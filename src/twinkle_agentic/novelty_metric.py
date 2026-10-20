# Copyright (c) ModelScope Contributors. All rights reserved.
"""A novelty score a program can compute, for a group of sibling proposals.

Why this exists
---------------
If the novelty term in a challenger reward comes from an LLM judge, measure it
before trusting it. On two recorded runs it did not discriminate: over 1439 judged
proposals in one run, 93.3% scored exactly 0 and 66.2% of 225 groups scored
identically on every sibling; over a second run's 4589, 98.5% scored 0 and 91.6%
of 783 groups were constant. GRPO subtracts the group mean and divides by
the group standard deviation, so a term that is constant across a group
contributes **exactly zero gradient** -- not a small effect, a mathematically
zero one. Those runs had a novelty term wired into the reward and the term did
nothing.

:func:`constant_group_rate` is how that was measured, and it is exported so any
replacement has to be judged the same way.

What replaces it
----------------
Two things a program can measure, both computed **inside the group**, because
that is the contrast GRPO actually uses:

``distance``
    ``1 - max cosine`` to the most similar sibling. Answers "is there anyone in
    this group I am a near-duplicate of". High when a proposal is the odd one
    out, low when it is one of eight versions of one idea.
``spread``
    ``1 / cluster size``, where siblings closer than ``cluster_threshold`` to
    each other are one cluster. Answers "how many of us are the same idea",
    which the max-cosine term alone can miss: five near-identical proposals all
    have a low distance, but they are not eight different proposals.

The vectoriser is fitted on the whole corpus of statements from the run, not on
one group. That detail is load-bearing: these statements share their scaffolding
almost verbatim (``## Task:``, ``### Raw Input``, ``Create the file``), and IDF
fitted per group would give that boilerplate the highest weight of anything in
the text. Fitted across the run, a phrase that appears in every statement
carries almost no weight and the cosine reflects the content.

Nothing here is a model or a GPU: TF-IDF over word n-grams plus a union-find over
a threshold is a few milliseconds per group, so it can run inside the loop on
every proposal without touching the sampler.

One consequence of fitting on the corpus is worth stating rather than
discovering: ``TfidfVectorizer`` drops tokens absent from the vocabulary it was
fitted on, so a word the corpus never used is invisible to the score. Inside a
run the corpus is every statement produced so far and the group being scored is a
subset of it, which is the intended arrangement. A caller that fits on one run
and scores another gets a scorer that cannot see the difference.

The calibration set is :meth:`~twinkle_agentic.challenger.task_bank.TaskBank.sibling_groups`,
which is why that method exists and why it keeps a group id per statement.

Dependencies
------------
``scikit-learn`` and ``numpy``, both imported inside the methods that need them
rather than at module scope. A run that never scores novelty pays neither, and
an environment without scikit-learn can still import this module, read
:func:`constant_group_rate` over recorded scores, and find out what is missing
only at the point it asks for a scorer.
"""
from __future__ import annotations

import collections

__all__ = ['NoveltyScorer', 'constant_group_rate', 'spread']


def _union_find_components(n: int, edges) -> list[int]:
    """Size of each node's connected component, as a list indexed by node."""
    parent = list(range(n))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a, b in edges:
        root_a, root_b = find(a), find(b)
        if root_a != root_b:
            parent[root_b] = root_a
    sizes = collections.Counter(find(i) for i in range(n))
    return [sizes[find(i)] for i in range(n)]


class NoveltyScorer:
    """Scores sibling proposals against each other. Fit once, score many.

    Args:
        corpus: every statement the run has produced, or any large sample of
            them. Used only to weight the n-grams; it is not a reference set, and
            the score never compares a proposal against it.
        ngram_range: word n-grams for the vectoriser. ``(1, 2)`` keeps common
            compounds together without the sparsity of longer n-grams on texts
            that run to 1.6k characters.
        max_features: cap on the vocabulary.
        cluster_threshold: cosine at or above which two siblings are one cluster.
            Omit to calibrate it from ``calibration_groups``.
        cluster_quantile: which quantile of the observed sibling cosines becomes
            the threshold when one is not given.
        calibration_groups: groups of sibling statements, i.e. what
            :meth:`TaskBank.sibling_groups` returns.
        weight_distance: weight of the ``distance`` term.
        weight_spread: weight of the ``spread`` term. The two are separate
            signals, so the blend is explicit rather than baked into one formula.
        sublinear_tf: damp term frequency, which matters when one statement
            repeats a phrase the others use once.
    """

    def __init__(self,
                 corpus,
                 *,
                 ngram_range=(1, 2),
                 max_features: int = 200000,
                 cluster_threshold: float | None = None,
                 cluster_quantile: float = 0.75,
                 calibration_groups=None,
                 weight_distance: float = 0.5,
                 weight_spread: float = 0.5,
                 sublinear_tf: bool = True) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.cluster_quantile = float(cluster_quantile)
        self.cluster_threshold = float(cluster_threshold) if cluster_threshold is not None else None
        self.weight_distance = float(weight_distance)
        self.weight_spread = float(weight_spread)
        self.vectoriser = TfidfVectorizer(
            ngram_range=ngram_range,
            max_features=max_features,
            sublinear_tf=sublinear_tf,
            strip_accents='unicode',
            lowercase=True,
            token_pattern=r'\b\w[\w\-]+\b|[^\s\w]')
        texts = [text for text in corpus if (text or '').strip()]
        if not texts:
            raise ValueError('NoveltyScorer needs at least one non-empty statement')
        self.vectoriser.fit(texts)
        self.n_fit = len(texts)
        self.n_features = len(self.vectoriser.vocabulary_)
        self.calibrated: dict[str, float] | None = None
        if self.cluster_threshold is None:
            # The threshold is a property of the corpus, not a constant to choose.
            # Measured on two recorded runs: the within-group sibling cosine had a
            # median of 0.046 and 0.044 and a p99 of 0.104 and 0.210, so a
            # hand-picked 0.55 left *every* group entirely in singleton clusters
            # (9 of 9 and 28 of 28) and the spread term measured nothing at all.
            # Calibrating at the 0.75 quantile put the threshold at 0.056 and
            # 0.059 -- about ten times lower. Taking the quantile of the observed
            # distribution keeps the same reading, "the closest quarter of pairs
            # are the same idea", on any corpus, with no number to re-tune when
            # the wording shifts.
            pairs = self._pairwise(calibration_groups or [])
            if pairs:
                pairs.sort()
                index = min(len(pairs) - 1, max(0, int(round(self.cluster_quantile * (len(pairs) - 1)))))
                self.cluster_threshold = pairs[index]
                self.calibrated = {
                    'quantile': self.cluster_quantile,
                    'threshold': self.cluster_threshold,
                    'pairs': len(pairs)
                }
            else:
                self.cluster_threshold = 0.5

    def _pairwise(self, groups) -> list[float]:
        """Within-group sibling cosines: the distribution the threshold comes from."""
        out: list[float] = []
        for statements in groups:
            texts = [text for text in statements if (text or '').strip()]
            if len(texts) < 2:
                continue
            matrix = self.vectoriser.transform(texts)
            similarity = (matrix @ matrix.T).toarray()
            for i in range(len(texts)):
                for j in range(i + 1, len(texts)):
                    out.append(float(similarity[i, j]))
        return out

    def score(self, statements, *, normalize: bool = True) -> list[float]:
        """One score per statement, in the same order. Empty input scores ``[]``.

        With ``normalize`` the raw mix is min-max mapped onto ``[0, 1]`` inside
        the group. That is not cosmetic: GRPO uses only a proposal's position
        relative to its siblings, so a term varying by 10% around a mean it
        shares with everyone is worth 10% of the ranking, and replacing a judge
        is pointless unless this axis decides its share of the ranking. The
        mapping is monotone, so the order -- the only thing the update sees -- is
        unchanged.
        """
        import numpy as np

        if not statements:
            return []
        if len(statements) == 1:
            # A group of one has nothing to differ from. 1.0 rather than 0.0, so
            # it does not read as "this proposal is a duplicate of something".
            return [1.0]
        matrix = self.vectoriser.transform([statement or '' for statement in statements])
        # Rows are L2-normalised by TfidfVectorizer (norm='l2' by default), so the
        # inner product is the cosine.
        similarity = (matrix @ matrix.T).toarray()
        np.fill_diagonal(similarity, -1.0)
        max_similarity = similarity.max(axis=1)
        edges = [(i, j) for i in range(len(statements)) for j in range(i + 1, len(statements))
                 if similarity[i, j] >= self.cluster_threshold]
        sizes = _union_find_components(len(statements), edges)
        raw = []
        for index in range(len(statements)):
            distance = max(0.0, 1.0 - float(max_similarity[index]))
            raw.append(self.weight_distance * distance + self.weight_spread * (1.0 / sizes[index]))
        if normalize and len(raw) > 1:
            low, high = min(raw), max(raw)
            if high > low:
                return [(value - low) / (high - low) for value in raw]
        return raw


def spread(values) -> float:
    """``max - min``: the thing that has to be non-zero for GRPO to see a term."""
    values = [float(value) for value in values if value is not None]
    return (max(values) - min(values)) if len(values) > 1 else 0.0


def constant_group_rate(groups) -> float:
    """Fraction of groups whose scores are all identical, i.e. contribute nothing.

    This is the number a novelty term is judged on, and the reason it is exported
    rather than kept in a notebook: a replacement that is not measured this way
    cannot be compared to what it replaced. On the judge it measured 0.66 and
    0.824 across two recorded runs.
    """
    total = constant = 0
    for values in groups:
        values = [float(value) for value in values if value is not None]
        if len(values) < 2:
            continue
        total += 1
        constant += spread(values) == 0.0
    return (constant / total) if total else 0.0
