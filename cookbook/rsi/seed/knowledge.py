# Copyright (c) ModelScope Contributors. All rights reserved.
"""Knowledge-card seeding, surfaced where ``create_seeder`` looks for it.

The routing and the adapter live in
:mod:`twinkle_agentic.challenger.knowledge`, so they are importable and testable
without a cookbook on ``sys.path``. This module is the re-export that lets
``cookbook/rsi/rsi_grpo.py`` write ``from seed import KnowledgeCardSeeder``
alongside the other seeders.

See that module for what a card is, why retrieving one beats inventing a topic,
and how the epoch drives coverage of the bank.
"""
from twinkle_agentic.challenger.knowledge import KnowledgeCardSeeder

__all__ = ['KnowledgeCardSeeder']
