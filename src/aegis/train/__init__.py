"""Closed-loop training (MuJoCo-first slice).

Gym-style wrapper + behavior-cloning trainer live here. The safety gateway
stays mandatory inside the env step — training never bypasses it.
"""

from __future__ import annotations
