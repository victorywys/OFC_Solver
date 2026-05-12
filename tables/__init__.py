"""Precomputed tables for runtime acceleration.

Each table is (data structure + Collector builder + lookup API). They all
share a uniform interface:

    >>> from tables import FoulProbTable, FoulProbCollector
    >>> # build:
    >>> col = FoulProbCollector()
    >>> for rec in records: col.observe(rec)
    >>> table = col.result()
    >>> # lookup:
    >>> p_foul = table.lookup(state_sig)

Tables built from self-play traces are typed `Collector`-compatible so
they slot directly into `simulation.SelfPlay.run_parallel(...)`. Tables
built by other means (e.g. enumerating fantasy hands, running the
fantasy solver) live next to them and follow the same `result()` /
`save` / `load` API.
"""

from .fantasy_cache import (
    FantasyArrangementCache,
    FantasyArrangementCacheCollector,
    RecordingFantasyPolicy,
)
from .fantasy_ev import (
    FantasyEVCollector,
    FantasyEVTable,
)
from .foul_prob import FoulProbCollector, FoulProbTable
from .opening_book import OpeningBookCollector, OpeningBookTable
from .canonical_opening import (
    CanonicalOpeningBookTable,
    canonicalize,
    apply_inverse,
    enumerate_canonical_hands,
)
from .policy_prior import PolicyPriorCollector, PolicyPriorTable
from .signatures import (
    action_signature,
    canonical_action,
    fantasy_hand_signature,
    state_signature,
    street1_hand_signature,
    turn_state_signature,
)
from .loader import load_run_as_policy, load_tables
from .table_aware_policy import TableAwareConfig, TableAwarePolicy
from .transposition import TranspositionTable
from .welford import Welford

__all__ = [
    "FoulProbTable",
    "FoulProbCollector",
    "PolicyPriorTable",
    "PolicyPriorCollector",
    "OpeningBookTable",
    "OpeningBookCollector",
    "CanonicalOpeningBookTable",
    "canonicalize",
    "apply_inverse",
    "enumerate_canonical_hands",
    "FantasyEVTable",
    "FantasyEVCollector",
    "FantasyArrangementCache",
    "FantasyArrangementCacheCollector",
    "RecordingFantasyPolicy",
    "TranspositionTable",
    "Welford",
    "TableAwareConfig",
    "TableAwarePolicy",
    "load_tables",
    "load_run_as_policy",
    "state_signature",
    "turn_state_signature",
    "action_signature",
    "canonical_action",
    "fantasy_hand_signature",
    "street1_hand_signature",
]
