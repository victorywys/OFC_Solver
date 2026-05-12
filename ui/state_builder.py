"""Build a `GameState` from a JSON-shaped spec.

Spec shape (all card lists are arrays of strings like "As", "Td", "*1"):

    {
      "street": 3,                       # 1..5 (current street being played)
      "to_act": 0,                       # 0 or 1 — which player asks for advice
      "auto_fill_opponent": true,        # default true: synthesize opponent
                                          # via heuristic play through prior
                                          # streets when their data is empty.
      "dead_cards": ["2c", "7d"],        # optional cards seen elsewhere;
                                          # removed from the deck so rollouts
                                          # treat them as not in any hand.
      "players": [
        {
          "fantasy_tier": 0,             # 0 / 14 / 15 / 16 / 17
          "board": {
            "top":      ["As", "Kc"],
            "middle":   ["Qh", "Jd", "Ts"],
            "bottom":   ["7s", "7c", "8h"],
            "discards": ["2c", "3d"]
          },
          "pending":   ["6c", "6d", "9s"]   # cards the player has in hand
        },
        { ...same shape for player 1... }
      ]
    }

Either player's `pending` may be empty (e.g. opponent hasn't been dealt
this street yet, or you don't know their cards). If empty AND
`auto_fill_opponent` is true, the opponent's full pre-current-street
history is synthesized by heuristic self-play in isolation, and they're
dealt a pending hand for the current street. This keeps rollouts
self-consistent (opponent doesn't foul because of missing history).

Rules / validation
------------------
* No card may appear twice across all known positions.
* Row sizes must respect capacities (3/5/5).
* Pending size must match the street rules (5 on street 1 normal,
  3 on streets 2-5 normal, 14-17 in fantasy).
* `to_act`'s pending cannot be empty (we need cards to recommend on).
"""

from __future__ import annotations

import random
from typing import Any

from engine.cards import NUM_CARDS, parse_card
from engine.deck import Deck
from engine.fantasy import FantasyTier
from state.board import (
    PlayerBoard,
    ROW_CAPACITY,
    SLOT_BOTTOM,
    SLOT_DISCARD,
    SLOT_MIDDLE,
    SLOT_TOP,
)
from state.game_state import GameState, HandState, MAX_PLAYERS, N_NORMAL_STREETS


def _coerce_cards(value: Any) -> list[int]:
    """Accept list of int or list of str; return list of int card-ids."""
    if value is None:
        return []
    out: list[int] = []
    for v in value:
        if isinstance(v, int):
            if not (0 <= v < NUM_CARDS):
                raise ValueError(f"card id out of range: {v}")
            out.append(v)
        elif isinstance(v, str):
            out.append(parse_card(v))
        else:
            raise ValueError(f"unrecognized card value: {v!r}")
    return out


def _default_player() -> dict:
    return {
        "fantasy_tier": 0,
        "board": {"top": [], "middle": [], "bottom": [], "discards": []},
        "pending": [],
    }


def _is_player_empty(p: dict) -> bool:
    """True iff every list in this player spec is empty."""
    bd = p.get("board") or {}
    if any(bd.get(k) for k in ("top", "middle", "bottom", "discards")):
        return False
    if p.get("pending"):
        return False
    return True


def _synthesize_opponent(
    excluded_cards: set[int],
    target_street: int,
    seed: int,
) -> tuple[HandState, list[int]]:
    """Heuristically play one player through streets 1..target_street.

    Cards in `excluded_cards` are removed from the synthesized deck so
    they can't be dealt to the opponent. After completion the opponent's
    `pending` holds their freshly-dealt hand for `target_street`, and
    streets 1..(target_street - 1) are filled in on their board.

    Returns (opponent HandState, leftover deck cards) so the caller can
    use the leftover cards as the master deck.
    """
    # Local imports to avoid circular dependencies and keep import-time
    # cost of state_builder low.
    from ai.heuristic_policy import HeuristicPolicy

    rng = random.Random(seed)
    available = [c for c in range(NUM_CARDS) if c not in excluded_cards]
    rng.shuffle(available)

    deck = Deck.__new__(Deck)
    deck._cards = available
    deck._rng = rng

    # Build a 2-player game where index 0 is a "ghost" who is already
    # finished (so deal_street() never deals to them), and index 1 is
    # the real opponent we're synthesizing.
    ghost = HandState()
    ghost.finished = True
    opp = HandState()
    opp.fantasy_tier = FantasyTier.NORMAL
    fake_gs = GameState(
        deck=deck, hands=(ghost, opp), current_street=0
    )

    policy = HeuristicPolicy(seed=rng.randint(0, 2**31 - 1))

    # Play streets 1..(target_street - 1) and place each one heuristically.
    for street in range(1, target_street):
        fake_gs.deal_street()
        action = policy.act(fake_gs, 1)
        fake_gs.step(1, action)

    # Deal the current street's pending so the opponent enters the actual
    # game with a full hand to play on this street.
    fake_gs.deal_street()
    return opp, list(deck._cards)


def build_game_state(
    spec: dict,
    *,
    seed: int = 0,
    auto_fill_opponent: bool = True,
) -> GameState:
    """Construct a `GameState` matching `spec`. See module docstring."""
    if not isinstance(spec, dict):
        raise ValueError("spec must be a dict")

    street = int(spec.get("street", 1))
    if not (1 <= street <= N_NORMAL_STREETS):
        raise ValueError(f"street must be in 1..{N_NORMAL_STREETS}")

    # spec-level toggle overrides the kwarg default.
    if "auto_fill_opponent" in spec:
        auto_fill_opponent = bool(spec["auto_fill_opponent"])

    dead_cards_raw = spec.get("dead_cards") or []
    dead_cards = _coerce_cards(dead_cards_raw)

    players_in = spec.get("players") or [_default_player(), _default_player()]
    n_players = len(players_in)
    if not (2 <= n_players <= MAX_PLAYERS):
        raise ValueError(
            f"players must have 2..{MAX_PLAYERS} entries; got {n_players}"
        )

    to_act = int(spec.get("to_act", 0))
    if not (0 <= to_act < n_players):
        raise ValueError(f"to_act must be in 0..{n_players - 1}")

    # Parse each player's cards.
    parsed: list[dict] = []
    all_known: list[int] = []
    for p in players_in:
        bd = p.get("board") or {}
        top = _coerce_cards(bd.get("top"))
        mid = _coerce_cards(bd.get("middle"))
        bot = _coerce_cards(bd.get("bottom"))
        disc = _coerce_cards(bd.get("discards"))
        pending = _coerce_cards(p.get("pending"))
        tier_int = int(p.get("fantasy_tier", 0))
        if tier_int not in (0, 14, 15, 16, 17):
            raise ValueError(
                f"fantasy_tier must be 0/14/15/16/17, got {tier_int}"
            )
        # capacity checks
        if len(top) > ROW_CAPACITY[SLOT_TOP]:
            raise ValueError(f"top has {len(top)} > {ROW_CAPACITY[SLOT_TOP]}")
        if len(mid) > ROW_CAPACITY[SLOT_MIDDLE]:
            raise ValueError(f"middle has {len(mid)} > {ROW_CAPACITY[SLOT_MIDDLE]}")
        if len(bot) > ROW_CAPACITY[SLOT_BOTTOM]:
            raise ValueError(f"bottom has {len(bot)} > {ROW_CAPACITY[SLOT_BOTTOM]}")

        all_known.extend(top + mid + bot + disc + pending)
        parsed.append(
            dict(top=top, mid=mid, bot=bot, disc=disc, pending=pending,
                 tier=FantasyTier(tier_int))
        )

    # Duplicate detection (across both players AND dead cards).
    full_known = all_known + list(dead_cards)
    if len(full_known) != len(set(full_known)):
        seen = set()
        dups = []
        for c in full_known:
            if c in seen:
                dups.append(c)
            seen.add(c)
        raise ValueError(f"duplicate cards (incl. dead): {dups}")

    # Pending-size validation for the player who's asking.
    p_act = parsed[to_act]
    npending = len(p_act["pending"])
    if npending == 0:
        raise ValueError("to_act player has no pending cards (nothing to recommend)")
    tier = p_act["tier"]
    if tier != FantasyTier.NORMAL:
        if npending != tier.n_cards:
            raise ValueError(
                f"fantasy tier {tier.name} expects {tier.n_cards} pending, "
                f"got {npending}"
            )
    elif street == 1:
        if npending != 5:
            raise ValueError(f"street 1 normal expects 5 pending, got {npending}")
    else:
        if npending != 3:
            raise ValueError(f"streets 2-5 normal expect 3 pending, got {npending}")

    # Placed-count consistency check (catches the most common mistake:
    # user picks street K but the board cards are off by a street).
    # Only enforced for normal (non-fantasy) play because fantasy compresses
    # the entire layout into a single action on street 1.
    if tier == FantasyTier.NORMAL:
        # cards on rows BEFORE this street's action:
        #   street 1: 0 placed, 0 discards (the deal hasn't happened yet)
        #   street 2: 5 placed, 0 discards (street 1 deals 5 / places 5 / discards 0)
        #   street 3: 7 placed, 1 discard
        #   street 4: 9 placed, 2 discards
        #   street 5: 11 placed, 3 discards
        expected_placed = {1: 0, 2: 5, 3: 7, 4: 9, 5: 11}[street]
        expected_disc = {1: 0, 2: 0, 3: 1, 4: 2, 5: 3}[street]
        actual_placed = len(p_act["top"]) + len(p_act["mid"]) + len(p_act["bot"])
        actual_disc = len(p_act["disc"])
        if actual_placed != expected_placed:
            raise ValueError(
                f"player {to_act}: street {street} expects "
                f"{expected_placed} placed cards before this action, "
                f"but board has {actual_placed}"
            )
        if actual_disc != expected_disc:
            raise ValueError(
                f"player {to_act}: street {street} expects "
                f"{expected_disc} discards before this action, "
                f"but board has {actual_disc}"
            )

    # ------------------------------------------------------------------
    # Optional opponent synthesis. We only auto-fill an opponent slot if
    # (a) the caller opted in, (b) that opponent's spec is fully empty,
    # (c) target street is >= 2 (street 1 with empty opponents is fine:
    # deal_street will hand them their initial 5 cards inside rollouts),
    # and (d) the opponent is NORMAL tier (we don't synthesize fantasy).
    # ------------------------------------------------------------------
    opp_was_synthesized = False
    excluded: set[int] = set(all_known) | set(dead_cards)
    remaining: list[int] | None = None

    if street >= 2 and auto_fill_opponent:
        for i in range(n_players):
            if i == to_act:
                continue
            if not _is_player_empty(players_in[i]):
                continue
            if parsed[i]["tier"] != FantasyTier.NORMAL:
                continue
            opp_hs, leftover = _synthesize_opponent(
                excluded_cards=excluded,
                target_street=street,
                seed=(seed ^ 0xA5A5A5) + i,
            )
            parsed[i] = dict(
                top=list(opp_hs.board.rows[SLOT_TOP]),
                mid=list(opp_hs.board.rows[SLOT_MIDDLE]),
                bot=list(opp_hs.board.rows[SLOT_BOTTOM]),
                disc=list(opp_hs.board.discards),
                pending=list(opp_hs.pending),
                tier=opp_hs.fantasy_tier,
            )
            # Now those synthesized cards are also "used" — add them to the
            # exclusion set so subsequent opponents can't be dealt the same.
            excluded |= set(
                parsed[i]["top"] + parsed[i]["mid"] + parsed[i]["bot"]
                + parsed[i]["disc"] + parsed[i]["pending"]
            )
            opp_was_synthesized = True
            remaining = leftover

    if remaining is None:
        # Build remaining deck (ordered card pool — Deck.deal pops from the end).
        known_set = set()
        for p in parsed:
            known_set |= set(
                p["top"] + p["mid"] + p["bot"] + p["disc"] + p["pending"]
            )
        known_set |= set(dead_cards)
        remaining = [c for c in range(NUM_CARDS) if c not in known_set]

    rng = random.Random(seed)
    rng.shuffle(remaining)
    deck = Deck.__new__(Deck)
    deck._cards = remaining
    deck._rng = rng

    # Construct hands
    hands: list[HandState] = []
    for p in parsed:
        hs = HandState()
        hs.fantasy_tier = p["tier"]
        for c in p["top"]:
            hs.board.place(c, SLOT_TOP)
        for c in p["mid"]:
            hs.board.place(c, SLOT_MIDDLE)
        for c in p["bot"]:
            hs.board.place(c, SLOT_BOTTOM)
        for c in p["disc"]:
            hs.board.discards.append(c)
        hs.pending = list(p["pending"])
        hs.finished = hs.board.is_full() and not hs.pending
        hands.append(hs)

    gs = GameState(deck=deck, hands=tuple(hands), current_street=street)
    # Attach a hint flag so callers (e.g. the HTTP server) can surface
    # whether the opponent was auto-filled. Not part of the dataclass
    # schema; just an annotated attribute.
    gs.opp_was_synthesized = opp_was_synthesized  # type: ignore[attr-defined]
    return gs


__all__ = ["build_game_state"]
