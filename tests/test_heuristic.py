import pytest

from ai.heuristic_policy import (
    DEFAULT_WEIGHTS,
    HeuristicPolicy,
    RowProfile,
    _foul_penalty,
    _profile_row,
    _row_strength,
    score_action,
)
from ai.random_policy import RandomPolicy
from engine.cards import parse_cards
from engine.fantasy import FantasyTier
from state.action import Action, enumerate_initial_actions, enumerate_pineapple_actions
from state.board import (
    PlayerBoard,
    ROW_CAPACITY,
    SLOT_BOTTOM,
    SLOT_DISCARD,
    SLOT_MIDDLE,
    SLOT_TOP,
)
from state.game_state import GameState, N_NORMAL_STREETS


# -------------------------- profile_row ---------------------------
def test_profile_pair_detection():
    p = _profile_row(parse_cards("As Ad 7c"), ROW_CAPACITY[SLOT_TOP])
    assert p.max_mult == 2
    assert p.max_mult_rank == 12  # rank A


def test_profile_trips_detection():
    p = _profile_row(parse_cards("Qs Qd Qc"), ROW_CAPACITY[SLOT_TOP])
    assert p.max_mult == 3
    assert p.max_mult_rank == 10


def test_profile_flush_count():
    p = _profile_row(parse_cards("As Ks Qs 2d"), ROW_CAPACITY[SLOT_BOTTOM])
    assert p.max_suit_count == 3


def test_profile_straight_run():
    p = _profile_row(parse_cards("9c Tc Jc"), ROW_CAPACITY[SLOT_BOTTOM])
    assert p.longest_run >= 3


def test_profile_with_joker_helps_pair():
    p = _profile_row(parse_cards("Qs *1"), ROW_CAPACITY[SLOT_TOP])
    # joker can virtually become Q -> max_mult=2
    assert p.max_mult == 2


# -------------------------- foul penalty ---------------------------
def test_foul_penalty_when_top_dominates():
    # top has trips, middle has empty -> top_min(TRIPS) > mid_max(QUADS)? no,
    # mid empty can still reach quads. So instead force top trips and small middle
    top = _profile_row(parse_cards("As Ad Ac"), 3)
    mid = _profile_row(parse_cards("2c 3d 4h 5s 7c"), 5)  # high card max only
    bot = _profile_row(parse_cards(""), 5)
    pen = _foul_penalty(top, mid, bot, DEFAULT_WEIGHTS)
    assert pen >= DEFAULT_WEIGHTS.w_foul  # certain foul vs middle


def test_no_foul_penalty_balanced():
    # very early state with no committed pairs anywhere -> no ordering risk
    top = _profile_row(parse_cards("2c 3d"), 3)
    mid = _profile_row(parse_cards("5s 7c"), 5)
    bot = _profile_row(parse_cards("Ks Qd"), 5)
    pen = _foul_penalty(top, mid, bot, DEFAULT_WEIGHTS)
    # only the smooth expected-category term may contribute; should be small
    assert pen < DEFAULT_WEIGHTS.w_foul / 4


def test_no_ordering_penalty_when_lower_row_has_stronger_category():
    """Regression for the user-reported AA-on-top vs two-pair-on-middle bug.

    With the previous tuple layout ``(max_mult, max_mult_rank, second_mult)``
    a pair of aces on top compared *greater* than a two-pair on middle
    (rank-A=12 dominated second_mult=2 at the wrong tuple position),
    firing a ~110-point order-violation penalty even though the two-pair
    is the strictly stronger poker hand. The corrected layout
    ``(max_mult, second_mult, max_mult_rank)`` distinguishes one-pair
    from two-pair before the kicker rank, so the penalty must not fire.
    """
    top = _profile_row(parse_cards("Ac As"), 3)              # pair A, 2/3
    mid = _profile_row(parse_cards("5s 6s 9h 5d 6d"), 5)     # two pair 5s/6s
    bot = _profile_row(parse_cards("8h Jh *1 8d"), 5)        # trips 8 (joker)
    pen = _foul_penalty(top, mid, bot, DEFAULT_WEIGHTS)
    # rows are already correctly ordered (PAIR < TWO_PAIR < TRIPS):
    # only the smooth expected-category term may contribute, which is
    # strictly below one order_violation tick (22 pts).
    assert pen < DEFAULT_WEIGHTS.w_order_violation, (
        f"spurious ordering penalty {pen:.2f}"
    )


def test_ordering_penalty_still_fires_when_top_truly_dominates():
    """Belt-and-suspenders: the tuple-comparison arm must still fire when
    the upper row really is at risk of out-ranking the lower row.

    Top has a pair of aces (PAIR=1), middle is two high cards with no
    pair (HIGH_CARD=0). Top is currently stronger than middle and may
    still get worse — penalty *should* fire.
    """
    top = _profile_row(parse_cards("Ac As"), 3)
    mid = _profile_row(parse_cards("3c 7d"), 5)              # no pair yet
    bot = _profile_row(parse_cards(""), 5)
    pen = _foul_penalty(top, mid, bot, DEFAULT_WEIGHTS)
    assert pen >= DEFAULT_WEIGHTS.w_order_violation


def test_score_aa_top_fantasy_play_outranks_safe_drop_on_user_position():
    """End-to-end regression for the exact position reported by the user.

    Mid-hand board (street 4, fantasy NORMAL):
        TOP:    Ac
        MIDDLE: 5s 6s 9h 5d
        BOTTOM: 8h Jh *1 8d
        Discards: Ah Ts
        Hand:   As Kc 6d

    The fantasy-entering action (As -> TOP, 6d -> MIDDLE, Kc -> X)
    achieves AA on top with no foul risk, while the heuristic's old
    top pick (Kc/6d to top/middle, As to discard) gives up the AA-pair
    fantasy entry entirely. After the fix the fantasy play must score
    competitively with the safe drop (i.e. the spurious foul penalty
    that previously made it -18.35 vs +66.85 must be gone).
    """
    board = PlayerBoard()
    for c in parse_cards("Ac"):
        board.place(c, SLOT_TOP)
    for c in parse_cards("5s 6s 9h 5d"):
        board.place(c, SLOT_MIDDLE)
    for c in parse_cards("8h Jh *1 8d"):
        board.place(c, SLOT_BOTTOM)
    for c in parse_cards("Ah Ts"):
        board.place(c, SLOT_DISCARD)
    # AA top, 6d to middle, Kc discarded
    fantasy_action = Action(placements=(
        (parse_cards("As")[0], SLOT_TOP),
        (parse_cards("6d")[0], SLOT_MIDDLE),
        (parse_cards("Kc")[0], SLOT_DISCARD),
    ))
    # safe drop: As discarded, K to top, 6d to top
    safe_action = Action(placements=(
        (parse_cards("Kc")[0], SLOT_TOP),
        (parse_cards("6d")[0], SLOT_TOP),
        (parse_cards("As")[0], SLOT_DISCARD),
    ))
    s_fantasy = score_action(fantasy_action, board)
    s_safe = score_action(safe_action, board)
    # The previously-buggy run gave s_fantasy.foul_penalty ~110 and
    # s_fantasy.total ~ -18. After the fix the foul penalty must be
    # zero (rows are correctly ordered) and the fantasy bonus (~14)
    # plus AA top royalty (~9) should make the score positive.
    assert s_fantasy.foul_penalty == 0.0, (
        f"unexpected foul penalty on a legitimate AA-top fantasy play: "
        f"{s_fantasy.foul_penalty:.2f}"
    )
    assert s_fantasy.total > 0, (
        f"fantasy-entering action should score > 0 after fix; got "
        f"{s_fantasy.total:.2f}"
    )



# -------------------------- score_action ---------------------------
def test_score_prefers_pair_to_bottom_over_top():
    # 5 cards: As Ad 5c 6d 7h. Putting AA on bottom should beat AA on top.
    cards = parse_cards("As Ad 5c 6d 7h")
    acts = enumerate_initial_actions(cards)
    bot_pair = next(
        a for a in acts
        if all(s == SLOT_BOTTOM for c, s in a.placements if c in parse_cards("As Ad"))
    )
    top_pair = next(
        a for a in acts
        if all(s == SLOT_TOP for c, s in a.placements if c in parse_cards("As Ad"))
    )
    sb = score_action(bot_pair, PlayerBoard()).total
    st = score_action(top_pair, PlayerBoard()).total
    # putting AA on bottom is safer (avoids foul risk) and has higher row weight
    assert sb > st


def test_score_avoids_certain_foul_via_top_trips():
    # Trips on top with the rest being garbage -> should be penalized vs.
    # putting the trips on bottom.
    cards = parse_cards("As Ad Ac 5c 7h")
    acts = enumerate_initial_actions(cards)

    aaa_top = next(
        a for a in acts
        if all(s == SLOT_TOP for c, s in a.placements if c in parse_cards("As Ad Ac"))
    )
    aaa_bot = next(
        a for a in acts
        if all(s == SLOT_BOTTOM for c, s in a.placements if c in parse_cards("As Ad Ac"))
    )
    s_top = score_action(aaa_top, PlayerBoard()).total
    s_bot = score_action(aaa_bot, PlayerBoard()).total
    assert s_bot > s_top


def test_score_breakdown_components():
    cards = parse_cards("As Ad 5c 6d 7h")
    a = enumerate_initial_actions(cards)[0]
    sc = score_action(a, PlayerBoard())
    # total must equal sum of components
    assert sc.total == pytest.approx(
        sc.row_strength + sc.royalty_bonus + sc.fantasy_bonus
        - sc.foul_penalty - sc.discard_penalty
    )


# -------------------------- HeuristicPolicy.act ---------------------------
def test_heuristic_policy_returns_legal_action():
    gs = GameState.new(seed=11)
    gs.deal_street()
    pol = HeuristicPolicy(seed=0)
    a = pol.act(gs, 0)
    cards = list(gs.hands[0].pending)
    placed_cards = [c for c, _ in a.placements]
    assert sorted(placed_cards) == sorted(cards)


def test_heuristic_policy_completes_full_hand_no_foul_typical():
    """Self-play: heuristic vs heuristic over many seeds; foul rate.

    A pure greedy heuristic without lookahead caps around 25-35% foul because
    late-street forced placements can't always be avoided. The Phase 4
    rollout policy will drive this far lower.
    """
    pol = HeuristicPolicy(seed=0)
    fouls = 0
    n_games = 60
    for seed in range(n_games):
        gs = GameState.new(seed=seed)
        for _ in range(N_NORMAL_STREETS):
            gs.deal_street()
            for p in (0, 1):
                gs.step(p, pol.act(gs, p))
        sb = gs.score()
        if sb.a_foul:
            fouls += 1
        if sb.b_foul:
            fouls += 1
    foul_rate = fouls / (2 * n_games)
    # baseline ceiling for a greedy heuristic; rollouts will improve drastically.
    assert foul_rate < 0.40, f"foul rate too high: {foul_rate:.2%}"
    # but should clearly beat fully-random (~77% foul)
    assert foul_rate < 0.50


def test_random_policy_returns_legal_action():
    gs = GameState.new(seed=2)
    gs.deal_street()
    pol = RandomPolicy(seed=0)
    a = pol.act(gs, 0)
    cards = list(gs.hands[0].pending)
    assert sorted(c for c, _ in a.placements) == sorted(cards)


def test_heuristic_beats_random_on_average():
    """Smoke test: over many seeded games, heuristic should beat random
    significantly. Tight bound: average score per hand > +5 chips."""
    n = 60
    total = 0
    for seed in range(n):
        gs = GameState.new(seed=seed)
        h = HeuristicPolicy(seed=seed)
        r = RandomPolicy(seed=seed + 1000)
        for _ in range(N_NORMAL_STREETS):
            gs.deal_street()
            gs.step(0, h.act(gs, 0))
            gs.step(1, r.act(gs, 1))
        sb = gs.score()
        total += sb.total_a
    avg = total / n
    assert avg > 5.0, f"heuristic avg vs random = {avg:.2f}, expected > 5"


def test_heuristic_deterministic_for_fixed_seed_and_state():
    gs1 = GameState.new(seed=33)
    gs2 = GameState.new(seed=33)
    gs1.deal_street()
    gs2.deal_street()
    h1 = HeuristicPolicy(seed=7)
    h2 = HeuristicPolicy(seed=7)
    assert h1.act(gs1, 0) == h2.act(gs2, 0)
