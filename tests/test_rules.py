from engine.cards import parse_cards
from engine.fantasy import (
    FantasyTier,
    fantasy_tier_from_top,
    maintains_fantasy,
    next_fantasy_tier,
)
from engine.evaluator import evaluate_3, evaluate_5
from engine.rules import is_foul, is_valid
from engine.scoring import Board, score_match


# ---------- validity ----------
def test_valid_board():
    top = parse_cards("Qs Qd 5c")           # pair Q
    mid = parse_cards("As Kd Ac Kh 2s")      # two pair AAKK
    bot = parse_cards("Ts Td Tc Th 9s")      # quads
    assert is_valid(top, mid, bot)
    assert not is_foul(top, mid, bot)


def test_foul_top_beats_middle():
    top = parse_cards("As Ad Ac")            # trips of A
    mid = parse_cards("Ks Kd 2c 3h 4s")      # pair K
    bot = parse_cards("Ts Td Tc Th 9s")      # quads
    assert is_foul(top, mid, bot)


def test_foul_middle_beats_bottom():
    top = parse_cards("2s 3d 4c")
    mid = parse_cards("As Ad Ac Kh Ks")      # full house
    bot = parse_cards("Js Jd Jc 2h 3s")      # trips
    assert is_foul(top, mid, bot)


def test_foul_strictly_greater_or_equal():
    # if mid == bot exactly (both straights with same high), is_valid should hold
    top = parse_cards("2s 3d 4c")
    mid = parse_cards("Ts Jd Qc Kh As")      # A-high straight
    bot = parse_cards("Th Jh Qh Ks Ad")      # A-high straight (different suits)
    assert is_valid(top, mid, bot)


# ---------- fantasy entry ----------
def test_qq_enters_f14():
    r = evaluate_3(parse_cards("Qs Qd 2c"))
    assert fantasy_tier_from_top(r) == FantasyTier.F14


def test_kk_enters_f15():
    r = evaluate_3(parse_cards("Ks Kd 2c"))
    assert fantasy_tier_from_top(r) == FantasyTier.F15


def test_aa_enters_f16():
    r = evaluate_3(parse_cards("As Ad 2c"))
    assert fantasy_tier_from_top(r) == FantasyTier.F16


def test_trips_enters_f17():
    r = evaluate_3(parse_cards("2s 2d 2c"))
    assert fantasy_tier_from_top(r) == FantasyTier.F17


def test_jj_does_not_enter():
    r = evaluate_3(parse_cards("Js Jd 2c"))
    assert fantasy_tier_from_top(r) == FantasyTier.NORMAL


# ---------- fantasy maintenance: NEVER UPGRADES ----------
def test_f14_with_aa_does_not_upgrade_to_f16():
    # entered F14 via QQ; new round has AA top -> still F14
    new_top = evaluate_3(parse_cards("As Ad 2c"))
    new_bot = evaluate_5(parse_cards("As Kd Qc Jh 9s"))  # nothing special
    assert next_fantasy_tier(FantasyTier.F14, new_top, new_bot) == FantasyTier.F14


def test_f17_maintains_only_with_trips_top_or_quads_bottom():
    # F17 with new top JJ -> top condition fails, but bottom quads keeps F17
    new_top = evaluate_3(parse_cards("Js Jd 2c"))
    new_bot = evaluate_5(parse_cards("9s 9d 9c 9h Ks"))  # quads
    assert maintains_fantasy(FantasyTier.F17, new_top, new_bot)
    # bottom merely full house -> drop to NORMAL
    new_bot2 = evaluate_5(parse_cards("As Ad Ac Kh Ks"))
    assert not maintains_fantasy(FantasyTier.F17, new_top, new_bot2)
    assert next_fantasy_tier(FantasyTier.F17, new_top, new_bot2) == FantasyTier.NORMAL


def test_f14_maintain_with_kk_top_ok():
    # F14 entered with QQ. Top KK satisfies "QQ+ pair" entry condition -> maintain
    new_top = evaluate_3(parse_cards("Ks Kd 2c"))
    new_bot = evaluate_5(parse_cards("As Kd Qc Jh 9s"))
    assert next_fantasy_tier(FantasyTier.F14, new_top, new_bot) == FantasyTier.F14


def test_f16_drops_with_kk():
    # F16 entry condition was AA; KK top is NOT AA, so unless bottom quads+, drop.
    new_top = evaluate_3(parse_cards("Ks Kd 2c"))
    new_bot = evaluate_5(parse_cards("As Kd Qc Jh 9s"))
    assert next_fantasy_tier(FantasyTier.F16, new_top, new_bot) == FantasyTier.NORMAL


# ---------- scoring ----------
def _board(top, mid, bot):
    return Board(tuple(parse_cards(top)), tuple(parse_cards(mid)), tuple(parse_cards(bot)))


def test_scoop_double():
    a = _board("As Ad Kc", "Ts Td Tc Th 9s", "9s 8s 7s 6s 5s")  # AAK / quads / SF
    b = _board("2s 3d 4c", "5s 7d 9c Jh Qh", "6c 8c Tc Kh Ah")  # high / Q-hi / A-hi
    sb = score_match(a, b)
    assert not sb.a_foul and not sb.b_foul
    # A wins all 3 lines
    assert sb.line_score_a == 3
    assert sb.scoop_bonus_a == 3
    # royalties: a top AAK -> pair A => 2; mid quads => 12; bot SF => 12
    assert sb.a_royalties == 2 + 12 + 12
    # b: top high_card => 1, mid high_card => 1, bot high_card => 0
    assert sb.b_royalties == 1 + 1 + 0
    assert sb.total_a == 3 + 3 + (2 + 12 + 12) - (1 + 1)


def test_foul_loses_all_plus_doubled():
    a = _board("As Ad Ac", "2s 3d 4c 5h 7s", "8s 9d Tc Jh Qs")  # trips top, weaker mid -> foul
    b = _board("2s 3d 4c", "5s 7d 9c Jh Qs", "Ts Td Tc Th 9c")  # valid
    sb = score_match(a, b)
    assert sb.a_foul and not sb.b_foul
    assert sb.line_score_a == -3
    assert sb.scoop_bonus_a == -3
    # b: top high_card=1, mid high_card=1, bot quads=8 -> 10
    assert sb.b_royalties == 10


def test_split_lines():
    a = _board("Ks Kd 2c", "As Ad Kc Kh 2s", "Ts Td Tc Th 9s")  # KK top / two pair / quads
    b = _board("As Ad 2c", "Js Jd Jc 2h 3s", "9s 9d 9c 9h Ks")  # AA top / trips / quads(9)
    sb = score_match(a, b)
    # top: AA > KK -> b wins top    (-1 for a)
    # mid: AAKK two pair > JJJ trips? two_pair vs trips: TRIPS=3, TWO_PAIR=2, so trips wins -> b (-1)
    # bot: TTTT > 9999 -> a (+1)
    assert sb.line_score_a == -1
    assert sb.scoop_bonus_a == 0


# ---------- joker-aware foul resolution ----------
from engine.cards import JOKER_1, JOKER_2, parse_card  # noqa: E402
from engine.evaluator import PAIR, QUADS, TRIPS  # noqa: E402
from engine.rules import resolve_board  # noqa: E402


def test_joker_on_top_avoids_foul_via_low_pairing():
    # Without joker awareness this layout would foul: joker on top maxes
    # to KKK (trips), beating middle pair-A. The player can choose a low
    # rank for the joker to keep top as merely PAIR(K) and avoid fouling.
    top = (parse_card("Ks"), parse_card("Kh"), JOKER_1)
    mid = (parse_card("As"), parse_card("Ah"), parse_card("Qd"),
           parse_card("Jd"), parse_card("Td"))
    bot = (parse_card("2c"), parse_card("3c"), parse_card("4c"),
           parse_card("5c"), parse_card("6c"))
    t, m, b, fouled = resolve_board(top, mid, bot)
    assert not fouled
    assert t[0] == PAIR and t[1][0] == 11   # pair of K (rank 11)
    assert is_valid(top, mid, bot)
    assert not is_foul(top, mid, bot)


def test_joker_resolver_picks_max_royalty_when_safe():
    # Two jokers in middle, bottom is straight-flush so quads-A on middle
    # is safe (SF > Q). The resolver should pick joker=Ac/Ad to make
    # quads-A (royalty 12) rather than trips-A (royalty 2).
    top = (parse_card("Ks"), parse_card("Kh"), parse_card("Tc"))
    mid = (parse_card("As"), parse_card("Ah"), JOKER_1, JOKER_2,
           parse_card("Qd"))
    bot = (parse_card("2c"), parse_card("3c"), parse_card("4c"),
           parse_card("5c"), parse_card("6c"))
    t, m, b, fouled = resolve_board(top, mid, bot)
    assert not fouled
    assert m[0] == QUADS
    assert m[1][0] == 12  # quads of A


def test_joker_resolver_falls_back_when_no_safe_assignment():
    # Top is trip-A (cat=TRIPS=3). Middle is at most a pair regardless of
    # joker substitution, and bottom is a flush so middle ≤ pair < trips
    # → always fouls.
    top = (parse_card("As"), parse_card("Ah"), parse_card("Ad"))
    mid = (parse_card("2s"), parse_card("3d"), parse_card("4c"),
           parse_card("9h"), JOKER_1)
    bot = (parse_card("7c"), parse_card("8c"), parse_card("9c"),
           parse_card("Tc"), parse_card("Jc"))
    t, m, b, fouled = resolve_board(top, mid, bot)
    assert fouled
    assert is_foul(top, mid, bot)


def test_joker_resolver_no_jokers_unchanged():
    # No-joker fast path must match the per-row evaluators exactly.
    top = (parse_card("Qs"), parse_card("Qd"), parse_card("5c"))
    mid = (parse_card("As"), parse_card("Kd"), parse_card("Ac"),
           parse_card("Kh"), parse_card("2s"))
    bot = (parse_card("Ts"), parse_card("Td"), parse_card("Tc"),
           parse_card("Th"), parse_card("9s"))
    t, m, b, fouled = resolve_board(top, mid, bot)
    assert not fouled
    assert (t, m, b) == (evaluate_3(top), evaluate_5(mid), evaluate_5(bot))


def test_joker_score_match_uses_resolved_assignment():
    # Two boards: A has a joker on top that, if greedy-maxed, would foul;
    # the resolver's pick lets A play a clean (non-fouled) board.
    a = Board(
        top=(parse_card("Ks"), parse_card("Kh"), JOKER_1),
        middle=(parse_card("As"), parse_card("Ah"), parse_card("Qd"),
                parse_card("Jd"), parse_card("Td")),
        bottom=(parse_card("2c"), parse_card("3c"), parse_card("4c"),
                parse_card("5c"), parse_card("6c")),
    )
    b = Board(
        top=(parse_card("2h"), parse_card("3h"), parse_card("4h")),
        middle=(parse_card("5h"), parse_card("6h"), parse_card("7h"),
                parse_card("8d"), parse_card("9d")),
        bottom=(parse_card("Tc"), parse_card("Jc"), parse_card("Qc"),
                parse_card("Kc"), parse_card("Ac")),
    )
    sb = score_match(a, b)
    # A's board is non-fouled thanks to joker-aware resolution.
    assert not sb.a_foul
