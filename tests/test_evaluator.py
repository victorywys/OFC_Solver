from engine.cards import parse_cards
from engine.evaluator import (
    FLUSH,
    FULL_HOUSE,
    HIGH_CARD,
    PAIR,
    QUADS,
    ROYAL_FLUSH,
    STRAIGHT,
    STRAIGHT_FLUSH,
    TRIPS,
    TWO_PAIR,
    evaluate_3,
    evaluate_5,
)


# ---------- 5-card no joker ----------
def _ev(s):
    return evaluate_5(parse_cards(s))


def test_high_card():
    r = _ev("As Kd Qc Jh 9s")
    assert r[0] == HIGH_CARD
    assert r[1] == (12, 11, 10, 9, 7)


def test_pair():
    r = _ev("As Ad Kc Jh 9s")
    assert r[0] == PAIR
    assert r[1][0] == 12  # pair of A
    # kickers descending
    assert r[1][1:] == (11, 9, 7)


def test_two_pair():
    r = _ev("As Ad Kc Kh 9s")
    assert r[0] == TWO_PAIR
    assert r[1] == (12, 11, 7)


def test_trips():
    r = _ev("As Ad Ac Kh 9s")
    assert r[0] == TRIPS
    assert r[1] == (12, 11, 7)


def test_straight_high():
    r = _ev("Ts Jd Qc Kh As")
    assert r[0] == STRAIGHT
    assert r[1] == (12,)


def test_straight_wheel():
    r = _ev("As 2d 3c 4h 5s")
    assert r[0] == STRAIGHT
    assert r[1] == (3,)  # rank index of 5


def test_flush():
    r = _ev("As Js 8s 4s 2s")
    assert r[0] == FLUSH
    assert r[1] == (12, 9, 6, 2, 0)


def test_full_house():
    r = _ev("As Ad Ac Kh Ks")
    assert r[0] == FULL_HOUSE
    assert r[1] == (12, 11)


def test_quads():
    r = _ev("As Ad Ac Ah Ks")
    assert r[0] == QUADS
    assert r[1] == (12, 11)


def test_straight_flush():
    r = _ev("9s 8s 7s 6s 5s")
    assert r[0] == STRAIGHT_FLUSH
    assert r[1] == (7,)


def test_steel_wheel():
    r = _ev("As 2s 3s 4s 5s")
    assert r[0] == STRAIGHT_FLUSH
    assert r[1] == (3,)


def test_royal_flush():
    r = _ev("As Ks Qs Js Ts")
    assert r[0] == ROYAL_FLUSH


def test_no_straight_with_pair():
    # 9 8 7 6 6 is two pair (oh wait, 6 6 is pair of 6, but with 9,8,7 high)
    # actually 9 8 7 6 + 6 = pair of 6 with 9,8,7 kickers
    r = _ev("9s 8d 7c 6h 6s")
    assert r[0] == PAIR


def test_full_house_ordering():
    a = _ev("As Ad Ac 2h 2s")
    b = _ev("Ks Kd Kc Ah As")
    # AAA22 > KKKAA in full house ordering (trips rank dominates)
    assert a > b


def test_flush_kicker_ordering():
    a = _ev("As Js 8s 4s 2s")
    b = _ev("As Js 8s 4s 3s")
    assert b > a  # 3 > 2 as last kicker


def test_straight_beats_trips():
    s = _ev("Ts Jd Qc Kh As")
    t = _ev("As Ad Ac Kh 9s")
    assert s > t


# ---------- 5-card with jokers ----------
def test_one_joker_makes_quads():
    # 4 aces + joker -> joker becomes 5th card; best is quads w/ best kicker
    r = evaluate_5(parse_cards("As Ad Ac Ah *1"))
    assert r[0] == QUADS
    assert r[1][0] == 12


def test_one_joker_makes_royal_flush():
    r = evaluate_5(parse_cards("As Ks Qs Js *1"))
    assert r[0] == ROYAL_FLUSH


def test_one_joker_full_house():
    r = evaluate_5(parse_cards("As Ad Kc Kh *1"))
    # joker fills to make AAAKK > AAAA? no, with 1 joker we have AA KK + 1
    # best is AAA KK (joker as A) full house, OR AA KKK (joker as K).
    # AAAKK > KKKAA so we want AAA KK
    assert r[0] == FULL_HOUSE
    assert r[1] == (12, 11)


def test_two_jokers_make_quads_or_better():
    # 3 of a kind plus 2 jokers -> 5 of a kind impossible; best legal: quads or SF
    # AAA + two jokers: jokers can be any 2 distinct cards.
    # Options:
    #   * make AAAA + kicker -> QUADS (great)
    #   * make AAAKK -> full house (worse)
    # Even better: A A A + two jokers matching same suit -> straight flush?
    # cards: As Ad Ac + jokers => flush requires same suit; we have 3 different
    # suits already, so flush impossible. So quads of A.
    r = evaluate_5(parse_cards("As Ad Ac *1 *2"))
    assert r[0] == QUADS
    assert r[1][0] == 12


def test_two_jokers_with_two_aces_can_be_quads():
    r = evaluate_5(parse_cards("As Ad Kc *1 *2"))
    # best: quads of A with kicker K
    assert r[0] == QUADS
    assert r[1] == (12, 11)


def test_two_jokers_can_make_royal_flush():
    # A K Q + two jokers all same suit -> royal flush
    r = evaluate_5(parse_cards("As Ks Qs *1 *2"))
    assert r[0] == ROYAL_FLUSH


# ---------- 3-card top ----------
def test_top_high_card():
    r = evaluate_3(parse_cards("As Kd Qc"))
    assert r[0] == HIGH_CARD


def test_top_pair():
    r = evaluate_3(parse_cards("As Ad Qc"))
    assert r[0] == PAIR
    assert r[1] == (12, 10)


def test_top_trips():
    r = evaluate_3(parse_cards("As Ad Ac"))
    assert r[0] == TRIPS
    assert r[1] == (12,)


def test_top_with_one_joker_makes_pair_of_higher():
    r = evaluate_3(parse_cards("Qs Jd *1"))
    assert r[0] == PAIR
    assert r[1] == (10, 9)


def test_top_with_one_joker_paired_real_makes_trips():
    r = evaluate_3(parse_cards("Qs Qd *1"))
    assert r[0] == TRIPS
    assert r[1] == (10,)


def test_top_with_two_jokers_makes_trips():
    r = evaluate_3(parse_cards("Qs *1 *2"))
    assert r[0] == TRIPS
    assert r[1] == (10,)


def test_cross_row_compare_top_trips_lt_middle_full_house():
    top = evaluate_3(parse_cards("Qs Qd Qc"))      # TRIPS = 3
    mid = evaluate_5(parse_cards("As Ad Ac Kh Ks"))  # FULL_HOUSE = 6
    assert mid > top
