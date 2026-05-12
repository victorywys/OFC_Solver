from engine.cards import parse_cards
from engine.evaluator import evaluate_3, evaluate_5
from engine.royalties import (
    DEFAULT_ROYALTIES,
    STANDARD_PINEAPPLE,
    royalty_bottom,
    royalty_middle,
    royalty_top,
)


def test_top_pair_royalties_default():
    p55 = evaluate_3(parse_cards("5s 5d 2c"))
    p66 = evaluate_3(parse_cards("6s 6d 2c"))
    pAA = evaluate_3(parse_cards("As Ad 2c"))
    assert royalty_top(p55) == 1
    assert royalty_top(p66) == 2
    assert royalty_top(pAA) == 2  # default per spec: any 66+ = 2


def test_top_trips_royalty_default():
    t = evaluate_3(parse_cards("Qs Qd Qc"))
    assert royalty_top(t) == 4


def test_middle_royalty_default_table():
    assert royalty_middle(evaluate_5(parse_cards("As Ks Qs Js Ts"))) == 20  # royal
    assert royalty_middle(evaluate_5(parse_cards("9s 8s 7s 6s 5s"))) == 20  # straight flush
    assert royalty_middle(evaluate_5(parse_cards("As Ad Ac Ah Ks"))) == 12  # quads
    assert royalty_middle(evaluate_5(parse_cards("As Ad Ac Kh Ks"))) == 8   # full house
    assert royalty_middle(evaluate_5(parse_cards("As Js 8s 4s 2s"))) == 4   # flush
    assert royalty_middle(evaluate_5(parse_cards("Ts Jd Qc Kh As"))) == 2   # straight
    assert royalty_middle(evaluate_5(parse_cards("As Ad Ac Kh 9s"))) == 2   # trips
    assert royalty_middle(evaluate_5(parse_cards("As Ad Kc Kh 9s"))) == 1   # two pair


def test_bottom_royalty_default_table():
    assert royalty_bottom(evaluate_5(parse_cards("As Ks Qs Js Ts"))) == 25
    assert royalty_bottom(evaluate_5(parse_cards("9s 8s 7s 6s 5s"))) == 12
    assert royalty_bottom(evaluate_5(parse_cards("As Ad Ac Ah Ks"))) == 8
    assert royalty_bottom(evaluate_5(parse_cards("As Ad Ac Kh Ks"))) == 4
    assert royalty_bottom(evaluate_5(parse_cards("As Js 8s 4s 2s"))) == 2
    assert royalty_bottom(evaluate_5(parse_cards("Ts Jd Qc Kh As"))) == 1
    assert royalty_bottom(evaluate_5(parse_cards("As Ad Ac Kh 9s"))) == 0
    assert royalty_bottom(evaluate_5(parse_cards("As Ad Kc Kh 9s"))) == 0


def test_standard_pineapple_preset():
    # standard top: 66=1, AA=9, 222=10, AAA=22
    p66 = evaluate_3(parse_cards("6s 6d 2c"))
    pAA = evaluate_3(parse_cards("As Ad 2c"))
    t22 = evaluate_3(parse_cards("2s 2d 2c"))
    tAA = evaluate_3(parse_cards("As Ad Ac"))
    assert royalty_top(p66, STANDARD_PINEAPPLE) == 1
    assert royalty_top(pAA, STANDARD_PINEAPPLE) == 9
    assert royalty_top(t22, STANDARD_PINEAPPLE) == 10
    assert royalty_top(tAA, STANDARD_PINEAPPLE) == 22

    # standard bottom: royal = 25, SF = 15, quads = 10
    rf = evaluate_5(parse_cards("As Ks Qs Js Ts"))
    sf = evaluate_5(parse_cards("9s 8s 7s 6s 5s"))
    q = evaluate_5(parse_cards("As Ad Ac Ah Ks"))
    assert royalty_bottom(rf, STANDARD_PINEAPPLE) == 25
    assert royalty_bottom(sf, STANDARD_PINEAPPLE) == 15
    assert royalty_bottom(q, STANDARD_PINEAPPLE) == 10


def test_default_royalties_singleton_is_safe():
    # ensure we have a usable default and it doesn't error on a low-rank pair top
    p22 = evaluate_3(parse_cards("2s 2d 5c"))
    assert royalty_top(p22, DEFAULT_ROYALTIES) == 1
