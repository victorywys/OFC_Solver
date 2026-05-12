from engine.cards import (
    JOKER_1,
    JOKER_2,
    NUM_CARDS,
    RANK_A,
    RANK_T,
    SUIT_C,
    SUIT_S,
    card_rank,
    card_str,
    card_suit,
    full_deck,
    is_joker,
    make_card,
    parse_card,
    parse_cards,
)


def test_card_pack_unpack_roundtrip():
    for r in range(13):
        for s in range(4):
            c = make_card(r, s)
            assert card_rank(c) == r
            assert card_suit(c) == s
            assert not is_joker(c)


def test_parse_known():
    assert parse_card("As") == make_card(RANK_A, SUIT_S)
    assert parse_card("Tc") == make_card(RANK_T, SUIT_C)
    assert parse_card("*1") == JOKER_1
    assert parse_card("*2") == JOKER_2


def test_parse_cards_string():
    cs = parse_cards("As Kd 2c *1")
    assert cs[0] == parse_card("As")
    assert cs[-1] == JOKER_1


def test_card_str_inverse():
    for c in range(52):
        assert parse_card(card_str(c)) == c
    assert card_str(JOKER_1) == "*1"
    assert card_str(JOKER_2) == "*2"


def test_full_deck_size():
    assert len(full_deck()) == NUM_CARDS == 54
    assert len(set(full_deck())) == NUM_CARDS


def test_jokers_have_no_rank_suit():
    assert card_rank(JOKER_1) == -1
    assert card_suit(JOKER_2) == -1
    assert is_joker(JOKER_1) and is_joker(JOKER_2)
