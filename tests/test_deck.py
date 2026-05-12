from engine.cards import NUM_CARDS, full_deck
from engine.deck import Deck, remaining_after


def test_deck_has_54_cards_with_jokers():
    d = Deck(seed=0)
    assert len(d) == NUM_CARDS == 54


def test_deck_no_jokers():
    d = Deck(seed=0, include_jokers=False)
    assert len(d) == 52


def test_deck_seed_reproducible():
    a = Deck(seed=42).deal(13)
    b = Deck(seed=42).deal(13)
    assert a == b


def test_deal_shrinks():
    d = Deck(seed=1)
    n = len(d)
    d.deal(5)
    assert len(d) == n - 5


def test_remove_known():
    d = Deck(seed=2)
    cards = d.deal(13)
    # already-dealt cards should not be in deck
    assert all(c not in d.cards() for c in cards)


def test_remaining_after_helper():
    known = full_deck()[:5]
    rem = remaining_after(known)
    assert len(rem) == NUM_CARDS - 5
    assert all(c not in rem for c in known)
