ZERO_SCORE = 0
ONE_SCORE = 3
SOME_SCORE = 5
ALL_SCORE = 10

def report_score(num_matches, target):
    """
    Report the score for a given number of matches.

    Inputs:
        num_matches [int]: the number of matches
        target [int]: the target number of matches

    Returns [int]: the score for the specified number of
      matches and target.
    """
    assert target > 0
    result = 0
    if num_matches == 0:
        result = ZERO_SCORE
    if num_matches == 1 and target != 1:
        result = ONE_SCORE
    if 1 < num_matches < target:
        result = SOME_SCORE
    if num_matches == target:
        result = ALL_SCORE
    return result


def count_cards(cards, lb, ub):
    """
    Count the number of cards in a given list
    that are within a given range.

    Inputs:
        cards [List[int]]: the list of cards
        lb [int]: the lower bound for the cards (inclusive)
        ub [int]: the upper bound for the cards (inclusive)

    Returns [int]: the number of cards
    that are within the inclusive range
    """
    assert lb <= ub
    count = 0
    for i in cards:
        if lb <= i <= ub:
            count += 1
    return count


def has_card_v1(cards, target_card):
    """
    Determines if the given card is in the given list

    Inputs:
    cards [List[int]]: the list of cards
    target_card [int]: the target card

    Returns [bool]: whether the target is in the list or not

    """
    result = False
    for i in cards:
        if i == target_card:
            result = True
            break
    return result



def has_card_v2(cards, target_card):
    """
    Determines if the given card is in the given list

    Inputs:
    cards [List[int]]: the list of cards
    target_card [int]: the target card

    Returns [bool]: whether the target is in the list or not
    """
    for i in cards:
        if i == target_card:
            return True
    return False


def has_at_least_one(cards_on_hand, cards_to_match):
    """
    Determines if two given lists contain at least
    one of the same card

    Inputs:
    cards_on_hand [List[int]]: the list of cards
    cards_to_match [List[int]]: the list of cards being compared

    Returns [bool]: whether the two lists contain at least
    one of the same card or not
    """
    # assume there is at least one card to match
    # and that the cards to match are unique
    N = len(cards_to_match)
    assert N >= 1
    assert len(set(cards_to_match)) == N
    for target in cards_on_hand:
        if has_card_v1(cards_to_match, target):
            return True
    return False



def has_all(cards_on_hand, cards_to_match):
    """
    Checks if the list of cards on hand contains every
    card in the other list of cards

    Inputs:
    cards_on_hand [List[int]]: the list of cards
    cards_to_match [List[int]]: the list of cards being compared

    Returns [bool]: whether cards on hand contains every
    card in cards to match

    """
    # assume there is at least one card to match
    # and that the cards to match are unique
    N = len(cards_to_match)
    assert N >= 1
    assert len(set(cards_to_match)) == N
    result = None
    for i in cards_to_match:
        if has_card_v1(cards_on_hand, i):
            result = True
        if not has_card_v1(cards_on_hand, i):
            return False
    return result



def has_exactly_one(cards_on_hand, cards_to_match):
    """
    Compares two given lists to see if they have
    exactly one shared element

    Inputs:
    cards_on_hand [List[int]]: the list of cards
    cards_to_match [List[int]]: the list of cards being compared

    Returns [bool]: whether the lists share exactly one
    element or not
    """
    # assume there is at least one card to match
    # and that the cards to match are unique
    N = len(cards_to_match)
    assert N >= 1
    assert len(set(cards_to_match)) == N
    count = 0
    for i in cards_to_match:
        if has_card_v1(cards_on_hand, i):
            count += 1
            if count > 1:
                return False
    if count == 1:
        return True
    return False


def compute_score(cards_on_hand, cards_to_match):
    """
    Report the score for a number of matches between two given lists

    Inputs:
    cards_on_hand [List[int]]: the list of cards
    cards_to_match [List[int]]: the list of cards being compared

    Returns [int]: the score correlated with the
    number of matches between the two lists
    """
    # assume there is at least one card to match
    # and that the cards to match are unique
    N = len(cards_to_match)
    assert N >= 1
    assert len(set(cards_to_match)) == N
    count = 0
    for i in cards_to_match:
        if has_card_v1(cards_on_hand, i):
            count += 1
    return report_score(count, len(cards_to_match))


def card_index_v1(cards, target_card):
    """
    Reports the index of the first occurence of a
    target if the target is found within the given list

    Inputs:
    cards [List[int]]: the list of cards
    target_card [int]: the given target

    Returns [int]: the index of the target within
    the list if it exists
    """
    for i, item in enumerate(cards):
        if item == target_card:
            return i
    return None


def card_index_v2(cards, target_card):
    """
    Reports the index of the first occurence of a
    target if the target is found within the given list

    Inputs:
    cards [List[int]]: the list of cards
    target_card [int]: the given target

    Returns [int]: the index of the target within
    the list if it exists
    """
    i = 0
    while i in range(len(cards)):
        for x in cards:
            i += 1
            if x == target_card:
                return i - 1
    return None
