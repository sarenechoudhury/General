def cards_equal(card1, card2):
    """
    Determine whether or not two cards are equal.

    Input:
        card1 [tuple]: first card
        card2 [tuple]: second card

    Returns [boolean]: True if the cards are equal, False otherwise.
    """
    return list(card1) == list(card2)


def create_deck(size):
    """
    Create a deck of cards of a given size. A deck is made of
        cards with values 1 to size of black cards and 1 to size of
        red cards.

    Input:
        size [int]: the size of the deck

    Returns [List[tuple]]: A deck of cards.
    """

    deck_black = list(range(1, (size+1)))
    deck_red = list(range(1, (size+1)))
    deck_black.extend(deck_red)
    deck_black.sort()
    tf_list = []
    i = 0
    while i <= size:
        tf_list.append(True)
        tf_list.append(False)
        i += 1
    final = list(zip(deck_black, tf_list))
    return final



def cut_deck(deck):
    """
    Split the deck of cards in half. If the deck has an odd
        number of cards, the second half should get the
        extra card.

    Input:
        deck [List[tuple]]: the deck

    Returns [List[tuple]], [List[tuple]]: Two halves of the deck.
    """
    first = deck[0:int((len(deck))/2)]
    second = deck[int((len(deck))/2):]
    final = (first, second)
    return final



def flip_color(cards):
    """
    Flip the color (black to red, red to black) of each card
        in a list of cards in-place.

    Input:
        cards [List[tuple]]: a list of cards

    Returns [None]: Nothing, modifies the list of cards in-place.
    """
    for i, card in enumerate(cards):
        id, color = card
        cards[i] = (id, not color)




def change_value(cards, num):
    """
    Create a list of cards with each card's original value
        incremented by a number.

    Input:
        cards [List[tuple]]: a list of cards
        num [int]: the number to increment

    Returns [List[tuple]]: A list of cards with new values.
    """
    final = []
    for i in cards:
        id, color = i
        final.append((id + num, color))
    return final



def split_by_color(cards):
    """
    Split the black and red cards in a list of cards
        into two separate lists.

    Input:
        cards [List[tuple]]: a list of cards

    Returns [List[tuple]], [List[tuple]]: A list of black cards
        and a list of red cards.
    """
    black = []
    red = []
    for i in cards:
        id, color = i
        if color:
            black.append(i)
        else:
            red.append(i)
    final = black, red 
    return final


def lose_points(cards, target):
    """
    Change the points on the target card to zero in-place.

    Input:
        cards [List[tuple]]: a list of cards
        target [tuple]: the target card

    Returns [None]: Nothing, modifies the list of cards in-place.
    """
    for i, card in enumerate(cards):
        id, color = card
        if card == target:
            cards[i] = (int(not id), color)


           
def remove_from_middle(cards, start, end):
    """
    Create a list of cards from the middle of a list of
        cards, from start to end, inclusive. Create a list
        of the remaining cards.

    Input:
        cards [List[tuple]]: a list of cards
        start [int]: the starting card
        end [int]: the ending card

    Returns [List[tuple]], [List[tuple]]: A list of cards from the
        middle and a list of the remaining cards.
    """
    assert start >= 0
    assert end < len(cards)
    assert start <= end
    assert len(cards) > 0

    removed = cards[start:end+1]
    remaining = cards[:start] + cards[end+1:]
    final = removed, remaining
    return final


def count_longest_run(cards):
    """
    Find the number of cards in the longest run of
        cards of the same color.

    Input:
        cards [List[tuple]]: a list of cards

    Returns [int]: The number of the longest run of
        cards of the same color.
    """
    if not cards:
        return 0
    longest = 0
    count = 1
    for i in range(len(cards)-1):
        if cards[i][1] == cards[i+1][1]:
            count += 1
        else:
            longest = max(longest, count)
            count = 1

    longest = max(longest, count)
    return longest    
