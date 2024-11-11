def add_one_and_multiply(a, x):
    """
    Add 1 to a, and multiply by x.

    Inputs:
        a (int): an integer value
        x (int): another integer value

    Returns (int): The result of adding 1 to a and then multiplying by x.
    """
    b = a + 1
    return b * x


def same_parity(a, b):
    """
    Determine whether or not two integers have the same parity.
    Integers have the same parity if they are both even or both odd.

    Inputs:
        a (int): an integer value
        b (int): another integer value

    Returns (boolean): True if a and b have the same parity, False otherwise.
    """
    result = (((a%2)+(b%2) == 0) or ((a%2)+(b%2) == 2))
    return result


def fractional_part(n, d):
    """
    Find the fractional part of n/d as a float.
    You may assume that d is not zero.
    Hint: Use just the arithmetic operators +, -, *, /, %, //

    Inputs:
        n (int): the numerator
        d (int): the denominator

    Returns (float): The fractional part of n divided by d.
    """
    x = n/d
    result = x - int(x)
    return result


def peep(p, e):
    """
    Determine whether or not peep = pp^e.

    Inputs:
      p (int): the first digit
      e (int): the second digit

    Returns (boolean): True if peep = pp^e, False otherwise.
    """
    word = (p*1000) + (e*100) + (e*10) + p
    ppe = ((p*10) + p)**e
    result = (word == ppe)
    return result


def is_valid_color(r, g, b):
    """
    Determine whether or not an RGB color is valid.
    A color is valid if each R, G, B component is an integer between
    0 and 255 (inclusive).

    Inputs:
        r (number): red component
        g (number): green component
        b (number): blue component

    Returns (boolean): True if the color is valid, False otherwise.
    """
    result = (0 <= r <= 255) and (0 <= g <= 255) and (0 <= b <= 255) \
    and isinstance(r, int) and isinstance(r, int) and isinstance(r, int)
    return result


def is_grey(r, g, b):
    """
    Determine whether or not an RGB color is grey.
    An RGB color is grey if all R, G, B values are equal.

    Inputs:
        r (int): red component
        g (int): green component
        b (int): blue component

    Returns (boolean): True if the color is grey, False otherwise.
    """
    result = (is_valid_color(r, g, b)) and (r == g == b)
    return result


def make_greyscale(r, g, b):
    """
    Calculate the greyscale version of an RGB color by averaging
    the red, green, and blue components.

    Inputs:
        r (int): red component
        g (int): green component
        b (int): blue component

    Returns (int): The greyscale value of the color.
    """
    a = (is_valid_color(r, g, b)) and (r + g + b)/3
    result = round(a)
    return result


def brightness(r, g, b):
    """
    Calculate the perceived brightness of an RGB color using the
    formula: (0.21 x R) + (0.72 x G) + (0.07 x B).

    Inputs:
        r (int): red component
        g (int): green component
        b (int): blue component

    Returns (float): The perceived brightness of a color.
    """
    result = (is_valid_color(r, g, b)) and \
    ((0.21 * r) + (0.72 * g) + (0.07 * b))
    return result


def is_bright(r, g, b):
    """
    Determine whether or not an RGB color is "bright".
    An RGB color is bright if its perceived brightness strictly greater
    than 200.

    Inputs:
        r (int): red component
        g (int): green component
        b (int): blue component

    Returns (boolean): True if color is bright, False otherwise.
    """
    result = (is_valid_color(r, g, b)) and (brightness(r, g, b) > 200)
    return result


def is_primary(r, g, b):
    """
    Determine whether or not an RGB color is a primary color.
    An RGB color is a primary color if one of its RGB components is
    255 and the rest are 0.

    Inputs:
        r (int): red component
        g (int): green component
        b (int): blue component

    Returns (boolean): True if the color is a pimary color, false otherwise.
    """
    result = (is_valid_color(r, g, b)) and ((r == 255) and \
    (g == 0) and (b == 0)) or ((r == 0) and (g == 255) and \
    (b == 0)) or ((r == 0) and (g == 0) and (b == 255))
    return result
