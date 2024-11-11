COLOR_SIMILARITY_THRESHOLD = 10
GREEN_SCREEN_THRESHOLD = 0.7

def color_distance(color1, color2):
    """
    Compute the Euclidean distance between two colors.

    Inputs:
        color1 (Tuple[int, int, int]): a color
        color2 (Tuple[int, int, int]): a color

    Returns:
        float: The Euclidean distance between the two colors.
    """
    sum_of_squares = sum((x - y) ** 2 for x, y in zip(color1, color2))
    return sum_of_squares ** 0.5

def count_close_colors(img, target_color):
    """
    Count the number of cells that hold a color close to
    the target color.  Count within the region only, if
    specified.

    Inputs:
        img [List[List[Tuple(int, int, int)]]: an image
        target_color [Tuple[int, int, int]]: a color

    Returns [int]: the number of cells that are within threshold
      distance of the target color
    """
    count = 0
    for row in img:
        for color in row:
            distance = color_distance(color, target_color)
            if distance <= COLOR_SIMILARITY_THRESHOLD:
                count += 1
    return count


def gen_identity_mask(N, M):
    """
    Generate a mask with N rows and M colors where every
    entry is True.

    Inputs:
        N [int]: the number of rows
        M [int]: the number of columns

    Returns [List[List[bool]]: an N x M identity mask
    """
    return [[True] * M for i in range(N)]

def combine_using_mask(image1, image2, mask):
    """
    Creates an image where the color in location (i, j) of the result
    is the color from location (i, j) in image1 if the mask value in
    location (i, j) is True and color from location (i, j) in image2
    otherwise.

    Inputs:
        image1 [List[List[Tuple[int, int, int]]]]: the first image
        image2 [List[List[Tuple[int, int, int]]]]: the second image

    Returns [List[List[Tuple[int, int, int]]]]: the combined image.
    """
    res_image = []
    for row_mask, row1, row2 in zip(mask, image1, image2):
        res_row = []
        for is_use_image1, pixel1, pixel2 in zip(row_mask, row1, row2):
            res_pixel = pixel1 if is_use_image1 else pixel2
            res_row.append(res_pixel)
        res_image.append(res_row)
    return res_image


def flip_region_in_mask(mask, region):
    """
    Flip the mask locations in a region in-place.

    Inputs:
        img [List[List[Tuple(int, int, int)]]: an image
        region [Tuple[Tuple[int, int], Tuple[int, int]]]: a region defined
          using two locations.
    """
    top_left, bottom_right = region
    for x in range(top_left[0], bottom_right[0] + 1):
        for y in range(top_left[1], bottom_right[1] + 1):
            mask[x][y] = not mask[x][y]

def loc_radius_to_region(grid, loc, radius):
    """
    Given a grid, a location within that grid, and a radius,
    compute the two-location representation of the region with the specified
    radious around the location.

    Inputs:
        grid [List[List[Any]]]: the grid of interest
        loc [Tuple[int, int]]: the location
        radius [int]: the radius

    Returns [Tuple[Tuple[int, int], Tuple[int, int]]]: the
        two-location representation of the region.

    """
    top_left = (max(loc[0] - radius, 0), max(loc[1] - radius, 0))
    bottom_right = (min(loc[0] + radius, len(grid) - 1), \
                     min(loc[1] + radius, len(grid[0]) - 1))
    return top_left, bottom_right

def count_colors_in_region(img, target_color, region):
    """
    Count the number of colors close to the target color within
    the specified region.

    Inputs:
        img [List[List[Tuple[int, int, int]]]]: an image
        target_color [Tuple[int, int, int]]: the target color
        region [Tuple[Tuple[int, int], Tuple[int, int]]]: the
        region defined by two locations

    Returns:
        [int]: the number of colors close to the target color within the region
    """
    top_left, bottom_right = region
    count = 0
    for x in range(top_left[0], bottom_right[0] + 1):
        for y in range(top_left[1], bottom_right[1] + 1):
            if color_distance(img[x][y], target_color) <= \
                COLOR_SIMILARITY_THRESHOLD:
                count += 1
    return count

def count_pixels_in_region(region):
    """
    Count the number of pixels in the specified region.

    Inputs:
        region [Tuple[Tuple[int, int], Tuple[int, int]]]: the region
        defined by two locations

    Returns:
        [int]: the number of pixels in the region
    """
    top_left, bottom_right = region
    width = bottom_right[0] - top_left[0] + 1
    height = bottom_right[1] - top_left[1] + 1
    return width * height

def is_region_above_threshold(img, target_color, region):
    """
    Check if more than 50% of the pixels in the specified
    region are close to the target color.

    Inputs:
        img [List[List[Tuple[int, int, int]]]]: an image
        target_color [Tuple[int, int, int]]: the target color
        region [Tuple[Tuple[int, int], Tuple[int, int]]]: the region
        defined by two locations

    Returns:
        [bool]: True if the region meets the threshold, False otherwise
    """
    total_pixels = count_pixels_in_region(region)
    close_color_count = count_colors_in_region(img, target_color, region)
    return close_color_count/total_pixels > 0.5

def gen_green_screen_mask(img, green_screen_color, radius):
    """
    Generate mask where the mask entry corresponding to a pixel
    will be True if the pixel is close to the green screen color AND
    more than 50% of the cells in its region are close to the green
    screen color and False otherwise.

    Inputs:
        img [List[List[Tuple[int, int, int]]]]: an RGB image
        green_screen_color [Tuple[int, int, int]]: the RGB color of the
          green screen
        radius [int]: the radius defining a region.

    Returns [List[List[bool]]: a mask where cells that close to the
      green screen colors and mostly near cells with colors close to
      the green screen color are True and all other cells are False.

    """
    mask = []
    for i in range(len(img)):
        row_mask = []
        for j in range(len(img[0])):
            region = loc_radius_to_region(img, (i, j), radius)
            close_to_target = color_distance(img[i][j], green_screen_color) <= COLOR_SIMILARITY_THRESHOLD
            meets_threshold = is_region_above_threshold(img, green_screen_color, region)
            row_mask.append(close_to_target and meets_threshold)
        mask.append(row_mask)
    return mask
