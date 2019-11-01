import numpy as np

def to_pixel_no(coordinate):
        row = coordinate[1]
        column = coordinate[0]
        if row % 2 == 0:
            return 20 * row + column
        else:
            return 20 * row + 19 - column

def pixel_to_matrix_cord(pixel_no):
    row = int(pixel_no / 20)
    column = pixel_no % 20
    if row % 2 == 1:
        column = 19 - column

    return (column, row)  # x, y


def to_zigzag_layout(color_array):
    """
    counters the effect of the zigzag layout that every even row
    has its pixels in reversed order if the pixels are set in ascending order
    """
    c_zigzag = color_array.copy()
    c_zigzag[1::2] = c_zigzag[1::2, ::-1]
    return c_zigzag


def rotate(cords, center, rotrad=0):
    x = cords[0]
    y = cords[1]
    cx = center[0]
    cy = center[1]

    x = int(np.round(np.cos(rotrad)*(x-cx) + np.sin(rotrad)*(y-cy))) + cx
    y = int(np.round(-np.sin(rotrad)*(x-cx) + np.cos(rotrad)*(y-cy))) + cy

    return x, y
