#!/usr/bin/python3

import os.path
import sys

import cv2
import numpy as np
import argparse

KERNEL = np.ones((5, 5), np.uint8)


# 1.  grayscale
def gray(image) -> np.ndarray[any, any]:
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# 2. blur
def blurred(image) -> np.ndarray[any, any]:
    return cv2.GaussianBlur(image, (7, 7), 0)


# 3. bilateral_filtered
def bilateral_filtered(image) -> np.ndarray[any, any]:
    return cv2.bilateralFilter(image, 9, 75, 75)


# 4. edges
def edges(image) -> np.ndarray[any, any]:
    return cv2.Canny(image, 100, 200)


# 5. binary
def binary(image) -> np.ndarray[any, any]:
    _gray = gray(image)
    _, _binary = cv2.threshold(_gray, 127, 255, cv2.THRESH_BINARY)
    return _binary


# 6. dilate
def dilated(image) -> np.ndarray[any, any]:
    _binary = binary(image)
    return cv2.dilate(_binary, KERNEL, iterations=2)


# 7. erode
def eroded(image) -> np.ndarray[any, any]:
    _binary = binary(image)
    return cv2.erode(_binary, KERNEL, iterations=2)


# 8. median blur
def median_blurred(image) -> np.ndarray[any, any]:
    return cv2.medianBlur(image, 5)


# 9. horizontal flip
def flipped_horizontally(image) -> np.ndarray[any, any]:
    return cv2.flip(image, 1)


# 10. flip vertically
def flipped_vertically(image) -> np.ndarray[any, any]:
    return cv2.flip(image, 0)


# 11. increase contrast (clahe)
def clahe_applied(image) -> np.ndarray[any, any]:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    _gray = gray(image)
    return clahe.apply(_gray)


# 12. rotate
def rotated(image) -> np.ndarray[any, any]:
    rows, cols = image.shape[:2]
    center = (cols // 2, rows // 2)
    matrix = cv2.getRotationMatrix2D(center, 45, 1.0)
    return cv2.warpAffine(image, matrix, (cols, rows))


# 13. resize
def resized_half(image) -> np.ndarray[any, any]:
    rows, cols = image.shape[:2]
    return cv2.resize(image, (cols // 2, rows // 2))


# 14. increase saturation
def more_saturated(image) -> np.ndarray[any, any]:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv[..., 1] = cv2.multiply(hsv[..., 1], 2)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


# 15. decrease brightness
def darker(image) -> np.ndarray[any, any]:
    return cv2.addWeighted(image, 0.5, np.zeros(image.shape, image.dtype), 0, 0)


# 16. overlay text
def text_overlay(image) -> np.ndarray[any, any]:
    _text_overlay = image.copy()
    cv2.putText(_text_overlay, "Wowowowowow!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    return _text_overlay


# 17. convert to rgb
def rgb(image) -> np.ndarray[any, any]:
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


# 18. invert image color
def inverted(image) -> np.ndarray[any, any]:
    return cv2.bitwise_not(image)


# 19. modify column color
def modified_column(image) -> np.ndarray[any, any]:
    _modified_column = image.copy()
    _modified_column[:, 100] = [0, 255, 0]  # Change column at index 100 to green
    return _modified_column


# 20. add circle overlay
def add_circle(image) -> np.ndarray[any, any]:
    # Specify the circle parameters
    center_coordinates = (image.shape[1] // 2, image.shape[0] // 2)
    radius = 50
    color = (0, 255, 0)  # BGR
    thickness = -1

    return cv2.circle(image, center_coordinates, radius, color, thickness)


def transform_image(input_path):
    # Load the input image
    image = cv2.imread(input_path)

    available_transformations = {
        "gray": gray,
        "blurred": blurred,
        "bilateral_filtered": bilateral_filtered,
        "edges": edges,
        "binary": binary,
        "dilated": dilated,
        "eroded": eroded,
        "median_blurred": median_blurred,
        "flipped_horizontally": flipped_horizontally,
        "flipped_vertically": flipped_vertically,
        "clahe_applied": clahe_applied,
        "rotated": rotated,
        "resized_half": resized_half,
        "more_saturated": more_saturated,
        "darker": darker,
        "text_overlay": text_overlay,
        "rgb": rgb,
        "inverted": inverted,
        "modified_column": modified_column,
        "add_circle": add_circle
    }

    truncated_path = os.path.dirname(input_path)
    truncated_path = os.path.join(truncated_path, "output")

    if not os.path.exists(truncated_path):
        os.makedirs(truncated_path)

    print(f'Saving output in {truncated_path}/')
    for key in available_transformations:
        func = available_transformations.get(key)
        mod_output_path = os.path.join(truncated_path, f'image_{key}.jpg')
        output_image = func(image)
        cv2.imwrite(mod_output_path, output_image)
        print(f"Modification {key} saved as {mod_output_path}")


def usage():
    print(f'Usage: {sys.argv[0]} --input <path>')


if __name__ == "__main__":
    parser = argparse.ArgumentParser("image_processing")
    parser.add_argument(
        "--input", help="Path to input image", type=str
    )
    args = parser.parse_args()

    try:
        if not os.path.exists(args.input):
            usage()
            raise Exception(f'Could not find input path: {args.input}')
    except TypeError:
        usage()
        raise

    transform_image(args.input)
