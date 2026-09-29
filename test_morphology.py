import random
import unittest

from morphology import (
    StructuringElement,
    closing,
    dilate,
    erode,
    opening,
)


def reference_erode(image, element):
    height, width = len(image), len(image[0])
    result = [[None for _ in range(width)] for _ in range(height)]
    for row in range(height):
        for column in range(width):
            best = None
            for dr, dc in element.members:
                source_row, source_column = row + dr, column + dc
                if 0 <= source_row < height and 0 <= source_column < width:
                    candidate = image[source_row][source_column] - element.weights[(dr, dc)]
                    if best is None or candidate < best:
                        best = candidate
            result[row][column] = best
    return result


def reference_dilate(image, element):
    height, width = len(image), len(image[0])
    result = [[None for _ in range(width)] for _ in range(height)]
    for row in range(height):
        for column in range(width):
            best = None
            for dr, dc in element.members:
                source_row, source_column = row - dr, column - dc
                if 0 <= source_row < height and 0 <= source_column < width:
                    candidate = image[source_row][source_column] + element.weights[(dr, dc)]
                    if best is None or candidate > best:
                        best = candidate
            result[row][column] = best
    return result


class MorphologyTest(unittest.TestCase):
    def test_asymmetric_reflected_shape(self):
        image = [
            [0, 1, 0],
            [1, 1, 0],
            [0, 0, 0],
        ]
        element = StructuringElement(
            [
                (0, 0),
                (1, 1),
                (-1, -1),
            ]
        )
        self.assertEqual(erode(image, element, method="brute"), reference_erode(image, element))
        self.assertEqual(dilate(image, element, method="brute"), reference_dilate(image, element))
        self.assertEqual(dilate(image, element)[2][2], 1)
        self.assertEqual(erode(image, element)[0][0], 0)

    def test_boundary_clips_instead_of_padding_inf(self):
        image = [[0, 1, 1]]
        element = StructuringElement.rectangle(1, 3, anchor=(0, 0))
        self.assertEqual(erode(image, element), [[0, 1, 1]])
        self.assertEqual(dilate(image, element), [[0, 1, 1]])

        centered = StructuringElement.rectangle(1, 5)
        self.assertEqual(erode(image, centered), [[0, 0, 0]])
        self.assertEqual(dilate(image, centered), [[1, 1, 1]])

    def test_single_pixel_and_element_larger_than_image(self):
        large_flat = StructuringElement.rectangle(5, 5)
        large_nonflat = StructuringElement.gray_kernel(
            [
                [-3, -2, -1, 0, 1],
                [-2, -1, 0, 1, 2],
                [-1, 0, 0, 2, 3],
                [0, 1, 2, 3, 4],
                [1, 2, 3, 4, 5],
            ]
        )
        for element in (large_flat, large_nonflat):
            for image in ([[7]], [[1]]):
                self.assertEqual(erode(image, element), image)
                self.assertEqual(dilate(image, element), image)
                self.assertEqual(opening(image, element), image)
                self.assertEqual(closing(image, element), image)

        binary = [[1]]
        self.assertEqual(erode(binary, large_flat), [[1]])
        self.assertEqual(dilate(binary, large_flat), [[1]])

    def test_constant_images(self):
        for fill in (0, 1, 37):
            image = [[fill for _ in range(5)] for _ in range(4)]
            for element in (
                StructuringElement.rectangle(7, 3),
                StructuringElement.cross(3),
            ):
                self.assertEqual(erode(image, element), image)
                self.assertEqual(dilate(image, element), image)
                self.assertEqual(opening(image, element), image)
                self.assertEqual(closing(image, element), image)

    def test_two_by_two_binary_large_element(self):
        image = [
            [1, 0],
            [0, 0],
        ]
        element = StructuringElement.rectangle(5, 5)
        self.assertEqual(erode(image, element), [[0, 0], [0, 0]])
        self.assertEqual(dilate(image, element), [[1, 1], [1, 1]])
        self.assertEqual(opening(image, element), [[0, 0], [0, 0]])
        self.assertEqual(closing(image, element), [[1, 1], [1, 1]])

    def test_separable_matches_reference(self):
        random.seed(2909)
        cases = [
            (10, 7, 3, 5, True),
            (7, 11, 5, 3, False),
            (8, 8, 1, 7, True),
            (5, 5, 7, 1, False),
            (4, 4, 5, 5, True),
        ]
        for case_index, (height, width, se_height, se_width, binary) in enumerate(cases):
            image = [
                [random.randrange(2) if binary else random.randrange(100) for _ in range(width)]
                for _ in range(height)
            ]
            element = StructuringElement.rectangle(se_height, se_width)
            for operation, reference in (
                (erode, reference_erode),
                (dilate, reference_dilate),
            ):
                expected = reference(image, element)
                self.assertEqual(operation(image, element, method="separable"), expected)
                self.assertEqual(operation(image, element, method="brute"), expected)
                self.assertEqual(operation(image, element, method="auto"), expected)

    def test_additive_nonflat_rectangle_is_separable(self):
        row_weights = [-1, 0, 2]
        column_weights = [-2, 0, 4]
        kernel = [
            [row_weight + column_weight for column_weight in column_weights]
            for row_weight in row_weights
        ]
        element = StructuringElement.gray_kernel(kernel)
        self.assertIsNotNone(element.separable_components())

        random.seed(17)
        image = [[random.randrange(100) - 50 for _ in range(8)] for _ in range(6)]
        self.assertEqual(erode(image, element, method="separable"), reference_erode(image, element))
        self.assertEqual(dilate(image, element, method="separable"), reference_dilate(image, element))

    def test_arbitrary_shapes_match_reference(self):
        random.seed(4242)
        cross = StructuringElement.cross(2)
        diamond = StructuringElement.flat_kernel(
            [
                [False, False, True, False, False],
                [False, True, True, True, False],
                [True, True, True, True, True],
                [False, True, True, True, False],
                [False, False, True, False, False],
            ]
        )
        asymmetric = StructuringElement(
            [
                (0, 0),
                (1, 0),
                (0, 1),
                (-2, 1),
                (1, -2),
            ]
        )
        nonflat = StructuringElement.gray_kernel(
            [
                [None, -2, None],
                [1, 0, -3],
                [None, 2, None],
            ]
        )
        elements = (cross, diamond, asymmetric, nonflat)

        for index, element in enumerate(elements):
            if index != 0:
                self.assertIsNone(element.separable_components())
            image = [
                [random.randrange(2) if index % 2 else random.randrange(80) - 40 for _ in range(12)]
                for _ in range(9)
            ]
            for operation, reference in (
                (erode, reference_erode),
                (dilate, reference_dilate),
            ):
                expected = reference(image, element)
                self.assertEqual(operation(image, element), expected)
                self.assertEqual(operation(image, element, method="brute"), expected)

    def test_idempotence(self):
        random.seed(99)
        elements = [
            StructuringElement.rectangle(5, 3),
            StructuringElement.rectangle(4, 4, anchor=(1, 1)),
            StructuringElement.cross(2),
            StructuringElement(
                [
                    (0, 0),
                    (1, -1),
                    (-1, 1),
                    (2, 2),
                ]
            ),
            StructuringElement.gray_kernel(
                [
                    [-1, None, 2],
                    [None, 0, None],
                    [3, None, -2],
                ]
            ),
        ]
        for index, element in enumerate(elements):
            image = [
                [random.randrange(2) if index < 2 else random.randrange(120) - 60 for _ in range(13)]
                for _ in range(11)
            ]
            for method in ("brute", "separable", "auto"):
                if method == "separable" and element.separable_components() is None:
                    with self.assertRaises(ValueError):
                        opening(image, element, method=method)
                    with self.assertRaises(ValueError):
                        closing(image, element, method=method)
                    continue

                opened_once = opening(image, element, method)
                opened_twice = opening(opened_once, element, method)
                opened_thrice = opening(opened_twice, element, method)
                self.assertEqual(opened_once, opened_twice)
                self.assertEqual(opened_once, opened_thrice)

                closed_once = closing(image, element, method)
                closed_twice = closing(closed_once, element, method)
                closed_thrice = closing(closed_twice, element, method)
                self.assertEqual(closed_once, closed_twice)
                self.assertEqual(closed_once, closed_thrice)

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            StructuringElement([(1, 0)])
        with self.assertRaises(ValueError):
            StructuringElement.rectangle(0, 3)
        with self.assertRaises(ValueError):
            erode([[]], StructuringElement.cross(1))
        with self.assertRaises(ValueError):
            erode([[1]], (0, 0))


if __name__ == "__main__":
    unittest.main(verbosity=2)
