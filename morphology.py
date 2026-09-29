from collections import deque
from math import isfinite


ERODE = "erode"
DILATE = "dilate"


class MorphologyError(ValueError):
    pass


def _finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and isfinite(value)


class StructuringElement:
    def __init__(self, members):
        weights = {}
        try:
            iterator = iter(members)
        except TypeError as error:
            raise MorphologyError("members must be iterable") from error
        for member in iterator:
            if not isinstance(member, tuple) or len(member) not in (2, 3):
                raise MorphologyError("each member must be (dr, dc) or (dr, dc, weight)")
            if len(member) == 2:
                dr, dc = member
                weight = 0
            else:
                dr, dc, weight = member
            if not isinstance(dr, int) or isinstance(dr, bool):
                raise MorphologyError("row offset must be an integer")
            if not isinstance(dc, int) or isinstance(dc, bool):
                raise MorphologyError("column offset must be an integer")
            if not _finite_number(weight):
                raise MorphologyError("structuring-element weight must be a finite number")
            weights[(dr, dc)] = weight
        if (0, 0) not in weights:
            raise MorphologyError("the structuring element must contain its origin (0, 0)")
        self.weights = weights
        self.members = tuple(sorted(weights))
        self.flat = all(weight == 0 for weight in weights.values())

    @classmethod
    def from_kernel(cls, kernel, anchor=None, flat=False):
        if not isinstance(kernel, list) or not kernel or not all(isinstance(row, list) for row in kernel):
            raise MorphologyError("kernel must be a non-empty rectangular list of rows")
        height = len(kernel)
        width = len(kernel[0])
        if width == 0 or any(len(row) != width for row in kernel):
            raise MorphologyError("kernel must be a non-empty rectangle")
        if anchor is None:
            anchor = (height // 2, width // 2)
        if (
            not isinstance(anchor, tuple)
            or len(anchor) != 2
            or not all(isinstance(value, int) and not isinstance(value, bool) for value in anchor)
        ):
            raise MorphologyError("anchor must be an integer (row, column) tuple")
        anchor_row, anchor_column = anchor
        if not 0 <= anchor_row < height or not 0 <= anchor_column < width:
            raise MorphologyError("anchor is outside the kernel")

        members = []
        for row_index, row in enumerate(kernel):
            for column_index, entry in enumerate(row):
                if entry is None or entry is False:
                    continue
                if entry is True:
                    weight = 0
                elif flat:
                    if entry == 0:
                        continue
                    weight = 0
                elif _finite_number(entry):
                    weight = entry
                else:
                    raise MorphologyError("kernel entries must be None, False, True, or finite numbers")
                members.append((row_index - anchor_row, column_index - anchor_column, weight))

        element = cls(members)
        if (0, 0) not in element.weights:
            raise MorphologyError("the anchored kernel cell must belong to the structuring element")
        return element

    @classmethod
    def flat_kernel(cls, kernel, anchor=None):
        return cls.from_kernel(kernel, anchor=anchor, flat=True)

    @classmethod
    def gray_kernel(cls, kernel, anchor=None):
        return cls.from_kernel(kernel, anchor=anchor, flat=False)

    @classmethod
    def rectangle(cls, height, width, anchor=None):
        if not isinstance(height, int) or isinstance(height, bool) or height <= 0:
            raise MorphologyError("rectangle height must be a positive integer")
        if not isinstance(width, int) or isinstance(width, bool) or width <= 0:
            raise MorphologyError("rectangle width must be a positive integer")
        kernel = [[True for _ in range(width)] for _ in range(height)]
        return cls.from_kernel(kernel, anchor=anchor)

    @classmethod
    def cross(cls, radius):
        if not isinstance(radius, int) or isinstance(radius, bool) or radius < 0:
            raise MorphologyError("cross radius must be a non-negative integer")
        size = radius * 2 + 1
        kernel = [[False for _ in range(size)] for _ in range(size)]
        for index in range(size):
            kernel[radius][index] = True
            kernel[index][radius] = True
        return cls.from_kernel(kernel, anchor=(radius, radius))

    def separable_components(self):
        rows = sorted({dr for dr, _ in self.members})
        columns = sorted({dc for _, dc in self.members})
        if set(self.members) != {(dr, dc) for dr in rows for dc in columns}:
            return None

        origin_weight = self.weights[(0, 0)]
        vertical_weights = {dr: self.weights[(dr, 0)] - origin_weight for dr in rows}
        horizontal_weights = {dc: self.weights[(0, dc)] for dc in columns}
        for dr in rows:
            for dc in columns:
                expected = vertical_weights[dr] + horizontal_weights[dc]
                if self.weights[(dr, dc)] != expected:
                    return None

        vertical = StructuringElement((dr, 0, vertical_weights[dr]) for dr in rows)
        horizontal = StructuringElement((0, dc, horizontal_weights[dc]) for dc in columns)
        return vertical, horizontal

    def __repr__(self):
        return f"StructuringElement({self.members})"


def _validate_image(image):
    if not isinstance(image, list) or not image or not isinstance(image[0], list):
        raise MorphologyError("image must be a non-empty list of rows")
    width = len(image[0])
    if width == 0 or any(not isinstance(row, list) or len(row) != width for row in image):
        raise MorphologyError("image must be a non-empty rectangle")
    for row in image:
        for value in row:
            if isinstance(value, bool):
                continue
            if not _finite_number(value):
                raise MorphologyError("image values must be finite integers or floats")
    return len(image), width


def _brute(image, element, operation):
    height = len(image)
    width = len(image[0])
    items = [(dr, dc, element.weights[(dr, dc)]) for dr, dc in element.members]
    result = [[0 for _ in range(width)] for _ in range(height)]

    for row in range(height):
        for column in range(width):
            best = None
            for dr, dc, weight in items:
                if operation == ERODE:
                    source_row = row + dr
                    source_column = column + dc
                    if 0 <= source_row < height and 0 <= source_column < width:
                        candidate = image[source_row][source_column] - weight
                        if best is None or candidate < best:
                            best = candidate
                else:
                    source_row = row - dr
                    source_column = column - dc
                    if 0 <= source_row < height and 0 <= source_column < width:
                        candidate = image[source_row][source_column] + weight
                        if best is None or candidate > best:
                            best = candidate
            result[row][column] = best
    return result


def _morph_line(values, pairs, operation):
    length = len(values)
    offsets = [offset for offset, _ in pairs]
    flat_and_contiguous = all(weight == 0 for _, weight in pairs)
    if flat_and_contiguous:
        low_offset = min(offsets)
        high_offset = max(offsets)
        if offsets == list(range(low_offset, high_offset + 1)):
            queue = deque()
            added = -1
            result = [0] * length
            for index in range(length):
                if operation == ERODE:
                    source_low = index + low_offset
                    source_high = index + high_offset
                else:
                    source_low = index - high_offset
                    source_high = index - low_offset
                source_low = max(source_low, 0)
                source_high = min(source_high, length - 1)

                while added < source_high:
                    added += 1
                    value = values[added]
                    if operation == ERODE:
                        while queue and values[queue[-1]] >= value:
                            queue.pop()
                    else:
                        while queue and values[queue[-1]] <= value:
                            queue.pop()
                    queue.append(added)
                while queue[0] < source_low:
                    queue.popleft()
                result[index] = values[queue[0]]
            return result

    result = [0] * length
    for index in range(length):
        best = None
        for offset, weight in pairs:
            source = index + offset if operation == ERODE else index - offset
            if 0 <= source < length:
                candidate = values[source] - weight if operation == ERODE else values[source] + weight
                if best is None or (operation == ERODE and candidate < best) or (
                    operation == DILATE and candidate > best
                ):
                    best = candidate
        result[index] = best
    return result


def _apply_line(image, line_element, axis, operation):
    height = len(image)
    width = len(image[0])
    if axis == 0:
        pairs = []
        for dr, dc in line_element.members:
            if dc != 0:
                raise MorphologyError("a vertical line component cannot contain a column offset")
            pairs.append((dr, line_element.weights[(dr, dc)]))
        pairs.sort()
        result = [[0 for _ in range(width)] for _ in range(height)]
        for column in range(width):
            values = [image[row][column] for row in range(height)]
            filtered = _morph_line(values, pairs, operation)
            for row in range(height):
                result[row][column] = filtered[row]
        return result

    pairs = []
    for dr, dc in line_element.members:
        if dr != 0:
            raise MorphologyError("a horizontal line component cannot contain a row offset")
        pairs.append((dc, line_element.weights[(dr, dc)]))
    pairs.sort()
    return [_morph_line(row, pairs, operation) for row in image]


def _separable(image, element, operation):
    components = element.separable_components()
    if components is None:
        raise MorphologyError("this structuring element has no supported one-dimensional decomposition")
    vertical, horizontal = components
    result = _apply_line(image, vertical, 0, operation)
    return _apply_line(result, horizontal, 1, operation)


def _morph(image, element, operation, method):
    _validate_image(image)
    if not isinstance(element, StructuringElement):
        raise MorphologyError("element must be a StructuringElement")
    if method == "brute":
        return _brute(image, element, operation)
    if method == "separable":
        return _separable(image, element, operation)
    if method != "auto":
        raise MorphologyError("method must be 'auto', 'brute', or 'separable'")
    if element.separable_components() is not None:
        return _separable(image, element, operation)
    return _brute(image, element, operation)


def erode(image, element, method="auto"):
    return _morph(image, element, ERODE, method)


def dilate(image, element, method="auto"):
    return _morph(image, element, DILATE, method)


def opening(image, element, method="auto"):
    return dilate(erode(image, element, method), element, method)


def closing(image, element, method="auto"):
    return erode(dilate(image, element, method), element, method)
