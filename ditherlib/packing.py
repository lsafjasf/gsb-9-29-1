"""Bit packing of palette indices at 1/2/4/8 bits per pixel, MSB first."""

VALID_DEPTHS = (1, 2, 4, 8)


def _check_bits(bits):
    if bits not in VALID_DEPTHS:
        raise ValueError(f"bits must be one of {VALID_DEPTHS}, got {bits!r}")


def packed_row_size(width, bits):
    _check_bits(bits)
    if width < 0:
        raise ValueError("width must be >= 0")
    return (width * bits + 7) // 8


def pack_indices(indices, bits):
    """Pack index values into bytes, MSB first, zero-padded at the tail."""
    _check_bits(bits)
    mask = (1 << bits) - 1
    out = bytearray()
    acc = 0
    nbits = 0
    for value in indices:
        if not 0 <= value <= mask:
            raise ValueError(f"index {value!r} does not fit in {bits} bits")
        acc = (acc << bits) | value
        nbits += bits
        while nbits >= 8:
            nbits -= 8
            out.append((acc >> nbits) & 0xFF)
        acc &= (1 << nbits) - 1
    if nbits:
        out.append((acc << (8 - nbits)) & 0xFF)
    return bytes(out)


def unpack_indices(data, bits, count):
    """Inverse of pack_indices; reads exactly `count` indices."""
    _check_bits(bits)
    if count < 0:
        raise ValueError("count must be >= 0")
    mask = (1 << bits) - 1
    out = []
    acc = 0
    nbits = 0
    for byte in data:
        acc = (acc << 8) | byte
        nbits += 8
        while nbits >= bits and len(out) < count:
            nbits -= bits
            out.append((acc >> nbits) & mask)
        acc &= (1 << nbits) - 1
        if len(out) == count:
            break
    if len(out) != count:
        raise ValueError(f"not enough data: wanted {count} indices, got {len(out)}")
    return out


def pack_rows(index_rows, bits):
    """Pack a 2D image row by row (each row independently padded)."""
    return b"".join(pack_indices(row, bits) for row in index_rows)


def unpack_rows(data, bits, width, height):
    """Inverse of pack_rows for a width x height image."""
    row_bytes = packed_row_size(width, bits)
    if len(data) != row_bytes * height:
        raise ValueError(
            f"expected {row_bytes * height} bytes, got {len(data)}")
    return [
        unpack_indices(data[y * row_bytes:(y + 1) * row_bytes], bits, width)
        for y in range(height)
    ]
