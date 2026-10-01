"""Floyd-Steinberg error diffusion, streaming row by row (serpentine)."""

from .palettes import clamp8


class StreamingDitherer:
    """Row-by-row Floyd-Steinberg diffusion.

    Keeps only two width-sized error rows, so memory is O(width),
    independent of image height. Error aimed outside the canvas is dropped
    and accounted for in `dropped`; it never wraps to the other edge and
    never reaches another image.
    """

    def __init__(self, width, palette, serpentine=True):
        if width < 1:
            raise ValueError("width must be >= 1")
        self.width = width
        self.palette = palette
        self.serpentine = serpentine
        self._er = [0.0] * width
        self._eg = [0.0] * width
        self._eb = [0.0] * width
        self._y = 0
        self.dropped = [0.0, 0.0, 0.0]
        self.clamp_delta = [0.0, 0.0, 0.0]
        self.sum_in = [0.0, 0.0, 0.0]
        self.sum_out = [0.0, 0.0, 0.0]

    def process_row(self, row):
        """Consume one input row (RGB tuples), return palette-index row."""
        if len(row) != self.width:
            raise ValueError(
                f"expected {self.width} pixels, got {len(row)}")
        w = self.width
        colors = self.palette.colors
        nearest = self.palette.nearest_index
        er, eg, eb = self._er, self._eg, self._eb
        nr, ng, nb = [0.0] * w, [0.0] * w, [0.0] * w
        l2r = (not self.serpentine) or (self._y & 1) == 0
        dx = 1 if l2r else -1
        out = [0] * w
        si, so = self.sum_in, self.sum_out
        dropped, clampd = self.dropped, self.clamp_delta
        for x in range(w) if l2r else range(w - 1, -1, -1):
            pr, pg, pb = row[x]
            si[0] += pr
            si[1] += pg
            si[2] += pb
            ar, ag, ab = pr + er[x], pg + eg[x], pb + eb[x]
            cr, cg, cb = clamp8(ar), clamp8(ag), clamp8(ab)
            clampd[0] += ar - cr
            clampd[1] += ag - cg
            clampd[2] += ab - cb
            idx = nearest(cr, cg, cb)
            out[x] = idx
            qr, qg, qb = colors[idx]
            so[0] += qr
            so[1] += qg
            so[2] += qb
            fr, fg, fb = cr - qr, cg - qg, cb - qb
            xs_ = x + dx
            if 0 <= xs_ < w:
                er[xs_] += fr * 0.4375
                eg[xs_] += fg * 0.4375
                eb[xs_] += fb * 0.4375
            else:
                dropped[0] += fr * 0.4375
                dropped[1] += fg * 0.4375
                dropped[2] += fb * 0.4375
            xl = x - dx
            if 0 <= xl < w:
                nr[xl] += fr * 0.1875
                ng[xl] += fg * 0.1875
                nb[xl] += fb * 0.1875
            else:
                dropped[0] += fr * 0.1875
                dropped[1] += fg * 0.1875
                dropped[2] += fb * 0.1875
            nr[x] += fr * 0.3125
            ng[x] += fg * 0.3125
            nb[x] += fb * 0.3125
            xr = x + dx
            if 0 <= xr < w:
                nr[xr] += fr * 0.0625
                ng[xr] += fg * 0.0625
                nb[xr] += fb * 0.0625
            else:
                dropped[0] += fr * 0.0625
                dropped[1] += fg * 0.0625
                dropped[2] += fb * 0.0625
        self._er, self._eg, self._eb = nr, ng, nb
        self._y += 1
        return out

    def residual(self):
        """Unconsumed error in the row buffer (below the last processed row)."""
        return [sum(self._er), sum(self._eg), sum(self._eb)]

    def conservation_residual(self):
        """Per-channel accounting gap; must be ~0 for error conservation."""
        res = self.residual()
        return [
            (si - so) - (d + c + rv)
            for si, so, d, c, rv in zip(
                self.sum_in, self.sum_out, self.dropped,
                self.clamp_delta, res)
        ]


def diffuse(rows, width, palette, serpentine=True):
    """Whole-image convenience wrapper built on the streaming core."""
    ditherer = StreamingDitherer(width, palette, serpentine=serpentine)
    return [ditherer.process_row(row) for row in rows], ditherer
