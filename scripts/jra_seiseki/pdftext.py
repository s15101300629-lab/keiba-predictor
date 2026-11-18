"""Low level text extraction: characters with positions, grouped into lines."""

from __future__ import annotations

from dataclasses import dataclass, field

import pymupdf

from .glyphs import GlyphResolver, is_symbol_font

# The A4-landscape pages are printed as two columns ("halves").  The right half is the
# left one shifted by this many points, so table columns can be addressed with one
# set of x coordinates.
RIGHT_HALF_OFFSET = 411.0


@dataclass
class Char:
    x0: float
    y0: float
    x1: float
    y1: float
    oy: float  # baseline
    t: str
    size: float
    span: int  # global span sequence number (keeps drawing order inside a span)
    idx: int

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2


@dataclass
class Line:
    y: float
    chars: list[Char] = field(default_factory=list)

    @property
    def x0(self) -> float:
        return self.chars[0].x0

    def text(self, space_factor: float = 0.6) -> str:
        return chars_text(self.chars, space_factor)


def order_chars(chars: list[Char]) -> list[Char]:
    """Sort by x, but keep the drawing order of characters inside one span.

    Full-width brackets have overlapping boxes ("）（"), so pure x sorting can swap them.
    """
    blocks: dict[int, list[Char]] = {}
    for c in chars:
        blocks.setdefault(c.span, []).append(c)
    ordered = sorted(blocks.values(), key=lambda b: min(c.x0 for c in b))
    return [c for b in ordered for c in sorted(b, key=lambda c: c.idx)]


def chars_text(chars: list[Char], space_factor: float = 0.6) -> str:
    out = []
    prev = None
    for c in order_chars(chars):
        if prev is not None and c.x0 - prev.x1 > space_factor * max(c.size, prev.size):
            out.append(" ")
        out.append(c.t)
        prev = c
    return "".join(out)


def group_lines(chars: list[Char], tol: float = 1.0) -> list[Line]:
    lines: list[Line] = []
    for c in sorted(chars, key=lambda c: (c.oy, c.x0)):
        if lines and abs(lines[-1].y - c.oy) <= tol:
            lines[-1].chars.append(c)
        else:
            lines.append(Line(c.oy, [c]))
    for ln in lines:
        ln.chars = order_chars(ln.chars)
    return lines


def page_chars(page: pymupdf.Page, resolver: GlyphResolver) -> tuple[list[Char], list[Char]]:
    """Return (left_half, right_half) characters; right half x is shifted to align."""
    codes: dict[tuple[str, float, float], int] = {}
    for block in page.get_text("rawdict")["blocks"]:
        for line in block.get("lines", []):
            for span in line["spans"]:
                if is_symbol_font(span["font"]):
                    for ch in span["chars"]:
                        ox, oy = ch["origin"]
                        codes[(span["font"], round(ox, 1), round(oy, 1))] = ord(ch["c"])

    mid = page.rect.width / 2
    left: list[Char] = []
    right: list[Char] = []
    idx = 0
    for sidx, span in enumerate(page.get_texttrace()):
        font = span["font"]
        symbol = is_symbol_font(font)
        size = span["size"]
        for unicode, gid, origin, bbox in span["chars"]:
            x0, y0, x1, y1 = bbox
            if symbol:
                code = codes.get((font, round(origin[0], 1), round(origin[1], 1)))
                t = resolver.text(font, code, gid, y1 - y0)
            else:
                t = chr(unicode) if unicode > 0 else ""
            idx += 1
            if not t or t.isspace():
                continue
            target, dx = (left, 0.0) if (x0 + x1) / 2 < mid else (right, RIGHT_HALF_OFFSET)
            target.append(Char(x0 - dx, y0, x1 - dx, y1, origin[1], t, size, sidx, idx))
    return left, right
