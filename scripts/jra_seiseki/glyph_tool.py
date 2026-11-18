"""Maintenance tool: render glyphs missing from glyph_map.json so they can be labelled.

    python -m jra_seiseki.glyph_tool sheet out.png some.pdf other.pdf
        -> contact sheet; each cell shows the glyph hash, the glyph and its context
    python -m jra_seiseki.glyph_tool add <hash> <text> [<hash> <text> ...]
        -> add labels ('' for decorations) to glyph_map.json

After adding labels run ``python -m jra_seiseki update --force`` (or bump PARSER_VERSION).
Requires Pillow for ``sheet``.
"""

from __future__ import annotations

import io
import json
import sys

import pymupdf

from .glyphs import DIAGRAM_MIN_HEIGHT, GLYPH_MAP_PATH, GlyphResolver, is_symbol_font, load_glyph_map


def unknown_glyphs(paths: list[str]) -> dict[str, tuple[str, int, tuple]]:
    known = load_glyph_map()
    found: dict[str, tuple[str, int, tuple]] = {}
    for path in paths:
        doc = pymupdf.open(path)
        res = GlyphResolver(doc, known)
        for pno, page in enumerate(doc):
            codes = {}
            for b in page.get_text("rawdict")["blocks"]:
                for ln in b.get("lines", []):
                    for sp in ln["spans"]:
                        if is_symbol_font(sp["font"]):
                            for ch in sp["chars"]:
                                ox, oy = ch["origin"]
                                codes[(sp["font"], round(ox, 1), round(oy, 1))] = ord(ch["c"])
            for sp in page.get_texttrace():
                if not is_symbol_font(sp["font"]):
                    continue
                for _u, gid, org, bbox in sp["chars"]:
                    if bbox[3] - bbox[1] >= DIAGRAM_MIN_HEIGHT:
                        continue
                    code = codes.get((sp["font"], round(org[0], 1), round(org[1], 1)))
                    h = res.glyph_hash(sp["font"], code, gid)
                    if h and h not in known and h not in found:
                        found[h] = (path, pno, tuple(bbox))
    return found


def sheet(out: str, paths: list[str]) -> None:
    from PIL import Image, ImageDraw

    found = unknown_glyphs(paths)
    if not found:
        print("no unknown glyphs")
        return
    w, h = 420, 70
    img = Image.new("RGB", (w, h * len(found)), "white")
    draw = ImageDraw.Draw(img)
    docs: dict[str, pymupdf.Document] = {}
    for i, (gh, (path, pno, bbox)) in enumerate(found.items()):
        doc = docs.setdefault(path, pymupdf.open(path))
        r = pymupdf.Rect(bbox)
        glyph = doc[pno].get_pixmap(dpi=500, clip=r)
        ctx = doc[pno].get_pixmap(dpi=300, clip=pymupdf.Rect(r.x0 - 40, r.y0 - 1, r.x1 + 25, r.y1 + 1))
        for pix, x, size in ((glyph, 120, (60, 60)), (ctx, 190, (225, 60))):
            im = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
            im.thumbnail(size)
            img.paste(im, (x, i * h + 5))
        draw.text((5, i * h + 25), gh, fill="blue")
        print(gh, path, pno)
    img.save(out)
    print(f"wrote {out}")


def add(pairs: list[str]) -> None:
    m = load_glyph_map()
    for gh, text in zip(pairs[::2], pairs[1::2]):
        m[gh] = text
    GLYPH_MAP_PATH.write_text(json.dumps(dict(sorted(m.items())), ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    cmd, *rest = sys.argv[1:]
    if cmd == "sheet":
        sheet(rest[0], rest[1:])
    elif cmd == "add":
        add(rest)
    else:
        raise SystemExit(__doc__)
