"""Decode JRA's custom "EdiF" symbol fonts embedded in the 成績表 PDFs.

JRA draws circled numbers (人気, 払戻の組番), fractions (着差 ½ ¾ ¼), marks such as
(地) / [外] / (混合) and some 外字 kanji with subsetted Type1 fonts that have no
ToUnicode map, so ordinary text extraction returns control characters for them.

Each glyph is identified by a hash of its outline (with subroutines inlined), which is
stable across PDFs, and looked up in ``glyph_map.json``.  The font program is chosen
by matching the PDF ``/Differences`` encoding against the glyph id reported by MuPDF,
because several distinct subset fonts can share the same base name on one page.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
from dataclasses import dataclass
from pathlib import Path

import pymupdf
from fontTools.cffLib import CFFFontSet
from fontTools.pens.recordingPen import RecordingPen

GLYPH_MAP_PATH = Path(__file__).with_name("glyph_map.json")

# Glyphs whose em box is at least this tall (pt) are course diagrams; they carry no text.
DIAGRAM_MIN_HEIGHT = 13.0
UNKNOWN = "〓"


def load_glyph_map() -> dict[str, str]:
    return json.loads(GLYPH_MAP_PATH.read_text(encoding="utf-8"))


@dataclass
class _SymbolFont:
    encoding: dict[int, str]
    charset: list[str]
    hashes: dict[str, str]


def _parse_differences(text: str) -> dict[int, str]:
    enc: dict[int, str] = {}
    code = 0
    for tok in re.findall(r"/[^\s/\]\[]+|\d+", text):
        if tok.startswith("/"):
            enc[code] = tok[1:]
            code += 1
        else:
            code = int(tok)
    return enc


def is_symbol_font(name: str) -> bool:
    return "EdiF" in name


class GlyphResolver:
    def __init__(self, doc: pymupdf.Document, glyph_map: dict[str, str] | None = None):
        self.glyph_map = glyph_map if glyph_map is not None else load_glyph_map()
        self.fonts: dict[str, list[_SymbolFont]] = {}
        self.unknown: dict[str, int] = {}
        seen: set[int] = set()
        for page in doc:
            for xref, _ext, ftype, basefont, *_ in page.get_fonts(full=True):
                if ftype != "Type1" or not is_symbol_font(basefont) or xref in seen:
                    continue
                seen.add(xref)
                font = self._load(doc, xref)
                if font:
                    self.fonts.setdefault(basefont.split("+")[-1], []).append(font)

    @staticmethod
    def _load(doc: pymupdf.Document, xref: int) -> _SymbolFont | None:
        try:
            _name, _ext, _type, buf = doc.extract_font(xref)
            cff = CFFFontSet()
            cff.decompile(io.BytesIO(buf), None)
        except Exception:  # noqa: BLE001 - malformed / non-CFF font: just skip it
            return None
        top = cff[cff.fontNames[0]]
        charset = list(top.charset)
        encoding: dict[int, str] = {}
        if not isinstance(top.Encoding, str):
            encoding = {i: n for i, n in enumerate(top.Encoding) if n != ".notdef"}
        kind, value = doc.xref_get_key(xref, "Encoding")
        if kind == "xref":
            _k, diff = doc.xref_get_key(int(value.split()[0]), "Differences")
            encoding.update(_parse_differences(diff))
        elif kind == "dict":
            m = re.search(r"/Differences\s*\[(.*?)\]", value, re.S)
            if m:
                encoding.update(_parse_differences(m.group(1)))
        hashes = {}
        for name in charset:
            pen = RecordingPen()
            top.CharStrings[name].draw(pen)
            hashes[name] = hashlib.sha1(repr(pen.value).encode()).hexdigest()[:12]
        return _SymbolFont(encoding, charset, hashes)

    def glyph_hash(self, font: str, code: int | None, gid: int) -> str | None:
        candidates = self.fonts.get(font, [])
        found = {
            f.hashes[f.charset[gid]]
            for f in candidates
            if gid < len(f.charset) and code is not None and f.encoding.get(code) == f.charset[gid]
        }
        if not found:
            found = {f.hashes[f.charset[gid]] for f in candidates if gid < len(f.charset)}
        return found.pop() if len(found) == 1 else None

    def text(self, font: str, code: int | None, gid: int, height: float) -> str:
        """Text for one symbol glyph ('' for decorations, UNKNOWN when unmapped)."""
        if height >= DIAGRAM_MIN_HEIGHT:
            return ""
        h = self.glyph_hash(font, code, gid)
        if h is not None and h in self.glyph_map:
            return self.glyph_map[h]
        key = h or f"ambiguous:{font}:{gid}"
        self.unknown[key] = self.unknown.get(key, 0) + 1
        return UNKNOWN
