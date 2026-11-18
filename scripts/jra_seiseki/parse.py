"""Parse a JRA 成績表 PDF (one race day at one venue) into structured data."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pymupdf

from .glyphs import UNKNOWN, GlyphResolver
from .pdftext import Char, Line, chars_text, group_lines, page_chars

VENUES = {
    "sapporo": ("01", "札幌"),
    "hakodate": ("02", "函館"),
    "fukushima": ("03", "福島"),
    "niigata": ("04", "新潟"),
    "tokyo": ("05", "東京"),
    "nakayama": ("06", "中山"),
    "chukyo": ("07", "中京"),
    "kyoto": ("08", "京都"),
    "hanshin": ("09", "阪神"),
    "kokura": ("10", "小倉"),
}

FILE_RE = re.compile(r"(?P<year>\d{4})-(?P<kai>\d{1,2})(?P<venue>[a-z]+)(?P<day>\d{1,2})$")

CIRCLED = {chr(0x2460 + i): i + 1 for i in range(20)}  # ①..⑳
FRACTIONS = {"½": ".5", "¾": ".75", "¼": ".25"}

# Normalise full-width digits / punctuation used throughout the PDFs.
_NUM_TRANS = str.maketrans(
    "０１２３４５６７８９，．：＋－（）　",
    "0123456789,.:+-() ",
)

# --- table column boundaries (x of the character centre, left-half coordinates) -------
COLUMNS = [
    ("waku", 15.0, 28.0),
    ("umaban", 28.0, 42.0),
    ("name", 42.0, 110.5),
    ("sexage", 110.5, 133.0),
    ("weight", 133.0, 150.0),
    ("jockey", 150.0, 193.3),
    ("owner", 193.3, 235.5),
    ("trainer", 235.5, 263.5),
    ("region", 263.5, 276.0),
    ("breeder", 276.0, 303.5),
    ("equipment", 303.5, 308.5),
    ("body", 308.5, 355.3),  # 馬体重 + 増減 + タイム (split with regexes)
    ("margin", 355.3, 383.0),
    ("odds", 383.0, 402.8),
    ("pop", 402.8, 420.0),
]

HORSE_MARKS = ("(地)", "[地]", "(外)", "[外]")
JOCKEY_MARKS = "▲△☆◇★"
STATUS_WORDS = {
    "出走取消": "取消",
    "競走除外": "除外",
    "競走中止": "中止",
    "失格": "失格",
}
BET_TYPES = ["単勝", "複勝", "枠連", "馬連", "馬単", "ワイド", "3連複", "3連単", "5重勝"]


def norm(s: str) -> str:
    return s.translate(_NUM_TRANS)


def to_int(s: str | None) -> int | None:
    if s is None:
        return None
    s = norm(s).replace(",", "").strip()
    return int(s) if re.fullmatch(r"[+-]?\d+", s) else None


def to_float(s: str | None) -> float | None:
    if s is None:
        return None
    s = norm(s).replace(",", "").strip()
    try:
        return float(s)
    except ValueError:
        return None


def time_to_seconds(s: str | None) -> float | None:
    if not s:
        return None
    m = re.fullmatch(r"(?:(\d+):)?(\d+\.\d)", norm(s))
    if not m:
        return None
    return round(int(m.group(1) or 0) * 60 + float(m.group(2)), 1)


def join_name(s: str) -> str:
    """Drop layout spaces from Japanese names but keep them between Latin words."""
    s = re.sub(r"\s+", " ", s).strip()
    return re.sub(r"(?<![A-Za-z.]) | (?![A-Za-z])", "", s)


def cell_text(chars: list[Char]) -> str:
    """Multi-line cell text: sub-lines top to bottom, each left to right."""
    if not chars:
        return ""
    lines = group_lines(chars, tol=1.2)
    return "".join(ln.text() for ln in lines)


@dataclass
class Segment:
    page: int
    half: str
    lines: list[Line]
    chars: list[Char]


def _is_race_header(line: Line) -> re.Match | None:
    if line.x0 > 30:
        return None
    return re.match(r"\s*(\d{5})\s*(\d{1,2})月\s*(\d{1,2})日\s*(.*)", line.text())


def split_races(doc: pymupdf.Document, resolver: GlyphResolver) -> list[list[Segment]]:
    """Cut the two-column page flow into per-race lists of segments."""
    races: list[list[Segment]] = []
    for pno, page in enumerate(doc):
        for half, chars in zip("LR", page_chars(page, resolver)):
            if not chars:
                continue
            lines = group_lines(chars)
            header_ys = [ln.y for ln in lines if _is_race_header(ln)]
            # Things printed just above a header (WIN5 banner, furigana, race marks)
            # belong to that race.
            cuts = [y - 14 for y in header_ys]
            bounds = [float("-inf")] + cuts + [float("inf")]
            for i in range(len(bounds) - 1):
                lo, hi = bounds[i], bounds[i + 1]
                seg_lines = [ln for ln in lines if lo <= ln.y < hi]
                if not seg_lines:
                    continue
                seg_chars = [c for ln in seg_lines for c in ln.chars]
                seg = Segment(pno, half, seg_lines, seg_chars)
                if i == 0:
                    if races and not (pno == 0 and half == "L"):
                        races[-1].append(seg)  # continuation of the previous race
                    # else: page title / column header -> ignore
                else:
                    races.append([seg])
    return races


# ----------------------------------------------------------------------------------
# race header
# ----------------------------------------------------------------------------------


def parse_header(seg: Segment, table_top: float) -> dict:
    lines = [ln for ln in seg.lines if ln.y < table_top]
    head = next(ln for ln in lines if _is_race_header(ln))
    m = _is_race_header(head)
    hy = head.y
    info: dict = {"serial": m.group(1), "month": int(m.group(2)), "day": int(m.group(3))}

    rest = norm(m.group(4))
    wm = re.match(r"\s*(\S+)\s+(.+?)\s*\(", rest)
    if wm:
        info["weather"] = wm.group(1)
        info["going"] = re.sub(r"\s+", " ", wm.group(2)).strip()

    near = [c for c in seg.chars if hy - 8 <= c.oy <= hy + 9]
    big = [c for c in near if c.size >= 9 and c.t.isdigit() and 195 <= c.cx <= 240]
    info["race_no"] = int("".join(c.t for c in sorted(big, key=lambda c: c.x0))) if big else None

    def is_mark(c: Char) -> bool:
        return any(b in c.t for b in "()[]")  # symbol-font labels use ASCII brackets

    name_chars = [
        c for c in near if 236 <= c.cx < 378 and c.size >= 5.5 and c.oy >= hy - 3 and not is_mark(c)
    ]
    info["race_name"] = re.sub(r"^競?走", "", re.sub(r"\s+", "", chars_text(name_chars)))

    marks = [c for c in seg.chars if 330 <= c.cx <= 385 and hy - 12 <= c.oy <= hy + 9 and is_mark(c)]
    mark_text = "".join(c.t for c in sorted(marks, key=lambda c: (round(c.oy), c.x0)))
    info["marks"] = re.findall(r"[(\[][^)\]]*[)\]]", mark_text)

    dist = [c for c in near if c.cx > 375 and (c.t.isdigit() or c.t in ",，")]
    info["distance"] = to_int("".join(c.t for c in sorted(dist, key=lambda c: c.x0)))

    text_lines = [norm(ln.text()) for ln in lines]
    for t in text_lines:
        if m2 := re.search(r"発走\s*(\d{1,2})時(\d{2})分", t):
            info["start_time"] = f"{int(m2.group(1)):02d}:{m2.group(2)}"
        if m2 := re.search(r"\(((?:芝|ダート|障害)[^)]*)\)\s*$", t):
            course = m2.group(1)
            info["course"] = course
            parts = course.split("・")
            info["surface"] = parts[0]
        if "負担重量" in t:
            cond, _, burden = t.partition("負担重量は")
            info["conditions"] = cond.strip(" ;；、")
            info["weight_rule"] = burden.strip(" 、,")
        if t.replace(" ", "").startswith("本賞"):
            info["prize"] = [to_int(x) for x in re.findall(r"([\d,]+)円", t)]
        if m2 := re.search(r"コースレコード\s*([\d:.]+)", t):
            info["course_record"] = m2.group(1)
    return info


# ----------------------------------------------------------------------------------
# horse table
# ----------------------------------------------------------------------------------


def find_anchors(seg: Segment, top: float, bottom: float) -> list[Char]:
    anchors = [
        c
        for c in seg.chars
        if 19 <= c.cx <= 27 and c.t.isdigit() and c.size >= 5.5 and top <= c.oy < bottom
    ]
    # A real row has the horse number right next to the frame number.
    nums = [c for c in seg.chars if 28 <= c.cx <= 42 and c.t.isdigit()]
    kana = [c for c in seg.chars if 42 <= c.cx <= 110.5 and re.match(r"[\u30a0-\u30ffA-Za-z]", c.t)]
    wgt = [c for c in seg.chars if 133 <= c.cx <= 150 and c.t.isdigit()]

    def near(cs: list[Char], a: Char, tol: float) -> bool:
        return any(abs(c.oy - a.oy) < tol for c in cs)

    anchors = [a for a in anchors if near(nums, a, 1.5) and near(kana, a, 3) and near(wgt, a, 4)]
    anchors.sort(key=lambda c: c.oy)
    out: list[Char] = []
    for c in anchors:
        if not out or c.oy - out[-1].oy > 4:
            out.append(c)
    return out


def split_columns(chars: list[Char]) -> dict[str, list[Char]]:
    cols: dict[str, list[Char]] = {name: [] for name, _a, _b in COLUMNS}
    for c in chars:
        for name, a, b in COLUMNS:
            if a <= c.cx < b:
                cols[name].append(c)
                break
    return cols


def parse_row(chars: list[Char]) -> dict:
    cols = split_columns(chars)
    row: dict = {}
    row_text = norm(chars_text(chars)).replace(" ", "")

    row["waku"] = to_int("".join(c.t for c in cols["waku"]))
    row["umaban"] = to_int("".join(c.t for c in cols["umaban"] if c.t.isdigit()))

    mark = "".join(c.t for c in cols["umaban"] + cols["name"] if c.t in HORSE_MARKS)
    name = join_name(cell_text([c for c in cols["name"] if c.t not in HORSE_MARKS]))
    row["horse"] = name
    row["horse_mark"] = mark or None

    sexage = cell_text(cols["sexage"]).replace(" ", "")
    m = re.match(r"([牡牝セ騸])?(\d+)(.*)", sexage)
    if m:
        row["sex"] = "セ" if m.group(1) == "騸" else m.group(1)
        row["age"] = int(m.group(2))
        row["color"] = m.group(3) or None
    else:
        row["sex"], row["age"], row["color"] = None, None, sexage or None

    wchars = cols["weight"]
    if wchars:
        biggest = max(c.size for c in wchars)
        main = [c for c in wchars if c.size >= biggest - 0.5]
        sub = [c for c in wchars if c.size < biggest - 0.5]
        row["weight"] = to_float(chars_text(main, 9).replace(" ", ""))
        row["weight_before_allowance"] = to_float(chars_text(sub, 9).replace(" ", "")) if sub else None
    else:
        row["weight"] = row["weight_before_allowance"] = None

    jockey = join_name(norm(cell_text(cols["jockey"])))
    jmark = ""
    while jockey and jockey[0] in JOCKEY_MARKS:
        jmark += jockey[0]
        jockey = jockey[1:]
    row["jockey"] = jockey
    row["jockey_mark"] = jmark or None
    row["owner"] = join_name(cell_text(cols["owner"]))
    row["trainer"] = join_name(cell_text(cols["trainer"]))
    row["region"] = join_name(cell_text(cols["region"]))
    row["breeder"] = join_name(cell_text(cols["breeder"]))
    row["equipment"] = cell_text(cols["equipment"]).replace(" ", "") or None

    status = None
    for word, st in STATUS_WORDS.items():
        if word in row_text:
            status = st
    row["status"] = status

    body = norm(chars_text(cols["body"])).replace(" ", "")
    if status in ("取消", "除外", "中止"):
        body = re.sub(r"\(?(出走取消|競走除外|競走中止)\)?", "", body)
    time_m = re.search(r"((?:\d:)?\d{2}\.\d|〃)$", body)
    time_s = None
    if time_m:
        time_s = time_m.group(1)
        body = body[: time_m.start()]
    bm = re.match(r"(\d{3})?(.*)$", body)
    row["body_weight"] = to_int(bm.group(1)) if bm else None
    diff = (bm.group(2) if bm else "").replace("±", "+")
    row["body_weight_diff"] = to_int(diff) if re.fullmatch(r"[+-]\d+", diff) else None
    row["time_raw"] = time_s

    margin = norm(chars_text(cols["margin"])).replace(" ", "")
    row["margin"] = None if status in ("取消", "除外", "中止") else (margin or None)

    row["odds"] = to_float(chars_text(cols["odds"]).replace(" ", ""))
    pop = "".join(c.t for c in cols["pop"])
    row["popularity"] = CIRCLED.get(pop[:1]) if pop else None
    return row


def parse_table(race: list[Segment]) -> tuple[list[dict], dict]:
    rows: list[dict] = []
    head_info: dict | None = None
    starters = None
    for i, seg in enumerate(race):
        # Excluded / scratched horses are printed below the "(N頭)" line, so the
        # table ends at the sales block.
        end = next(
            (ln.y for ln in seg.lines if re.search(r"売得金|単勝[:：]\s*[\d，,]", ln.text())),
            float("inf"),
        )
        for ln in seg.lines:
            if m := re.search(r"[（(](\d+)頭[）)]", ln.text()):
                starters = int(m.group(1))
        start = float("-inf")
        if i == 0:
            hdr = next(ln for ln in seg.lines if _is_race_header(ln))
            start = hdr.y + 12
            # Long weight rules wrap onto lines that start with digits; the table
            # only begins below the prize / record lines.
            for ln in seg.lines:
                if hdr.y < ln.y < end and ln.x0 < 60 and re.search(r"本\s*賞|付加賞|負担重量", ln.text()):
                    start = max(start, ln.y + 3)
        anchors = find_anchors(seg, start, end)
        if i == 0:
            top = anchors[0].oy - 6 if anchors else end
            head_info = parse_header(seg, top)
        for j, a in enumerate(anchors):
            lo = (anchors[j - 1].oy + a.oy) / 2 if j else a.oy - 7
            hi = (a.oy + anchors[j + 1].oy) / 2 if j + 1 < len(anchors) else min(a.oy + 7, end - 3)
            rows.append(parse_row([c for c in seg.chars if lo <= c.cy < hi]))
        if end != float("inf"):
            break
    info = head_info or {}
    info["starters"] = starters
    return rows, info


def finalize_rows(rows: list[dict]) -> None:
    prev_time = None
    pos = 0
    finishers = 0
    for r in rows:
        t = r.pop("time_raw")
        if t == "〃":
            t = prev_time
        if r["status"] in ("取消", "除外", "中止"):
            r["time"] = None
            r["finish"] = None
            continue
        r["time"] = t
        r["time_sec"] = time_to_seconds(t)
        prev_time = t
        finishers += 1
        if r["margin"] != "同着":
            pos = finishers
        r["finish"] = pos
    for r in rows:
        r.setdefault("time_sec", None)
    # Fill missing popularity (e.g. unknown glyph) from odds rank.
    with_odds = [r for r in rows if r["odds"] is not None]
    if any(r["popularity"] is None for r in with_odds):
        for rank, r in enumerate(sorted(with_odds, key=lambda r: r["odds"]), 1):
            if r["popularity"] is None:
                r["popularity"] = rank
                r["popularity_estimated"] = True


# ----------------------------------------------------------------------------------
# everything below the table
# ----------------------------------------------------------------------------------


def _squash_bets(t: str) -> str:
    t = re.sub(r"単\s*勝", "単勝", t)
    t = re.sub(r"複\s*勝", "複勝", t)
    t = re.sub(r"枠\s*連", "枠連", t)
    t = re.sub(r"馬\s*連", "馬連", t)
    t = re.sub(r"馬\s*単", "馬単", t)
    t = re.sub(r"ワ\s*イ\s*ド", "ワイド", t)
    t = re.sub(r"3\s*連\s*複", "3連複", t)
    t = re.sub(r"3\s*連\s*単", "3連単", t)
    return t


PAY_TOKEN_RE = re.compile(
    r"(?P<type>単\s*勝|複\s*勝|枠\s*連|馬\s*連|馬\s*単|ワ\s*イ\s*ド|3\s*連\s*複|3\s*連\s*単|5\s*重\s*勝)"
    r"|(?P<combo>[①-⑳〓]+)"
    r"|\((?P<waku>\d+(?:-\d+)*)\)"
    r"|(?P<amount>[\d,]+)円"
    r"|(?P<none>発売なし|特払|不成立)"
)


def parse_payouts(pairs: list[tuple[str, str, Line, int]]) -> list[dict]:
    """払戻金 block.  Entries are assigned to the bet label on their left in the same
    band, because stacked entries (e.g. three 複勝) put lines out of reading order."""
    labels, items = [], []
    for _t, _c, line, si in pairs:
        text, xs = "", []
        prev = None
        for c in line.chars:
            if prev is not None and c.x0 - prev.x1 > 0.6 * max(c.size, prev.size):
                text += " "
                xs.append(c.x0)
            text += c.t
            xs.extend([c.x0] * len(c.t))
            prev = c
        text = norm(text)
        for m in PAY_TOKEN_RE.finditer(text):
            x = xs[m.start()]
            if m.group("type"):
                labels.append((si, line.y, x, re.sub(r"\s+", "", m.group("type"))))
            else:
                items.append((si, line.y, x, m))
    if not labels:
        return []
    rows: list[list] = []  # [seg, y, [labels]]
    for lab in sorted(labels, key=lambda l: (l[0], l[1])):
        if rows and rows[-1][0] == lab[0] and abs(rows[-1][1] - lab[1]) < 3:
            rows[-1][2].append(lab)
        else:
            rows.append([lab[0], lab[1], [lab]])
    grouped: dict[tuple, list] = {}
    for si, y, x, m in items:
        cands = [r for r in rows if r[0] == si] or rows
        row = min(cands, key=lambda r: abs(r[1] - y))
        left = [l for l in row[2] if l[2] <= x + 1]
        lab = max(left, key=lambda l: l[2]) if left else min(row[2], key=lambda l: l[2])
        grouped.setdefault(lab, []).append((y, x, m))
    out = []
    for lab in sorted(grouped, key=lambda l: (l[0], l[1], l[2])):
        combo = None
        for _y, _x, m in sorted(grouped[lab], key=lambda t: (round(t[0]), t[1])):
            if m.group("combo"):
                combo = [CIRCLED.get(ch) for ch in m.group("combo")]
            elif m.group("waku"):
                combo = [int(v) for v in m.group("waku").split("-")]
            elif m.group("amount"):
                out.append({"type": lab[3], "combo": combo, "payout": to_int(m.group("amount"))})
                combo = None
            elif m.group("none"):
                out.append({"type": lab[3], "combo": None, "payout": None, "note": m.group("none")})
    order = {t: i for i, t in enumerate(BET_TYPES)}
    out.sort(key=lambda p: order.get(p["type"], 99))  # stable within a type
    return out


def parse_lower(pairs: list[tuple[str, str, Line, int]]) -> dict:
    lines = [p[0] for p in pairs]
    clines = [p[1] for p in pairs]
    out: dict = {"sales": {}, "payouts": [], "laps": [], "passing_times": [], "corners": [], "notes": []}

    def idx(pred) -> int | None:
        return next((i for i, t in enumerate(lines) if pred(t)), None)

    vote_start = idx(lambda t: "単勝票数" in t)
    # The payout block starts after the 売得金 lines ("単勝: 36,741,400円" etc.).
    pay_start = idx(lambda t: "円" in t and ":" not in t and "票数" not in t)
    if pay_start is not None and vote_start is not None and pay_start > vote_start:
        pay_start = None
    lap_start = idx(lambda t: "ハロンタイム" in t)
    pass_start = idx(lambda t: "通過タイム" in t)
    corner_start = idx(lambda t: t.startswith("コーナー") or "通過コーナー順" in t)
    win_start = idx(lambda t: t.startswith("勝馬の"))

    # 売得金
    for t in lines[: pay_start or 0]:
        for k, v in re.findall(r"(単勝|複勝|枠連|馬連|馬単|ワイド|3連複|3連単|5重勝|計)\s*:\s*([\d,]+)円", t):
            out["sales"][k] = to_int(v)

    # 払戻金
    if pay_start is not None:
        out["payouts"] = parse_payouts(pairs[pay_start : vote_start or lap_start])

    # 票数 -> popularity of each winning combination
    if vote_start is not None:
        pops: dict[str, list[tuple[int | None, int | None]]] = {}
        for t in lines[vote_start : lap_start]:
            t = _squash_bets(t)
            m = re.search(r"(単勝|複勝|枠連|馬連|馬単|ワイド|3連複|3連単|5重勝)票数\s*計\s*([\d,]+)", t)
            if not m:
                continue
            btype = m.group(1)
            out["sales"].setdefault("votes", {})[btype] = to_int(m.group(2))
            for v, p in re.findall(r"([\d,]+)\s*\((\d+)番人気\)", t):
                pops.setdefault(btype, []).append((to_int(v), int(p)))
        for btype, items in pops.items():
            targets = [p for p in out["payouts"] if p["type"] == btype and p.get("payout") is not None]
            for p, (votes, pop) in zip(targets, items):
                p["votes"] = votes
                p["popularity"] = pop

    def times(seg: list[str]) -> list[float | None]:
        vals = []
        for t in seg:
            vals += re.findall(r"(?:\d+:)?\d+\.\d", t.replace(" ", ""))
        return [time_to_seconds(v) for v in vals]

    if lap_start is not None:
        end = pass_start if pass_start is not None else corner_start
        out["laps"] = times([lines[lap_start].split("ハロンタイム", 1)[1]] + lines[lap_start + 1 : end])
    if pass_start is not None:
        end = corner_start if corner_start is not None else win_start
        out["passing_times"] = times(
            [lines[pass_start].split("通過タイム", 1)[1]] + lines[pass_start + 1 : end]
        )

    for t in lines:
        if m := re.search(r"上り\s*4F\s*([\d.:]+)\s*[―-]\s*3F\s*([\d.:]+)", t):
            out["last_4f"] = time_to_seconds(m.group(1))
            out["last_3f"] = time_to_seconds(m.group(2))
        elif m := re.search(r"上り\s*3F\s*([\d.:]+)", t):
            out["last_3f"] = time_to_seconds(m.group(1))

    if corner_start is not None:
        end = win_start if win_start is not None else len(lines)
        text = " ".join(clines[corner_start:end])
        text = re.sub(r"「[^」]*」", " ", text)
        text = re.sub(r"上り.*?3F\s*[\d.:]+", " ", text)
        text = text.replace("コーナー", " ").replace("通過順位", " ")
        for tok in text.split():
            if re.fullmatch(r"[1-4①-④]", tok):
                out["corners"].append({"corner": tok, "order": ""})
            elif out["corners"] and re.search(r"\d", tok):
                out["corners"][-1]["order"] += tok
    notes_started = False
    for t in lines[(win_start or len(lines)) :]:
        if t.startswith(("〔", "※", "(", "（")):
            notes_started = True
            out["notes"].append(t)
        elif notes_started and out["notes"]:
            out["notes"][-1] += t
    return out


def corner_positions(seq: str) -> dict[int, int]:
    """'8,9(7,14,16)(1,3,13)12-5=4' -> {umaban: position at that corner}."""
    positions: dict[int, int] = {}
    count = 0
    for m in re.finditer(r"\(([^)]*)\)|(\d+)", seq):
        group = [int(x) for x in re.findall(r"\d+", m.group(1))] if m.group(1) is not None else [int(m.group(2))]
        for u in group:
            positions[u] = count + 1
        count += len(group)
    return positions


# ----------------------------------------------------------------------------------


def corner_text(line: Line) -> str:
    """Corner-order lines: separate the corner label from its sequence by pen advance.

    Bounding boxes are full em boxes even for half-width glyphs, so the generic gap
    rule cannot see the space between '3' and '(5,14)...'.
    """
    out = []
    prev = None
    for c in line.chars:
        if prev is not None and c.x0 - prev.x0 > 1.4 * c.size:
            out.append(" ")
        out.append(c.t)
        prev = c
    return norm("".join(out)).strip()


def lower_lines(race: list[Segment]) -> list[tuple[str, str, Line, int]]:
    out: list[tuple[str, str, Line, int]] = []
    started = False
    for si, seg in enumerate(race):
        for ln in seg.lines:
            t = ln.text()
            if not started:
                if re.search(r"[（(]\d+頭[）)]", t):
                    started = True
                continue
            out.append((norm(t).strip(), corner_text(ln), ln, si))
    return out


def parse_pdf(path: str | Path, source_url: str | None = None) -> dict:
    path = Path(path)
    doc = pymupdf.open(path)
    resolver = GlyphResolver(doc)
    fm = FILE_RE.match(path.stem)
    meta: dict = {"file": path.name, "source_url": source_url}
    if fm:
        venue_code, venue_name = VENUES.get(fm.group("venue"), (None, fm.group("venue")))
        meta.update(
            year=int(fm.group("year")),
            kai=int(fm.group("kai")),
            venue=fm.group("venue"),
            venue_code=venue_code,
            venue_name=venue_name,
            day=int(fm.group("day")),
        )
    races = []
    for race in split_races(doc, resolver):
        rows, info = parse_table(race)
        if "障害" in (info.get("race_name") or "") + (info.get("conditions") or ""):
            info["surface"] = "障害"  # course stays e.g. "芝・ダート"
        finalize_rows(rows)
        lower = parse_lower(lower_lines(race))
        corners = [corner_positions(c["order"]) for c in lower["corners"]]
        for r in rows:
            r["corner_positions"] = [c.get(r["umaban"]) for c in corners] or None
        if fm and info.get("race_no"):
            info["race_id"] = (
                f"{meta['year']}{meta['venue_code']}{meta['kai']:02d}{meta['day']:02d}{info['race_no']:02d}"
            )
            info["date"] = f"{meta['year']}-{info['month']:02d}-{info['day']:02d}"
        races.append({**info, "runners": len(rows), "results": rows, **lower})
    if races and races[0].get("date"):
        meta["date"] = races[0]["date"]
    meta["races"] = races
    meta["unknown_glyphs"] = resolver.unknown
    meta["warnings"] = validate(meta)
    return meta


def validate(day: dict) -> list[str]:
    warns = []
    for race in day["races"]:
        rid = race.get("race_id") or race.get("serial")
        if not race.get("race_no"):
            warns.append(f"{rid}: race number not found")
        if not race["results"]:
            warns.append(f"{rid}: no result rows")
        nums = [r["umaban"] for r in race["results"]]
        if sorted(n for n in nums if n) != list(range(1, len(nums) + 1)):
            warns.append(f"{rid}: horse numbers not 1..N: {nums}")
        for r in race["results"]:
            if r["status"] is None and r["time_sec"] is None:
                warns.append(f"{rid}: no time for #{r['umaban']} {r['horse']}")
            if UNKNOWN in (r["horse"] + r["jockey"] + r["trainer"]):
                warns.append(f"{rid}: unknown glyph in #{r['umaban']} {r['horse']}/{r['jockey']}/{r['trainer']}")
        if not any(p["type"] == "単勝" for p in race["payouts"]):
            warns.append(f"{rid}: no 単勝 payout")
    return warns
