"""JRA のオッズのページ（accessO）から、馬連・ワイド・馬単・3連複・3連単の全部の組み合わせのオッズを読む（2026-11-24、依頼者の案 8）.

単勝・複勝のページ（pw151ou…）に、ほかの券種のページの cname（pw154＝馬連、pw155＝ワイド、pw156＝馬単、pw157＝3連複、pw158＝3連単）がある。
check_jra_day.py が確定オッズを毎日 data/jra-odds/日付.json に保存し、期待値の順（evPlan）と実際の組み合わせのオッズを使う確率（案 2）を後から確かめる
キー：馬連・ワイド・3連複は小さい順「1-2」「1-2-3」、馬単・3連単は着順「2-1」「3-1-2」。ワイドは [下限, 上限]
"""

from __future__ import annotations

import re

TYPES = {"pw154": "馬連", "pw155": "ワイド", "pw156": "馬単", "pw157": "3連複", "pw158": "3連単"}


def _num(s: str) -> float | None:
    s = re.sub(r"<[^>]+>", "", s).replace(",", "").strip()
    try:
        return float(s)
    except ValueError:
        return None


def links(page_151: str, race_key: str) -> dict[str, str]:
    """単勝・複勝のページから、同じレースのほかの券種のページの cname. race_key は cname の中のレースを表す部分（pw151ou の後ろ、Z の前）."""
    out = {}
    for c in re.findall(r"'(pw15[4-8]ou[^']+)'", page_151):
        if race_key in c:
            out[TYPES[c[:5]]] = c
    return out


def parse(kind: str, page: str) -> dict:
    out: dict = {}
    if kind in ("馬連", "ワイド", "馬単", "3連複"):
        cls = {"馬連": "umaren", "ワイド": "wide", "馬単": "umatan", "3連複": "fuku3"}[kind]
        for cap, body in re.findall(r'<table class="basic narrow-xy ' + cls + r'">\s*<caption>(.*?)</caption>(.*?)</table>', page, re.S):
            head = [int(x) for x in re.findall(r"\d+", re.sub(r"<[^>]+>", " ", cap))]
            for th, td in re.findall(r'<th scope="row">(\d+)</th>\s*(<td.*?</td>)', body, re.S):
                nums = head + [int(th)]
                if kind == "ワイド":
                    mn, mx = re.search(r'class="min">([\d.,]+)<', td), re.search(r'class="max">([\d.,]+)<', td)
                    if mn:
                        out["-".join(map(str, sorted(nums)))] = [_num(mn[1]), _num(mx[1]) if mx else None]
                    continue
                v = _num(td)
                if v is None:
                    continue
                key = nums if kind == "馬単" else sorted(nums)
                out["-".join(map(str, key))] = v
    elif kind == "3連単":
        # 表の前に「1着 n」「2着 m」の札、表の行が3着
        for m in re.finditer(r'<table class="basic narrow-xy tan3">(.*?)</table>', page, re.S):
            before = page[max(0, m.start() - 700):m.start()]
            f1 = re.findall(r'<span>1着</span></div><div class="num">(\d+)</div>', before)
            f2 = re.findall(r'<span>2着</span></div><div class="num">(\d+)</div>', before)
            if not f1 or not f2:
                continue
            a, b = int(f1[-1]), int(f2[-1])
            for th, td in re.findall(r'<th scope="row">(\d+)</th>\s*(<td.*?</td>)', m.group(1), re.S):
                v = _num(td)
                if v is not None:
                    out[f"{a}-{b}-{th}"] = v
    return out
