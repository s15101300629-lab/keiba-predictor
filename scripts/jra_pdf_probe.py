from __future__ import annotations
import json, re, urllib.request
from pathlib import Path

URL = "https://jra.jp/datafile/seiseki/report/2025/2025-1kyoto1.pdf"
OUT = Path("jra-pdf-probe")
OUT.mkdir(exist_ok=True)
req = urllib.request.Request(URL, headers={"User-Agent":"Mozilla/5.0","Accept":"application/pdf"})
with urllib.request.urlopen(req, timeout=45) as res:
    raw, content_type = res.read(), res.headers.get("Content-Type", "")
if not raw.startswith(b"%PDF"):
    raise RuntimeError(f"Not a PDF: {content_type}, {len(raw)} bytes")
import pymupdf
(OUT/"source.pdf").write_bytes(raw)

def norm(s):
    return re.sub(r"\s+", "", re.sub(r"[\x00-\x1f]", "", s))

def center_y(item):
    return (item["bbox"][1]+item["bbox"][3])/2

doc = pymupdf.open(stream=raw, filetype="pdf")
doc[0].get_pixmap(matrix=pymupdf.Matrix(1.5,1.5), alpha=False).save(str(OUT/"page-1-preview.png"))
pages, blocks, lines, spans = [], [], [], []
font_xrefs = {}
for page in doc:
    for xref, ext, typ, basefont, resource_name, encoding, referencer in page.get_fonts(full=True):
        font_xrefs[basefont.split("+")[-1]] = xref

# The JRA PDF uses subset fonts for circled popularity digits. Decode each
# subset's /Differences array rather than treating control characters as ranks.
popularity_maps = {}
for font_name, xref in font_xrefs.items():
    font_obj = doc.xref_object(xref)
    enc_match = re.search(r"/Encoding\s+(\d+)\s+0\s+R", font_obj)
    if not enc_match:
        continue
    enc_obj = doc.xref_object(int(enc_match.group(1)))
    diff_match = re.search(r"/Differences\s*\[(.*?)\]", enc_obj, re.S)
    if not diff_match:
        continue
    current = None
    mapping = {}
    for token in re.findall(r"\d+|/[^\s\[\]]+", diff_match.group(1)):
        if token.isdigit():
            current = int(token)
        elif current is not None:
            glyph = token[1:]
            glyph_match = re.fullmatch(r"c(\d{3})", glyph)
            if glyph_match:
                popularity = int(glyph_match.group(1)) - 32
                if 1 <= popularity <= 18:
                    mapping[current] = popularity
            current += 1
    if mapping:
        popularity_maps[font_name] = mapping

for pn, page in enumerate(doc, 1):
    txt = page.get_text("text", sort=True)
    pages.append({"page":pn,"characters":len(txt),"text":txt})
    for b in page.get_text("dict")["blocks"]:
        if b.get("type") != 0: continue
        bt = "\n".join("".join(s["text"] for s in ln["spans"]) for ln in b["lines"]).strip()
        if bt: blocks.append({"page":pn,"bbox":list(b["bbox"]),"text":bt})
        for ln in b["lines"]:
            t = "".join(s["text"] for s in ln["spans"]).strip()
            if t: lines.append({"page":pn,"bbox":list(ln["bbox"]),"text":t})
            for sp in ln["spans"]:
                if sp["text"] :
                    spans.append({"page":pn,"bbox":list(sp["bbox"]),"text":sp["text"],"font":sp["font"]})
combined = "\n\n".join(f"===== PAGE {p['page']} =====\n{p['text']}" for p in pages)
(OUT/"sample-extracted.txt").write_text(combined,encoding="utf-8")
for name,data in [("layout-blocks.json",blocks),("layout-lines.json",lines),("layout-spans.json",spans)]:
    (OUT/name).write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding="utf-8")

headers=[]
for b in blocks:
    m=re.search(r"第\s*(\d+)\s*競走",b["text"])
    if m: headers.append({"page":b["page"],"x":b["bbox"][0],"y0":b["bbox"][1],"y1":b["bbox"][3],"no":int(m.group(1)),"text":b["text"]})
races, errors=[],[]
coat=r"黒鹿|青鹿|青毛|鹿毛|栃栗|鹿|栗|芦|白毛|白|青"
identity_re=re.compile(rf"(.+?)(牡|牝|せん|セン)?(\d)({coat})")
for h in headers:
    left=h["x"]<420
    col=lambda x: x["page"]==h["page"] and ((x["bbox"][0]<420)==left)
    cb=[x for x in blocks if col(x)]
    cl=[x for x in lines if col(x)]
    cs=[x for x in spans if col(x)]
    declared=decl_y=None
    for b in cb:
        m=re.search(r"（\s*(\d+)\s*頭\s*）",b["text"])
        if m:
            declared=int(m.group(1))
            count_line=next((x for x in cl if re.search(r"（\s*\d+\s*頭\s*）",x["text"])),None)
            decl_y=count_line["bbox"][1] if count_line else b["bbox"][3]
            break
    if declared is None:
        errors.append(f"R{h['no']}: headcount missing"); continue

    number_x=(18,42) if left else (429,452)
    identity_x=(45,145) if left else (455,555)
    number_lines=[x for x in cl if number_x[0]<=x["bbox"][0]<=number_x[1]
                  and x["bbox"][1]>=h["y1"]+25 and x["bbox"][1]<decl_y]
    horses=[]
    for nl in number_lines:
        raw_number_line=norm(nl["text"])
        nm=re.match(r"^([1-8])(\d{1,2})(.*)$",raw_number_line)
        if not nm: continue
        frame_no,horse_no=int(nm.group(1)),int(nm.group(2))
        ny=center_y(nl)
        fragments=[x for x in cl if identity_x[0]<=x["bbox"][0]<identity_x[1]
                   and abs(center_y(x)-ny)<=2.0]
        fragments.sort(key=lambda x:x["bbox"][0])
        identity=norm("".join(x["text"] for x in fragments))
        match=identity_re.search(identity)
        source="identity_column"
        if not match:
            fallback=nm.group(3)
            match=identity_re.search(fallback)
            if match:
                identity=fallback
                source="number_line_remainder"
        if not match:
            errors.append(f"R{h['no']} horse {horse_no}: identity parse failed {identity!r}")
            continue

        # Weight may be embedded after coat color or printed as small adjusted
        # weight digits on a separate baseline (e.g. 55 above 53).
        weight_fragments=[x for x in cl if 130<=(x["bbox"][0] if left else x["bbox"][0]-410.8)<150
                          and abs(center_y(x)-ny)<=4.0 and re.fullmatch(r"\d{2}",norm(x["text"]))]
        carried_weight=None
        if weight_fragments:
            carried_weight=int(norm(max(weight_fragments,key=center_y)["text"]))
        else:
            weight_match=re.search(rf"(?:{coat})(\d{{2}})$",identity)
            if weight_match: carried_weight=int(weight_match.group(1))

        # Extract official body weight, change, time, margin and odds from
        # their fixed columns. Empty values remain null when JRA omits them.
        body_x=(300,380) if left else (710,790)
        odds_x=(380,420) if left else (790,830)
        row_body=[x for x in cl if body_x[0]<=x["bbox"][0]<body_x[1] and abs(center_y(x)-ny)<=3.0]
        body_text=min(row_body,key=lambda x:abs(center_y(x)-ny))["text"] if row_body else ""
        body_match=re.search(r"(\d{3})\s*([＋+－−±])\s*(\d{1,2})",body_text)
        body_weight=int(body_match.group(1)) if body_match else None
        body_change=None
        if body_match:
            amount=int(body_match.group(3))
            body_change=amount if body_match.group(2) in ("＋","+") else (-amount if body_match.group(2) in ("－","−") else 0)
        time_match=re.search(r"(\d{1,2})[:：](\d{2})[．.](\d)",body_text)
        finish_time=(f"{int(time_match.group(1))}:{time_match.group(2)}.{time_match.group(3)}" if time_match else None)
        margin_text=None
        if time_match:
            suffix=norm(body_text[time_match.end():])
            margin_text=suffix or None

        odds_lines=[x for x in cl if odds_x[0]<=x["bbox"][0]<odds_x[1] and abs(center_y(x)-ny)<=3.0]
        odds_text=min(odds_lines,key=lambda x:abs(center_y(x)-ny))["text"] if odds_lines else ""
        odds_match=re.search(r"(\d+)[．.](\d)",odds_text)
        win_odds=float(odds_match.group(1)+"."+odds_match.group(2)) if odds_match else None
        popularity=None
        glyph_x_min=402 if left else 813
        popularity_spans=[x for x in cs if x["bbox"][0]>=glyph_x_min and abs(center_y(x)-ny)<=3.0
                          and len(x["text"])==1 and ord(x["text"])<32]
        if popularity_spans:
            glyph=popularity_spans[0]
            popularity=popularity_maps.get(glyph["font"],{}).get(ord(glyph["text"]))
        if popularity is None:
            errors.append(f"R{h['no']} horse {horse_no}: popularity glyph not decoded")

        jockey_x=(150,213) if left else (560,623)
        jockey_parts=[x for x in cl if jockey_x[0]<=x["bbox"][0]<jockey_x[1] and abs(center_y(x)-ny)<=2.5]
        jockey_parts.sort(key=lambda x:x["bbox"][0])
        jockey_raw="".join(x["text"] for x in jockey_parts).strip() or None

        horses.append({"frameNumber":frame_no,"horseNumber":horse_no,
                       "horseName":match.group(1),"sex":match.group(2),
                       "age":int(match.group(3)),"coatColor":match.group(4),
                       "carriedWeightKg":carried_weight,
                       "bodyWeightKg":body_weight,"bodyWeightChangeKg":body_change,
                       "finishTime":finish_time,"marginText":margin_text,
                       "winOdds":win_odds,"popularity":popularity,
                       "jockeyRaw":jockey_raw,
                       "identitySource":source,"sourcePage":h["page"],"sourceY":round(ny,2)})

    horses.sort(key=lambda x:x["sourceY"])
    for order,horse in enumerate(horses,1): horse["finishOrder"]=order
    nums=[x["horseNumber"] for x in horses]
    pops=[x["popularity"] for x in horses]
    if len(nums)!=declared: errors.append(f"R{h['no']}: rows {len(nums)} != declared {declared}")
    if len(set(nums))!=len(nums): errors.append(f"R{h['no']}: duplicate horse numbers {nums}")
    if any(n<1 or n>18 for n in nums): errors.append(f"R{h['no']}: horse number outside JRA range {nums}")
    if any(p is None or p<1 or p>declared for p in pops): errors.append(f"R{h['no']}: invalid popularity values {pops}")
    if len(set(pops))!=len(pops): errors.append(f"R{h['no']}: duplicate popularity ranks {pops}")
    if any(x["winOdds"] is None or x["winOdds"]<=0 for x in horses): errors.append(f"R{h['no']}: missing/invalid win odds")
    if any(x["carriedWeightKg"] is None for x in horses): errors.append(f"R{h['no']}: missing carried weight")

    nearby=[b["text"] for b in cb if h["y0"]-20 <= b["bbox"][1] <= h["y1"]+60]
    card_text="\n".join([h["text"],*nearby])
    year_match=re.search(r"(\d{4})年",card_text)
    day_match=re.search(r"(\d+)月(\d+)日",card_text)
    date=f"{int(year_match.group(1)):04d}-{int(day_match.group(1)):02d}-{int(day_match.group(2)):02d}" if year_match and day_match else None
    distance_match=re.search(r"(\d)[，,](\d{3})",h["text"])
    surface_match=re.search(r"（(芝|ダート|障害)[^）]*）",card_text)
    venue_match=re.search(r"\d{4}年\d+([^）]+)",card_text)
    weather_match=re.search(r"\d+月\d+日(晴|曇|雨|小雨|雪)",card_text)
    going_match=re.search(r"\d+月\d+日(?:晴|曇|雨|小雨|雪)?\s*(良|稍重|重|不良)",card_text)
    races.append({"raceNumber":h["no"],"date":date,
                  "venue":venue_match.group(1) if venue_match else None,
                  "distanceMeters":int(distance_match.group(1)+distance_match.group(2)) if distance_match else None,
                  "surface":surface_match.group(1) if surface_match else None,
                  "weather":weather_match.group(1) if weather_match else None,
                  "going":going_match.group(1) if going_match else None,
                  "declaredRunnerCount":declared,"parsedRunnerCount":len(horses),
                  "horses":horses})

if len(headers)!=12: errors.append(f"race headers: {len(headers)} != 12")
if len({r["raceNumber"] for r in races})!=len(races): errors.append("duplicate race numbers")
for race in races:
    for field in ("date","venue","distanceMeters","surface","going"):
        if race[field] is None: errors.append(f"R{race['raceNumber']}: missing race metadata {field}")

summary={"source":URL,"httpContentType":content_type,"pdfBytes":len(raw),"pageCount":len(doc),
         "pageCharacters":[p["characters"] for p in pages],"layoutBlockCount":len(blocks),
         "layoutLineCount":len(lines),"layoutSpanCount":len(spans),
         "popularityFontMaps":popularity_maps,
         "raceCount":len(headers),"parsedRaceCount":len(races),
         "parsedHorseRows":sum(len(r["horses"]) for r in races),"validationErrorCount":len(errors),
         "validationErrors":errors}
(OUT/"parsed-races.json").write_text(json.dumps(races,ensure_ascii=False,indent=2),encoding="utf-8")
(OUT/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
print(json.dumps(summary,ensure_ascii=False))
if len(doc)==0 or len(combined)<1000 or len(headers)!=12 or errors:
    raise RuntimeError(f"Structured validation failed: {len(errors)} errors")
