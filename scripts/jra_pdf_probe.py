from __future__ import annotations
import hashlib, json, os, re, urllib.request
from pathlib import Path

URL = os.environ.get("JRA_PDF_URL", "https://jra.jp/datafile/seiseki/report/2025/2025-1kyoto1.pdf")
URL_YEAR_MATCH = re.search(r"/report/(\d{4})/", URL)
URL_YEAR = int(URL_YEAR_MATCH.group(1)) if URL_YEAR_MATCH else None
OUT = Path(os.environ.get("JRA_OUTPUT_DIR", "jra-pdf-probe"))
DIAGNOSTICS = os.environ.get("JRA_DIAGNOSTICS", "1") == "1"
EXPECTED_RACES = int(os.environ.get("JRA_EXPECTED_RACES", "12"))
OUT.mkdir(exist_ok=True)
req = urllib.request.Request(URL, headers={"User-Agent":"Mozilla/5.0","Accept":"application/pdf"})
with urllib.request.urlopen(req, timeout=45) as res:
    raw, content_type = res.read(), res.headers.get("Content-Type", "")
if not raw.startswith(b"%PDF"):
    raise RuntimeError(f"Not a PDF: {content_type}, {len(raw)} bytes")
import pymupdf
if DIAGNOSTICS:
    (OUT/"source.pdf").write_bytes(raw)

def norm(s):
    return re.sub(r"\s+", "", re.sub(r"[\x00-\x1f]", "", s))

def center_y(item):
    return (item["bbox"][1]+item["bbox"][3])/2


def time_to_seconds(value):
    normalized=value.replace("：",":").replace("．",".")
    if ":" in normalized:
        minutes,seconds=normalized.split(":",1)
        return round(int(minutes)*60+float(seconds),1)
    return round(float(normalized),1)

doc = pymupdf.open(stream=raw, filetype="pdf")
if DIAGNOSTICS:
    doc[0].get_pixmap(matrix=pymupdf.Matrix(1.5,1.5), alpha=False).save(str(OUT/"page-1-preview.png"))
pages, blocks, lines, spans, chars = [], [], [], [], []
popularity_maps = {}
for pn, page in enumerate(doc, 1):
    for xref, ext, typ, basefont, resource_name, encoding, referencer in page.get_fonts(full=True):
        font_name=basefont.split("+")[-1]
        font_obj=doc.xref_object(xref)
        enc_match=re.search(r"/Encoding\s+(\d+)\s+0\s+R",font_obj)
        if not enc_match: continue
        enc_obj=doc.xref_object(int(enc_match.group(1)))
        diff_match=re.search(r"/Differences\s*\[(.*?)\]",enc_obj,re.S)
        if not diff_match: continue
        current=None
        mapping={}
        for token in re.findall(r"\d+|/[^\s\[\]]+",diff_match.group(1)):
            if token.isdigit():
                current=int(token)
            elif current is not None:
                gm=re.fullmatch(r"c(\d{3})",token[1:])
                if gm:
                    rank=int(gm.group(1))-32
                    if 1<=rank<=18: mapping[current]=rank
                current+=1
        if mapping: popularity_maps[f"{pn}:{font_name}"]=mapping

    txt=page.get_text("text",sort=True)
    pages.append({"page":pn,"characters":len(txt),"text":txt})
    for b in page.get_text("dict")["blocks"]:
        if b.get("type")!=0: continue
        bt="\n".join("".join(s["text"] for s in ln["spans"]) for ln in b["lines"]).strip()
        if bt: blocks.append({"page":pn,"bbox":list(b["bbox"]),"text":bt})
        for ln in b["lines"]:
            t="".join(s["text"] for s in ln["spans"]).strip()
            if t: lines.append({"page":pn,"bbox":list(ln["bbox"]),"text":t})
            for sp in ln["spans"]:
                # Keep control characters: some are the circled popularity glyphs.
                if sp["text"]:
                    spans.append({"page":pn,"bbox":list(sp["bbox"]),"text":sp["text"],"font":sp["font"]})
    for b in page.get_text("rawdict")["blocks"]:
        if b.get("type") != 0: continue
        for ln in b["lines"]:
            for sp in ln["spans"]:
                for ch in sp.get("chars", []):
                    chars.append({"page":pn,"bbox":list(ch["bbox"]),"text":ch["c"],"font":sp["font"]})

combined="\n\n".join(f"===== PAGE {p['page']} =====\n{p['text']}" for p in pages)
if DIAGNOSTICS:
    (OUT/"sample-extracted.txt").write_text(combined,encoding="utf-8")
if DIAGNOSTICS:
    for name,data in [("layout-blocks.json",blocks),("layout-lines.json",lines),("layout-spans.json",spans),("layout-chars.json",chars)]:
        (OUT/name).write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding="utf-8")

headers=[]
for b in blocks:
    m=re.search(r"第\s*(\d+)\s*競走",b["text"])
    normalized_header=norm(b["text"])
    year_or_venue=bool(re.search(r"\d{4}年",normalized_header) or re.search(r"[（(]\d{2}(?:札幌|函館|福島|新潟|東京|中山|中京|京都|阪神|小倉)\d+[）)]",normalized_header))
    if m and re.match(r"^\d{5}",normalized_header) and year_or_venue: headers.append({"page":b["page"],"x":b["bbox"][0],"y0":b["bbox"][1],"y1":b["bbox"][3],"no":int(m.group(1)),"text":b["text"]})
races,errors=[],[]
coat=r"黒鹿|青鹿|青毛|鹿毛|栃栗|鹿|栗|芦|白毛|白|青"
identity_re=re.compile(rf"(.+?)(牡|牝|せん|セン)?(\d)({coat})")
for h in headers:
    left=h["x"]<420
    col=lambda x:x["page"]==h["page"] and ((x["bbox"][0]<420)==left)
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
        errors.append(f"R{h['no']}: headcount missing")
        continue

    number_x=(18,42) if left else (429,452)
    identity_x=(45,145) if left else (449.5,555)
    number_lines=[x for x in cl if number_x[0]<=x["bbox"][0]<=number_x[1]
                  and x["bbox"][1]>=h["y1"]+25 and x["bbox"][1]<decl_y]
    horses=[]
    previous_time=None
    previous_margin=None
    for nl in number_lines:
        raw_number_line_source=nl["text"]
        nm=re.match(r"^([1-8])(\d{1,2})(.*)$",norm(raw_number_line_source))
        if not nm: continue
        frame_no,horse_no=int(nm.group(1)),int(nm.group(2))
        ny=center_y(nl)
        identity_parts=[x for x in cl if identity_x[0]<=x["bbox"][0]<identity_x[1]
                        and abs(center_y(x)-ny)<=2.0]
        identity_parts.sort(key=lambda x:x["bbox"][0])
        identity_raw="".join(x["text"] for x in identity_parts)
        identity=norm(identity_raw)
        im=identity_re.search(identity)
        source="identity_column"
        if not im:
            fallback_raw=re.sub(r"^\s*[1-8]\s*\d{1,2}", "", raw_number_line_source)
            fallback=norm(fallback_raw)
            im=identity_re.search(fallback)
            if im:
                identity=fallback
                identity_raw=fallback_raw
                source="number_line_remainder"
        if not im:
            errors.append(f"R{h['no']} horse {horse_no}: identity parse failed {identity!r}")
            continue

        # Weight adjustment is printed in a small, separate line when applicable.
        weight_x=(130,150) if left else (540,560)
        weight_parts=[x for x in cl if weight_x[0]<=x["bbox"][0]<weight_x[1]
                      and abs(center_y(x)-ny)<=4.0 and re.fullmatch(r"\d{2}",norm(x["text"]))]
        carried_weight=None
        if weight_parts:
            carried_weight=int(norm(max(weight_parts,key=center_y)["text"]))
        else:
            wm=re.search(rf"(?:{coat})(\d{{2}})$",identity)
            if wm: carried_weight=int(wm.group(1))

        # Use separate PDF text spans and their x-coordinates for the narrow
        # official columns. This avoids confusing a two-digit weight change
        # with the first digit of a race time.
        def nearest_span(xmin,xmax):
            found=[x for x in cs if xmin<=x["bbox"][0]<xmax and abs(center_y(x)-ny)<=3.0]
            return min(found,key=lambda x:abs(center_y(x)-ny)) if found else None
        body_span=nearest_span(307.5,320.9) if left else nearest_span(718,731.3)
        change_span=nearest_span(320.9,334.4) if left else nearest_span(731.3,744.9)
        time_span=nearest_span(334.4,355.4) if left else nearest_span(744.9,765.8)
        margin_span=nearest_span(355.4,386) if left else nearest_span(765.8,797)
        odds_span=nearest_span(386,402.9) if left else nearest_span(797,813.3)
        glyph_span=nearest_span(402.9,411) if left else nearest_span(813.3,821)

        body_parts=[x for x in cs if (300<=x["bbox"][0]<320.9 if left else 710<=x["bbox"][0]<731.3) and abs(center_y(x)-ny)<=3.0]
        body_parts.sort(key=lambda x:x["bbox"][0])
        body_text=norm("".join(x["text"] for x in body_parts))
        body_match=re.search(r"(?<!\d)(\d{3})(?!\d)",body_text)
        body_weight=int(body_match.group(1)) if body_match else None
        change_parts=[x for x in cs if (320.9<=x["bbox"][0]<334.4 if left else 731.3<=x["bbox"][0]<744.9) and abs(center_y(x)-ny)<=3.0]
        change_parts.sort(key=lambda x:x["bbox"][0])
        change_text=norm("".join(x["text"] for x in change_parts))
        change_match=re.search(r"([＋+－−±])(\d{1,2})",change_text)
        body_change=None
        if change_match:
            n=int(change_match.group(2))
            body_change=n if change_match.group(1) in ("＋","+") else (-n if change_match.group(1) in ("－","−") else 0)

        time_parts=[x for x in cs if (334.4<=x["bbox"][0]<355.4 if left else 744.9<=x["bbox"][0]<765.8) and abs(center_y(x)-ny)<=3.0]
        time_parts.sort(key=lambda x:x["bbox"][0])
        time_raw=norm("".join(x["text"] for x in time_parts))
        time_match=re.fullmatch(r"(?:(\d{1,2})[:：])?(\d{1,2})[．.](\d)",time_raw)
        if time_match:
            minutes=int(time_match.group(1) or 0)
            seconds=int(time_match.group(2))
            finish_time=f"{minutes}:{seconds:02d}.{time_match.group(3)}" if minutes else f"{seconds}.{time_match.group(3)}"
        elif time_raw=="〃":
            finish_time=previous_time
        else:
            finish_time=None
        if finish_time is not None: previous_time=finish_time

        margin_parts=[x for x in cs if (355.4<=x["bbox"][0]<380 if left else 765.8<=x["bbox"][0]<790) and abs(center_y(x)-ny)<=3.0]
        margin_parts.sort(key=lambda x:x["bbox"][0])
        margin_raw=norm("".join(x["text"] for x in margin_parts))
        margin_text=previous_margin if margin_raw=="〃" else (margin_raw or None)
        if margin_text is not None: previous_margin=margin_text
        status_terms=("競走中止","中止","失格","降着","落馬","出走取消","競走除外","取消","除外")
        result_status=next((term for term in status_terms if term in time_raw or term in margin_raw),None)
        finish_time_seconds=time_to_seconds(finish_time) if finish_time else None

        odds_parts=[x for x in cs if (380<=x["bbox"][0]<402.9 if left else 790<=x["bbox"][0]<813.3) and abs(center_y(x)-ny)<=3.0]
        odds_parts.sort(key=lambda x:x["bbox"][0])
        odds_raw=norm("".join(x["text"] for x in odds_parts))
        odds_match=re.search(r"(\d+)[．.](\d)",odds_raw)
        win_odds=float(odds_match.group(1)+"."+odds_match.group(2)) if odds_match else None

        popularity=None
        if glyph_span and len(glyph_span["text"])==1:
            glyph_code=ord(glyph_span["text"])
            popularity=popularity_maps.get(f"{h['page']}:{glyph_span['font']}",{}).get(glyph_code)

        jockey_x=(150,192.1) if left else (560,602.8)
        jockey_parts=[x for x in chars if x["page"]==h["page"] and jockey_x[0]<=x["bbox"][0]<jockey_x[1] and abs(center_y(x)-ny)<=2.5]
        jockey_parts.sort(key=lambda x:x["bbox"][0])
        jockey_raw=norm("".join(x["text"] for x in jockey_parts)) or None
        mark_match=re.match(r"^([△▲☆★◇▽◆]+)",jockey_raw or "")
        jockey_mark=mark_match.group(1) if mark_match else None
        jockey_name=(jockey_raw[len(jockey_mark):] if jockey_mark else jockey_raw) or None
        if jockey_name: jockey_name=jockey_name.replace("．",".")

        horse_name=im.group(1)
        horse_sex=im.group(2)
        # JRA prints gelding (せん) with a custom-font glyph in the sex
        # column. When the horse identity had to fall back to the number line,
        # inspect that dedicated column instead of losing the glyph.
        sex_x=(108,115) if left else (518,525)
        sex_markers=[x for x in chars if x["page"]==h["page"]
                     and sex_x[0]<=x["bbox"][0]<sex_x[1]
                     and abs(center_y(x)-ny)<=2.5
                     and len(x["text"])==1 and ord(x["text"])<32
                     and x["font"].startswith("EdiF-")]
        if horse_sex is None and (re.search(r"[\x01-\x1f]",identity_raw) or sex_markers):
            horse_sex="せん"
        horse_age=int(im.group(3))
        birth_year=URL_YEAR-horse_age if URL_YEAR else None
        horse_key=f"{norm(horse_name)}|{horse_sex or '不明'}|{birth_year or '不明'}"
        horses.append({"frameNumber":frame_no,"horseNumber":horse_no,
                       "horseName":horse_name,"sex":horse_sex,"sexMarkerCode":sex_marker_code,"age":horse_age,
                       "birthYear":birth_year,"horseKey":horse_key,
                       "coatColor":im.group(4),"carriedWeightKg":carried_weight,
                       "bodyWeightKg":body_weight,"bodyWeightChangeKg":body_change,
                       "finishTime":finish_time,"finishTimeSeconds":finish_time_seconds,
                       "marginText":margin_text,"resultStatus":result_status,
                       "winOdds":win_odds,"popularity":popularity,
                       "jockeyNameRaw":jockey_raw,"jockeyAllowanceMark":jockey_mark,"jockeyName":jockey_name,"identitySource":source,
                       "sourcePage":h["page"],"sourceY":round(ny,2)})

    horses.sort(key=lambda x:x["sourceY"])
    rank=0
    tie_group_size=1
    for horse in horses:
        if horse["resultStatus"]:
            horse["finishOrder"]=None
        elif rank==0:
            rank=1
            horse["finishOrder"]=rank
        elif horse["marginText"]=="同着":
            horse["finishOrder"]=rank
            tie_group_size+=1
        else:
            rank += tie_group_size
            horse["finishOrder"]=rank
            tie_group_size=1
    nums=[x["horseNumber"] for x in horses]
    pops=[x["popularity"] for x in horses]
    if len(nums)!=declared: errors.append(f"R{h['no']}: rows {len(nums)} != declared {declared}")
    if len(set(nums))!=len(nums): errors.append(f"R{h['no']}: duplicate horse numbers {nums}")
    if any(n<1 or n>18 for n in nums): errors.append(f"R{h['no']}: horse number outside JRA range {nums}")
    if any(p is None or p<1 or p>declared for p in pops): errors.append(f"R{h['no']}: invalid popularity values {pops}")
    ranked=sorted([x for x in horses if x["winOdds"] is not None and x["popularity"] is not None],key=lambda x:x["winOdds"])
    for a,b in zip(ranked,ranked[1:]):
        if a["winOdds"]<b["winOdds"] and a["popularity"]>b["popularity"]:
            errors.append(f"R{h['no']}: popularity/odds order conflict {a['horseNumber']}:{a['winOdds']}/{a['popularity']} > {b['horseNumber']}:{b['winOdds']}/{b['popularity']}")
    if any(x["winOdds"] is None or x["winOdds"]<=0 for x in horses): errors.append(f"R{h['no']}: missing/invalid win odds")
    if any(x["carriedWeightKg"] is None for x in horses): errors.append(f"R{h['no']}: missing carried weight")
    if any(x["jockeyName"] is None for x in horses): errors.append(f"R{h['no']}: missing jockey name")
    if any(x["finishOrder"] == 1 and x["finishTime"] is None for x in horses): errors.append("R winner: finish time missing")

    nearby=[b["text"] for b in cb if h["y0"]-20<=b["bbox"][1]<=h["y1"]+60]
    card_text="\n".join([h["text"],*nearby])
    year_match=re.search(r"(\d{4})年",card_text)
    day_match=re.search(r"(\d+)月(\d+)日",card_text)
    date_year=int(year_match.group(1)) if year_match else URL_YEAR
    date=f"{date_year:04d}-{int(day_match.group(1)):02d}-{int(day_match.group(2)):02d}" if date_year and day_match else None
    distance_match=re.search(r"(\d)[，,](\d{3})",h["text"])
    index_match=re.match(r"(\d{5})",norm(h["text"]))
    start_match=re.search(r"発走(\d+)時(\d+)分",card_text)
    direction_match=re.search(r"（(?:芝|ダート|障害)[^）]*(右|左)",card_text)
    layout_match=re.search(r"（(?:芝|ダート|障害)[^）]*(内|外)",card_text)
    surface_match=re.search(r"（(芝|ダート|障害)[^）]*）",card_text)
    venue_match=re.search(r"\d{4}年\d+([^）]+)",card_text)
    venue_name=venue_match.group(1) if venue_match else None
    if venue_name is None:
        legacy_venue=re.search(r"[（(]\d{2}(札幌|函館|福島|新潟|東京|中山|中京|京都|阪神|小倉)\d+[）)]",card_text)
        venue_name=legacy_venue.group(1) if legacy_venue else None
    weather_match=re.search(r"\d+月\d+日(晴|曇|雨|小雨|雪)",card_text)
    going_match=re.search(r"\d+月\d+日(?:晴|曇|雨|小雨|雪)?\s*(良|稍重|重|不良)",card_text)
    lower_blocks=[b for b in cb if b["bbox"][1]>300]
    lap_block=next((b for b in lower_blocks if "ハロンタイム" in b["text"]),None)
    passing_block=next((b for b in lower_blocks if "コーナー通過順位" in norm(b["text"])),None)
    corner_block=next((b for b in lower_blocks if "通過コーナー順" in b["text"]),None)
    lap_raw=norm(lap_block["text"]) if lap_block else ""
    passing_raw=norm(passing_block["text"]) if passing_block else ""
    corner_raw=corner_block["text"] if corner_block else ""
    lap_match=re.search(r"ハロンタイム\s*([0-9０-９．.―ー\-]+)",lap_raw)
    lap_times=[float(v.replace("．",".")) for v in re.findall(r"\d+[．.]\d",lap_match.group(1))] if lap_match else []
    passing_times_raw=re.findall(r"(?:\d{1,2}[:：])?\d{1,2}[．.]\d",passing_raw)
    passing_times=[time_to_seconds(value) for value in passing_times_raw]
    corner_order_match=re.search(r"通過コーナー順([0-9→]+)",norm(corner_raw))
    corner_order=[int(v) for v in re.findall(r"\d",corner_order_match.group(1))] if corner_order_match else []
    last_match=re.search(r"上り4F(\d{2}[．.]\d)[―ー\-]+3F(\d{2}[．.]\d)",norm(corner_raw))
    last4f=float(last_match.group(1).replace("．",".")) if last_match else None
    last3f=float(last_match.group(2).replace("．",".")) if last_match else None
    race_key=f"{date}|{venue_name}|{h['no']:02d}" if date and venue_name else None
    races.append({"raceNumber":h["no"],"jraIndexNumber":index_match.group(1) if index_match else None,"raceKey":race_key,"date":date,
                  "venue":venue_name,
                  "distanceMeters":int(distance_match.group(1)+distance_match.group(2)) if distance_match else None,
                  "surface":surface_match.group(1) if surface_match else None,
                  "direction":direction_match.group(1) if direction_match else None,
                  "courseLayout":layout_match.group(1) if layout_match else None,
                  "startTime":f"{int(start_match.group(1)):02d}:{int(start_match.group(2)):02d}" if start_match else None,
                  "weather":weather_match.group(1) if weather_match else None,
                  "going":going_match.group(1) if going_match else None,
                  "lapTimes":lap_times,"passingTimes":passing_times,"passingTimesRaw":passing_times_raw,
                  "cornerOrder":corner_order,"last4F":last4f,"last3F":last3f,
                  "cornerPassageRaw":corner_raw or None,
                  "declaredRunnerCount":declared,"parsedRunnerCount":len(horses),
                  "horses":horses})

if EXPECTED_RACES > 0 and len(headers)!=EXPECTED_RACES: errors.append(f"race headers: {len(headers)} != {EXPECTED_RACES}")
if not headers: errors.append("no race headers found")
if len({r["raceNumber"] for r in races})!=len(races): errors.append("duplicate race numbers")
for race in races:
    for field in ("date","venue","distanceMeters","surface","going"):
        if race[field] is None: errors.append(f"R{race['raceNumber']}: missing race metadata {field}")
    if race["surface"] != "障害":
        if not race["lapTimes"]: errors.append(f"R{race['raceNumber']}: missing lap times")
        if not race["passingTimes"]: errors.append(f"R{race['raceNumber']}: missing passing times")
        if race["lapTimes"] and len(race["passingTimes"]) != len(race["lapTimes"])-1:
            errors.append(f"R{race['raceNumber']}: passing-time count {len(race['passingTimes'])} != lap-time count minus one {len(race['lapTimes'])-1}")
        if any(b<=a for a,b in zip(race["passingTimes"],race["passingTimes"][1:])):
            errors.append(f"R{race['raceNumber']}: passing times are not strictly increasing")
        if not race["cornerOrder"]: errors.append(f"R{race['raceNumber']}: missing corner order")
        if race["last4F"] is None or race["last3F"] is None:
            errors.append(f"R{race['raceNumber']}: missing final 4F/3F")
        winner=next((h for h in race["horses"] if h["finishOrder"]==1),None)
        if winner is None or winner["finishTimeSeconds"] is None:
            errors.append(f"R{race['raceNumber']}: winner finish time missing")
        if winner and winner["finishTimeSeconds"] is not None and race["lapTimes"]:
            winner_seconds=winner["finishTimeSeconds"]
            if abs(sum(race["lapTimes"])-winner_seconds)>0.11:
                    errors.append(f"R{race['raceNumber']}: lap sum {sum(race['lapTimes']):.1f} != winner time {winner_seconds:.1f}")
        if len(race["lapTimes"])>=4 and race["last4F"] is not None:
            if abs(sum(race["lapTimes"][-4:])-race["last4F"])>0.11:
                errors.append(f"R{race['raceNumber']}: final 4F does not match last four lap times")
        if len(race["lapTimes"])>=3 and race["last3F"] is not None:
            if abs(sum(race["lapTimes"][-3:])-race["last3F"])>0.11:
                errors.append(f"R{race['raceNumber']}: final 3F does not match last three lap times")

summary={"source":URL,"sourceSha256":hashlib.sha256(raw).hexdigest(),"httpContentType":content_type,"pdfBytes":len(raw),"pageCount":len(doc),
         "pageCharacters":[p["characters"] for p in pages],"layoutBlockCount":len(blocks),
         "layoutLineCount":len(lines),"layoutSpanCount":len(spans),"layoutCharCount":len(chars),
         "popularityFontMapCount":len(popularity_maps),
         "raceCount":len(headers),"parsedRaceCount":len(races),
         "parsedHorseRows":sum(len(r["horses"]) for r in races),
         "missingBodyWeightCount":sum(h["bodyWeightKg"] is None for r in races for h in r["horses"]),
         "missingSexCount":sum(h["sex"] is None for r in races for h in r["horses"]),
         "missingMarginCount":sum(h["marginText"] is None for r in races for h in r["horses"]),
         "validationErrorCount":len(errors),
         "validationErrors":errors}
(OUT/"parsed-races.json").write_text(json.dumps(races,ensure_ascii=False,indent=2),encoding="utf-8")
(OUT/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
print(json.dumps(summary,ensure_ascii=False))
if len(doc)==0 or len(combined)<1000 or not headers or errors:
    raise RuntimeError(f"Structured validation failed: {len(errors)} errors")
