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

def norm(s):
    return re.sub(r"\s+", "", re.sub(r"[\x00-\x1f]", "", s))

doc = pymupdf.open(stream=raw, filetype="pdf")
pages, blocks, lines = [], [], []
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
combined = "\n\n".join(f"===== PAGE {p['page']} =====\n{p['text']}" for p in pages)
(OUT/"sample-extracted.txt").write_text(combined,encoding="utf-8")
for name,data in [("layout-blocks.json",blocks),("layout-lines.json",lines)]:
    (OUT/name).write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding="utf-8")

headers=[]
for b in blocks:
    m=re.search(r"第\s*(\d+)\s*競走",b["text"])
    if m: headers.append({"page":b["page"],"x":b["bbox"][0],"y1":b["bbox"][3],"no":int(m.group(1)),"text":b["text"]})
races, errors=[],[]
coat=r"黒鹿|青鹿|青毛|鹿毛|栃栗|鹿|栗|芦|白毛|白|青"
identity_re=re.compile(rf"(.+?)(牡|牝|せん|セン)?(\d)({coat})")
for h in headers:
    left=h["x"]<420
    col=lambda x: x["page"]==h["page"] and ((x["bbox"][0]<420)==left)
    cb=[x for x in blocks if col(x)]
    cl=[x for x in lines if col(x)]
    declared=decl_y=None
    for b in cb:
        m=re.search(r"（\s*(\d+)\s*頭\s*）",b["text"])
        if m:
            declared=int(m.group(1))
            # A PDF text block can contain the final horse row and the
            # (N頭) label. Use the label's own line y, not the block top.
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
        ny=(nl["bbox"][1]+nl["bbox"][3])/2
        fragments=[x for x in cl if identity_x[0]<=x["bbox"][0]<identity_x[1]
                   and abs((x["bbox"][1]+x["bbox"][3])/2-ny)<=2.0]
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
        horses.append({"frameNumber":frame_no,"horseNumber":horse_no,
                       "horseName":match.group(1),"sex":match.group(2),
                       "age":int(match.group(3)),"coatColor":match.group(4),
                       "identitySource":source,"sourcePage":h["page"],"sourceY":round(ny,2)})

    horses.sort(key=lambda x:x["sourceY"])
    for order,horse in enumerate(horses,1): horse["finishOrder"]=order
    nums=[x["horseNumber"] for x in horses]
    if len(nums)!=declared: errors.append(f"R{h['no']}: rows {len(nums)} != declared {declared}")
    if len(set(nums))!=len(nums): errors.append(f"R{h['no']}: duplicate horse numbers {nums}")
    if any(n<1 or n>18 for n in nums): errors.append(f"R{h['no']}: horse number outside JRA range {nums}")
    ht=norm(h["text"])
    dm=re.search(r"(\d{4})年(\d+)月(\d+)日",ht)
    dist=re.search(r"(\d)[，,](\d{3})",ht)
    surf=re.search(r"（(芝|ダート|障害)[^）]*）",ht)
    vm=re.search(r"\d{4}年\d+([^）]+)",ht)
    races.append({"raceNumber":h["no"],
                  "date":f"{int(dm.group(1)):04d}-{int(dm.group(2)):02d}-{int(dm.group(3)):02d}" if dm else None,
                  "venue":vm.group(1) if vm else None,
                  "distanceMeters":int(dist.group(1)+dist.group(2)) if dist else None,
                  "surface":surf.group(1) if surf else None,
                  "declaredRunnerCount":declared,"parsedRunnerCount":len(horses),
                  "horses":horses})
summary={"source":URL,"httpContentType":content_type,"pdfBytes":len(raw),"pageCount":len(doc),
         "pageCharacters":[p["characters"] for p in pages],"layoutBlockCount":len(blocks),
         "layoutLineCount":len(lines),"raceCount":len(headers),"parsedRaceCount":len(races),
         "parsedHorseRows":sum(len(r["horses"]) for r in races),"validationErrorCount":len(errors),
         "validationErrors":errors}
(OUT/"parsed-races.json").write_text(json.dumps(races,ensure_ascii=False,indent=2),encoding="utf-8")
(OUT/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
print(json.dumps(summary,ensure_ascii=False))
if len(doc)==0 or len(combined)<1000 or len(headers)!=12 or errors:
    raise RuntimeError(f"Structured validation failed: {len(errors)} errors")
