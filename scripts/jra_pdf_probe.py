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
for h in headers:
    left=h["x"]<420
    col=lambda x: x["page"]==h["page"] and ((x["bbox"][0]<420)==left)
    cb=[x for x in blocks if col(x)]
    cl=[x for x in lines if col(x)]
    declared=decl_y=None
    for b in cb:
        m=re.search(r"（\s*(\d+)\s*頭\s*）",b["text"])
        if m: declared,decl_y=int(m.group(1)),b["bbox"][1]; break
    if declared is None:
        errors.append(f"R{h['no']}: headcount missing"); continue
    candidates=[x for x in cl if x["bbox"][1]>=h["y1"]+25 and x["bbox"][1]<decl_y+1]
    candidates.sort(key=lambda x:(sum(x["bbox"][1:4:2])/2,x["bbox"][0]))
    rows=[]
    for ln in candidates:
        cy=(ln["bbox"][1]+ln["bbox"][3])/2
        if not rows or abs(cy-rows[-1]["y"])>3.0: rows.append({"y":cy,"lines":[ln]})
        else:
            rows[-1]["lines"].append(ln)
            rows[-1]["y"]=sum((z["bbox"][1]+z["bbox"][3])/2 for z in rows[-1]["lines"])/len(rows[-1]["lines"])
    horses=[]
    for row in rows:
        numline=next((norm(x["text"]) for x in row["lines"] if x["bbox"][0]<(25 if left else 435)),"")
        nm=re.match(r"^([1-8])\s*(\d{1,2})",numline)
        if not nm: continue
        xmin,xmax=(40,155) if left else (450,585)
        # Join all fragments from the identity column around the horse-number baseline.
        fr=[x for x in cl if xmin<=x["bbox"][0]<xmax and abs((x["bbox"][1]+x["bbox"][3])/2-row["y"])<=4.5]
        fr.sort(key=lambda x:(x["bbox"][0],x["bbox"][1]))
        identity=norm("".join(x["text"] for x in fr))
        im=re.search(rf"(.+?)(牡|牝|せん|セン)(\d)({coat})",identity)
        if not im:
            errors.append(f"R{h['no']} horse {nm.group(2)} identity={identity!r}")
            continue
        horses.append({"frameNumber":int(nm.group(1)),"horseNumber":int(nm.group(2)),"horseName":im.group(1),"sex":im.group(2),"age":int(im.group(3)),"coatColor":im.group(4),"sourcePage":h["page"],"sourceY":round(row["y"],2)})
    nums=[x["horseNumber"] for x in horses]
    if len(nums)!=declared: errors.append(f"R{h['no']}: rows {len(nums)} != declared {declared}")
    if sorted(nums)!=list(range(1,declared+1)): errors.append(f"R{h['no']}: horse numbers {sorted(nums)}")
    ht=norm(h["text"])
    dm=re.search(r"(\d{4})年(\d+)月(\d+)日",ht)
    dist=re.search(r"(\d)[，,](\d{3})",ht)
    surf=re.search(r"（(芝|ダート|障害)[^）]*）",ht)
    vm=re.search(r"\d{4}年\d+([^）]+)",ht)
    races.append({"raceNumber":h["no"],"date":f"{int(dm.group(1)):04d}-{int(dm.group(2)):02d}-{int(dm.group(3)):02d}" if dm else None,"venue":vm.group(1) if vm else None,"distanceMeters":int(dist.group(1)+dist.group(2)) if dist else None,"surface":surf.group(1) if surf else None,"declaredRunnerCount":declared,"parsedRunnerCount":len(horses),"horses":horses})
summary={"source":URL,"httpContentType":content_type,"pdfBytes":len(raw),"pageCount":len(doc),"pageCharacters":[p["characters"] for p in pages],"layoutBlockCount":len(blocks),"layoutLineCount":len(lines),"raceCount":len(headers),"parsedRaceCount":len(races),"parsedHorseRows":sum(len(r["horses"]) for r in races),"validationErrorCount":len(errors),"validationErrors":errors}
(OUT/"parsed-races.json").write_text(json.dumps(races,ensure_ascii=False,indent=2),encoding="utf-8")
(OUT/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
print(json.dumps(summary,ensure_ascii=False))
if len(doc)==0 or len(combined)<1000 or len(headers)!=12 or errors: raise RuntimeError(f"Structured validation failed: {len(errors)} errors")
