// これから走るレースの出走馬の父・母父・母母父のうち、data/pedigree/stallions.json にない（または年替わりで取り直す）種牡馬の
// 産駒成績を、レース前に取ってファイルに入れる（2026-11-25、依頼者の指示）。
// アプリ（レース詳細 API）は種牡馬の成績をまずこのファイルから読み、ないものだけ netkeiba から取るので、表示が速くなる。
//   npx tsx scripts/ml/collect_upcoming_pedigree.ts [YYYYMMDD ...]   # 日付を省くと、日本時間の今日から4日分
// 取るもの：開催一覧（日ごと）→ 出馬表（レースごと）→ data/pedigree/all/horses.json にない馬の血統表 → 足りない種牡馬の成績（2ページずつ）。
// 1回に取るのは「予測する出走馬」の分だけ（依頼者の方針）。馬の血統（horses.json）も一緒に保存する（毎週の再学習の collect_all_pedigree.ts と同じ形）
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import path from "node:path";
import { extractPedigree } from "@/app/lib/netkeiba";
import { jstToday, parseStallionPages, stallionNeedsRefresh, PEDIGREE_SOURCES, type StallionRaw } from "@/app/lib/pedigree";
import type { CompactPedigree } from "./collect_all_pedigree";

const ROOT = path.resolve(__dirname, "../..");
const PED = path.join(ROOT, "data/pedigree");
const WAIT_MS = Number(process.env.PEDIGREE_WAIT_MS ?? 1000);

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));
const readJson = <T>(file: string, fallback: T): T => (existsSync(file) ? JSON.parse(readFileSync(file, "utf8")) as T : fallback);
const writeJson = (file: string, value: unknown) => { mkdirSync(path.dirname(file), { recursive: true }); writeFileSync(file, JSON.stringify(value) + "\n"); };
let requests = 0;

// アプリと同じく、User-Agent は Node のまま（ブラウザのふりはしない）。1本ずつ・WAIT_MS 間隔
async function get(url: string): Promise<string | null> {
  for (let attempt = 0; attempt < 3; attempt++) {
    try {
      await sleep(WAIT_MS);
      const res = await fetch(url, { headers: { Accept: "text/html" }, signal: AbortSignal.timeout(20000) });
      requests++;
      if (res.status === 404) return null;
      if (!res.ok) throw new Error("HTTP " + res.status);
      return new TextDecoder("utf-8").decode(new Uint8Array(await res.arrayBuffer()));
    } catch (error) {
      if (attempt === 2) { console.error("  ! " + url + ": " + error); return null; }
      await sleep(3000 * (attempt + 1));
    }
  }
  return null;
}

function jstDates(n: number): string[] {
  const [y, m, d] = jstToday().split("-").map(Number);
  return Array.from({ length: n }, (_, k) => {
    const t = new Date(Date.UTC(y, m - 1, d + k));
    return `${t.getUTCFullYear()}${String(t.getUTCMonth() + 1).padStart(2, "0")}${String(t.getUTCDate()).padStart(2, "0")}`;
  });
}

const uniq = <T>(xs: T[]) => [...new Set(xs)];

async function main() {
  const args = process.argv.slice(2).filter((a) => /^\d{8}$/.test(a));
  const dates = args.length ? args : jstDates(4);
  const horseFile = path.join(PED, "all", "horses.json");
  const stallionFile = path.join(PED, "stallions.json");
  const horses = readJson<Record<string, CompactPedigree | null>>(horseFile, {});
  const stallions = readJson<Record<string, StallionRaw>>(stallionFile, {});

  // 1) 開催一覧 → レース
  const raceIds: string[] = [];
  for (const date of dates) {
    const html = await get(`https://race.netkeiba.com/top/race_list_sub.html?kaisai_date=${date}`);
    const ids = uniq([...(html ?? "").matchAll(/race_id=(\d{12})/g)].map((m) => m[1]));
    console.log(`${date}: ${ids.length} races`);
    raceIds.push(...ids);
  }

  // 2) 出馬表 → 馬
  const horseIds: string[] = [];
  const overseasCheck: string[] = []; // 海外の走（フランスの区間タイム）を調べる馬：重賞（G1〜G3）の出走馬と、海外の厩舎の馬（francegalop.py upcoming が読む）
  for (const id of uniq(raceIds)) {
    const html = await get(`https://race.netkeiba.com/race/shutuba.html?race_id=${id}`);
    horseIds.push(...[...(html ?? "").matchAll(/\/horse\/(\d{10})/g)].map((m) => m[1]));
    const graded = /Icon_GradeType[123]\b/.test(html ?? "");
    for (const row of (html ?? "").split(/<tr/)) {
      const h = row.match(/\/horse\/(\d{10})/);
      if (h && (graded || row.includes("海外"))) overseasCheck.push(h[1]);
    }
  }
  writeJson(path.join(ROOT, "data/overseas/upcoming_overseas.json"), { dates, horses: uniq(overseasCheck) });
  const runners = uniq(horseIds);
  const horseTodo = runners.filter((id) => !horses[id]);
  console.log(`horses: ${runners.length} runners, ${horseTodo.length} pedigrees to fetch`);

  // 3) 血統表がない馬（主に新馬）→ 父・母父・母母父
  for (const id of horseTodo) {
    const html = await get(`https://db.netkeiba.com/horse/ped/${id}/`);
    if (html === null) continue;
    const ped = extractPedigree(html);
    if (!ped) continue;
    const at = (p: string) => { const a = ped.ancestors.find((x) => x.position === p); return a ? [a.horseId, a.name] as [string | null, string] : null; };
    horses[id] = { F: at("F"), MF: at("MF"), MMF: at("MMF"), line: ped.sireLine };
  }
  if (horseTodo.length) writeJson(horseFile, horses);

  // 4) 種牡馬：ファイルにないもの、年が変わって取り直すもの（アプリの判定と同じ stallionNeedsRefresh）
  const names = new Map<string, string>();
  for (const id of runners) {
    for (const s of PEDIGREE_SOURCES) {
      const a = horses[id]?.[s.position as "F" | "MF" | "MMF"];
      if (a?.[0]) names.set(a[0], a[1]);
    }
  }
  const todo = [...names.keys()].filter((id) => !stallions[id] || stallionNeedsRefresh(stallions[id]));
  console.log(`stallions: ${names.size} referenced, ${todo.length} to fetch`, todo.map((id) => names.get(id)).join("・"));
  const today = jstToday();
  for (const id of todo) {
    const url = (t: number) => `https://db.netkeiba.com/horse/sire/${id}/?syusai=10&mode=1&type=${t}`;
    const going = await get(url(1));
    const distance = await get(url(2));
    if (going || distance) stallions[id] = parseStallionPages(id, names.get(id) ?? null, { going, distance }, today);
  }
  if (todo.length) writeJson(stallionFile, stallions);
  console.log(`done (${requests} requests)`);
}

main().catch((e) => { console.error(e); process.exit(1); });
