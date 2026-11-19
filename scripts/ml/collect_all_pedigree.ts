// 成績表の全平地レース（2022年〜）の出走馬について、netkeiba から 馬ID・父/母父/母母父・その産駒成績を集める。
// 本体モデル（LightGBM）に血統の特徴量を入れるための学習データ。
//   npx tsx scripts/ml/collect_all_pedigree.ts [--max-minutes N]
// 取得済みは飛ばすので、途中で止めても続きから再開できる。保存先: data/pedigree/all/
import { existsSync, mkdirSync, readdirSync, readFileSync, writeFileSync } from "node:fs";
import path from "node:path";
import { cleanText, extractPedigree } from "@/app/lib/netkeiba";
import { parseStallionPages, PEDIGREE_SOURCES, type StallionRaw } from "@/app/lib/pedigree";
import type { GradedRace, HorsePedigree } from "./collect_pedigree";

const ROOT = path.resolve(__dirname, "../..");
const SEISEKI = path.join(ROOT, "data/jra-seiseki/json");
const PED = path.join(ROOT, "data/pedigree");
const OUT = path.join(PED, "all");
// netkeiba への負担を抑えるため、GitHub Actions では環境変数で同時数と間隔を絞る
const CONCURRENCY = Number(process.env.PEDIGREE_CONCURRENCY ?? 3);
const WAIT_MS = Number(process.env.PEDIGREE_WAIT_MS ?? 300);

// race_id -> { 馬番: 馬ID }
type RaceHorses = Record<string, Record<string, string>>;
// 馬ID -> 父・母父・母母父 の [ID, 名前]（ID がない祖先は null）と父系統
export type CompactPedigree = { F: [string | null, string] | null; MF: [string | null, string] | null; MMF: [string | null, string] | null; line: string | null };

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));
const readJson = <T>(file: string, fallback: T): T => (existsSync(file) ? JSON.parse(readFileSync(file, "utf8")) as T : fallback);
const writeJson = (file: string, value: unknown) => { mkdirSync(path.dirname(file), { recursive: true }); writeFileSync(file, JSON.stringify(value) + "\n"); };

const maxMinutesArg = process.argv.indexOf("--max-minutes");
const deadline = maxMinutesArg > 0 ? Date.now() + Number(process.argv[maxMinutesArg + 1]) * 60000 : Infinity;
let requests = 0;

async function get(url: string): Promise<string | null> {
  for (let attempt = 0; attempt < 3; attempt++) {
    try {
      await sleep(WAIT_MS);
      const res = await fetch(url, { headers: { "User-Agent": "Mozilla/5.0" }, signal: AbortSignal.timeout(20000) });
      requests++;
      if (res.status === 404) return null;
      if (!res.ok) throw new Error("HTTP " + res.status);
      return await res.text();
    } catch (error) {
      if (attempt === 2) { console.error("  ! " + url + ": " + error); return null; }
      await sleep(3000 * (attempt + 1));
    }
  }
  return null;
}

async function pool<T>(items: T[], label: string, every: number, save: () => void, work: (item: T) => Promise<void>) {
  let next = 0;
  let done = 0;
  await Promise.all(Array.from({ length: CONCURRENCY }, async () => {
    while (next < items.length && Date.now() < deadline) {
      await work(items[next++]);
      if (++done % every === 0) { save(); console.log(`  ${label} ${done}/${items.length} (${requests} requests)`); }
    }
  }));
  save();
}

function flatRaceIds(): string[] {
  const ids: string[] = [];
  for (const year of readdirSync(SEISEKI).sort()) {
    for (const file of readdirSync(path.join(SEISEKI, year)).filter((f) => f.endsWith(".json"))) {
      const day = JSON.parse(readFileSync(path.join(SEISEKI, year, file), "utf8"));
      for (const r of day.races) if (r.race_id && ["芝", "ダート"].includes(r.surface)) ids.push(r.race_id);
    }
  }
  return ids.sort();
}

function parseRaceHorseIds(html: string): Record<string, string> {
  const table = html.match(/<table\b[^>]*race_table_01[\s\S]*?<\/table>/)?.[0] ?? "";
  const map: Record<string, string> = {};
  for (const row of table.match(/<tr\b[\s\S]*?<\/tr>/g) ?? []) {
    const cells = [...row.matchAll(/<td\b[^>]*>([\s\S]*?)<\/td>/g)].map((m) => m[1]);
    const id = row.match(/\/horse\/(\w+)\//)?.[1];
    const umaban = cleanText(cells[2] ?? "");
    if (id && /^\d+$/.test(umaban)) map[umaban] = id;
  }
  return map;
}

async function main() {
  const raceFile = path.join(OUT, "race_horses.json");
  const horseFile = path.join(OUT, "horses.json");
  const stallionFile = path.join(PED, "stallions.json");
  const raceHorses = readJson<RaceHorses>(raceFile, {});
  const horses = readJson<Record<string, CompactPedigree | null>>(horseFile, {});
  const stallions = readJson<Record<string, StallionRaw>>(stallionFile, {});

  // 重賞で取得済みのものを流用
  const graded = readJson<Record<string, GradedRace>>(path.join(PED, "graded_races.json"), {});
  for (const r of Object.values(graded)) {
    if (!raceHorses[r.raceId] && r.runners.every((h) => h.horseId)) {
      raceHorses[r.raceId] = Object.fromEntries(r.runners.map((h) => [String(h.umaban), h.horseId!]));
    }
  }
  const gradedHorses = readJson<Record<string, HorsePedigree | null>>(path.join(PED, "horses.json"), {});
  const compact = (ped: { ancestors: { position: string; horseId: string | null; name: string }[]; sireLine: string | null }): CompactPedigree => {
    const at = (p: string) => { const a = ped.ancestors.find((x) => x.position === p); return a ? [a.horseId, a.name] as [string | null, string] : null; };
    return { F: at("F"), MF: at("MF"), MMF: at("MMF"), line: ped.sireLine };
  };
  for (const [id, ped] of Object.entries(gradedHorses)) if (ped && !horses[id]) horses[id] = compact(ped);

  // 1) レース → 馬ID
  const raceTodo = flatRaceIds().filter((id) => !raceHorses[id]);
  console.log(`races: ${raceTodo.length} to map`);
  await pool(raceTodo, "races", 200, () => writeJson(raceFile, raceHorses), async (id) => {
    const html = await get(`https://db.netkeiba.com/race/${id}/`);
    if (html) { const map = parseRaceHorseIds(html); if (Object.keys(map).length) raceHorses[id] = map; }
  });

  // 2) 馬 → 父・母父・母母父
  const ids = [...new Set(Object.values(raceHorses).flatMap((m) => Object.values(m)))];
  const horseTodo = ids.filter((id) => !(id in horses));
  console.log(`horses: ${ids.length} referenced, ${horseTodo.length} to fetch`);
  await pool(horseTodo, "horses", 500, () => writeJson(horseFile, horses), async (id) => {
    const html = await get(`https://db.netkeiba.com/horse/ped/${id}/`);
    if (html === null) return;
    const ped = extractPedigree(html);
    horses[id] = ped ? compact(ped) : null;
  });

  // 3) 父・母父・母母父 → 産駒成績（年度別・種牡馬/BMS）
  const names = new Map<string, string>();
  for (const h of Object.values(horses)) {
    for (const s of PEDIGREE_SOURCES) {
      const a = h?.[s.position as "F" | "MF" | "MMF"];
      if (a?.[0]) names.set(a[0], a[1]);
    }
  }
  const stallionTodo = [...names.keys()].filter((id) => !stallions[id]);
  console.log(`stallions: ${names.size} referenced, ${stallionTodo.length} to fetch`);
  const today = new Date().toISOString().slice(0, 10);
  await pool(stallionTodo, "stallions", 100, () => writeJson(stallionFile, stallions), async (id) => {
    const url = (t: number) => `https://db.netkeiba.com/horse/sire/${id}/?syusai=10&mode=1&type=${t}`;
    const going = await get(url(1));
    const distance = await get(url(2));
    if (going || distance) stallions[id] = parseStallionPages(id, names.get(id) ?? null, { going, distance }, today);
  });
  console.log(Date.now() >= deadline ? "time limit reached; run again to continue" : "done", `(${requests} requests)`);
}

main().catch((e) => { console.error(e); process.exit(1); });
