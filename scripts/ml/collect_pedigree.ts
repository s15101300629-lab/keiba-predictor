// 重賞の出走馬について、netkeiba から馬ID・血統・父/母父/母母父の産駒成績を集めて data/pedigree に保存する。
//   npx tsx scripts/ml/collect_pedigree.ts [年 ...]      (既定: 2023〜今年)
// 取得済みのレース・馬・種牡馬は飛ばすので、途中で止めても続きから再開できる。
import { existsSync, mkdirSync, readdirSync, readFileSync, writeFileSync } from "node:fs";
import path from "node:path";
import { cleanText, extractPedigree, type Pedigree } from "@/app/lib/netkeiba";
import { allMaleSources, parseStallionPages, type StallionRaw } from "@/app/lib/pedigree";

const ROOT = path.resolve(__dirname, "../..");
const SEISEKI = path.join(ROOT, "data/jra-seiseki/json");
const OUT = path.join(ROOT, "data/pedigree");
const CONCURRENCY = 3;
const WAIT_MS = 300;
const PED_GENERATIONS = 4; // 血統表を保存する世代数（血統の補正の検証で4代まで使う）
const GRADED = /[(（](?:G|Ｇ)(?:Ⅰ|Ⅱ|Ⅲ|1|2|3|１|２|３)[)）]/;

export type GradedRace = {
  raceId: string; date: string; raceName: string; surface: string; distance: number; going: string | null;
  runners: { umaban: number; horse: string; horseId: string | null; finish: number | null }[];
};
export type HorsePedigree = {
  horseId: string; sire: string | null; dam: string | null; damsire: string | null; damdam: string | null;
  sireLine: string | null;
  ancestors: Pedigree["ancestors"]; // PED_GENERATIONS 代まで
  crosses: Pedigree["crosses"];
};

const readJson = <T>(file: string, fallback: T): T => (existsSync(file) ? JSON.parse(readFileSync(file, "utf8")) as T : fallback);
const writeJson = (file: string, value: unknown) => {
  mkdirSync(path.dirname(file), { recursive: true });
  writeFileSync(file, JSON.stringify(value, null, 0) + "\n");
};
const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

let requests = 0;
async function get(url: string): Promise<string | null> {
  for (let attempt = 0; attempt < 3; attempt++) {
    try {
      await sleep(WAIT_MS);
      const res = await fetch(url, { headers: { "User-Agent": "Mozilla/5.0" }, signal: AbortSignal.timeout(20000) });
      requests++;
      if (res.status === 404) return null;
      if (!res.ok) throw new Error("HTTP " + res.status);
      const buf = new Uint8Array(await res.arrayBuffer());
      const charset = (res.headers.get("content-type") ?? "").match(/charset=([\w-]+)/i)?.[1]?.toLowerCase();
      return new TextDecoder(charset === "euc-jp" ? "euc-jp" : "utf-8").decode(buf);
    } catch (error) {
      if (attempt === 2) { console.error("  ! " + url + ": " + error); return null; }
      await sleep(2000 * (attempt + 1));
    }
  }
  return null;
}

async function pool<T>(items: T[], work: (item: T, index: number) => Promise<void>) {
  let next = 0;
  await Promise.all(Array.from({ length: CONCURRENCY }, async () => {
    while (next < items.length) { const i = next++; await work(items[i], i); }
  }));
}

function listGradedRaces(years: number[]): GradedRace[] {
  const races: GradedRace[] = [];
  for (const year of years) {
    const dir = path.join(SEISEKI, String(year));
    if (!existsSync(dir)) continue;
    for (const file of readdirSync(dir).filter((f) => f.endsWith(".json")).sort()) {
      const day = JSON.parse(readFileSync(path.join(dir, file), "utf8"));
      for (const r of day.races) {
        if (!GRADED.test(r.race_name ?? "") || !["芝", "ダート"].includes(r.surface) || !r.race_id) continue;
        races.push({
          raceId: r.race_id, date: r.date, raceName: r.race_name, surface: r.surface, distance: r.distance,
          going: r.going ?? null,
          runners: r.results.filter((h: { status: string | null }) => h.status !== "取消" && h.status !== "除外")
            .map((h: { umaban: number; horse: string; finish: number | null; status: string | null }) => ({
              umaban: h.umaban, horse: h.horse, horseId: null, finish: h.status === "中止" ? null : h.finish,
            })),
        });
      }
    }
  }
  return races.sort((a, b) => a.raceId.localeCompare(b.raceId));
}

/** netkeiba のレース結果ページから 馬番 → 馬ID. */
function parseRaceHorseIds(html: string): Map<number, string> {
  const table = html.match(/<table\b[^>]*race_table_01[\s\S]*?<\/table>/)?.[0] ?? "";
  const map = new Map<number, string>();
  for (const row of table.match(/<tr\b[\s\S]*?<\/tr>/g) ?? []) {
    const cells = [...row.matchAll(/<td\b[^>]*>([\s\S]*?)<\/td>/g)].map((m) => m[1]);
    const id = row.match(/\/horse\/(\w+)\//)?.[1];
    const umaban = Number(cleanText(cells[2] ?? ""));
    if (id && Number.isInteger(umaban) && umaban > 0) map.set(umaban, id);
  }
  return map;
}

async function main() {
  const thisYear = new Date().getFullYear();
  const years = process.argv.slice(2).map(Number).filter(Boolean);
  const targetYears = years.length ? years : Array.from({ length: thisYear - 2022 }, (_, i) => 2023 + i);
  const racesFile = path.join(OUT, "graded_races.json");
  const horsesFile = path.join(OUT, "horses.json");
  const stallionsFile = path.join(OUT, "stallions.json");
  const races = readJson<Record<string, GradedRace>>(racesFile, {});
  const horses = readJson<Record<string, HorsePedigree | null>>(horsesFile, {});
  const stallions = readJson<Record<string, StallionRaw>>(stallionsFile, {});

  // 1) レース → 馬ID
  const pending = listGradedRaces(targetYears).filter((r) => !races[r.raceId]?.runners.every((h) => h.horseId));
  console.log(`graded races ${targetYears.join(",")}: ${pending.length} to map`);
  await pool(pending, async (race, i) => {
    const html = await get(`https://db.netkeiba.com/race/${race.raceId}/`);
    if (!html) return;
    const ids = parseRaceHorseIds(html);
    race.runners.forEach((h) => { h.horseId = ids.get(h.umaban) ?? null; });
    races[race.raceId] = race;
    if (i % 20 === 19) { writeJson(racesFile, races); console.log(`  races ${i + 1}/${pending.length}`); }
  });
  writeJson(racesFile, races);

  // 2) 馬 → 血統（3代まで・クロス）
  const horseIds = [...new Set(Object.values(races).flatMap((r) => r.runners.map((h) => h.horseId)).filter((id): id is string => !!id))];
  // 未取得、または保存している世代が足りない馬
  const horseTodo = horseIds.filter((id) => !(id in horses) ||
    (horses[id] !== null && Math.max(...horses[id]!.ancestors.map((a) => a.generation)) < PED_GENERATIONS));
  console.log(`horses: ${horseIds.length} referenced, ${horseTodo.length} to fetch`);
  await pool(horseTodo, async (id, i) => {
    const html = await get(`https://db.netkeiba.com/horse/ped/${id}/`);
    const ped = html ? extractPedigree(html) : null;
    horses[id] = ped ? {
      horseId: id, sire: ped.sire, dam: ped.dam, damsire: ped.damsire, damdam: ped.damdam, sireLine: ped.sireLine,
      ancestors: ped.ancestors.filter((a) => a.generation <= PED_GENERATIONS), crosses: ped.crosses,
    } : null;
    if (i % 100 === 99) { writeJson(horsesFile, horses); console.log(`  horses ${i + 1}/${horseTodo.length}`); }
  });
  writeJson(horsesFile, horses);

  // 3) 父・母父・母母父 → 産駒成績（年度別）
  const stallionNames = new Map<string, string>();
  for (const h of Object.values(horses)) {
    for (const s of allMaleSources(PED_GENERATIONS)) {
      const a = h?.ancestors.find((x) => x.position === s.position);
      if (a?.horseId) stallionNames.set(a.horseId, a.name);
    }
  }
  const stallionTodo = [...stallionNames.keys()].filter((id) => !stallions[id]);
  console.log(`stallions: ${stallionNames.size} referenced, ${stallionTodo.length} to fetch`);
  const today = new Date().toISOString().slice(0, 10);
  await pool(stallionTodo, async (id, i) => {
    const url = (type: number) => `https://db.netkeiba.com/horse/sire/${id}/?syusai=10&mode=1&type=${type}`;
    const going = await get(url(1));
    const distance = await get(url(2));
    if (!going && !distance) return;
    stallions[id] = parseStallionPages(id, stallionNames.get(id) ?? null, { going, distance }, today);
    if (i % 50 === 49) { writeJson(stallionsFile, stallions); console.log(`  stallions ${i + 1}/${stallionTodo.length}`); }
  });
  writeJson(stallionsFile, stallions);
  console.log(`done: ${requests} requests`);
}

main().catch((error) => { console.error(error); process.exit(1); });
