// 成績表の全平地レースの出走馬について、父・母父・母母父の産駒成績（レース前年まで）から血統の特徴量を作る。
//   npx tsx scripts/ml/pedigree_all_features.ts  -> data/pedigree/all/runner_features.csv
// 入力：data/pedigree/all/race_horses.json（レース→馬番→馬ID）、horses.json（馬→父・母父・母母父）、data/pedigree/stallions.json
// 列（祖先 F・MF・MMF ごと）：
//   heavy：重・不良の3着内率 ÷ 良（縮小推定）、wet：稍重〜不良 ÷ 良、starts：今回の芝ダでの産駒の出走数、
//   top3：今回の芝ダでの産駒の3着内率、dist：今回の距離区分の3着内率 ÷ 全距離区分の平均
import { existsSync, readdirSync, readFileSync, writeFileSync } from "node:fs";
import path from "node:path";
import { PEDIGREE_SOURCES, stallionAptitude, stallionRecord, type StallionRaw, type StallionRecord } from "@/app/lib/pedigree";
import type { CompactPedigree } from "./collect_all_pedigree";

const ROOT = path.resolve(__dirname, "../..");
const SEISEKI = path.join(ROOT, "data/jra-seiseki/json");
const PED = path.join(ROOT, "data/pedigree");
const raceHorses = JSON.parse(readFileSync(path.join(PED, "all/race_horses.json"), "utf8")) as Record<string, Record<string, string>>;
const horses = JSON.parse(readFileSync(path.join(PED, "all/horses.json"), "utf8")) as Record<string, CompactPedigree | null>;
const stallions = JSON.parse(readFileSync(path.join(PED, "stallions.json"), "utf8")) as Record<string, StallionRaw>;

const cache = new Map<string, StallionRecord | null>();
function record(id: string, role: "sire" | "damsire", throughYear: number): StallionRecord | null {
  const key = `${id}|${role}|${throughYear}`;
  if (!cache.has(key)) cache.set(key, stallions[id] ? stallionRecord(stallions[id], role, throughYear) : null);
  return cache.get(key)!;
}

const cols = PEDIGREE_SOURCES.flatMap((s) => ["heavy", "wet", "starts", "top3", "dist"].map((k) => `${s.position}_${k}`));
const lines = ["race_id,umaban,horse_id," + cols.join(",")];
let races = 0;
let missing = 0;
for (const year of readdirSync(SEISEKI).sort()) {
  for (const file of readdirSync(path.join(SEISEKI, year)).filter((f) => f.endsWith(".json")).sort()) {
    const day = JSON.parse(readFileSync(path.join(SEISEKI, year, file), "utf8"));
    for (const race of day.races) {
      if (!race.race_id || !["芝", "ダート"].includes(race.surface)) continue;
      const map = raceHorses[race.race_id];
      if (!map) { missing++; continue; }
      races++;
      const throughYear = Number(race.date.slice(0, 4)) - 1;
      for (const [umaban, horseId] of Object.entries(map)) {
        const ped = horses[horseId];
        const values = PEDIGREE_SOURCES.flatMap((s) => {
          const id = ped?.[s.position as "F" | "MF" | "MMF"]?.[0];
          const rec = id ? record(id, s.role as "sire" | "damsire", throughYear) : null;
          if (!rec) return ["", "", "", "", ""];
          const apt = stallionAptitude(rec, race.surface, race.distance ?? null);
          const g = rec.going[race.surface] ?? {};
          let n = 0;
          let t = 0;
          for (const c of Object.values(g)) { if (c) { n += c[0] + c[1] + c[2] + c[3]; t += c[0] + c[1] + c[2]; } }
          const bands = Object.values(apt.bandRates);
          const mean = bands.length ? bands.reduce((a, b) => a + b, 0) / bands.length : 0;
          const dist = apt.band && apt.bandRates[apt.band] !== undefined && mean > 0 ? apt.bandRates[apt.band] / mean : "";
          return [apt.heavyRatio ?? "", apt.wetRatio ?? "", n, n ? t / n : "", dist];
        });
        lines.push([race.race_id, umaban, horseId, ...values].join(","));
      }
    }
  }
}
writeFileSync(path.join(PED, "all/runner_features.csv"), lines.join("\n") + "\n");
console.log(`races ${races}, races without horse ids ${missing}, rows ${lines.length - 1}`);
if (!existsSync(path.join(PED, "all/runner_features.csv"))) process.exit(1);
