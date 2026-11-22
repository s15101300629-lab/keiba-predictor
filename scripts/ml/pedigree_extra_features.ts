// 血統の追加候補（compare_pedigree2.py の比較用）。成績表の全平地レースの出走馬について、父・母父・母母父の産駒成績（レース前年まで）から作る。
//   npx tsx scripts/ml/pedigree_extra_features.ts  -> data/pedigree/all/runner_features_extra.csv（git には入れない）
// 列（どれも父・母父・母母父を 50:25:12.5 で平均。値のない祖先は除いて重みを割り直す）
//   pb_band_top3：今回の芝ダ×距離区分での産駒の3着内率（出走数が少ない区分は、その芝ダ全体の率に寄せる）
//   pb_going_top3：今回の芝ダ×今回の馬場（良・稍重・重・不良）での3着内率（同じく寄せる）
//   pb_win：今回の芝ダでの産駒の勝率
//   pb_recent_top3：今回の芝ダでの、レース前年までの直近3年の3着内率（今の産駒の質）
//   pb_f_band_top3：父だけの pb_band_top3
import { readdirSync, readFileSync, writeFileSync } from "node:fs";
import path from "node:path";
import { PEDIGREE_SOURCES, stallionRecord, type StallionRaw, type StallionRecord } from "@/app/lib/pedigree";
import type { CompactPedigree } from "./collect_all_pedigree";

const ROOT = path.resolve(__dirname, "../..");
const SEISEKI = path.join(ROOT, "data/jra-seiseki/json");
const PED = path.join(ROOT, "data/pedigree");
const raceHorses = JSON.parse(readFileSync(path.join(PED, "all/race_horses.json"), "utf8")) as Record<string, Record<string, string>>;
const horses = JSON.parse(readFileSync(path.join(PED, "all/horses.json"), "utf8")) as Record<string, CompactPedigree | null>;
const stallions = JSON.parse(readFileSync(path.join(PED, "stallions.json"), "utf8")) as Record<string, StallionRaw>;
const SHRINK = 30;
type C = [number, number, number, number];
const st = (c: C) => c[0] + c[1] + c[2] + c[3];
const t3 = (c: C) => c[0] + c[1] + c[2];
const sub = (a: C, b: C | undefined): C => (b ? a.map((v, i) => Math.max(0, v - b[i])) as C : a);
const sum = (cs: (C | undefined)[]): C => cs.reduce<C>((a, c) => (c ? a.map((v, i) => v + c[i]) as C : a), [0, 0, 0, 0]);

const cache = new Map<string, StallionRecord | null>();
function rec(id: string, role: "sire" | "damsire", through: number): StallionRecord | null {
  const key = `${id}|${role}|${through}`;
  if (!cache.has(key)) cache.set(key, stallions[id] ? stallionRecord(stallions[id], role, through) : null);
  return cache.get(key)!;
}

function values(r: StallionRecord, older: StallionRecord | null, surface: string, distance: number | null, going: string | null) {
  const g = r.going[surface] ?? {};
  const all = sum(Object.values(g) as C[]);
  if (!st(all)) return null;
  const rate = t3(all) / st(all);
  const bands = r.distance[surface] ?? [];
  let band = distance === null ? undefined : bands.find((b) => b.upper !== null && distance <= b.upper);
  if (!band && distance !== null) band = bands.find((b) => b.upper === null);
  const bc = band?.counts as C | undefined;
  const gc = going ? (g as Record<string, C>)[going] : undefined;
  const recentAll = sub(all, older ? sum(Object.values(older.going[surface] ?? {}) as C[]) : undefined);
  return {
    pb_band_top3: bc ? (t3(bc) + SHRINK * rate) / (st(bc) + SHRINK) : null,
    pb_going_top3: gc ? (t3(gc) + SHRINK * rate) / (st(gc) + SHRINK) : null,
    pb_win: all[0] / st(all),
    pb_recent_top3: st(recentAll) ? (t3(recentAll) + SHRINK * rate) / (st(recentAll) + SHRINK) : null,
  };
}

const KEYS = ["pb_band_top3", "pb_going_top3", "pb_win", "pb_recent_top3"] as const;
const lines = ["race_id,umaban," + [...KEYS, "pb_f_band_top3"].join(",")];
for (const year of readdirSync(SEISEKI).sort()) {
  if (year < "2019") continue;
  for (const file of readdirSync(path.join(SEISEKI, year)).filter((f) => f.endsWith(".json")).sort()) {
    const day = JSON.parse(readFileSync(path.join(SEISEKI, year, file), "utf8"));
    for (const race of day.races) {
      if (!race.race_id || !["芝", "ダート"].includes(race.surface)) continue;
      const map = raceHorses[race.race_id];
      if (!map) continue;
      const through = Number(race.date.slice(0, 4)) - 1;
      const going = ["良", "稍重", "重", "不良"].includes(race.going) ? race.going : null;
      for (const [umaban, horseId] of Object.entries(map)) {
        const ped = horses[horseId];
        const per = PEDIGREE_SOURCES.map((s) => {
          const id = ped?.[s.position as "F" | "MF" | "MMF"]?.[0];
          const r = id ? rec(id, s.role as "sire" | "damsire", through) : null;
          return r ? values(r, rec(id!, s.role as "sire" | "damsire", through - 3), race.surface, race.distance ?? null, going) : null;
        });
        const out = KEYS.map((k) => {
          let s = 0, w = 0;
          per.forEach((v, i) => { const x = v?.[k]; if (x !== null && x !== undefined) { s += PEDIGREE_SOURCES[i].weight * x; w += PEDIGREE_SOURCES[i].weight; } });
          return w ? s / w : "";
        });
        lines.push([race.race_id, umaban, ...out, per[0]?.pb_band_top3 ?? ""].join(","));
      }
    }
  }
}
writeFileSync(path.join(PED, "all/runner_features_extra.csv"), lines.join("\n") + "\n");
console.log("rows", lines.length - 1);
