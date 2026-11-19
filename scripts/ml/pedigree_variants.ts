// 血統の道悪倍率を、使う祖先の範囲を変えて計算する（補正の比較用）。
//   npx tsx scripts/ml/pedigree_variants.ts  -> data/pedigree/heavy_variants.csv
import { readFileSync, writeFileSync } from "node:fs";
import path from "node:path";
import {
  allMaleSources, PEDIGREE_SOURCES, pedigreeAptitudes, pedigreeSource, stallionRecord,
  type PedigreeSource, type StallionRaw,
} from "@/app/lib/pedigree";
import type { GradedRace, HorsePedigree } from "./collect_pedigree";

export const VARIANTS: Record<string, PedigreeSource[]> = {
  v0_current: PEDIGREE_SOURCES, // 父・母父・母母父
  v1_male_gen3: allMaleSources(3), // 3代までの牡馬すべて
  v2_male_gen4: allMaleSources(4), // 4代までの牡馬すべて
  v3_current_plus_mmmf: [...PEDIGREE_SOURCES, pedigreeSource("MMMF")], // 牝系をもう1段
};

const OUT = path.resolve(__dirname, "../../data/pedigree");
const races = JSON.parse(readFileSync(path.join(OUT, "graded_races.json"), "utf8")) as Record<string, GradedRace>;
const horses = JSON.parse(readFileSync(path.join(OUT, "horses.json"), "utf8")) as Record<string, HorsePedigree | null>;
const stallions = JSON.parse(readFileSync(path.join(OUT, "stallions.json"), "utf8")) as Record<string, StallionRaw>;

const names = Object.keys(VARIANTS);
const lines = ["race_id,umaban," + names.join(",")];
for (const race of Object.values(races).sort((a, b) => a.raceId.localeCompare(b.raceId))) {
  const throughYear = Number(race.date.slice(0, 4)) - 1;
  const ratios = names.map((name) => {
    const sources = VARIANTS[name];
    const records = race.runners.map((h) => {
      const ped = h.horseId ? horses[h.horseId] : null;
      return sources.map((s) => {
        const id = ped?.ancestors.find((a) => a.position === s.position)?.horseId;
        return id && stallions[id] ? stallionRecord(stallions[id], s.role, throughYear) : null;
      });
    });
    return pedigreeAptitudes(records, race.surface, race.distance, null, sources).map((a) => a.heavyRatio);
  });
  race.runners.forEach((h, i) => lines.push([race.raceId, h.umaban, ...ratios.map((r) => r[i] ?? "")].join(",")));
}
writeFileSync(path.join(OUT, "heavy_variants.csv"), lines.join("\n") + "\n");
console.log(`wrote ${lines.length - 1} rows`);
