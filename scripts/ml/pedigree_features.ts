// data/pedigree の重賞データから、2段目モデル用の血統特徴量を作る（レース前年までの産駒成績を使用）。
//   npx tsx scripts/ml/pedigree_features.ts   -> data/pedigree/graded_features.csv
import { readFileSync, writeFileSync } from "node:fs";
import path from "node:path";
import {
  PEDIGREE_SOURCES, pedigreeAptitudes, stackFeatures, stallionRecord, type StallionRaw,
} from "@/app/lib/pedigree";
import type { GradedRace, HorsePedigree } from "./collect_pedigree";

const OUT = path.resolve(__dirname, "../../data/pedigree");
const races = JSON.parse(readFileSync(path.join(OUT, "graded_races.json"), "utf8")) as Record<string, GradedRace>;
const horses = JSON.parse(readFileSync(path.join(OUT, "horses.json"), "utf8")) as Record<string, HorsePedigree | null>;
const stallions = JSON.parse(readFileSync(path.join(OUT, "stallions.json"), "utf8")) as Record<string, StallionRaw>;

const lines = ["race_id,date,umaban,horse_id,finish,label,n_sources,ped_rate,ped_dev,dist_fit,heavy_fit,distance_type,heavy_label"];
for (const race of Object.values(races).sort((a, b) => a.raceId.localeCompare(b.raceId))) {
  const throughYear = Number(race.date.slice(0, 4)) - 1;
  const going = race.going?.replace(/^\(.*?\)/, "") ?? null;
  const isHeavy = going === "重" || going === "不良";
  const records = race.runners.map((h) => {
    const ped = h.horseId ? horses[h.horseId] : null;
    return PEDIGREE_SOURCES.map((s) => {
      const id = ped?.ancestors.find((a) => a.position === s.position)?.horseId;
      return id && stallions[id] ? stallionRecord(stallions[id], s.role, throughYear) : null;
    });
  });
  const apts = pedigreeAptitudes(records, race.surface, race.distance, going);
  race.runners.forEach((h, i) => {
    const f = stackFeatures(0.5, apts[i], isHeavy);
    lines.push([
      race.raceId, race.date, h.umaban, h.horseId ?? "", h.finish ?? "", h.finish !== null && h.finish <= 3 ? 1 : 0,
      apts[i].sources.length, apts[i].conditionRate ?? "", f.ped_dev, f.dist_fit, f.heavy_fit,
      apts[i].distanceType ?? "", apts[i].heavyLabel ?? "",
    ].join(","));
  });
}
writeFileSync(path.join(OUT, "graded_features.csv"), lines.join("\n") + "\n");
console.log(`wrote ${lines.length - 1} rows`);
