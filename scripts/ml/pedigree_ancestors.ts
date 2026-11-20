// 重賞の出走馬について、父・母父・母母父それぞれの道悪倍率と産駒の重・不良の出走数を書き出す（重みの比較用）。
//   npx tsx scripts/ml/pedigree_ancestors.ts  -> data/pedigree/heavy_ancestors.csv
import { readFileSync, writeFileSync } from "node:fs";
import path from "node:path";
import { PEDIGREE_SOURCES, stallionAptitude, stallionRecord, type StallionRaw } from "@/app/lib/pedigree";
import type { GradedRace, HorsePedigree } from "./collect_pedigree";

const OUT = path.resolve(__dirname, "../../data/pedigree");
const races = JSON.parse(readFileSync(path.join(OUT, "graded_races.json"), "utf8")) as Record<string, GradedRace>;
const horses = JSON.parse(readFileSync(path.join(OUT, "horses.json"), "utf8")) as Record<string, HorsePedigree | null>;
const stallions = JSON.parse(readFileSync(path.join(OUT, "stallions.json"), "utf8")) as Record<string, StallionRaw>;

const cols = PEDIGREE_SOURCES.flatMap((s) => [`${s.position}_ratio`, `${s.position}_starts`]);
const lines = ["race_id,umaban," + cols.join(",")];
for (const race of Object.values(races).sort((a, b) => a.raceId.localeCompare(b.raceId))) {
  const throughYear = Number(race.date.slice(0, 4)) - 1; // レース前年までの産駒成績
  for (const h of race.runners) {
    const ped = h.horseId ? horses[h.horseId] : null;
    const values = PEDIGREE_SOURCES.flatMap((s) => {
      const id = ped?.ancestors.find((a) => a.position === s.position)?.horseId;
      const rec = id && stallions[id] ? stallionRecord(stallions[id], s.role, throughYear) : null;
      const apt = rec ? stallionAptitude(rec, race.surface, race.distance) : null;
      return [apt?.heavyRatio ?? "", apt ? apt.heavyStarts : ""];
    });
    lines.push([race.raceId, h.umaban, ...values].join(","));
  }
}
writeFileSync(path.join(OUT, "heavy_ancestors.csv"), lines.join("\n") + "\n");
console.log(`wrote ${lines.length - 1} rows`);
