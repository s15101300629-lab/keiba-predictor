// 重賞出走馬の netkeiba 戦績から、2段目モデル用の特徴量を作る（レース日より前の戦績だけ）。
//   npx tsx scripts/ml/history_features.ts  -> data/pedigree/history_features.csv
import { readFileSync, writeFileSync } from "node:fs";
import path from "node:path";
import { HISTORY_FEATURES, historyFeatures, type HistoryValues } from "@/app/lib/history";
import type { GradedRace } from "./collect_pedigree";

const OUT = path.resolve(__dirname, "../../data/pedigree");
const races = JSON.parse(readFileSync(path.join(OUT, "graded_races.json"), "utf8")) as Record<string, GradedRace>;
const histories = JSON.parse(readFileSync(path.join(OUT, "histories.json"), "utf8")) as Record<string, HistoryValues[]>;

const lines = ["race_id,date,umaban,horse_id,label," + HISTORY_FEATURES.join(",")];
let missing = 0;
for (const race of Object.values(races).sort((a, b) => a.raceId.localeCompare(b.raceId))) {
  for (const h of race.runners) {
    const history = h.horseId ? histories[h.horseId] : undefined;
    if (!history) { missing++; continue; }
    const f = historyFeatures(history, race.date);
    lines.push([race.raceId, race.date, h.umaban, h.horseId, h.finish !== null && h.finish <= 3 ? 1 : 0,
      ...HISTORY_FEATURES.map((k) => (Number.isNaN(f[k]) ? "" : f[k]))].join(","));
  }
}
writeFileSync(path.join(OUT, "history_features.csv"), lines.join("\n") + "\n");
console.log(`wrote ${lines.length - 1} rows (${missing} runners without history)`);
