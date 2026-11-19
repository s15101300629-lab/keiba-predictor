// Python (scripts/ml) と TypeScript (app/lib/ml) の特徴量・予測値が一致するか確認する。
//   node --experimental-strip-types scripts/ml/parity.ts
import { readFileSync } from "node:fs";
import { addFieldFeatures, addFormDeviation, normJockey, runnerFeatures, type Store } from "../../app/lib/ml/features.ts";
import { predictProbability, type Model } from "../../app/lib/ml/model.ts";

type Row = { race_id: string; date: string; horse: string; jockey: string | null; trainer: string | null; features: Record<string, number | null>; pred: number };
const root = new URL("../../data/ml/", import.meta.url);
const store = JSON.parse(readFileSync(new URL("store.json", root), "utf8")) as Store;
const model = JSON.parse(readFileSync(new URL("model.json", root), "utf8")) as Model;
const rows = JSON.parse(readFileSync(new URL("parity.json", root), "utf8")) as Row[];

const num = (v: number | null) => (v === null ? null : v);
let worstFeature = 0;
let worstPred = 0;
let failures = 0;
const byRace = new Map<string, Row[]>();
for (const row of rows) byRace.set(row.race_id, [...(byRace.get(row.race_id) ?? []), row]);
for (const raceRows of byRace.values()) {
  const f0 = raceRows[0].features;
  const race = {
    date: raceRows[0].date, venue: f0.venue!, surface: f0.surface!, distance: num(f0.distance),
    going: num(f0.going), fieldSize: f0.field_size!, prize1stMan: num(f0.prize_1st),
  };
  const computed = raceRows.map((row) => runnerFeatures(store, race, {
    horse: row.horse, jockeyKey: normJockey(row.jockey) || null, trainerKey: normJockey(row.trainer) || null,
    waku: num(row.features.waku), umaban: num(row.features.umaban), weight: num(row.features.weight),
    age: num(row.features.age), sex: row.features.sex === null ? null : ["牡", "牝", "セ"][row.features.sex],
    bodyWeight: num(row.features.body_weight), bodyWeightDiff: num(row.features.body_weight_diff),
  }));
  addFormDeviation(computed);
  addFieldFeatures(computed);
  raceRows.forEach((row, i) => {
    for (const name of model.features) {
      const expected = row.features[name];
      const actual = computed[i][name];
      const same = expected === null ? Number.isNaN(actual) : Math.abs(actual - expected) <= 1e-9;
      if (!same) { failures++; console.log(`${row.race_id} ${row.horse} ${name}: python=${expected} ts=${actual}`); }
      else if (expected !== null) worstFeature = Math.max(worstFeature, Math.abs(actual - expected));
    }
    worstPred = Math.max(worstPred, Math.abs(predictProbability(model, computed[i]) - row.pred));
  });
}
console.log(`rows ${rows.length}, feature mismatches ${failures}, max feature diff ${worstFeature}, max prediction diff ${worstPred}`);
if (failures || worstPred > 1e-9) process.exit(1);
