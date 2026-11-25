// 一致テストの中身（parity.ts・parity_day.ts で共通）
import { addFieldFeatures, addFormDeviation, addOpponentFeatures, addSimilarCondFeatures, fillForeignPeople, normJockey, runnerFeatures, storeHorseKey, type RunnerInput, type Store } from "../../app/lib/ml/features.ts";
import { predictProbability, type Model } from "../../app/lib/ml/model.ts";
import { treeShap } from "../../app/lib/ml/shap.ts";

export type Row = { race_id: string; date: string; horse: string; foreign?: boolean; extra?: Store["horses"][string]; shap?: number[]; owner?: string | null; breeder?: string | null; turn?: number | null; sband?: number | null; jockey: string | null; trainer: string | null; features: Record<string, number | null>; pred: number };
export type Mismatch = { race_id: string; horse: string; name: string; python: number | null; ts: number };
export type ParityResult = { rows: number; failures: number; worstFeature: number; worstPred: number; worstShap: number; mismatches: Mismatch[] };

const num = (v: number | null) => (v === null ? null : v);
// ratingsFromStore：store のレーティングをそのまま使う（その日が store の最終日より後のとき。外国馬のレーティングの初めの値も確かめる）
export function runParity(store: Store, model: Model, rows: Row[], quiet = false, ratingsFromStore = false): ParityResult {
let worstFeature = 0;
let worstPred = 0;
let worstShap = 0;
let failures = 0;
const mismatches: Mismatch[] = [];
const byRace = new Map<string, Row[]>();
for (const row of rows) byRace.set(row.race_id, [...(byRace.get(row.race_id) ?? []), row]);
for (const raceRows of byRace.values()) {
  const f0 = raceRows[0].features;
  const race = {
    date: raceRows[0].date, venue: f0.venue!, surface: f0.surface!, distance: num(f0.distance),
    going: num(f0.going), fieldSize: f0.field_size!, prize1stMan: num(f0.prize_1st),
    turn: raceRows[0].turn ?? null, sband: raceRows[0].sband ?? null,
    cStraight: num(f0.c_straight), cElev: num(f0.c_elev), cLap: num(f0.c_lap),
    // 馬場の細かい値は、アプリでは JRA の馬場情報から取る。ここでは Python の値（含水率の平均との差は TS で計算して比べる）
    baba: { cushion: f0.baba_cushion ?? null, moist: f0.baba_moist ?? null, course: f0.course_letter ?? null },
  };
  // レーティングは store に最終日の更新後の点数しかない（アプリが予測するのは最終日より後のレース）。
  // 過去のレースでは、Python がそのレース前の点数で計算しているので、その値（行の elom・elos）を入れて比べる
  const ratings: Store["ratings"] = {};
  for (const row of raceRows) {
    const r = row.features;
    const s = [r.elos ?? 1500, r.elos_n ?? 0];
    ratings[row.horse] = [r.elom ?? 1500, r.elom_n ?? 0, ...(race.surface === 0 ? s : [1500, 0]), ...(race.surface === 0 ? [1500, 0] : s)] as [number, number, number, number, number, number];
  }
  const raceStore: Store = ratingsFromStore ? store : { ...store, ratings };
  const keys = raceRows.map((row) => storeHorseKey(raceStore, row.horse));
  const runners: RunnerInput[] = raceRows.map((row, i) => ({
    horse: keys[i], jockeyKey: normJockey(row.jockey) || null, trainerKey: normJockey(row.trainer) || null,
    foreign: !!row.foreign, extraRuns: row.extra,
    waku: num(row.features.waku), umaban: num(row.features.umaban), weight: num(row.features.weight),
    age: num(row.features.age), sex: row.features.sex === null ? null : ["牡", "牝", "セ"][row.features.sex],
    bodyWeight: num(row.features.body_weight), bodyWeightDiff: num(row.features.body_weight_diff),
    trainRank: num(row.features.train_rank), owner: row.owner ?? null, breeder: row.breeder ?? null,
    pedigree: Object.fromEntries(["ped_heavy", "ped_wet", "ped_top3", "ped_dist", "pb_band_top3", "pb_going_top3", "pb_win", "pb_recent_top3", "pb_f_band_top3"]
      .map((k) => [k, num(row.features[k])])),
  }));
  const computed = runners.map((r) => runnerFeatures(raceStore, race, r));
  fillForeignPeople(raceStore, race, computed, runners);
  addFormDeviation(computed);
  addFieldFeatures(computed);
  addOpponentFeatures(raceStore, race, computed, keys);
  addSimilarCondFeatures(raceStore, race, computed, keys);
  raceRows.forEach((row, i) => {
    for (const name of model.features) {
      const expected = row.features[name];
      const actual = computed[i][name];
      const same = expected === null ? Number.isNaN(actual) : Math.abs(actual - expected) <= 1e-9;
      if (same && expected !== null && actual !== expected && process.env.PARITY_DEBUG) console.log(`diff ${row.horse} ${name} ${actual - expected}`);
      if (!same) { failures++; mismatches.push({ race_id: row.race_id, horse: row.horse, name, python: expected, ts: actual }); if (!quiet) console.log(`${row.race_id} ${row.horse} ${name}: python=${expected} ts=${actual}`); }
      else if (expected !== null) worstFeature = Math.max(worstFeature, Math.abs(actual - expected));
    }
    const dp = Math.abs(predictProbability(model, computed[i]) - row.pred);
    if (dp > 1e-9 && process.env.PARITY_DEBUG) {
      const base: Record<string, number> = {};
      for (const [k, v] of Object.entries(row.features)) base[k] = v === null ? NaN : v;
      for (const name of model.features) {
        const swapped = { ...base, [name]: computed[i][name] };
        if (Math.abs(predictProbability(model, swapped) - row.pred) > 1e-9) console.log(`split ${row.horse} ${name} py=${base[name]} ts=${computed[i][name]}`);
      }
    }
    worstPred = Math.max(worstPred, dp);
    if (row.shap) { // SHAP（TreeSHAP）も Python（LightGBM の pred_contrib）と比べる
      const phi = treeShap(model, computed[i]);
      phi.forEach((v, j) => { worstShap = Math.max(worstShap, Math.abs(v - row.shap![j])); });
    }
  });
}
return { rows: rows.length, failures, worstFeature, worstPred, worstShap, mismatches };
}
