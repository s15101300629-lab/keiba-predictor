// Python (scripts/ml) と TypeScript (app/lib/ml) の特徴量・予測値が一致するか確認する。
//   node --experimental-strip-types scripts/ml/parity.ts
import { readFileSync } from "node:fs";
import type { Store } from "../../app/lib/ml/features.ts";
import type { Model } from "../../app/lib/ml/model.ts";
import { runParity, type Row } from "./parity_core.ts";

const root = new URL("../../data/ml/", import.meta.url);
const store = JSON.parse(readFileSync(new URL("store.json", root), "utf8")) as Store;
const model = JSON.parse(readFileSync(new URL("model.json", root), "utf8")) as Model;
const rows = JSON.parse(readFileSync(new URL("parity.json", root), "utf8")) as Row[];
const r = runParity(store, model, rows);
console.log(`rows ${r.rows}, feature mismatches ${r.failures}, max feature diff ${r.worstFeature}, max prediction diff ${r.worstPred}, max shap diff ${r.worstShap}`);
if (r.failures || r.worstPred > 1e-9 || r.worstShap > 1e-6) process.exit(1);
