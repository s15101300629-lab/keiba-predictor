// 外国馬の扱い（2026-11-26）の一致テストの TS 側（parity_foreign.py の結果と比べる）。
//   node --experimental-strip-types scripts/ml/parity_foreign.ts
import { readFileSync } from "node:fs";
import type { Store } from "../../app/lib/ml/features.ts";
import type { Model } from "../../app/lib/ml/model.ts";
import { runParity, type Row } from "./parity_core.ts";

const root = new URL("../../data/", import.meta.url);
const read = (p: string) => JSON.parse(readFileSync(new URL(p, root), "utf8"));
const store = read("ml/store.json") as Store;
const model = read("ml/model.json") as Model;
store.jockeyCareer = read("overseas/jockey_career.json").jockeys;
try { store.ratingInit = read("overseas/rating_init.json").coef; } catch { store.ratingInit = null; }
const rows = read("ml/parity_foreign.json") as (Row & { extra: Store["horses"][string] | null })[];
// レーティングは Python の行の値（そのレースの前の点数。外国馬の初めの値も入っている）で比べる（parity.ts と同じ）
const r = runParity(store, model, rows.map((x) => ({ ...x, extra: x.extra ?? undefined })));
console.log(`rows ${r.rows}（外国馬 ${rows.filter((x) => x.foreign).length}）, feature mismatches ${r.failures}, max feature diff ${r.worstFeature}, max prediction diff ${r.worstPred}`);
if (r.failures || r.worstPred > 1e-9) process.exit(1);
