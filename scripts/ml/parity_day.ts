// 毎日の答え合わせで、その日の全レース・全項目の一致テスト（2026-11-24、依頼者の案 5）。
//   node --experimental-strip-types scripts/ml/parity_day.ts 2026-11-29   # → data/ml/day_check/parity/2026-11-29.json
// check_jra_day.py が書く data/ml/day_check/inputs/日付.json.gz の、Python で計算した本体の入力（x）と、
// アプリと同じ TS の計算（store.json から）を比べる。その日が store の最終日より後のとき（＝アプリが予測するときと同じ状態）だけ意味がある
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { gunzipSync } from "node:zlib";
import type { Store } from "../../app/lib/ml/features.ts";
import { predictProbability, type Model } from "../../app/lib/ml/model.ts";
import { runParity, type Row } from "./parity_core.ts";

const root = new URL("../../data/ml/", import.meta.url);
const date = process.argv[2];
type H = { n: number; horse: string; top3: number; x: (number | null)[]; meta?: Record<string, string | null> };
const inp = JSON.parse(gunzipSync(readFileSync(new URL(`day_check/inputs/${date}.json.gz`, root))).toString("utf8")) as {
  model_dir?: string; features: string[]; races: { race_id: string; date: string; turn: number | null; sband: number | null; horses: H[] }[] };
const store = JSON.parse(readFileSync(new URL("store.json", root), "utf8")) as Store;
const model = JSON.parse(readFileSync(new URL("model.json", root), "utf8")) as Model;
const out: Record<string, unknown> = { date, storeLastDate: store.lastDate };
if ((inp.model_dir ?? "") !== "" || date <= store.lastDate || inp.features.join() !== model.features.join()) {
  out.skipped = "その日が store の最終日以前か、本番と違うモデルの入力なので比べない";
} else {
  const rows: Row[] = inp.races.flatMap((r) => r.horses.map((h) => ({
    race_id: r.race_id, date: r.date, horse: h.horse, turn: r.turn, sband: r.sband,
    jockey: h.meta?.jockey ?? null, trainer: h.meta?.trainer ?? null, owner: h.meta?.owner ?? null, breeder: h.meta?.breeder ?? null,
    features: Object.fromEntries(inp.features.map((f, k) => [f, h.x[k]])), pred: h.top3,
  })));
  // レーティングは store の最終日の点数（その日は store より後なので、Python の値と同じはず）
  const r = runParity(store, model, rows, true);
  Object.assign(out, { races: inp.races.length, rows: r.rows, failures: r.failures, worstFeature: r.worstFeature, worstPred: r.worstPred,
    byFeature: Object.entries(r.mismatches.reduce((m, x) => ({ ...m, [x.name]: (m[x.name] ?? 0) + 1 }), {} as Record<string, number>)).sort((a, b) => b[1] - a[1]),
    examples: r.mismatches.slice(0, 50) });
  void predictProbability;
}
mkdirSync(new URL("day_check/parity/", root), { recursive: true });
writeFileSync(new URL(`day_check/parity/${date}.json`, root), JSON.stringify(out, null, 1) + "\n");
console.log(JSON.stringify({ ...out, examples: undefined }));
if ((out.failures as number) > 0) process.exitCode = 1;
