// 予測と SHAP の時間を、木の本数ごとに測る（2026-11-25、seed 5本の平均を表示を遅くせずに入れる比較）。
//   npx tsx scripts/ml/bench_predict.ts 1551 7750 1900     # 木の本数（今の1本・5本の平均・案1 など）ごとに、18頭×3モデル分の時間
// レース詳細の API と同じ関数（predictProbability・treeShap）を、data/ml/parity.json の行（本番の入力）で動かす。
// 木の本数は data/ml/model.json の木を先頭から使い、足りなければくり返して増やす（同じ設定の木なので、1本あたりの手間は同じとみなす）。
// 案2（学習率5倍・1本の木を約1/5にした5本）は、合計の木の数が今と同じなので、今の1本と同じ本数で測る
import { readFileSync } from "node:fs";
import path from "node:path";
import { predictProbability, type Model } from "@/app/lib/ml/model";
import { treeShap } from "@/app/lib/ml/shap";

const ML = path.resolve(__dirname, "../../data/ml");
const base = JSON.parse(readFileSync(path.join(ML, "model.json"), "utf8")) as Model;
const rows = (JSON.parse(readFileSync(path.join(ML, "parity.json"), "utf8")) as { features: Record<string, number> }[]).slice(0, 18);
const withTrees = (n: number): Model => ({ ...base, trees: Array.from({ length: n }, (_, k) => base.trees[k % base.trees.length]) });

function bench(n: number, reps = 5) {
  const m = withTrees(n);
  const times = { predict: [] as number[], shap: [] as number[] };
  for (let r = 0; r < reps + 1; r++) {
    let t = performance.now();
    // API は3着以内・連対・勝率の3つを予測する（同じ本数とみなす）
    for (let k = 0; k < 3; k++) for (const row of rows) predictProbability(m, row.features);
    const tp = performance.now() - t;
    t = performance.now();
    for (const row of rows) treeShap(m, row.features); // SHAP は3着以内のモデルだけ
    const ts = performance.now() - t;
    if (r > 0) { times.predict.push(tp); times.shap.push(ts); } // 1回目は温め
  }
  const med = (a: number[]) => [...a].sort((x, y) => x - y)[Math.floor(a.length / 2)];
  return { trees: n, predict_ms: Math.round(med(times.predict)), shap_ms: Math.round(med(times.shap)) };
}

const ns = process.argv.slice(2).map(Number).filter((n) => n > 0);
for (const n of ns.length ? ns : [base.trees.length]) console.log(JSON.stringify(bench(n)));
