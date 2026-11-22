// 人気薄専用の2段目モデル（AI印の☆）。scripts/ml/train_longshot.py が data/ml/model_longshot.json に書き出す。
// 6番人気以下の馬について、本体の182項目＋AI・市場の確率・オッズ・人気から3着以内の確率を出し、
// 「その確率 ÷ 市場だけを補正した3着以内確率」が threshold（1.2）以上なら☆の候補にする。
// 検証（scripts/ml/star_eval.py、確定オッズ）：2024〜2026年の複勝回収率 89・86・95%（市場の確率の帯をそろえた基準 74・73・75%）。
import { harville } from "./market";
import { predictRaw, type Model } from "./ml/model";

export type LongshotModel = Model & {
  min_pop: number; threshold: number;
  market_cal: { coef: number; intercept: number };
};
export type LongshotInput = {
  n: number; // 馬番
  x: (number | null)[]; // 本体モデルの入力（model.json の features の順。null は欠損）
  top3: number; // 本体の3着以内確率（合計をそろえる前）
  win: number; // 本体の勝率（そろえる前）
};
export type LongshotResult = { popularity: number; probability: number; market: number; ratio: number; extra: number[] };

const clip = (p: number, e: number) => Math.min(Math.max(p, e), 1 - e);
const logit6 = (p: number) => { const q = clip(p, 1e-6); return Math.log(q / (1 - q)); };
const sigmoid = (z: number) => 1 / (1 + Math.exp(-z));

/** 単勝オッズ（馬番 → オッズ）がある馬だけで計算する。6番人気以下の馬だけ結果を返す（それ以外は返さない）. */
export function longshotRatios(model: LongshotModel, horses: LongshotInput[], winOdds: Map<number, number>): Map<number, LongshotResult> {
  const out = new Map<number, LongshotResult>();
  const hs = horses.filter((h) => (winOdds.get(h.n) ?? 0) > 0 && h.top3 > 0 && h.win > 0);
  if (hs.length < 3) return out;
  // 市場：単勝オッズの逆数を合計1に、3着以内はハービルの式（backtest_value.prepare と同じ）
  const inv = new Map(hs.map((h) => [h.n, 1 / winOdds.get(h.n)!]));
  const invSum = hs.reduce((s, h) => s + inv.get(h.n)!, 0);
  const mktWin = new Map(hs.map((h) => [h.n, inv.get(h.n)! / invSum]));
  const mktTop3 = harville(mktWin).top3;
  const top3Sum = hs.reduce((s, h) => s + h.top3, 0);
  const winSum = hs.reduce((s, h) => s + h.win, 0);
  const target = Math.min(3, hs.length);
  // 人気：単勝オッズの低い順（同じオッズは馬番の順）
  const pop = new Map([...hs].sort((a, b) => winOdds.get(a.n)! - winOdds.get(b.n)! || a.n - b.n).map((h, i) => [h.n, i + 1]));
  for (const h of hs) {
    const popularity = pop.get(h.n)!;
    if (popularity < model.min_pop) continue;
    const aiTop3 = clip((h.top3 * target) / top3Sum, 1e-4);
    const mTop3 = clip(mktTop3.get(h.n)!, 1e-4);
    const extra = [logit6(aiTop3), logit6(mTop3), Math.log(winOdds.get(h.n)!), logit6(h.win / winSum), logit6(mktWin.get(h.n)!), popularity];
    const probability = sigmoid(predictRaw(model, [...h.x.map((v) => (v == null ? NaN : v)), ...extra]));
    const market = sigmoid(model.market_cal.coef * logit6(mTop3) + model.market_cal.intercept);
    out.set(h.n, { popularity, probability, market, ratio: probability / market, extra });
  }
  return out;
}
