// 人気薄専用の2段目（AI印の☆）：Python（train_longshot.py）と TypeScript（app/lib/longshot.ts）の入力・予測が一致するか.
//   npx tsx scripts/ml/parity_longshot.ts
import { readFileSync } from "node:fs";
import { longshotRatios, type LongshotModel } from "../../app/lib/longshot";

type H = { umaban: number; horse: string; x: (number | null)[]; top3: number; win: number; odds: number; extra: number[]; p: number; market: number };
const model = JSON.parse(readFileSync(new URL("../../data/ml/model_longshot.json", import.meta.url), "utf8")) as
  LongshotModel & { parity: { race_id: string; horses: H[] }[] };

let worstExtra = 0, worstP = 0, worstMarket = 0, checked = 0, failures = 0;
for (const race of model.parity) {
  const odds = new Map(race.horses.map((h) => [h.umaban, h.odds]));
  const res = longshotRatios(model, race.horses.map((h) => ({ n: h.umaban, x: h.x, top3: h.top3, win: h.win })), odds);
  for (const h of race.horses) {
    const pop = h.extra[5];
    const r = res.get(h.umaban);
    if (pop < model.min_pop) {
      if (r) { failures++; console.log("人気が違う", race.race_id, h.horse); }
      continue;
    }
    if (!r) { failures++; console.log("結果がない", race.race_id, h.horse); continue; }
    checked++;
    const de = Math.max(...h.extra.map((v, k) => Math.abs(v - r.extra[k])));
    const dp = Math.abs(h.p - r.probability);
    const dm = Math.abs(h.market - r.market);
    worstExtra = Math.max(worstExtra, de); worstP = Math.max(worstP, dp); worstMarket = Math.max(worstMarket, dm);
    if (de > 1e-9 || dp > 1e-9 || dm > 1e-9) {
      failures++;
      console.log("ずれ", race.race_id, h.horse, { de, dp, dm });
    }
  }
}
console.log(JSON.stringify({ races: model.parity.length, checked, worstExtra, worstP, worstMarket, failures }));
if (failures) process.exit(1);
