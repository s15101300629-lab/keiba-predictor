// 印から作る買い目の計算の一致テスト（scripts/ml/parity_bets.py が作った例と、app/lib/markBets.ts の計算を比べる）
//   npx tsx scripts/ml/parity_bets.ts
import { readFileSync } from "node:fs";
import path from "node:path";
import { BENTER_L2, BENTER_L3, markBetCandidates, strengthsFromTop3 } from "@/app/lib/markBets";

const data = JSON.parse(readFileSync(path.resolve(__dirname, "../../data/ml/parity_bets.json"), "utf8")) as {
  lambda: [number, number]; races: { t3: number[]; s: number[]; umaren: Record<string, number>; trio: Record<string, number>; trifecta: Record<string, number> }[] };
if (data.lambda[0] !== BENTER_L2 || data.lambda[1] !== BENTER_L3) { console.log("λ が違う", data.lambda, BENTER_L2, BENTER_L3); process.exit(1); }
let worstS = 0, worstP = 0;
for (const r of data.races) {
  const s = strengthsFromTop3(r.t3);
  s.forEach((x, i) => { worstS = Math.max(worstS, Math.abs(x - r.s[i])); });
  // 全頭に印を付けて、全部の組を出す（馬番＝添字＋1）
  const hs = r.s.map((_, i) => ({ id: String(i), num: i + 1, mark: "△" as const, aiWin: r.s[i], odds: null }));
  const all = markBetCandidates(hs, r.s)!;
  const cmp = (list: { key: string; prob: number }[], ref: Record<string, number>, ordered: boolean) => {
    for (const c of list) {
      const k = c.key.split("-").map((x) => Number(x) - 1).join("-");
      if (!(k in ref)) { if (ordered) continue; throw new Error("no key " + k); }
      worstP = Math.max(worstP, Math.abs(c.prob - ref[k]));
    }
  };
  cmp(all["馬連"], r.umaren, false);
  cmp(all["3連複"], r.trio, false);
  cmp(all["3連単"].filter((c) => c.nums.every((n) => n <= 8)).map((c) => ({ key: c.nums.join("-"), prob: c.prob })), r.trifecta, true);
}
console.log(`強さの差 最大 ${worstS.toExponential(2)}、組み合わせの確率の差 最大 ${worstP.toExponential(2)}`);
if (worstS > 1e-6 || worstP > 1e-9) process.exit(1);
