// AI の印（◎○▲☆△消）。決め方は scripts/ml/backtest_marks.py（▲は決め方 B）と同じ。上から順に付け、1頭に印は1つ。▲☆△消は複数頭に付いてよい。
// 1. ◎：AI の勝率1位。「割高」（目安の倍率 0.9 以下）なら次の馬
// 2. ○：残りで AIスコア（3着以内確率）1位
// 3. ▲：残りのうち AI の勝率が上位4頭以内の馬で、単勝の回収見込み（AI と市場を合わせた勝率 × 単勝オッズ）が一番高い1頭
// 4. ☆：残りで、6番人気以下かつ「人気薄専用の2段目の3着以内確率 ÷ 補正した市場」が 1.2 以上（app/lib/longshot.ts。2段目が出ないときは☆なし）
// 5. △：残りで、AIスコアが6位以内
// 6. 消：残りで、3番人気以内かつ「割高」
// オッズがないときは目安が出せないので、◎は勝率1位、▲は残りで勝率1位、☆・消はなし。

export type AiMark = "◎" | "○" | "▲" | "☆" | "△" | "消";
export const AI_MARK_ORDER: AiMark[] = ["◎", "○", "▲", "☆", "△", "消"];
export type MarkInput = {
  id: string;
  win: number; // AI の勝率（そろえる前のモデルの出力）
  top3: number; // AIスコア（3着以内確率）
  ratio: number | null; // 目安の倍率（オッズがないと null）
  odds: number | null; // 単勝オッズ
  popularity: number | null;
  longshot: number | null; // 人気薄専用の2段目の倍率（6番人気以下だけ。オッズがない・2段目がないと null）
};
export type MarkResult = { mark: AiMark; reason: string };

const STAR = 1.2; // data/ml/model_longshot.json の threshold と同じ
const STAR_MIN_POP = 6;
const EXPENSIVE = 0.9;

export function assignAiMarks(hs: MarkInput[], winCoef: { ai: number; market: number } | null): Map<string, MarkResult> {
  const out = new Map<string, MarkResult>();
  if (!hs.length) return out;
  const hasOdds = hs.every((h) => h.odds != null && h.ratio != null) && !!winCoef;
  const byWin = [...hs].sort((a, b) => b.win - a.win);
  const byTop3 = [...hs].sort((a, b) => b.top3 - a.top3);
  const winRank = new Map(byWin.map((h, i) => [h.id, i]));
  const top3Rank = new Map(byTop3.map((h, i) => [h.id, i]));
  const free = (h: MarkInput) => !out.has(h.id);
  const expensive = (h: MarkInput) => h.ratio != null && h.ratio <= EXPENSIVE;

  const honmei = byWin.find((h) => !hasOdds || !expensive(h));
  if (honmei) out.set(honmei.id, { mark: "◎", reason: "AIの勝率1位" + (honmei !== byWin[0] ? "（勝率1位の馬は割高のため次点）" : "") });
  const taikou = byTop3.find(free);
  if (taikou) out.set(taikou.id, { mark: "○", reason: `AIスコア${top3Rank.get(taikou.id)! + 1}位` });

  if (hasOdds) {
    // 単勝の回収見込み ∝ exp(ai·log AI勝率 + market·log 市場勝率) × 単勝オッズ（レースの中での比べ方は、合計をそろえなくても同じ）
    const winSum = hs.reduce((s, h) => s + h.win, 0);
    const invSum = hs.reduce((s, h) => s + 1 / h.odds!, 0);
    const score = (h: MarkInput) => winCoef!.ai * Math.log(h.win / winSum) + winCoef!.market * Math.log(1 / h.odds! / invSum) + Math.log(h.odds!);
    const cand = hs.filter((h) => free(h) && winRank.get(h.id)! < 4);
    if (cand.length) {
      const best = cand.reduce((a, b) => (score(b) > score(a) ? b : a));
      out.set(best.id, { mark: "▲", reason: `AIの勝率${winRank.get(best.id)! + 1}位のうち、単勝の妙味が一番` });
    }
    for (const h of hs) {
      if (free(h) && h.popularity != null && h.popularity >= STAR_MIN_POP && h.longshot != null && h.longshot >= STAR) {
        out.set(h.id, { mark: "☆", reason: `${h.popularity}番人気だが、人気薄AIでオッズの割に来やすい（×${h.longshot.toFixed(2)}）` });
      }
    }
  } else {
    const san = byWin.find(free);
    if (san) out.set(san.id, { mark: "▲", reason: `AIの勝率${winRank.get(san.id)! + 1}位` });
  }
  for (const h of byTop3) if (free(h) && top3Rank.get(h.id)! < 6) out.set(h.id, { mark: "△", reason: `AIスコア${top3Rank.get(h.id)! + 1}位` });
  if (hasOdds) {
    for (const h of hs) {
      if (free(h) && h.popularity != null && h.popularity <= 3 && expensive(h)) {
        out.set(h.id, { mark: "消", reason: `${h.popularity}番人気だが割高（目安 ×${h.ratio!.toFixed(2)}）` });
      }
    }
  }
  return out;
}
