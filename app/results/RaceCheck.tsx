import type { Pick, RaceDetail } from "./page";

// 1レース分：AI 上位3頭（着順で ○ ✕）と、実際の1〜3着（AI 何位か）
const finishText = (f: number | null) => (f === null ? "中止" : `${f}着`);

function AiPick({ h, rank }: { h: Pick; rank: number }) {
  const hit = h.finish !== null && h.finish <= 3;
  return <li className={hit ? "hit" : "miss"}>
    <span className="check-rank">AI{rank}</span>
    <span className="check-num">{h.umaban}</span>
    <span className="check-name">{h.horse}</span>
    <span className="check-meta">{h.pop ? `${h.pop}人気` : ""}</span>
    <span className="check-mark">{hit ? "○" : "✕"} {finishText(h.finish)}</span>
  </li>;
}

export default function RaceCheck({ race }: { race: RaceDetail }) {
  return <article className="check-race">
    <header>
      <b>{race.venue}{race.race_no}R</b> {race.race_name}
      <small>{race.course}{race.going ? `・${race.going}` : ""}</small>
      <span className={`check-score s${race.ai_hits}`}>{race.ai_hits}/3</span>
    </header>
    <ul className="check-ai">{race.ai_top3.map((h, i) => <AiPick key={h.umaban} h={h} rank={i + 1} />)}</ul>
    <p className="check-result">結果　{race.result_top3.map((h, i) => <span key={h.umaban}>
      {h.finish ?? i + 1}着 <b>{h.umaban}</b> {h.horse}<em>（AI{h.ai_rank}位{h.pop ? `・${h.pop}人気` : ""}）</em></span>)}</p>
  </article>;
}
