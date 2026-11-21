"use client";
import { useMemo, useState } from "react";

export type Pick = { umaban: number; horse: string; p?: number; finish: number | null; pop: number | null; odds?: number | null; ai_rank?: number };
export type RaceDetail = {
  race_id: string; venue: string; race_no: number; race_name: string; course: string; going: string | null;
  ai_top3: Pick[]; result_top3: Pick[]; ai_hits: number; pop_hits: number;
};
export type Day = { date: string; detail: RaceDetail[] };

const VENUE_ORDER = ["札幌", "函館", "福島", "新潟", "東京", "中山", "中京", "京都", "阪神", "小倉"];
const pct = (x: number) => (Number.isNaN(x) ? "–" : `${(x * 100).toFixed(1)}%`);
const fmtDate = (d: string) => {
  const [y, m, dd] = d.split("-").map(Number);
  return `${m}/${dd}（${"日月火水木金土"[new Date(Date.UTC(y, m - 1, dd)).getUTCDay()]}）`;
};

function stats(races: RaceDetail[]) {
  const n = races.length;
  const top1 = races.map((r) => r.ai_top3[0]);
  const withOdds = top1.every((h) => h.odds != null);
  return {
    n,
    ai3: n ? races.reduce((s, r) => s + r.ai_hits, 0) / (3 * n) : NaN,
    pop3: n ? races.reduce((s, r) => s + r.pop_hits, 0) / (3 * n) : NaN,
    win: n ? top1.filter((h) => h.finish === 1).length / n : NaN,
    in3: n ? top1.filter((h) => h.finish !== null && h.finish <= 3).length / n : NaN,
    roi: n && withOdds ? top1.reduce((s, h) => s + (h.finish === 1 ? h.odds! : 0), 0) / n : NaN,
  };
}

function Summary({ races, label }: { races: RaceDetail[]; label: string }) {
  const s = stats(races);
  return <div className="check-summary">
    <p className="check-summary-label">{label}<span>{s.n}レース</span></p>
    <div className="check-summary-grid">
      <div><small>AI 3位以内の馬が<br />3着以内に来た割合</small><b>{pct(s.ai3)}</b><em>人気順なら {pct(s.pop3)}</em></div>
      <div><small>AI 1位の馬が<br />1着になった割合</small><b>{pct(s.win)}</b><em>3着以内 {pct(s.in3)}</em></div>
      <div><small>AI 1位の単勝を<br />買った回収率</small><b>{pct(s.roi)}</b><em>確定オッズ</em></div>
    </div>
  </div>;
}

type Row = { umaban: number; horse: string; aiRank: number | null; finish: number | null; pop: number | null; label: "的中" | "外れ" | "見逃し" };

/** AI 予想の上位3頭と、実際の1〜3着をまとめた行（着順の順。AI が3位以内に入れたが4着以下の馬は後ろ）. */
function rowsOf(race: RaceDetail): Row[] {
  const rows = new Map<number, Row>();
  race.ai_top3.forEach((h, i) => {
    const hit = h.finish !== null && h.finish <= 3;
    rows.set(h.umaban, { umaban: h.umaban, horse: h.horse, aiRank: i + 1, finish: h.finish, pop: h.pop, label: hit ? "的中" : "外れ" });
  });
  race.result_top3.forEach((h, i) => {
    if (!rows.has(h.umaban)) rows.set(h.umaban, { umaban: h.umaban, horse: h.horse, aiRank: h.ai_rank ?? null, finish: h.finish ?? i + 1, pop: h.pop, label: "見逃し" });
  });
  return [...rows.values()].sort((a, b) => (a.finish ?? 99) - (b.finish ?? 99));
}

function RaceCard({ race }: { race: RaceDetail }) {
  return <article className="check-race">
    <header>
      <b>{race.race_no}R</b> {race.race_name}
      <small>{race.course}{race.going ? `・${race.going}` : ""}</small>
      <span className={`check-score s${race.ai_hits}`}>{race.ai_hits}/3 的中</span>
    </header>
    <table className="check-table">
      <thead><tr><th>馬番・馬名</th><th>AI予想</th><th>結果</th><th></th></tr></thead>
      <tbody>{rowsOf(race).map((r) => <tr key={r.umaban} className={"row-" + r.label}>
        <td><b className="check-num">{r.umaban}</b>{r.horse}{r.pop ? <small>{r.pop}人気</small> : null}</td>
        <td className={r.aiRank !== null && r.aiRank <= 3 ? "strong" : ""}>{r.aiRank !== null ? `${r.aiRank}位` : "–"}</td>
        <td className={r.finish !== null && r.finish <= 3 ? "strong" : ""}>{r.finish !== null ? `${r.finish}着` : "中止"}</td>
        <td><span className={"check-label " + r.label}>{r.label}</span></td>
      </tr>)}</tbody>
    </table>
  </article>;
}

export default function ResultsView({ days }: { days: Day[] }) {
  const [date, setDate] = useState(days[0]?.date ?? "");
  const day = days.find((d) => d.date === date) ?? days[0];
  const venues = useMemo(() => [...new Set((day?.detail ?? []).map((r) => r.venue))]
    .sort((a, b) => VENUE_ORDER.indexOf(a) - VENUE_ORDER.indexOf(b)), [day]);
  const [venue, setVenue] = useState("");
  const v = venues.includes(venue) ? venue : venues[0] ?? "";
  const races = (day?.detail ?? []).filter((r) => r.venue === v).sort((a, b) => a.race_no - b.race_no);
  const all = days.flatMap((d) => d.detail);

  return <main className="shell check-page">
    <h1 className="check-title">AI成績</h1>
    <p className="check-lead">各レースで、AI 予想の順位（AIスコアの順）と実際の着順を並べたもの。
      AI が3位以内に予想して3着以内に来た馬は「的中」、来なかった馬は「外れ」、AI が4位以下に予想したのに3着以内に来た馬は「見逃し」。
      予想はレース前の情報だけで、レースの後に出し直したもの。毎日 17:30 ごろに更新</p>
    {days.length === 0 ? <p className="notice">まだデータがない。</p> : <>
      <Summary races={all} label="累計" />
      <div className="date-tabs" role="tablist" aria-label="開催日">
        {days.map((d) => <button key={d.date} type="button" role="tab" aria-selected={d.date === day.date}
          className={d.date === day.date ? "date-tab active" : "date-tab"} onClick={() => setDate(d.date)}>{fmtDate(d.date)}</button>)}
      </div>
      <div className="venue-tabs" role="tablist" aria-label="競馬場">
        {venues.map((x) => <button key={x} type="button" role="tab" aria-selected={x === v}
          className={x === v ? "venue-tab active" : "venue-tab"} onClick={() => setVenue(x)}>{x}</button>)}
      </div>
      <Summary races={races} label={`${fmtDate(day.date)} ${v}`} />
      <div className="check-races">{races.map((r) => <RaceCard key={r.race_id} race={r} />)}</div>
    </>}
  </main>;
}
