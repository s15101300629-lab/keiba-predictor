"use client";
import { useMemo, useState } from "react";

export type Pick = { umaban: number; horse: string; p?: number; finish: number | null; pop: number | null; odds?: number | null; ai_rank?: number };
export type RaceDetail = {
  race_id: string; venue: string; race_no: number; race_name: string; course: string; going: string | null;
  ai_top3: Pick[]; result_top3: Pick[]; ai_hits: number; pop_hits: number;
};
export type Drift = { feature: string; flags: string[]; missing_day: number; missing_train: number; mean_z: number | null };
export type Day = { date: string; detail: RaceDetail[]; drift?: Drift[] | null };
type BetTicket = { key: string; marks: string[]; prob: number; odds: number | null; hit: boolean; pay: number };
export type BetRace = {
  race_id: string; venue: string; race_no: number; race_name: string; shinba: boolean; payouts_ok: boolean; odds_ok: boolean;
  marks: { n: number; horse: string; mark: string; finish: number | null; pop: number | null; win_pay: number; place_pay: number }[];
  bets: Record<"prob" | "ev", Record<string, BetTicket[]>>; stars: { kind: string; key: string; hit: boolean; pay: number }[];
};
export type BetDay = { date: string; races: BetRace[] };

const BET_TYPES = ["ワイド", "馬連", "馬単", "3連複", "3連単"];
const STAR_KINDS = ["複勝☆", "ワイド◎-☆", "ワイド○-☆"];
const MARKS = ["◎", "○", "▲", "☆", "△", "消"];
const yen = (x: number) => x.toLocaleString("ja-JP") + "円";

/** 券種ごと：買ったレース数・点数・当たったレース数・払戻. */
function betStats(races: BetRace[], mode: "prob" | "ev") {
  const rows = BET_TYPES.map((t) => {
    let racesN = 0, points = 0, hitRaces = 0, ret = 0;
    for (const r of races) {
      const ts = r.bets[mode][t] ?? [];
      if (!ts.length) continue;
      racesN++; points += ts.length; ret += ts.reduce((s, x) => s + x.pay, 0);
      if (ts.some((x) => x.hit)) hitRaces++;
    }
    return { label: t, racesN, points, hitRaces, ret };
  });
  const stars = STAR_KINDS.map((k) => {
    const ts = races.flatMap((r) => r.stars.filter((x) => x.kind === k));
    return { type: k.startsWith("複勝") ? "複勝" : "ワイド", label: k.replace("☆", " ☆").replace("複勝 ☆", "☆の複勝"), racesN: ts.length, points: ts.length, hitRaces: ts.filter((x) => x.hit).length,
      ret: ts.reduce((s, x) => s + x.pay, 0) };
  });
  return { rows, stars };
}

function markStats(races: BetRace[]) {
  return MARKS.map((m) => {
    const hs = races.flatMap((r) => r.marks.filter((h) => h.mark === m));
    const n = hs.length;
    return { m, n, win: n ? hs.filter((h) => h.finish === 1).length / n : NaN, in3: n ? hs.filter((h) => h.finish != null && h.finish <= 3).length / n : NaN,
      winRoi: n ? hs.reduce((s, h) => s + h.win_pay, 0) / (100 * n) : NaN, placeRoi: n ? hs.reduce((s, h) => s + h.place_pay, 0) / (100 * n) : NaN };
  });
}

function BetSummary({ races, label }: { races: BetRace[]; label: string }) {
  const [mode, setMode] = useState<"prob" | "ev">("prob");
  const [skip, setSkip] = useState(false);
  const rs = races.filter((r) => r.payouts_ok && r.odds_ok && !(skip && r.shinba));
  const { rows, stars } = betStats(rs, mode);
  const total = rows.reduce((a, r) => ({ points: a.points + r.points, ret: a.ret + r.ret }), { points: 0, ret: 0 });
  const line = (r: { label: string; racesN: number; points: number; hitRaces: number; ret: number }, unit: string) => <tr key={r.label}>
    <th>{r.label}</th><td>{r.points}点</td><td>{r.hitRaces}/{r.racesN}{unit}</td><td>{r.racesN ? pct(r.hitRaces / r.racesN) : "–"}</td><td>{yen(r.ret)}</td>
    <td className={r.points && r.ret >= r.points * 100 ? "strong" : ""}>{r.points ? pct(r.ret / (r.points * 100)) : "–"}</td></tr>;
  return <div className="check-summary bet-summary">
    <p className="check-summary-label">{label} 買い方の成績<span>{rs.length}レース</span></p>
    <div className="bet-summary-toolbar">
      <div className="sort-toggle" role="group" aria-label="買い目の選び方">
        <button type="button" className={mode === "prob" ? "active" : ""} onClick={() => setMode("prob")}>確率順</button>
        <button type="button" className={mode === "ev" ? "active" : ""} onClick={() => setMode("ev")} disabled={!races.some((r) => r.payouts_ok && r.odds_ok && r.bets.ev && Object.values(r.bets.ev).some((tickets) => tickets.length > 0))}>期待値順</button>
      </div>
      <div className="sort-toggle" role="group" aria-label="対象レース">
        <button type="button" className={skip ? "active" : ""} aria-pressed={skip} onClick={() => setSkip(!skip)}>新馬を除く</button>
      </div>
    </div>
    <section className="bet-result-section">
      <h4 className="bet-result-heading">印から作る買い目</h4>
      <div className="bet-table-wrap">
        <table className="check-table bet-stats primary-bet-stats">
          <thead><tr><th>券種</th><th>点数</th><th>的中レース数</th><th>的中率</th><th>金額</th><th>回収率</th></tr></thead>
          <tbody>{rows.map((r) => line(r, ""))}
            <tr className="total"><th>合計</th><td>{total.points}点</td><td>–</td><td>–</td><td>{yen(total.ret)}</td><td>{total.points ? pct(total.ret / (total.points * 100)) : "–"}</td></tr></tbody>
        </table>
      </div>
    </section>
    <section className="bet-result-section">
      <h4 className="bet-result-heading">☆の買い目</h4>
      <div className="bet-table-wrap compact-bet-table-wrap">
        <table className="check-table bet-stats star-bet-stats">
          <thead><tr><th>券種</th><th>購入点数</th><th>的中点数</th><th>的中率</th><th>回収率</th></tr></thead>
          <tbody>{stars.map((r) => <tr key={r.label}>
            <th>{r.label}</th><td>{r.points}点</td><td>{r.hitRaces}点</td>
            <td>{r.points ? pct(r.hitRaces / r.points) : "–"}</td>
            <td className={r.points && r.ret >= r.points * 100 ? "strong" : ""}>{r.points ? pct(r.ret / (r.points * 100)) : "–"}</td>
          </tr>)}</tbody>
        </table>
      </div>
    </section>
    <section className="bet-result-section">
      <h4 className="bet-result-heading">AI印別の成績</h4>
      <div className="bet-table-wrap compact-bet-table-wrap">
        <table className="check-table bet-stats mark-bet-stats">
          <thead><tr><th>AI印</th><th>頭数</th><th>勝率</th><th>3着内率</th><th>単勝回収率</th><th>複勝回収率</th></tr></thead>
          <tbody>{markStats(rs).map((r) => <tr key={r.m}><th>{r.m}</th><td>{r.n}頭</td>
            <td>{pct(r.win)}</td><td>{pct(r.in3)}</td><td>{pct(r.winRoi)}</td><td>{pct(r.placeRoi)}</td></tr>)}</tbody>
        </table>
      </div>
    </section>
    <p className="check-note">1点100円・確定の払戻。印と買い目は画面と同じ計算だが、オッズは確定オッズで決めた（レース前に見た印・買い目と違うことがある）。
      期待値順もオッズは確定オッズ。的中したレース＝その券種の買い目のどれかが当たったレースの割合</p>
  </div>;
}

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

type Row = { umaban: number; horse: string; mark: string | null; aiRank: number | null; finish: number | null; pop: number | null; label: "的中" | "外れ" | "見逃し" };

/** AI 予想の上位3頭と、実際の1〜3着をまとめた行（着順の順。AI が3位以内に入れたが4着以下の馬は後ろ）. */
function rowsOf(race: RaceDetail, betRace?: BetRace): Row[] {
  const markByHorse = new Map((betRace?.marks ?? []).map((h) => [h.n, h.mark]));
  const rows = new Map<number, Row>();
  race.ai_top3.forEach((h, i) => {
    const hit = h.finish !== null && h.finish <= 3;
    rows.set(h.umaban, { umaban: h.umaban, horse: h.horse, mark: markByHorse.get(h.umaban) ?? null, aiRank: i + 1, finish: h.finish, pop: h.pop, label: hit ? "的中" : "外れ" });
  });
  race.result_top3.forEach((h, i) => {
    if (!rows.has(h.umaban)) rows.set(h.umaban, { umaban: h.umaban, horse: h.horse, mark: markByHorse.get(h.umaban) ?? null, aiRank: h.ai_rank ?? null, finish: h.finish ?? i + 1, pop: h.pop, label: "見逃し" });
  });
  return [...rows.values()].sort((a, b) => (a.finish ?? 99) - (b.finish ?? 99));
}

function RaceCard({ race, bets }: { race: RaceDetail; bets?: BetRace }) {
  return <article className="check-race">
    <header>
      <b>{race.race_no}R</b> {race.race_name}
      <small>{race.course}{race.going ? `・${race.going}` : ""}</small>
      <span className={`check-score s${race.ai_hits}`}>{race.ai_hits}/3 的中</span>
    </header>
    <table className="check-table">
      <thead><tr><th>馬番・馬名</th><th>AI印</th><th>AI順位</th><th>実着順</th><th>判定</th></tr></thead>
      <tbody>{rowsOf(race, bets).map((r) => <tr key={r.umaban} className={"row-" + r.label}>
        <td><b className="check-num">{r.umaban}</b>{r.horse}{r.pop ? <small>{r.pop}人気</small> : null}</td>
        <td className="strong">{r.mark ?? "–"}</td>
        <td className={r.aiRank !== null && r.aiRank <= 3 ? "strong" : ""}>{r.aiRank !== null ? `${r.aiRank}位` : "–"}</td>
        <td className={r.finish !== null && r.finish <= 3 ? "strong" : ""}>{r.finish !== null ? `${r.finish}着` : "中止"}</td>
        <td><span className={"check-label " + r.label}>{r.label}</span></td>
      </tr>)}</tbody>
    </table>
  </article>;
}

export default function ResultsView({ days, betDays = [], view = "daily" }: { days: Day[]; betDays?: BetDay[]; view?: "daily" | "cumulative" }) {
  const [date, setDate] = useState(days[0]?.date ?? "");
  const day = days.find((d) => d.date === date) ?? days[0];
  const venues = useMemo(() => [...new Set((day?.detail ?? []).map((r) => r.venue))]
    .sort((a, b) => VENUE_ORDER.indexOf(a) - VENUE_ORDER.indexOf(b)), [day]);
  const [venue, setVenue] = useState("");
  const v = venues.includes(venue) ? venue : venues[0] ?? "";
  const races = (day?.detail ?? []).filter((r) => r.venue === v).sort((a, b) => a.race_no - b.race_no);
  const all = days.flatMap((d) => d.detail);
  const betAll = betDays.flatMap((d) => d.races);
  const betDay = betDays.find((d) => d.date === day?.date);
  const betRaces = (betDay?.races ?? []).filter((r) => r.venue === v);
  const betById = new Map((betDay?.races ?? []).map((r) => [r.race_id, r]));

  return <main className="shell check-page">
    <h1 className="check-title">{view === "cumulative" ? "累計成績" : "日別成績"}</h1>
    <p className="check-lead">AI上位3頭の着順と、実際の3着以内の馬を比較します。「的中」はAI上位予想が3着以内、「見逃し」はAI予想外で実際に3着以内に入った馬です。</p>
    {days.length === 0 ? <p className="notice">まだデータがない。</p> : view === "cumulative" ? <>
      <p className="check-lead">全開催日の成績を集計。</p>
      <Summary races={all} label="累計" />
      {betAll.length > 0 && <BetSummary races={betAll} label="累計" />}
    </> : <>
      <div className="date-tabs" role="tablist" aria-label="開催日">
        {days.map((d) => <button key={d.date} type="button" role="tab" aria-selected={d.date === day.date}
          className={d.date === day.date ? "date-tab active" : "date-tab"} onClick={() => setDate(d.date)}>{fmtDate(d.date)}</button>)}
      </div>
      <div className="venue-tabs" role="tablist" aria-label="競馬場">
        {venues.map((x) => <button key={x} type="button" role="tab" aria-selected={x === v}
          className={x === v ? "venue-tab active" : "venue-tab"} onClick={() => setVenue(x)}>{x}</button>)}
      </div>
      {day.drift && day.drift.length > 0 && <details className="check-drift">
        <summary>入力の見張り：学習データとずれが大きい項目 {day.drift.length}</summary>
        <p className="check-note">その日の全レースの入力を、学習データの直近1年と比べた（欠けの割合・平均）。不具合で入力がおかしくなっていないかの目安</p>
        <ul>{day.drift.slice(0, 20).map((x) => <li key={x.feature}><b>{x.feature}</b>（{x.flags.join("・")}）欠け {pct(x.missing_day)}（学習 {pct(x.missing_train)}）
          {x.mean_z != null ? `・平均のずれ ${x.mean_z}` : ""}</li>)}</ul>
      </details>}
      <Summary races={races} label={`${fmtDate(day.date)} ${v}`} />
      {betRaces.length > 0 && <BetSummary races={betRaces} label={`${fmtDate(day.date)} ${v}`} />}
      <div className="check-races">{races.map((r) => <RaceCard key={r.race_id} race={r} bets={betById.get(r.race_id)} />)}</div>
    </>}
  </main>;
}
