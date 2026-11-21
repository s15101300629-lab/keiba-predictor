import { readdirSync, readFileSync } from "node:fs";
import path from "node:path";
import RaceCheck from "./RaceCheck";

// 答え合わせ：data/ml/day_check/日付.json（scripts/ml/check_jra_day.py が毎日作る）を読み、AI 上位3頭と実際の結果を並べる
export const metadata = { title: "答え合わせ | Racehorse Insight Analytics" };

export type Pick = { umaban: number; horse: string; p?: number; finish: number | null; pop: number | null; odds?: number | null; ai_rank?: number };
export type RaceDetail = {
  race_id: string; venue: string; race_no: number; race_name: string; course: string; going: string | null;
  ai_top3: Pick[]; result_top3: Pick[]; ai_hits: number; pop_hits: number;
};
type Day = { date: string; races: number; detail: RaceDetail[] };

function loadDays(): Day[] {
  const dir = path.join(process.cwd(), "data", "ml", "day_check");
  let files: string[] = [];
  try { files = readdirSync(dir).filter((f) => /^\d{4}-\d{2}-\d{2}\.json$/.test(f)); } catch { return []; }
  return files.sort().reverse().map((f) => JSON.parse(readFileSync(path.join(dir, f), "utf8")) as Day);
}

function stats(races: RaceDetail[]) {
  const n = races.length;
  const top1 = races.map((r) => r.ai_top3[0]);
  const winBack = top1.reduce((s, h) => s + (h.finish === 1 && h.odds ? h.odds * 100 : 0), 0);
  const withOdds = top1.filter((h) => h.odds != null).length;
  return {
    n,
    ai3: n ? races.reduce((s, r) => s + r.ai_hits, 0) / (3 * n) : NaN,
    pop3: n ? races.reduce((s, r) => s + r.pop_hits, 0) / (3 * n) : NaN,
    win: n ? top1.filter((h) => h.finish === 1).length / n : NaN,
    in3: n ? top1.filter((h) => h.finish !== null && h.finish <= 3).length / n : NaN,
    roi: withOdds === n && n ? winBack / (100 * n) : NaN,
  };
}

const pct = (x: number) => (Number.isNaN(x) ? "–" : `${(x * 100).toFixed(1)}%`);
const fmtDate = (d: string) => { const [y, m, dd] = d.split("-"); return `${y}/${Number(m)}/${Number(dd)}（${"日月火水木金土"[new Date(Date.UTC(Number(y), Number(m) - 1, Number(dd))).getUTCDay()]}）`; };

function Summary({ races, label }: { races: RaceDetail[]; label: string }) {
  const s = stats(races);
  return <div className="check-summary">
    <p className="check-summary-label">{label}<span>{s.n}レース</span></p>
    <div className="check-summary-grid">
      <div><small>AI上位3頭の3着内率</small><b>{pct(s.ai3)}</b><em>人気順 {pct(s.pop3)}</em></div>
      <div><small>AI 1位の勝率</small><b>{pct(s.win)}</b><em>3着内 {pct(s.in3)}</em></div>
      <div><small>AI 1位の単勝回収率</small><b>{pct(s.roi)}</b><em>確定オッズ</em></div>
    </div>
  </div>;
}

export default function ResultsPage() {
  const days = loadDays();
  const all = days.flatMap((d) => d.detail);
  return <main className="shell check-page">
    <h1 className="check-title">答え合わせ</h1>
    <p className="check-lead">各レースの AI 上位3頭（AIスコア順）と実際の結果。予想はレース前の情報だけで、レースの後にモデルで出し直したもの。毎日 17:30 ごろに更新</p>
    {days.length === 0 ? <p className="notice">まだ答え合わせのデータがない。</p> : <>
      <Summary races={all} label="累計" />
      {days.map((d, i) => <details key={d.date} className="check-day" open={i === 0}>
        <summary>{fmtDate(d.date)}<span>{pct(stats(d.detail).ai3)}</span></summary>
        <Summary races={d.detail} label="この日" />
        <div className="check-races">{d.detail.map((r) => <RaceCheck key={r.race_id} race={r} />)}</div>
      </details>)}
    </>}
  </main>;
}
