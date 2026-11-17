"use client";
import Link from "next/link";
import { useCallback, useEffect, useMemo, useState } from "react";
import { useParams } from "next/navigation";

type Conditions = {
  raceName: string | null; date: string | null; venue: string | null;
  raceNumber: number | null; startTime: string | null; surface: string | null;
  distanceMeters: number | null; direction: string | null; courseLayout: string | null;
  courseLetter: string | null; weather: string | null; going: string | null;
};
type Entry = {
  frameNumber: number | null; horseNumber: number | null; sexAge: string | null;
  carriedWeight: number | null; jockey: string | null; trainer: string | null;
  bodyWeight: number | null; popularity: number | null;
};
type PerformanceStats = {
  parsedStarts: number; wins: number; top3: number; averageFinish: number | null;
  gradedStarts: number; gradedWins: number;
  sameVenueStarts: number; sameVenueTop3: number;
  similarDistanceStarts: number; similarDistanceTop3: number;
  sameGoingStarts: number; sameGoingTop3: number;
};
type Horse = {
  horseId: string; name: string; status: string; raceEntry: Entry | null;
  profile: { label: string; value: string }[] | null;
  performances: { values: Record<string, string> }[]; historyReturned: number;
  performanceStats?: PerformanceStats;
};
type ResponseData = {
  ok: boolean; raceId: string; raceConditions: Conditions; horseCount: number;
  completeCount: number; partialCount: number; failedCount: number;
  results: Horse[]; error?: string;
};
type CachedRace = { savedAt: number; data: ResponseData };
type FactorScore = { label: string; score: number; starts: number; rate: number };

const CLIENT_CACHE_TTL_MS = 5 * 60 * 1000;
const cacheKey = (id: string) => "keiba-race-detail-v1:" + id;
const marksCacheKey = (id: string) => "keiba-race-marks-v1:" + id;
const MARK_OPTIONS = ["", "◎", "○", "▲", "△", "☆", "注", "消"];

function readMarks(id: string): Record<string, string> {
  try {
    const raw = window.localStorage.getItem(marksCacheKey(id));
    if (!raw) return {};
    const parsed = JSON.parse(raw) as Record<string, unknown>;
    if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) return {};
    const validEntries: [string, string][] = Object.entries(parsed).filter(
      (entry): entry is [string, string] =>
        typeof entry[1] === "string" && MARK_OPTIONS.includes(entry[1])
    );
    return Object.fromEntries(validEntries);
  } catch { return {}; }
}
function saveMarks(id: string, marks: Record<string, string>) {
  try { window.localStorage.setItem(marksCacheKey(id), JSON.stringify(marks)); }
  catch { /* 保存できない場合も画面操作は継続する */ }
}
function readCachedRace(id: string): CachedRace | null {
  try {
    const raw = window.localStorage.getItem(cacheKey(id));
    if (!raw) return null;
    const parsed = JSON.parse(raw) as CachedRace;
    if (!parsed || typeof parsed.savedAt !== "number" || !parsed.data ||
      parsed.data.raceId !== id || !parsed.data.raceConditions ||
      !Array.isArray(parsed.data.results)) return null;
    return parsed;
  } catch { return null; }
}
function saveCachedRace(id: string, data: ResponseData, savedAt: number) {
  try { window.localStorage.setItem(cacheKey(id), JSON.stringify({ savedAt, data })); }
  catch { /* キャッシュできない場合も画面表示は継続する */ }
}
function display(value: string | number | null | undefined) {
  return value === null || value === undefined || value === "" ? "" : String(value);
}
function formatSavedAt(timestamp: number | null) {
  if (!timestamp) return "保存データなし";
  return "保存データ：" + new Date(timestamp).toLocaleString("ja-JP", {
    month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit",
  });
}

function buildFactorScores(horses: Horse[]): Map<string, FactorScore[]> {
  const factors = [
    { label: "全体成績", get: (s: PerformanceStats) => ({ starts: s.parsedStarts, rate: s.parsedStarts ? s.top3 / s.parsedStarts : 0 }) },
    { label: "重賞実績", get: (s: PerformanceStats) => ({ starts: s.gradedStarts, rate: s.gradedStarts ? s.gradedWins / s.gradedStarts : 0 }) },
    { label: "競馬場適性", get: (s: PerformanceStats) => ({ starts: s.sameVenueStarts, rate: s.sameVenueStarts ? s.sameVenueTop3 / s.sameVenueStarts : 0 }) },
    { label: "距離適性", get: (s: PerformanceStats) => ({ starts: s.similarDistanceStarts, rate: s.similarDistanceStarts ? s.similarDistanceTop3 / s.similarDistanceStarts : 0 }) },
    { label: "馬場適性", get: (s: PerformanceStats) => ({ starts: s.sameGoingStarts, rate: s.sameGoingStarts ? s.sameGoingTop3 / s.sameGoingStarts : 0 }) },
  ];
  const result = new Map<string, FactorScore[]>();
  for (const factor of factors) {
    const values = horses.map((horse) => {
      const stats = horse.performanceStats;
      return { horseId: horse.horseId, ...(stats ? factor.get(stats) : { starts: 0, rate: 0 }) };
    }).filter((item) => item.starts > 0);
    if (!values.length) continue;
    const mean = values.reduce((sum, item) => sum + item.rate, 0) / values.length;
    const variance = values.reduce((sum, item) => sum + Math.pow(item.rate - mean, 2), 0) / values.length;
    const sd = Math.sqrt(variance);
    for (const item of values) {
      const score = sd === 0 ? 50 : Math.max(20, Math.min(80, 50 + 10 * (item.rate - mean) / sd));
      const list = result.get(item.horseId) ?? [];
      list.push({ label: factor.label, score: Math.round(score * 10) / 10, starts: item.starts, rate: item.rate });
      result.set(item.horseId, list);
    }
  }
  return result;
}

function HorseDetailCard({ horse, mark, onMarkChange }: {
  horse: Horse; mark: string; onMarkChange: (horseId: string, mark: string) => void;
}) {
  const [expanded, setExpanded] = useState(false);
  const entry = horse.raceEntry;
  const profile = horse.profile ?? [];
  const profileValue = (label: string) => profile.find((item) => item.label.includes(label))?.value;
  const histories = horse.performances ?? [];
  const facts: [string, string][] = ([
    ["枠番", entry?.frameNumber == null ? "" : String(entry.frameNumber)],
    ["馬体重", entry?.bodyWeight == null ? "" : String(entry.bodyWeight) + "kg"],
    ["人気", entry?.popularity == null ? "" : String(entry.popularity)],
    ["調教師", entry?.trainer ?? ""],
    ["父", profileValue("父") ?? ""], ["母", profileValue("母") ?? ""], ["母父", profileValue("母父") ?? ""],
  ] as [string, string][]).filter(([, value]) => value !== "");
  return <details className="horse-detail" onToggle={(event) => setExpanded(event.currentTarget.open)}>
    <summary className="horse-summary">
      {entry?.horseNumber != null && <span className="horse-number-badge">{entry.horseNumber}番</span>}
      <span className="horse-main">
        <b>{horse.name}</b>
        {[entry?.sexAge, entry?.jockey, entry?.carriedWeight != null ? String(entry.carriedWeight) + "kg" : null].filter(Boolean).length > 0 &&
          <small>{[entry?.sexAge, entry?.jockey, entry?.carriedWeight != null ? String(entry.carriedWeight) + "kg" : null].filter(Boolean).join(" ・ ")}</small>}
      </span>
      <span className="horse-expand">{expanded ? "－" : "＋"}</span>
    </summary>
    <select className="horse-mark-select" aria-label={horse.name + "の印"} value={mark}
      onChange={(event) => onMarkChange(horse.horseId, event.target.value)}>
      {MARK_OPTIONS.map((option) => <option key={option || "none"} value={option}>{option || "印なし"}</option>)}
    </select>
    {expanded && <div className="horse-body">
      {facts.length > 0 && <div className="horse-facts">
        {facts.map(([label, value]) => <div className="horse-fact" key={label}><span>{label}：</span>{value}</div>)}
      </div>}
      {histories.length > 0 && <div className="history-table-wrap">
        <b style={{ fontSize: 12 }}>過去の戦績（{histories.length}走）</b>
        <table className="history-table">
          <thead><tr>{Object.keys(histories[0].values).map((key) => <th key={key}>{key}</th>)}</tr></thead>
          <tbody>{histories.map((history, index) => <tr key={index}>
            {Object.keys(histories[0].values).map((key) => <td key={key}>{history.values[key] ?? ""}</td>)}
          </tr>)}</tbody>
        </table>
      </div>}
    </div>}
  </details>;
}

function PredictionHorseCard({ horse, scores }: { horse: Horse; scores: FactorScore[] }) {
  const entry = horse.raceEntry;
  return <article className="prediction-horse">
    <div className="prediction-horse-head">
      {entry?.horseNumber != null && <span className="horse-number-badge">{entry.horseNumber}番</span>}
      <div className="horse-main"><b>{horse.name}</b>
        <small>{[entry?.jockey, entry?.carriedWeight != null ? String(entry.carriedWeight) + "kg" : null].filter(Boolean).join(" ・ ")}</small>
      </div>
    </div>
    {scores.length > 0 ? <div className="prediction-factors">
      {scores.map((factor) => <div className="deviation-row" key={factor.label}>
        <div className="deviation-label"><span>{factor.label}</span><b>偏差値 {factor.score.toFixed(1)}</b></div>
        <div className="deviation-track" role="img" aria-label={factor.label + "の偏差値 " + factor.score.toFixed(1)}>
          <div className="deviation-fill" style={{ width: Math.max(0, Math.min(100, factor.score / 80 * 100)) + "%" }} />
          <span className="deviation-average" />
        </div>
        <small>対象 {factor.starts}走 ・ 3着以内率 {Math.round(factor.rate * 100)}%</small>
      </div>)}
    </div> : <p className="prediction-no-data">評価に使える戦績データがありません</p>}
  </article>;
}

export default function RaceDetailPage() {
  const params = useParams<{ raceId: string }>();
  const raceId = params.raceId;
  const [data, setData] = useState<ResponseData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [refreshing, setRefreshing] = useState(false);
  const [cachedAt, setCachedAt] = useState<number | null>(null);
  const [cacheChecked, setCacheChecked] = useState(false);
  const [marks, setMarks] = useState<Record<string, string>>({});
  const [activeTab, setActiveTab] = useState<"detail" | "prediction">("detail");

  const load = useCallback(async (forceRefresh: boolean, background: boolean) => {
    try {
      if (forceRefresh || background) setRefreshing(true);
      else setLoading(true);
      setError("");
      const response = await fetch("/api/races/" + raceId + (forceRefresh ? "?refresh=1" : ""), { cache: "no-store" });
      const result = (await response.json()) as ResponseData;
      if (!response.ok || !result.raceConditions || !Array.isArray(result.results)) {
        throw new Error(result.error || "レース情報の取得に失敗しました（HTTP " + response.status + "）");
      }
      const savedAt = Date.now();
      setData(result); setCachedAt(savedAt); saveCachedRace(raceId, result, savedAt);
    } catch (err) {
      const message = err instanceof Error ? err.message : "レース情報の取得に失敗しました。";
      if (background || forceRefresh) setError("最新情報への更新に失敗しました。表示中の保存データを利用しています。 " + message);
      else setError(message);
    } finally { setLoading(false); setRefreshing(false); }
  }, [raceId]);

  useEffect(() => {
    if (!/^\d{12}$/.test(raceId)) {
      setError("レースIDが正しくありません。"); setLoading(false); setCacheChecked(true); return;
    }
    setMarks(readMarks(raceId));
    const cached = readCachedRace(raceId);
    setCacheChecked(true);
    if (cached) {
      setData(cached.data); setCachedAt(cached.savedAt); setLoading(false);
      if (Date.now() - cached.savedAt >= CLIENT_CACHE_TTL_MS) void load(false, true);
    } else void load(false, false);
  }, [raceId, load]);

  const conditions = data?.raceConditions;
  const horses = data?.results ?? [];
  const factorScores = useMemo(() => buildFactorScores(horses), [horses]);
  const handleMarkChange = (horseId: string, mark: string) => {
    setMarks((current) => {
      const next = { ...current };
      if (mark) next[horseId] = mark; else delete next[horseId];
      saveMarks(raceId, next); return next;
    });
  };

  return <main className="race-detail-shell">
    <Link href="/" className="back-link">← 開催レース一覧へ戻る</Link>
    {loading && cacheChecked && !data && <div className="state-panel loading-panel">
      <span className="loading-spinner" aria-hidden="true" /><span>初回データを取得しています…</span>
    </div>}
    {!data && !loading && error && <div className="state-panel error-panel"><b>レース情報を取得できませんでした</b><p>{error}</p></div>}
    {data && conditions && <>
      <header className="detail-hero">
        <p className="eyebrow">RACE DETAIL</p><h1>{conditions.raceName || "レース詳細"}</h1>
        <p className="detail-meta">{display(conditions.date)} ・ {display(conditions.venue)}競馬場 ・ {display(conditions.raceNumber)}R
          <br />{display(conditions.startTime)} 発走</p>
      </header>
      <button type="button" className="refresh-button" onClick={() => void load(true, false)} disabled={refreshing || loading}>
        {refreshing && <span className="loading-spinner" aria-hidden="true" />}
        {refreshing ? "最新情報を取得中…" : "最新情報に更新"}
      </button>
      <p className="detail-meta">{formatSavedAt(cachedAt)}{refreshing && " ・ バックグラウンドで最新情報を確認中"}</p>
      {error && <p className="detail-meta" role="status">{error}</p>}
      <div className="race-tabs" role="tablist" aria-label="レース情報">
        <button type="button" role="tab" aria-selected={activeTab === "detail"} className={"race-tab" + (activeTab === "detail" ? " active" : "")}
          onClick={() => setActiveTab("detail")}>レース詳細</button>
        <button type="button" role="tab" aria-selected={activeTab === "prediction"} className={"race-tab" + (activeTab === "prediction" ? " active" : "")}
          onClick={() => setActiveTab("prediction")}>AI予測結果</button>
      </div>
      {activeTab === "detail" && <div role="tabpanel">
        <section className="detail-section">
          <h2>レース条件</h2><div className="condition-grid">
            {([
              ["コース", [conditions.surface, conditions.distanceMeters ? String(conditions.distanceMeters) + "m" : null].filter(Boolean).join(" ")],
              ["回り・コース", [conditions.direction, conditions.courseLayout, conditions.courseLetter].filter(Boolean).join(" ")],
              ["天候", conditions.weather ?? ""], ["馬場状態", conditions.going ?? ""],
            ] as [string, string][]).filter(([, value]) => value !== "").map(([label, value]) =>
              <div className="condition-item" key={label}><small>{label}</small><b>{value}</b></div>)}
          </div>
        </section>
        <section className="detail-section">
          <h2>出走馬 <span style={{ color: "var(--muted)", fontSize: 12 }}>（{data.horseCount}頭）</span></h2>
          {horses.map((horse) => <HorseDetailCard key={horse.horseId} horse={horse}
            mark={marks[horse.horseId] ?? ""} onMarkChange={handleMarkChange} />)}
        </section>
      </div>}
      {activeTab === "prediction" && <section className="prediction-panel" role="tabpanel">
        <div className="prediction-intro"><h2>各馬の評価</h2>
          <p>偏差値50を出走馬の平均として、各項目の成績率を相対比較しています。バーが長いほど、このレースの出走馬内で高い評価です。</p>
        </div>
        {horses.map((horse) => <PredictionHorseCard key={horse.horseId} horse={horse}
          scores={factorScores.get(horse.horseId) ?? []} />)}
      </section>}
    </>}
  </main>;
}
