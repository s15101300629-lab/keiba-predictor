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
  sameVenueSurfaceStarts: number; sameVenueSurfaceTop3: number;
  similarDistanceStarts: number; similarDistanceTop3: number;
  sameGoingStarts: number; sameGoingTop3: number;
  sameJockeyStarts: number; sameJockeyTop3: number;
  sameFrameStarts: number; sameFrameTop3: number;
  positionStarts: number; positionGainTotal: number;
  positionForwardStarts: number; positionForwardTop3: number;
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
type FactorScore = { label: string; score: number; starts: number; rate: number; rank: number; rankedCount: number };
type OverallScore = { score: number | null; rank: number | null; rankedCount: number };
type CategoryRank = { label: string; rank: number | null; total: number; detail: string };

const CLIENT_CACHE_TTL_MS = 5 * 60 * 1000;
const cacheKey = (id: string) => "keiba-race-detail-v3:" + id;
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
    { label: "競馬場・馬場適性", get: (s: PerformanceStats) => ({ starts: s.sameVenueSurfaceStarts, rate: s.sameVenueSurfaceStarts ? s.sameVenueSurfaceTop3 / s.sameVenueSurfaceStarts : 0 }) },
    { label: "距離適性", get: (s: PerformanceStats) => ({ starts: s.similarDistanceStarts, rate: s.similarDistanceStarts ? s.similarDistanceTop3 / s.similarDistanceStarts : 0 }) },
    { label: "馬場適性", get: (s: PerformanceStats) => ({ starts: s.sameGoingStarts, rate: s.sameGoingStarts ? s.sameGoingTop3 / s.sameGoingStarts : 0 }) },
    { label: "騎手相性", get: (s: PerformanceStats) => ({ starts: s.sameJockeyStarts, rate: s.sameJockeyStarts ? s.sameJockeyTop3 / s.sameJockeyStarts : 0 }) },
    { label: "枠順実績", get: (s: PerformanceStats) => ({ starts: s.sameFrameStarts, rate: s.sameFrameStarts ? s.sameFrameTop3 / s.sameFrameStarts : 0 }) },
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
    const scored = values.map((item) => ({
      ...item,
      score: sd === 0 ? 50 : Math.max(20, Math.min(80, 50 + 10 * (item.rate - mean) / sd)),
    })).sort((a, b) => b.score - a.score);
    let currentRank = 0;
    let previousScore: number | null = null;
    scored.forEach((item, index) => {
      if (previousScore === null || Math.abs(previousScore - item.score) >= 0.05) currentRank = index + 1;
      previousScore = item.score;
      const list = result.get(item.horseId) ?? [];
      list.push({
        label: factor.label,
        score: Math.round(item.score * 10) / 10,
        starts: item.starts,
        rate: item.rate,
        rank: currentRank,
        rankedCount: scored.length,
      });
      result.set(item.horseId, list);
    });
  }
  return result;
}

function buildOverallScores(horses: Horse[], factorScores: Map<string, FactorScore[]>): Map<string, OverallScore> {
  const raw = horses.map((horse) => {
    const values = factorScores.get(horse.horseId) ?? [];
    return { horseId: horse.horseId, score: values.length ? values.reduce((sum, item) => sum + item.score, 0) / values.length : null };
  });
  const ranked = raw.filter((item): item is { horseId: string; score: number } => item.score !== null)
    .sort((a, b) => b.score - a.score);
  const result = new Map<string, OverallScore>();
  let currentRank = 0;
  let previousScore: number | null = null;
  ranked.forEach((item, index) => {
    if (previousScore === null || Math.abs(previousScore - item.score) >= 0.05) currentRank = index + 1;
    previousScore = item.score;
    result.set(item.horseId, { score: Math.round(item.score * 10) / 10, rank: currentRank, rankedCount: ranked.length });
  });
  raw.filter((item) => item.score === null).forEach((item) => {
    result.set(item.horseId, { score: null, rank: null, rankedCount: ranked.length });
  });
  return result;
}

function buildCategoryRanks(
  horses: Horse[],
  factorScores: Map<string, FactorScore[]>,
  distanceMeters: number | null
): Map<string, CategoryRank[]> {
  const result = new Map<string, CategoryRank[]>();
  const categories = ["騎手", "調教", "血統", "距離適性", "コース適性", "馬場適性", "枠順", "種牡馬", "前走距離", "ポジション"];
  for (const horse of horses) {
    result.set(horse.horseId, categories.map((label) => ({
      label, rank: null, total: 0, detail: "評価に必要なデータが未取得",
    })));
  }
  const distanceValues = horses.map((horse) => {
    const value = (factorScores.get(horse.horseId) ?? []).find((item) => item.label === "距離適性");
    return value ? { horseId: horse.horseId, score: value.score, starts: value.starts, rate: value.rate } : null;
  }).filter((value): value is { horseId: string; score: number; starts: number; rate: number } => value !== null)
    .sort((a, b) => b.score - a.score);
  let distanceRank = 0;
  let previousDistanceScore: number | null = null;
  distanceValues.forEach((value, index) => {
    if (previousDistanceScore === null || Math.abs(previousDistanceScore - value.score) >= 0.05) distanceRank = index + 1;
    previousDistanceScore = value.score;
    const row = result.get(value.horseId)?.find((item) => item.label === "距離適性");
    if (row) {
      row.rank = distanceRank;
      row.total = distanceValues.length;
      row.detail = `近似距離 ${value.starts}走・3着以内率 ${Math.round(value.rate * 100)}%`;
    }
  });
  const assignFactorRank = (factorLabel: string, categoryLabel: string, detailBuilder: (value: FactorScore) => string) => {
    const values = horses.map((horse) => {
      const factor = (factorScores.get(horse.horseId) ?? []).find((item) => item.label === factorLabel);
      return factor ? { horseId: horse.horseId, factor } : null;
    }).filter((value): value is { horseId: string; factor: FactorScore } => value !== null)
      .sort((a, b) => b.factor.score - a.factor.score);
    let rank = 0;
    let previousScore: number | null = null;
    values.forEach((value, index) => {
      if (previousScore === null || Math.abs(previousScore - value.factor.score) >= 0.05) rank = index + 1;
      previousScore = value.factor.score;
      const row = result.get(value.horseId)?.find((item) => item.label === categoryLabel);
      if (row) {
        row.rank = rank;
        row.total = values.length;
        row.detail = detailBuilder(value.factor);
      }
    });
  };
  assignFactorRank("競馬場・馬場適性", "コース適性", (value) =>
    `同競馬場・同芝ダート ${value.starts}走・3着以内率 ${Math.round(value.rate * 100)}%`);
  assignFactorRank("馬場適性", "馬場適性", (value) =>
    `同馬場状態 ${value.starts}走・3着以内率 ${Math.round(value.rate * 100)}%`);
  assignFactorRank("騎手相性", "騎手", (value) =>
    `今回の騎手との過去騎乗 ${value.starts}走・3着以内率 ${Math.round(value.rate * 100)}%`);
  assignFactorRank("枠順実績", "枠順", (value) =>
    `今回と同じ枠番 ${value.starts}走・3着以内率 ${Math.round(value.rate * 100)}%`);

  const positionValues = horses.map((horse) => {
    const stats = horse.performanceStats;
    if (!stats || stats.positionStarts === 0) return null;
    return {
      horseId: horse.horseId,
      starts: stats.positionStarts,
      averageGain: stats.positionGainTotal / stats.positionStarts,
    };
  }).filter((value): value is { horseId: string; starts: number; averageGain: number } => value !== null)
    .sort((a, b) => b.averageGain - a.averageGain);
  let positionRank = 0;
  let previousPositionValue: number | null = null;
  positionValues.forEach((value, index) => {
    if (previousPositionValue === null || Math.abs(previousPositionValue - value.averageGain) >= 0.05) {
      positionRank = index + 1;
    }
    previousPositionValue = value.averageGain;
    const row = result.get(value.horseId)?.find((item) => item.label === "ポジション");
    if (row) {
      row.rank = positionRank;
      row.total = positionValues.length;
      const direction = value.averageGain >= 0 ? "平均順位上昇" : "平均順位後退";
      row.detail = `4角からゴールまで ${direction} ${Math.abs(value.averageGain).toFixed(1)}位（${value.starts}走）`;
    }
  });

  for (const horse of horses) {
    const sire = horse.profile?.find((item) => /^父$/.test(item.label.replace(/\s/g, "")))?.value ?? "";
    const damsire = horse.profile?.find((item) => /母父/.test(item.label.replace(/\s/g, "")))?.value ?? "";
    const bloodline = [sire ? `父：${sire}` : "", damsire ? `母父：${damsire}` : ""].filter(Boolean).join(" / ");
    const rows = result.get(horse.horseId) ?? [];
    const bloodlineRow = rows.find((item) => item.label === "血統");
    const sireRow = rows.find((item) => item.label === "種牡馬");
    if (bloodline && bloodlineRow) {
      bloodlineRow.detail = bloodline + "（血統情報取得済み・適性比較は未実装）";
    }
    if (sire && sireRow) {
      sireRow.detail = `父：${sire}（種牡馬成績の比較は未実装）`;
    }
  }

  if (distanceMeters !== null) {
    const lastDistances = horses.map((horse) => {
      const first = horse.performances?.[0]?.values ?? {};
      const entry = Object.entries(first).find(([key]) => /距離/.test(key.replace(/\s+/g, "")));
      const match = entry?.[1]?.match(/(\d{3,4})\s*m?/i);
      return match ? { horseId: horse.horseId, distance: Number(match[1]), gap: Math.abs(Number(match[1]) - distanceMeters) } : null;
    }).filter((value): value is { horseId: string; distance: number; gap: number } => value !== null)
      .sort((a, b) => a.gap - b.gap);
    let previousGap: number | null = null;
    let previousRank = 0;
    lastDistances.forEach((value, index) => {
      if (previousGap === null || previousGap !== value.gap) previousRank = index + 1;
      previousGap = value.gap;
      const row = result.get(value.horseId)?.find((item) => item.label === "前走距離");
      if (row) {
        row.rank = previousRank;
        row.total = lastDistances.length;
        row.detail = `前走 ${value.distance}m・今回との差 ${value.gap}m`;
      }
    });
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
    {scores.length > 0 ? <div className="prediction-logic-list">
      {scores.map((factor) => <div className="prediction-logic-row" key={factor.label}>
        <div><b>{factor.label}</b><span>{factor.starts}走中 {Math.round(factor.rate * 100)}%が3着以内</span></div>
        <strong>{factor.rank}位 / {factor.rankedCount}頭</strong>
      </div>)}
    </div> : <p className="prediction-no-data">評価に使える戦績データがありません</p>}
  </article>;
}

function OverallHorseCard({ horse, overall, categories, factors }: {
  horse: Horse; overall: OverallScore; categories: CategoryRank[]; factors: FactorScore[];
}) {
  const entry = horse.raceEntry;
  return <article className="prediction-horse">
    <div className="prediction-horse-head">
      {entry?.horseNumber != null && <span className="horse-number-badge">{entry.horseNumber}番</span>}
      <div className="horse-main"><b>{horse.name}</b>
        <small>{[entry?.jockey, entry?.carriedWeight != null ? String(entry.carriedWeight) + "kg" : null].filter(Boolean).join(" ・ ")}</small>
      </div>
    </div>
    <div className="overall-score-block">
      <div className="deviation-label"><span>総合偏差値</span>
        <b>{overall.score === null ? "算出不可" : `偏差値 ${overall.score.toFixed(1)} ・ ${overall.rank}位 / ${overall.rankedCount}頭`}</b></div>
      {overall.score !== null && <div className="deviation-track" role="img" aria-label={`総合偏差値 ${overall.score.toFixed(1)}`}>
        <div className="deviation-fill" style={{ width: Math.max(0, Math.min(100, overall.score / 80 * 100)) + "%" }} />
        <span className="deviation-average" />
      </div>}
    </div>
    <div className="overall-factor-list">
      {factors.map((factor) => <div className="deviation-row" key={factor.label}>
        <div className="deviation-label"><span>{factor.label}</span><b>偏差値 {factor.score.toFixed(1)}</b></div>
        <div className="deviation-track" role="img" aria-label={factor.label + "偏差値 " + factor.score.toFixed(1)}>
          <div className="deviation-fill" style={{ width: Math.max(0, Math.min(100, factor.score / 80 * 100)) + "%" }} />
          <span className="deviation-average" />
        </div>
      </div>)}
    </div>
    <div className="category-rank-list">
      {categories.map((item) => <div className="category-rank-row" key={item.label}>
        <b>{item.label}</b>
        <span>{item.rank === null ? (item.detail.includes("取得済み") || item.detail.includes("未実装") ? "情報あり・評価未実装" : "算出不可") : `${item.rank}位 / ${item.total}頭`}</span>
        <small>{item.detail}</small>
      </div>)}
    </div>
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
  const [activeTab, setActiveTab] = useState<"detail" | "prediction" | "overall">("detail");

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
  const overallScores = useMemo(() => buildOverallScores(horses, factorScores), [horses, factorScores]);
  const categoryRanks = useMemo(() => buildCategoryRanks(horses, factorScores, conditions?.distanceMeters ?? null), [horses, factorScores, conditions?.distanceMeters]);
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
        <button type="button" role="tab" aria-selected={activeTab === "overall"} className={"race-tab" + (activeTab === "overall" ? " active" : "")}
          onClick={() => setActiveTab("overall")}>総合評価</button>
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
        <div className="prediction-intro"><h2>AI予測ロジックの判定材料</h2>
          <p>過去成績から全体成績・重賞実績・競馬場と芝ダート・距離・馬場・今回の騎手・今回の枠番の実績を比較しています。</p>
        </div>
        {horses.map((horse) => <PredictionHorseCard key={horse.horseId} horse={horse}
          scores={factorScores.get(horse.horseId) ?? []} />)}
      </section>}
      {activeTab === "overall" && <section className="prediction-panel" role="tabpanel">
        <div className="prediction-intro"><h2>全馬の総合評価</h2>
          <p>総合偏差値は、過去成績から算出できた最大7項目の偏差値の平均です。血統情報は取得できた馬のみ表示し、調教データは未取得のため評価に含めていません。</p>
        </div>
        {horses.map((horse) => <OverallHorseCard key={horse.horseId} horse={horse}
          overall={overallScores.get(horse.horseId) ?? { score: null, rank: null, rankedCount: 0 }}
          categories={categoryRanks.get(horse.horseId) ?? []}
          factors={factorScores.get(horse.horseId) ?? []} />)}
      </section>}
    </>}
  </main>;
}
