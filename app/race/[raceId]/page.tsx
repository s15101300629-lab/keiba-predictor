"use client";

import Link from "next/link";
import { useCallback, useEffect, useState } from "react";
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
type Horse = {
  horseId: string; name: string; status: string; raceEntry: Entry | null;
  profile: { label: string; value: string }[] | null;
  performances: { values: Record<string, string> }[]; historyReturned: number;
};
type ResponseData = {
  ok: boolean; raceId: string; raceConditions: Conditions; horseCount: number;
  completeCount: number; partialCount: number; failedCount: number;
  results: Horse[]; error?: string;
};
type CachedRace = {
  savedAt: number;
  data: ResponseData;
};

const CLIENT_CACHE_TTL_MS = 5 * 60 * 1000;
const cacheKey = (raceId: string) => "keiba-race-detail-v1:" + raceId;

function readCachedRace(raceId: string): CachedRace | null {
  try {
    const raw = window.localStorage.getItem(cacheKey(raceId));
    if (!raw) return null;
    const parsed = JSON.parse(raw) as CachedRace;
    if (
      !parsed ||
      typeof parsed.savedAt !== "number" ||
      !parsed.data ||
      parsed.data.raceId !== raceId ||
      !parsed.data.raceConditions ||
      !Array.isArray(parsed.data.results)
    ) return null;
    return parsed;
  } catch {
    return null;
  }
}

function saveCachedRace(raceId: string, data: ResponseData, savedAt: number) {
  try {
    const cached: CachedRace = { savedAt, data };
    window.localStorage.setItem(cacheKey(raceId), JSON.stringify(cached));
  } catch {
    // ストレージ容量不足・利用不可の場合も、画面表示とAPI取得は継続する
  }
}

function display(value: string | number | null | undefined) {
  return value === null || value === undefined || value === "" ? "未発表・未取得" : String(value);
}

function formatSavedAt(timestamp: number | null) {
  if (!timestamp) return "保存データなし";
  return "保存データ：" + new Date(timestamp).toLocaleString("ja-JP", {
    month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit",
  });
}

function HorseCard({ horse }: { horse: Horse }) {
  const [expanded, setExpanded] = useState(false);
  const entry = horse.raceEntry;
  const profile = horse.profile ?? [];
  const profileValue = (label: string) => profile.find((item) => item.label.includes(label))?.value;
  const histories = horse.performances ?? [];

  return (
    <details className="horse-detail" onToggle={(event) => setExpanded(event.currentTarget.open)}>
      <summary className="horse-summary">
        <span className="horse-number-badge">{display(entry?.horseNumber)}番</span>
        <span className="horse-main">
          <b>{horse.name}</b>
          <small>{[entry?.sexAge, entry?.jockey, entry?.carriedWeight ? String(entry.carriedWeight) + "kg" : null].filter(Boolean).join(" ・ ") || "出走情報の一部は未発表"}</small>
        </span>
        <span className="horse-expand">{expanded ? "－" : "＋"}</span>
      </summary>
      {expanded && (
        <div className="horse-body">
          <div className="horse-facts">
            <div className="horse-fact"><span>枠番：</span>{display(entry?.frameNumber)}</div>
            <div className="horse-fact"><span>馬体重：</span>{entry?.bodyWeight ? String(entry.bodyWeight) + "kg" : "未発表"}</div>
            <div className="horse-fact"><span>人気：</span>{display(entry?.popularity)}</div>
            <div className="horse-fact"><span>調教師：</span>{display(entry?.trainer)}</div>
            <div className="horse-fact"><span>父：</span>{display(profileValue("父"))}</div>
            <div className="horse-fact"><span>母：</span>{display(profileValue("母"))}</div>
            <div className="horse-fact"><span>母父：</span>{display(profileValue("母父"))}</div>
          </div>
          {histories.length > 0 ? (
            <div className="history-table-wrap">
              <b style={{ fontSize: 12 }}>過去の戦績（取得分 {histories.length}走）</b>
              <table className="history-table">
                <thead><tr>{Object.keys(histories[0].values).map((key) => <th key={key}>{key}</th>)}</tr></thead>
                <tbody>{histories.map((history, index) => <tr key={index}>{Object.keys(histories[0].values).map((key) => <td key={key}>{display(history.values[key])}</td>)}</tr>)}</tbody>
              </table>
            </div>
          ) : <p className="detail-meta">過去の戦績は取得できませんでした。</p>}
        </div>
      )}
    </details>
  );
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

  const load = useCallback(async (forceRefresh: boolean, background: boolean) => {
    try {
      if (forceRefresh || background) setRefreshing(true);
      else setLoading(true);
      setError("");
      const refreshQuery = forceRefresh ? "?refresh=1" : "";
      const response = await fetch("/api/races/" + raceId + refreshQuery, { cache: "no-store" });
      const result = (await response.json()) as ResponseData;
      if (!response.ok || !result.raceConditions || !Array.isArray(result.results)) {
        throw new Error(result.error || "レース情報の取得に失敗しました（HTTP " + response.status + "）");
      }
      const savedAt = Date.now();
      setData(result);
      setCachedAt(savedAt);
      saveCachedRace(raceId, result, savedAt);
    } catch (err) {
      const message = err instanceof Error ? err.message : "レース情報の取得に失敗しました。";
      if (background || forceRefresh) {
        setError("最新情報への更新に失敗しました。表示中の保存データを利用しています。 " + message);
      } else {
        setError(message);
      }
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }, [raceId]);

  useEffect(() => {
    if (!/^\d{12}$/.test(raceId)) {
      setError("レースIDが正しくありません。");
      setLoading(false);
      setCacheChecked(true);
      return;
    }

    const cached = readCachedRace(raceId);
    setCacheChecked(true);
    if (cached) {
      setData(cached.data);
      setCachedAt(cached.savedAt);
      setLoading(false);

      if (Date.now() - cached.savedAt >= CLIENT_CACHE_TTL_MS) {
        void load(false, true);
      }
    } else {
      void load(false, false);
    }
  }, [raceId, load]);

  const conditions = data?.raceConditions;
  const horses = data?.results ?? [];

  return (
    <main className="race-detail-shell">
      <Link href="/" className="back-link">← 開催レース一覧へ戻る</Link>
      {loading && !cacheChecked && <div className="state-panel">初回データを確認しています…</div>}
      {loading && cacheChecked && !data && <div className="state-panel">初回データを取得しています…</div>}
      {!data && !loading && error && <div className="state-panel error-panel"><b>レース情報を取得できませんでした</b><p>{error}</p></div>}
      {data && conditions && (
        <>
          <header className="detail-hero">
            <p className="eyebrow">RACE DETAIL</p>
            <h1>{conditions.raceName || "レース詳細"}</h1>
            <p className="detail-meta">
              {display(conditions.date)} ・ {display(conditions.venue)}競馬場 ・ {display(conditions.raceNumber)}R
              <br />{display(conditions.startTime)} 発走
            </p>
          </header>
          <button
            type="button"
            className="refresh-button"
            onClick={() => void load(true, false)}
            disabled={refreshing || loading}
          >
            {refreshing ? "最新情報を取得中…" : "最新情報に更新"}
          </button>
          <p className="detail-meta">
            {formatSavedAt(cachedAt)}
            {refreshing ? " ・ バックグラウンドで最新情報を確認中" : " ・ 保存データは5分ごとに自動確認"}
          </p>
          {error && <p className="detail-meta" role="status">{error}</p>}
          <section className="detail-section">
            <h2>レース条件</h2>
            <div className="condition-grid">
              {[
                ["コース", [conditions.surface, conditions.distanceMeters ? String(conditions.distanceMeters) + "m" : null].filter(Boolean).join(" ")],
                ["回り・コース", [conditions.direction, conditions.courseLayout, conditions.courseLetter].filter(Boolean).join(" ")],
                ["天候", conditions.weather],
                ["馬場状態", conditions.going],
              ].map(([label, value]) => (
                <div className="condition-item" key={label}><small>{label}</small><b>{display(value)}</b></div>
              ))}
            </div>
          </section>
          <section className="detail-section">
            <h2>出走馬 <span style={{ color: "var(--muted)", fontSize: 12 }}>（{data.horseCount}頭）</span></h2>
            <p className="detail-meta">取得完了 {data.completeCount}頭 ・ 一部取得 {data.partialCount}頭 ・ 取得失敗 {data.failedCount}頭</p>
            {horses.map((horse) => <HorseCard key={horse.horseId} horse={horse} />)}
          </section>
        </>
      )}
    </main>
  );
}
