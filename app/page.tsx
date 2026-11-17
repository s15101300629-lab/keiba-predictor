"use client";

import { useEffect, useMemo, useState } from "react";
import Link from "next/link";

type RaceItem = {
  raceId: string;
  raceDate: string;
  venue: string;
  raceNumber: number;
  raceTitle: string;
  startTime: string | null;
  course: string | null;
  runners: number | null;
};

type RaceListResponse = {
  ok: boolean;
  month?: string;
  year?: number;
  startDate?: string;
  endDate?: string;
  raceDates?: string[];
  races?: RaceItem[];
  error?: string;
};

const VENUE_ORDER = ["札幌", "函館", "福島", "新潟", "東京", "中山", "中京", "京都", "阪神", "小倉"];
const CLIENT_CACHE_TTL_MS = 10 * 60 * 1000;
const RACE_LIST_CACHE_KEY = "keiba-race-list-v1";
type CachedRaceList = { savedAt: number; data: RaceListResponse };

function getJstToday() {
  const parts = new Intl.DateTimeFormat("en-CA", { timeZone: "Asia/Tokyo", year: "numeric", month: "2-digit", day: "2-digit" }).formatToParts(new Date());
  const year = parts.find((part) => part.type === "year")?.value ?? "";
  const month = parts.find((part) => part.type === "month")?.value ?? "";
  const day = parts.find((part) => part.type === "day")?.value ?? "";
  return { year: Number(year), month, today: year + month + day };
}
function readRaceListCache(): CachedRaceList | null {
  try {
    const raw = window.localStorage.getItem(RACE_LIST_CACHE_KEY);
    if (!raw) return null;
    const cached = JSON.parse(raw) as CachedRaceList;
    if (!cached || typeof cached.savedAt !== "number" || !cached.data?.ok ||
      !Array.isArray(cached.data.races) || !Array.isArray(cached.data.raceDates) ||
      typeof cached.data.month !== "string" || typeof cached.data.year !== "number") return null;
    return cached;
  } catch { return null; }
}
function saveRaceListCache(data: RaceListResponse) {
  try { window.localStorage.setItem(RACE_LIST_CACHE_KEY, JSON.stringify({ savedAt: Date.now(), data })); }
  catch { /* 保存できない環境でも通常の取得・表示は継続する */ }
}

function formatTabDate(date: string) {
  if (!/^\d{8}$/.test(date)) return date;
  const value = new Date(Date.UTC(Number(date.slice(0, 4)), Number(date.slice(4, 6)) - 1, Number(date.slice(6, 8))));
  const weekday = ["日", "月", "火", "水", "木", "金", "土"][value.getUTCDay()];
  return `${Number(date.slice(4, 6))}/${Number(date.slice(6, 8))}（${weekday}）`;
}

export default function Home() {
  const [races, setRaces] = useState<RaceItem[]>([]);
  const [raceDates, setRaceDates] = useState<string[]>([]);
  const [month, setMonth] = useState("");
  const [selectedDate, setSelectedDate] = useState("");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    let active = true;
    const today = getJstToday();
    const cached = readRaceListCache();
    const cacheMatchesMonth = cached?.data.year === today.year && cached?.data.month === today.month;
    const cacheHasToday = cached?.data.startDate === today.today;
    const hasCache = Boolean(cached && cacheMatchesMonth);
    if (hasCache && cached) {
      const dates = cached.data.raceDates ?? [];
      setRaces(cached.data.races ?? []);
      setRaceDates(dates);
      setMonth(cached.data.month ?? "");
      setSelectedDate((current) => dates.includes(current) ? current : dates[0] ?? "");
      setLoading(false);
    }
    const cacheIsFresh = Boolean(cached && cacheMatchesMonth && cacheHasToday &&
      Date.now() - cached.savedAt < CLIENT_CACHE_TTL_MS);
    if (cacheIsFresh) return () => { active = false; };
    async function load() {
      try {
        if (!hasCache) setLoading(true);
        setError("");
        const response = await fetch("/api/race-list", { cache: "no-store" });
        const data = (await response.json()) as RaceListResponse;
        if (!response.ok || !data.ok) throw new Error(data.error || `開催情報の取得に失敗しました（HTTP ${response.status}）`);
        if (active) {
          const dates = data.raceDates ?? [];
          setRaces(data.races ?? []);
          setRaceDates(dates);
          setMonth(data.month ?? "");
          setSelectedDate((current) => dates.includes(current) ? current : dates[0] ?? "");
          saveRaceListCache(data);
        }
      } catch (err) {
        if (active && !hasCache) setError(err instanceof Error ? err.message : "開催情報の取得に失敗しました。");
      } finally { if (active) setLoading(false); }
    }
    void load();
    return () => { active = false; };
  }, []);

  const grouped = useMemo(() => {
    const selectedRaces = races.filter((race) => race.raceDate === selectedDate);
    return VENUE_ORDER
      .map((venue) => ({
        venue,
        races: selectedRaces
          .filter((race) => race.venue === venue)
          .sort((a, b) => a.raceNumber - b.raceNumber),
      }))
      .filter((group) => group.races.length > 0);
  }, [races, selectedDate]);

  return (
    <main className="shell">
      <header className="topbar">
        <div className="brand"><span className="brand-mark">K</span><div><strong>KEIBA PREDICTOR</strong><small>PERSONAL JRA ANALYSIS</small></div></div>
        <span className="private-pill">PRIVATE</span>
      </header>

      <div className="month-row">
        <span className="date-chip">{month ? `${Number(month)}月` : "取得中"}</span>
      </div>

      <section className="venue-section">
        <div className="venue-heading"><h3>検証用レース</h3><span>過去開催</span></div>
        <div className="race-list">
          <Link href="/race/202608050411" className="race-card race-card-link">
            <div className="race-time">15:40<span>11R</span></div>
            <div className="race-info">
              <div className="race-title">エリザベス女王杯</div>
              <p>11/15（日）京都 ・ 芝2200m ・ 18頭</p>
            </div>
            <span className="race-number">検証</span>
          </Link>
        </div>
      </section>

      {loading && (
        <div className="state-panel loading-panel">
          <span className="loading-spinner" aria-hidden="true" />
          <span>当月の開催情報を取得しています…</span>
        </div>
      )}
      {!loading && error && (
        <div className="state-panel error-panel">
          <b>開催情報を取得できませんでした</b>
          <p>{error}</p>
          <button className="retry-button" onClick={() => window.location.reload()}>再読み込み</button>
        </div>
      )}
      {!loading && !error && raceDates.length === 0 && (
        <div className="state-panel">
          <b>開催予定はありません</b>
          <p>当月のこれからの開催日から、表示対象レースを取得できませんでした。</p>
        </div>
      )}

      {!loading && !error && raceDates.length > 0 && (
        <>
          <div className="date-tabs" role="tablist" aria-label="開催日">
            {raceDates.map((date) => (
              <button
                key={date}
                type="button"
                role="tab"
                aria-selected={selectedDate === date}
                className={selectedDate === date ? "date-tab active" : "date-tab"}
                onClick={() => setSelectedDate(date)}
              >
                {formatTabDate(date)}
              </button>
            ))}
          </div>

          {grouped.length === 0 ? (
            <div className="state-panel">この日の表示対象レースはありません。</div>
          ) : (
            <div className="venue-list">
              {grouped.map(({ venue, races: venueRaces }) => (
                <section className="venue-section" key={venue}>
                  <div className="venue-heading"><h3>{venue}競馬場</h3><span>{venueRaces.length}レース</span></div>
                  <div className="race-list">
                    {venueRaces.map((race) => (
                      <Link href={`/race/${race.raceId}`} className="race-card race-card-link" key={race.raceId}>
                        <div className="race-time">{race.startTime ?? "--:--"}<span>{race.raceNumber}R</span></div>
                        <div className="race-info">
                          <div className="race-title">{race.raceTitle}</div>
                          <p>{[race.course, race.runners !== null ? `${race.runners}頭` : null].filter(Boolean).join(" ・ ") || "詳細情報は未取得"}</p>
                
                        </div>
                        <span className="race-number">{race.raceNumber}R</span>
                      </Link>
                    ))}
                  </div>
                </section>
              ))}
            </div>
          )}
        </>
      )}
    </main>
  );
}
