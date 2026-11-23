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
  isMain?: boolean;
  grade?: string | null;
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
const RACE_LIST_CACHE_KEY = "keiba-race-list-v2"; // v2：全レース（以前は重賞・メインだけ）
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

// トップで選んだ日・競馬場のタブとスクロール位置（レースを開いて戻ったときに元の画面に戻すため。タブを閉じると消える）
const HOME_VIEW_KEY = "keiba-home-view-v1";
type HomeView = { date?: string; venue?: string; scrollY?: number };
function readHomeView(): HomeView {
  try { return JSON.parse(window.sessionStorage.getItem(HOME_VIEW_KEY) ?? "{}") as HomeView; } catch { return {}; }
}
function saveHomeView(patch: HomeView) {
  try { window.sessionStorage.setItem(HOME_VIEW_KEY, JSON.stringify({ ...readHomeView(), ...patch })); } catch { /* 保存できなくても表示は続ける */ }
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
  const [venue, setVenue] = useState("注目"); // 「注目」＝全場の重賞・メイン、競馬場名＝その場の全レース
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [restored, setRestored] = useState(false);
  const [viewReady, setViewReady] = useState(false); // 前の競馬場のタブを読み終えたか（読む前に「重賞・メイン」で上書きしないため）

  // 前に選んでいた競馬場のタブを戻す
  useEffect(() => {
    if (viewReady) return;
    const saved = readHomeView();
    if (saved.venue) setVenue(saved.venue);
    setViewReady(true);
  }, [viewReady]);
  useEffect(() => { if (viewReady && selectedDate) saveHomeView({ date: selectedDate }); }, [viewReady, selectedDate]);
  useEffect(() => { if (viewReady) saveHomeView({ venue }); }, [viewReady, venue]);

  useEffect(() => {
    let active = true;
    const pickDate = (dates: string[], current: string) => {
      const saved = readHomeView().date;
      if (current && dates.includes(current)) return current;
      if (saved && dates.includes(saved)) return saved;
      return dates[0] ?? "";
    };
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
      setSelectedDate((current) => pickDate(dates, current));
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
          setSelectedDate((current) => pickDate(dates, current));
          saveRaceListCache(data);
        }
      } catch (err) {
        if (active && !hasCache) setError(err instanceof Error ? err.message : "開催情報の取得に失敗しました。");
      } finally { if (active) setLoading(false); }
    }
    void load();
    return () => { active = false; };
  }, []);

  // 選んだ日の競馬場（タブ）。「注目」は全場の重賞・メインを発走時刻順、競馬場のタブはその場の全レースをレース番号順
  const venues = useMemo(() => [...new Set(races.filter((r) => r.raceDate === selectedDate).map((r) => r.venue))]
    .sort((a, b) => VENUE_ORDER.indexOf(a) - VENUE_ORDER.indexOf(b)), [races, selectedDate]);
  const dayRaces = useMemo(() => {
    const day = races.filter((race) => race.raceDate === selectedDate);
    if (venue === "注目" || !venues.includes(venue)) {
      return day.filter((r) => r.isMain || r.grade)
        .sort((a, b) => (a.startTime ?? "99:99").localeCompare(b.startTime ?? "99:99") ||
          VENUE_ORDER.indexOf(a.venue) - VENUE_ORDER.indexOf(b.venue));
    }
    return day.filter((r) => r.venue === venue).sort((a, b) => a.raceNumber - b.raceNumber);
  }, [races, selectedDate, venue, venues]);

  // スクロール位置を覚えて、一覧が出たら1回だけ戻す
  useEffect(() => {
    if (restored || !viewReady || loading || dayRaces.length === 0) return;
    const y = readHomeView().scrollY;
    if (y) window.requestAnimationFrame(() => window.scrollTo(0, y));
    setRestored(true);
  }, [restored, viewReady, loading, dayRaces.length]);
  useEffect(() => {
    if (!restored) return;
    let timer = 0;
    const onScroll = () => { window.clearTimeout(timer); timer = window.setTimeout(() => saveHomeView({ scrollY: window.scrollY }), 150); };
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => { window.removeEventListener("scroll", onScroll); window.clearTimeout(timer); };
  }, [restored]);

  return (
    <main className="shell">
      <p className="month-caption">{month ? `${Number(month)}月の開催` : "開催情報を取得中"}</p>

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
              <button key={date} type="button" role="tab" aria-selected={selectedDate === date}
                className={selectedDate === date ? "date-tab active" : "date-tab"}
                onClick={() => { setSelectedDate(date); saveHomeView({ scrollY: 0 }); }}>
                {formatTabDate(date)}
              </button>
            ))}
          </div>
          <div className="venue-tabs" role="tablist" aria-label="競馬場">
            {["注目", ...venues].map((v) => (
              <button key={v} type="button" role="tab" aria-selected={venue === v || (v === "注目" && !venues.includes(venue))}
                className={venue === v || (v === "注目" && !venues.includes(venue)) ? "venue-tab active" : "venue-tab"}
                onClick={() => { setVenue(v); saveHomeView({ scrollY: 0 }); }}>
                {v === "注目" ? "重賞・メイン" : v}
              </button>
            ))}
          </div>
          {dayRaces.length === 0
            ? <div className="state-panel">この日の表示対象レースはありません。</div>
            : <div className="race-list">{dayRaces.map((race) => <RaceCard race={race} key={race.raceId} />)}</div>}
        </>
      )}

      <section className="sample-section">
        <h3>検証用レース</h3>
        <RaceCard race={{ raceId: "202608050411", raceDate: "20261115", venue: "京都", raceNumber: 11,
          raceTitle: "エリザベス女王杯", startTime: "15:40", course: "芝2200m", runners: 18, grade: "G1" }}
          dateLabel="11/15（日）" />
      </section>
    </main>
  );
}

function RaceCard({ race, dateLabel }: { race: RaceItem; dateLabel?: string }) {
  const title = race.raceTitle.replace(/[（(]\s*G[123IⅠⅡⅢ]+\s*[)）]/gi, "").trim();
  const surface = race.course?.startsWith("芝") ? "turf" : race.course?.startsWith("ダ") ? "dirt" : "";
  return (
    <Link href={`/race/${race.raceId}`} className="race-card race-card-link">
      <span className={"grade-badge " + (race.grade ? race.grade.toLowerCase() : race.isMain ? "main" : "plain")}>
        {race.grade ?? (race.isMain ? "メイン" : `${race.raceNumber}R`)}</span>
      <div className="race-info">
        <div className="race-title">{title}</div>
        <p>
          <b>{dateLabel ? dateLabel + " " : ""}{race.venue}{race.raceNumber}R</b>
          {race.startTime && <span>{race.startTime}</span>}
          {race.course && <span className={"surface " + surface}>{race.course}</span>}
          {race.runners !== null && <span>{race.runners}頭</span>}
        </p>
      </div>
      <span className="race-arrow" aria-hidden="true">›</span>
    </Link>
  );
}
