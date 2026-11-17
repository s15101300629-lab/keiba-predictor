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
  raceDates?: string[];
  races?: RaceItem[];
  error?: string;
};

const VENUE_ORDER = ["札幌", "函館", "福島", "新潟", "東京", "中山", "中京", "京都", "阪神", "小倉"];

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
    async function load() {
      try {
        setLoading(true);
        setError("");
        const response = await fetch("/api/race-list", { cache: "no-store" });
        const data = (await response.json()) as RaceListResponse;
        if (!response.ok || !data.ok) {
          throw new Error(data.error || `開催情報の取得に失敗しました（HTTP ${response.status}）`);
        }
        if (active) {
          const dates = data.raceDates ?? [];
          setRaces(data.races ?? []);
          setRaceDates(dates);
          setMonth(data.month ?? "");
          setSelectedDate(dates[0] ?? "");
        }
      } catch (err) {
        if (active) setError(err instanceof Error ? err.message : "開催情報の取得に失敗しました。");
      } finally {
        if (active) setLoading(false);
      }
    }
    load();
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
