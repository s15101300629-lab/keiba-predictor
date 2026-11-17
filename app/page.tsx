"use client";

import Link from "next/link";
import { useEffect, useMemo, useState } from "react";

type RaceItem = {
  raceId: string;
  venue: string;
  raceNumber: number;
  raceTitle: string;
  startTime: string | null;
  course: string | null;
  runners: number | null;
  isMain: boolean;
  grade: string | null;
  labels: string[];
};

type RaceListResponse = {
  ok: boolean;
  date?: string;
  raceCount?: number;
  races?: RaceItem[];
  error?: string;
};

const VENUE_ORDER = ["札幌", "函館", "福島", "新潟", "東京", "中山", "中京", "京都", "阪神", "小倉"];

function formatDate(date: string) {
  if (!/^\d{8}$/.test(date)) return date;
  return `${date.slice(0, 4)}年${Number(date.slice(4, 6))}月${Number(date.slice(6, 8))}日`;
}

export default function Home() {
  const [races, setRaces] = useState<RaceItem[]>([]);
  const [date, setDate] = useState("");
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
          setRaces(data.races ?? []);
          setDate(data.date ?? "");
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
    const groups = new Map<string, RaceItem[]>();
    for (const venue of VENUE_ORDER) {
      const venueRaces = races
        .filter((race) => race.venue === venue)
        .sort((a, b) => a.raceNumber - b.raceNumber);
      if (venueRaces.length) groups.set(venue, venueRaces);
    }
    return Array.from(groups.entries());
  }, [races]);

  return (
    <main className="shell">
      <header className="topbar">
        <div className="brand"><span className="brand-mark">K</span><div><strong>KEIBA PREDICTOR</strong><small>PERSONAL JRA ANALYSIS</small></div></div>
        <span className="private-pill">PRIVATE</span>
      </header>

      <section className="hero">
        <p className="eyebrow">TODAY'S RACE CENTER</p>
        <h1>今日の注目レースを、<br /><span>競馬場ごとに。</span></h1>
        <p className="hero-copy">当日のJRA開催情報から、各競馬場のメインレースと重賞だけを表示します。平場のレースは表示しません。</p>
        <div className="hero-stats">
          <div><small>対象</small><b>JRA</b></div>
          <div><small>表示対象</small><b>メイン・重賞</b></div>
          <div><small>開催日</small><b>{date ? formatDate(date) : "取得中"}</b></div>
        </div>
      </section>

      <section className="section-head">
        <div><p className="eyebrow">FEATURED RACES</p><h2>本日のメイン・重賞</h2></div>
        <span className="date-chip">{date ? formatDate(date) : "本日"}</span>
      </section>

      {loading && <div className="state-panel">本日の開催レースを取得しています…</div>}
      {!loading && error && (
        <div className="state-panel error-panel">
          <b>開催情報を取得できませんでした</b>
          <p>{error}</p>
          <button className="retry-button" onClick={() => window.location.reload()}>再読み込み</button>
        </div>
      )}
      {!loading && !error && grouped.length === 0 && (
        <div className="state-panel">
          <b>本日の表示対象レースはありません</b>
          <p>JRAの開催がない日、または開催一覧からメイン・重賞を取得できなかった場合に表示されます。</p>
        </div>
      )}

      <div className="venue-list">
        {grouped.map(([venue, venueRaces]) => (
          <section className="venue-section" key={venue}>
            <div className="venue-heading"><h3>{venue}競馬場</h3><span>{venueRaces.length}レース</span></div>
            <div className="race-list">
              {venueRaces.map((race) => (
                <article className="race-card" key={race.raceId}>
                  <div className="race-time">{race.startTime ?? "--:--"}<span>{race.raceNumber}R</span></div>
                  <div className="race-info">
                    <div className="race-title">{race.raceTitle}</div>
                    <p>{[race.course, race.runners !== null ? `${race.runners}頭` : null].filter(Boolean).join(" ・ ") || "詳細情報は未取得"}</p>
                    <div className="race-labels">
                      {race.isMain && <span className="race-label main-label">メイン</span>}
                      {race.grade && <span className="race-label grade-label">重賞 {race.grade}</span>}
                    </div>
                  </div>
                  <span className="race-number">{race.raceNumber}R</span>
                </article>
              ))}
            </div>
          </section>
        ))}
      </div>

      <section className="section-head lower"><div><p className="eyebrow">DATA TOOLS</p><h2>データ取り込み</h2></div></section>
      <Link href="/import" className="import-link">出馬表・血統・過去戦績の取り込みを開く →</Link>
      <footer>KEIBA PREDICTOR <span>個人利用向け・開発中</span></footer>
    </main>
  );
}
