"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
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
  bodyWeight: number | null; odds: number | null; popularity: number | null;
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
function display(value: string | number | null | undefined) {
  return value === null || value === undefined || value === "" ? "未発表・未取得" : String(value);
}

export default function RaceDetailPage() {
  const params = useParams<{ raceId: string }>();
  const raceId = params.raceId;
  const [data, setData] = useState<ResponseData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    let active = true;
    async function load() {
      try {
        setLoading(true);
        setError("");
        const response = await fetch("/api/races/" + raceId, { cache: "no-store" });
        const result = (await response.json()) as ResponseData;
        if (!response.ok || !result.ok) {
          throw new Error(result.error || "レース情報の取得に失敗しました（HTTP " + response.status + "）");
        }
        if (active) setData(result);
      } catch (err) {
        if (active) setError(err instanceof Error ? err.message : "レース情報の取得に失敗しました。");
      } finally {
        if (active) setLoading(false);
      }
    }
    if (/^\d{12}$/.test(raceId)) load();
    else { setError("レースIDが正しくありません。"); setLoading(false); }
    return () => { active = false; };
  }, [raceId]);

  const conditions = data?.raceConditions;
  const horses = data?.results ?? [];

  return (
    <main className="race-detail-shell">
      <Link href="/" className="back-link">← 開催レース一覧へ戻る</Link>
      {loading && <div className="state-panel">レース情報と出走馬のデータを取得しています…</div>}
      {!loading && error && <div className="state-panel error-panel"><b>レース情報を取得できませんでした</b><p>{error}</p></div>}
      {!loading && data && conditions && (
        <>
          <header className="detail-hero">
            <p className="eyebrow">RACE DETAIL</p>
            <h1>{conditions.raceName || "レース詳細"}</h1>
            <p className="detail-meta">
              {display(conditions.date)} ・ {display(conditions.venue)}競馬場 ・ {display(conditions.raceNumber)}R
              <br />{display(conditions.startTime)} 発走
            </p>
          </header>
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
            {horses.map((horse) => {
              const entry = horse.raceEntry;
              const profile = horse.profile ?? [];
              const profileValue = (label: string) => profile.find((item) => item.label.includes(label))?.value;
              const histories = horse.performances ?? [];
              return (
                <details className="horse-detail" key={horse.horseId}>
                  <summary className="horse-summary">
                    <span className="horse-number-badge">{display(entry?.horseNumber)}番</span>
                    <span className="horse-main">
                      <b>{horse.name}</b>
                      <small>{[entry?.sexAge, entry?.jockey, entry?.carriedWeight ? String(entry.carriedWeight) + "kg" : null].filter(Boolean).join(" ・ ") || "出走情報の一部は未発表"}</small>
                    </span>
                    <span className="horse-expand">＋</span>
                  </summary>
                  <div className="horse-body">
                    <div className="horse-facts">
                      <div className="horse-fact"><span>枠番：</span>{display(entry?.frameNumber)}</div>
                      <div className="horse-fact"><span>馬体重：</span>{entry?.bodyWeight ? String(entry.bodyWeight) + "kg" : "未発表"}</div>
                      <div className="horse-fact"><span>オッズ：</span>{display(entry?.odds)}</div>
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
                </details>
              );
            })}
          </section>
        </>
      )}
    </main>
  );
}
