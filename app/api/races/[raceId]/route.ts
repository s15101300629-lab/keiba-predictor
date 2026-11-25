import { NextResponse } from "next/server";
import { AsyncLocalStorage } from "node:async_hooks";
import { readFile } from "node:fs/promises";
import path from "node:path";
import { cleanText, decodeEntities, extractPedigree, extractTables } from "@/app/lib/netkeiba";
import { fetchJraBaba, fetchJraBabaDetail } from "@/app/lib/jraBaba";
import { addDisplayDeviation, formFromHistory, heavyRunsInHistory, type FormDisplay } from "@/app/lib/form";
import { addFieldFeatures, addFormDeviation, addOpponentFeatures, addSimilarCondFeatures, courseInfo, expectedPace, fillForeignPeople, jockeyStats, resolveCareerJockey, resolveTrainer, runningStyle, storeHorseKey, trainerStats, resolveJockey, runnerFeatures, TRAIN_RANK, type RunnerInput, type Store } from "@/app/lib/ml/features";
import { normalizeSum, predictProbability, type Model } from "@/app/lib/ml/model";
import {
  PEDIGREE_SOURCES, parseStallionPages, pedigreeAptitudes, pedigreeModelValues, stackFeatures, stackProbability, stallionNeedsRefresh, stallionRecord,
  type StackModel, type StallionRaw, type StallionRecord,
} from "@/app/lib/pedigree";

export const runtime = "nodejs";

const CONCURRENCY = 4;

type HorseEntry = {
  horseId: string;
  name: string;
};

type RaceEntry = {
  horseId: string;
  horseName: string;
  scratched: boolean; // 出走取消・競走除外
  frameNumber: number | null;
  horseNumber: number | null;
  sexAge: string | null;
  carriedWeight: number | null;
  jockey: string | null;
  trainer: string | null;
  bodyWeight: number | null;
  bodyWeightChange: number | null;
  popularity: number | null;
  // netkeiba のオッズの番号（行の中の id="odds-1_NN"）。枠順の確定後は馬番と同じ。確定前は出馬表の並びの仮の番号で、
  // 予想オッズ（status "yoso"）はこの番号で返る（2026-11-25 に気づいた。前は馬番がないと馬にオッズが付かなかった）
  oddsKey: number | null;
  fieldStatus: {
    frameNumber: string;
    horseNumber: string;
    sexAge: string;
    carriedWeight: string;
    jockey: string;
    bodyWeight: string;
    popularity: string;
  };
};

// 調教評価（race.netkeiba.com の調教ページ。ログインなしで見える短評と A〜D。scripts/ml/collect_training.py と同じ読み方）
function parseTrainingPage(html: string): Map<number, { rank: string | null; critic: string | null }> {
  const out = new Map<number, { rank: string | null; critic: string | null }>();
  const strip = (t: string) => decodeEntities(t.replace(/<[^>]+>/g, " ")).replace(/\s+/g, " ").trim() || null;
  for (const m of html.matchAll(/<tr class\s*=\s*"OikiriDataHead\d+ HorseList"[^>]*>([\s\S]*?)<\/tr>/g)) {
    const row = m[1];
    const umaban = row.match(/class="Umaban">\s*(\d+)/)?.[1];
    if (!umaban) continue;
    const critic = row.match(/class="Training_Critic">([\s\S]*?)<\/td>/)?.[1];
    const rank = row.match(/class="Rank_[^"]*">([\s\S]*?)<\/td>/)?.[1];
    out.set(Number(umaban), { rank: rank ? strip(rank) : null, critic: critic ? strip(critic) : null });
  }
  return out;
}

// --- 海外のレース（成績表にない）を過去走として足す（依頼者の判断。学習データには海外の走がないので、効き方は検証できていない）---
const JRA_VENUES = ["札幌", "函館", "福島", "新潟", "東京", "中山", "中京", "京都", "阪神", "小倉"];
const NAR_VENUES = ["門別", "盛岡", "水沢", "浦和", "船橋", "大井", "川崎", "金沢", "笠松", "名古屋", "園田", "姫路", "高知", "佐賀", "帯広", "ばんえい"];
// 格 → JRA の同じ格の1着賞金の中央値（万円。成績表 2024〜2026年）。条件戦は 1勝クラス とみなす
const OVERSEAS_PRIZE: [RegExp, number][] = [
  [/\((?:GI|G1)\)/, 17000], [/\((?:GII|G2)\)/, 5700], [/\((?:GIII|G3)\)/, 4100],
  [/\(L\)/, 2400], [/\(OP\)/, 2200], [/新馬/, 750], [/未勝利/, 560],
];
const OVERSEAS_PRIZE_DEFAULT = 800;
// 着差がないので、着順ごとの勝ち馬とのタイム差の中央値（成績表 2020年〜）で代える
const TIME_DIFF_BY_FINISH = [0, 0, 0.2, 0.4, 0.6, 0.7, 0.9, 1.1, 1.2, 1.4, 1.6, 1.7, 1.9, 2.2, 2.4, 2.7, 3.1];
const OVERSEAS_GOING: Record<string, number> = { "良": 0, "稍": 1, "重": 2, "不": 3 };

function isOverseasVenue(venue: string): boolean {
  if (!venue) return false;
  return !JRA_VENUES.some((v) => venue.includes(v)) && !NAR_VENUES.some((v) => venue.includes(v));
}

// 海外の走の見積もり（scripts/ml/overseas.py。2026-11-24 採用：外国馬・遠征帰りの日本馬で logloss が一番良かった）
// racedb.com で分かった馬ごとの1着賞金（万円）・タイム差（data/overseas/runs.json）と、主要国以外は格を1つ下げる決まり・能力指数の式（impute.json）
type OverseasImpute = { coef: Record<string, { ab: number[]; tidx: number[] }>; major_venues: string[]; grade_prize: [string, number][]; lower: Record<string, number> };
type OverseasData = { imp: OverseasImpute | null; runs: Record<string, Record<string, { prize?: number; time_diff?: number }>>;
  venueMap: { map: [string, number, number][]; default: [number, number] } | null };
let overseasData: Promise<OverseasData> | null = null;
function loadOverseas(): Promise<OverseasData> {
  overseasData ??= (async () => {
    const dir = path.join(process.cwd(), "data", "overseas");
    const read = (n: string) => readFile(path.join(dir, n), "utf8").then((t) => JSON.parse(t)).catch(() => null);
    const [imp, runs, venueMap] = await Promise.all([read("impute.json"), read("runs.json"), read("venue_map.json")]);
    return { imp, runs: runs ?? {}, venueMap };
  })();
  return overseasData;
}

/** netkeiba の戦績のうち海外のレースを、store.horses と同じ形の過去走にする（見積もりのデータがあれば overseas.py と同じ見積もり）. */
function overseasRuns(performances: { values: Record<string, string> }[], horseId = "", od: OverseasData | null = null): Store["horses"][string] {
  const runs: Store["horses"][string] = [];
  for (const p of performances) {
    const v = p.values;
    if (!isOverseasVenue(v["開催"] ?? "")) continue;
    const date = (v["日付"] ?? "").replace(/\//g, "-");
    const course = (v["距離"] ?? "").match(/^(芝|ダ)(\d+)/);
    const field = Number(v["頭数"]);
    const finish = Number(v["着順"]);
    if (!/^\d{4}-\d{2}-\d{2}$/.test(date) || !course || !(field > 1)) continue;
    const ok = Number.isInteger(finish) && finish > 0;
    const name = v["レース名"] ?? "";
    let prize = OVERSEAS_PRIZE.find(([re]) => re.test(name))?.[1] ?? OVERSEAS_PRIZE_DEFAULT;
    const weight = Number(v["斤量"]);
    let timeDiff = ok ? TIME_DIFF_BY_FINISH[Math.min(finish, TIME_DIFF_BY_FINISH.length - 1)] : -1;
    let tidx: number | null = null;
    let ab: number | null = null;
    const imp = od?.imp;
    if (imp) {
      const x = od.runs[horseId]?.[date];
      const major = imp.major_venues.some((m) => (v["開催"] ?? "").includes(m));
      const g = imp.grade_prize.slice(0, 5).find(([pat]) => new RegExp(pat).test(name)); // G1〜OP
      // 主要国（欧州・香港・ドバイなど）の格のあるレースは JRA の同じ格の値（2026-11-26。scripts/ml/overseas.py の improve と同じ）
      if (major && g) prize = g[1];
      else if (x?.prize != null) prize = x.prize;
      else if (!major && g) prize = imp.lower[String(g[1])] ?? g[1];
      if (x?.time_diff != null && ok) timeDiff = x.time_diff;
      if (ok && timeDiff >= 0) {
        const c = imp.coef[course[1] === "芝" ? "0" : "1"];
        const z = [1, Math.log(Math.max(prize, 100)), Math.min(timeDiff, 3), Math.log(field)];
        const dot = (w: number[]) => Math.round(w.reduce((acc, a, i) => acc + a * z[i], 0) * 1e4) / 1e4;
        ab = dot(c.ab);
        tidx = dot(c.tidx);
      }
    }
    // 海外の競馬場を近い JRA のコースに読み替える（data/overseas/venue_map.json。check_jra_day.py の overseas_course と同じ）
    const surf = course[1] === "芝" ? 0 : 1;
    const vm = od?.venueMap;
    const [jv, layout] = vm ? (vm.map.find(([pat]) => (v["開催"] ?? "").includes(pat))?.slice(1) as [number, number] | undefined) ?? vm.default : [0, 0];
    const ci = vm ? courseInfo(jv, surf, [4, 5, 7].includes(jv) ? "左" : "右", surf === 0 && layout === 1) : null;
    runs.push([
      date, jv, surf, Number(course[2]), OVERSEAS_GOING[(v["馬場"] ?? "").slice(0, 1)] ?? -1, field,
      ok ? finish : -1, timeDiff,
      -1, -1, -1, prize, Number.isFinite(weight) && weight > 0 ? weight : null, -1, -1, tidx, 0, ab,
      null, null, null, null, ci?.turn ?? null, ci?.sband ?? null, null, ci?.cStraight ?? null, ci?.cElev ?? null,
    ]);
  }
  return runs;
}

// 馬体重の発表前に使う値（初出走馬の性別ごとの中央値。学習データ 2020年〜）
const DEBUT_BODY_WEIGHT: Record<string, number> = { "牡": 476, "牝": 448, "セ": 468 };

/** netkeiba の戦績から、レース日より前の一番新しい馬体重（「480(+2)」の 480）。計不・海外などは飛ばす */
/**
 * ラップタイプ（表示用。モデルには入れない）：netkeiba の戦績の本物の上がり3F から、その馬自身のペースの変化を見る。
 * 加速の度合い＝（上がり3F より前の 200m あたり）−（上がり3F の 200m あたり）。プラスなら終いに加速している。
 * レース日より前の近5走（1000m を超える平地・タイムと上り・通過がある走）の平均。区切りは JRA の結果ページの 2026年9〜11月の約7,200走の3等分
 * （加速 0.35秒以上・−0.28秒以下）。加速が大きい走のうち、最後のコーナーが前半分なら加速型、後ろ半分なら終い伸び型
 */
function lapType(performances: { values: Record<string, string> }[], raceDate: string): { type: string; accel: number; last4: number; runs: number } | null {
  const sel: { accel: number; last4: number }[] = [];
  for (const p of performances) {
    const v = p.values;
    const date = (v["日付"] ?? "").replace(/\//g, "-");
    if (!(date < raceDate)) continue;
    const dist = Number((v["距離"] ?? "").match(/^[芝ダ](\d+)/)?.[1]);
    const t = (v["タイム"] ?? "").match(/^(?:(\d+):)?(\d+)\.(\d)$/);
    const agari = Number(v["上り"]);
    const corners = (v["通過"] ?? "").split("-").map(Number).filter((x) => x > 0);
    const field = Number(v["頭数"]);
    if (!(dist > 1000) || !t || !(agari > 0) || !corners.length || !(field > 1)) continue;
    const time = Number(t[1] ?? 0) * 60 + Number(t[2]) + Number(t[3]) / 10;
    sel.push({ accel: (time - agari) / ((dist - 600) / 200) - agari / 3, last4: corners[corners.length - 1] / field });
    if (sel.length >= 5) break;
  }
  if (!sel.length) return null;
  const accel = sel.reduce((a, x) => a + x.accel, 0) / sel.length;
  const last4 = sel.reduce((a, x) => a + x.last4, 0) / sel.length;
  const type = accel >= 0.35 ? (last4 <= 0.5 ? "加速型" : "終い伸び型") : accel <= -0.28 ? "失速型" : "持続型";
  return { type, accel, last4, runs: sel.length };
}

function previousBodyWeight(performances: { values: Record<string, string> }[], raceDate: string): number | null {
  let best: { date: string; weight: number } | null = null;
  for (const p of performances) {
    const date = (p.values["日付"] ?? "").replace(/\//g, "-");
    const weight = (p.values["馬体重"] ?? "").match(/^\s*(\d{3})/);
    if (!weight || !date || date >= raceDate) continue;
    if (!best || date > best.date) best = { date, weight: Number(weight[1]) };
  }
  return best?.weight ?? null;
}

function detectCharset(contentType: string | null, bytes: Uint8Array) {
  const fromHeader = contentType?.match(/charset\s*=\s*["']?([\w-]+)/i)?.[1];
  const head = new TextDecoder("latin1").decode(bytes.subarray(0, 4096));
  const fromMeta = head.match(/<meta\b[^>]*charset\s*=\s*["']?([\w-]+)/i)?.[1];
  const charset = (fromHeader || fromMeta || "").toLowerCase();
  if (/^(euc-?jp|x-euc-jp)$/.test(charset)) return "euc-jp";
  if (/^(shift[_-]?jis|sjis|windows-31j|cp932|x-sjis)$/.test(charset)) return "shift_jis";
  if (/^utf-?8$/.test(charset)) return "utf-8";
  return null;
}

// ページ取得の内訳（2026-11-25、表示を速くする調べ）：1回のリクエストの中で、取ったページを種類ごとに数と時間（足し算。同時に取るので合計は待ち時間より長い）で数える
type FetchStat = { n: number; ms: number; bytes: number };
const fetchStats = new AsyncLocalStorage<Record<string, FetchStat>>();
function fetchKind(url: string): string {
  if (url.includes("/race/shutuba")) return "出馬表";
  if (url.includes("/race/oikiri")) return "調教";
  if (url.includes("db.sp.netkeiba.com/horse/")) return "馬のプロフィール";
  if (url.includes("/horse/result/")) return "戦績";
  if (url.includes("/horse/ped/")) return "血統表";
  if (url.includes("/horse/sire/")) return "種牡馬の成績";
  return "その他";
}

async function fetchHtml(
  url: string,
  revalidateSeconds: number,
  forceRefresh: boolean,
  fallbackEncoding: "utf-8" | "euc-jp" = "utf-8"
) {
  const tFetch = performance.now();
  const response = await fetch(url, {
    ...(forceRefresh
      ? { cache: "no-store" as const }
      : { cache: "force-cache" as const, next: { revalidate: revalidateSeconds } }),
    headers: { Accept: "text/html" },
    signal: AbortSignal.timeout(10000),
  });
  const buffer = new Uint8Array(await response.arrayBuffer());
  const st = fetchStats.getStore();
  if (st) {
    const k = st[fetchKind(url)] ??= { n: 0, ms: 0, bytes: 0 };
    k.n += 1; k.ms += performance.now() - tFetch; k.bytes += buffer.byteLength;
  }
  // netkeiba のページは EUC-JP から UTF-8 に移行しているため、宣言された文字コードで読む
  const encoding = detectCharset(response.headers.get("content-type"), buffer) ?? fallbackEncoding;
  return {
    status: response.status,
    ok: response.ok,
    html: new TextDecoder(encoding).decode(buffer),
    bytes: buffer.byteLength,
  };
}




// レースのクラス（買い目の見送りの判断。scripts/ml/backtest_bets2.py の race_class と同じ分け方）
function raceClassOf(name: string, text: string): string {
  const all = name + " " + text;
  if (/[（(](?:Ｇ|G)[ⅠⅡⅢ123I]+[）)]|GI|GII|GIII|Ｇ[ⅠⅡⅢ]/.test(name)) return "重賞";
  if (/[（(](?:Ｌ|L)[）)]|リステッド/.test(all)) return "L";
  if (/新馬/.test(all)) return "新馬";
  if (/未勝利/.test(name) || /^[^除]*未勝利(?!馬)/.test(text)) return "未勝利";
  if (/1勝クラス|１勝クラス|500万/.test(all)) return "1勝";
  if (/2勝クラス|２勝クラス|1000万/.test(all)) return "2勝";
  if (/3勝クラス|３勝クラス|1600万/.test(all)) return "3勝";
  return "OP";
}

type RaceConditions = {
  raceClass?: string | null;
  raceName: string | null;
  date: string | null;
  venue: string | null;
  meeting: number | null;
  day: number | null;
  raceNumber: number | null;
  startTime: string | null;
  surface: string | null;
  distanceMeters: number | null;
  direction: string | null;
  courseLayout: string | null;
  courseLetter: string | null;
  weather: string | null;
  weatherForecast: boolean; // netkeiba の天候が「予報」（まだ発表前）
  conditionSource?: string; // 天候・馬場を JRA の馬場情報から取ったとき「JRA（11月23日 9:20 取得）」など
  going: string | null;
  fieldStatus: Record<string, string>;
};

function extractRaceConditions(html: string, raceId: string): RaceConditions {
  const pageText = cleanText(html);
  const titleMatch = html.match(/<title\b[^>]*>([\s\S]*?)<\/title>/i);
  const titleText = titleMatch ? cleanText(titleMatch[1]) : "";
  const headingMatch = html.match(/<h1\b[^>]*>([\s\S]*?)<\/h1>/i);
  const headingText = headingMatch ? cleanText(headingMatch[1]) : "";

  const raceDataMatch = html.match(
    /<div\b(?=[^>]*\bclass=["'][^"']*\bRaceData01\b)[^>]*>([\s\S]*?)<\/div>/i
  );
  const raceDataText = raceDataMatch ? cleanText(raceDataMatch[1]) : "";
  const conditionText = raceDataText || pageText;

  const raceNameMatch =
    headingText.match(/(([ぁ-んァ-ヶ一-龠A-Za-zＡ-Ｚａ-ｚ0-9０-９ー・]*)(?:（(?:春|秋)）|\((?:春|秋)\)|チャンピオンシップ|ステークス|フューチュリティ|フィリーズ|記念|賞|杯|優駿牝馬|オークス|エリ女杯?|春天|秋天|有馬|宝塚|日本ダービー|東京優駿|ヴィクトリアマイル|ヴィクトリア[ＭM]|マイル|スプリント|クラシック|グランドジャンプ|大障害|[ＣC]?[ＳS]|[ＦF][ＳS]|[ＪJ][ＦF]|[ＣC]))/) ||
    titleText.match(/^(.+?)(?:\(|（| 出馬表| レース情報)/);
  const dateMatch = pageText.match(/(20\d{2})年\s*(\d{1,2})月\s*(\d{1,2})日/);
  const meetingMatch = pageText.match(/(\d+)回\s*([^\s　]+)\s*(\d+)日目/);

  const startMatch = conditionText.match(/(\d{1,2}:\d{2})\s*発走/);
  const trackMatch = conditionText.match(/(芝|ダート|ダ|障害|障)\s*(\d{3,4})\s*m/i);
  const trackDetail = trackMatch
    ? conditionText.slice(trackMatch.index! + trackMatch[0].length)
    : "";
  const directionMatch = trackDetail.match(/(?:[（(]\s*)?(右|左|直線)/);
  const layoutMatch = trackDetail.match(/(?:[（(]\s*)?(外|内)(?:\s|[A-D]|[）)]|$)/);
  const courseLetterMatch = trackDetail.match(/(?:^|[\s（(])([A-D])(?:コース)?(?=$|[\s）)/])/);

  // 天候・馬場：前の書き方は「天候:晴 / 馬場:良」。2026-11 の書き方は、天候がアイコン（aria-label="晴"、発表前は直前に「予報」）、
  // 馬場は <span class="Item03|Item04"> 良</span> の文字だけ
  const raceDataHtml = raceDataMatch?.[1] ?? "";
  const weatherMatch = conditionText.match(/天候\s*[:：]\s*(小雨|小雪|晴|曇|雨|雪)/)
    ?? raceDataHtml.match(/Icon_Weather[^>]*aria-label=["'](小雨|小雪|晴|曇|雨|雪)["']/);
  const weatherForecast = /<span>\s*予報\s*<\/span>/.test(raceDataHtml);
  const goingRaw = conditionText.match(/馬場\s*[:：]\s*(稍重|不良|良|重)/)?.[1]
    ?? raceDataHtml.match(/class=["']Item0[34]["'][^>]*>\s*(稍重|不良|良|重|稍|不)\s*</)?.[1] ?? null;
  const goingMatch = goingRaw ? [goingRaw, goingRaw === "稍" ? "稍重" : goingRaw === "不" ? "不良" : goingRaw] : null;

  const raceNumber = Number(raceId.slice(-2));
  const date = dateMatch
    ? dateMatch[1] + "-" + dateMatch[2].padStart(2, "0") + "-" + dateMatch[3].padStart(2, "0")
    : null;
  const meeting = meetingMatch ? Number(meetingMatch[1]) : null;
  const day = meetingMatch ? Number(meetingMatch[3]) : null;
  const venue = meetingMatch?.[2] ?? null;

  const surface = trackMatch
    ? trackMatch[1] === "ダ" || trackMatch[1] === "ダート"
      ? "ダート"
      : trackMatch[1] === "障" || trackMatch[1] === "障害"
        ? "障害"
        : "芝"
    : null;
  const distanceMeters = trackMatch ? Number(trackMatch[2]) : null;
  const direction = directionMatch?.[1] ?? null;
  const courseLayout = layoutMatch?.[1] === "外"
    ? "外回り"
    : layoutMatch?.[1] === "内"
      ? "内回り"
      : null;
  const startTime = startMatch?.[1] ?? null;
  const weather = weatherMatch?.[1] ?? null;
  const going = goingMatch?.[1] ?? null;

  const raceData2 = cleanText(html.match(/<div\b(?=[^>]*\bclass=["'][^"']*\bRaceData02\b)[^>]*>([\s\S]*?)<\/div>/i)?.[1] ?? "");
  return {
    raceClass: raceClassOf(headingText + " " + titleText, raceData2),
    raceName: raceNameMatch?.[1]?.trim() ?? null,
    date,
    venue,
    meeting,
    day,
    raceNumber: Number.isFinite(raceNumber) ? raceNumber : null,
    startTime,
    surface,
    distanceMeters,
    direction,
    courseLayout,
    courseLetter: courseLetterMatch?.[1] ?? null,
    weather,
    weatherForecast,
    going,
    fieldStatus: {
      raceName: raceNameMatch ? "available" : "unavailable",
      date: date ? "available" : "unavailable",
      venue: venue ? "available" : "unavailable",
      meeting: meeting !== null ? "available" : "unavailable",
      day: day !== null ? "available" : "unavailable",
      raceNumber: Number.isFinite(raceNumber) ? "available" : "unavailable",
      startTime: startTime ? "available" : "unavailable",
      surface: surface ? "available" : "unavailable",
      distanceMeters: distanceMeters !== null ? "available" : "unavailable",
      direction: direction ? "available" : "unavailable",
      courseLayout: courseLayout ? "available" : "not_applicable_or_unpublished",
      courseLetter: surface && surface !== "芝" ? "not_applicable" : courseLetterMatch ? "available" : "not_found",
      weather: weather ? "available" : "not_published_or_not_found",
      going: going ? "available" : "not_published_or_not_found",
    },
  };
}
function extractHorses(html: string): HorseEntry[] {
  const horses: HorseEntry[] = [];
  const seen = new Set<string>();
  const anchorRegex = /<a\b([^>]*)>([\s\S]*?)<\/a>/gi;
  let anchorMatch: RegExpExecArray | null;

  while ((anchorMatch = anchorRegex.exec(html)) !== null) {
    const hrefMatch = anchorMatch[1].match(/\bhref\s*=\s*(["'])(.*?)\1/i);
    if (!hrefMatch) continue;

    const href = decodeEntities(hrefMatch[2]);
    const idMatch =
      href.match(/horse_id=(\d+)/i) || href.match(/\/horse\/(\d+)/i);
    if (!idMatch) continue;

    const horseId = idMatch[1];
    if (seen.has(horseId)) continue;

    const name = cleanText(anchorMatch[2]);
    if (!name) continue;

    seen.add(horseId);
    horses.push({ horseId, name });
  }

  return horses;
}

function parseNumber(value: string | undefined): number | null {
  if (!value) return null;
  const normalized = value.replace(/,/g, "").trim();
  if (!/^\d+(\.\d+)?$/.test(normalized)) return null;
  const parsed = Number(normalized);
  return Number.isFinite(parsed) ? parsed : null;
}

function extractRaceEntries(html: string): RaceEntry[] {
  const tableRegex = /<table\b[^>]*>([\s\S]*?)<\/table>/gi;
  const rowRegex = /<tr\b([^>]*)>([\s\S]*?)<\/tr>/gi;
  const cellRegex = /<t[dh]\b[^>]*>([\s\S]*?)<\/t[dh]>/gi;
  let tableMatch: RegExpExecArray | null;

  while ((tableMatch = tableRegex.exec(html)) !== null) {
    const rawRows: string[][] = [];
    const rowAttrs: string[] = [];
    let rowMatch: RegExpExecArray | null;
    rowRegex.lastIndex = 0;

    while ((rowMatch = rowRegex.exec(tableMatch[1])) !== null) {
      const cells: string[] = [];
      let cellMatch: RegExpExecArray | null;
      cellRegex.lastIndex = 0;

      while ((cellMatch = cellRegex.exec(rowMatch[2])) !== null) {
        cells.push(cellMatch[1]);
      }
      if (cells.length) { rawRows.push(cells); rowAttrs.push(rowMatch[1]); }
    }

    const headerIndex = rawRows.findIndex((cells) => {
      const text = cells.map(cleanText).join(" ");
      return text.includes("馬名") && text.includes("性齢") &&
        text.includes("斤量") && text.includes("騎手");
    });
    if (headerIndex < 0) continue;

    const headers = rawRows[headerIndex].map(cleanText);
    const indexOf = (label: string) =>
      headers.findIndex((header) => header.includes(label));

    const indexes = {
      frame: indexOf("枠"),
      number: indexOf("馬番"),
      name: indexOf("馬名"),
      sexAge: indexOf("性齢"),
      carriedWeight: indexOf("斤量"),
      jockey: indexOf("騎手"),
      trainer: indexOf("厩舎"),
      bodyWeight: indexOf("馬体重"),
      popularity: indexOf("人気"),
    };

    const entries: RaceEntry[] = [];

    for (const [rowIndex, rawCells] of rawRows.entries()) {
      if (rowIndex <= headerIndex) continue;
      const rowHtml = rawCells.join(" ");
      const horseIdMatch =
        rowHtml.match(/\/horse\/(\d+)/i) ||
        rowHtml.match(/horse_id=(\d+)/i);
      if (!horseIdMatch) continue;

      const values = rawCells.map(cleanText);
      // 出走取消・競走除外：行の class に Cancel が付くか、セルに「取消」「除外」と書かれる
      const scratched = /cancel/i.test(rowAttrs[rowIndex] ?? "") || values.some((v) => v === "取消" || v === "除外");
      const get = (index: number) =>
        index >= 0 ? values[index] ?? "" : "";

      const horseCell = indexes.name >= 0 ? rawCells[indexes.name] : "";
      const horseAnchor = horseCell.match(/<a\b[^>]*>([\s\S]*?)<\/a>/i);
      const horseName = horseAnchor
        ? cleanText(horseAnchor[1])
        : get(indexes.name);

      const frameText = get(indexes.frame);
      const numberText = get(indexes.number);
      const sexAgeText = get(indexes.sexAge);
      const carriedText = get(indexes.carriedWeight);
      const jockeyText = get(indexes.jockey);
      const bodyText = get(indexes.bodyWeight);
      const popularityText = get(indexes.popularity);

      const bodyMatch = bodyText.match(/(\d{3})\s*\(([+-]?\d+)\)/);
      // 「464」だけ、または初出走の「464(初出走)」など増減が数でないもの（2026-11-22 まで新馬は馬体重なし扱いで、性別の中央値で計算していた）
      const bodyOnlyMatch = bodyText.match(/^\s*(\d{3})\s*(?:\([^)0-9+-]*\))?\s*$/);
      const sexAge = /^[牡牝セ]\s*\d+$/.test(sexAgeText)
        ? sexAgeText.replace(/\s+/g, "")
        : null;
      const jockey = jockeyText && !/^(○○|--|-)$/u.test(jockeyText)
        ? jockeyText
        : null;

      entries.push({
        horseId: horseIdMatch[1],
        horseName,
        scratched,
        frameNumber: parseNumber(frameText),
        horseNumber: parseNumber(numberText),
        sexAge,
        carriedWeight: parseNumber(carriedText),
        jockey,
        trainer: get(indexes.trainer) || null,
        bodyWeight: bodyMatch
          ? Number(bodyMatch[1])
          : bodyOnlyMatch
            ? Number(bodyOnlyMatch[1])
            : null,
        bodyWeightChange: bodyMatch ? Number(bodyMatch[2]) : null,
        popularity: parseNumber(popularityText),
        oddsKey: (() => { const m = rowHtml.match(/id=["']odds-1_(\d+)["']/); return m ? Number(m[1]) : null; })(),
        fieldStatus: {
          frameNumber: parseNumber(frameText) === null ? "not_published" : "available",
          horseNumber: parseNumber(numberText) === null ? "not_published" : "available",
          sexAge: sexAge ? "available" : "unavailable",
          carriedWeight: parseNumber(carriedText) === null ? "unavailable" : "available",
          jockey: jockey ? "available" : "not_confirmed",
          bodyWeight: bodyMatch || bodyOnlyMatch ? "available" : "not_published",
          popularity: parseNumber(popularityText) === null ? "not_published" : "available",
        },
      });
    }

    if (entries.length > 0) return entries;
  }

  return [];
}

function extractProfile(tables: string[][][]) {
  const table = tables.find((candidate) =>
    candidate.some((row) =>
      ["父", "母", "調教師"].some((label) => row[0]?.includes(label))
    ) && candidate.length >= 5
  );

  if (!table) return null;

  return table
    .filter((row) => row.length >= 2 && row[0] && row[1])
    .map((row) => ({
      label: row[0],
      value: row.slice(1).join(" "),
    }));
}

const EXCLUDED_HISTORY_COLUMNS = [
  /映像/,
  /オッズ/,
  /馬場指数/,
  /タイム指数/,
  /スタート指数/,
  /上がり指数/,
  /追走指数/,
  /厩舎コメント/,
  /備考/,
  /賞金/,
];

function extractPerformances(tables: string[][][]) {
  const table = tables.find((candidate) =>
    candidate[0]?.includes("レース名") &&
    candidate[0]?.includes("着順") &&
    candidate[0]?.includes("騎手")
  );

  if (!table || table.length < 2) return null;

  const sourceHeaders = table[0];
  const visibleColumns = sourceHeaders
    .map((header, index) => ({ header, index }))
    .filter(({ header }) => {
      const normalized = header.replace(/\s+/g, "");
      return !EXCLUDED_HISTORY_COLUMNS.some((pattern) => pattern.test(normalized));
    });
  const headers = visibleColumns.map(({ header }) => header);
  const rows = table.slice(1);

  return {
    headers,
    countOnPage: table.length - 1,
    rows: rows.map((cells) => ({
      values: Object.fromEntries(
        visibleColumns.map(({ header, index }) => [header, cells[index] ?? ""])
      ),
    })),
  };
}


async function fetchAllPerformances(
  horseId: string,
  forceRefresh: boolean
): Promise<{ performance: ReturnType<typeof extractPerformances>; pagesFetched: number; failedPages: number[] }> {
  const baseUrl = `https://db.netkeiba.com/horse/result/${horseId}/`;
  const firstPage = await fetchHtml(baseUrl, 604800, forceRefresh, "euc-jp");
  if (!firstPage.ok) return { performance: null, pagesFetched: 0, failedPages: [1] };

  const firstPerformance = extractPerformances(extractTables(firstPage.html));
  if (!firstPerformance) return { performance: null, pagesFetched: 1, failedPages: [] };

  const pageNumbers = new Set<number>();
  const pageLinkRegex = /[?&]page=(\d+)/gi;
  let pageMatch: RegExpExecArray | null;
  while ((pageMatch = pageLinkRegex.exec(firstPage.html)) !== null) {
    const pageNumber = Number(pageMatch[1]);
    if (Number.isInteger(pageNumber) && pageNumber > 1 && pageNumber <= 50) pageNumbers.add(pageNumber);
  }

  const rows = [...firstPerformance.rows];
  const failedPages: number[] = [];
  let pagesFetched = 1;
  const pagesToFetch = pageNumbers.size
    ? [...pageNumbers].sort((a, b) => a - b)
    : firstPerformance.rows.length >= 20
      ? Array.from({ length: 49 }, (_, index) => index + 2)
      : [];

  for (const pageNumber of pagesToFetch) {
    const page = await fetchHtml(baseUrl + `?page=${pageNumber}`, 604800, forceRefresh, "euc-jp");
    if (!page.ok) {
      failedPages.push(pageNumber);
      if (!pageNumbers.size) break;
      continue;
    }
    const parsed = extractPerformances(extractTables(page.html));
    if (!parsed || parsed.rows.length === 0) {
      if (!pageNumbers.size) break;
      continue;
    }
    pagesFetched++;
    // 1ページ目に全戦績が載っている場合、2ページ目以降も同じ内容が返る。新しい走がなければ打ち切る
    const known = new Set(rows.map((row) => JSON.stringify(row.values)));
    const fresh = parsed.rows.filter((row) => !known.has(JSON.stringify(row.values)));
    if (fresh.length === 0) break;
    rows.push(...fresh);
  }

  const uniqueRows = new Map<string, { values: Record<string, string> }>();
  for (const row of rows) uniqueRows.set(JSON.stringify(row.values), row);
  return {
    performance: {
      headers: firstPerformance.headers,
      countOnPage: uniqueRows.size,
      rows: [...uniqueRows.values()],
    },
    pagesFetched,
    failedPages,
  };
}

type PerformanceStats = {
  parsedStarts: number; wins: number; top3: number; averageFinish: number | null;
  gradedStarts: number; gradedWins: number;
  sameVenueStarts: number; sameVenueTop3: number;
  sameVenueSurfaceStarts: number; sameVenueSurfaceTop3: number;
  similarDistanceStarts: number; similarDistanceTop3: number;
  sameGoingStarts: number; sameGoingTop3: number;
  positionStarts: number; positionGainTotal: number;
  positionForwardStarts: number; positionForwardTop3: number;
  goingUsed: string | null;
  top2: number; gradedTop2: number; gradedTop3: number;
  sameVenueWins: number; sameVenueTop2: number; sameVenueSurfaceWins: number; sameVenueSurfaceTop2: number;
  similarDistanceWins: number; similarDistanceTop2: number; sameGoingWins: number; sameGoingTop2: number;
};

// 学習済みモデルと特徴量用データ（scripts/ml/train.py が data/ml に書き出す）
type StackAsset = StackModel & { adopted: boolean; validation?: Record<string, unknown> };
type HeavyAdjustment = {
  beta: number; runners: number; goings?: string[]; chosen?: string; mle_beta?: number; years_improved?: number;
  per_year?: Record<string, unknown>;
};
// 種牡馬の成績（2026-11-25、表示を速くする案1）：まず取得済みの data/pedigree/stallions.json（学習と同じ元。週1回の再学習で新しく走った馬の祖先を足す）から読む。
// 1レースで 60〜126ページ・14〜30MB を netkeiba から取っていたのをやめるため。ファイルにない種牡馬（初めて走る外国馬の祖先など）と、
// 年が変わって使う年の成績がファイルにまだ入っていない種牡馬（stallionNeedsRefresh）だけページから取り、解析した結果をサーバーの中で7日ためる
let storedStallions: Promise<Record<string, StallionRaw>> | null = null;
const loadStoredStallions = () => (storedStallions ??= readFile(path.join(process.cwd(), "data", "pedigree", "stallions.json"), "utf8")
  .then((t) => JSON.parse(t) as Record<string, StallionRaw>).catch((): Record<string, StallionRaw> => ({})));
const liveStallions = new Map<string, { raw: StallionRaw; at: number }>();
const LIVE_STALLION_MS = 7 * 24 * 3600 * 1000;

let mlAssets: Promise<{
  model: Model; store: Store; metrics: Record<string, unknown>; stack: StackAsset | null; heavy: HeavyAdjustment | null;
  winModel: Model | null; top2Model: Model | null; valueModel: unknown; marksEvidence: unknown; betsEvidence: unknown; longshotFeatures: string[] | null;
  calibration: { top3: number; top2: number; win: number } | null;
  betsSkipClasses: string[] | null; betsExtras: unknown;
} | null> | null = null;
function loadMlAssets() {
  mlAssets ??= (async () => {
    try {
      const dir = path.join(process.cwd(), "data", "ml");
      const [model, store, metrics] = await Promise.all(
        ["model.json", "store.json", "metrics.json"].map(async (name) => JSON.parse(await readFile(path.join(dir, name), "utf8")))
      );
      // 2段目（AI確率 + 血統）。未作成なら1段目だけで予測する
      const stack = await readFile(path.join(dir, "stack.json"), "utf8").then((text) => JSON.parse(text) as StackAsset).catch(() => null);
      // 道悪未経験 × 血統の道悪倍率 の補正（scripts/ml/train_stack.py が検証して書き出す）
      const heavy = await readFile(path.join(dir, "heavy_adjustment.json"), "utf8").then((text) => JSON.parse(text) as HeavyAdjustment).catch(() => null);
      // 勝率（1着）・連対率（2着以内）のモデル（scripts/ml/train.py が書き出す）。なければ出さない
      // 画面の目安（妙味あり・妥当・割高）の係数（scripts/ml/backtest_value.py が書き出す）。なければ目安は出さない
      const valueModel = await readFile(path.join(dir, "value_combine.json"), "utf8").then((text) => JSON.parse(text) as unknown).catch(() => null);
      // AI印の過去の成績（scripts/ml/backtest_marks.py。▲は決め方 B）
      const marksEvidence = await readFile(path.join(dir, "marks_backtest.json"), "utf8").then((text) => {
        const json = JSON.parse(text) as { years: Record<string, Record<string, unknown>> };
        return Object.fromEntries(Object.entries(json.years).map(([y, v]) => [y, { marks: v.triangle_B, popLe3: v.pop_le3 }]));
      }).catch(() => null);
      // AI印から作る買い目の過去の成績（scripts/ml/backtest_mark_bets.py の「合わせた勝率｜確率上位 N点」。券種ごとの点数は app/lib/markBets.ts）
      // 2026-11-24 から：backtest_bets2.py の「案1・ハービル（決まった点数）」（強さの逆算）。見送りは新馬だけ（案3）、☆の買い目（案5）も
      const bets2 = await readFile(path.join(dir, "bets2_backtest.json"), "utf8").then((text) => JSON.parse(text) as {
        years: Record<string, Record<string, unknown>> }).catch(() => null);
      const bets2Ready = !!bets2 && Object.values(bets2.years).every((v) => v["案1・ハービル（決まった点数）"]);
      // 前の年で決めたクラスを見送る形は3年中1年で悪化したので、新馬だけ（依頼者の判断）
      const betsSkipClasses = ["新馬"];
      const betsExtras = bets2Ready ? Object.fromEntries(Object.entries(bets2!.years).map(([y, v]) => [y, v["案5"]])) : null;
      const betsEvidence = bets2Ready
        ? Object.fromEntries(Object.entries(bets2!.years).map(([y, v]) => [y, v["案1・ハービル（決まった点数）"]]))
        : await readFile(path.join(dir, "mark_bets_backtest.json"), "utf8").then((text) => {
        const json = JSON.parse(text) as { years: Record<string, { rules: Record<string, Record<string, unknown>> }> };
        const points: Record<string, number> = { ワイド: 3, 馬連: 3, 馬単: 3, "3連複": 5, "3連単": 6 };
        return Object.fromEntries(Object.entries(json.years).map(([y, v]) => [y, Object.fromEntries(Object.entries(points)
          .map(([t, n]) => [t, v.rules[`合わせた勝率｜確率上位 ${n}点`]?.[t] ?? null]))]));
      }).catch(() => null);
      const [winModel, top2Model] = await Promise.all(["model_win.json", "model_top2.json"].map((name) =>
        readFile(path.join(dir, name), "utf8").then((text) => JSON.parse(text) as Model).catch(() => null)));
      // 人気薄専用の2段目（AI印の☆。scripts/ml/train_longshot.py）。入力の並びだけ見る（推論は /api/longshot）
      // メリハリの補正の T（案 10。scripts/ml/backtest_value.py の prepare が一番新しい年で決めて calibration.json に書く）。なければ 1
      const calibration = await readFile(path.join(dir, "calibration.json"), "utf8")
        .then((text) => JSON.parse(text) as { top3: number; top2: number; win: number }).catch(() => null);
      const longshotFeatures = await readFile(path.join(dir, "model_longshot.json"), "utf8")
        .then((text) => (JSON.parse(text) as { features: string[] }).features).catch(() => null);
      // 外国馬の扱い（2026-11-26）：短期免許の騎手の JRA での通算（scripts/ml/foreign_jockeys.py）・レーティングの初めの値の式（overseas_eval.py）
      const ov = path.join(process.cwd(), "data", "overseas");
      const [jockeyCareer, ratingInit] = await Promise.all(["jockey_career.json", "rating_init.json"].map((name) =>
        readFile(path.join(ov, name), "utf8").then((text) => JSON.parse(text)).catch(() => null)));
      (store as Store).jockeyCareer = jockeyCareer?.jockeys ?? undefined;
      (store as Store).ratingInit = ratingInit?.coef ?? null;
      (store as Store).foreignTrainingRuns = jockeyCareer?.foreign_training_runs ?? undefined;
      return { model, store, metrics, stack, heavy, winModel, top2Model, valueModel, marksEvidence, betsEvidence, betsSkipClasses, betsExtras, longshotFeatures, calibration };
    } catch {
      mlAssets = null; // 次のリクエストで再試行
      return null;
    }
  })();
  return mlAssets;
}

// コース別・枠番別の成績（scripts/ml/course_waku.py が書き出す。表示用）
type CourseWaku = { years: string[]; courses: Record<string, Record<string, [number, number]>> };
let courseWakuAsset: Promise<CourseWaku | null> | null = null;
function loadCourseWaku() {
  courseWakuAsset ??= readFile(path.join(process.cwd(), "data", "ml", "course_waku.json"), "utf8")
    .then((text) => JSON.parse(text) as CourseWaku)
    .catch(() => { courseWakuAsset = null; return null; });
  return courseWakuAsset;
}

function extractFirstPrize(html: string): number | null {
  const m = cleanText(html).match(/本賞金\s*[:：]\s*(\d+(?:\.\d+)?)/); // 「15000,6000,…万円」の先頭 = 1着
  return m ? Number(m[1]) : null; // 万円
}

const GOING_CODE: Record<string, number> = { "良": 0, "稍重": 1, "重": 2, "不良": 3 };


function summarizePerformances(
  allRows: { values: Record<string, string> }[],
  raceConditions: RaceConditions
): PerformanceStats {
  // 予測するレースより前の走だけを数える（過去のレースを開いたとき、そのレース自身の結果を含めない）
  const raceDate = raceConditions.date?.replace(/-/g, "/") ?? null;
  const rows = raceDate ? allRows.filter((row) => (row.values["日付"] ?? "") < raceDate) : allRows;
  // 馬場が未発表のときは「良」として数える（AI予測と同じ扱い）
  const conditions = { ...raceConditions, going: raceConditions.going ?? "良" };
  const stats: PerformanceStats = {
    parsedStarts: 0, wins: 0, top3: 0, averageFinish: null,
    gradedStarts: 0, gradedWins: 0, sameVenueStarts: 0, sameVenueTop3: 0,
    sameVenueSurfaceStarts: 0, sameVenueSurfaceTop3: 0,
    similarDistanceStarts: 0, similarDistanceTop3: 0, sameGoingStarts: 0, sameGoingTop3: 0,
    positionStarts: 0, positionGainTotal: 0, positionForwardStarts: 0, positionForwardTop3: 0,
    goingUsed: conditions.going, top2: 0, gradedTop2: 0, gradedTop3: 0,
    sameVenueWins: 0, sameVenueTop2: 0, sameVenueSurfaceWins: 0, sameVenueSurfaceTop2: 0,
    similarDistanceWins: 0, similarDistanceTop2: 0, sameGoingWins: 0, sameGoingTop2: 0,
  };
  const venueNames = ["札幌", "函館", "福島", "新潟", "東京", "中山", "中京", "京都", "阪神", "小倉"];
  const finishes: number[] = [];
  for (const row of rows) {
    const values = row.values;
    const findValue = (patterns: RegExp[]): string => {
      const entry = Object.entries(values).find(([key]) => {
        const normalized = key.replace(/\s+/g, "");
        return patterns.some((pattern) => pattern.test(normalized));
      });
      return entry?.[1] ?? "";
    };
    const finishText = findValue([/^着順$/]);
    const finishMatch = finishText.match(/^\s*(\d{1,2})/);
    const finish = finishMatch ? Number(finishMatch[1]) : null;
    if (finish !== null && finish > 0) {
      finishes.push(finish);
      if (finish === 1) stats.wins++;
      if (finish <= 3) stats.top3++;
      if (finish <= 2) stats.top2++;
    }
    const raceTitle = findValue([/^レース名$/]);
    if (/\((?:J・)?G(?:I{1,3}|[1-3])\)|G[1-3]|Ｇ[１２３]|GⅠ|GⅡ|GⅢ|重賞/.test(raceTitle)) {
      stats.gradedStarts++;
      if (finish === 1) stats.gradedWins++;
      if (finish !== null && finish <= 2) stats.gradedTop2++;
      if (finish !== null && finish <= 3) stats.gradedTop3++;
    }
    const venueText = findValue([/^開催$/]);
    const pastVenue = venueNames.find((name) => venueText.includes(name));
    if (conditions.venue && pastVenue === conditions.venue && finish !== null) {
      stats.sameVenueStarts++;
      if (finish <= 3) stats.sameVenueTop3++;
      if (finish <= 2) stats.sameVenueTop2++;
      if (finish === 1) stats.sameVenueWins++;
    }

    const passageText = findValue([/^通過$/, /通過順位/]);
    const passagePositions = passageText.match(/\d+/g)?.map(Number) ?? [];
    const fieldSizeText = findValue([/^頭数$/]);
    const fieldSizeMatch = fieldSizeText.match(/\d+/);
    const fieldSize = fieldSizeMatch ? Number(fieldSizeMatch[0]) : null;
    const finalCornerPosition = passagePositions.length
      ? passagePositions[passagePositions.length - 1]
      : null;
    if (finish !== null && finalCornerPosition !== null && finalCornerPosition > 0) {
      stats.positionStarts++;
      stats.positionGainTotal += finalCornerPosition - finish;
      if (fieldSize !== null && fieldSize > 0 &&
        finalCornerPosition <= Math.ceil(fieldSize / 3)) {
        stats.positionForwardStarts++;
        if (finish <= 3) stats.positionForwardTop3++;
      }
    }

    const distanceText = findValue([/^距離$/]);
    const distanceMatch = distanceText.match(/(芝|ダート|ダ|障害|障)?\s*(\d{3,4})\s*m?/i);
    if (distanceMatch && conditions.distanceMeters !== null) {
      const pastSurface = distanceMatch[1] === "ダ" || distanceMatch[1] === "ダート" ? "ダート"
        : distanceMatch[1] === "障" || distanceMatch[1] === "障害" ? "障害"
        : distanceMatch[1] === "芝" ? "芝" : null;
      const pastDistance = Number(distanceMatch[2]);
      const surfaceMatches = !conditions.surface || !pastSurface || conditions.surface === pastSurface;
      if (conditions.venue && pastVenue === conditions.venue &&
        conditions.surface && pastSurface === conditions.surface && finish !== null) {
        stats.sameVenueSurfaceStarts++;
        if (finish <= 3) stats.sameVenueSurfaceTop3++;
      if (finish <= 2) stats.sameVenueSurfaceTop2++;
      if (finish === 1) stats.sameVenueSurfaceWins++;
      }
      if (surfaceMatches && Math.abs(pastDistance - conditions.distanceMeters) <= 200 && finish !== null) {
        stats.similarDistanceStarts++;
        if (finish <= 3) stats.similarDistanceTop3++;
      if (finish <= 2) stats.similarDistanceTop2++;
      if (finish === 1) stats.similarDistanceWins++;
      }
    }
    const goingText = findValue([/^馬場$/, /^馬場状態$/, /^状態$/]);
    const normalizeGoing = (value: string) => {
      const normalized = value.replace(/\s+/g, "");
      if (normalized.includes("不良")) return "不良";
      if (normalized.includes("稍")) return "稍重";
      if (normalized.includes("重")) return "重";
      if (normalized.includes("良")) return "良";
      return "";
    };
    if (conditions.going && normalizeGoing(goingText) === normalizeGoing(conditions.going) && finish !== null) {
      stats.sameGoingStarts++;
      if (finish <= 3) stats.sameGoingTop3++;
      if (finish <= 2) stats.sameGoingTop2++;
      if (finish === 1) stats.sameGoingWins++;
    }
  }
  stats.parsedStarts = finishes.length;
  stats.averageFinish = finishes.length
    ? Math.round((finishes.reduce((sum, value) => sum + value, 0) / finishes.length) * 10) / 10
    : null;
  return stats;
}

export async function GET(request: Request, context: { params: Promise<{ raceId: string }> }) {
  return fetchStats.run({}, () => getRace(request, context));
}

async function getRace(
  _request: Request,
  context: { params: Promise<{ raceId: string }> }
) {
  const startedAt = Date.now();
  // 表示時間の内訳（2026-11-25）：ページ取得・特徴量・予測・SHAP・それ以外。Server-Timing の見出しとログに出す
  const timing: Record<string, number> = { fetch: 0, assets: 0, features: 0, predict: 0, shap: 0 };
  const phase: Record<string, number> = {}; // ページ取得の中の段階の待ち時間（出馬表と調教・JRA の馬場・各馬のページ）
  const timed = <T,>(name: keyof typeof timing, fn: () => T): T => {
    const t0 = performance.now();
    try { return fn(); } finally { timing[name] += performance.now() - t0; }
  };
  const t0All = performance.now();
  const { raceId } = await context.params;
  const searchParams = new URL(_request.url).searchParams;
  const summaryOnly = searchParams.get("summary") === "1";
  const forceRefresh = searchParams.get("refresh") === "1";
  // weights=1：出馬表だけ取り直す（オッズの取得のあとに、発表された馬体重を予想に入れるため。ほかのページはためた分を使う）
  const refreshEntries = forceRefresh || searchParams.get("weights") === "1";

  if (!/^\d{12}$/.test(raceId)) {
    return NextResponse.json(
      { ok: false, error: "raceId must be a 12-digit JRA race ID." },
      { status: 400 }
    );
  }

  try {
    const raceUrl =
      `https://race.netkeiba.com/race/shutuba.html?race_id=${raceId}`;
    const tRace = performance.now();
    const [race, trainingPage] = await Promise.all([
      fetchHtml(raceUrl, 3600, refreshEntries),
      fetchHtml(`https://race.netkeiba.com/race/oikiri.html?race_id=${raceId}`, 3600, forceRefresh).catch(() => null),
    ]);
    const training = trainingPage?.ok ? parseTrainingPage(trainingPage.html) : new Map<number, { rank: string | null; critic: string | null }>();

    if (!race.ok) {
      return NextResponse.json(
        {
          ok: false,
          step: "race_page",
          raceId,
          status: race.status,
        },
        { status: 502 }
      );
    }

    phase.race = performance.now() - tRace;
    const raceConditions = extractRaceConditions(race.html, raceId);
    // 天候・馬場は JRA の馬場情報（同じ競馬場・同じ開催日のもの）から取る（app/lib/jraBaba.ts。10分ためる）
    const tJra = performance.now();
    const jraPages = await fetchJraBaba(forceRefresh).catch(() => []);
    phase.jra = performance.now() - tJra;
    const jra = jraPages.find((b) => b && raceConditions.venue && b.venue === raceConditions.venue && b.date === raceConditions.date) ?? null;
    const jraGoing = jra ? (raceConditions.surface === "ダート" ? jra.dirt : jra.turf) : null;
    // netkeiba の天候・馬場は使わない（依頼者の指示、2026-11-21）。JRA に同じ日のページがなければ未発表
    raceConditions.weather = jra?.weather ?? null;
    raceConditions.weatherForecast = false;
    raceConditions.going = jraGoing;
    if (jra && (jraGoing || jra.weather)) raceConditions.conditionSource = "JRA" + (jra.asOf ? "（" + jra.asOf + "）" : "");
    // 馬場状態が未発表のときだけ、画面で選んだ想定の馬場（?going=重 など）を使う
    const goingParam = searchParams.get("going");
    let goingSource: "announced" | "user" | "default" = raceConditions.going ? "announced" : "default";
    if (!raceConditions.going && goingParam && goingParam in GOING_CODE) {
      raceConditions.going = goingParam;
      raceConditions.fieldStatus.going = "assumed_by_user";
      goingSource = "user";
    }
    const horses = extractHorses(race.html);
    const raceEntries = extractRaceEntries(race.html);

    if (horses.length === 0) {
      return NextResponse.json(
        {
          ok: false,
          step: "horse_extraction",
          raceId,
          raceStatus: race.status,
          error: "No horse IDs were found on the race page.",
        },
        { status: 502 }
      );
    }

    const entryByHorseId = new Map(
      raceEntries.map((entry) => [entry.horseId, entry])
    );

    const results: Record<string, unknown>[] = new Array(horses.length);
    // 父・母父・母母父の産駒成績ページ（馬場別・距離別）。同じ馬は1回だけ取得する。
    // 学習データと揃えるため、レース前年までの成績を使う
    const throughYear = raceConditions.date ? Number(raceConditions.date.slice(0, 4)) - 1 : null;
    const stallionRaws = new Map<string, Promise<StallionRaw | null>>();
    const loadStallion = async (stallionId: string, role: "sire" | "damsire", name: string | null, yearsBack = 0) => {
      let raw = stallionRaws.get(stallionId);
      if (!raw) {
        raw = (async () => {
          const st = fetchStats.getStore();
          const count = (kind: string) => { if (st) (st[kind] ??= { n: 0, ms: 0, bytes: 0 }).n += 1; };
          // 使う年（レースの前の年）の成績がそろっているか：その年より後に取ったか、その年に産駒が走っていない
          const usable = (r: StallionRaw) => !stallionNeedsRefresh(r, (throughYear ?? new Date().getFullYear() - 1) + 1);
          const stored = (await loadStoredStallions())[stallionId];
          if (stored && usable(stored)) { count("種牡馬（手元のファイル）"); return stored; }
          const cached = liveStallions.get(stallionId);
          if (!forceRefresh && cached && Date.now() - cached.at < LIVE_STALLION_MS && usable(cached.raw)) { count("種牡馬（サーバーにためた分）"); return cached.raw; }
          const url = (type: number) => `https://db.netkeiba.com/horse/sire/${stallionId}/?syusai=10&mode=1&type=${type}`;
          const [going, distance] = await Promise.all([1, 2].map((type) => fetchHtml(url(type), 604800, forceRefresh).catch(() => null)));
          const parsed = parseStallionPages(stallionId, name, {
            going: going?.ok ? going.html : null, distance: distance?.ok ? distance.html : null,
          }, new Date().toISOString().slice(0, 10));
          if (going?.ok || distance?.ok) {
            liveStallions.set(stallionId, { raw: parsed, at: Date.now() });
            if (liveStallions.size > 2000) liveStallions.delete(liveStallions.keys().next().value!);
          }
          return parsed;
        })();
        stallionRaws.set(stallionId, raw);
      }
      const value = await raw;
      if (yearsBack && throughYear === null) return null;
      return value ? stallionRecord(value, role, throughYear === null ? null : throughYear - yearsBack) : null;
    };
    const pedigreeRecords: (StallionRecord | null)[][] = horses.map(() => []);
    const pedigreeOlder: (StallionRecord | null)[][] = horses.map(() => []); // 3年前までの成績（血統の pb_recent_top3 用）
    let nextIndex = 0;

    async function worker() {
      while (true) {
        const index = nextIndex++;
        if (index >= horses.length) return;

        const horse = horses[index];

        try {
          const raceEntry = entryByHorseId.get(horse.horseId) ?? null;
          const profileUrl = `https://db.sp.netkeiba.com/horse/${horse.horseId}/`;
          const pedigreeUrl = `https://db.netkeiba.com/horse/ped/${horse.horseId}/`;
          const [profilePage, historyResult, pedigreePage] = await Promise.all([
            fetchHtml(profileUrl, 604800, forceRefresh),
            fetchAllPerformances(horse.horseId, forceRefresh),
            fetchHtml(pedigreeUrl, 2592000, forceRefresh).catch(() => null),
          ]);
          const pedigree = pedigreePage?.ok ? extractPedigree(pedigreePage.html) : null;
          pedigreeRecords[index] = await Promise.all(PEDIGREE_SOURCES.map((source) => {
            const ancestor = pedigree?.ancestors.find((item) => item.position === source.position);
            return ancestor?.horseId ? loadStallion(ancestor.horseId, source.role, ancestor.name) : Promise.resolve(null);
          }));
          pedigreeOlder[index] = await Promise.all(PEDIGREE_SOURCES.map((source) => {
            const ancestor = pedigree?.ancestors.find((item) => item.position === source.position);
            return ancestor?.horseId ? loadStallion(ancestor.horseId, source.role, ancestor.name, 3) : Promise.resolve(null);
          }));
          if (!profilePage.ok && !historyResult.performance) {
            results[index] = {
              horseId: horse.horseId, name: horse.name, raceEntry,
              status: "fetch_failed",
              pedigree,
              profileHttpStatus: profilePage.status,
              historyFailedPages: historyResult.failedPages,
            };
            continue;
          }
          const profile = profilePage.ok ? extractProfile(extractTables(profilePage.html)) : null;
          const performance = historyResult.performance;
          const performanceRows = performance?.rows ?? [];
          const performanceStats = summarizePerformances(performanceRows, raceConditions);
          results[index] = {
            horseId: horse.horseId, name: horse.name, raceEntry,
            status: profile && performance && pedigree && historyResult.failedPages.length === 0 ? "complete" : "partial",
            pedigree,
            pedigreeHttpStatus: pedigreePage?.status ?? null,
            sourceStatus: profilePage.status,
            historyPagesFetched: historyResult.pagesFetched,
            historyFailedPages: historyResult.failedPages,
            responseBytes: profilePage.bytes, profile,
            performanceCountOnPage: performance?.countOnPage ?? null,
            performanceHeaders: performance?.headers ?? [],
            performances: performance?.rows ?? [],
            historyReturned: performanceRows.length, performanceStats,
          };
        } catch (error) {
          const raceEntry = entryByHorseId.get(horse.horseId) ?? null;
          results[index] = {
            horseId: horse.horseId, name: horse.name, raceEntry,
            status: "fetch_failed",
            error: error instanceof Error ? error.message : "Unknown error",
          };
        }
      }
    }

    const workerCount = Math.min(CONCURRENCY, horses.length);
    const tHorses = performance.now();
    await Promise.all(
      Array.from({ length: workerCount }, () => worker())
    );
    phase.horses = performance.now() - tHorses;

    // --- 血統適性・調子偏差値（表示用）・AI予測 ---
    const surfaceName = raceConditions.surface === "芝" || raceConditions.surface === "ダート" ? raceConditions.surface : null;
    const aptitudes = surfaceName
      ? pedigreeAptitudes(pedigreeRecords, surfaceName, raceConditions.distanceMeters, raceConditions.going)
      : horses.map(() => null);
    const forms: FormDisplay[] = results.map((item) =>
      formFromHistory((item as { performances?: { values: Record<string, string> }[] }).performances ?? [], raceConditions.date));
    addDisplayDeviation(forms);

    timing.fetch += performance.now() - t0All; // ここまではページの取得と解析
    const tAssets = performance.now();
    const ml = await loadMlAssets();
    timing.assets += performance.now() - tAssets;
    // 出走取消・競走除外の馬は、頭数・出走馬の中での比較・予測から外す（学習データと同じ扱い）
    const activeIdx = horses.map((_, i) => i).filter((i) => !entryByHorseId.get(horses[i].horseId)?.scratched);
    let mlSummary: Record<string, unknown> = { status: "unavailable" };
    const predictions: (Record<string, unknown> | null)[] = horses.map(() => null);
    if (ml && surfaceName && raceConditions.date) {
      const goingAssumed = raceConditions.going === null;
      const venueCode = Number(raceId.slice(4, 6));
      const surfaceCode = surfaceName === "芝" ? 0 : 1;
      const course = courseInfo(venueCode, surfaceCode, raceConditions.direction, raceConditions.courseLayout === "外回り");
      // クッション値・含水率・使用コース（JRA の馬場情報。features.py の BABA_FEATURES）
      const tBaba = performance.now();
      const babaDetail = raceConditions.venue && raceConditions.date
        ? await fetchJraBabaDetail(raceConditions.venue, raceConditions.date, forceRefresh).catch(() => null) : null;
      timing.fetch += performance.now() - tBaba;
      const raceInput = {
        baba: babaDetail ? { cushion: babaDetail.cushion, moist: surfaceCode === 0 ? babaDetail.moistTurf : babaDetail.moistDirt, course: babaDetail.course } : null,
        date: raceConditions.date,
        venue: venueCode,
        surface: surfaceCode,
        ...course,
        distance: raceConditions.distanceMeters,
        going: goingAssumed ? 0 : GOING_CODE[raceConditions.going!] ?? 0,
        fieldSize: activeIdx.length,
        prize1stMan: extractFirstPrize(race.html),
      };
      const overseasInfo = await loadOverseas();
      const tFeat = performance.now();
      const rows = horses.map((horse, index) => {
        const entry = entryByHorseId.get(horse.horseId) ?? null;
        const sexAge = entry?.sexAge?.match(/^([牡牝セ])(\d+)$/);
        // 馬体重は当日の発表。学習データには欠けがないので、発表前は前走の馬体重・増減0 で計算する
        // （前走の馬体重がない馬は、学習データの初出走馬の性別ごとの中央値。scripts/ml/eval_body_weight.py で検証）
        const performances = (results[index] as { performances?: { values: Record<string, string> }[] }).performances ?? [];
        const overseas = overseasRuns(performances, horse.horseId, overseasInfo);
        const assumedBodyWeight = entry?.bodyWeight == null
          ? previousBodyWeight(performances, raceConditions.date!) ?? DEBUT_BODY_WEIGHT[sexAge?.[1] ?? ""] ?? DEBUT_BODY_WEIGHT["牡"]
          : null;
        // 外国馬（海外の厩舎）。成績表の馬名「カランダガン（Calandagan（IRE））（仏）」とつなぐ（storeHorseKey）
        const foreign = /^海外/.test(entry?.trainer ?? "");
        const horseKey = storeHorseKey(ml.store, horse.name);
        const jockeyKey = (foreign ? resolveCareerJockey(entry?.jockey, ml.store, raceInput.date) : null) ?? resolveJockey(entry?.jockey, ml.store, raceInput.date);
        const trainerKey = resolveTrainer(entry?.trainer, ml.store, raceInput.date, horseKey);
        const runner: RunnerInput = {
          horse: horseKey, jockeyKey, trainerKey, foreign,
          waku: entry?.frameNumber ?? null, umaban: entry?.horseNumber ?? null,
          weight: entry?.carriedWeight ?? null,
          age: sexAge ? Number(sexAge[2]) : null, sex: sexAge?.[1] ?? null,
          bodyWeight: entry?.bodyWeight ?? assumedBodyWeight,
          bodyWeightDiff: entry?.bodyWeight == null ? 0 : entry?.bodyWeightChange ?? null,
          extraRuns: overseas,
          pedigree: pedigreeModelValues(pedigreeRecords[index] ?? [], surfaceName, raceConditions.distanceMeters,
            raceConditions.going ?? "良", pedigreeOlder[index] ?? []),
          trainRank: TRAIN_RANK[(entry?.horseNumber != null ? training.get(entry.horseNumber)?.rank : null) ?? ""] ?? null,
          // 馬主・生産者（netkeiba の馬のプロフィール。名前は features.ts の normOwner で成績表の書き方にそろえる）
          owner: (results[index] as { profile?: { label: string; value: string }[] | null }).profile?.find((p) => p.label === "馬主")?.value ?? null,
          breeder: (results[index] as { profile?: { label: string; value: string }[] | null }).profile?.find((p) => p.label.startsWith("生産者"))?.value ?? null,
        };
        const features = runnerFeatures(ml.store, raceInput, runner);
        return { jockeyKey, trainerKey, features, runner, horseKey, foreign, estimated: [] as string[], bodyWeightAssumed: assumedBodyWeight !== null, overseasRuns: overseas.length };
      });
      // 外国馬の調教師・馬主・生産者などを、同じレースの日本馬の中央値で埋める（埋めた項目は画面で「推定」）
      fillForeignPeople(ml.store, raceInput, activeIdx.map((i) => rows[i].features), activeIdx.map((i) => rows[i].runner))
        .forEach((keys, k) => { rows[activeIdx[k]].estimated = keys; });
      addFormDeviation(activeIdx.map((i) => rows[i].features));
      addFieldFeatures(activeIdx.map((i) => rows[i].features));
      addOpponentFeatures(ml.store, raceInput, activeIdx.map((i) => rows[i].features), activeIdx.map((i) => rows[i].horseKey));
      addSimilarCondFeatures(ml.store, raceInput, activeIdx.map((i) => rows[i].features), activeIdx.map((i) => rows[i].horseKey));
      timing.features += performance.now() - tFeat;
      // メリハリの補正（案 10）：モデルの log-odds に T を掛ける。この後、合計を 3・2・1 にそろえる
      const temper = (p: number, T: number | undefined) => {
        if (!T || T === 1) return p;
        const q = Math.min(Math.max(p, 1e-9), 1 - 1e-9);
        return 1 / (1 + Math.exp(-T * Math.log(q / (1 - q))));
      };
      const baseProbs = timed("predict", () => rows.map((row) => temper(predictProbability(ml.model, row.features), ml.calibration?.top3)));
      const useStack = !!ml.stack?.adopted;
      const isHeavy = raceConditions.going === "重" || raceConditions.going === "不良";
      const stacked = baseProbs.map((p, i) => (useStack ? stackProbability(ml.stack!, stackFeatures(p, aptitudes[i], isHeavy)) : p));
      // 道悪の後からの補正は 2026-11-20 にやめた（全レースで4年とも悪化。血統の道悪倍率は本体モデルの入力 ped_heavy・ped_wet に入れた）
      // 表示用：重・不良の経験がない馬に、血統の道悪の得意・苦手を出す
      const heavyNotes = horses.map((_, i) => {
        const apt = aptitudes[i];
        const performances = (results[i] as { performances?: { values: Record<string, string> }[] }).performances ?? [];
        const inexperienced = rows[i].features.heavy_n === 0 && heavyRunsInHistory(performances, surfaceName, raceConditions.date) === 0;
        if (!inexperienced || apt?.heavyRatio == null) return null;
        const label = apt.heavyLabel;
        return {
          applied: false, show: label === "得意" || label === "苦手", label, ratio: apt.heavyRatio,
          before: stacked[i], after: stacked[i], goingKnown: !goingAssumed, beta: 0,
          going: raceConditions.going, goingAssumedByUser: goingSource === "user",
        };
      });
      // 出走馬の中で3着以内確率の合計を 3 にそろえる（学習データの検証で3年とも logloss 改善。順位は変わらない）
      const probs = [...stacked];
      const normalized = normalizeSum(activeIdx.map((i) => stacked[i]), 3);
      activeIdx.forEach((i, k) => { probs[i] = normalized[k]; });
      // 勝率・連対率。3着以内確率を超えないようにそろえる
      // 連対率・勝率も T を掛けて、出走馬の合計を 2・1 にそろえる（案 10）
      const rawTop2 = timed("predict", () => rows.map((row) => ml.top2Model ? temper(predictProbability(ml.top2Model, row.features), ml.calibration?.top2) : null));
      const baseWinProbs = timed("predict", () => rows.map((row) => ml.winModel ? temper(predictProbability(ml.winModel, row.features), ml.calibration?.win) : null)); // AI印の◎・▲（そろえる前の勝率）
      const normIn = (vals: (number | null)[], target: number) => {
        const out = [...vals];
        if (activeIdx.every((i) => vals[i] != null)) {
          const z = normalizeSum(activeIdx.map((i) => vals[i]!), target);
          activeIdx.forEach((i, k) => { out[i] = z[k]; });
        }
        return out;
      };
      const top2N = normIn(rawTop2, 2);
      const winN = normIn(baseWinProbs, 1);
      const top2Probs = rows.map((_, i) => top2N[i] != null ? Math.min(top2N[i]!, probs[i]) : null);
      const winProbs = rows.map((_, i) => winN[i] != null ? Math.min(winN[i]!, top2Probs[i] ?? probs[i]) : null);
      const order = activeIdx.map((i) => ({ p: probs[i], i })).sort((a, b) => b.p - a.p);
      // 各観点（SHAP）は、2026-11-25 から内訳を開いたときに /api/shap で計算する（レースを開くたびに全馬で約0.6秒かかっていたため）。
      // 画面は下の inputRow と probability を送る
      const longshotReady = !!ml.longshotFeatures && ml.model.features.every((name, k) => ml.longshotFeatures![k] === name);
      order.forEach(({ i }, rank) => {
        const f = rows[i].features;
        const num = (v: number) => (Number.isNaN(v) ? null : v);
        predictions[i] = {
          probability: probs[i], baseProbability: baseProbs[i], pedigreeIncluded: useStack, rank: rank + 1,
          winProbability: winProbs[i], top2Probability: top2Probs[i], baseWinProbability: baseWinProbs[i],
          heavyNote: heavyNotes[i],
          modelFormDeviation: num(f.form_dev), priorRuns: f.n_prior, overseasRuns: rows[i].overseasRuns,
          // 外国馬：学習データの外国馬の走の数（注意の表示）と、日本馬の中央値などで埋めた項目（補足の表の「推定」）
          foreign: rows[i].foreign ? { estimated: rows[i].estimated, trainingRuns: ml.store.foreignTrainingRuns ?? null, values: Object.fromEntries(
            ["elom", "elos", "jockey_n", "jockey_top3", "trainer_n", "trainer_top3", "owner_n_y", "owner_top3_y", "breeder_n_y", "breeder_top3_y"].map((k) => [k, num(f[k])])) } : null,
          training: (() => { const n = entryByHorseId.get(horses[i].horseId)?.horseNumber; return n != null ? training.get(n) ?? null : null; })(),
          jockeyName: rows[i].jockeyKey,
          pace: {
            style: runningStyle(f), posFirstMean: num(f.pos_first_mean), nigeRate: num(f.nige_rate),
            highPace: { n: f.hp_n, top3: num(f.hp_top3) }, slowPace: { n: f.sp_n, top3: num(f.sp_top3) },
            relPosRank: num(f.rel_pos_rank),
          },
          jockey: jockeyStats(ml.store, raceInput, rows[i].jockeyKey),
          trainerName: rows[i].trainerKey,
          trainer: trainerStats(ml.store, raceInput, rows[i].trainerKey),
          extras: {
            prizeRatioLast: num(f.prize_ratio_last), distChange: num(f.dist_change), surfaceChange: num(f.surface_change),
            weightChange: num(f.weight_change), jockeyChange: num(f.jockey_change),
            tidxMean5: num(f.tidx_mean5), tidxBest5: num(f.tidx_best5), wakuCourseN: f.waku_course_n, wakuCourseDiff: num(f.waku_course_diff),
          },
          // 各観点の順位（画面）に使う、モデルで効いている項目。能力指数の直近加重平均・推定上がり偏差の直近加重平均・今回の条件での能力
          ability: { ab: num(f.ab_wmean), agari: num(f.ag_dev_wmean), cond: num(f.cond_ab), early: num(f.early_gap_wmean) },
          lapType: lapType((results[i] as { performances?: { values: Record<string, string> }[] }).performances ?? [], raceConditions.date!),
          // 本体モデルの入力（model.json の features の並び）。人気薄専用の2段目（☆）と、AIスコアの内訳（/api/shap）に使う
          inputRow: ml.model.features.map((name) => { const v = f[name]; return v == null || Number.isNaN(v) ? null : v; }),
        };
      });
      const f0 = rows[activeIdx[0]]?.features;
      mlSummary = {
        status: "available", dataUntil: ml.store.lastDate, goingAssumed, goingSource, valueModel: ml.valueModel, marksEvidence: ml.marksEvidence, betsEvidence: ml.betsEvidence,
          betsSkipClasses: ml.betsSkipClasses, betsExtras: ml.betsExtras,
        longshotReady, featureCount: ml.model.features.length,
        bodyWeightAssumed: activeIdx.some((i) => rows[i].bodyWeightAssumed),
        // このレースの結果がすでに学習データに入っている（答えを見て学習した状態）
        raceInTraining: raceConditions.date <= ml.store.lastDate,
        pace: f0 ? {
          nigeCandidates: f0.field_nige_cands, frontRunners: f0.field_front, frontRatio: f0.field_front_ratio,
          expected: expectedPace(f0.field_nige_cands, f0.field_front_ratio),
        } : null,
        pedigreeIncluded: useStack, stackValidation: ml.stack?.validation ?? null,
        heavyAdjustment: ml.heavy ? {
          beta: ml.heavy.beta, chosen: ml.heavy.chosen ?? null, goings: ml.heavy.goings ?? null,
          runners: ml.heavy.runners, yearsImproved: ml.heavy.years_improved ?? null,
        } : null,
        prize1stMan: raceInput.prize1stMan, metrics: {
          auc: ml.metrics.auc, raceTop3Hit: ml.metrics.race_top3_hit, testYear: ml.metrics.test_year,
        },
      };
    }
    // 枠順（表示用）：同じコースでの、この枠の3着内率とコース全体の3着内率
    const courseWaku = await loadCourseWaku();
    const courseCells = courseWaku && raceConditions.venue && raceConditions.surface && raceConditions.distanceMeters
      ? courseWaku.courses[`${raceConditions.venue}-${raceConditions.surface}-${raceConditions.distanceMeters}`] ?? null : null;
    const courseTotal = courseCells ? Object.values(courseCells).reduce((sum, [n, t]) => [sum[0] + n, sum[1] + t], [0, 0]) : null;
    results.forEach((item, i) => {
      if (!item) return;
      const frame = (item as { raceEntry?: RaceEntry | null }).raceEntry?.frameNumber;
      const cell = courseCells && frame != null ? courseCells[String(frame)] : null;
      item.courseFrame = cell && courseTotal ? {
        starts: cell[0], top3: cell[1], courseStarts: courseTotal[0], courseTop3: courseTotal[1], years: courseWaku!.years,
      } : null;
      item.prediction = predictions[i];
      item.form = forms[i];
      item.pedigreeAptitude = aptitudes[i];
    });

    const completeCount = results.filter(
      (item) => (item as { status?: string })?.status === "complete"
    ).length;
    const partialCount = results.filter(
      (item) => (item as { status?: string })?.status === "partial"
    ).length;
    const failedCount = results.filter(
      (item) => (item as { status?: string })?.status === "fetch_failed"
    ).length;

    const matchedEntryCount = horses.filter((horse) =>
      entryByHorseId.has(horse.horseId)
    ).length;

    const totalMs = performance.now() - t0All;
    const otherMs = Math.max(0, totalMs - Object.values(timing).reduce((a, b) => a + b, 0));
    const serverTiming = [...Object.entries(timing).map(([k, v]) => `${k};dur=${v.toFixed(1)}`), `other;dur=${otherMs.toFixed(1)}`, `total;dur=${totalMs.toFixed(1)}`].join(", ");
    const kinds = Object.fromEntries(Object.entries(fetchStats.getStore() ?? {}).map(([k, v]) => [k, { n: v.n, ms: Math.round(v.ms), kb: Math.round(v.bytes / 1024) }]));
    console.log(`[timing] race ${raceId} ${JSON.stringify({ ...Object.fromEntries(Object.entries(timing).map(([k, v]) => [k, Math.round(v)])), other: Math.round(otherMs), total: Math.round(totalMs), horses: horses.length, phase: Object.fromEntries(Object.entries(phase).map(([k, v]) => [k, Math.round(v)])), fetches: kinds })}`);
    return NextResponse.json(
      {
        ok: failedCount === 0 && partialCount === 0 && matchedEntryCount === horses.length,
        raceId,
        raceConditions,
        racePageStatus: race.status,
        racePageBytes: race.bytes,
        ml: mlSummary,
        horseCount: horses.length,
        raceEntryCount: raceEntries.length,
        matchedEntryCount,
        concurrency: CONCURRENCY,
        completeCount,
        partialCount,
        failedCount,
        ...(summaryOnly
          ? {
              sampleRaceEntries: results.slice(0, 3).map((item) => {
                const result = item as {
                  name?: string;
                  status?: string;
                  raceEntry?: RaceEntry | null;
                };
                return {
                  name: result.name ?? null,
                  status: result.status ?? null,
                  raceEntry: result.raceEntry ?? null,
                };
              }),
            }
          : { results }),
        elapsedMs: Date.now() - startedAt,
        notes: {
          frameAndHorseNumber: "Not published until the draw is announced; null is expected before then.",
          bodyWeight: "Usually published on race day; null before publication is expected.",
          jockey: "A provisional placeholder is reported as not_confirmed.",
        },
      },
      {
        headers: {
          "Content-Type": "application/json; charset=utf-8",
          "Cache-Control": "no-store",
          "Server-Timing": serverTiming,
        },
      }
    );
  } catch (error) {
    return NextResponse.json(
      {
        ok: false,
        raceId,
        error: error instanceof Error ? error.message : "Unknown error",
        elapsedMs: Date.now() - startedAt,
      },
      { status: 502 }
    );
  }
}