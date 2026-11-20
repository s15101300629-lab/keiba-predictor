import { NextResponse } from "next/server";
import { readFile } from "node:fs/promises";
import path from "node:path";
import { cleanText, decodeEntities, extractPedigree, extractTables } from "@/app/lib/netkeiba";
import { addDisplayDeviation, formFromHistory, heavyRunsInHistory, type FormDisplay } from "@/app/lib/form";
import { addFieldFeatures, addFormDeviation, expectedPace, jockeyStats, resolveTrainer, runningStyle, trainerStats, resolveJockey, runnerFeatures, type Store } from "@/app/lib/ml/features";
import { predictProbability, type Model } from "@/app/lib/ml/model";
import {
  PEDIGREE_SOURCES, parseStallionPages, pedigreeAptitudes, stackFeatures, stackProbability, stallionRecord,
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

// 馬体重の発表前に使う値（初出走馬の性別ごとの中央値。学習データ 2020年〜）
const DEBUT_BODY_WEIGHT: Record<string, number> = { "牡": 476, "牝": 448, "セ": 468 };

/** netkeiba の戦績から、レース日より前の一番新しい馬体重（「480(+2)」の 480）。計不・海外などは飛ばす */
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

async function fetchHtml(
  url: string,
  revalidateSeconds: number,
  forceRefresh: boolean,
  fallbackEncoding: "utf-8" | "euc-jp" = "utf-8"
) {
  const response = await fetch(url, {
    ...(forceRefresh
      ? { cache: "no-store" as const }
      : { cache: "force-cache" as const, next: { revalidate: revalidateSeconds } }),
    headers: { Accept: "text/html" },
    signal: AbortSignal.timeout(10000),
  });
  const buffer = new Uint8Array(await response.arrayBuffer());
  // netkeiba のページは EUC-JP から UTF-8 に移行しているため、宣言された文字コードで読む
  const encoding = detectCharset(response.headers.get("content-type"), buffer) ?? fallbackEncoding;
  return {
    status: response.status,
    ok: response.ok,
    html: new TextDecoder(encoding).decode(buffer),
    bytes: buffer.byteLength,
  };
}




type RaceConditions = {
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

  const weatherMatch = conditionText.match(/天候\s*[:：]\s*(小雨|小雪|晴|曇|雨|雪)/);
  const goingMatch = conditionText.match(/馬場\s*[:：]\s*(稍重|不良|良|重)/);

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

  return {
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
      const bodyOnlyMatch = bodyText.match(/^\s*(\d{3})\s*$/);
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
let mlAssets: Promise<{
  model: Model; store: Store; metrics: Record<string, unknown>; stack: StackAsset | null; heavy: HeavyAdjustment | null;
  winModel: Model | null; top2Model: Model | null;
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
      const [winModel, top2Model] = await Promise.all(["model_win.json", "model_top2.json"].map((name) =>
        readFile(path.join(dir, name), "utf8").then((text) => JSON.parse(text) as Model).catch(() => null)));
      return { model, store, metrics, stack, heavy, winModel, top2Model };
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

export async function GET(
  _request: Request,
  context: { params: Promise<{ raceId: string }> }
) {
  const startedAt = Date.now();
  const { raceId } = await context.params;
  const searchParams = new URL(_request.url).searchParams;
  const summaryOnly = searchParams.get("summary") === "1";
  const forceRefresh = searchParams.get("refresh") === "1";

  if (!/^\d{12}$/.test(raceId)) {
    return NextResponse.json(
      { ok: false, error: "raceId must be a 12-digit JRA race ID." },
      { status: 400 }
    );
  }

  try {
    const raceUrl =
      `https://race.netkeiba.com/race/shutuba.html?race_id=${raceId}`;
    const race = await fetchHtml(raceUrl, 3600, forceRefresh);

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

    const raceConditions = extractRaceConditions(race.html, raceId);
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
    const loadStallion = async (stallionId: string, role: "sire" | "damsire", name: string | null) => {
      let raw = stallionRaws.get(stallionId);
      if (!raw) {
        raw = (async () => {
          const url = (type: number) => `https://db.netkeiba.com/horse/sire/${stallionId}/?syusai=10&mode=1&type=${type}`;
          const [going, distance] = await Promise.all([1, 2].map((type) => fetchHtml(url(type), 604800, forceRefresh).catch(() => null)));
          return parseStallionPages(stallionId, name, {
            going: going?.ok ? going.html : null, distance: distance?.ok ? distance.html : null,
          }, new Date().toISOString().slice(0, 10));
        })();
        stallionRaws.set(stallionId, raw);
      }
      const value = await raw;
      return value ? stallionRecord(value, role, throughYear) : null;
    };
    const pedigreeRecords: (StallionRecord | null)[][] = horses.map(() => []);
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
    await Promise.all(
      Array.from({ length: workerCount }, () => worker())
    );

    // --- 血統適性・調子偏差値（表示用）・AI予測 ---
    const surfaceName = raceConditions.surface === "芝" || raceConditions.surface === "ダート" ? raceConditions.surface : null;
    const aptitudes = surfaceName
      ? pedigreeAptitudes(pedigreeRecords, surfaceName, raceConditions.distanceMeters, raceConditions.going)
      : horses.map(() => null);
    const forms: FormDisplay[] = results.map((item) =>
      formFromHistory((item as { performances?: { values: Record<string, string> }[] }).performances ?? [], raceConditions.date));
    addDisplayDeviation(forms);

    const ml = await loadMlAssets();
    // 出走取消・競走除外の馬は、頭数・出走馬の中での比較・予測から外す（学習データと同じ扱い）
    const activeIdx = horses.map((_, i) => i).filter((i) => !entryByHorseId.get(horses[i].horseId)?.scratched);
    let mlSummary: Record<string, unknown> = { status: "unavailable" };
    const predictions: (Record<string, unknown> | null)[] = horses.map(() => null);
    if (ml && surfaceName && raceConditions.date) {
      const goingAssumed = raceConditions.going === null;
      const raceInput = {
        date: raceConditions.date,
        venue: Number(raceId.slice(4, 6)),
        surface: surfaceName === "芝" ? 0 : 1,
        distance: raceConditions.distanceMeters,
        going: goingAssumed ? 0 : GOING_CODE[raceConditions.going!] ?? 0,
        fieldSize: activeIdx.length,
        prize1stMan: extractFirstPrize(race.html),
      };
      const rows = horses.map((horse, index) => {
        const entry = entryByHorseId.get(horse.horseId) ?? null;
        const sexAge = entry?.sexAge?.match(/^([牡牝セ])(\d+)$/);
        // 馬体重は当日の発表。学習データには欠けがないので、発表前は前走の馬体重・増減0 で計算する
        // （前走の馬体重がない馬は、学習データの初出走馬の性別ごとの中央値。scripts/ml/eval_body_weight.py で検証）
        const performances = (results[index] as { performances?: { values: Record<string, string> }[] }).performances ?? [];
        const assumedBodyWeight = entry?.bodyWeight == null
          ? previousBodyWeight(performances, raceConditions.date!) ?? DEBUT_BODY_WEIGHT[sexAge?.[1] ?? ""] ?? DEBUT_BODY_WEIGHT["牡"]
          : null;
        const jockeyKey = resolveJockey(entry?.jockey, ml.store, raceInput.date);
        const trainerKey = resolveTrainer(entry?.trainer, ml.store, raceInput.date, horse.name);
        const features = runnerFeatures(ml.store, raceInput, {
          horse: horse.name, jockeyKey, trainerKey,
          waku: entry?.frameNumber ?? null, umaban: entry?.horseNumber ?? null,
          weight: entry?.carriedWeight ?? null,
          age: sexAge ? Number(sexAge[2]) : null, sex: sexAge?.[1] ?? null,
          bodyWeight: entry?.bodyWeight ?? assumedBodyWeight,
          bodyWeightDiff: entry?.bodyWeight == null ? 0 : entry?.bodyWeightChange ?? null,
        });
        return { jockeyKey, trainerKey, features, bodyWeightAssumed: assumedBodyWeight !== null };
      });
      addFormDeviation(activeIdx.map((i) => rows[i].features));
      addFieldFeatures(activeIdx.map((i) => rows[i].features));
      const baseProbs = rows.map((row) => predictProbability(ml.model, row.features));
      const useStack = !!ml.stack?.adopted;
      const isHeavy = raceConditions.going === "重" || raceConditions.going === "不良";
      const stacked = baseProbs.map((p, i) => (useStack ? stackProbability(ml.stack!, stackFeatures(p, aptitudes[i], isHeavy)) : p));
      // 道悪未経験の馬が重・不良を走る場合: オッズ × 血統の道悪倍率^β（2段目に血統が入っているときは二重にかけない）
      const beta = ml.heavy?.beta ?? 0;
      const heavyNotes = horses.map((_, i) => {
        const apt = aptitudes[i];
        const performances = (results[i] as { performances?: { values: Record<string, string> }[] }).performances ?? [];
        const inexperienced = rows[i].features.heavy_n === 0 && heavyRunsInHistory(performances, surfaceName, raceConditions.date) === 0;
        if (!inexperienced || apt?.heavyRatio == null) return null;
        const label = apt.heavyLabel;
        // 対象の馬場: heavy_adjustment.json の goings（稍重・重・不良）
        const applies = !!raceConditions.going && (ml.heavy?.goings ?? ["重", "不良"]).includes(raceConditions.going) && !useStack && beta !== 0;
        const before = stacked[i];
        const after = applies
          ? 1 / (1 + Math.exp(-(Math.log(before / (1 - before)) + beta * Math.log(apt.heavyRatio))))
          : before;
        // 判定が「普通」の馬は、補正があっても表示はしない
        const show = label === "得意" || label === "苦手";
        return {
          applied: applies, show, label, ratio: apt.heavyRatio, before, after, goingKnown: !goingAssumed, beta,
          going: raceConditions.going, goingAssumedByUser: goingSource === "user",
        };
      });
      const probs = stacked.map((p, i) => heavyNotes[i]?.after ?? p);
      // 勝率・連対率（道悪補正はかけない）。3着以内確率を超えないようにそろえる
      const top2Probs = rows.map((row, i) => ml.top2Model ? Math.min(predictProbability(ml.top2Model, row.features), probs[i]) : null);
      const winProbs = rows.map((row, i) => ml.winModel ? Math.min(predictProbability(ml.winModel, row.features), top2Probs[i] ?? probs[i]) : null);
      const order = activeIdx.map((i) => ({ p: probs[i], i })).sort((a, b) => b.p - a.p);
      order.forEach(({ i }, rank) => {
        const f = rows[i].features;
        const num = (v: number) => (Number.isNaN(v) ? null : v);
        predictions[i] = {
          probability: probs[i], baseProbability: baseProbs[i], pedigreeIncluded: useStack, rank: rank + 1,
          winProbability: winProbs[i], top2Probability: top2Probs[i],
          heavyNote: heavyNotes[i],
          modelFormDeviation: num(f.form_dev), priorRuns: f.n_prior,
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
        };
      });
      const f0 = rows[activeIdx[0]]?.features;
      mlSummary = {
        status: "available", dataUntil: ml.store.lastDate, goingAssumed, goingSource,
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