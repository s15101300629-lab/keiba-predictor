import { NextResponse } from "next/server";
import { readFile } from "node:fs/promises";
import path from "node:path";
import { cleanText, decodeEntities, extractPedigree, extractTables } from "@/app/lib/netkeiba";
import { addDisplayDeviation, formFromHistory, heavyRunsInHistory, type FormDisplay } from "@/app/lib/form";
import { addFormDeviation, jockeyStats, resolveJockey, runnerFeatures, type Store } from "@/app/lib/ml/features";
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
  const rowRegex = /<tr\b[^>]*>([\s\S]*?)<\/tr>/gi;
  const cellRegex = /<t[dh]\b[^>]*>([\s\S]*?)<\/t[dh]>/gi;
  let tableMatch: RegExpExecArray | null;

  while ((tableMatch = tableRegex.exec(html)) !== null) {
    const rawRows: string[][] = [];
    let rowMatch: RegExpExecArray | null;
    rowRegex.lastIndex = 0;

    while ((rowMatch = rowRegex.exec(tableMatch[1])) !== null) {
      const cells: string[] = [];
      let cellMatch: RegExpExecArray | null;
      cellRegex.lastIndex = 0;

      while ((cellMatch = cellRegex.exec(rowMatch[1])) !== null) {
        cells.push(cellMatch[1]);
      }
      if (cells.length) rawRows.push(cells);
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

    for (const rawCells of rawRows.slice(headerIndex + 1)) {
      const rowHtml = rawCells.join(" ");
      const horseIdMatch =
        rowHtml.match(/\/horse\/(\d+)/i) ||
        rowHtml.match(/horse_id=(\d+)/i);
      if (!horseIdMatch) continue;

      const values = rawCells.map(cleanText);
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
    rows.push(...parsed.rows);
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
};

type RaceHistoryMetric = { starts: number; wins: number; top3: number; winRate: number; placeRate: number };
type RaceHistoryStats = {
  status: "available" | "partial" | "unavailable";
  requestedYears: number[]; fetchedYears: number[]; failedYears: number[];
  frameYears: number[]; jockeyYears: number[];
  totalEntries: number;
  frameStats: Record<string, RaceHistoryMetric>;
  jockeyStats: Record<string, RaceHistoryMetric>;
};
type RaceHistoryForHorse = {
  frame: RaceHistoryMetric | null; jockey: RaceHistoryMetric | null;
  frameYears: number[]; jockeyYears: number[];
} | null;
type JraHistoricalEntry = {
  frameNumber: number | null; horseNumber: number; horseName: string;
  jockey: string | null; finish: number | null;
};

function normalizeJockeyName(value: string | null | undefined) {
  return (value ?? "").normalize("NFKC").replace(/[\s　.・]/g, "").toLowerCase();
}


function createEmptyRaceHistoryStats(requestedYears: number[], status: RaceHistoryStats["status"]): RaceHistoryStats {
  return { status, requestedYears, fetchedYears: [], failedYears: [...requestedYears], frameYears: [], jockeyYears: [], totalEntries: 0, frameStats: {}, jockeyStats: {} };
}

function extractJraHistoricalEntries(html: string): JraHistoricalEntry[] {
  const tableRegex = /<table\b[^>]*>([\s\S]*?)<\/table>/gi;
  const rowRegex = /<tr\b[^>]*>([\s\S]*?)<\/tr>/gi;
  const cellRegex = /<(t[dh])\b([^>]*)>([\s\S]*?)<\/\1>/gi;
  let tableMatch: RegExpExecArray | null;
  const cellText = (value: string) => cleanText(
    value.replace(/<img\b[^>]*\balt\s*=\s*(["'])(.*?)\1[^>]*>/gi, " $2 ")
  );

  while ((tableMatch = tableRegex.exec(html)) !== null) {
    const rawRows: { raw: string; cells: { text: string; raw: string; attrs: string; tag: string }[] }[] = [];
    let rowMatch: RegExpExecArray | null;
    rowRegex.lastIndex = 0;
    while ((rowMatch = rowRegex.exec(tableMatch[1])) !== null) {
      const cells: { text: string; raw: string; attrs: string; tag: string }[] = [];
      let cellMatch: RegExpExecArray | null;
      cellRegex.lastIndex = 0;
      while ((cellMatch = cellRegex.exec(rowMatch[1])) !== null) {
        cells.push({ text: cellText(cellMatch[3]), raw: cellMatch[3], attrs: cellMatch[2], tag: cellMatch[1].toLowerCase() });
      }
      if (cells.length) rawRows.push({ raw: rowMatch[1], cells });
    }

    const firstDataRow = rawRows.findIndex((row) =>
      row.cells.some((cell) => cell.tag === "td") &&
      /^\d{1,2}$/.test(row.cells[0]?.text ?? "") &&
      row.cells.length >= 5
    );
    if (firstDataRow < 1) continue;

    const grid: ({ text: string; raw: string } | undefined)[][] = [];
    for (let rowIndex = 0; rowIndex < rawRows.length; rowIndex++) {
      const row = rawRows[rowIndex];
      grid[rowIndex] ??= [];
      let column = 0;
      for (const cell of row.cells) {
        while (grid[rowIndex][column] !== undefined) column++;
        const colspan = Math.max(1, Number(cell.attrs.match(/\bcolspan\s*=\s*["']?(\d+)/i)?.[1] ?? 1));
        const rowspan = Math.max(1, Number(cell.attrs.match(/\browspan\s*=\s*["']?(\d+)/i)?.[1] ?? 1));
        for (let r = rowIndex; r < Math.min(rawRows.length, rowIndex + rowspan); r++) {
          grid[r] ??= [];
          for (let c = column; c < column + colspan; c++) {
            grid[r][c] = { text: cell.text, raw: cell.raw };
          }
        }
        column += colspan;
      }
    }

    const headerRows = grid.slice(0, firstDataRow);
    const width = Math.max(...headerRows.map((row) => row.length));
    const headers = Array.from({ length: width }, (_, column) =>
      headerRows.map((row) => row[column]?.text ?? "").join("")
    );
    const findColumn = (pattern: RegExp) => headers.findIndex((value) => pattern.test(value.replace(/\s+/g, "")));
    const columns = {
      finish: findColumn(/着順/),
      frame: findColumn(/枠/),
      number: findColumn(/馬番/),
      name: findColumn(/馬名/),
      jockey: findColumn(/騎手名?/),
    };
    if (columns.finish < 0 || columns.number < 0 || columns.name < 0 || columns.jockey < 0) continue;

    const entries: JraHistoricalEntry[] = [];
    for (let rowIndex = firstDataRow; rowIndex < rawRows.length; rowIndex++) {
      const rawRow = rawRows[rowIndex];
      if (!rawRow.cells.some((cell) => cell.tag === "td") || !/^\d{1,2}$/.test(rawRow.cells[0]?.text ?? "")) continue;
      const row = grid[rowIndex] ?? [];
      const get = (index: number) => row[index];
      const horseNumber = parseNumber(get(columns.number)?.text);
      const horseName = get(columns.name)?.text ?? "";
      if (horseNumber === null || !horseName) continue;
      const frameCell = get(columns.frame);
      const frameAlt = frameCell?.raw.match(/\balt\s*=\s*(["'])(.*?)\1/i)?.[2] ?? "";
      const frameText = (frameAlt + " " + (frameCell?.text ?? "")).trim();
      const frameMatch = frameText.match(/(?:枠)?\s*(\d+)(?:\D|$)/);
      const finishMatch = (get(columns.finish)?.text ?? "").match(/^\s*(\d+)/);
      const jockey = get(columns.jockey)?.text ?? "";
      entries.push({
        frameNumber: frameMatch ? Number(frameMatch[1]) : null,
        horseNumber,
        horseName,
        jockey: jockey && !/^(?:-|--|取消|除外)$/u.test(jockey) ? jockey : null,
        finish: finishMatch ? Number(finishMatch[1]) : null,
      });
    }
    if (entries.length) return entries;
  }
  return [];
}
function toRaceHistoryMetric(value: { starts: number; wins: number; top3: number }): RaceHistoryMetric {
  return { ...value, winRate: value.starts ? value.wins / value.starts : 0, placeRate: value.starts ? value.top3 / value.starts : 0 };
}

async function fetchRaceHistoryStats(conditions: RaceConditions, forceRefresh: boolean): Promise<RaceHistoryStats | null> {
  const raceName = (conditions.raceName ?? "").replace(/[\s　]/g, "");
  if (!raceName.includes("エリザベス女王杯")) return null;
  const targetYear = conditions.date ? Number(conditions.date.slice(0, 4)) : NaN;
  if (!Number.isInteger(targetYear)) return null;
  const requestedYears = Array.from({ length: 10 }, (_, i) => targetYear - i - 1);
  const archiveUrl = "https://www.jra.go.jp/datafile/seiseki/g1/eliza/";
  const archive = await fetchHtml(archiveUrl, 86400, forceRefresh);
  if (!archive.ok) return createEmptyRaceHistoryStats(requestedYears, "unavailable");

  const yearUrls = new Map<number, string>();
  const links = /href\s*=\s*(["'])([^"']*eliza(20\d{2})\.html[^"']*)\1/gi;
  let match: RegExpExecArray | null;
  while ((match = links.exec(archive.html)) !== null) {
    const year = Number(match[3]);
    if (requestedYears.includes(year)) {
      try { yearUrls.set(year, new URL(decodeEntities(match[2]), archiveUrl).toString()); } catch { /* ignore invalid href */ }
    }
  }
  const failedYears = requestedYears.filter((year) => !yearUrls.has(year));
  const pending = requestedYears.filter((year) => yearUrls.has(year));
  const fetched: { year: number; venue: string | null; entries: JraHistoricalEntry[] }[] = [];
  let next = 0;
  async function worker() {
    while (true) {
      const i = next++;
      if (i >= pending.length) return;
      const year = pending[i];
      const page = await fetchHtml(yearUrls.get(year)!, 2592000, forceRefresh);
      if (!page.ok) { failedYears.push(year); continue; }
      const entries = extractJraHistoricalEntries(page.html);
      if (!entries.length) { failedYears.push(year); continue; }
      const pageText = cleanText(page.html);
      const venueMatch = pageText.match(/\d+回\s*(札幌|函館|福島|新潟|東京|中山|中京|京都|阪神|小倉)\s*\d+日/);
      fetched.push({ year, venue: venueMatch?.[1] ?? null, entries });
    }
  }
  await Promise.all(Array.from({ length: Math.min(CONCURRENCY, pending.length) }, () => worker()));

  const frameCounts: Record<string, { starts: number; wins: number; top3: number }> = {};
  const jockeyCounts: Record<string, { starts: number; wins: number; top3: number }> = {};
  let totalEntries = 0;
  const frameYears = fetched.filter((race) => race.venue === conditions.venue).map((race) => race.year).sort((a, b) => a - b);
  const jockeyYears = fetched.map((race) => race.year).sort((a, b) => a - b);
  for (const race of fetched) for (const entry of race.entries) {
    totalEntries++;
    const add = (bucket: Record<string, { starts: number; wins: number; top3: number }>, key: string) => {
      const item = bucket[key] ?? (bucket[key] = { starts: 0, wins: 0, top3: 0 });
      item.starts++;
      if (entry.finish === 1) item.wins++;
      if (entry.finish !== null && entry.finish <= 3) item.top3++;
    };
    if (race.venue === conditions.venue && entry.frameNumber !== null) add(frameCounts, String(entry.frameNumber));
    if (entry.jockey) add(jockeyCounts, normalizeJockeyName(entry.jockey));
  }
  const fetchedYears = fetched.map((race) => race.year).sort((a, b) => a - b);
  return {
    status: fetchedYears.length === 0 ? "unavailable" : failedYears.length ? "partial" : "available",
    requestedYears, fetchedYears, failedYears: failedYears.sort((a, b) => a - b),
    frameYears, jockeyYears, totalEntries,
    frameStats: Object.fromEntries(Object.entries(frameCounts).map(([k, v]) => [k, toRaceHistoryMetric(v)])),
    jockeyStats: Object.fromEntries(Object.entries(jockeyCounts).map(([k, v]) => [k, toRaceHistoryMetric(v)])),
  };
}

// 学習済みモデルと特徴量用データ（scripts/ml/train.py が data/ml に書き出す）
type StackAsset = StackModel & { adopted: boolean; validation?: Record<string, unknown> };
type HeavyAdjustment = { beta: number; runners: number; mle_beta?: number; years_improved?: number; per_year?: Record<string, unknown> };
let mlAssets: Promise<{
  model: Model; store: Store; metrics: Record<string, unknown>; stack: StackAsset | null; heavy: HeavyAdjustment | null;
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
      return { model, store, metrics, stack, heavy };
    } catch {
      mlAssets = null; // 次のリクエストで再試行
      return null;
    }
  })();
  return mlAssets;
}

function extractFirstPrize(html: string): number | null {
  const m = cleanText(html).match(/本賞金\s*[:：]\s*(\d+(?:\.\d+)?)/); // 「15000,6000,…万円」の先頭 = 1着
  return m ? Number(m[1]) : null; // 万円
}

const GOING_CODE: Record<string, number> = { "良": 0, "稍重": 1, "重": 2, "不良": 3 };

function getRaceHistoryForHorse(entry: RaceEntry | null, history: RaceHistoryStats | null): RaceHistoryForHorse {
  if (!entry || !history) return null;
  return {
    frame: entry.frameNumber === null ? null : history.frameStats[String(entry.frameNumber)] ?? null,
    jockey: entry.jockey ? history.jockeyStats[normalizeJockeyName(entry.jockey)] ?? null : null,
    frameYears: history.frameYears,
    jockeyYears: history.jockeyYears,
  };
}

function summarizePerformances(
  rows: { values: Record<string, string> }[],
  conditions: RaceConditions
): PerformanceStats {
  const stats: PerformanceStats = {
    parsedStarts: 0, wins: 0, top3: 0, averageFinish: null,
    gradedStarts: 0, gradedWins: 0, sameVenueStarts: 0, sameVenueTop3: 0,
    sameVenueSurfaceStarts: 0, sameVenueSurfaceTop3: 0,
    similarDistanceStarts: 0, similarDistanceTop3: 0, sameGoingStarts: 0, sameGoingTop3: 0,
    positionStarts: 0, positionGainTotal: 0, positionForwardStarts: 0, positionForwardTop3: 0,
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
    }
    const raceTitle = findValue([/^レース名$/]);
    if (/G[1-3]|Ｇ[１２３]|GⅠ|GⅡ|GⅢ|重賞/i.test(raceTitle)) {
      stats.gradedStarts++;
      if (finish === 1) stats.gradedWins++;
    }
    const venueText = findValue([/^開催$/]);
    const pastVenue = venueNames.find((name) => venueText.includes(name));
    if (conditions.venue && pastVenue === conditions.venue && finish !== null) {
      stats.sameVenueStarts++;
      if (finish <= 3) stats.sameVenueTop3++;
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
      }
      if (surfaceMatches && Math.abs(pastDistance - conditions.distanceMeters) <= 200 && finish !== null) {
        stats.similarDistanceStarts++;
        if (finish <= 3) stats.similarDistanceTop3++;
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
    const raceHistoryStats = await fetchRaceHistoryStats(raceConditions, forceRefresh);

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
              raceHistory: getRaceHistoryForHorse(raceEntry, raceHistoryStats),
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
            raceHistory: getRaceHistoryForHorse(raceEntry, raceHistoryStats),
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
            raceHistory: getRaceHistoryForHorse(raceEntry, raceHistoryStats),
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
        fieldSize: horses.length,
        prize1stMan: extractFirstPrize(race.html),
      };
      const rows = horses.map((horse) => {
        const entry = entryByHorseId.get(horse.horseId) ?? null;
        const sexAge = entry?.sexAge?.match(/^([牡牝セ])(\d+)$/);
        const jockeyKey = resolveJockey(entry?.jockey, ml.store, raceInput.date);
        const features = runnerFeatures(ml.store, raceInput, {
          horse: horse.name, jockeyKey,
          waku: entry?.frameNumber ?? null, umaban: entry?.horseNumber ?? null,
          weight: entry?.carriedWeight ?? null,
          age: sexAge ? Number(sexAge[2]) : null, sex: sexAge?.[1] ?? null,
          bodyWeight: entry?.bodyWeight ?? null, bodyWeightDiff: entry?.bodyWeightChange ?? null,
        });
        return { jockeyKey, features };
      });
      addFormDeviation(rows.map((row) => row.features));
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
        const applies = isHeavy && !useStack && beta !== 0;
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
      const order = probs.map((p, i) => ({ p, i })).sort((a, b) => b.p - a.p);
      order.forEach(({ i }, rank) => {
        const f = rows[i].features;
        const num = (v: number) => (Number.isNaN(v) ? null : v);
        predictions[i] = {
          probability: probs[i], baseProbability: baseProbs[i], pedigreeIncluded: useStack, rank: rank + 1,
          heavyNote: heavyNotes[i],
          modelFormDeviation: num(f.form_dev), priorRuns: f.n_prior,
          jockeyName: rows[i].jockeyKey,
          jockey: jockeyStats(ml.store, raceInput, rows[i].jockeyKey),
        };
      });
      mlSummary = {
        status: "available", dataUntil: ml.store.lastDate, goingAssumed, goingSource,
        pedigreeIncluded: useStack, stackValidation: ml.stack?.validation ?? null,
        heavyAdjustment: ml.heavy ? { beta: ml.heavy.beta, runners: ml.heavy.runners, yearsImproved: ml.heavy.years_improved ?? null } : null,
        prize1stMan: raceInput.prize1stMan, metrics: {
          auc: ml.metrics.auc, raceTop3Hit: ml.metrics.race_top3_hit, testYear: ml.metrics.test_year,
        },
      };
    }
    results.forEach((item, i) => {
      if (!item) return;
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
        raceHistoryStats,
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