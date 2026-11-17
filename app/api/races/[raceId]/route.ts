import { NextResponse } from "next/server";

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

function decodeEntities(value: string) {
  return value
    .replace(/&nbsp;/gi, " ")
    .replace(/&amp;/gi, "&")
    .replace(/&quot;/gi, '"')
    .replace(/&#39;/gi, "'")
    .replace(/&lt;/gi, "<")
    .replace(/&gt;/gi, ">");
}

function cleanText(value: string) {
  return decodeEntities(
    value
      .replace(/<script\b[^>]*>[\s\S]*?<\/script>/gi, "")
      .replace(/<style\b[^>]*>[\s\S]*?<\/style>/gi, "")
      .replace(/<[^>]*>/g, "")
  )
    .replace(/\s+/g, " ")
    .trim();
}

function extractTables(html: string) {
  const tables: string[][][] = [];
  const tableRegex = /<table\b[^>]*>([\s\S]*?)<\/table>/gi;
  let tableMatch: RegExpExecArray | null;

  while ((tableMatch = tableRegex.exec(html)) !== null) {
    const rows: string[][] = [];
    const rowRegex = /<tr\b[^>]*>([\s\S]*?)<\/tr>/gi;
    let rowMatch: RegExpExecArray | null;

    while ((rowMatch = rowRegex.exec(tableMatch[1])) !== null) {
      const cells: string[] = [];
      const cellRegex = /<t[dh]\b[^>]*>([\s\S]*?)<\/t[dh]>/gi;
      let cellMatch: RegExpExecArray | null;

      while ((cellMatch = cellRegex.exec(rowMatch[1])) !== null) {
        cells.push(cleanText(cellMatch[1]));
      }

      if (cells.length > 0) rows.push(cells);
    }

    if (rows.length > 0) tables.push(rows);
  }

  return tables;
}

async function fetchHtml(url: string, revalidateSeconds: number, forceRefresh: boolean) {
  const response = await fetch(url, {
    ...(forceRefresh
      ? { cache: "no-store" as const }
      : { cache: "force-cache" as const, next: { revalidate: revalidateSeconds } }),
    headers: { Accept: "text/html" },
    signal: AbortSignal.timeout(10000),
  });

  const buffer = await response.arrayBuffer();

  return {
    status: response.status,
    ok: response.ok,
    html: new TextDecoder("utf-8").decode(new Uint8Array(buffer)),
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
    headingText.match(/([ぁ-んァ-ヶ一-龠A-Za-z0-9ー・]+(?:チャンピオンシップ|ステークス|Ｓ|S|記念|賞))/) ||
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


type PerformanceStats = {
  parsedStarts: number; wins: number; top3: number; averageFinish: number | null;
  gradedStarts: number; gradedWins: number;
  sameVenueStarts: number; sameVenueTop3: number;
  similarDistanceStarts: number; similarDistanceTop3: number;
  sameGoingStarts: number; sameGoingTop3: number;
};

function summarizePerformances(rows: { values: Record<string, string> }[], conditions: RaceConditions): PerformanceStats {
  const stats: PerformanceStats = {
    parsedStarts: 0, wins: 0, top3: 0, averageFinish: null,
    gradedStarts: 0, gradedWins: 0, sameVenueStarts: 0, sameVenueTop3: 0,
    similarDistanceStarts: 0, similarDistanceTop3: 0, sameGoingStarts: 0, sameGoingTop3: 0,
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
    const distanceText = findValue([/^距離$/]);
    const distanceMatch = distanceText.match(/(芝|ダート|ダ|障害|障)?\s*(\d{3,4})\s*m?/i);
    if (distanceMatch && conditions.distanceMeters !== null) {
      const pastSurface = distanceMatch[1] === "ダ" || distanceMatch[1] === "ダート" ? "ダート"
        : distanceMatch[1] === "障" || distanceMatch[1] === "障害" ? "障害"
        : distanceMatch[1] === "芝" ? "芝" : null;
      const pastDistance = Number(distanceMatch[2]);
      const surfaceMatches = !conditions.surface || !pastSurface || conditions.surface === pastSurface;
      if (surfaceMatches && Math.abs(pastDistance - conditions.distanceMeters) <= 200 && finish !== null) {
        stats.similarDistanceStarts++;
        if (finish <= 3) stats.similarDistanceTop3++;
      }
    }
    const goingText = findValue([/^馬場$/, /^馬場状態$/, /^状態$/]);
    if (conditions.going && goingText && goingText.includes(conditions.going) && finish !== null) {
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

type JraRating = {
  value: number;
  raceDate: string;
  raceName: string;
  source: string;
};

type JraRatingDiagnostics = {
  sourceUrl: string | null;
  httpStatus: number | null;
  publishedDate: string | null;
  parsedRowCount: number;
  matchedHorseCount: number;
  ratingCount: number;
  error: string | null;
};

type JraRatingMap = Map<string, JraRating> & {
  diagnostics: JraRatingDiagnostics;
};

function createJraRatingMap(): JraRatingMap {
  const ratings = new Map<string, JraRating>() as JraRatingMap;
  ratings.diagnostics = {
    sourceUrl: null,
    httpStatus: null,
    publishedDate: null,
    parsedRowCount: 0,
    matchedHorseCount: 0,
    ratingCount: 0,
    error: null,
  };
  return ratings;
}

function normalizeHorseName(value: string) {
  return value
    .replace(/[（(][^）)]*[）)]/g, "")
    .replace(/[\s　]+/g, "")
    .trim();
}

function normalizeRaceName(value: string) {
  return normalizeHorseName(value)
    .replace(/[Ａ-Ｚａ-ｚ０-９]/g, (character) =>
      String.fromCharCode(character.charCodeAt(0) - 0xFEE0)
    )
    .replace(/[Ｓｓ]/g, "S")
    .replace(/[Ｃｃ]/g, "C")
    .replace(/ステークス/g, "S")
    .replace(/カップ/g, "C")
    .toUpperCase();
}

function normalizeRaceNameForMatch(value: string) {
  const normalized = normalizeRaceName(value);
  const aliases: Record<string, string> = {
    "マイルCS": "マイルチャンピオンシップ",
    "マイルチャンピオンシップ": "マイルチャンピオンシップ",
    "スプリンターズS": "スプリンターズS",
    "スプリンターズステークス": "スプリンターズS",
    "ジャパンC": "ジャパンC",
    "ジャパンカップ": "ジャパンC",
    "NHKマイルC": "NHKマイルC",
    "NHKマイルカップ": "NHKマイルC",
    "チャンピオンズC": "チャンピオンズC",
    "チャンピオンズカップ": "チャンピオンズC",
  };
  return aliases[normalized] ?? normalized;
}

function parseRatingValue(value: string) {
  const match = value.match(/(\d{2,3})(?:\s*[SMILE](?:\s*,\s*[SMILE])*)?(?:\s*[（(]T[）)])?\s*$/i);
  if (!match) return null;
  const rating = Number(match[1]);
  return rating >= 50 && rating <= 150 ? rating : null;
}

function extractMonthlyJraRatings(
  html: string,
  year: number,
  targetDate: string,
  sourceUrl: string
) {
  const ratings = new Map<string, JraRating>();
  const tables = extractTables(html);
  const table = tables.find((candidate) =>
    candidate.some((row) =>
      row.some((cell) => cell.includes("実施日")) &&
      row.some((cell) => cell.includes("第1着馬名レーティング"))
    )
  );
  if (!table) return ratings;

  const headerIndex = table.findIndex((row) =>
    row.some((cell) => cell.includes("実施日")) &&
    row.some((cell) => cell.includes("第1着馬名レーティング"))
  );

  for (const row of table.slice(headerIndex + 1)) {
    const dateMatch = (row[0] ?? "").match(/(\d{1,2})月\s*(\d{1,2})日/);
    if (!dateMatch) continue;

    const raceDate = year + "-" + dateMatch[1].padStart(2, "0") + "-" + dateMatch[2].padStart(2, "0");
    if (raceDate >= targetDate) continue;

    const raceName = row[2] ?? "";
    for (const cell of row.slice(3, 7)) {
      const ratingMatches = cell.matchAll(/([^\d]+?)(\d{2,3})(?:\s*[SMILE](?:\s*,\s*[SMILE])*)?(?=\s|$)/gi);
      for (const ratingMatch of ratingMatches) {
        const horseName = normalizeHorseName(ratingMatch[1]);
        const value = Number(ratingMatch[2]);
        if (!horseName || value < 50 || value > 150) continue;

        const existing = ratings.get(horseName);
        if (!existing || raceDate > existing.raceDate) {
          ratings.set(horseName, { value, raceDate, raceName, source: sourceUrl });
        }
      }
    }
  }

  return ratings;
}

const G1_RATING_SLUGS: Record<string, string> = {
  "マイルチャンピオンシップ": "mile",
};

function findG1RaceUrl(
  html: string,
  raceName: string | null,
  targetDate: string,
  venue: string | null,
  baseUrl: string
) {
  const wantedName = raceName ? normalizeRaceNameForMatch(raceName) : "";
  if (!wantedName) return null;

  const rows = [...html.matchAll(/<tr\b[^>]*>([\s\S]*?)<\/tr>/gi)];
  const candidates: string[] = [];

  for (const row of rows) {
    const rowHtml = row[1];
    const cells = [...rowHtml.matchAll(/<t[dh]\b[^>]*>([\s\S]*?)<\/t[dh]>/gi)]
      .map((cell) => cleanText(cell[1]));
    const rowText = cleanText(rowHtml);
    const dateMatch = rowText.match(/(20\d{2})年\s*(\d{1,2})月\s*(\d{1,2})日/);
    if (!dateMatch) continue;

    const rowDate = dateMatch[1] + "-" +
      dateMatch[2].padStart(2, "0") + "-" +
      dateMatch[3].padStart(2, "0");
    if (rowDate !== targetDate) continue;

    const normalizedVenue = venue ? normalizeHorseName(venue) : "";
    const venueMatches = normalizedVenue
      ? cells.some((cell) => normalizeHorseName(cell) === normalizedVenue)
      : false;
    if (normalizedVenue && !venueMatches) continue;

    const raceNameMatches = cells.some((cell) =>
      normalizeRaceNameForMatch(cell) === wantedName
    );
    if (!raceNameMatches) continue;

    const anchors = [...rowHtml.matchAll(/<a\b([^>]*)>([\s\S]*?)<\/a>/gi)];
    const links = anchors
      .map((anchor) => {
        const href = anchor[1].match(/\bhref\s*=\s*(["'])(.*?)\1/i)?.[2];
        return href
          ? {
              href: new URL(decodeEntities(href), baseUrl).toString(),
              label: cleanText(anchor[2]),
            }
          : null;
      })
      .filter((link): link is { href: string; label: string } => link !== null);

    const nameMatchedLink = links.find((link) =>
      normalizeRaceNameForMatch(link.label) === wantedName
    );
    if (nameMatchedLink) {
      candidates.push(nameMatchedLink.href);
      continue;
    }

    if (links.length === 1) {
      candidates.push(links[0].href);
      continue;
    }

    // JRAの年度別一覧で、開催前のレース行にリンクがない場合の確認済みURL。
    const slug = G1_RATING_SLUGS[wantedName];
    if (links.length === 0 && slug) {
      candidates.push(
        new URL(
          "datafile/ranking/g1/" + targetDate.slice(0, 4) + "/" + slug + ".html",
          "https://www.jra.go.jp/"
        ).toString()
      );
    }
  }

  return candidates.length === 1 ? candidates[0] : null;
}
async function fetchG1PreRatings(
  horses: HorseEntry[],
  raceName: string | null,
  targetDate: string,
  venue: string | null,
  year: number,
  forceRefresh: boolean
): Promise<JraRatingMap> {
  const result = createJraRatingMap();
  const yearListUrl = "https://www.jra.go.jp/datafile/ranking/g1/" + year + ".html";
  result.diagnostics.sourceUrl = yearListUrl;

  let yearPage: Awaited<ReturnType<typeof fetchHtml>>;
  try {
    yearPage = await fetchHtml(yearListUrl, 21600, forceRefresh);
  } catch (error) {
    result.diagnostics.error = error instanceof Error ? error.message : "年度別G1一覧の取得に失敗しました";
    return result;
  }
  result.diagnostics.httpStatus = yearPage.status;
  if (!yearPage.ok) {
    result.diagnostics.error = "年度別G1一覧のHTTP取得に失敗しました";
    return result;
  }

  const targetUrl = findG1RaceUrl(yearPage.html, raceName, targetDate, venue, yearListUrl);
  if (!targetUrl) {
    result.diagnostics.error =
      "年度別G1一覧に開催日・レース名が一致するリンクが見つかりませんでした";
    return result;
  }
  result.diagnostics.sourceUrl = targetUrl;

  let detail: Awaited<ReturnType<typeof fetchHtml>>;
  try {
    detail = await fetchHtml(targetUrl, 21600, forceRefresh);
  } catch (error) {
    result.diagnostics.error = error instanceof Error ? error.message : "G1プレレーティングページの取得に失敗しました";
    return result;
  }
  result.diagnostics.httpStatus = detail.status;
  if (!detail.ok) {
    result.diagnostics.error = "G1プレレーティングページのHTTP取得に失敗しました";
    return result;
  }

  const pageText = cleanText(detail.html);
  const [, targetMonthText, targetDayText] = targetDate.split("-");
  const targetMonthDay = Number(targetMonthText) + "月" + Number(targetDayText) + "日";
  const expectedRaceName = raceName ? normalizeRaceNameForMatch(raceName) : "";
  const pageRaceName = normalizeRaceName(pageText);
  const raceNameConfirmed = Boolean(expectedRaceName && pageRaceName.includes(expectedRaceName));
  const raceDateConfirmed = pageText.includes(targetMonthDay);
  const venueConfirmed = !venue || pageText.includes(venue + "競馬");
  if (!raceNameConfirmed || !raceDateConfirmed || !venueConfirmed) {
    result.diagnostics.error =
      "取得したJRA詳細ページが対象レースと一致しませんでした（レース名・開催日・競馬場を照合）";
    return result;
  }

  const publishedMatch = pageText.match(/(20\d{2})年\s*(\d{1,2})月\s*(\d{1,2})日/);
  const publishedDate = publishedMatch
    ? publishedMatch[1] + "-" + publishedMatch[2].padStart(2, "0") + "-" + publishedMatch[3].padStart(2, "0")
    : null;
  result.diagnostics.publishedDate = publishedDate;
  if (!publishedDate) {
    result.diagnostics.error = "プレレーティングページの公開日を読み取れません";
    return result;
  }
  if (publishedDate > targetDate) {
    result.diagnostics.error = "対象レース日より後に公開されたページのため除外しました";
    return result;
  }

  const wantedNames = new Set(horses.map((horse) => normalizeHorseName(horse.name)));
  const matchedNames = new Set<string>();
  const rows = [...detail.html.matchAll(/<tr\b[^>]*>([\s\S]*?)<\/tr>/gi)];
  result.diagnostics.parsedRowCount = rows.length;

  for (const rowMatch of rows) {
    const cells = [...rowMatch[1].matchAll(/<t[dh]\b[^>]*>([\s\S]*?)<\/t[dh]>/gi)]
      .map((cell) => cleanText(cell[1]));
    if (cells.length < 2) continue;

    const nameCell = cells.find((cell) => wantedNames.has(normalizeHorseName(cell)));
    if (!nameCell) continue;

    const horseName = normalizeHorseName(nameCell);
    matchedNames.add(horseName);
    const value = cells
      .filter((cell) => normalizeHorseName(cell) !== horseName)
      .map((cell) => parseRatingValue(cell))
      .find((candidate) => candidate !== null) ?? null;
    if (value === null) continue;

    result.set(horseName, {
      value,
      raceDate: publishedDate,
      raceName: "GⅠプレレーティング",
      source: targetUrl,
    });
  }

  // 表のセル構造が異なる場合の代替解析。次の出走馬名までを対象にする。
  for (const horseName of wantedNames) {
    if (result.has(horseName)) continue;
    const index = pageText.indexOf(horseName);
    if (index < 0) continue;
    matchedNames.add(horseName);

    const afterName = pageText.slice(index + horseName.length);
    let segmentEnd = afterName.length;
    for (const otherName of wantedNames) {
      if (otherName === horseName) continue;
      const nextIndex = afterName.indexOf(otherName);
      if (nextIndex >= 0 && nextIndex < segmentEnd) segmentEnd = nextIndex;
    }

    const ratingMatch = afterName.slice(0, segmentEnd).match(/\b(\d{2,3})\s*[SMILE]\b/i);
    if (!ratingMatch) continue;
    const value = Number(ratingMatch[1]);
    if (value < 50 || value > 150) continue;

    result.set(horseName, {
      value,
      raceDate: publishedDate,
      raceName: "GⅠプレレーティング",
      source: targetUrl,
    });
  }

  result.diagnostics.matchedHorseCount = matchedNames.size;
  result.diagnostics.ratingCount = result.size;
  if (result.size === 0) {
    result.diagnostics.error = "ページは取得できましたが、出走馬のレーティングを照合できませんでした";
  } else if (result.size < wantedNames.size) {
    result.diagnostics.error = "一部の出走馬のみレーティングを照合できました";
  }
  return result;
}
async function fetchJraRatings(
  horses: HorseEntry[],
  raceName: string | null,
  targetDate: string | null,
  venue: string | null,
  forceRefresh: boolean
) {
  const result = createJraRatingMap();
  if (!targetDate) {
    result.diagnostics.error = "レース日を取得できないためJRAレーティングを検索できません";
    return result;
  }

  const [yearText, monthText] = targetDate.split("-");
  const targetYear = Number(yearText);
  const targetMonth = Number(monthText);

  const g1Ratings = await fetchG1PreRatings(
    horses,
    raceName,
    targetDate,
    venue,
    targetYear,
    forceRefresh
  );
  result.diagnostics = g1Ratings.diagnostics;
  for (const [name, rating] of g1Ratings) result.set(name, rating);

  const pages: { year: number; url: string }[] = [];
  for (let offset = 0; offset < 24; offset++) {
    let year = targetYear;
    let month = targetMonth - offset;
    while (month <= 0) {
      month += 12;
      year--;
    }
    pages.push({
      year,
      url: "https://www.jra.go.jp/datafile/ranking/jyusyo/" + year + "/" + String(month).padStart(2, "0") + ".html",
    });
  }

  const wantedNames = new Set(horses.map((horse) => normalizeHorseName(horse.name)));
  const monthlyFoundNames = new Set<string>();

  for (let offset = 0; offset < pages.length; offset += 4) {
    const batch = pages.slice(offset, offset + 4);
    const fetched = await Promise.all(batch.map(async (page) => {
      try {
        const response = await fetchHtml(page.url, 604800, forceRefresh);
        return response.ok ? { ...page, html: response.html } : null;
      } catch {
        return null;
      }
    }));

    for (const page of fetched) {
      if (!page) continue;
      const monthlyRatings = extractMonthlyJraRatings(
        page.html,
        page.year,
        targetDate,
        page.url
      );

      for (const name of wantedNames) {
        const rating = monthlyRatings.get(name);
        const existing = result.get(name);
        if (rating) monthlyFoundNames.add(name);
        if (rating && (!existing || rating.raceDate > existing.raceDate)) {
          result.set(name, rating);
        }
      }
    }

    if (monthlyFoundNames.size === wantedNames.size) break;
  }

  result.diagnostics.ratingCount = result.size;
  return result;
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

    const jraRatings = await fetchJraRatings(horses, raceConditions.raceName, raceConditions.date, raceConditions.venue, forceRefresh);
    const results: Record<string, unknown>[] = new Array(horses.length);
    let nextIndex = 0;

    async function worker() {
      while (true) {
        const index = nextIndex++;
        if (index >= horses.length) return;

        const horse = horses[index];

        try {
          const url = `https://db.sp.netkeiba.com/horse/${horse.horseId}/`;
          const page = await fetchHtml(url, 604800, forceRefresh);

          if (!page.ok) {
            results[index] = {
              horseId: horse.horseId,
              name: horse.name,
              raceEntry: entryByHorseId.get(horse.horseId) ?? null,
              rating: jraRatings.get(normalizeHorseName(horse.name)) ?? null,
              status: "fetch_failed",
              httpStatus: page.status,
            };
            continue;
          }

          const tables = extractTables(page.html);
          const profile = extractProfile(tables);
          const performance = extractPerformances(tables);
          const performanceRows = performance?.rows ?? [];
          const performanceStats = summarizePerformances(performanceRows, raceConditions);

          results[index] = {
            horseId: horse.horseId,
            name: horse.name,
            raceEntry: entryByHorseId.get(horse.horseId) ?? null,
            rating: jraRatings.get(normalizeHorseName(horse.name)) ?? null,
            status: profile && performance ? "complete" : "partial",
            sourceStatus: page.status,
            responseBytes: page.bytes,
            profile,
            performanceCountOnPage: performance?.countOnPage ?? null,
            performanceHeaders: performance?.headers ?? [],
            performances: performance?.rows ?? [],
            historyReturned: performanceRows.length,
            performanceStats,
          };
        } catch (error) {
          results[index] = {
            horseId: horse.horseId,
            name: horse.name,
            raceEntry: entryByHorseId.get(horse.horseId) ?? null,
            rating: jraRatings.get(normalizeHorseName(horse.name)) ?? null,
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
        ratingDiagnostics: jraRatings.diagnostics,
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