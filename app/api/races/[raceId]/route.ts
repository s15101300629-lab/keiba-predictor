import { NextResponse } from "next/server";

export const runtime = "nodejs";

const CONCURRENCY = 4;
const MAX_HISTORY = 10;

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
  odds: number | null;
  popularity: number | null;
  fieldStatus: {
    frameNumber: string;
    horseNumber: string;
    sexAge: string;
    carriedWeight: string;
    jockey: string;
    bodyWeight: string;
    odds: string;
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

async function fetchHtml(url: string) {
  const response = await fetch(url, {
    cache: "no-store",
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
  const titleMatch = html.match(/<title\\b[^>]*>([\\s\\S]*?)<\\/title>/i);
  const titleText = titleMatch ? cleanText(titleMatch[1]) : "";
  const headingMatch = html.match(/<h1\\b[^>]*>([\\s\\S]*?)<\\/h1>/i);
  const headingText = headingMatch ? cleanText(headingMatch[1]) : "";
  const raceNameMatch =
    headingText.match(/([ぁ-んァ-ヶ一-龠A-Za-z0-9ー・]+(?:ステークス|Ｓ|S|記念|賞))/) ||
    titleText.match(/^(.+?)(?:\\(|（| 出馬表| レース情報)/);
  const dateMatch = pageText.match(/(20\\d{2})年\\s*(\\d{1,2})月\\s*(\\d{1,2})日/);
  const meetingMatch = pageText.match(/(\\d+)回\\s*([^\\s　]+)\\s*(\\d+)日目/);
  const startMatch = pageText.match(/(\\d{1,2}:\\d{2})発走\\s*\\/\\s*(芝|ダ|障)(?:ート)?\\s*(\\d{3,4})m\\s*(右|左|直線)?/);
  const weatherMatch = pageText.match(/天候\\s*[:：]\\s*(晴|曇|雨|小雨|雪|小雪)/);
  const goingMatch = pageText.match(/馬場\\s*[:：]\\s*(良|稍重|重|不良)/);
  const courseLetterMatch = pageText.match(/([A-D])コース/);
  const courseLayout = /外回り|芝・外/.test(pageText)
    ? "外回り"
    : /内回り|芝・内/.test(pageText)
      ? "内回り"
      : null;
  const raceNumber = Number(raceId.slice(-2));

  const date = dateMatch
    ? `${dateMatch[1]}-${dateMatch[2].padStart(2, "0")}-${dateMatch[3].padStart(2, "0")}`
    : null;
  const meeting = meetingMatch ? Number(meetingMatch[1]) : null;
  const day = meetingMatch ? Number(meetingMatch[3]) : null;
  const venue = meetingMatch?.[2] ?? null;
  const surface = startMatch
    ? startMatch[2] === "ダ" || startMatch[2] === "障"
      ? startMatch[2] === "ダ" ? "ダート" : "障害"
      : "芝"
    : null;
  const distanceMeters = startMatch ? Number(startMatch[3]) : null;
  const direction = startMatch?.[4] ?? null;
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
      courseLetter: courseLetterMatch ? "available" : "not_found",
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
      odds: indexOf("単勝オッズ"),
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
      const oddsText = get(indexes.odds);
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
        odds: parseNumber(oddsText),
        popularity: parseNumber(popularityText),
        fieldStatus: {
          frameNumber: parseNumber(frameText) === null ? "not_published" : "available",
          horseNumber: parseNumber(numberText) === null ? "not_published" : "available",
          sexAge: sexAge ? "available" : "unavailable",
          carriedWeight: parseNumber(carriedText) === null ? "unavailable" : "available",
          jockey: jockey ? "available" : "not_confirmed",
          bodyWeight: bodyMatch || bodyOnlyMatch ? "available" : "not_published",
          odds: parseNumber(oddsText) === null ? "not_published" : "available",
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

function extractPerformances(tables: string[][][]) {
  const table = tables.find((candidate) =>
    candidate[0]?.includes("レース名") &&
    candidate[0]?.includes("着順") &&
    candidate[0]?.includes("騎手")
  );

  if (!table || table.length < 2) return null;

  const headers = table[0];
  const rows = table.slice(1, MAX_HISTORY + 1);

  return {
    headers,
    countOnPage: table.length - 1,
    rows: rows.map((cells) => ({
      values: Object.fromEntries(
        headers.map((header, index) => [header, cells[index] ?? ""])
      ),
      rawCells: cells,
    })),
  };
}

export async function GET(
  _request: Request,
  context: { params: Promise<{ raceId: string }> }
) {
  const startedAt = Date.now();
  const { raceId } = await context.params;
  const summaryOnly = new URL(_request.url).searchParams.get("summary") === "1";

  if (!/^\d{12}$/.test(raceId)) {
    return NextResponse.json(
      { ok: false, error: "raceId must be a 12-digit JRA race ID." },
      { status: 400 }
    );
  }

  try {
    const raceUrl =
      `https://race.netkeiba.com/race/shutuba.html?race_id=${raceId}`;
    const race = await fetchHtml(raceUrl);

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
    const results: Record<string, unknown>[] = new Array(horses.length);
    let nextIndex = 0;

    async function worker() {
      while (true) {
        const index = nextIndex++;
        if (index >= horses.length) return;

        const horse = horses[index];

        try {
          const url = `https://db.sp.netkeiba.com/horse/${horse.horseId}/`;
          const page = await fetchHtml(url);

          if (!page.ok) {
            results[index] = {
              horseId: horse.horseId,
              name: horse.name,
              raceEntry: entryByHorseId.get(horse.horseId) ?? null,
              status: "fetch_failed",
              httpStatus: page.status,
            };
            continue;
          }

          const tables = extractTables(page.html);
          const profile = extractProfile(tables);
          const performance = extractPerformances(tables);

          results[index] = {
            horseId: horse.horseId,
            name: horse.name,
            raceEntry: entryByHorseId.get(horse.horseId) ?? null,
            status: profile && performance ? "complete" : "partial",
            sourceStatus: page.status,
            responseBytes: page.bytes,
            profile,
            performanceCountOnPage: performance?.countOnPage ?? null,
            performanceHeaders: performance?.headers ?? [],
            performances: performance?.rows ?? [],
            historyReturned: performance?.rows.length ?? 0,
          };
        } catch (error) {
          results[index] = {
            horseId: horse.horseId,
            name: horse.name,
            raceEntry: entryByHorseId.get(horse.horseId) ?? null,
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
        horseCount: horses.length,
        raceEntryCount: raceEntries.length,
        matchedEntryCount,
        concurrency: CONCURRENCY,
        historyLimit: MAX_HISTORY,
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
          odds: "Null means no numeric odds were present in the fetched page at request time.",
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