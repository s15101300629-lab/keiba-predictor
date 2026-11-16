import { NextResponse } from "next/server";

export const runtime = "nodejs";

const CONCURRENCY = 4;
const MAX_HISTORY = 10;

type HorseEntry = {
  horseId: string;
  name: string;
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

function extractHorses(html: string): HorseEntry[] {
  const horses: HorseEntry[] = [];
  const seen = new Set<string>();
  const anchorRegex = /<a\b([^>]*)>([\s\S]*?)<\/a>/gi;

  let anchorMatch: RegExpExecArray | null;

  while ((anchorMatch = anchorRegex.exec(html)) !== null) {
    const hrefMatch = anchorMatch[1].match(/\bhref\s*=\s*(["'])(.*?)\1/i);
    if (!hrefMatch) continue;

    const href = decodeEntities(hrefMatch[2]);
    if (!href.includes("/modal/horse.html")) continue;

    const idMatch = href.match(/horse_id=(\d+)/i);
    if (!idMatch) continue;

    const horseId = idMatch[1];
    if (seen.has(horseId)) continue;

    seen.add(horseId);
    horses.push({
      horseId,
      name: cleanText(anchorMatch[2]),
    });
  }

  return horses;
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

  if (!/^\d{12}$/.test(raceId)) {
    return NextResponse.json(
      { ok: false, error: "raceId must be a 12-digit JRA race ID." },
      { status: 400 }
    );
  }

  try {
    const raceUrl =
      `https://race.sp.netkeiba.com/race/shutuba.html?race_id=${raceId}`;
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

    const horses = extractHorses(race.html);

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
      (item) =>
        (item as { status?: string })?.status === "complete"
    ).length;

    const partialCount = results.filter(
      (item) =>
        (item as { status?: string })?.status === "partial"
    ).length;

    const failedCount = results.filter(
      (item) =>
        (item as { status?: string })?.status === "fetch_failed"
    ).length;

    return NextResponse.json(
      {
        ok: failedCount === 0 && partialCount === 0,
        raceId,
        racePageStatus: race.status,
        horseCount: horses.length,
        concurrency: CONCURRENCY,
        historyLimit: MAX_HISTORY,
        completeCount,
        partialCount,
        failedCount,
        results,
        elapsedMs: Date.now() - startedAt,
        notes: {
          bodyWeight: "Not collected by this endpoint yet.",
          odds: "Not collected by this endpoint yet.",
          raceCardNumbersAndJockeys:
            "Not collected by this endpoint yet.",
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
