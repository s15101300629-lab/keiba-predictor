
import { NextResponse } from "next/server";

export const runtime = "nodejs";

const CONCURRENCY = 4;

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
  const tableRegex =
    /<table\b[^>]*>([\s\S]*?)<\/table>/gi;

  let tableMatch: RegExpExecArray | null;

  while ((tableMatch = tableRegex.exec(html)) !== null) {
    const rows: string[][] = [];
    const rowRegex =
      /<tr\b[^>]*>([\s\S]*?)<\/tr>/gi;

    let rowMatch: RegExpExecArray | null;

    while ((rowMatch = rowRegex.exec(tableMatch[1])) !== null) {
      const cells: string[] = [];
      const cellRegex =
        /<t[dh]\b[^>]*>([\s\S]*?)<\/t[dh]>/gi;

      let cellMatch: RegExpExecArray | null;

      while ((cellMatch = cellRegex.exec(rowMatch[1])) !== null) {
        cells.push(cleanText(cellMatch[1]));
      }

      if (cells.length > 0) {
        rows.push(cells);
      }
    }

    if (rows.length > 0) {
      tables.push(rows);
    }
  }

  return tables;
}

async function fetchHtml(url: string) {
  const response = await fetch(url, {
    cache: "no-store",
    headers: {
      Accept: "text/html",
    },
    signal: AbortSignal.timeout(10000),
  });

  const buffer = await response.arrayBuffer();

  return {
    status: response.status,
    ok: response.ok,
    html: new TextDecoder("utf-8").decode(
      new Uint8Array(buffer)
    ),
    bytes: buffer.byteLength,
  };
}

export async function GET() {
  const startedAt = Date.now();

  const raceUrl =
    "https://race.sp.netkeiba.com/race/shutuba.html?race_id=202608050511";

  try {
    const race = await fetchHtml(raceUrl);

    if (!race.ok) {
      return NextResponse.json(
        {
          ok: false,
          step: "race_page",
          status: race.status,
        },
        { status: 502 }
      );
    }

    const horses: {
      horseId: string;
      name: string;
    }[] = [];

    const anchorRegex =
      /<a\b([^>]*)>([\s\S]*?)<\/a>/gi;

    let anchorMatch: RegExpExecArray | null;

    while ((anchorMatch = anchorRegex.exec(race.html)) !== null) {
      const hrefMatch = anchorMatch[1].match(
        /\bhref\s*=\s*(["'])(.*?)\1/i
      );

      if (!hrefMatch) continue;

      const href = decodeEntities(hrefMatch[2]);

      if (!href.includes("/modal/horse.html")) continue;

      const idMatch = href.match(/horse_id=(\d+)/i);
      if (!idMatch) continue;

      const horseId = idMatch[1];

      if (horses.some((horse) => horse.horseId === horseId)) {
        continue;
      }

      horses.push({
        horseId,
        name: cleanText(anchorMatch[2]),
      });
    }

    const results: Record<string, unknown>[] =
      new Array(horses.length);

    let nextIndex = 0;

    async function worker() {
      while (true) {
        const index = nextIndex++;

        if (index >= horses.length) {
          return;
        }

        const horse = horses[index];

        try {
          const url =
            `https://db.sp.netkeiba.com/horse/${horse.horseId}/`;

          const page = await fetchHtml(url);

          if (!page.ok) {
            results[index] = {
              horseId: horse.horseId,
              name: horse.name,
              ok: false,
              status: page.status,
            };
            continue;
          }

          const tables = extractTables(page.html);

          const profile = tables.find((table) =>
            table.some((row) =>
              ["父", "母", "調教師"].some((label) =>
                row[0]?.includes(label)
              )
            ) && table.length >= 5
          ) || [];

          const performance = tables.find((table) =>
            table[0]?.includes("レース名") &&
            table[0]?.includes("着順") &&
            table[0]?.includes("騎手")
          );

          results[index] = {
            horseId: horse.horseId,
            name: horse.name,
            ok: page.ok && Boolean(performance),
            responseBytes: page.bytes,
            profileFound: profile.length > 0,
            profile,
            performanceFound: Boolean(performance),
            performanceCount: performance
              ? performance.length - 1
              : 0,
            performanceHeaders: performance?.[0] || [],
            recentPerformances: performance
              ? performance.slice(1, 4)
              : [],
          };
        } catch (error) {
          results[index] = {
            horseId: horse.horseId,
            name: horse.name,
            ok: false,
            error:
              error instanceof Error
                ? error.message
                : "Unknown error",
          };
        }
      }
    }

    const workerCount = Math.min(
      CONCURRENCY,
      horses.length
    );

    await Promise.all(
      Array.from(
        { length: workerCount },
        () => worker()
      )
    );

    const successCount = results.filter(
      (item) => item?.ok === true
    ).length;

    return NextResponse.json(
      {
        ok: true,
        raceStatus: race.status,
        raceHorseCount: horses.length,
        testHorseCount: results.length,
        concurrency: CONCURRENCY,
        successCount,
        failureCount: results.length - successCount,
        results,
        elapsedMs: Date.now() - startedAt,
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
        error:
          error instanceof Error
            ? error.message
            : "Unknown error",
        elapsedMs: Date.now() - startedAt,
      },
      { status: 502 }
    );
  }
}
