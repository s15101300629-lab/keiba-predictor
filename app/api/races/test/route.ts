
import { NextResponse } from "next/server";

export const runtime = "nodejs";

function decodeEntities(value: string) {
  return value
    .replace(/&nbsp;/gi, " ")
    .replace(/&amp;/gi, "&")
    .replace(/&quot;/gi, '"')
    .replace(/&#39;/gi, "'")
    .replace(/&lt;/gi, "<")
    .replace(/&gt;/gi, ">")
    .replace(/&#(\d+);/g, (_, code) =>
      String.fromCodePoint(Number(code))
    )
    .replace(/&#x([0-9a-f]+);/gi, (_, code) =>
      String.fromCodePoint(parseInt(code, 16))
    );
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

export async function GET() {
  const startedAt = Date.now();
  const horseId = "2021104824";

  const targetUrl =
    `https://db.sp.netkeiba.com/horse/${horseId}/`;

  try {
    const response = await fetch(targetUrl, {
      cache: "no-store",
      headers: {
        Accept: "text/html",
      },
      signal: AbortSignal.timeout(10000),
    });

    const buffer = await response.arrayBuffer();
    const html = new TextDecoder("utf-8").decode(
      new Uint8Array(buffer)
    );

    const titleMatch = html.match(
      /<title[^>]*>([\s\S]*?)<\/title>/i
    );

    const title = titleMatch
      ? cleanText(titleMatch[1])
      : null;

    const tables = extractTables(html);

    const profileRows = tables.find((table) =>
      table.some((row) =>
        ["父", "母", "調教師"].some((label) =>
          row[0]?.includes(label)
        )
      ) && table.length >= 5
    ) || [];

    const performanceTable = tables.find((table) =>
      table[0]?.includes("レース名") &&
      table[0]?.includes("着順") &&
      table[0]?.includes("騎手")
    );

    return NextResponse.json(
      {
        ok: response.ok,
        status: response.status,
        title,
        horseId,
        profileRows,
        performanceFound: Boolean(performanceTable),
        performanceHeaders: performanceTable?.[0] || [],
        performanceCount: performanceTable
          ? performanceTable.length - 1
          : 0,
        performances: performanceTable
          ? performanceTable.slice(1, 11)
          : [],
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
