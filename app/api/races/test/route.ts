
import { NextResponse } from "next/server";

export const runtime = "nodejs";

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
    const bytes = new Uint8Array(buffer);
    const html = new TextDecoder("utf-8").decode(bytes);

    const titleMatch = html.match(
      /<title[^>]*>([\s\S]*?)<\/title>/i
    );

    const title = titleMatch
      ? cleanText(titleMatch[1])
      : null;

    const headings = Array.from(
      html.matchAll(/<h[1-4]\b[^>]*>([\s\S]*?)<\/h[1-4]>/gi),
      (match) => cleanText(match[1])
    ).filter(Boolean);

    const tables: {
      rowCount: number;
      rows: string[][];
    }[] = [];

    const tableRegex =
      /<table\b[^>]*>([\s\S]*?)<\/table>/gi;

    let tableMatch: RegExpExecArray | null;

    while ((tableMatch = tableRegex.exec(html)) !== null) {
      const tableHtml = tableMatch[1];
      const rows: string[][] = [];

      const rowRegex =
        /<tr\b[^>]*>([\s\S]*?)<\/tr>/gi;

      let rowMatch: RegExpExecArray | null;

      while ((rowMatch = rowRegex.exec(tableHtml)) !== null) {
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
        tables.push({
          rowCount: rows.length,
          rows: rows.slice(0, 12),
        });
      }
    }

    const keywords = [
      "血統",
      "父",
      "母",
      "母父",
      "競走成績",
      "レース名",
      "着順",
      "距離",
      "タイム",
      "騎手",
    ];

    const foundKeywords = keywords.filter((word) =>
      html.includes(word)
    );

    return NextResponse.json(
      {
        ok: response.ok,
        status: response.status,
        contentType: response.headers.get("content-type"),
        title,
        responseBytes: bytes.length,
        headings: headings.slice(0, 30),
        tableCount: tables.length,
        tables: tables.slice(0, 8),
        foundKeywords,
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
