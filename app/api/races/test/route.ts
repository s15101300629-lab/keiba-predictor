
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

export async function GET() {
  const startedAt = Date.now();

  const targetUrl =
    "https://race.sp.netkeiba.com/race/shutuba.html?race_id=202608050511";

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

    const rows: {
      horseId: string;
      cells: string[];
      rowText: string;
    }[] = [];

    const rowRegex = /<tr\b[^>]*>([\s\S]*?)<\/tr>/gi;
    let rowMatch: RegExpExecArray | null;

    while ((rowMatch = rowRegex.exec(html)) !== null) {
      const rowHtml = rowMatch[1];

      const horseIdMatch = rowHtml.match(
        /horse_id=(\d+)/i
      );

      if (!horseIdMatch) continue;

      const cells: string[] = [];
      const cellRegex =
        /<t[dh]\b[^>]*>([\s\S]*?)<\/t[dh]>/gi;

      let cellMatch: RegExpExecArray | null;

      while ((cellMatch = cellRegex.exec(rowHtml)) !== null) {
        cells.push(cleanText(cellMatch[1]));
      }

      rows.push({
        horseId: horseIdMatch[1],
        cells,
        rowText: cleanText(rowHtml),
      });
    }

    const horseIds = Array.from(
      new Set(
        Array.from(
          html.matchAll(/horse_id=(\d+)/gi),
          (match) => match[1]
        )
      )
    );

    return NextResponse.json(
      {
        ok: response.ok,
        status: response.status,
        contentType: response.headers.get("content-type"),
        title,
        responseBytes: bytes.length,
        horseIdCount: horseIds.length,
        tableRowCount: rows.length,
        rows: rows.slice(0, 30),
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
