
import { NextResponse } from "next/server";

export const runtime = "nodejs";

function cleanText(value: string) {
  return value
    .replace(/<[^>]*>/g, "")
    .replace(/&nbsp;/gi, " ")
    .replace(/&amp;/gi, "&")
    .replace(/&quot;/gi, '"')
    .replace(/&#39;/gi, "'")
    .replace(/&lt;/gi, "<")
    .replace(/&gt;/gi, ">")
    .replace(/\s+/g, " ")
    .trim();
}

function decodeHtml(value: string) {
  return value
    .replace(/&amp;/gi, "&")
    .replace(/&quot;/gi, '"')
    .replace(/&#39;/gi, "'")
    .replace(/&lt;/gi, "<")
    .replace(/&gt;/gi, ">");
}

export async function GET() {
  const startedAt = Date.now();

  try {
    const response = await fetch(
      "https://race.sp.netkeiba.com/?pid=race_calendar",
      {
        cache: "no-store",
        headers: {
          Accept: "text/html",
        },
        signal: AbortSignal.timeout(10000),
      }
    );

    const buffer = await response.arrayBuffer();
    const bytes = new Uint8Array(buffer);
    const html = new TextDecoder("utf-8").decode(bytes);

    const titleMatch = html.match(
      /<title[^>]*>([\s\S]*?)<\/title>/i
    );

    const title = titleMatch
      ? cleanText(titleMatch[1])
      : null;

    const links: {
      text: string;
      href: string;
    }[] = [];

    const anchorRegex =
      /<a\b([^>]*)>([\s\S]*?)<\/a>/gi;

    let match: RegExpExecArray | null;

    while ((match = anchorRegex.exec(html)) !== null) {
      const attributes = match[1];
      const innerHtml = match[2];

      const hrefMatch = attributes.match(
        /\bhref\s*=\s*(["'])(.*?)\1/i
      );

      if (!hrefMatch) continue;

      const href = decodeHtml(hrefMatch[2]);
      const text = cleanText(innerHtml);

      if (!href || !text) continue;

      if (
        /race_id=|kaisai_date=|race_list|calendar|date=/i.test(
          href
        ) ||
        /開催|レース/.test(text)
      ) {
        links.push({
          text,
          href,
        });
      }
    }

    return NextResponse.json(
      {
        ok: response.ok,
        status: response.status,
        contentType: response.headers.get("content-type"),
        title,
        responseBytes: bytes.length,
        totalAnchorCount: (
          html.match(/<a\b/gi) || []
        ).length,
        relevantLinkCount: links.length,
        links: links.slice(0, 100),
        elapsedMs: Date.now() - startedAt,
      },
      {
        headers: {
          "Content-Type": "application/json; charset=utf-8",
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
