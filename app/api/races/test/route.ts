
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

  const targetUrl =
    "https://race.sp.netkeiba.com/?pid=race_list&kaisai_id=2026080505&kaisai_date=20261121";

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

      if (
        /race_id=|race\/|race_result|shutuba/i.test(href)
      ) {
        let absoluteHref = href;

        try {
          absoluteHref = new URL(
            href,
            targetUrl
          ).toString();
        } catch {
          // URL形式を解釈できない場合は元の値を保持
        }

        links.push({
          text,
          href: absoluteHref,
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
        raceLinkCount: links.length,
        links: links.slice(0, 100),
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
