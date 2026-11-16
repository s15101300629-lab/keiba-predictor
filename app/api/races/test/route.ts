
import { NextResponse } from "next/server";

export const runtime = "nodejs";

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

    const html = await response.text();

    const titleMatch = html.match(
      /<title[^>]*>([\s\S]*?)<\/title>/i
    );

    const title = titleMatch
      ? titleMatch[1]
          .replace(/<[^>]*>/g, "")
          .replace(/\s+/g, " ")
          .trim()
      : null;

    return NextResponse.json({
      ok: response.ok,
      status: response.status,
      title,
      responseBytes: Buffer.byteLength(html, "utf8"),
      elapsedMs: Date.now() - startedAt,
    });
  } catch {
    return NextResponse.json(
      {
        ok: false,
        error: "netkeibaへの接続に失敗しました。",
        elapsedMs: Date.now() - startedAt,
      },
      { status: 502 }
    );
  }
}
