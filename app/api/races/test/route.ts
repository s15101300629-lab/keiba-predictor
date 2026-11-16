
import { NextResponse } from "next/server";

export const runtime = "nodejs";

function extractTitle(html: string) {
  const match = html.match(
    /<title[^>]*>([\s\S]*?)<\/title>/i
  );

  return match
    ? match[1]
        .replace(/<[^>]*>/g, "")
        .replace(/\s+/g, " ")
        .trim()
    : null;
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

    const utf8Html = new TextDecoder("utf-8").decode(bytes);
    const shiftJisHtml = new TextDecoder("shift_jis").decode(bytes);

    const titleMatch = utf8Html.match(
      /<title[^>]*>([\s\S]*?)<\/title>/i
    );

    const titleRaw = titleMatch ? titleMatch[1] : null;

    const titleBytes = titleRaw
      ? Array.from(
          new TextEncoder().encode(titleRaw)
        )
          .map((b) => b.toString(16).padStart(2, "0"))
          .join(" ")
      : null;

    return NextResponse.json({
      ok: response.ok,
      status: response.status,
      contentType: response.headers.get("content-type"),
      titleUtf8: extractTitle(utf8Html),
      titleShiftJis: extractTitle(shiftJisHtml),
      titleRaw,
      titleBytes,
      responseBytes: bytes.length,
      elapsedMs: Date.now() - startedAt,
    });
  } catch (error) {
    return NextResponse.json(
      {
        ok: false,
        error: error instanceof Error
          ? error.message
          : "Unknown error",
        elapsedMs: Date.now() - startedAt,
      },
      { status: 502 }
    );
  }
}
