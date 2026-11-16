
import { NextResponse } from "next/server";

export const runtime = "nodejs";

function cleanTitle(value: string | null) {
  if (value === null) return null;

  return value
    .replace(/<[^>]*>/g, "")
    .replace(/\s+/g, " ")
    .trim();
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

    // 1バイト単位でHTMLのタグ位置を特定する
    const byteView = new TextDecoder("latin1").decode(bytes);

    const titleMatch = /<title\b[^>]*>([\s\S]*?)<\/title>/i.exec(
      byteView
    );

    let titleUtf8: string | null = null;
    let titleShiftJis: string | null = null;
    let titleBytesHex: string | null = null;

    if (titleMatch && titleMatch.index !== undefined) {
      const titleStart =
        titleMatch.index + titleMatch[0].indexOf(">") + 1;

      const titleEnd = titleStart + titleMatch[1].length;

      // 元のHTMLからタイトル部分のバイト列を直接取得
      const originalTitleBytes = bytes.slice(
        titleStart,
        titleEnd
      );

      // 同じ生バイトを2種類の文字コードで解釈
      titleUtf8 = cleanTitle(
        new TextDecoder("utf-8").decode(originalTitleBytes)
      );

      titleShiftJis = cleanTitle(
        new TextDecoder("shift_jis").decode(originalTitleBytes)
      );

      titleBytesHex = Array.from(originalTitleBytes)
        .map((b) => b.toString(16).padStart(2, "0"))
        .join(" ");
    }

    return NextResponse.json(
      {
        ok: response.ok,
        status: response.status,
        contentType: response.headers.get("content-type"),
        titleUtf8,
        titleShiftJis,
        titleBytesHex,
        responseBytes: bytes.length,
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
