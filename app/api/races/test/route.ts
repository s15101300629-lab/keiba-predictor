
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

    const buffer = await response.arrayBuffer();
    const bytes = new Uint8Array(buffer);

    const contentType =
      response.headers.get("content-type") || "";

    // HTTPヘッダーから文字コードを取得
    let charsetMatch = contentType.match(
      /charset\s*=\s*["']?([^;"'\s]+)/i
    );

    // HTTPヘッダーにない場合はHTML内を確認
    if (!charsetMatch) {
      const headerText = new TextDecoder("latin1").decode(
        bytes.slice(0, 4096)
      );

      charsetMatch = headerText.match(
        /<meta[^>]+charset\s*=\s*["']?([^"'\s/>;]+)/i
      );
    }

    const charset = charsetMatch
      ? charsetMatch[1].trim().toLowerCase()
      : "utf-8";

    // 指定された文字コードでHTMLを変換
    let html: string;

    try {
      html = new TextDecoder(charset).decode(bytes);
    } catch {
      html = new TextDecoder("utf-8").decode(bytes);
    }

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
      contentType,
      detectedCharset: charset,
      title,
      responseBytes: bytes.length,
      elapsedMs: Date.now() - startedAt,
    });
  } catch (error) {
    return NextResponse.json(
      {
        ok: false,
        error: "netkeibaへの接続に失敗しました。",
        detail:
          error instanceof Error
            ? error.message
            : "Unknown error",
        elapsedMs: Date.now() - startedAt,
      },
      { status: 502 }
    );
  }
}
