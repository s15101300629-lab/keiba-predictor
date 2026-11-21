
import { NextRequest, NextResponse } from "next/server";

export async function middleware(request: NextRequest) {
  const password = process.env.APP_PASSWORD;
  const secret = process.env.AUTH_SECRET;
  const path = request.nextUrl.pathname;

  if (!password || !secret) {
    return new NextResponse("Server configuration error", {
      status: 500,
    });
  }

  if (path === "/login" || path === "/api/login" || /^\/(logo|logo-mark|icon|apple-icon)\.png$/.test(path)) {
    return NextResponse.next();
  }

  const token = request.cookies.get("app_auth")?.value;

  if (token) {
    try {
      const key = await crypto.subtle.importKey(
        "raw",
        new TextEncoder().encode(secret),
        { name: "HMAC", hash: "SHA-256" },
        false,
        ["verify"]
      );

      const normalized = token.replace(/-/g, "+").replace(/_/g, "/");
      const padded = normalized + "=".repeat((4 - normalized.length % 4) % 4);
      const signature = Uint8Array.from(
        atob(padded),
        (c) => c.charCodeAt(0)
      );

      const valid = await crypto.subtle.verify(
        "HMAC",
        key,
        signature,
        new TextEncoder().encode(password)
      );

      if (valid) {
        return NextResponse.next();
      }
    } catch {
      // 無効な認証情報はログイン画面へ
    }
  }

  return NextResponse.redirect(new URL("/login", request.url));
}

export const config = {
  matcher: ["/((?!_next/static|_next/image|favicon.ico).*)"],
};
