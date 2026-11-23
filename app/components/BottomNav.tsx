"use client";
import Link from "next/link";
import { usePathname } from "next/navigation";

// 画面下に固定するタブ：レース予想（トップ・レース詳細）と AI成績を切り替える
export default function BottomNav() {
  const pathname = usePathname() ?? "/";
  const onResults = pathname.startsWith("/results");
  return <nav className="bottom-nav" aria-label="画面の切り替え">
    <Link href="/" className={onResults ? "bottom-nav-item" : "bottom-nav-item active"} aria-current={onResults ? undefined : "page"}>
      <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 5h16M4 12h16M4 19h10" /></svg>
      <span>レース予想</span>
    </Link>
    <Link href="/results" className={onResults ? "bottom-nav-item active" : "bottom-nav-item"} aria-current={onResults ? "page" : undefined}>
      <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M5 20V11M12 20V5M19 20v-6" /></svg>
      <span>AI成績</span>
    </Link>
  </nav>;
}
