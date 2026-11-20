"use client";
import Link from "next/link";
import { usePathname } from "next/navigation";

// 全画面共通のヘッダー：アプリ名（トップへ戻る）。レース詳細では一覧へ戻るボタンも出す
export default function SiteHeader() {
  const pathname = usePathname() ?? "/";
  const onRace = pathname.startsWith("/race/");
  return <header className="site-header">
    <div className="site-header-inner">
      <Link href="/" className="brand" aria-label="トップへ戻る">
        <span className="brand-mark">K</span>
        <span><strong>KEIBA PREDICTOR</strong><small>JRA 重賞・メインレースの AI 予想</small></span>
      </Link>
      {onRace && <Link href="/" className="header-back">← レース一覧</Link>}
    </div>
  </header>;
}
