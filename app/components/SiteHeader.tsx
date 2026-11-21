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
        <img src="/logo-mark.png" alt="" className="brand-logo" width={40} height={40} />
        <span><strong>RACEHORSE INSIGHT</strong><small>ANALYTICS</small></span>
      </Link>
      {onRace && <Link href="/" className="header-back">← レース一覧</Link>}
    </div>
  </header>;
}
