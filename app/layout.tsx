import type { Metadata, Viewport } from "next";
import "./globals.css";
import SiteFooter from "./components/SiteFooter";
import SiteHeader from "./components/SiteHeader";
import BottomNav from "./components/BottomNav";

export const metadata: Metadata = {
  title: "Racehorse Insight Analytics",
  description: "自分専用のJRA競馬予想・分析アプリ",
  applicationName: "Racehorse Insight Analytics",
  appleWebApp: { capable: true, statusBarStyle: "black-translucent", title: "Racehorse Insight" },
};
export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  viewportFit: "cover",
  themeColor: "#0b0a08",
};
export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <html lang="ja"><body><SiteHeader />{children}<SiteFooter /><BottomNav /></body></html>;
}