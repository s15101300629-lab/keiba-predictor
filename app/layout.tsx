import type { Metadata, Viewport } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "KEIBA PREDICTOR",
  description: "自分専用のJRA競馬予想・分析アプリ",
  applicationName: "KEIBA PREDICTOR",
  appleWebApp: { capable: true, statusBarStyle: "black-translucent", title: "KEIBA" },
};
export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  themeColor: "#090f1d",
};
export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <html lang="ja"><body>{children}</body></html>;
}