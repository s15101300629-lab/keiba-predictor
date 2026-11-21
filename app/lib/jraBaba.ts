// JRA の馬場情報ページ（https://www.jra.go.jp/keiba/baba/index.html・index2.html・index3.html …）の解析。
// 開催中の競馬場ごとに1ページ。開催日・何時現在・天候・芝とダートの馬場状態が載る（当日は9時ごろ、前日は正午ごろに更新）。
// ページは Shift_JIS。取得はレース詳細 API（route.ts の fetchHtml）で行い、ここでは文字列だけを扱う。

export const JRA_BABA_PAGES = ["index.html", "index2.html", "index3.html", "index4.html"].map((p) => "https://www.jra.go.jp/keiba/baba/" + p);

export type JraBaba = {
  venue: string; // 東京・京都 など
  date: string; // YYYY-MM-DD（開催日）
  asOf: string | null; // 「11月21日（土曜）10時現在」など
  weather: string | null;
  turf: string | null; // 良・稍重・重・不良
  dirt: string | null;
};

const VENUES = ["札幌", "函館", "福島", "新潟", "東京", "中山", "中京", "京都", "阪神", "小倉"];
const GOINGS = ["稍重", "不良", "良", "重"];

export function parseJraBaba(html: string): JraBaba | null {
  const head = html.match(/<h2>([\s\S]*?)<span class="date">([\s\S]*?)<\/span>/);
  if (!head) return null;
  const venue = VENUES.find((v) => head[1].includes(v + "競馬"));
  const d = head[2].match(/(\d{4})年(\d{1,2})月(\d{1,2})日/);
  if (!venue || !d) return null;
  const asOf = html.match(/馬場状態<span class="time">\s*[（(]?([\s\S]*?)[）)]?\s*<\/span>/)?.[1]?.trim() ?? null;
  const weather = html.match(/天候：<strong>\s*([^<\s]+)\s*<\/strong>/)?.[1] ?? null;
  const going = (label: string) => {
    const v = html.match(new RegExp(`<h4>${label}</h4></div>\\s*<div class="content">\\s*<p>\\s*([^<]+?)\\s*</p>`))?.[1] ?? null;
    return v && GOINGS.includes(v) ? v : null;
  };
  return {
    venue, date: `${d[1]}-${d[2].padStart(2, "0")}-${d[3].padStart(2, "0")}`, asOf,
    weather, turf: going("芝"), dirt: going("ダート"),
  };
}
