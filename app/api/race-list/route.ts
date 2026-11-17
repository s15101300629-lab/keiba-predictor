import { NextResponse } from "next/server";

export const runtime = "nodejs";

const VENUES: Record<string, string> = {
  "01": "札幌",
  "02": "函館",
  "03": "福島",
  "04": "新潟",
  "05": "東京",
  "06": "中山",
  "07": "中京",
  "08": "京都",
  "09": "阪神",
  "10": "小倉",
};

type RaceItem = {
  raceId: string;
  venue: string;
  raceNumber: number;
  raceTitle: string;
  startTime: string | null;
  course: string | null;
  runners: number | null;
  isMain: boolean;
  grade: string | null;
  labels: string[];
};

function decodeEntities(value: string) {
  return value
    .replace(/&nbsp;/gi, " ")
    .replace(/&amp;/gi, "&")
    .replace(/&quot;/gi, '"')
    .replace(/&#39;/gi, "'")
    .replace(/&lt;/gi, "<")
    .replace(/&gt;/gi, ">");
}

function textOf(value: string) {
  return decodeEntities(value.replace(/<[^>]*>/g, " "))
    .replace(/\s+/g, " ")
    .trim();
}

function extractInner(html: string, className: string) {
  const escaped = className.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  const match = html.match(
    new RegExp(`<([a-z0-9]+)\\b[^>]*class=["'][^"']*\\b${escaped}\\b[^"']*["'][^>]*>([\\s\\S]*?)<\\/\\1>`, "i")
  );
  return match?.[2] ?? "";
}

function parseGrade(title: string): string | null {
  const normalized = title
    .replace(/Ｇ/g, "G")
    .replace(/[Ⅰ１]/g, "1")
    .replace(/[Ⅱ２]/g, "2")
    .replace(/[Ⅲ３]/g, "3")
    .replace(/\s+/g, "");
  const match = normalized.match(/G([123])/i);
  return match ? `G${match[1]}` : null;
}

function parseRaces(html: string): RaceItem[] {
  const races: RaceItem[] = [];
  const seen = new Set<string>();
  const itemRegex = /<li\b[^>]*class=["'][^"']*RaceList_DataItem[^"']*["'][^>]*>([\s\S]*?)<\/li>/gi;
  let itemMatch: RegExpExecArray | null;

  while ((itemMatch = itemRegex.exec(html)) !== null) {
    const item = itemMatch[1];
    const linkMatch = item.match(/<a\b([^>]*)>([\s\S]*?)<\/a>/i);
    if (!linkMatch) continue;

    const hrefMatch = linkMatch[1].match(/href=["']([^"']+)["']/i);
    const raceIdMatch = hrefMatch?.[1].match(/race_id=(\d{12})/);
    if (!raceIdMatch) continue;

    const raceId = raceIdMatch[1];
    if (seen.has(raceId)) continue;
    const venue = VENUES[raceId.slice(4, 6)];
    if (!venue) continue;

    const linkHtml = linkMatch[2];
    const raceNumberText = textOf(extractInner(linkHtml, "Race_Num")) || textOf(linkHtml);
    const raceNumberMatch = raceNumberText.match(/(\d{1,2})\s*R/i);
    if (!raceNumberMatch) continue;
    const raceNumber = Number(raceNumberMatch[1]);

    const title = textOf(extractInner(linkHtml, "ItemTitle")) || `${raceNumber}R`;
    const timeText = textOf(extractInner(linkHtml, "RaceList_Itemtime"));
    const courseText = textOf(extractInner(linkHtml, "RaceList_ItemLong"));
    const runnersText = textOf(extractInner(linkHtml, "RaceList_Itemnumber"));
    const timeMatch = timeText.match(/\d{1,2}:\d{2}/);
    const runnersMatch = runnersText.match(/(\d+)\s*頭/);
    const grade = parseGrade(`${title} ${textOf(item)}`);
    const isMain = raceNumber === 11;

    if (!isMain && !grade) continue;

    const labels: string[] = [];
    if (isMain) labels.push("メイン");
    if (grade) labels.push(`重賞 ${grade}`);

    seen.add(raceId);
    races.push({
      raceId,
      venue,
      raceNumber,
      raceTitle: title,
      startTime: timeMatch?.[0] ?? null,
      course: courseText || null,
      runners: runnersMatch ? Number(runnersMatch[1]) : null,
      isMain,
      grade,
      labels,
    });
  }

  const venueOrder = ["札幌", "函館", "福島", "新潟", "東京", "中山", "中京", "京都", "阪神", "小倉"];
  return races.sort((a, b) => venueOrder.indexOf(a.venue) - venueOrder.indexOf(b.venue) || a.raceNumber - b.raceNumber);
}

function getJstDate(offsetDays: number): string {
  const parts = new Intl.DateTimeFormat("en-US", {
    timeZone: "Asia/Tokyo",
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
  }).formatToParts(new Date());
  const year = Number(parts.find((part) => part.type === "year")?.value);
  const month = Number(parts.find((part) => part.type === "month")?.value);
  const day = Number(parts.find((part) => part.type === "day")?.value);
  const date = new Date(Date.UTC(year, month - 1, day + offsetDays));
  return String(date.getUTCFullYear()) +
    String(date.getUTCMonth() + 1).padStart(2, "0") +
    String(date.getUTCDate()).padStart(2, "0");
}

function hasRaceEntries(html: string): boolean {
  const itemRegex = /<li\b[^>]*class=["'][^"']*RaceList_DataItem[^"']*["'][^>]*>([\s\S]*?)<\/li>/gi;
  let itemMatch: RegExpExecArray | null;
  while ((itemMatch = itemRegex.exec(html)) !== null) {
    if (/race_id=\d{12}/i.test(itemMatch[1])) return true;
  }
  return false;
}

async function fetchRaceList(date: string): Promise<{ html: string; status: number }> {
  const response = await fetch(
    "https://race.netkeiba.com/top/race_list_sub.html?kaisai_date=" + date,
    {
      cache: "force-cache",
      next: { revalidate: 3600 },
      headers: { Accept: "text/html" },
      signal: AbortSignal.timeout(10000),
    }
  );
  if (!response.ok) {
    throw new Error("開催一覧の取得に失敗しました（HTTP " + response.status + " / " + date + "）");
  }
  const buffer = await response.arrayBuffer();
  return {
    html: new TextDecoder("utf-8").decode(new Uint8Array(buffer)),
    status: response.status,
  };
}

export async function GET(request: Request) {
  const url = new URL(request.url);
  const requestedDate = url.searchParams.get("date");

  if (requestedDate && !/^\d{8}$/.test(requestedDate)) {
    return NextResponse.json(
      { ok: false, error: "date must be YYYYMMDD." },
      { status: 400 }
    );
  }

  try {
    let date: string | null = requestedDate;
    let html = "";
    let sourceStatus = 200;

    if (requestedDate) {
      const result = await fetchRaceList(requestedDate);
      html = result.html;
      sourceStatus = result.status;
    } else {
      // 閲覧日ではなく、今日以降で最初にJRA開催が確認できた日を採用する。
      // 各日付の取得結果は1時間キャッシュし、最大21日先まで検索する。
      for (let offset = 0; offset <= 21; offset += 1) {
        const candidateDate = getJstDate(offset);
        const result = await fetchRaceList(candidateDate);
        if (hasRaceEntries(result.html)) {
          date = candidateDate;
          html = result.html;
          sourceStatus = result.status;
          break;
        }
      }
    }

    if (!date) {
      return NextResponse.json({
        ok: true,
        date: null,
        raceCount: 0,
        venues: [],
        races: [],
        error: "今後21日以内にJRAの開催日を確認できませんでした。",
      });
    }

    const races = parseRaces(html);
    const venues = Array.from(new Set(races.map((race) => race.venue)));

    return NextResponse.json(
      {
        ok: true,
        date,
        sourceStatus,
        raceCount: races.length,
        venues,
        races,
        filter: "次回開催日の各競馬場の11R（メイン）および重賞のみ。平場は除外。",
      },
      {
        headers: {
          "Content-Type": "application/json; charset=utf-8",
          "Cache-Control": "no-store",
        },
      }
    );
  } catch (error) {
    return NextResponse.json(
      {
        ok: false,
        error: error instanceof Error ? error.message : "Unknown error",
      },
      { status: 502 }
    );
  }
}
