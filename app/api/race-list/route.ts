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
  const anchorRegex = /<a\\b([^>]*)>([\\s\\S]*?)<\\/a>/gi;
  let anchorMatch: RegExpExecArray | null;

  while ((anchorMatch = anchorRegex.exec(html)) !== null) {
    const hrefMatch = anchorMatch[1].match(/href=["']([^"']+)["']/i);
    const raceIdMatch = hrefMatch?.[1].match(/race_id=(\\d{12})/);
    if (!raceIdMatch) continue;

    const raceId = raceIdMatch[1];
    if (seen.has(raceId)) continue;
    const venue = VENUES[raceId.slice(4, 6)];
    if (!venue) continue;

    const anchorStart = anchorMatch.index;
    const listStart = html.lastIndexOf("<li", anchorStart);
    const divStart = html.lastIndexOf("<div", anchorStart);
    const contextStart = Math.max(listStart, divStart);
    const listEnd = html.indexOf("</li>", anchorMatch.index + anchorMatch[0].length);
    const divEnd = html.indexOf("</div>", anchorMatch.index + anchorMatch[0].length);
    const contextEndCandidates = [listEnd >= 0 ? listEnd + 5 : -1, divEnd >= 0 ? divEnd + 6 : -1].filter((value) => value > anchorMatch!.index);
    const contextEnd = contextEndCandidates.length ? Math.min(...contextEndCandidates) : anchorMatch.index + anchorMatch[0].length;
    const context = html.slice(contextStart >= 0 ? contextStart : anchorMatch.index, contextEnd);
    const linkHtml = anchorMatch[2];
    const contextText = textOf(context);
    const anchorText = textOf(linkHtml);

    const raceNumberText = textOf(extractInner(linkHtml, "Race_Num")) || contextText;
    const raceNumberMatch = raceNumberText.match(/(\\d{1,2})\\s*R/i) || anchorText.match(/(\\d{1,2})\\s*R/i);
    if (!raceNumberMatch) continue;
    const raceNumber = Number(raceNumberMatch[1]);

    const titleCandidates = [
      extractInner(context, "ItemTitle"),
      extractInner(context, "RaceName"),
      extractInner(linkHtml, "ItemTitle"),
      extractInner(linkHtml, "RaceName"),
    ].map(textOf).filter(Boolean);
    const raceTitle = titleCandidates[0] || anchorText.replace(/\\b\\d{1,2}\\s*R\\b/i, "").trim() || `${raceNumber}R`;
    const grade = parseGrade(`${raceTitle} ${contextText}`);
    const isMain = raceNumber === 11;
    if (!isMain && !grade) continue;

    const timeText = textOf(extractInner(context, "RaceList_Itemtime")) || contextText;
    const courseText = textOf(extractInner(context, "RaceList_ItemLong")) || "";
    const runnersText = textOf(extractInner(context, "RaceList_Itemnumber")) || contextText;
    const timeMatch = timeText.match(/\\d{1,2}:\\d{2}/);
    const runnersMatch = runnersText.match(/(\\d+)\\s*頭/);
    const labels: string[] = [];
    if (isMain) labels.push("メイン");
    if (grade) labels.push(`重賞 ${grade}`);

    seen.add(raceId);
    races.push({
      raceId,
      venue,
      raceNumber,
      raceTitle,
      startTime: timeMatch?.[0] ?? null,
      course: courseText || null,
      runners: runnersMatch ? Number(runnersMatch[1]) : null,
      isMain,
      grade,
      labels,
    });
  }

  return races.sort((a, b) => VENUE_ORDER.indexOf(a.venue) - VENUE_ORDER.indexOf(b.venue) || a.raceNumber - b.raceNumber);
}

const VENUE_ORDER = ["札幌", "函館", "福島", "新潟", "東京", "中山", "中京", "京都", "阪神", "小倉"];

export async function GET(request: Request) {
  const url = new URL(request.url);
  const requestedDate = url.searchParams.get("date");
  const date =
    requestedDate ??
    new Intl.DateTimeFormat("en-CA", {
      timeZone: "Asia/Tokyo",
      year: "numeric",
      month: "2-digit",
      day: "2-digit",
    }).format(new Date()).replace(/-/g, "");

  if (!/^\d{8}$/.test(date)) {
    return NextResponse.json(
      { ok: false, error: "date must be YYYYMMDD." },
      { status: 400 }
    );
  }

  try {
    const response = await fetch(
      `https://race.netkeiba.com/top/race_list_sub.html?kaisai_date=${date}`,
      {
        cache: "no-store",
        headers: { Accept: "text/html" },
        signal: AbortSignal.timeout(10000),
      }
    );

    if (!response.ok) {
      return NextResponse.json(
        { ok: false, date, sourceStatus: response.status, error: "開催レース一覧を取得できませんでした。" },
        { status: 502 }
      );
    }

    const buffer = await response.arrayBuffer();
    const html = new TextDecoder("utf-8").decode(new Uint8Array(buffer));
    const races = parseRaces(html);
    const venues = Array.from(new Set(races.map((race) => race.venue)));

    return NextResponse.json(
      {
        ok: true,
        date,
        sourceStatus: response.status,
        raceCount: races.length,
        venues,
        races,
        filter: "各競馬場の11R（メイン）および重賞のみ。平場は除外。",
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
        date,
        error: error instanceof Error ? error.message : "Unknown error",
      },
      { status: 502 }
    );
  }
}
