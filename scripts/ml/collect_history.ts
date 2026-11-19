// 重賞出走馬（data/pedigree/graded_races.json）の netkeiba 全戦績を集め、data/pedigree/histories.json に保存する。
//   npx tsx scripts/ml/collect_history.ts          取得済みの馬は飛ばす（再開可能）
//   npx tsx scripts/ml/collect_history.ts --refresh 全頭を取り直す（新しい出走を反映）
import { existsSync, readFileSync, writeFileSync } from "node:fs";
import path from "node:path";
import { parseHistoryPage, type HistoryValues } from "@/app/lib/history";
import type { GradedRace } from "./collect_pedigree";

const OUT = path.resolve(__dirname, "../../data/pedigree");
const CONCURRENCY = 3;
const WAIT_MS = 300;
const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

async function get(url: string): Promise<string | null> {
  for (let attempt = 0; attempt < 3; attempt++) {
    try {
      await sleep(WAIT_MS);
      const res = await fetch(url, { headers: { "User-Agent": "Mozilla/5.0" }, signal: AbortSignal.timeout(20000) });
      if (res.status === 404) return null;
      if (!res.ok) throw new Error("HTTP " + res.status);
      return await res.text();
    } catch (error) {
      if (attempt === 2) { console.error("  ! " + url + ": " + error); return null; }
      await sleep(2000 * (attempt + 1));
    }
  }
  return null;
}

async function main() {
  const refresh = process.argv.includes("--refresh");
  const races = JSON.parse(readFileSync(path.join(OUT, "graded_races.json"), "utf8")) as Record<string, GradedRace>;
  const file = path.join(OUT, "histories.json");
  const histories = existsSync(file) ? JSON.parse(readFileSync(file, "utf8")) as Record<string, HistoryValues[]> : {};
  const ids = [...new Set(Object.values(races).flatMap((r) => r.runners.map((h) => h.horseId)).filter((x): x is string => !!x))];
  const todo = ids.filter((id) => refresh || !histories[id]);
  console.log(`histories: ${ids.length} horses, ${todo.length} to fetch`);
  let next = 0;
  let done = 0;
  await Promise.all(Array.from({ length: CONCURRENCY }, async () => {
    while (next < todo.length) {
      const id = todo[next++];
      const base = `https://db.netkeiba.com/horse/result/${id}/`;
      const first = await get(base);
      if (!first) continue;
      const rows = parseHistoryPage(first) ?? [];
      // 戦績が複数ページに分かれている場合
      const pages = [...new Set([...first.matchAll(/[?&]page=(\d+)/g)].map((m) => Number(m[1])))].filter((p) => p > 1).sort((a, b) => a - b);
      for (const p of pages) {
        const html = await get(base + "?page=" + p);
        rows.push(...(html ? parseHistoryPage(html) ?? [] : []));
      }
      histories[id] = rows;
      if (++done % 100 === 0) { writeFileSync(file, JSON.stringify(histories) + "\n"); console.log(`  ${done}/${todo.length}`); }
    }
  }));
  writeFileSync(file, JSON.stringify(histories) + "\n");
  console.log(`done: ${done} horses`);
}

main().catch((e) => { console.error(e); process.exit(1); });
