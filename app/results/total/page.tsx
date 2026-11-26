import { readdirSync, readFileSync } from "node:fs";
import path from "node:path";
import ResultsView, { type BetDay, type Day } from "../ResultsView";

export const metadata = { title: "累計成績 | Racehorse Insight Analytics" };

function loadDays(): Day[] {
  const dir = path.join(process.cwd(), "data", "ml", "day_check");
  let files: string[] = [];
  try { files = readdirSync(dir).filter((f) => /^\d{4}-\d{2}-\d{2}\.json$/.test(f)); } catch { return []; }
  return files.sort().reverse().map((f) => {
    const d = JSON.parse(readFileSync(path.join(dir, f), "utf8")) as Day;
    return { date: d.date, detail: d.detail, drift: d.drift ?? null };
  });
}

function loadBetDays(): BetDay[] {
  const dir = path.join(process.cwd(), "data", "ml", "day_bets");
  let files: string[] = [];
  try { files = readdirSync(dir).filter((f) => /^\d{4}-\d{2}-\d{2}\.json$/.test(f)); } catch { return []; }
  return files.map((f) => JSON.parse(readFileSync(path.join(dir, f), "utf8")) as BetDay);
}

export default function CumulativeResultsPage() {
  return <ResultsView days={loadDays()} betDays={loadBetDays()} view="cumulative" />;
}
