// AIスコアの内訳（観点ごとの点。SHAP）を、内訳を開いたときにだけ計算する（2026-11-25、表示を速くするため）。
// 前はレース詳細の API がレースを開くたびに全馬の SHAP を計算していた（18頭で約0.6秒。ページをためた後は待ち時間の約6割）。
// 画面は、レース詳細の API が返す各馬の本体の入力（inputRow。☆の /api/longshot と同じ）と 3着以内確率を送る。
// 観点の点（scorePoints）と順位は出走馬全部の SHAP を比べて出すので、1頭の内訳でもレースの全馬の分を計算する。結果はためて使い回す
import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";
import path from "node:path";
import { NextResponse } from "next/server";
import type { Model } from "@/app/lib/ml/model";
import { treeShap } from "@/app/lib/ml/shap";
import { groupPoints, scorePoints } from "@/app/lib/ml/shapGroups";

export const runtime = "nodejs";

type Row = { key: string; inputRow: (number | null)[]; probability: number };
export type ShapGroupOut = { group: string; phi: number; points: number; top: { name: string; phi: number; points: number; value: number | null }[] };

let modelPromise: Promise<Model> | null = null;
const loadModel = () => (modelPromise ??= readFile(path.join(process.cwd(), "data", "ml", "model.json"), "utf8").then((t) => JSON.parse(t) as Model));

// 同じ入力（同じレース・同じ入力の行）の結果をためる（新しいものから 50 件）
const cache = new Map<string, Record<string, ShapGroupOut[]>>();

export async function POST(request: Request) {
  const t0 = performance.now();
  let body: { rows?: Row[] };
  try { body = await request.json(); } catch { return NextResponse.json({ ok: false, error: "JSON が読めない" }, { status: 400 }); }
  const rows = body.rows ?? [];
  if (!rows.length) return NextResponse.json({ ok: false, error: "rows がない" }, { status: 400 });
  const model = await loadModel();
  if (rows.some((r) => !Array.isArray(r.inputRow) || r.inputRow.length !== model.features.length || !(r.probability > 0))) {
    // モデルが変わった後に、前の入力が送られてきたとき（画面のためた結果）。取り直してもらう
    return NextResponse.json({ ok: false, error: "入力の項目の数がモデルと合わない（レースを更新してから開き直す）" }, { status: 409 });
  }
  const key = createHash("sha1").update(JSON.stringify([model.trees.length, rows])).digest("hex");
  let groups = cache.get(key);
  const hit = !!groups;
  if (!groups) {
    const feats = rows.map((r) => Object.fromEntries(model.features.map((name, k) => [name, r.inputRow[k] ?? Number.NaN])));
    const phis = feats.map((f) => treeShap(model, f));
    const pts = scorePoints(phis, rows.map((r) => r.probability));
    groups = Object.fromEntries(rows.map((r, i) => [r.key, groupPoints(model.features, phis[i], pts[i]).map((g) => ({
      group: g.group, phi: g.phi, points: g.points,
      top: g.top.map((t) => { const v = feats[i][t.name]; return { name: t.name, phi: t.phi, points: t.points, value: Number.isNaN(v) ? null : v }; }),
    }))]));
    cache.set(key, groups);
    if (cache.size > 50) cache.delete(cache.keys().next().value!);
  }
  const ms = performance.now() - t0;
  return NextResponse.json({ ok: true, groups, cached: hit }, { headers: { "Server-Timing": `shap;dur=${ms.toFixed(1)}`, "Cache-Control": "no-store" } });
}
