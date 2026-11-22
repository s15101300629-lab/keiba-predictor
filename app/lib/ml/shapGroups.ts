// AI予想の「各観点」：SHAP の寄与（app/lib/ml/shap.ts）を、モデルの入力のまとまりごとに足す。
// 観点の順位＝出走馬の中で、その観点がAIスコアを押し上げた量の順。モデルに入っていない観点は出さない。

export const SHAP_GROUPS = [
  "能力指数", "タイム指数", "出走レースレベル", "上がり・ラップタイプ", "脚質", "ポジション", "調子", "レース間隔", "前走距離",
  "条件適性", "枠順", "騎手", "調教師", "馬主・生産者", "血統", "調教", "馬体・斤量・年齢", "前走からの変化", "レース条件",
] as const;
export type ShapGroup = (typeof SHAP_GROUPS)[number];

// 2026-11-22 に依頼者と決めた観点（キャプチャの「勝ち馬サーチ」「走行解析」に合わせて細かくした）
const EXACT: Record<string, ShapGroup> = {
  surface: "レース条件", distance: "レース条件", going: "レース条件", field_size: "レース条件", venue: "レース条件",
  c_straight: "レース条件", c_elev: "レース条件", c_lap: "レース条件",
  waku: "枠順", umaban: "枠順", umaban_ratio: "枠順",
  prize_1st: "出走レースレベル", field_form_mean: "出走レースレベル",
  weight: "馬体・斤量・年齢", age: "馬体・斤量・年齢", sex: "馬体・斤量・年齢", body_weight: "馬体・斤量・年齢", body_weight_diff: "馬体・斤量・年齢",
  n_prior: "調子", days_since_last: "レース間隔", g_layoff: "レース間隔",
  dist_change: "前走距離", fit_change: "前走距離", fit_change_n: "前走距離",
  surface_change: "前走からの変化", weight_change: "前走からの変化", jockey_change: "前走からの変化",
  last1_victim: "前走からの変化", victim_n5: "前走からの変化",
  pos_last_mean: "ポジション", rel_pos_rank: "ポジション", pos_std: "ポジション",
};
const PREFIX: [string, ShapGroup][] = [
  ["c4_", "ポジション"],
  ["cond_ab", "条件適性"], ["fit_", "条件適性"], ["surface_", "条件適性"], ["dist_", "条件適性"], ["venue_", "条件適性"], ["wet_", "条件適性"],
  ["sim_", "条件適性"], ["band_", "条件適性"], ["sc_", "条件適性"], ["scp_", "条件適性"],
  ["ab_", "能力指数"], ["tidx_", "タイム指数"], ["rel_tidx", "タイム指数"], ["rank_tidx", "タイム指数"],
  ["ag_", "上がり・ラップタイプ"], ["early_gap", "上がり・ラップタイプ"], ["fade_", "上がり・ラップタイプ"],
  ["form_", "調子"], ["last1_", "調子"], ["g_", "調子"],
  ["pos_", "脚質"], ["nige_", "脚質"], ["senko_", "脚質"], ["hp_", "脚質"], ["sp_", "脚質"], ["field_", "脚質"], ["pace_", "脚質"],
  ["prize_ratio", "出走レースレベル"], ["rel_prize", "出走レースレベル"], ["rank_prize", "出走レースレベル"], ["lv_", "出走レースレベル"], ["h2h_", "出走レースレベル"],
  ["jockey_", "騎手"], ["trainer_", "調教師"], ["owner_", "馬主・生産者"], ["breeder_", "馬主・生産者"], ["ped_", "血統"], ["train_rank", "調教"],
];

export function shapGroupOf(feature: string): ShapGroup | null {
  if (EXACT[feature]) return EXACT[feature];
  return PREFIX.find(([p]) => feature.startsWith(p))?.[1] ?? null;
}

// 中身の表示用の、項目の短い名前（ないものは項目名のまま）
const LABELS: Record<string, string> = {
  ab_wmean: "能力指数（近走の直近加重平均）", ab_best: "能力指数（近走の最高）", ab_last: "能力指数（前走）", ab_trend: "能力指数の傾き",
  ab_wmean_rank: "能力指数の出走馬の中の順位", ab_wmean_dmax: "能力指数の、出走馬の最高との差", ab_wmean_dmean: "能力指数の、出走馬の平均との差",
  ab_wmean_d3: "能力指数の、3番手との差", ab_wmean_dtop3: "能力指数の、上位3頭との差", ab_best_dmax: "最高の能力指数の、出走馬の最高との差",
  tidx_mean5: "タイム指数（近5走平均）", tidx_best5: "タイム指数（近5走最高）", tidx_last1: "タイム指数（前走）",
  ag_dev_wmean: "推定上がりの速さ（直近加重平均）", ag_dev_best: "推定上がりの速さ（最高）", ag_rank_wmean: "推定上がりの順位",
  form_dev: "調子偏差値", form_raw: "調子（近5走）", form_finish_mean: "近5走の着順", form_time_mean: "近5走の勝ち馬との差",
  last1_finish_c: "前走の着順", last1_time_c: "前走の勝ち馬との差", n_prior: "過去の出走数", days_since_last: "前走からの間隔",
  cond_ab: "今回の条件での能力", cond_ab_rank: "今回の条件での能力の順位", cond_ab_dmax: "今回の条件での能力の、最高との差",
  fit_dist: "距離の得意さ", fit_venue: "競馬場の得意さ", fit_going: "馬場の得意さ", fit_turn: "回りの得意さ", fit_shape: "直線の長さの得意さ",
  surface_top3: "同じ芝ダの3着内率", dist_top3: "近い距離の3着内率", venue_top3: "同じ競馬場の3着内率", wet_top3: "稍重以上の3着内率",
  pos_first_mean: "最初のコーナーの位置", pos_last_mean: "最後のコーナーの位置", nige_rate: "逃げた割合", senko_rate: "先行した割合",
  rel_pos_rank: "出走馬の中の位置取りの順位", field_front_ratio: "先行馬の割合",
  prize_ratio_last: "クラスの上下（前走比）", prize_ratio_mean5: "クラスの上下（近5走比）", rel_prize_ratio_mean5: "クラスの、出走馬平均との差",
  lv_field_form: "前走の相手の調子", lv_field_tidx: "前走の相手のタイム指数", h2h_ahead: "今回の相手への先着率",
  jockey_top3: "騎手の今年の3着内率", jockey_venue_top3: "騎手の同じ競馬場の3着内率", jockey_course_top3: "騎手の同じコースの3着内率", jockey_top3_py: "騎手の前年の3着内率",
  trainer_top3: "調教師の今年の3着内率", trainer_venue_top3: "調教師の同じ競馬場の3着内率", trainer_top3_py: "調教師の前年の3着内率",
  owner_top3_y: "馬主の今年の3着内率", owner_top3_py: "馬主の前年の3着内率", breeder_top3_y: "生産者の今年の3着内率", breeder_top3_py: "生産者の前年の3着内率",
  ped_top3: "血統の今回の芝ダでの3着内率", ped_dist: "血統の距離の得意さ", ped_heavy: "血統の道悪倍率", ped_wet: "血統の稍重〜不良の倍率",
  train_rank: "調教評価", train_rank_rel: "調教評価の、出走馬平均との差", train_rank_pos: "調教評価の出走馬の中の順位",
  weight: "斤量", age: "年齢", sex: "性別", body_weight: "馬体重", body_weight_diff: "馬体重の増減",
  dist_change: "距離の変化", surface_change: "芝ダ替わり", weight_change: "斤量の変化", jockey_change: "騎手の乗り替わり",
  waku: "枠番", umaban: "馬番", field_size: "頭数", prize_1st: "1着賞金",
  jockey_venue_n: "騎手の同じ競馬場の騎乗数", jockey_course_n: "騎手の同じコースの騎乗数", jockey_n_py: "騎手の前年の騎乗数", hp_n: "ハイペースの出走数", sp_n: "スローペースの出走数",
  trainer_venue_n: "調教師の同じ競馬場の出走数", trainer_n_py: "調教師の前年の出走数", owner_n_py: "馬主の前年の出走数", breeder_n_py: "生産者の前年の出走数",
  fit_shape_n: "直線の得意さの元の走数", fit_going_n: "馬場の得意さの元の走数", fit_turn_n: "回りの得意さの元の走数", fit_change_n: "延長・短縮の得意さの元の走数",
  pace_hi_n: "ハイペースの走数", pace_sl_n: "スローペースの走数", lv_after_n: "前走の相手のその後の出走数", sim_straight_n: "直線の長さが近いコースの出走数",
  band_straight_n: "直線が同じ区分のコースの出走数", band_straight_top3: "直線が同じ区分のコースの3着内率", band_elev_n: "高低差が同じ区分のコースの出走数",
  band_elev_top3: "高低差が同じ区分のコースの3着内率", band_both_n: "直線・高低差が同じ区分のコースの出走数", sc_n: "似た条件の出走数", scp_n: "似た条件（ペースまで）の出走数",
  ab_std: "能力指数のばらつき", ab_n: "能力指数のある走の数", ab_wmean_z: "能力指数の出走馬の中の偏差", ab_wmean_fstd: "出走馬の能力指数のばらつき",
  ab_best_rank: "最高の能力指数の順位", ab_best_z: "最高の能力指数の偏差", ab_best_dmean: "最高の能力指数の、出走馬の平均との差",
  ab_wmean_top3_gap: "上位3頭の抜け具合", ab_wmean_close: "能力指数の近い馬の数",
  rel_tidx_mean5: "タイム指数の、出走馬平均との差", rank_tidx_mean5: "タイム指数の順位", rel_tidx_best5: "最高タイム指数の、出走馬平均との差", rank_tidx_best5: "最高タイム指数の順位",
  ag_rank_best: "推定上がりの最高の順位", early_gap_wmean: "600m地点の差", early_gap_best: "600m地点の差（最高）", fade_wmean: "先頭のラップとの差", fade_best: "先頭のラップとの差（最高）",
  ag_dev_wmean_rank: "推定上がりの出走馬の中の順位", ag_dev_wmean_z: "推定上がりの偏差", ag_dev_wmean_dmean: "推定上がりの、出走馬平均との差", ag_dev_wmean_dmax: "推定上がりの、最高との差", ag_dev_wmean_fstd: "出走馬の推定上がりのばらつき",
  g_last2: "直近2走の伸び", g_m3_m10: "近3走と近10走の差", g_last_best: "前走と最高の差", g_up_streak: "連続で良くなった数", g_layoff: "休み明けの変化",
  fit_dist_n: "距離の得意さの元の走数", fit_venue_n: "競馬場の得意さの元の走数", fit_change: "距離延長・短縮の得意さ", fit_surface: "芝ダの得意さ",
  cond_ab_d3: "今回の条件での能力の、3番手との差", cond_ab_dtop3: "今回の条件での能力の、上位3頭との差", cond_ab_top3_gap: "今回の条件での上位3頭の抜け具合", cond_ab_close: "今回の条件で能力の近い馬の数",
  surface_n: "同じ芝ダの出走数", dist_n: "近い距離の出走数", venue_n: "同じ競馬場の出走数", wet_n: "稍重以上の出走数",
  sim_straight_top3: "直線の長さが近いコースの3着内率", band_both_top3: "直線・高低差が同じ区分のコースの3着内率",
  sc_top3: "似た条件（距離・馬場・ペース）の3着内率", sc_ab_best: "似た条件での最高の能力指数", sc_ab_mean: "似た条件での能力指数の平均", sc_fin_best: "似た条件での最高の着順", sc_strong: "似た条件で強い相手と走った数", sc_strong_ab: "似た条件で強い相手と走ったときの能力指数",
  scp_top3: "似た条件（ペースまで）の3着内率", scp_ab_best: "似た条件（ペースまで）での最高の能力指数",
  hp_top3: "ハイペースでの3着内率", sp_top3: "スローペースでの3着内率", pace_hi: "ハイペースでの能力の差", pace_sl: "スローペースでの能力の差", pos_std: "位置取りのばらつき",
  field_nige_cands: "出走馬の逃げ候補の数", field_front: "出走馬の先行馬の数",
  rank_prize_ratio_mean5: "クラスの順位", field_form_mean: "出走馬全体の調子", lv_after_top3: "前走の相手のその後の3着内率", lv_beaten_form: "前走で先着した相手の調子", lv_lost_form: "前走で先着された相手の調子",
  h2h_n: "今回の相手との対戦数", h2h_adj: "直接対決の見込みとの差", h2h_margin: "直接対決のタイム差",
  jockey_n: "騎手の今年の騎乗数", trainer_n: "調教師の今年の出走数", owner_n_y: "馬主の今年の出走数", breeder_n_y: "生産者の今年の出走数",
  umaban_ratio: "馬番÷頭数", surface: "芝・ダート", distance: "距離", going: "馬場", venue: "競馬場", c_straight: "直線の長さ", c_elev: "高低差", c_lap: "一周の距離",
  last1_victim: "前走で不利を受けた", victim_n5: "近5走で不利を受けた数",
  c4_rate: "このコースで、ふだんの4角の位置からの3着内率", c4_adv: "このコースで、ふだんの4角の位置の有利さ", c4_bias: "このコースの前残りの強さ", c4_n: "このコース・その位置の走の数",
};
export const featureLabel = (name: string) => LABELS[name] ?? name;

export type GroupContribution = { group: ShapGroup; phi: number; top: { name: string; phi: number }[] };

/** 1頭の SHAP（features と同じ並び）を観点ごとに足す。top は観点の中で寄与の大きい項目（絶対値の大きい順に3つ）. */
export function groupShap(features: string[], phi: number[]): GroupContribution[] {
  const out = new Map<ShapGroup, GroupContribution>();
  features.forEach((name, k) => {
    const g = shapGroupOf(name);
    if (!g) return;
    const c = out.get(g) ?? { group: g, phi: 0, top: [] };
    c.phi += phi[k];
    c.top.push({ name, phi: phi[k] });
    out.set(g, c);
  });
  for (const c of out.values()) c.top = c.top.sort((a, b) => Math.abs(b.phi) - Math.abs(a.phi)).slice(0, 3);
  return SHAP_GROUPS.filter((g) => out.has(g)).map((g) => out.get(g)!);
}
