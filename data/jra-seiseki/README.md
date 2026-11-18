# JRA レース成績データ（成績表PDFの変換結果）

[JRA レース成績データ](https://www.jra.go.jp/datafile/seiseki/) の「成績表（PDF）」
（`report/{年}.html` から辿れる開催日・競馬場ごとの PDF、例 `2026-5tokyo4.pdf` = 2026年 5回東京 4日目）を
`scripts/jra_seiseki` で変換したものです。発売票数 PDF（`…-hyo-…`）は対象外です。

## 更新方法

- GitHub Actions `.github/workflows/jra-seiseki.yml` が土日の夜と月曜に今年分の未変換 PDF を取得・変換してコミットします。
  過去年は Actions の「Run workflow」で `years` に `2025 2024` のように指定します。
- 手元で実行する場合:

  ```sh
  pip install -r scripts/jra_seiseki/requirements.txt
  cd scripts
  python -m jra_seiseki update                  # 今年の未変換分
  python -m jra_seiseki update --year 2025      # 指定年
  python -m jra_seiseki parse some.pdf --pretty # 1ファイルを JSON で標準出力へ
  ```

## ファイル構成

| パス | 内容 |
| --- | --- |
| `json/{年}/{ファイル名}.json` | 1開催日・1競馬場分の全情報（下記スキーマ） |
| `csv/{年}/races.csv` | 1行 = 1レース |
| `csv/{年}/results.csv` | 1行 = 1頭（着順順） |
| `csv/{年}/payouts.csv` | 1行 = 1払戻（的中組番ごと） |
| `manifest.json` | 変換済み PDF の URL・SHA-256・警告 |

`race_id` は netkeiba 形式の12桁 `YYYY + 場コード(2) + 回(2) + 日目(2) + レース番号(2)` で、アプリ側の `raceId` と同じです。
場コード: 01札幌 02函館 03福島 04新潟 05東京 06中山 07中京 08京都 09阪神 10小倉。

## results.csv の主な列

| 列 | 説明 |
| --- | --- |
| `finish` | 着順（同着は同順位。取消・除外・中止は空） |
| `status` | `取消` / `除外` / `中止` / `失格`（通常は空） |
| `waku`, `umaban` | 枠番・馬番 |
| `horse_mark` | `(外)` 外国産馬 / `[外]` 外国調教馬 / `(地)` 地方競馬出身馬 / `[地]` 地方競馬所属馬 |
| `sex`, `age`, `color` | 性（牡/牝/セ）・年齢・毛色 |
| `weight` / `weight_before_allowance` | 負担重量 / 減量前の重量（見習騎手の減量がある場合） |
| `jockey`, `jockey_mark` | 騎手（外国人騎手は `(仏)` 等の国名付き）・減量記号 `▲△☆◇★` |
| `trainer`, `owner`, `region`, `breeder` | 調教師・馬主・産地・生産牧場 |
| `equipment` | 馬装具（`B` ブリンカー 等） |
| `body_weight`, `body_weight_diff` | 馬体重・前走比増減 |
| `time`, `time_sec` | 走破タイム（`〃` は前の馬と同タイムとして展開済み） |
| `margin` | 着差（`½` `1¼` `クビ` `ハナ` `アタマ` `大差` `同着` など、PDF表記のまま） |
| `odds`, `popularity` | 単勝オッズ・単勝人気 |
| `corner_positions` | 各コーナー通過順位（`races.csv` の `corners` の順に `-` 区切り） |
| `horse_id`, `sire`, `dam`, `damsire` | 馬ID・父・母・母の父（`data/jra-pedigree` を取得済みの馬のみ。詳細は同ディレクトリの README） |

## races.csv の主な列

`race_name`（特別レース名 or 条件）, `conditions`（出走条件）, `weight_rule`（負担重量）, `marks`（`(国際)(指定)(混合)(特指)[指定]` 等）,
`surface`（芝/ダート/障害）, `course`（例 `芝・右・外`）, `distance`(m), `weather`, `going`（馬場状態）, `start_time`,
`starters`（出走頭数）, `runners`（取消・除外を含む行数）, `prize_1st`（1着本賞）, `laps`（ハロンタイム、`-` 区切り秒）,
`passing_times`（通過タイム）, `last_4f` / `last_3f`（レース上り）, `corners`（`3:8,9(7,14,16)… 4:…` 通過順位の原文。
PDF の「通過コーナー順」の表記どおりで、長距離戦の2周目などは `③` `④` のように丸数字のラベルになります）, `notes`（競走中止・騎手変更などの付記）。
障害レースは `surface=障害`、`course` に走路（`芝` / `芝・ダート` など）が入ります。

## payouts.csv

`type`（単勝/複勝/枠連/馬連/馬単/ワイド/3連複/3連単/5重勝）, `combo`（馬番・枠番を `-` 区切り。馬単・3連単は着順通り）,
`payout`（100円あたり払戻金）, `popularity`（その組番の人気）, `votes`（的中票数）, `note`（`発売なし` 等）。

## 変換の仕組みと注意

- PDF は2段組で、人気の丸数字・着差の分数・(地)(外) 等の記号・一部の外字は独自の埋め込みフォント（ToUnicode なし）で描かれています。
  グリフの輪郭ハッシュ → 文字の対応表 `scripts/jra_seiseki/glyph_map.json` で復元しています。
  未知のグリフは `〓` として出力し、`manifest.json` の `unknown_glyphs` に記録されます（対応表に追記して `--force` で再変換）。
- 人気が読めなかった場合のみオッズ順から推定し、JSON の該当馬に `popularity_estimated: true` を付けます。
- 各ファイルの `warnings`（馬番の欠落、タイム欠落など）は `manifest.json` に残ります。
