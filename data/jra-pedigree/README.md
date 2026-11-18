# JRA 出走馬の血統データ

`data/jra-seiseki`（成績表PDFの変換結果）に載っている出走馬について、JRA 公式サイトの
[過去のレース結果](https://www.jra.go.jp/JRADB/accessS.html) と「競走馬情報」ページから馬IDと血統を取得したものです。
`scripts/jra_seiseki/pedigree.py`（`python -m jra_seiseki pedigree`）で生成します。

## ファイル構成

| パス | 内容 |
| --- | --- |
| `race_horses/{年}.csv` | 1行 = 1出走馬。`race_id`・`umaban`（馬番）→ `horse_id` の対応 |
| `horses.csv` | 1行 = 1頭の血統・プロフィール |

`horse_id` は JRA の10桁の馬IDで、netkeiba の `horse_id`（アプリの `horseId`）と同じ値です。
`data/jra-seiseki/csv/{年}/results.csv` にも `horse_id`・`sire`・`dam`・`damsire` 列として結合されます。

### race_horses/{年}.csv

| 列 | 説明 |
| --- | --- |
| `race_id` | 12桁のレースID（`results.csv` と同じ） |
| `umaban` | 馬番 |
| `horse` | JRA サイト上の馬名 |
| `horse_id` | 馬ID |
| `jradb_cname` | 競走馬情報ページを開くためのキー（取得処理用） |

### horses.csv

| 列 | 説明 |
| --- | --- |
| `horse_id`, `name`, `name_en`, `country` | 馬ID・馬名・欧字名・生産国（`JPN` 等） |
| `sex`, `color`, `birth_date` | 性別・毛色・生年月日 |
| `sire`, `sire_id` | 父（JRAに登録がある馬なら馬IDも） |
| `dam`, `dam_id` | 母 |
| `damsire`, `damdam` | 母の父・母の母 |
| `breeder`, `region`, `market` | 生産牧場・産地・取引市場 |
| `retired_date` | 抹消年月日（現役馬は空） |
| `fetched_at` | 取得日 |

## 更新方法

- GitHub Actions `.github/workflows/jra-pedigree.yml` が、成績表の変換ワークフロー完了後と毎日1回、
  未取得のレース・馬を取得してコミットします。1回あたり約5時間で打ち切り、次回は続きから再開します
  （初回は全5年分で約4〜5万ページあるため、数日かけて揃います）。
- 手元で実行する場合（標準ライブラリのみで動きます）:

  ```sh
  cd scripts
  python -m jra_seiseki pedigree                      # 変換済み全年の未取得分
  python -m jra_seiseki pedigree --year 2026 --max-minutes 30
  python -m jra_seiseki pedigree --year 2026 --limit 5   # 動作確認（5レース・5頭だけ）
  python -m jra_seiseki pedigree --refresh-horses     # 取得済みの馬も再取得（抹消日などの更新）
  ```

## 取得の仕組み

JRA サイトのページはサイトが発行したキー（`cname`）でしか開けないため、ページ内のリンクを順に辿ります:
過去レース検索トップ（月ごとのキー表）→ 月の開催一覧 → 開催日 → レース結果（出走馬の馬IDリンク）→ 競走馬情報。
リクエスト間隔は既定で 0.5 秒以上、同時2接続です（`--wait`, `--workers`）。
PDF と JRA サイトで馬名が異なる（外字など）場合も、馬番で対応付けます。
