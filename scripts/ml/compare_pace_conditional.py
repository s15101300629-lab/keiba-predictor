# 「ペースがこうなったら」を入力にするモデルの検証（1年ずつ外す）
# - 今のモデル / ペースを入力に加えたモデルに「実際のペース」を入れた場合 / 「違うペース」を入れた場合 / 全部「ミドル」の場合
import sys, json
sys.path.insert(0, '/home/user/keiba-predictor/scripts/ml')
import numpy as np, lightgbm as lgb
from sklearn.metrics import log_loss, roc_auc_score
from features import FEATURES, build_dataset, load_races
from train import DATA, OUT, PARAMS, TRAIN_FROM, race_top3_hit
pace = {r['race_id']: r['pace'] for r in load_races(DATA)}
rows, _ = build_dataset(DATA)
rows = [r for r in rows if r['date'] >= TRAIN_FROM and pace.get(r['race_id']) is not None]
for r in rows: r['race_pace'] = pace[r['race_id']]
rounds = json.loads((OUT / 'metrics.json').read_text())['best_iteration']
F2 = FEATURES + ['race_pace']
def mat(rs, feats, override=None):
    return np.array([[(override(r) if (f == 'race_pace' and override) else r[f]) for f in feats] for r in rs], dtype=np.float64)
def metrics(rs, p):
    y = [r['label'] for r in rs]
    return {'logloss': log_loss(y, p), 'auc': roc_auc_score(y, p), 'top3_hit': race_top3_hit(rs, p)}
years = sorted({r['date'][:4] for r in rows})
res = {}
for year in years:
    tr = [r for r in rows if r['date'][:4] != year]; te = [r for r in rows if r['date'][:4] == year]
    y = [r['label'] for r in tr]
    a = lgb.train(PARAMS, lgb.Dataset(mat(tr, FEATURES), label=y), rounds)
    b = lgb.train(PARAMS, lgb.Dataset(mat(tr, F2), label=y), rounds)
    out = {'今のモデル': metrics(te, a.predict(mat(te, FEATURES))),
           '実際のペース': metrics(te, b.predict(mat(te, F2))),
           '違うペース(+1)': metrics(te, b.predict(mat(te, F2, lambda r: (r['race_pace'] + 1) % 3))),
           '違うペース(+2)': metrics(te, b.predict(mat(te, F2, lambda r: (r['race_pace'] + 2) % 3))),
           '全部ミドル': metrics(te, b.predict(mat(te, F2, lambda r: 1)))}
    # 実際のペース別：今のモデル vs 実際のペースを入れたモデル
    for pc, name in [(0, 'スロー'), (1, 'ミドル'), (2, 'ハイ')]:
        sub = [r for r in te if r['race_pace'] == pc]
        out[f'{name}のレース'] = {'races': len({r["race_id"] for r in sub}),
                                 '今のモデル': metrics(sub, a.predict(mat(sub, FEATURES))),
                                 '実際のペース': metrics(sub, b.predict(mat(sub, F2)))}
    res[year] = out
    print(year, json.dumps({k: v for k, v in out.items() if 'レース' not in k}, ensure_ascii=False), file=sys.stderr)
json.dump(res, open(OUT / 'pace_conditional.json', 'w'), ensure_ascii=False, indent=1)
print('done')
