# 成績表の備考から「前走で不利を受けた（被害馬）」「前走で発走時に問題」を作り、1年ずつ外す比較
import sys, re, json, csv, glob, math
sys.path.insert(0, '/home/user/keiba-predictor/scripts/ml')
import numpy as np, lightgbm as lgb
from sklearn.metrics import log_loss, roc_auc_score
from features import FEATURES, build_dataset
from train import DATA, OUT, PARAMS, TRAIN_FROM, race_top3_hit
victims, starts, slow = set(), set(), set()   # (race_id, umaban) / (race_id, horse)
for f in glob.glob('/home/user/keiba-predictor/data/jra-seiseki/csv/*/races.csv'):
    for r in csv.DictReader(open(f)):
        for part in r['notes'].split(' / '):
            if '〔制裁〕' in part:
                for grp in re.findall(r'被害馬[:：]([^)）]+)', part):
                    for u in re.findall(r'(\d+)番', grp): victims.add((r['race_id'], int(u)))
            if '発走状況' in part:
                for sent in re.split(r'。', part):
                    for name in re.findall(r'([ァ-ヴー・Ａ-Ｚａ-ｚA-Za-z0-9]+)号は', sent):
                        starts.add((r['race_id'], name))
                        if '出遅れ' in sent: slow.add((r['race_id'], name))
print('victims', len(victims), 'start', len(starts), 'slow', len(slow), file=sys.stderr)
rows, _ = build_dataset(DATA)
by_horse = {}
for r in rows: by_horse.setdefault(r['horse'], []).append(r)
NEW = ['last1_victim', 'victim_n5', 'last1_start_trouble', 'start_trouble_n5']
for runs in by_horse.values():
    runs.sort(key=lambda r: r['date'])
    flags = []
    for r in runs:
        u = r['umaban']; u = None if u is None or (isinstance(u, float) and math.isnan(u)) else int(u)
        prev = flags[-5:]
        r['last1_victim'] = prev[-1][0] if prev else np.nan
        r['victim_n5'] = sum(p[0] for p in prev) if prev else np.nan
        r['last1_start_trouble'] = prev[-1][1] if prev else np.nan
        r['start_trouble_n5'] = sum(p[1] for p in prev) if prev else np.nan
        flags.append((int((r['race_id'], u) in victims), int((r['race_id'], r['horse']) in starts)))
rows = [r for r in rows if r['date'] >= TRAIN_FROM]
print('last1_victim=1:', sum(r['last1_victim'] == 1 for r in rows), 'last1_start=1:', sum(r['last1_start_trouble'] == 1 for r in rows), file=sys.stderr)
rounds = json.loads((OUT / 'metrics.json').read_text())['best_iteration']
SETS = {'base': FEATURES, '+victim': FEATURES + NEW[:2], '+victim+start': FEATURES + NEW}
years = sorted({r['date'][:4] for r in rows})
result = {}
for name, feats in SETS.items():
    per = {}
    for year in years:
        tr = [r for r in rows if r['date'][:4] != year]; te = [r for r in rows if r['date'][:4] == year]
        m = lambda rs: np.array([[r[f] for f in feats] for r in rs], dtype=np.float64)
        b = lgb.train(PARAMS, lgb.Dataset(m(tr), label=[r['label'] for r in tr]), num_boost_round=rounds)
        p = b.predict(m(te)); y = [r['label'] for r in te]
        per[year] = {'logloss': log_loss(y, p), 'auc': roc_auc_score(y, p), 'top3_hit': race_top3_hit(te, p)}
        print(name, year, per[year], file=sys.stderr)
    result[name] = {'per_year': per, 'mean': {k: float(np.mean([v[k] for v in per.values()])) for k in ('logloss', 'auc', 'top3_hit')}}
# 単純集計：前走で不利を受けた馬の3着内率
v1 = [r for r in rows if r['last1_victim'] == 1]; v0 = [r for r in rows if r['last1_victim'] == 0]
result['raw'] = {'victim_top3': sum(r['label'] for r in v1) / len(v1), 'victim_n': len(v1), 'other_top3': sum(r['label'] for r in v0) / len(v0)}
json.dump(result, open(OUT / 'feature_comparison_furi.json', 'w'), ensure_ascii=False, indent=1)
print(json.dumps({k: v.get('mean', v) for k, v in result.items()}, ensure_ascii=False, indent=1))
