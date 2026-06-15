# generate.py — versione GitHub Actions (no Colab)
import os, json, subprocess, sys, itertools, random
from datetime import datetime
import pandas as pd
import numpy as np

# ── KAGGLE SETUP (legge dai secrets GitHub) ──────────────────────
os.makedirs(os.path.expanduser('~/.kaggle'), exist_ok=True)
kaggle_cfg = {
    "username": os.environ["KAGGLE_USERNAME"],
    "key": os.environ["KAGGLE_KEY"]
}
with open(os.path.expanduser('~/.kaggle/kaggle.json'), 'w') as f:
    json.dump(kaggle_cfg, f)
os.chmod(os.path.expanduser('~/.kaggle/kaggle.json'), 0o600)

# ── DOWNLOAD DATASET ─────────────────────────────────────────────
subprocess.run(["kaggle", "datasets", "download", "-d",
    "martj42/international-football-results-from-1872-to-2017", "--unzip"], check=True)
subprocess.run(["kaggle", "datasets", "download", "-d",
    "cashncarry/fifaworldranking", "--unzip"], check=True)

print("✅ Dataset scaricati")

# ── CARICAMENTO ──────────────────────────────────────────────────
risultati = pd.read_csv('results.csv')
ranking   = pd.read_csv('fifa_ranking-2024-06-20.csv')

risultati['date']     = pd.to_datetime(risultati['date'])
ranking['rank_date']  = pd.to_datetime(ranking['rank_date'])

# ── FEATURE ENGINEERING ──────────────────────────────────────────
TORNEI_PESO = {
    'FIFA World Cup': 5, 'FIFA World Cup qualification': 3,
    'UEFA Euro': 4, 'Copa América': 4, 'African Cup of Nations': 4,
    'AFC Asian Cup': 4, 'Gold Cup': 3, 'UEFA Euro qualification': 2,
    'African Cup of Nations qualification': 2, 'AFC Asian Cup qualification': 2,
    'UEFA Nations League': 2, 'CONCACAF Nations League': 2, 'Friendly': 1,
}

df = risultati[risultati['date'] >= '2010-01-01'].copy()
df['tournament_weight'] = df['tournament'].map(TORNEI_PESO).fillna(1)

ranking_sorted = ranking.sort_values('rank_date')

def get_rank_at_date(team, date):
    mask = (ranking_sorted['country_full'] == team) & (ranking_sorted['rank_date'] <= date)
    rows = ranking_sorted[mask]
    if not rows.empty:
        return rows.iloc[-1]['rank'], rows.iloc[-1]['total_points']
    return 100, 500

def forma_squadra(team, date, n=10):
    partite = df[
        ((df['home_team'] == team) | (df['away_team'] == team)) &
        (df['date'] < date)
    ].tail(n)
    if partite.empty:
        return 0, 0, 0, 0
    punti_tot, gol_fatti, gol_subiti, peso_tot = 0, 0, 0, 0
    for _, r in partite.iterrows():
        peso = r['tournament_weight']
        gf, gs = (r['home_score'], r['away_score']) if r['home_team'] == team else (r['away_score'], r['home_score'])
        gol_fatti += gf * peso; gol_subiti += gs * peso; peso_tot += peso
        if gf > gs: punti_tot += 3 * peso
        elif gf == gs: punti_tot += 1 * peso
    return (punti_tot/peso_tot if peso_tot else 0,
            gol_fatti/peso_tot if peso_tot else 0,
            gol_subiti/peso_tot if peso_tot else 0,
            len(partite))

def head_to_head(team_a, team_b, date, n=10):
    scontri = df[
        (((df['home_team']==team_a) & (df['away_team']==team_b)) |
         ((df['home_team']==team_b) & (df['away_team']==team_a))) &
        (df['date'] < date)
    ].tail(n)
    if scontri.empty: return 0, 0, 0
    v_a, par, v_b = 0, 0, 0
    for _, r in scontri.iterrows():
        hs, as_ = (r['home_score'], r['away_score']) if r['home_team']==team_a else (r['away_score'], r['home_score'])
        if hs > as_: v_a += 1
        elif hs == as_: par += 1
        else: v_b += 1
    tot = len(scontri)
    return v_a/tot, par/tot, v_b/tot

print("Costruendo feature dataset...")
records = []
for _, row in df.iterrows():
    if pd.isna(row['home_score']) or pd.isna(row['away_score']): continue
    date, home, away = row['date'], row['home_team'], row['away_team']
    rank_h, pts_h = get_rank_at_date(home, date)
    rank_a, pts_a = get_rank_at_date(away, date)
    punti_h, gf_h, gs_h, _ = forma_squadra(home, date)
    punti_a, gf_a, gs_a, _ = forma_squadra(away, date)
    h2h_h, h2h_par, h2h_a = head_to_head(home, away, date)
    if row['home_score'] > row['away_score']: target = 0
    elif row['home_score'] == row['away_score']: target = 1
    else: target = 2
    records.append({
        'date': date, 'home_team': home, 'away_team': away,
        'home_score': row['home_score'], 'away_score': row['away_score'],
        'tournament': row['tournament'], 'tournament_weight': row['tournament_weight'],
        'neutral': int(row['neutral']),
        'rank_home': rank_h, 'rank_away': rank_a,
        'diff_rank': rank_h - rank_a, 'pts_home': pts_h, 'pts_away': pts_a,
        'diff_pts': pts_h - pts_a,
        'forma_punti_home': punti_h, 'forma_punti_away': punti_a, 'diff_forma': punti_h - punti_a,
        'gol_fatti_home': gf_h, 'gol_fatti_away': gf_a,
        'gol_subiti_home': gs_h, 'gol_subiti_away': gs_a,
        'h2h_home': h2h_h, 'h2h_away': h2h_a, 'h2h_par': h2h_par,
        'risultato': target,
    })

features_df = pd.DataFrame(records)
print(f"✅ {len(features_df)} partite")

# ── TRAINING ─────────────────────────────────────────────────────
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, log_loss

FEATURE_COLS = [
    'diff_rank', 'diff_pts', 'forma_punti_home', 'forma_punti_away', 'diff_forma',
    'gol_fatti_home', 'gol_fatti_away', 'gol_subiti_home', 'gol_subiti_away',
    'h2h_home', 'h2h_away', 'h2h_par', 'neutral', 'tournament_weight',
    'rank_home', 'rank_away',
]

features_df = features_df.sort_values('date').reset_index(drop=True)
X = features_df[FEATURE_COLS].fillna(0)
y = features_df['risultato']

split_idx = int(len(X) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

xgb_model = xgb.XGBClassifier(
    n_estimators=300, learning_rate=0.05, max_depth=4,
    subsample=0.8, colsample_bytree=0.8,
    eval_metric='mlogloss', random_state=42, n_jobs=-1
)

modello_finale = CalibratedClassifierCV(xgb_model, method='isotonic', cv=5)
print("Training in corso...")
modello_finale.fit(X_train, y_train)

y_pred  = modello_finale.predict(X_test)
y_proba = modello_finale.predict_proba(X_test)
acc = accuracy_score(y_test, y_pred)
ll  = log_loss(y_test, y_proba)
print(f"✅ Accuracy: {acc:.2%} | Log Loss: {ll:.4f}")

# ── ORACOLO ──────────────────────────────────────────────────────
SOGLIA_PAREGGIO = 0.27
NOW = pd.Timestamp.now()

def predici_con_soglia(proba):
    risultati_pred = []
    for p in proba:
        if p[1] >= SOGLIA_PAREGGIO: risultati_pred.append(1)
        else: risultati_pred.append(int(np.argmax([p[0], p[1], p[2]])))
    return risultati_pred

def oracolo(home, away, neutro=True):
    r_h, pts_h = get_rank_at_date(home, NOW)
    r_a, pts_a = get_rank_at_date(away, NOW)
    p_h, gf_h, gs_h, _ = forma_squadra(home, NOW)
    p_a, gf_a, gs_a, _ = forma_squadra(away, NOW)
    h2h_h, h2h_par, h2h_a = head_to_head(home, away, NOW)
    match = pd.DataFrame([{
        'diff_rank': r_h-r_a, 'diff_pts': pts_h-pts_a,
        'forma_punti_home': p_h, 'forma_punti_away': p_a, 'diff_forma': p_h-p_a,
        'gol_fatti_home': gf_h, 'gol_fatti_away': gf_a,
        'gol_subiti_home': gs_h, 'gol_subiti_away': gs_a,
        'h2h_home': h2h_h, 'h2h_away': h2h_a, 'h2h_par': h2h_par,
        'neutral': int(neutro), 'tournament_weight': 5,
        'rank_home': r_h, 'rank_away': r_a,
    }])
    proba = modello_finale.predict_proba(match)[0]
    pred = predici_con_soglia([proba])[0]
    return {'prob_home': round(float(proba[0]),3), 'prob_draw': round(float(proba[1]),3),
            'prob_away': round(float(proba[2]),3), 'prediction': pred}

# ── GIRONI ───────────────────────────────────────────────────────
GIRONI = {
    'A': ['Mexico','South Korea','South Africa','Czech Republic'],
    'B': ['Canada','Switzerland','Qatar','Bosnia and Herzegovina'],
    'C': ['Brazil','Morocco','Haiti','Scotland'],
    'D': ['United States','Paraguay','Australia','Turkey'],
    'E': ['Germany','Ecuador','Ivory Coast','Curacao'],
    'F': ['Netherlands','Japan','Tunisia','Sweden'],
    'G': ['Belgium','Egypt','Iran','New Zealand'],
    'H': ['Spain','Uruguay','Saudi Arabia','Cape Verde'],
    'I': ['France','Senegal','Norway','Iraq'],
    'J': ['Argentina','Algeria','Austria','Jordan'],
    'K': ['Portugal','Colombia','Uzbekistan','DR Congo'],
    'L': ['England','Croatia','Ghana','Panama'],
}

mondiali_reali = risultati[
    (risultati['tournament'] == 'FIFA World Cup') &
    (risultati['date'] >= '2026-06-01')
].copy()

risultati_reali = {}
for _, r in mondiali_reali.iterrows():
    key = (r['home_team'], r['away_team'])
    risultati_reali[key] = {
        'home_score': int(r['home_score']) if not pd.isna(r['home_score']) else None,
        'away_score': int(r['away_score']) if not pd.isna(r['away_score']) else None,
    }

def calcola_girone(lettera, squadre):
    partite = []
    for home, away in itertools.combinations(squadre, 2):
        reale_key, invertita = None, False
        if (home, away) in risultati_reali: reale_key = (home, away)
        elif (away, home) in risultati_reali: reale_key = (away, home); invertita = True
        if reale_key:
            r = risultati_reali[reale_key]
            if r['home_score'] is not None:
                hs, as_ = (r['away_score'], r['home_score']) if invertita else (r['home_score'], r['away_score'])
                partite.append({'home':home,'away':away,'status':'played',
                    'home_score':hs,'away_score':as_,'prob_home':None,'prob_draw':None,'prob_away':None,'prediction':None})
                continue
        try:
            pred = oracolo(home, away)
            partite.append({'home':home,'away':away,'status':'predicted',
                'home_score':None,'away_score':None,
                'prob_home':pred['prob_home'],'prob_draw':pred['prob_draw'],'prob_away':pred['prob_away'],
                'prediction':pred['prediction']})
        except Exception as e:
            print(f"  ⚠️ {home} vs {away}: {e}")
            partite.append({'home':home,'away':away,'status':'error',
                'home_score':None,'away_score':None,'prob_home':0.33,'prob_draw':0.33,'prob_away':0.34,'prediction':1})
    return partite

def calcola_classifica(squadre, partite):
    cls = {s: {'punti':0,'gf':0,'gs':0,'pg':0} for s in squadre}
    for p in partite:
        if p['status'] != 'played':
            pred = p['prediction'] if p['prediction'] is not None else 0
            if pred == 0: cls[p['home']]['punti'] += 3
            elif pred == 1: cls[p['home']]['punti'] += 1; cls[p['away']]['punti'] += 1
            else: cls[p['away']]['punti'] += 3
        else:
            hs, as_ = p['home_score'], p['away_score']
            cls[p['home']]['gf'] += hs; cls[p['home']]['gs'] += as_
            cls[p['away']]['gf'] += as_; cls[p['away']]['gs'] += hs
            if hs > as_: cls[p['home']]['punti'] += 3
            elif hs == as_: cls[p['home']]['punti'] += 1; cls[p['away']]['punti'] += 1
            else: cls[p['away']]['punti'] += 3
        cls[p['home']]['pg'] += 1; cls[p['away']]['pg'] += 1
    return sorted(cls.items(), key=lambda x: (-x[1]['punti'], -(x[1]['gf']-x[1]['gs'])))

# ── GENERA GIRONI ─────────────────────────────────────────────────
print("\nGenerando predizioni per tutti i gironi...")
output = {
    'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M'),
    'model_accuracy': f'{acc:.0%}',
    'model_logloss': f'{ll:.4f}',
    'groups': {}
}

for lettera, squadre in GIRONI.items():
    print(f"  Girone {lettera}...")
    partite = calcola_girone(lettera, squadre)
    classifica = calcola_classifica(squadre, partite)
    output['groups'][lettera] = {
        'teams': squadre, 'matches': partite,
        'standings': [
            {'team': team, 'points': stats['punti'], 'played': stats['pg'],
             'gf': stats['gf'], 'gs': stats['gs'], 'advances': idx < 2}
            for idx, (team, stats) in enumerate(classifica)
        ]
    }

# ── MONTE CARLO ──────────────────────────────────────────────────
print("\nPre-calcolo probabilità squadre...")
TUTTE = [s for sq in GIRONI.values() for s in sq]
squad_cache = {}
for team in TUTTE:
    rank, pts = get_rank_at_date(team, NOW)
    punti_f, gf, gs, _ = forma_squadra(team, NOW)
    squad_cache[team] = {'rank':rank,'pts':pts,'forma':punti_f,'gf':gf,'gs':gs}

prob_cache = {}
for i, t1 in enumerate(TUTTE):
    for t2 in TUTTE[i+1:]:
        p = oracolo(t1, t2)
        prob_cache[(t1,t2)] = [p['prob_home'], p['prob_draw'], p['prob_away']]
        prob_cache[(t2,t1)] = [p['prob_away'], p['prob_draw'], p['prob_home']]

def simula_ko(a, b):
    p = prob_cache.get((a,b), [0.33,0.33,0.34])
    r = random.random()
    if r < p[0]: return a
    elif r < p[0]+p[1]: return a if random.random()<0.5 else b
    else: return b

def simula_girone_fast(squadre):
    cls = {s:{'pt':0,'gf':0,'gs':0} for s in squadre}
    for home, away in itertools.combinations(squadre, 2):
        reale = risultati_reali.get((home,away)) or risultati_reali.get((away,home))
        if reale and reale['home_score'] is not None:
            inv = (away,home) in risultati_reali and (home,away) not in risultati_reali
            hs, as_ = (reale['away_score'],reale['home_score']) if inv else (reale['home_score'],reale['away_score'])
        else:
            p = prob_cache.get((home,away),[0.33,0.33,0.34])
            r = random.random()
            if r < p[0]: hs,as_ = 2,0
            elif r < p[0]+p[1]: hs,as_ = 1,1
            else: hs,as_ = 0,2
        cls[home]['gf']+=hs; cls[home]['gs']+=as_; cls[away]['gf']+=as_; cls[away]['gs']+=hs
        if hs>as_: cls[home]['pt']+=3
        elif hs==as_: cls[home]['pt']+=1; cls[away]['pt']+=1
        else: cls[away]['pt']+=3
    return sorted(cls.items(), key=lambda x:(-x[1]['pt'],-(x[1]['gf']-x[1]['gs']),-x[1]['gf']))

def simula_torneo():
    primi, secondi, terze = [], [], []
    for g, sq in GIRONI.items():
        cls = simula_girone_fast(sq)
        primi.append(cls[0][0]); secondi.append(cls[1][0]); terze.append((cls[2][0],cls[2][1]['pt']))
    terze_ok = [t for t,_ in sorted(terze,key=lambda x:-x[1])[:8]]
    r32 = []
    for i in range(12): r32.append((primi[i], terze_ok[i%8]))
    for i in range(0,12,2): r32.append((secondi[i],secondi[i+1]))
    seen, clean = set(), []
    for a,b in r32:
        k = frozenset([a,b])
        if k not in seen and a!=b: seen.add(k); clean.append((a,b))
        if len(clean)==16: break
    def fase(partite):
        v = [simula_ko(a,b) for a,b in partite]
        return v, [(v[i],v[i+1]) for i in range(0,len(v)-1,2)]
    v32,q = fase(clean); v8,s = fase(q); v4,f = fase(s)
    return simula_ko(f[0][0],f[0][1]) if f else v4[0]

N = 10000
print(f"Lancio {N} simulazioni Monte Carlo...")
conteggio = {}
for _ in range(N):
    v = simula_torneo()
    conteggio[v] = conteggio.get(v,0)+1

odds = {t: round(c/N*100,1) for t,c in sorted(conteggio.items(),key=lambda x:-x[1])}
output['world_cup_winner_odds'] = odds

# ── EXPORT ───────────────────────────────────────────────────────
with open('predictions.json','w',encoding='utf-8') as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

print("\n✅ predictions.json generato!")
print("🏆 Top 5:")
for t,p in list(odds.items())[:5]:
    print(f"  {t}: {p}%")