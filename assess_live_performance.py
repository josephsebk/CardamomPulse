#!/usr/bin/env python3
"""
Score the live forecast ledger against a naive random walk.

The ledger DB is gitignored, so the evaluation set is reconstructed from the
daily snapshots of cardamom_webapp/data/archive.csv in git history: each
snapshot records the forecasts still pending on that date, so their union
recovers every horizon, not just the 7-day slice the current file retains.

Forecasts are built as anchor*exp(return), so predicting a zero return IS the
random walk. Scoring against it isolates the model's only contribution and
avoids reading a price-level MAPE that the anchor largely supplies.

Setup (the repo clone is shallow by default):
  git fetch --unshallow origin main
  mkdir -p /tmp/hist
  for c in $(git log --format=%H origin/main -- cardamom_webapp/data/archive.csv); do
      git show $c:cardamom_webapp/data/archive.csv > /tmp/hist/$c.csv
  done
  git show origin/main:cardamom_webapp/data/archive.csv > /tmp/now/archive.csv

Then point SD at the parent of hist2/ and now/ below.
"""

import pandas as pd, numpy as np, glob
from math import sqrt
from statistics import NormalDist
SD='/tmp/claude-0/-home-user-CardamomPulse/90d62be0-610b-5de8-b3d7-206a7dbd9636/scratchpad'

# ── reconstruct full forecast ledger from all 204 production snapshots ──
frames=[]
for f in glob.glob(f'{SD}/hist2/*.csv'):
    try: d=pd.read_csv(f)
    except Exception: continue
    d=d.dropna(subset=['predicted_avg_price_inr_per_kg','model_run_date','horizon_days'])
    frames.append(d[['date','predicted_avg_price_inr_per_kg','model_run_date','horizon_days','model_version']])
led=pd.concat(frames,ignore_index=True)
led.columns=['target_date','pred','run_date','horizon','model_version']
led=led.drop_duplicates(subset=['target_date','run_date','horizon'])
led['target_date']=pd.to_datetime(led['target_date']); led['run_date']=pd.to_datetime(led['run_date'])
print(f"Forecast ledger reconstructed: {len(led)} unique forecasts")
print(f"  run dates {led['run_date'].min().date()} -> {led['run_date'].max().date()}")

a=pd.read_csv(f'{SD}/now/archive.csv'); a['date']=pd.to_datetime(a['date'])
px=(a.dropna(subset=['actual_avg_price_inr_per_kg']).drop_duplicates(subset=['date'])
    [['date','actual_avg_price_inr_per_kg']].rename(columns={'actual_avg_price_inr_per_kg':'actual'})
    .sort_values('date').reset_index(drop=True))

led=led.merge(px,left_on='target_date',right_on='date',how='left').drop(columns=['date'])
led=led.sort_values('run_date')
led=pd.merge_asof(led, px.rename(columns={'date':'run_date','actual':'anchor'}),
                  on='run_date', direction='backward')
v=led.dropna(subset=['actual','anchor']).copy()
v['err_model']=v['pred']-v['actual']; v['err_rw']=v['anchor']-v['actual']
v['ape_model']=v['err_model'].abs()/v['actual']; v['ape_rw']=v['err_rw'].abs()/v['actual']
v['dir_hit']=np.sign(v['pred']-v['anchor'])==np.sign(v['actual']-v['anchor'])
v.to_csv(f'{SD}/validated_sep.csv',index=False)
print(f"  matured & scored: {len(v)}\n")

def dm(e1,e2,h):
    d=np.abs(e1)-np.abs(e2); n=len(d)
    if n<8: return np.nan,np.nan
    s=np.var(d,ddof=0)
    for lag in range(1,int(h)):
        if lag>=n: break
        s+=2*(1-lag/h)*np.cov(d[lag:],d[:-lag],ddof=0)[0,1]
    if s<=0: return np.nan,np.nan
    stat=d.mean()/sqrt(s/n)*sqrt((n+1-2*h+h*(h-1)/n)/n)
    return stat, 2*(1-NormalDist().cdf(abs(stat)))

print("="*92)
print("FULL PRODUCTION HISTORY — model vs naive random walk")
print("="*92)
hdr=f"{'H':>4} {'n':>5} {'MAPE':>8} {'naive':>8} {'skill':>8} {'TheilU':>7} {'dir%':>6} {'bias':>9} {'DM p':>7}"
print(hdr); print('-'*len(hdr))
for h,d in v.groupby('horizon'):
    mae_m=d['err_model'].abs().mean(); mae_r=d['err_rw'].abs().mean()
    rm=np.sqrt((d['err_model']**2).mean()); rr=np.sqrt((d['err_rw']**2).mean())
    _,p=dm(d['err_model'].values,d['err_rw'].values,max(int(h),1))
    print(f"{int(h):>4} {len(d):>5} {d['ape_model'].mean()*100:7.2f}% {d['ape_rw'].mean()*100:7.2f}% "
          f"{1-mae_m/mae_r:+8.3f} {rm/rr:7.3f} {d['dir_hit'].mean()*100:5.1f}% {d['err_model'].mean():+9.1f} {p:7.3f}")

print(f"\nPOOLED: n={len(v)}  skill={1-v['err_model'].abs().mean()/v['err_rw'].abs().mean():+.3f}  "
      f"dir={v['dir_hit'].mean()*100:.1f}%  return-corr="
      f"{np.log(v['pred']/v['anchor']).corr(np.log(v['actual']/v['anchor'])):+.3f}")

# ── has anything changed in the last 6 weeks? ──
CUT=pd.Timestamp('2026-08-05')
print("\n\n"+"="*92)
print("BEFORE vs AFTER the 2026-08-05 assessment (by forecast run date)")
print("="*92)
print(f"{'H':>4} {'n<=Aug5':>8} {'skill':>8} {'dir%':>6}  |{'n>Aug5':>8} {'skill':>8} {'dir%':>6}  {'skill change':>13}")
print('-'*80)
for h,d in v.groupby('horizon'):
    old=d[d['run_date']<=CUT]; new=d[d['run_date']>CUT]
    def sk(x):
        if len(x)<5: return np.nan
        return 1-x['err_model'].abs().mean()/x['err_rw'].abs().mean()
    so,sn=sk(old),sk(new)
    ch=f"{sn-so:+13.3f}" if not (np.isnan(so) or np.isnan(sn)) else f"{'--':>13}"
    print(f"{int(h):>4} {len(old):>8} {so:>8.3f} {old['dir_hit'].mean()*100:5.1f}%  |{len(new):>8} "
          f"{sn:>8.3f} {new['dir_hit'].mean()*100:5.1f}%  {ch}")
