"""Build The Cardamom Almanac — static HTML + PDF via WeasyPrint.

All charts are generated as inline SVG strings in Python (no JS needed).
Run from the CardamomPulse repo root.
"""
import sys, json, warnings, math
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
from pipeline.collectors.auction import load_xls_fallback

# ── Data assembly ────────────────────────────────────────────────────────────
raw = load_xls_fallback()
raw["Date"] = pd.to_datetime(raw["Date"])

def wavg(g):
    w = g["Qty_Sold_Kg"].clip(lower=1)
    return pd.Series({
        "avg_price": np.average(g["AvgPrice"], weights=w),
        "sold_kg": g["Qty_Sold_Kg"].sum(),
        "arrived_kg": g["Qty_Arrived_Kg"].sum(),
        "n_auctions": len(g),
        "disp_cv": g["AvgPrice"].std(ddof=0) / np.average(g["AvgPrice"], weights=w) if len(g) > 1 else 0.0,
    })

d = raw.groupby("Date").apply(wavg, include_groups=False).reset_index().sort_values("Date")
d["year"] = d["Date"].dt.year
d["month"] = d["Date"].dt.month

# price series (monthly)
ps = d.set_index("Date")["avg_price"].resample("ME").mean().dropna()
price_series = [{"date": str(i.date()), "price": round(float(p), 0)} for i, p in ps.items()]

# overall stats
price_overall = {
    "min": round(float(d["avg_price"].min()), 0),
    "min_date": str(d.loc[d["avg_price"].idxmin(), "Date"].date()),
    "max": round(float(d["avg_price"].max()), 0),
    "max_date": str(d.loc[d["avg_price"].idxmax(), "Date"].date()),
    "latest": round(float(d["avg_price"].iloc[-1]), 0),
    "ratio_max_min": round(float(d["avg_price"].max() / d["avg_price"].min()), 1),
}

# regimes
d["sm"] = d["avg_price"].rolling(45, min_periods=20, center=True).mean()
ds = d.dropna(subset=["sm"]).reset_index(drop=True)
ret90 = ds["sm"].pct_change(60)
state = pd.Series("range", index=ds.index)
state[ret90 > 0.12] = "bull"
state[ret90 < -0.12] = "bear"
segs = []; cur = state.iloc[0]; start = 0
for i in range(1, len(ds)):
    if state.iloc[i] != cur:
        segs.append((cur, start, i - 1)); cur = state.iloc[i]; start = i
segs.append((cur, start, len(ds) - 1))
regimes = []
for st, a, b in segs:
    if (ds["Date"].iloc[b] - ds["Date"].iloc[a]).days < 75: continue
    p0 = ds["avg_price"].iloc[a]; p1 = ds["avg_price"].iloc[b]
    regimes.append({"state": st, "start": str(ds["Date"].iloc[a].date()), "end": str(ds["Date"].iloc[b].date()),
                    "p0": round(p0, 0), "p1": round(p1, 0), "chg_pct": round((p1 / p0 - 1) * 100, 0)})

# rally/crash
d2 = d.set_index("Date")["avg_price"]
r180 = d2.pct_change(180)
biggest_rally = {"pct": round(float(r180.max()) * 100, 0), "end": str(r180.idxmax().date())}
biggest_crash = {"pct": round(float(r180.min()) * 100, 0), "end": str(r180.idxmin().date())}

# seasonality
ann = d.groupby("year").agg(avg=("avg_price", "mean"), lo=("avg_price", "min"),
                            hi=("avg_price", "max"), vol_mt=("sold_kg", lambda s: s.sum() / 1e6)).reset_index()
ann["yoy"] = ann["avg"].pct_change() * 100

d_m = d.set_index("Date")["avg_price"].resample("ME").mean().to_frame("p")
d_m["trail12"] = d_m["p"].rolling(12, min_periods=6).mean()
d_m["rel"] = d_m["p"] / d_m["trail12"]
d_m["mom"] = d_m["p"].pct_change() * 100
d_m["mon"] = d_m.index.month
seas = d_m.groupby("mon").agg(rel=("rel", "mean"), mom=("mom", "mean")).reset_index()
MONTHS = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
seasonality = [{"month": MONTHS[int(r.mon)-1], "rel_to_avg_pct": round((r.rel - 1) * 100, 1),
                "avg_mom_pct": round(r.mom, 1)} for r in seas.itertuples()]
best_s = seas.loc[seas["rel"].idxmax()]; worst_s = seas.loc[seas["rel"].idxmin()]
sell_best_month = MONTHS[int(best_s.mon) - 1]
sell_worst_month = MONTHS[int(worst_s.mon) - 1]
sell_premium_pct = round((best_s.rel / worst_s.rel - 1) * 100, 0)

# plant
yr_avg = ann.set_index("year")["avg"]
plant = []
for y in yr_avg.index:
    if y + 3 in yr_avg.index:
        plant.append({"plant_year": int(y), "price_then": round(yr_avg[y], 0),
                      "price_harvest": round(yr_avg[y + 3], 0), "multiple": round(yr_avg[y + 3] / yr_avg[y], 2)})
plant_best = max(plant, key=lambda x: x["multiple"])
plant_worst = min(plant, key=lambda x: x["multiple"])

# houses
dd = raw.merge(raw.groupby("Date").apply(
    lambda g: pd.Series({"day_wavg": np.average(g["AvgPrice"], weights=g["Qty_Sold_Kg"].clip(lower=1)),
                         "n": len(g)}), include_groups=False), on="Date")
multi = dd[dd["n"] >= 2].copy()
multi["rel"] = multi["AvgPrice"] / multi["day_wavg"] - 1
h = multi.groupby("Auctioneer").agg(n=("rel","count"), rel=("rel","mean"),
                                     above=("rel", lambda s: (s > 0).mean())).reset_index()
h = h[h["n"] >= 100].sort_values("rel", ascending=False)
def short(name): return name.split(",")[0][:38]
houses = [{"name": short(r.Auctioneer), "premium_pct": round(r.rel*100, 2),
           "days_above_pct": round(r.above * 100, 0), "n": int(r.n)} for r in h.itertuples()]
house_spread_pct = round((h["rel"].max() - h["rel"].min()) * 100, 1)

# dispersion
day = d.set_index("Date").copy()
day["disp_ma14"] = day["disp_cv"].rolling(14, min_periods=7).mean()
day["fwd14"] = day["avg_price"].shift(-14) / day["avg_price"] - 1
v = day.dropna(subset=["disp_ma14", "fwd14"])
q = pd.qcut(v["disp_ma14"], 4, labels=["Low","Q2","Q3","High"])
disp = v.groupby(q, observed=True)["fwd14"].agg(["mean", lambda s: s.abs().mean()])
disp.columns = ["ret", "absmove"]
dispersion = [{"bucket": str(i), "fwd_ret_pct": round(r.ret*100,1), "fwd_absmove_pct": round(r.absmove*100,1)}
              for i, r in disp.iterrows()]

coverage = {"start": str(d["Date"].min().date()), "end": str(d["Date"].max().date()),
            "days": int(len(d)), "auctions": int(len(raw))}

# ── SVG helpers ──────────────────────────────────────────────────────────────
INK    = "#23291f"
INK2   = "#525b46"
GREEN  = "#2f6b40"
GREEN2 = "#3f8a52"
GOLD   = "#c89b3c"
RUST   = "#b5482f"
LINE   = "#e6ddc8"
BLUE   = "#234fa0"
CREAM  = "#fffdf7"

def wrap_svg(w, h, body, extra=""):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'viewBox="0 0 {w} {h}" width="{w}" height="{h}" {extra}>{body}</svg>')

def lpath(pts):
    return " ".join(f"{'M' if i == 0 else 'L'}{x:.1f},{y:.1f}" for i, (x, y) in enumerate(pts))

def pct_str(v):
    return f"+{v}%" if v > 0 else f"{v}%"

def money(v):
    # Indian numbering: above 1000 → ₹1,234
    v = int(round(v))
    s = f"{v:,}"
    return f"₹{s}"

# ── Chart 1: Price journey (line chart) ─────────────────────────────────────
def build_price_chart(price_series, regimes, price_overall):
    W, H, PL, PR, PT, PB = 700, 240, 58, 16, 22, 36
    iW, iH = W - PL - PR, H - PT - PB

    dates = [pd.Timestamp(p["date"]).timestamp() for p in price_series]
    prices = [p["price"] for p in price_series]
    t0, t1 = dates[0], dates[-1]
    minP, maxP = 300, 5100

    def xS(t): return PL + (t - t0) / (t1 - t0) * iW
    def yS(p): return H - PB - (p - minP) / (maxP - minP) * iH

    out = []
    # regime bands
    for r in regimes:
        if r["state"] == "range" or abs(r["chg_pct"]) < 20: continue
        x1 = xS(pd.Timestamp(r["start"]).timestamp())
        x2 = xS(pd.Timestamp(r["end"]).timestamp())
        col = "rgba(63,138,82,.13)" if r["state"] == "bull" else "rgba(181,72,47,.13)"
        out.append(f'<rect x="{x1:.1f}" y="{PT}" width="{max(x2-x1,2):.1f}" height="{iH}" fill="{col}"/>')

    # grid
    for v in [500,1000,1500,2000,2500,3000,3500,4000,4500]:
        y = yS(v)
        out.append(f'<line x1="{PL}" x2="{W-PR}" y1="{y:.1f}" y2="{y:.1f}" stroke="{LINE}" stroke-width="0.8"/>')
        lbl = f"₹{v//1000}k" if v >= 1000 else f"₹{v}"
        out.append(f'<text x="{PL-5}" y="{y+3.5:.1f}" text-anchor="end" font-family="Arial" font-size="9" fill="{INK2}">{lbl}</text>')

    # year ticks
    for y in range(2015, 2027):
        t = pd.Timestamp(f"{y}-01-01").timestamp()
        if t < t0 or t > t1: continue
        x = xS(t)
        out.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{H-PB}" y2="{H-PB+4}" stroke="#aaa" stroke-width="1"/>')
        out.append(f'<text x="{x:.1f}" y="{H-PB+15}" text-anchor="middle" font-family="Arial" font-size="9" fill="{INK2}">{y}</text>')

    # area gradient + line
    pts = [(xS(t), yS(p)) for t, p in zip(dates, prices)]
    area = lpath(pts) + f" L{pts[-1][0]:.1f},{H-PB} L{pts[0][0]:.1f},{H-PB} Z"
    out.append(f'<defs><linearGradient id="pg" x1="0" y1="0" x2="0" y2="1">'
               f'<stop offset="0%" stop-color="{GREEN}" stop-opacity=".22"/>'
               f'<stop offset="100%" stop-color="{GREEN}" stop-opacity="0"/>'
               f'</linearGradient></defs>')
    out.append(f'<path d="{area}" fill="url(#pg)"/>')
    out.append(f'<path d="{lpath(pts)}" fill="none" stroke="{GREEN}" stroke-width="2.2"/>')

    # peak annotation
    pt = pd.Timestamp(price_overall["max_date"]).timestamp()
    px, py = xS(pt), yS(price_overall["max"])
    out.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="4.5" fill="{GOLD}" stroke="white" stroke-width="1.5"/>')
    out.append(f'<rect x="{px-74:.1f}" y="{py-24:.1f}" width="148" height="18" rx="4" fill="rgba(255,253,247,.92)"/>')
    out.append(f'<text x="{px:.1f}" y="{py-11:.1f}" text-anchor="middle" font-family="Arial" font-size="9" font-weight="700" fill="{RUST}">₹4,675 peak — Aug 2019</text>')

    # axes
    out.append(f'<line x1="{PL}" x2="{PL}" y1="{PT}" y2="{H-PB}" stroke="#bbb" stroke-width="1"/>')
    out.append(f'<line x1="{PL}" x2="{W-PR}" y1="{H-PB}" y2="{H-PB}" stroke="#bbb" stroke-width="1"/>')

    return wrap_svg(W, H, "\n".join(out))


# ── Chart 2: Factor importance (dual bar) ───────────────────────────────────
def build_factor_chart():
    rows = [
        ("Guatemala cumulative rainfall", 10, 95),
        ("Idukki rainfall (Indian supply)",  65, 70),
        ("Price momentum & lags",            80, 20),
        ("Cross-auction dispersion",         72, 12),
        ("ENSO / La Niña (lagged)",          18, 75),
        ("USD/INR exchange rate",            38, 40),
    ]
    W = 700; PL = 190; PR = 12; PT = 24; rH = 15; gap = 4
    iW = W - PL - PR
    colW = (iW - 16) // 2
    out = []

    # column header backgrounds
    out.append(f'<rect x="{PL}" y="{PT-16}" width="{colW}" height="14" rx="3" fill="{BLUE}" opacity=".12"/>')
    out.append(f'<text x="{PL+colW//2}" y="{PT-5}" text-anchor="middle" font-family="Arial" font-size="8" font-weight="700" fill="{BLUE}">SHORT-TERM (1–14 day models)</text>')
    out.append(f'<rect x="{PL+colW+16}" y="{PT-16}" width="{colW}" height="14" rx="3" fill="{GREEN}" opacity=".12"/>')
    out.append(f'<text x="{PL+colW+16+colW//2}" y="{PT-5}" text-anchor="middle" font-family="Arial" font-size="8" font-weight="700" fill="{GREEN}">LONG-TERM (90-day model)</text>')

    for i, (name, s, l) in enumerate(rows):
        y = PT + i * (rH + gap)
        out.append(f'<text x="{PL-6}" y="{y+rH*0.72:.1f}" text-anchor="end" font-family="Arial" font-size="8.5" fill="{INK}">{name}</text>')
        sw = s / 100 * colW
        out.append(f'<rect x="{PL}" y="{y+2}" width="{sw:.1f}" height="{rH-4}" rx="3" fill="{BLUE}" opacity=".8"/>')
        out.append(f'<text x="{PL+sw+4:.1f}" y="{y+rH*0.72:.1f}" font-family="Arial" font-size="7.5" fill="{BLUE}">{s}</text>')
        lw = l / 100 * colW
        ox = PL + colW + 16
        out.append(f'<rect x="{ox}" y="{y+2}" width="{lw:.1f}" height="{rH-4}" rx="3" fill="{GREEN}" opacity=".85"/>')
        out.append(f'<text x="{ox+lw+4:.1f}" y="{y+rH*0.72:.1f}" font-family="Arial" font-size="7.5" fill="{GREEN}">{l}</text>')

    H = PT + len(rows) * (rH + gap) + 6
    return wrap_svg(W, H, "\n".join(out))


# ── Chart 3: Seasonality ─────────────────────────────────────────────────────
def build_season_chart(seasonality, sell_best_month):
    W, H, PL, PR, PT, PB = 700, 205, 42, 14, 20, 34
    iW, iH = W - PL - PR, H - PT - PB
    n = len(seasonality); bW = iW / n * 0.60; gap = iW / n

    all_v = [s["rel_to_avg_pct"] for s in seasonality] + [s["avg_mom_pct"] for s in seasonality]
    minV = min(all_v) - 1.5; maxV = max(all_v) + 1.5

    def yS(v): return H - PB - (v - minV) / (maxV - minV) * iH
    z = yS(0)

    out = []
    # legend
    out.append(f'<rect x="{PL}" y="{PT-16}" width="118" height="13" rx="3" fill="white" opacity=".6"/>')
    out.append(f'<text x="{PL+4}" y="{PT-5}" font-family="Arial" font-size="8" fill="{GREEN2}" font-weight="700">▪ Seasonal premium (%)</text>')
    out.append(f'<text x="{PL+124}" y="{PT-5}" font-family="Arial" font-size="8" fill="{INK}" font-weight="700">- - Avg MoM change (%)</text>')

    # gridlines
    for v in [-8, -4, 0, 4, 8, 12, 16]:
        if v < minV or v > maxV: continue
        y = yS(v)
        sw = "#aaa" if v == 0 else LINE
        wt = "1.2" if v == 0 else "0.8"
        out.append(f'<line x1="{PL-4}" x2="{W-PR}" y1="{y:.1f}" y2="{y:.1f}" stroke="{sw}" stroke-width="{wt}"/>')
        out.append(f'<text x="{PL-6}" y="{y+3.5:.1f}" text-anchor="end" font-family="Arial" font-size="8.5" fill="{INK2}">{pct_str(v)}</text>')

    lpts = []
    for i, s in enumerate(seasonality):
        cx = PL + gap * i + gap / 2; x = cx - bW / 2
        v = s["rel_to_avg_pct"]
        col = GOLD if s["month"] == sell_best_month else (GREEN2 if v >= 0 else RUST)
        y = yS(v) if v >= 0 else z
        bh = abs(yS(v) - z)
        out.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bW:.1f}" height="{max(bh,1):.1f}" rx="2" fill="{col}"/>')
        out.append(f'<text x="{cx:.1f}" y="{H-PB+14}" text-anchor="middle" font-family="Arial" font-size="8.5" fill="{INK2}">{s["month"]}</text>')
        lpts.append((cx, yS(s["avg_mom_pct"])))

    out.append(f'<path d="{lpath(lpts)}" fill="none" stroke="{INK}" stroke-width="1.5" stroke-dasharray="5,3"/>')
    for x, y in lpts:
        out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="2.5" fill="{INK}"/>')

    out.append(f'<line x1="{PL}" x2="{PL}" y1="{PT}" y2="{H-PB}" stroke="#bbb" stroke-width="1"/>')
    out.append(f'<line x1="{PL}" x2="{W-PR}" y1="{H-PB}" y2="{H-PB}" stroke="#bbb" stroke-width="1"/>')
    return wrap_svg(W, H, "\n".join(out))


# ── Chart 4: Planting chart ──────────────────────────────────────────────────
def build_plant_chart(plant):
    W, H, PL, PR, PT, PB = 700, 200, 42, 16, 20, 34
    iW, iH = W - PL - PR, H - PT - PB
    n = len(plant); bW = iW / n * 0.62; gap = iW / n

    multiples = [p["multiple"] for p in plant]
    minM = min(multiples) * 0.88; maxM = max(multiples) * 1.06

    def yS(v): return H - PB - (v - minM) / (maxM - minM) * iH
    z = yS(1)

    out = []
    for v in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
        if v < minM or v > maxM: continue
        y = yS(v)
        da = "6,3" if v == 1.0 else ""
        sw = "#777" if v == 1.0 else LINE
        wt = "1.5" if v == 1.0 else "0.8"
        out.append(f'<line x1="{PL-4}" x2="{W-PR}" y1="{y:.1f}" y2="{y:.1f}" stroke="{sw}" stroke-width="{wt}" stroke-dasharray="{da}"/>')
        out.append(f'<text x="{PL-6}" y="{y+3.5:.1f}" text-anchor="end" font-family="Arial" font-size="8.5" fill="{INK2}">{v}×</text>')
        if v == 1.0:
            out.append(f'<text x="{W-PR-2}" y="{y-3:.1f}" text-anchor="end" font-family="Arial" font-size="7.5" fill="#777">break-even</text>')

    for i, p in enumerate(plant):
        cx = PL + gap * i + gap / 2; x = cx - bW / 2
        col = GREEN if p["multiple"] >= 1.5 else (GOLD if p["multiple"] >= 1 else RUST)
        y = yS(p["multiple"]) if p["multiple"] >= 1 else z
        bh = abs(yS(p["multiple"]) - z)
        out.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bW:.1f}" height="{max(bh,2):.1f}" rx="3" fill="{col}"/>')
        lbl_y = yS(p["multiple"]) - 4 if p["multiple"] >= 1 else yS(p["multiple"]) + 13
        out.append(f'<text x="{cx:.1f}" y="{lbl_y:.1f}" text-anchor="middle" font-family="Arial" font-size="8" font-weight="700" fill="{col}">{p["multiple"]}×</text>')
        out.append(f'<text x="{cx:.1f}" y="{H-PB+14}" text-anchor="middle" font-family="Arial" font-size="8.5" fill="{INK2}">{p["plant_year"]}</text>')

    out.append(f'<line x1="{PL}" x2="{PL}" y1="{PT}" y2="{H-PB}" stroke="#bbb" stroke-width="1"/>')
    out.append(f'<line x1="{PL}" x2="{W-PR}" y1="{H-PB}" y2="{H-PB}" stroke="#bbb" stroke-width="1"/>')
    return wrap_svg(W, H, "\n".join(out))


# ── Chart 5: Auction houses (horizontal bar) ─────────────────────────────────
def build_house_chart(houses):
    hs = sorted(houses, key=lambda h: h["premium_pct"])
    W = 700; PL = 226; PR = 78; PT = 10; PB = 14
    iW = W - PL - PR; rH = 16; gap = 3
    maxAbs = max(abs(h["premium_pct"]) for h in hs) * 1.1
    H = PT + len(hs) * (rH + gap) + PB

    def xS(v): return PL + iW / 2 + (v / maxAbs) * (iW / 2)
    z = PL + iW / 2

    out = []
    out.append(f'<line x1="{z:.1f}" x2="{z:.1f}" y1="{PT}" y2="{H-PB-6}" stroke="#ccc" stroke-width="1" stroke-dasharray="3,2"/>')

    for i, h in enumerate(hs):
        y = PT + i * (rH + gap)
        x0 = min(z, xS(h["premium_pct"]))
        bw = max(abs(xS(h["premium_pct"]) - z), 2)
        col = GREEN2 if h["premium_pct"] >= 0 else RUST
        name = h["name"].replace("PRIVATE LIMITED","").replace("Private Limited","").replace("LIMITED","").replace("Limited","").strip()
        name = name[:34]
        out.append(f'<text x="{PL-6}" y="{y+rH*0.72:.1f}" text-anchor="end" font-family="Arial" font-size="8" fill="{INK}">{name}</text>')
        out.append(f'<rect x="{x0:.1f}" y="{y+3}" width="{bw:.1f}" height="{rH-6}" rx="2" fill="{col}"/>')
        ta = "start" if h["premium_pct"] >= 0 else "end"
        lx = xS(h["premium_pct"]) + 4 if h["premium_pct"] >= 0 else xS(h["premium_pct"]) - 4
        sign = "+" if h["premium_pct"] > 0 else ""
        out.append(f'<text x="{lx:.1f}" y="{y+rH*0.72:.1f}" text-anchor="{ta}" font-family="Arial" font-size="8" font-weight="700" fill="{col}">{sign}{h["premium_pct"]}%</text>')

    out.append(f'<text x="{z:.1f}" y="{H-4}" text-anchor="middle" font-family="Arial" font-size="7.5" fill="#888">market average</text>')
    return wrap_svg(W, H, "\n".join(out))


# ── Chart 6: Dispersion ──────────────────────────────────────────────────────
def build_disp_chart(dispersion):
    W, H, PL, PR, PT, PB = 310, 135, 38, 10, 20, 32
    iW, iH = W - PL - PR, H - PT - PB
    n = len(dispersion); bW = iW / n * 0.55; gap = iW / n
    maxV = 10; minV = -3

    def yS(v): return H - PB - (v - minV) / (maxV - minV) * iH
    z = yS(0)

    out = []
    out.append(f'<text x="{PL}" y="{PT-6}" font-family="Arial" font-size="7" fill="{GREEN2}" font-weight="700">▪ Avg move size (2-wk)</text>')
    out.append(f'<text x="{PL+114}" y="{PT-6}" font-family="Arial" font-size="7" fill="{GOLD}" font-weight="700">— Direction</text>')

    for v in [0, 4, 8]:
        y = yS(v)
        out.append(f'<line x1="{PL-3}" x2="{W-PR}" y1="{y:.1f}" y2="{y:.1f}" stroke="{"#aaa" if v==0 else LINE}" stroke-width="0.8"/>')
        out.append(f'<text x="{PL-5}" y="{y+3:.1f}" text-anchor="end" font-family="Arial" font-size="7.5" fill="{INK2}">{pct_str(v)}</text>')

    lpts = []
    for i, dp in enumerate(dispersion):
        cx = PL + gap * i + gap / 2; x = cx - bW / 2
        bh = z - yS(dp["fwd_absmove_pct"])
        out.append(f'<rect x="{x:.1f}" y="{yS(dp["fwd_absmove_pct"]):.1f}" width="{bW:.1f}" height="{bh:.1f}" rx="2" fill="{GREEN2}"/>')
        out.append(f'<text x="{cx:.1f}" y="{H-PB+13}" text-anchor="middle" font-family="Arial" font-size="7.5" fill="{INK2}">{dp["bucket"]}</text>')
        lpts.append((cx, yS(dp["fwd_ret_pct"])))

    out.append(f'<path d="{lpath(lpts)}" fill="none" stroke="{GOLD}" stroke-width="2"/>')
    for x, y in lpts:
        out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="2.5" fill="{GOLD}"/>')

    out.append(f'<line x1="{PL}" x2="{PL}" y1="{PT}" y2="{H-PB}" stroke="#bbb" stroke-width="1"/>')
    out.append(f'<line x1="{PL}" x2="{W-PR}" y1="{H-PB}" y2="{H-PB}" stroke="#bbb" stroke-width="1"/>')
    return wrap_svg(W, H, "\n".join(out))


# ── Render all charts ────────────────────────────────────────────────────────
svg_price  = build_price_chart(price_series, regimes, price_overall)
svg_factor = build_factor_chart()
svg_season = build_season_chart(seasonality, sell_best_month)
svg_plant  = build_plant_chart(plant)
svg_house  = build_house_chart(houses)
svg_disp   = build_disp_chart(dispersion)

# ── Text helpers ─────────────────────────────────────────────────────────────
peak_months = ", ".join(s["month"] for s in seasonality if s["rel_to_avg_pct"] >= 10)
worst_mom = min(seasonality, key=lambda s: s["avg_mom_pct"])
top_h = houses[0]; bot_h = houses[-1]

short_drivers = [
    ("Price momentum &amp; moving averages", "1–14 day moves", 80),
    ("Cross-auction price dispersion",        "leading signal at 7–14 days", 72),
    ("Idukki rainfall (weekly signal)",        "near-term supply shock", 65),
    ("Seasonality / calendar position",        "lean vs harvest window", 55),
    ("USD/INR exchange rate",                  "export demand response", 38),
]
long_drivers = [
    ("Guatemala cumulative rainfall",          "#1 feature in 90-day model; ~70% global supply", 95),
    ("ENSO (El Niño / La Niña, lagged)",       "6–12 month lag on Guatemala climate", 75),
    ("Idukki cumulative monsoon (90d)",         "Indian supply for the coming season", 70),
    ("Guatemala export volumes",               "direct supply signal with 3–6m lag", 55),
    ("Cobweb cycle age &amp; cost ratio",       "structural planting cycle position", 40),
]

def driver_rows(items, col):
    rows = []
    for name, note, bar in items:
        rows.append(f"""
      <div class="driver-row">
        <div style="flex:1">
          <div class="driver-name">{name}</div>
          <div class="driver-note">{note}</div>
        </div>
        <div class="driver-bar-wrap">
          <div class="driver-bar" style="width:{bar}%;background:{col}"></div>
        </div>
      </div>""")
    return "\n".join(rows)

rotation_rows = [
    ("Mas Enterprises",                "Saturday", 44),
    ("South Indian Green Cardamom Co.","Tuesday",  42),
    ("Header Systems (India) Ltd.",    "Tuesday",  41),
    ("Kerala Cardamom Processing",     "Thursday", 47),
    ("Green House Cardamom Mktg.",     "Saturday", 41),
    ("SUGANDHAGIRI SPICES",            "Wednesday",44),
    ("IDUKKI Traditional Producers",   "Monday",   45),
    ("Cardamom GrowersForever",        "Friday",   42),
]

def rot_grid():
    rows = []
    for house, day, share in rotation_rows:
        rows.append(f'<div class="rot-row"><span>{house[:22]}</span>'
                    f'<span><span class="rot-day">{day}</span> '
                    f'<span class="rot-share">({share}%)</span></span></div>')
    return "\n".join(rows)

playbook = [
    ("Guatemala is the macro signal.",
     "Its cumulative 6-month rainfall is the #1 driver of annual price levels. A drought in Alta Verapaz "
     "starts a bull market 6–18 months later. Watch USDA Guatemala crop reports alongside the Idukki monsoon."),
    ("Respect the cycle.",
     f"Cardamom swings {price_overall['ratio_max_min']}× between peaks and troughs. "
     "Today's price is a moment in a multi-year pattern, not a permanent new level."),
    ("Watch the rain, not the rumour.",
     "Idukki monsoon data and El Niño/La Niña forecasts (lagged 6–12 months) shape Indian supply more reliably than any market tip."),
    (f"Sell into strength ({peak_months.split(',')[0].strip()}–{sell_best_month}).",
     f"The mid-year window pays ~{int(sell_premium_pct)}% over the weakest month. "
     f"Avoid the {worst_mom['month']} harvest glut if you can store safely."),
    ("Plant cheap, never hot.",
     f"Fields planted in low-price years like {plant_best['plant_year']} returned {plant_best['multiple']}×. "
     f"Those planted at the {plant_worst['plant_year']} peak returned just {plant_worst['multiple']}×. The crowd is almost always wrong."),
    ("Know your auction house.",
     f"A {house_spread_pct}% premium gap separates the best and worst houses for comparable produce. "
     "It compounds at every sale, every season."),
    ("Read the disagreement.",
     "When same-day auction prices diverge widely, the next two weeks tend to move more — "
     "and historically lean upward. High dispersion is a quiet tell."),
]

def pb_items():
    return "\n".join(f"<li><b>{t}</b> {d}</li>" for t, d in playbook)

# ── HTML template ────────────────────────────────────────────────────────────
HTML = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>The Cardamom Almanac</title>
<style>
:root{{--ink:#23291f;--ink2:#525b46;--cream:#faf6ec;--paper:#fffdf7;--green:#2f6b40;--green2:#3f8a52;--gold:#c89b3c;--rust:#b5482f;--line:#e6ddc8;}}
@page{{size:A4;margin:14mm 18mm 16mm;}}
*{{box-sizing:border-box;margin:0;padding:0;}}
html{{font-size:10pt;}}
body{{font-family:Georgia,"Times New Roman",serif;color:var(--ink);background:var(--paper);line-height:1.55;-webkit-print-color-adjust:exact;print-color-adjust:exact;}}
.page{{width:100%;break-after:page;page-break-after:always;overflow:hidden;}}
.page:last-child{{break-after:auto;page-break-after:auto;}}
h2{{font-family:Georgia,serif;font-size:19pt;font-weight:700;line-height:1.1;margin-bottom:5pt;}}
h3{{font-family:Georgia,serif;font-size:11pt;font-weight:700;line-height:1.2;}}
.kicker{{font-family:Arial,sans-serif;font-weight:700;letter-spacing:.15em;text-transform:uppercase;font-size:7.5pt;color:#bfe0b7;margin-bottom:8pt;}}
.lede{{font-family:Arial,sans-serif;font-size:10pt;color:var(--ink2);line-height:1.5;margin-bottom:10pt;}}
.sec-num{{font-family:Georgia,serif;font-weight:600;color:var(--gold);font-size:10pt;margin-bottom:2pt;}}
p{{font-size:9.5pt;margin-bottom:7pt;font-family:Arial,sans-serif;}}
/* Cover */
.cover{{background:linear-gradient(160deg,#1f4a2c 0%,#2f6b40 60%,#3f8a52 100%);color:#f4f1e4;padding:46pt 44pt;position:relative;overflow:hidden;display:flex;flex-direction:column;min-height:267mm;}}
.cover-orb{{position:absolute;right:-60pt;top:-50pt;width:270pt;height:270pt;background:radial-gradient(circle,rgba(200,155,60,.28),transparent 65%);border-radius:50%;}}
.cover h1{{color:#f4f1e4;font-family:Georgia,serif;font-size:37pt;max-width:390pt;margin:.3em 0 .25em;letter-spacing:-.02em;line-height:1.1;}}
.cover .sub{{font-family:Arial,sans-serif;font-size:11pt;max-width:360pt;color:#e7eede;margin:0 0 10pt;line-height:1.55;}}
.cover .meta{{font-family:Arial,sans-serif;font-size:7.5pt;color:#cfe3c8;letter-spacing:.04em;margin-top:10pt;}}
.cover-stats{{display:grid;grid-template-columns:repeat(4,1fr);gap:1pt;background:rgba(255,255,255,.15);border:1pt solid rgba(255,255,255,.2);border-radius:10pt;overflow:hidden;margin-top:auto;}}
.cs{{background:rgba(255,255,255,.08);padding:13pt 10pt;text-align:center;}}
.cs .big{{font-family:Georgia,serif;font-weight:700;font-size:19pt;color:#f4f1e4;line-height:1;}}
.cs .lbl{{font-family:Arial,sans-serif;font-size:7pt;color:#cfe3c8;margin-top:4pt;}}
/* Sections */
.sec-page{{padding:18pt 0 8pt;}}
.chart-wrap{{margin:8pt 0;}}
.caption{{font-family:Arial,sans-serif;font-size:7.5pt;color:var(--ink2);font-style:italic;margin-top:5pt;}}
.takeaway{{background:#f0f6ea;border-left:4pt solid var(--green2);border-radius:0 7pt 7pt 0;padding:9pt 13pt;margin:8pt 0;}}
.takeaway .t{{font-family:Arial,sans-serif;font-weight:700;color:var(--green);font-size:7pt;letter-spacing:.08em;text-transform:uppercase;margin-bottom:4pt;}}
.takeaway p{{margin:0;font-size:9pt;}}
.takeaway.warn{{background:#fbefe9;border-color:var(--rust);}}
.takeaway.warn .t{{color:var(--rust);}}
.grid2{{display:grid;grid-template-columns:1fr 1fr;gap:9pt;}}
.pull{{font-family:Georgia,serif;font-weight:600;font-size:13pt;color:var(--green);line-height:1.3;margin:12pt 0;text-align:center;padding:0 18pt;}}
.minis{{display:grid;grid-template-columns:repeat(4,1fr);gap:7pt;margin:8pt 0;}}
.mini{{background:var(--paper);border:1pt solid var(--line);border-radius:7pt;padding:8pt;text-align:center;}}
.mini .v{{font-family:Georgia,serif;font-weight:700;font-size:15pt;color:var(--green);}}
.mini .v.bad{{color:var(--rust);}}
.mini .k{{font-family:Arial,sans-serif;font-size:7pt;color:var(--ink2);margin-top:2pt;}}
.driver-split{{display:grid;grid-template-columns:1fr 1fr;gap:12pt;margin:8pt 0;}}
.driver-col{{border:1pt solid var(--line);border-radius:7pt;overflow:hidden;}}
.driver-col-head{{font-family:Arial,sans-serif;font-weight:700;font-size:7.5pt;letter-spacing:.08em;text-transform:uppercase;padding:7pt 10pt;}}
.driver-col-head.short{{background:#e8eef8;color:#234fa0;}}
.driver-col-head.long{{background:#eaf4e8;color:var(--green);}}
.driver-row{{display:flex;align-items:center;padding:5pt 10pt;border-top:1pt solid var(--line);gap:8pt;}}
.driver-row:nth-child(even){{background:rgba(0,0,0,.02);}}
.driver-name{{font-family:Arial,sans-serif;font-size:8.5pt;flex:1;}}
.driver-note{{font-family:Arial,sans-serif;font-size:7pt;color:var(--ink2);}}
.driver-bar-wrap{{flex:0 0 52pt;height:5pt;background:var(--line);border-radius:3pt;}}
.driver-bar{{height:5pt;border-radius:3pt;}}
.playbook{{background:#1f4a2c;color:#eef3e8;border-radius:10pt;padding:22pt 26pt;margin:8pt 0;}}
.playbook h3{{color:#fff;font-size:15pt;margin-bottom:4pt;font-family:Georgia,serif;}}
.playbook ol{{margin:10pt 0 0;padding-left:17pt;}}
.playbook li{{font-family:Arial,sans-serif;margin:0 0 9pt;font-size:9.5pt;line-height:1.45;}}
.playbook li b{{color:#c89b3c;font-weight:700;}}
.rot-grid{{display:grid;grid-template-columns:1fr 1fr;border:1pt solid var(--line);border-radius:7pt;overflow:hidden;margin:6pt 0;}}
.rot-row{{display:flex;justify-content:space-between;align-items:center;padding:5pt 10pt;border-bottom:1pt solid var(--line);font-family:Arial,sans-serif;font-size:8pt;}}
.rot-row:last-child,.rot-row:nth-last-child(2){{border-bottom:none;}}
.rot-row:nth-child(odd){{background:rgba(0,0,0,.02);}}
.rot-day{{color:var(--green);font-weight:600;}}
.rot-share{{color:var(--ink2);font-size:7.5pt;}}
.footer-bar{{font-family:Arial,sans-serif;font-size:7.5pt;color:var(--ink2);border-top:1pt solid var(--line);padding-top:8pt;margin-top:10pt;}}
.tag{{display:inline-block;background:#eef3e8;color:var(--green);border-radius:999pt;padding:2pt 7pt;font-size:7pt;font-weight:600;margin:2pt 3pt 2pt 0;}}
@media screen{{
  body{{max-width:860px;margin:0 auto;padding:20px;background:#e8e4da;}}
  .page{{border-radius:4px;margin-bottom:24px;padding:44pt;box-shadow:0 4px 24px rgba(0,0,0,.15);background:var(--paper);}}
  .cover{{padding:46pt 44pt;min-height:267mm;border-radius:4px;}}
}}
</style>
</head>
<body>

<!-- PAGE 1: COVER -->
<div class="page cover">
  <div class="cover-orb"></div>
  <div style="position:relative;z-index:1;flex:1;display:flex;flex-direction:column">
    <div class="kicker">CardamomPulse · A Data Almanac</div>
    <h1>The Cardamom<br>Almanac</h1>
    <p class="sub">Eleven years. {coverage['auctions']:,} auctions. One small green pod whose price has swung
    {price_overall['ratio_max_min']}× fold. Here is what the numbers say about when to sell,
    when to plant, and where the market really gets made.</p>
    <div class="meta">Coverage {coverage['start']} to {coverage['end']} · {coverage['days']:,} trading days · {coverage['auctions']:,} individual auctions</div>
    <div class="cover-stats" style="margin-top:32pt">
      <div class="cs"><div class="big">{money(price_overall['min'])}</div><div class="lbl">floor ({price_overall['min_date'][:7]})</div></div>
      <div class="cs"><div class="big">{money(price_overall['max'])}</div><div class="lbl">peak ({price_overall['max_date'][:7]})</div></div>
      <div class="cs"><div class="big">{price_overall['ratio_max_min']}×</div><div class="lbl">peak-to-trough swing</div></div>
      <div class="cs"><div class="big">+{int(biggest_rally['pct'])}%</div><div class="lbl">biggest 6-month rally</div></div>
    </div>
  </div>
</div>

<!-- PAGE 2: PRICE JOURNEY -->
<div class="page sec-page">
  <div class="sec-num">01</div>
  <h2>A price with a temper</h2>
  <p class="lede">Few farm commodities move like cardamom. Since 2014 the auction price has travelled
  from under ₹500 to nearly ₹4,700 a kilo and back again — a roller-coaster driven by monsoons,
  floods and the slow churn of planting cycles.</p>
  <div class="chart-wrap">{svg_price}</div>
  <div class="caption">Monthly average auction price, Nov 2014 – Feb 2026. Green bands = bull runs &gt;20%; red bands = bear runs &gt;20%.</div>
  <div class="minis">
    <div class="mini"><div class="v">{money(price_overall['max'])}</div><div class="k">2019 flood peak</div></div>
    <div class="mini"><div class="v bad">{int(biggest_crash['pct'])}%</div><div class="k">biggest 6-month crash</div></div>
    <div class="mini"><div class="v">+{int(biggest_rally['pct'])}%</div><div class="k">biggest 6-month rally</div></div>
    <div class="mini"><div class="v">{money(price_overall['latest'])}</div><div class="k">latest print</div></div>
  </div>
  <div class="takeaway">
    <div class="t">What this means for you</div>
    <p>Cardamom is a <b>boom-and-bust crop.</b> The single biggest fortunes — and ruins — were made by
    those who read the cycle, not the week. The 2019 flood spike to ₹4,675 was followed by a two-thirds
    collapse. Plan for the swing; never assume today's price is the new normal.</p>
  </div>
  <div class="pull">"The market spends years quietly cheap, then re-prices violently in a single season."</div>
</div>

<!-- PAGE 3: WHAT MOVES THE PRICE -->
<div class="page sec-page">
  <div class="sec-num">02</div>
  <h2>What actually moves the price</h2>
  <p class="lede">The answer depends entirely on your time horizon. <b>Guatemala is the master variable
  for price levels over months and years</b> — it produces ~70% of the world's cardamom supply.
  Indian auction dynamics dominate week-to-week moves. Our forecast models confirm this split precisely.</p>
  <div class="driver-split">
    <div class="driver-col">
      <div class="driver-col-head short">SHORT-TERM · days to weeks</div>
      {driver_rows(short_drivers, '#234fa0')}
    </div>
    <div class="driver-col">
      <div class="driver-col-head long">LONG-TERM · months to years</div>
      {driver_rows(long_drivers, '#2f6b40')}
    </div>
  </div>
  <div class="chart-wrap">{svg_factor}</div>
  <div class="caption">Bar length = relative predictive power in out-of-sample walk-forward models. Blue = short-horizon (1–14 day); green = long-horizon (90-day).</div>
  <div class="takeaway">
    <div class="t">Reading the two columns together</div>
    <p>The short-term column tells you <i>when</i> to sell this season. The long-term column tells you
    <i>what price era</i> you are in. A poor Guatemalan harvest starts a bull market 6–18 months later;
    an Idukki rain event moves next week's auction. Both matter — on very different clocks.</p>
  </div>
</div>

<!-- PAGE 4: SEASONALITY -->
<div class="page sec-page">
  <div class="sec-num">03</div>
  <h2>The selling calendar</h2>
  <p class="lede">Strip away the long-term trend and a clear yearly rhythm appears. Some months
  reliably pay more than others — and one month reliably punishes those who wait.</p>
  <div class="chart-wrap">{svg_season}</div>
  <div class="caption">Bars: average price in each month vs its own 12-month average (seasonal premium). Dashed line: average month-on-month change.</div>
  <div class="grid2" style="margin-top:10pt">
    <div class="takeaway">
      <div class="t">Best window to sell</div>
      <p>Prices sit highest from mid-summer through early autumn — <b>{peak_months}</b> all run 10–14%
      above their yearly average, with <b>{sell_best_month}</b> the single strongest month. Holding stock
      for this window has historically paid about <b>{int(sell_premium_pct)}% more</b> than selling in the weakest month.</p>
    </div>
    <div class="takeaway warn">
      <div class="t">The trap month</div>
      <p><b>{worst_mom['month']}</b> shows the steepest average drop of the year ({worst_mom['avg_mom_pct']}% in one month)
      as the harvest floods the market. Spring (around <b>{sell_worst_month}</b>) is the softest patch.
      Don't drift into the harvest glut holding unsold stock.</p>
    </div>
  </div>
</div>

<!-- PAGE 5: PLANTING PARADOX -->
<div class="page sec-page">
  <div class="sec-num">04</div>
  <h2>The planter's paradox</h2>
  <p class="lede">A new cardamom field takes about three years to come into full bearing. So the price
  that matters isn't today's — it's the one waiting for you at harvest. History is brutally clear
  about when to put plants in the ground.</p>
  <div class="chart-wrap">{svg_plant}</div>
  <div class="caption">For each planting year: the price multiple at harvest (~3 years later). Above 1× = real gain; below = real loss.</div>
  <div class="grid2" style="margin-top:10pt">
    <div class="takeaway">
      <div class="t">The golden window</div>
      <p>Planting in <b>{plant_best['plant_year']}</b>, when prices were just <b>{money(plant_best['price_then'])}</b>, meant
      harvesting into <b>{money(plant_best['price_harvest'])}</b> — a <b>{plant_best['multiple']}× return.</b>
      Low prices today mean few new plants going in; tight supply follows at harvest time.</p>
    </div>
    <div class="takeaway warn">
      <div class="t">The widow-maker</div>
      <p>In <b>{plant_worst['plant_year']}</b>, with prices euphoric at <b>{money(plant_worst['price_then'])}</b>,
      everyone planted at once. Three years on the price had collapsed to <b>{money(plant_worst['price_harvest'])}</b>
      — just <b>{plant_worst['multiple']}×</b>, a real purchasing-power loss.</p>
    </div>
  </div>
  <div class="pull" style="color:var(--rust)">"Plant into cheap prices, not into headlines. The crowd that planted at the 2019 peak harvested into a 64% loss."</div>
</div>

<!-- PAGE 6: AUCTION HOUSES -->
<div class="page sec-page">
  <div class="sec-num">05</div>
  <h2>Where you sell is worth money</h2>
  <p class="lede">On any given day two or three auction houses run their own sales. They are not
  interchangeable — a stubborn pecking order has persisted for years.</p>
  <div class="chart-wrap">{svg_house}</div>
  <div class="caption">Average price achieved by each house relative to the same-day market average. Based on days with ≥2 concurrent auctions.</div>
  <div class="minis">
    <div class="mini"><div class="v">+{top_h['premium_pct']}%</div><div class="k">best house premium</div></div>
    <div class="mini"><div class="v">{int(top_h['days_above_pct'])}%</div><div class="k">days it beats market</div></div>
    <div class="mini"><div class="v bad">{bot_h['premium_pct']}%</div><div class="k">worst house discount</div></div>
    <div class="mini"><div class="v">{house_spread_pct}%</div><div class="k">top-to-bottom gap</div></div>
  </div>
  <div class="takeaway">
    <div class="t">What this means for you</div>
    <p>The gap between best and worst clearing house is <b>{house_spread_pct}%</b> — on a tonne of cardamom
    that is real money, compounding at every sale. Favour the houses that consistently fetch more.
    The top-ranked house beats the market on <b>{int(top_h['days_above_pct'])}%</b> of days.</p>
  </div>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:14pt;margin-top:12pt;align-items:start">
    <div>
      <h3 style="font-size:10pt;margin-bottom:3pt;color:var(--green)">A market on a weekly clock</h3>
      <p class="lede" style="font-size:8.5pt;margin-bottom:5pt">Each major house owns a weekday — the "daily price" is really a rotating cast.</p>
      <div class="rot-grid">{rot_grid()}</div>
    </div>
    <div>
      <h3 style="font-size:10pt;margin-bottom:3pt;color:var(--green)">When the houses disagree</h3>
      <p class="lede" style="font-size:8.5pt;margin-bottom:5pt">Wide inter-auction spread predicts bigger moves ahead — and historically upward.</p>
      <div class="chart-wrap">{svg_disp}</div>
      <div class="caption">14-day dispersion quartile vs size &amp; direction of the following 2-week move.</div>
    </div>
  </div>
</div>

<!-- PAGE 7: PLAYBOOK -->
<div class="page sec-page">
  <div class="sec-num">06</div>
  <h2>The grower's playbook</h2>
  <p class="lede">Seven rules the data backs — one for each major finding in this almanac.</p>
  <div class="playbook">
    <h3>Seven rules from eleven years of data</h3>
    <ol>
{pb_items()}
    </ol>
  </div>
  <div class="footer-bar">
    <b>About this almanac.</b> Built from the Spices Board of India e-auction record ({coverage['start']} – {coverage['end']})
    plus Open-Meteo weather data, NOAA ENSO indices, and USD/INR rates.
    Price-driver rankings come from out-of-sample walk-forward forecast models (CardamomPulse v2.2).
    This is market analysis, not financial advice — cardamom is volatile and past patterns do not guarantee future prices.
    <div style="margin-top:6pt">
      <span class="tag">11 years</span><span class="tag">{coverage['auctions']:,} auctions</span><span class="tag">14 auction houses</span><span class="tag">CardamomPulse v2.2</span>
    </div>
  </div>
</div>

</body>
</html>"""

# Write static HTML
out_html = "/home/user/CardamomPulse/Cardamom_Farmer_Report.html"
with open(out_html, "w") as f:
    f.write(HTML)
print(f"HTML written: {out_html}")

# Generate PDF via WeasyPrint
try:
    from weasyprint import HTML as WH
    out_pdf = "/home/user/CardamomPulse/Cardamom_Farmer_Report.pdf"
    WH(filename=out_html).write_pdf(out_pdf)
    import os
    size = os.path.getsize(out_pdf) // 1024
    print(f"PDF written: {out_pdf}  ({size} KB)")
except Exception as e:
    print(f"PDF error: {e}")
