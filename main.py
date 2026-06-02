"""
Trading Algorithmique et Arbitrage Statistique
Stratégie : Arbitrage Statistique sur indices CDS (CDX 5Y / ITX 5Y)
            ou sur cryptomonnaies (Bitcoin / Ethereum)
Modèles   : OLS glissant + Filtre de Kalman
M2 MFCA   - Université de Lille
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from matplotlib.dates import DateFormatter

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import load_crypto_data, load_cds_data
from src.models import adf_test, test_cointegration, rolling_ols, kalman_filter_regression
from src.strategy import compute_zscore, generate_signals, backtest, count_trades
from src.performance import compute_metrics, print_metrics_table

# ============================================================
#  PARAMÈTRES DE CONFIGURATION
# ============================================================
DATASET         = 'cds'     # 'cds'    → indices CDX / ITX (recommandé pour StatArb)
                             # 'crypto' → Bitcoin / Ethereum
DATA_PATH       = 'Lille1M2 - Evaluation.xlsx'
OUTPUT_DIR      = 'outputs'
INITIAL_CAPITAL = 100.0     # trésorerie initiale (€)
WINDOW          = 60        # fenêtre OLS glissant / Z-Score (jours)
ENTRY_THRESH    = 2.0       # seuil d'entrée (écarts-types)
EXIT_THRESH     = 0.5       # seuil de sortie (écarts-types)
KALMAN_DELTA    = 1e-4      # vitesse d'adaptation du filtre de Kalman
TRAIN_RATIO     = 0.60      # fraction données in-sample

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
#  CONFIGURATION DATASET
# ============================================================
if DATASET == 'cds':
    ANN      = 252          # jours de trading annuels (marchés obligataires)
    raw      = load_cds_data(DATA_PATH)
    y_col    = 'CDX'        # variable dépendante  (CDS price = spread × 5)
    x_col    = 'ITX'        # variable explicative
    y_label  = 'CDX 5Y (prix)'
    x_label  = 'ITX 5Y (prix)'
    title_ds = 'CDX 5Y / ITX 5Y — Indices CDS'
    C_y      = '#E63946'    # rouge   CDX
    C_x      = '#457B9D'    # bleu    ITX
else:
    ANN      = 365          # crypto ouvre 365 j/an
    raw      = load_crypto_data(DATA_PATH)
    y_col    = 'Ethereum'
    x_col    = 'Bitcoin'
    y_label  = 'Ethereum (USD)'
    x_label  = 'Bitcoin (USD)'
    title_ds = 'Ethereum / Bitcoin — Cryptomonnaies'
    C_y      = '#627EEA'    # bleu    ETH
    C_x      = '#F7931A'    # orange  BTC

# Couleurs communes
C_OLS = '#E63946'
C_KF  = '#2A9D8F'

# Styles matplotlib
plt.rcParams.update({
    'figure.dpi': 130,
    'font.family': 'DejaVu Sans',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'grid.alpha': 0.35,
})

BANNER = "=" * 68

# ============================================================
print(BANNER)
print(f"  TRADING ALGORITHMIQUE — ARBITRAGE STATISTIQUE")
print(f"  Dataset : {title_ds}")
print(f"  M2 MFCA - Université de Lille")
print(BANNER)

# ============================================================
# 1. CHARGEMENT DES DONNÉES
# ============================================================
print("\n[1/6] Chargement des données...")

prices     = raw[[y_col, x_col]].copy()
log_p      = np.log(prices)
n          = len(prices)
split_idx  = int(n * TRAIN_RATIO)
split_date = prices.index[split_idx]

log_y = log_p[y_col]
log_x = log_p[x_col]

print(f"  Observations : {n}")
print(f"  Période      : {prices.index[0].date()} → {prices.index[-1].date()}")
print(f"  In-sample    : jusqu'au {split_date.date()} ({split_idx} obs)")
print(f"  Out-of-sample: après {split_date.date()} ({n - split_idx} obs)")
print()
print(prices.describe().round(4).to_string())

# ============================================================
# 2. TESTS STATISTIQUES
# ============================================================
print(f"\n[2/6] Tests de stationnarité et de cointégration...")

print(f"\n  Test ADF sur les log-prix (H0 : racine unitaire) :")
adf_y = adf_test(log_y, f'log({y_col})')
adf_x = adf_test(log_x, f'log({x_col})')

print(f"\n  Test ADF sur les log-rendements :")
adf_test(log_y.diff().dropna(), f'Δlog({y_col})')
adf_test(log_x.diff().dropna(), f'Δlog({x_col})')

print(f"\n  Test de cointégration Engle-Granger (in-sample, H0 : pas de cointégration) :")
is_coint, p_coint = test_cointegration(
    log_y.iloc[:split_idx], log_x.iloc[:split_idx],
    f'log({y_col})', f'log({x_col})',
)
if not adf_y and not adf_x:
    print(f"  → Les deux séries sont I(1) : le test de cointégration est approprié.")
else:
    print(f"  → Au moins une série est I(0) : régression linéaire classique (OLS).")

# ============================================================
# 3. ESTIMATION DES MODÈLES
# ============================================================
print(f"\n[3/6] Estimation des modèles (y=log({y_col}), x=log({x_col}))...")

# OLS statique in-sample (référence)
X_tr = np.column_stack([np.ones(split_idx), log_x.iloc[:split_idx].values])
y_tr = log_y.iloc[:split_idx].values
c_stat, _, _, _ = np.linalg.lstsq(X_tr, y_tr, rcond=None)
alpha_s, beta_s = c_stat

# OLS glissant
alpha_ols, beta_ols, resid_ols = rolling_ols(log_y, log_x, window=WINDOW)

# Filtre de Kalman
alpha_kf, beta_kf, resid_kf = kalman_filter_regression(
    log_y, log_x, delta=KALMAN_DELTA
)

print(f"  OLS statique  (in-sample)  : α={alpha_s:.4f}, β={beta_s:.4f}")
print(f"  OLS glissant  (window={WINDOW}j) : β moyen={beta_ols.dropna().mean():.4f}, "
      f"β final={beta_ols.iloc[-1]:.4f}")
print(f"  Filtre Kalman (δ={KALMAN_DELTA}): β initial={beta_kf.iloc[0]:.4f}, "
      f"β final={beta_kf.iloc[-1]:.4f}")

# ============================================================
# 4. Z-SCORES ET SIGNAUX
# ============================================================
print(f"\n[4/6] Calcul des Z-Scores et signaux de trading...")

z_ols = compute_zscore(resid_ols, window=WINDOW)
z_kf  = compute_zscore(resid_kf,  window=WINDOW)

sig_ols = generate_signals(z_ols, entry=ENTRY_THRESH, exit_=EXIT_THRESH)
sig_kf  = generate_signals(z_kf,  entry=ENTRY_THRESH, exit_=EXIT_THRESH)

st_ols = count_trades(sig_ols)
st_kf  = count_trades(sig_kf)

print(f"  OLS glissant : long={st_ols['jours_long']}j, short={st_ols['jours_short']}j, "
      f"neutre={st_ols['jours_neutre']}j, trades={st_ols['nb_trades']}")
print(f"  Kalman       : long={st_kf['jours_long']}j, short={st_kf['jours_short']}j, "
      f"neutre={st_kf['jours_neutre']}j, trades={st_kf['nb_trades']}")

# ============================================================
# 5. BACKTESTING
# ============================================================
print(f"\n[5/6] Backtesting (capital = {INITIAL_CAPITAL:.0f} €, sans coûts de transaction)...")

pv_ols = backtest(log_y, log_x, beta_ols, sig_ols, INITIAL_CAPITAL)
pv_kf  = backtest(log_y, log_x, beta_kf,  sig_kf,  INITIAL_CAPITAL)

# Benchmarks (Buy & Hold)
r_y = log_y.diff().fillna(0.0)
r_x = log_x.diff().fillna(0.0)
pv_y_bh  = (INITIAL_CAPITAL * (1.0 + r_y).cumprod()).rename(f'{y_col} B&H')
pv_x_bh  = (INITIAL_CAPITAL * (1.0 + r_x).cumprod()).rename(f'{x_col} B&H')
pv_ew    = (INITIAL_CAPITAL * (1.0 + 0.5 * r_y + 0.5 * r_x).cumprod()).rename('50/50 B&H')

# ============================================================
# 6. MÉTRIQUES DE PERFORMANCE
# ============================================================
print(f"\n[6/6] Métriques de performance (facteur annualisation = {ANN} j/an)...")

m_ols  = compute_metrics(pv_ols,  'StatArb OLS',   ann=ANN)
m_kf   = compute_metrics(pv_kf,   'StatArb Kalman', ann=ANN)
m_y_bh = compute_metrics(pv_y_bh, f'{y_col} B&H',  ann=ANN)
m_x_bh = compute_metrics(pv_x_bh, f'{x_col} B&H',  ann=ANN)
m_ew   = compute_metrics(pv_ew,    '50/50 B&H',     ann=ANN)

print_metrics_table([m_ols, m_kf, m_y_bh, m_x_bh, m_ew])

best = m_kf if m_kf['sharpe'] >= m_ols['sharpe'] else m_ols
best_pv = pv_kf if m_kf['sharpe'] >= m_ols['sharpe'] else pv_ols
best_name = best['name']
print(f"\n  Meilleure stratégie : {best_name} "
      f"(Sharpe = {best['sharpe']:.2f}, CAGR = {best['cagr']:.2%})")

# ============================================================
# GRAPHIQUES
# ============================================================
print(f"\nGénération des graphiques → {OUTPUT_DIR}/")
date_fmt = DateFormatter('%Y')


# ——— Figure 1 : Vue d'ensemble des données ———
fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)
fig.suptitle(f'Vue d\'ensemble — {title_ds}', fontsize=14, fontweight='bold')

ax = axes[0]
ax2 = ax.twinx()
ax.plot(prices.index, prices[y_col], color=C_y, lw=1.2, label=f'{y_col} (gauche)')
ax2.plot(prices.index, prices[x_col], color=C_x, lw=1.2, label=f'{x_col} (droite)')
ax.set_ylabel(y_label, color=C_y); ax.tick_params(axis='y', labelcolor=C_y)
ax2.set_ylabel(x_label, color=C_x); ax2.tick_params(axis='y', labelcolor=C_x)
lines = ax.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0]
labs  = ax.get_legend_handles_labels()[1] + ax2.get_legend_handles_labels()[1]
ax.legend(lines, labs, loc='upper left', fontsize=9)
ax.set_title('Prix observés', fontsize=10)

ax = axes[1]
norm_y = 100 * prices[y_col] / prices[y_col].iloc[0]
norm_x = 100 * prices[x_col] / prices[x_col].iloc[0]
ax.plot(prices.index, norm_y, color=C_y, lw=1.2, label=y_col)
ax.plot(prices.index, norm_x, color=C_x, lw=1.2, label=x_col)
ax.axhline(100, color='grey', lw=0.8, ls='--')
ax.axvline(split_date, color='black', lw=0.8, ls=':', alpha=0.5, label='Fin in-sample')
ax.set_ylabel('Base 100'); ax.legend(fontsize=9)
ax.set_title('Performance normalisée (base 100)', fontsize=10)

ax = axes[2]
window_corr = 90
roll_corr = log_y.diff().rolling(window_corr).corr(log_x.diff())
ax.plot(prices.index, roll_corr, color='#457B9D', lw=1.2,
        label=f'Corrélation glissante ({window_corr}j)')
ax.axhline(roll_corr.mean(), color='grey', lw=0.8, ls='--',
           label=f'Moyenne = {roll_corr.mean():.2f}')
ax.set_ylabel('Corrélation'); ax.set_ylim(-0.2, 1.1)
ax.legend(fontsize=9)
ax.set_title('Corrélation des log-rendements (fenêtre 90 j)', fontsize=10)
ax.xaxis.set_major_formatter(date_fmt)

plt.tight_layout()
fig.savefig(f'{OUTPUT_DIR}/01_donnees_overview.png', bbox_inches='tight')
plt.close(fig)
print(f"  ✓ {OUTPUT_DIR}/01_donnees_overview.png")


# ——— Figure 2 : Bêta, résidus, nuage de points ———
fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=False)
fig.suptitle('Modèles — Évolution du Bêta et Résidus du Spread',
             fontsize=14, fontweight='bold')

ax = axes[0]
ax.plot(beta_kf.index,  beta_kf,  color=C_KF,  lw=1.3, label='Filtre de Kalman')
ax.plot(beta_ols.index, beta_ols, color=C_OLS, lw=1.0, alpha=0.75,
        label=f'OLS glissant ({WINDOW}j)')
ax.axhline(beta_s, color='grey', lw=0.9, ls='--',
           label=f'OLS statique β={beta_s:.3f}')
ax.axvline(split_date, color='black', lw=0.8, ls=':', alpha=0.5, label='Fin in-sample')
ax.set_ylabel('β'); ax.set_title(f'Bêta dynamique', fontsize=10)
ax.legend(fontsize=9, ncol=2); ax.xaxis.set_major_formatter(date_fmt)

ax = axes[1]
ax.plot(resid_kf.dropna().index,  resid_kf.dropna(),  color=C_KF,  lw=1.0,
        alpha=0.9, label='Kalman')
ax.plot(resid_ols.dropna().index, resid_ols.dropna(), color=C_OLS, lw=0.8,
        alpha=0.7, label=f'OLS ({WINDOW}j)')
ax.axhline(0, color='grey', lw=0.8, ls='--')
ax.axvline(split_date, color='black', lw=0.8, ls=':', alpha=0.5)
ax.set_ylabel('Résidu ε')
ax.set_title('Résidus du spread ε = log(y) − α − β·log(x)', fontsize=10)
ax.legend(fontsize=9); ax.xaxis.set_major_formatter(date_fmt)

ax = axes[2]
ax.scatter(log_x, log_y, color='#457B9D', alpha=0.15, s=4, label='Données')
x_range = np.linspace(log_x.min(), log_x.max(), 300)
ax.plot(x_range, alpha_s + beta_s * x_range, color=C_OLS, lw=1.8,
        label=f'OLS statique : α={alpha_s:.3f}, β={beta_s:.3f}')
ax.set_xlabel(f'log({x_col})'); ax.set_ylabel(f'log({y_col})')
ax.set_title(f'Nuage de points log({y_col}) vs log({x_col})', fontsize=10)
ax.legend(fontsize=9)

plt.tight_layout()
fig.savefig(f'{OUTPUT_DIR}/02_modeles_beta_residus.png', bbox_inches='tight')
plt.close(fig)
print(f"  ✓ {OUTPUT_DIR}/02_modeles_beta_residus.png")


# ——— Figure 3 : Z-Scores et signaux ———
fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
fig.suptitle('Z-Scores et Signaux de Trading', fontsize=14, fontweight='bold')

for ax, z, sig, label, color in [
    (axes[0], z_ols, sig_ols, f'OLS glissant ({WINDOW}j)', C_OLS),
    (axes[1], z_kf,  sig_kf,  'Filtre de Kalman',          C_KF),
]:
    idx = z.index
    for i in range(1, len(idx)):
        s = sig.iloc[i - 1]
        if s == 1:
            ax.axvspan(idx[i - 1], idx[i], color='#2A9D8F', alpha=0.12, lw=0)
        elif s == -1:
            ax.axvspan(idx[i - 1], idx[i], color='#E63946', alpha=0.12, lw=0)

    ax.plot(idx, z, color=color, lw=0.9, label=label)
    ax.axhline(0, color='grey', lw=0.6, ls='--', alpha=0.6)
    ax.axhline( ENTRY_THRESH, color='#E63946', lw=1.0, ls='--', alpha=0.8,
                label=f'+{ENTRY_THRESH}σ entrée short')
    ax.axhline(-ENTRY_THRESH, color='#2A9D8F', lw=1.0, ls='--', alpha=0.8,
                label=f'−{ENTRY_THRESH}σ entrée long')
    ax.axhline( EXIT_THRESH,  color='grey', lw=0.7, ls=':',
                label=f'±{EXIT_THRESH}σ sortie')
    ax.axhline(-EXIT_THRESH,  color='grey', lw=0.7, ls=':')
    ax.axvline(split_date, color='black', lw=0.8, ls=':', alpha=0.5, label='Fin in-sample')
    ax.set_ylabel('Z-Score'); ax.set_ylim(-5.5, 5.5)
    ax.legend(fontsize=8, ncol=3, loc='upper right')
    ax.set_title(label, fontsize=10)

axes[1].xaxis.set_major_formatter(date_fmt)
fig.text(0.5, 0.005,
         'Fond vert = Long spread  |  Fond rouge = Short spread',
         ha='center', fontsize=9, color='grey')
plt.tight_layout(rect=[0, 0.02, 1, 1])
fig.savefig(f'{OUTPUT_DIR}/03_zscore_signaux.png', bbox_inches='tight')
plt.close(fig)
print(f"  ✓ {OUTPUT_DIR}/03_zscore_signaux.png")


# ——— Figure 4 : Performance portefeuille ———
fig = plt.figure(figsize=(13, 12))
gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

# 4a — Valeur du portefeuille
ax_pv = fig.add_subplot(gs[0, :])
ax_pv.plot(pv_ols.index,  pv_ols,   color=C_OLS, lw=1.6, label='StatArb OLS')
ax_pv.plot(pv_kf.index,   pv_kf,    color=C_KF,  lw=1.6, label='StatArb Kalman')
ax_pv.plot(pv_y_bh.index, pv_y_bh,  color=C_y,   lw=1.0, ls='--', alpha=0.7,
           label=f'{y_col} B&H')
ax_pv.plot(pv_x_bh.index, pv_x_bh,  color=C_x,   lw=1.0, ls='--', alpha=0.7,
           label=f'{x_col} B&H')
ax_pv.plot(pv_ew.index,   pv_ew,    color='grey', lw=1.0, ls=':',  alpha=0.7,
           label='50/50 B&H')
ax_pv.axhline(INITIAL_CAPITAL, color='grey', lw=0.8, ls='--', alpha=0.4)
ax_pv.axvline(split_date, color='black', lw=0.8, ls=':', alpha=0.4, label='Fin in-sample')
ax_pv.set_ylabel('Valeur (€)')
ax_pv.set_title(f'Valeur du portefeuille — capital initial {INITIAL_CAPITAL:.0f} €',
                fontsize=11, fontweight='bold')
ax_pv.legend(fontsize=9, ncol=3)
ax_pv.xaxis.set_major_formatter(date_fmt)

# 4b — Drawdowns
ax_dd = fig.add_subplot(gs[1, :])
for pv, lbl, col in [
    (pv_ols,  'StatArb OLS',   C_OLS),
    (pv_kf,   'StatArb Kalman', C_KF),
    (pv_y_bh, f'{y_col} B&H',  C_y),
]:
    dd = (pv - pv.cummax()) / pv.cummax() * 100
    ax_dd.plot(dd.index, dd, lw=1.2, label=lbl, color=col, alpha=0.85)
ax_dd.fill_between(
    ((pv_ols - pv_ols.cummax()) / pv_ols.cummax()).index,
    ((pv_ols - pv_ols.cummax()) / pv_ols.cummax()) * 100,
    alpha=0.12, color=C_OLS,
)
ax_dd.axhline(0, color='grey', lw=0.7)
ax_dd.axvline(split_date, color='black', lw=0.8, ls=':', alpha=0.4)
ax_dd.set_ylabel('Drawdown (%)')
ax_dd.set_title('Drawdowns', fontsize=10)
ax_dd.legend(fontsize=9)
ax_dd.xaxis.set_major_formatter(date_fmt)

# 4c — Distribution des rendements actifs
ax_dist = fig.add_subplot(gs[2, 0])
active_ols = m_ols['_ret'][m_ols['_ret'] != 0]
active_kf  = m_kf['_ret'][m_kf['_ret'] != 0]
ax_dist.hist(active_ols * 100, bins=50, color=C_OLS, alpha=0.55,
             label='OLS', density=True)
ax_dist.hist(active_kf * 100,  bins=50, color=C_KF,  alpha=0.55,
             label='Kalman', density=True)
ax_dist.axvline(0, color='grey', lw=1)
ax_dist.set_xlabel('Rendement journalier (%)')
ax_dist.set_ylabel('Densité')
ax_dist.set_title('Distribution des rendements (jours actifs)', fontsize=10)
ax_dist.legend(fontsize=9)

# 4d — PnL mensuel (meilleure stratégie)
ax_pnl = fig.add_subplot(gs[2, 1])
monthly = best_pv.resample('ME').last().pct_change().dropna() * 100
bar_colors = [C_KF if v >= 0 else C_OLS for v in monthly.values]
ax_pnl.bar(range(len(monthly)), monthly.values, color=bar_colors, width=0.8, alpha=0.85)
ax_pnl.axhline(0, color='grey', lw=0.8)
ax_pnl.set_xlabel('Mois (index chronologique)')
ax_pnl.set_ylabel('Rendement mensuel (%)')
ax_pnl.set_title(f'PnL mensuel — {best_name}', fontsize=10)

fig.suptitle(f'Résultats du Backtest — {title_ds}',
             fontsize=14, fontweight='bold', y=1.005)
fig.savefig(f'{OUTPUT_DIR}/04_backtest_performance.png', bbox_inches='tight')
plt.close(fig)
print(f"  ✓ {OUTPUT_DIR}/04_backtest_performance.png")


# ——— Figure 5 : Analyse des résidus ———
from statsmodels.graphics.tsaplots import plot_acf
from scipy import stats as sp_stats

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle('Analyse des résidus — Filtre de Kalman',
             fontsize=13, fontweight='bold')

valid_kf = resid_kf.dropna()

ax = axes[0]
plot_acf(valid_kf, lags=min(40, len(valid_kf) // 5), ax=ax,
         color=C_KF, alpha=0.05, zero=False)
ax.set_title('Autocorrélation des résidus', fontsize=10)
ax.set_xlabel('Lag (jours)')

ax = axes[1]
std_resid = (valid_kf - valid_kf.mean()) / valid_kf.std()
(qt, qs), (slope, intercept, _) = sp_stats.probplot(std_resid, dist='norm')
ax.scatter(qt, qs, color=C_KF, s=4, alpha=0.45, label='Résidus Kalman normalisés')
ax.plot(qt, slope * np.array(qt) + intercept, color='grey', lw=1.5,
        ls='--', label='Référence N(0,1)')
ax.set_xlabel('Quantiles théoriques')
ax.set_ylabel('Quantiles observés')
ax.set_title('QQ-Plot des résidus normalisés', fontsize=10)
ax.legend(fontsize=9)

plt.tight_layout()
fig.savefig(f'{OUTPUT_DIR}/05_analyse_residus.png', bbox_inches='tight')
plt.close(fig)
print(f"  ✓ {OUTPUT_DIR}/05_analyse_residus.png")

# ============================================================
print(f"\n{BANNER}")
print(f"  Analyse terminée. Graphiques dans : {OUTPUT_DIR}/")
print(BANNER)
