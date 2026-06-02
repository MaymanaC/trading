"""
Stratégie d'arbitrage statistique basée sur le Z-Score des résidus.

Logique de trading :
  - Z < -seuil_entrée  →  Long spread  (acheter ETH, vendre BTC)
  - Z >  seuil_entrée  →  Short spread (vendre ETH, acheter BTC)
  - |Z| < seuil_sortie →  Clôture de position

Schéma d'allocation : Equal Capital (50/50 chaque jambe).
Trésorerie initiale : 100 €, pas de coûts de transaction, taux nul.
"""
import numpy as np
import pandas as pd


def compute_zscore(residuals: pd.Series, window: int = 60) -> pd.Series:
    """Z-Score glissant des résidus du spread."""
    mu = residuals.rolling(window=window, min_periods=window).mean()
    sigma = residuals.rolling(window=window, min_periods=window).std()
    zscore = (residuals - mu) / sigma
    return zscore.rename('zscore')


def generate_signals(
    zscore: pd.Series,
    entry: float = 2.0,
    exit_: float = 0.5,
) -> pd.Series:
    """
    Génère les signaux de position à partir du Z-Score.

    Retourne une Series de positions :
        +1  : long spread
        -1  : short spread
         0  : neutre (aucune position)
    """
    signals = pd.Series(0.0, index=zscore.index, name='signal')
    pos = 0

    for i in range(len(zscore)):
        z = zscore.iloc[i]
        if pd.isna(z):
            continue

        if pos == 0:
            if z < -entry:
                pos = 1    # ETH sous-évalué → long spread
            elif z > entry:
                pos = -1   # ETH sur-évalué  → short spread
        elif pos == 1:
            if z >= -exit_:
                pos = 0
        elif pos == -1:
            if z <= exit_:
                pos = 0

        signals.iloc[i] = pos

    return signals


def backtest(
    log_y: pd.Series,
    log_x: pd.Series,
    beta: pd.Series,
    signals: pd.Series,
    initial_capital: float = 100.0,
) -> pd.Series:
    """
    Backtest basé sur les log-rendements — sans biais prospectif.

    Rendement du spread : r_spread_t = r_y_t − β_{t-1} · r_x_t
    Rendement du portefeuille : r_port_t = signal_{t-1} · r_spread_t

    La valeur du portefeuille démarre à initial_capital (€).
    """
    r_y = log_y.diff()
    r_x = log_x.diff()

    spread_ret = r_y - beta.shift(1) * r_x
    port_ret = signals.shift(1) * spread_ret
    port_ret = port_ret.fillna(0.0)

    pv = initial_capital * (1.0 + port_ret).cumprod()
    return pv.rename('portfolio_value')


def count_trades(signals: pd.Series) -> dict:
    """Compte le nombre de trades et la durée moyenne."""
    transitions = signals.diff().fillna(0)
    entries = transitions[transitions != 0]

    n_long = (signals == 1).sum()
    n_short = (signals == -1).sum()
    n_flat = (signals == 0).sum()

    # Compter les allers-retours (ouvertures)
    openings = 0
    prev = 0
    for v in signals:
        if v != 0 and v != prev:
            openings += 1
        prev = v

    return {
        'jours_long': int(n_long),
        'jours_short': int(n_short),
        'jours_neutre': int(n_flat),
        'nb_trades': openings,
    }
