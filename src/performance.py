"""
Calcul des métriques de performance et affichage des résultats.
"""
import numpy as np
import pandas as pd


def compute_metrics(
    portfolio_value: pd.Series,
    name: str = 'Stratégie',
    ann: int = 252,
) -> dict:
    """
    Métriques de performance annualisées.

    Paramètres
    ----------
    ann : facteur d'annualisation (252 pour CDS, 365 pour crypto)
    """
    pv = portfolio_value.dropna()
    ret = pv.pct_change().dropna()

    years = (pv.index[-1] - pv.index[0]).days / 365.25
    total_ret = pv.iloc[-1] / pv.iloc[0] - 1
    cagr = (1.0 + total_ret) ** (1.0 / years) - 1
    vol = ret.std() * np.sqrt(ann)
    sharpe = (ret.mean() * ann) / (ret.std() * np.sqrt(ann)) if vol > 0 else 0

    rolling_max = pv.cummax()
    drawdown = (pv - rolling_max) / rolling_max
    max_dd = drawdown.min()
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0

    down_ret = ret[ret < 0]
    down_std = down_ret.std() * np.sqrt(ann) if len(down_ret) > 1 else np.nan
    sortino = (ret.mean() * ann) / down_std if (down_std and down_std > 0) else 0

    # Win rate uniquement sur les jours avec position active
    active = ret[ret != 0]
    win_rate = (active > 0).mean() if len(active) > 0 else 0.0

    return {
        'name': name,
        'total_return': total_ret,
        'cagr': cagr,
        'volatility': vol,
        'sharpe': sharpe,
        'sortino': sortino,
        'max_drawdown': max_dd,
        'calmar': calmar,
        'win_rate': win_rate,
        'final_value': pv.iloc[-1],
        '_pv': pv,
        '_ret': ret,
    }


def print_metrics_table(metrics_list: list) -> None:
    """Affiche un tableau comparatif des métriques."""
    names = [m['name'] for m in metrics_list]
    col_w = max(18, max(len(n) for n in names) + 2)
    lw = 26
    sep = '=' * (lw + (col_w + 1) * len(names))
    fmt_h = f"{{:<{lw}}}" + f" {{:>{col_w}}}" * len(names)
    fmt_r = f"{{:<{lw}}}" + f" {{:>{col_w}}}" * len(names)

    print(f"\n{sep}")
    print(fmt_h.format('Métrique', *names))
    print(sep)

    rows = [
        ('Rendement total',    'total_return', '{:.2%}'),
        ('CAGR',               'cagr',         '{:.2%}'),
        ('Volatilité (ann.)',  'volatility',   '{:.2%}'),
        ('Ratio de Sharpe',    'sharpe',        '{:.2f}'),
        ('Ratio de Sortino',   'sortino',       '{:.2f}'),
        ('Max Drawdown',       'max_drawdown',  '{:.2%}'),
        ('Ratio de Calmar',    'calmar',        '{:.2f}'),
        ('Taux de gains*',     'win_rate',      '{:.2%}'),
        ('Capital final (€)',  'final_value',   '{:.2f}'),
    ]
    for label, key, fmt in rows:
        values = [fmt.format(m[key]) for m in metrics_list]
        print(fmt_r.format(label, *values))
    print(sep)
    print('  * Taux de gains calculé sur les jours avec position active uniquement.')
