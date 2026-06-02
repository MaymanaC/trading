"""
Modèles statistiques pour l'arbitrage statistique.

- OLS glissant (regression linéaire sur fenêtre roulante)
- Filtre de Kalman (estimation dynamique du beta)
- Tests ADF et de cointégration (Engle-Granger)
"""
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint, adfuller


# ---------------------------------------------------------------------------
# Tests statistiques
# ---------------------------------------------------------------------------

def adf_test(series: pd.Series, name: str = 'Série') -> bool:
    """Test de Dickey-Fuller augmenté (H0 : racine unitaire)."""
    result = adfuller(series.dropna(), autolag='AIC')
    stationary = result[1] < 0.05
    label = 'Stationnaire ✓' if stationary else 'Racine unitaire'
    print(f"  {name:<40}: ADF={result[0]:>8.4f}, p={result[1]:.4f}  [{label}]")
    return stationary


def test_cointegration(
    s1: pd.Series, s2: pd.Series,
    name1: str = 'S1', name2: str = 'S2'
) -> tuple[bool, float]:
    """Test de cointégration d'Engle-Granger."""
    score, pvalue, _ = coint(s1, s2)
    coint_flag = pvalue < 0.05
    label = 'Cointégrées ✓' if coint_flag else 'Non cointégrées'
    print(f"  {name1} / {name2:<30}: score={score:.4f}, p={pvalue:.4f}  [{label}]")
    return coint_flag, pvalue


# ---------------------------------------------------------------------------
# Modèle 1 : OLS glissant
# ---------------------------------------------------------------------------

def rolling_ols(
    y: pd.Series, x: pd.Series, window: int = 60
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Régression OLS roulante : y_t = α_t + β_t · x_t + ε_t.

    Paramètres
    ----------
    y, x : Series de log-prix alignées
    window : taille de la fenêtre glissante (jours)

    Retourne alpha, beta, résidus (Series alignées sur l'index de y).
    """
    n = len(y)
    y_arr = y.values
    x_arr = x.values

    alphas = np.full(n, np.nan)
    betas = np.full(n, np.nan)
    residuals = np.full(n, np.nan)

    for i in range(window - 1, n):
        y_w = y_arr[i - window + 1: i + 1]
        x_w = x_arr[i - window + 1: i + 1]
        X = np.column_stack([np.ones(window), x_w])
        try:
            coeffs, _, _, _ = np.linalg.lstsq(X, y_w, rcond=None)
            alphas[i] = coeffs[0]
            betas[i] = coeffs[1]
            residuals[i] = y_arr[i] - coeffs[0] - coeffs[1] * x_arr[i]
        except np.linalg.LinAlgError:
            pass

    return (
        pd.Series(alphas, index=y.index, name='alpha_ols'),
        pd.Series(betas, index=y.index, name='beta_ols'),
        pd.Series(residuals, index=y.index, name='resid_ols'),
    )


# ---------------------------------------------------------------------------
# Modèle 2 : Filtre de Kalman
# ---------------------------------------------------------------------------

def kalman_filter_regression(
    y: pd.Series, x: pd.Series, delta: float = 1e-4
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Filtre de Kalman pour régression à coefficients variants dans le temps.

    Modèle d'état :
        θ_t = θ_{t-1} + w_t        w_t ~ N(0, Q)   (marche aléatoire)
        y_t = H_t · θ_t + v_t      v_t ~ N(0, R)   (observation)

    avec θ = [α, β]ᵀ et H_t = [1, x_t].
    Q = δ/(1-δ) · I  →  δ contrôle la vitesse d'adaptation du beta.

    Paramètres
    ----------
    delta : petite valeur (1e-5) = beta stable ; grande valeur (1e-3) = réactif
    """
    n = len(y)
    y_arr = y.values
    x_arr = x.values

    # Bruit de processus
    Q = (delta / (1.0 - delta)) * np.eye(2)

    # Estimation initiale par OLS sur les 30 premières observations
    warmup = min(30, n // 10)
    X_init = np.column_stack([np.ones(warmup), x_arr[:warmup]])
    theta, _, _, _ = np.linalg.lstsq(X_init, y_arr[:warmup], rcond=None)
    resid_init = y_arr[:warmup] - X_init @ theta
    R = float(np.var(resid_init)) or 1e-4  # Bruit d'observation

    P = np.eye(2) * 1.0  # Covariance initiale de l'état

    alphas = np.zeros(n)
    betas = np.zeros(n)
    residuals = np.zeros(n)

    for i in range(n):
        H = np.array([1.0, x_arr[i]])

        # --- Prédiction ---
        P_pred = P + Q

        # --- Innovation ---
        innovation = y_arr[i] - H @ theta
        S = float(H @ P_pred @ H) + R

        # --- Gain de Kalman ---
        K = (P_pred @ H) / S

        # --- Mise à jour ---
        theta = theta + K * innovation
        P = (np.eye(2) - np.outer(K, H)) @ P_pred

        alphas[i] = theta[0]
        betas[i] = theta[1]
        residuals[i] = innovation

    return (
        pd.Series(alphas, index=y.index, name='alpha_kf'),
        pd.Series(betas, index=y.index, name='beta_kf'),
        pd.Series(residuals, index=y.index, name='resid_kf'),
    )
