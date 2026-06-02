# Projet de Trading — Arbitrage Statistique M2 MFCA

**Cours :** Trading Algorithmique et Arbitrage Statistique (en Basse et Haute Fréquence)  
**Enseignant :** Diquel Dos Santos  
**Université de Lille — M2 MFCA**

---

## Stratégie implémentée

**Arbitrage Statistique** sur les indices CDS (CDX 5Y / ITX 5Y), avec comme modèle alternatif la paire Bitcoin / Ethereum.

- Modèle 1 : **Régression OLS glissante** (fenêtre 60 jours)
- Modèle 2 : **Filtre de Kalman** (coefficients α, β variants dans le temps)
- Signaux basés sur le **Z-Score** des résidus du spread
- Backtesting avec schéma **Equal Capital**, capital initial 100 €, sans coûts de transaction

---

## Lancer l'analyse

```bash
pip install -r requirements.txt
python main.py
```

Les graphiques sont enregistrés dans le dossier `outputs/`.

Pour utiliser les données **Bitcoin / Ethereum**, modifier dans `main.py` :
```python
DATASET = 'crypto'   # 'cds' (défaut) ou 'crypto'
```

---

## Paramètres (configurables dans `main.py`)

| Paramètre       | Valeur par défaut | Description                              |
|-----------------|-------------------|------------------------------------------|
| `WINDOW`        | 60                | Fenêtre OLS glissant / Z-Score (jours)   |
| `ENTRY_THRESH`  | 2.0               | Seuil d'entrée (σ)                       |
| `EXIT_THRESH`   | 0.5               | Seuil de sortie (σ)                      |
| `KALMAN_DELTA`  | 1e-4              | Vitesse d'adaptation du filtre de Kalman |
| `TRAIN_RATIO`   | 0.60              | Fraction in-sample                       |

---

## Résultats principaux (CDS — CDX/ITX)

| Métrique              | StatArb OLS | StatArb Kalman |
|-----------------------|-------------|----------------|
| CAGR                  | +22.22 %    | **+22.67 %**   |
| Ratio de Sharpe       | 1.19        | **1.52**        |
| Max Drawdown          | -25.51 %    | **-16.67 %**   |
| Taux de gains (actif) | 54.0 %      | **63.0 %**     |

Le **filtre de Kalman** surperforme l'OLS glissant grâce à son adaptation continue du bêta,
ce qui réduit le délai de réaction aux changements de régime du marché du crédit.

---

## Structure du projet

```
trading/
├── main.py                    # Script principal (lancer l'analyse)
├── requirements.txt           # Dépendances Python
├── Lille1M2 - Evaluation.xlsx # Données (CDS + Crypto)
├── Lille1M2 - Evaluation.docx # Sujet de l'évaluation
├── src/
│   ├── data_loader.py         # Chargement des données Excel
│   ├── models.py              # OLS glissant, Filtre de Kalman, ADF, cointégration
│   ├── strategy.py            # Z-Score, signaux, backtest
│   └── performance.py         # Métriques (Sharpe, Calmar, MaxDD…)
└── outputs/                   # Graphiques générés
    ├── 01_donnees_overview.png
    ├── 02_modeles_beta_residus.png
    ├── 03_zscore_signaux.png
    ├── 04_backtest_performance.png
    └── 05_analyse_residus.png
```
