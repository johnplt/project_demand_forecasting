# Prévision Dynamique de la Demande & Optimisation des Stocks

Ce projet propose une approche de bout en bout pour la prévision des ventes et l'optimisation des niveaux de stock avec une méthode de Machine Learning (LightGBM).


---

## Fonctionnalités Principales

* **Ingénierie de caractéristiques temporelles :** Prise en compte de la saisonnalité calendaire, des variables glissantes (rolling means/std) et des lags sans fuite de données (*Data Leakage*).
* **Benchmarking & Performance :** Comparaison entre une baseline simple (moyenne mobile 7j) et un modèle gradient boosting (LightGBM).
* **Explicabilité du modèle (SHAP) :** Analyse fine des facteurs d'influence (saisonnalité hebdomadaire, tendances de vente, effet magasin).
* **Simulateur d'impact financier :** Calcul interactif du coût total des invendus vs ruptures et recherche automatique du stock de sécurité (*Safety Stock Buffer*) optimal.
* **Interface Interactive :** Application web de pilotage développée avec **Streamlit**.

---

## Structure du projet

```text
.
├── data/                  # Données brutes et transformées
├── src/
│   ├── data_loader.py     # Chargement et nettoyage des données
│   ├── features.py        # Ingénierie des variables (Lags, Rolling stats)
│   ├── metrics.py         # Calculs WAPE, coûts financiers et optimisation buffer
│   └── train.py           # Split temporel, entraînement LightGBM & SHAP
├── app.py                 # Application Streamlit principale
├── pyproject.toml / uv.lock # Gestion des dépendances avec uv
└── README.md
```

## Installation et lancement avec uv
Cloner le dépôt :

```bash
git clone <URL_DU_REPO>
cd demand-forecasting-app
```

Créer l'environnement virtuel et installer les dépendances :

```bash
uv sync
```

Générer les données de simulation (si nécessaire) :

```bash
uv run python generate_mock_data.py
```
Lancer l'application Streamlit :

```bash
uv run streamlit run app.py
```