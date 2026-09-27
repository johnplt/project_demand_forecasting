import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from src.data_loader import load_raw_data, merge_and_clean_data
from src.features import prepare_full_features
from src.metrics import (
    compute_inventory_costs,
    compute_wape,
    optimize_safety_stock_buffer,
)
from src.train import time_based_train_test_split, train_lightgbm_model
import matplotlib.pyplot as plt
import shap
# -----------------------------------------------------------------------------
# Configuration de la page
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Prévision de la Demande & Optimisation des Stocks",
    page_icon="📦",
    layout="wide",
)

st.title("📦 Prévision Dynamique de la Demande & Optimisation des Stocks")
st.markdown(
    "**Démarche Data Science de bout en bout :** Du besoin P&L Supply Chain au pilotage interactif du stock de sécurité."
)

# -----------------------------------------------------------------------------
# Chargement et mise en cache des données (Exécution unique)
# -----------------------------------------------------------------------------


@st.cache_data
def get_processed_data():
    sales_df, stores_df = load_raw_data(
        "data/raw/train.csv", "data/raw/store.csv"
    )
    cleaned_df = merge_and_clean_data(sales_df, stores_df)
    full_df = prepare_full_features(cleaned_df)
    return full_df


# Chargement sécurisé avec fallback d'exemple si les fichiers ne sont pas encore présents
try:
    df = get_processed_data()
    data_loaded = True
except Exception as e:
    st.error(f"Erreur lors du chargement des données : {e}")
    data_loaded = False

# -----------------------------------------------------------------------------
# Structuration des Onglets principaux
# -----------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(
    [
        "1. Cadrage Métier/Matrice des coûts",
        "2. Ingénierie des données et variables temporelles",
        "3. Comparatif des modèles (Benchmark)",
        "4. Simulateur et stock de sécurité",
    ]
)

# =============================================================================
# ONGLET 1 : Cadrage métier
# =============================================================================
with tab1:
    st.subheader("🎯 Cadrage Business & Arbitrage des Erreurs")

    st.markdown("""
    En gestion des stocks et *Supply Chain*, une prévision parfaite n'existe pas. L'enjeu fondamental consiste à **arbitrer le coût des erreurs** plutôt que de chercher une performance statistique théorique :
    
    * **Sous-prévision (Rupture de stock) :** Entraîne une perte directe de chiffre d'affaires, une pénalité de service et une insatisfaction client.
    * **Sur-prévision (Sur-stockage) :** Entraîne des coûts d'immobilisation de trésorerie, d'entreposage et un risque de dépréciation/périssabilité.
    """)

    st.markdown("---")
    st.markdown("### 💶 Hypothèses financières du modèle")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            label="Coût d'une rupture (Sous-stock)",
            value="10,00 € / unité",
            delta="Marge brute perdue",
            delta_color="inverse",
        )
    with col2:
        st.metric(
            label="Coût du sur-stockage",
            value="2,00 € / unité",
            delta="Frais de possession",
            delta_color="inverse",
        )
    with col3:
        st.metric(
            label="Métrique métier clé",
            value="WAPE (%)",
            help="Weighted Absolute Percentage Error : Métrique standard en Supply Chain",
        )

    st.info(
        "**Objectif de l'application :** Démontrer comment ajuster le seuil de décision (Stock de sécurité) pour minimiser la somme globale des coûts d'erreur."
    )

    st.markdown("### 🍦 Exemple concret : L'histoire de la boutique")
    st.markdown("""
    Imagine un gérant de boutique (ex: un glacier ou une supérette) :
    * S'il commande **10 glaces de trop**, il paye $10 \\times 2,00 € = 20,00 €$ de frais d'invendus / stockage.
    * S'il lui manque **10 glaces**, 10 clients repartent les mains vides : c'est $10 \\times 10,00 € = 100,00 €$ de marge perdue.
    
    *Le rôle de notre modèle n'est pas seulement de deviner le chiffre exact, mais de **trouver le bon niveau de stock de sécurité** pour minimiser ce coût total.*
    """)

# =============================================================================
# ONGLET 2 : Ingénierie des données
# =============================================================================
with tab2:
    st.subheader(
        "🛠️ Construction des variables & prévention des fuites (zero leakage)"
    )

    st.markdown("""
    Pour prédire les ventes sans fuite d'information (*Data Leakage*), toutes les variables historiques (lags et statistiques glissantes) sont calculées **uniquement sur le passé strict** ($J-1$ et antérieurs).
    """)

    if data_loaded:
        st.markdown("### 📊 Extrait des données transformées")
        st.dataframe(
            df[
                [
                    "Date",
                    "Store",
                    "Sales",
                    "Sales_lag_7",
                    "Sales_rolling_mean_7",
                    "Sales_rolling_std_7",
                ]
            ].head(10),
            use_container_width=True,
        )

        st.markdown("### 📋 Aperçu & Profilage des Données")

        col_data1, col_data2 = st.columns(2)

        with col_data1:
            st.markdown("**Statistiques descriptives de la variable cible (`Sales`)**")
            st.dataframe(df["Sales"].describe().to_frame().T, use_container_width=True)

        with col_data2:
            st.markdown("**Couverture temporelle & répartition**")
            st.write(
                f"- **Période :** Du {df['Date'].min().strftime('%d/%m/%Y')} au {df['Date'].max().strftime('%d/%m/%Y')}"
            )
            st.write(f"- **Nombre de magasins distincts :** {df['Store'].nunique()}")
            st.write(f"- **Total des observations :** {len(df):,} jours/magasins")

        st.markdown("### 📈 Série temporelle d'un magasin type")
        store_id = st.selectbox(
            "Sélectionner un magasin pour l'analyse :",
            options=sorted(df["Store"].unique()),
        )
        sample_store_df = df[df["Store"] == store_id].sort_values("Date")

        fig_ts = px.line(
            sample_store_df,
            x="Date",
            y=["Sales", "Sales_rolling_mean_7"],
            labels={"value": "Ventes (Unités)", "variable": "Légende"},
            title=f"Historique des ventes et Moyenne Mobile (7 jours) - Magasin n°{store_id}",
        )
        st.plotly_chart(fig_ts, use_container_width=True)
    else:
        st.warning(
            "Veuillez placer les fichiers de données dans `data/raw/` pour afficher les visualisations interactives."
        )

# =============================================================================
# ONGLET 3 : Comparatif des Modèles
# =============================================================================
with tab3:
    st.subheader("🤖 Benchmarking : Modèle Baseline vs Machine Learning")

    st.markdown("""
    Nous comparons deux approches sur une fenêtre de test stricte (*Split temporel*) :
    1. **Baseline Naïve :** Moyenne mobile glissante sur 7 jours.
    2. **LightGBM :** Modèle d'apprentissage supervisé exploitant l'ensemble des caractéristiques calendaires et historiques.
    """)

    if data_loaded:
        max_date = df["Date"].max()
        test_days = 60
        cutoff_date = max_date - pd.Timedelta(days=test_days)

        # Affichage explicatif du split temporel
        st.info(
            f"📅 **Période d'Entraînement :** Jusqu'au {cutoff_date.strftime('%d/%m/%Y')} | "
            f"🧪 **Période de Test (Évaluation) :** Du {cutoff_date.strftime('%d/%m/%Y')} au {max_date.strftime('%d/%m/%Y')} ({test_days} jours)"
        )

        train_df, test_df = time_based_train_test_split(
            df, cutoff_date=cutoff_date
        )

        feature_cols = [
            c
            for c in df.columns
            if c
            not in [
                "Date",
                "Sales",
                "Customers",
                "Open",
                "StateHoliday",
                "SchoolHoliday",
            ]
        ]

        if st.button("🚀 Lancer l'entraînement et l'évaluation"):
            with st.spinner("Entraînement et analyse des logs en cours..."):
                # 1. Récupère les 4 éléments renvoyés par la nouvelle fonction
                model, preds, shap_values, X_test_sample = train_lightgbm_model(
                    train_df, test_df, feature_cols
                )

                y_true = test_df["Sales"].values
                preds_baseline = test_df["Sales_rolling_mean_7"].values
                preds_model = preds.values

                # 2. Stocke-les dans la session pour pouvoir les réutiliser (notamment SHAP)
                st.session_state["y_true"] = y_true
                st.session_state["y_pred"] = preds_model
                st.session_state["y_pred_baseline"] = preds_baseline
                st.session_state["test_df"] = test_df
                st.session_state["shap_values"] = shap_values  # <-- AJOUT
                st.session_state["X_test_sample"] = X_test_sample  # <-- AJOUT

                wape_base = compute_wape(y_true, preds_baseline)
                wape_lgb = compute_wape(y_true, preds_model)

            st.success("✅ Entraînement terminé avec succès !")

            # --- A. TRANSFORMATION DES LOGS EN INFOS VISUELLES ---
            st.markdown("### 🔍 Journal d'Exécution & Diagnostic (Ex-Logs)")

            col_log1, col_log2, col_log3 = st.columns(3)
            col_log1.metric(
                "Données d'entraînement", f"{len(train_df):,} lignes"
            )
            col_log2.metric("Données de test", f"{len(test_df):,} lignes")
            col_log3.metric("Variables utilisées", f"{len(feature_cols)} features")

            # --- B. PERFORMANCE GLOBALE ---
            st.markdown("### 📊 Résultats sur la Période de Test")
            c_res1, c_res2 = st.columns(2)
            c_res1.metric(
                "WAPE Baseline (Moyenne 7j)",
                f"{wape_base:.2f} %",
            )
            c_res2.metric(
                "WAPE LightGBM",
                f"{wape_lgb:.2f} %",
                delta=f"{wape_base - wape_lgb:.2f} % de réduction d'erreur",
            )

            # --- C. COMPARISON VISUELLE TEMPORELLE SUR LE TEST ---
            st.markdown("### 📈 Prédictions vs Réalité (Échantillon de Test)")

            # Agrégation par date pour voir la courbe globale sur la période test
            test_results_df = test_df[["Date", "Store", "Sales"]].copy()
            test_results_df["Baseline"] = preds_baseline
            test_results_df["LightGBM"] = preds_model

            daily_agg = (
                test_results_df.groupby("Date")[
                    ["Sales", "Baseline", "LightGBM"]
                ]
                .sum()
                .reset_index()
            )

            fig_compare = px.line(
                daily_agg,
                x="Date",
                y=["Sales", "Baseline", "LightGBM"],
                labels={"value": "Ventes Totales", "variable": "Légende"},
                title="Comparaison des Ventes Réelles vs Prédictions sur la période de Test",
                color_discrete_map={
                    "Sales": "black",
                    "Baseline": "orange",
                    "LightGBM": "#00CC96",
                },
            )
            st.plotly_chart(fig_compare, use_container_width=True)

            st.markdown("---")
            st.markdown("### 🧠 Explicabilité du Modèle (Valeurs SHAP)")
            st.markdown("""
            Les valeurs **SHAP (SHapley Additive exPlanations)** permettent d'interpréter le modèle :
            * **Quelles variables comptent le plus** dans la décision du modèle ?
            * **Quel est leur impact réel** (positif ou négatif) sur le niveau des ventes prédit ?
            """)
            
            if "shap_values" in st.session_state:
                shap_values = st.session_state["shap_values"]
                X_test_sample = st.session_state["X_test_sample"]
            
                col_shap1, col_shap2 = st.columns(2)
            
                with col_shap1:
                    st.markdown("**Importance globale des variables (Summary Plot)**")
                    fig_shap, ax = plt.subplots(figsize=(6, 4))
                    shap.summary_plot(
                        shap_values, X_test_sample, plot_type="bar", show=False
                    )
                    st.pyplot(fig_shap, clear_figure=True)
            
                with col_shap2:
                    st.markdown("**Impact et valeur des variables (Beeswarm Plot)**")
                    fig_beeswarm, ax2 = plt.subplots(figsize=(6, 4))
                    shap.summary_plot(shap_values, X_test_sample, show=False)
                    st.pyplot(fig_beeswarm, clear_figure=True)

            st.info(
                "💡 **Prochaine étape :** Rendez-vous dans l'**Onglet 4** pour simuler la conversion de ces écarts de prévision en impact financier (Euros)."
            )

# =============================================================================
# ONGLET 4 : Simulateur & stock de sécurité
# =============================================================================
with tab4:
    st.subheader(
        "🎛️ Simulation financière & optimisation du stock de sécurité"
    )

    st.markdown("""
    Ajustez le niveau de **stock de sécurité** (*Safety Stock Buffer*) pour observer son impact direct sur le coût global d'erreur.
    """)

    col_sim_left, col_sim_right = st.columns([1, 2])

    with col_sim_left:
        st.markdown("#### ⚙️ Paramètres de Simulation")
        cost_under = st.number_input(
            "Coût unitaire Rupture (€)", value=10.0, step=1.0
        )
        cost_over = st.number_input(
            "Coût unitaire sur-stockage (€)", value=2.0, step=0.5
        )

        buffer_slider = st.slider(
            "Stock de sécurité appliqué (%)",
            min_value=-20,
            max_value=30,
            value=0,
            step=1,
            help="Ajustement en % appliqué aux prédictions brutes du modèle",
        )

    with col_sim_right:
        st.markdown("#### 📊 Impact financier & Gain vs Baseline")

        if "y_true" in st.session_state and "y_pred" in st.session_state:
            y_true = st.session_state["y_true"]
            y_pred_base = st.session_state["y_pred"]
            y_pred_baseline = st.session_state.get("y_pred_baseline", None)

            # Application du buffer sur le modèle
            y_pred_adjusted = y_pred_base * (1 + (buffer_slider / 100.0))

            # Calcul des coûts du modèle
            costs_model = compute_inventory_costs(
                y_true,
                y_pred_adjusted,
                unit_cost_overstock=cost_over,
                unit_cost_understock=cost_under,
            )

            # Calcul des coûts de la Baseline (sans buffer ou avec buffer identique)
            if y_pred_baseline is not None:
                costs_baseline = compute_inventory_costs(
                    y_true,
                    y_pred_baseline * (1 + (buffer_slider / 100.0)),
                    unit_cost_overstock=cost_over,
                    unit_cost_understock=cost_under,
                )
                gain = costs_baseline["total_cost"] - costs_model["total_cost"]
            else:
                gain = 0.0

            # Cartes métriques comparatives
            c1, c2, c3 = st.columns(3)
            c1.metric(
                "Coût Modèle LightGBM",
                f"{costs_model['total_cost']:,.0f} €".replace(",", " "),
            )
            if y_pred_baseline is not None:
                c2.metric(
                    "Coût Baseline (Naïve)",
                    f"{costs_baseline['total_cost']:,.0f} €".replace(",", " "),
                )
                c3.metric(
                    "💰 Économie Réalisée",
                    f"{gain:,.0f} €".replace(",", " "),
                    delta=f"{gain:,.0f} € vs Baseline",
                )

            st.markdown("---")

            # Graphique d'optimisation du Buffer
            best_buf, df_opt = optimize_safety_stock_buffer(
                y_true,
                y_pred_base,
                cost_overstock=cost_over,
                cost_understock=cost_under,
            )

            fig_opt = px.line(
                df_opt,
                x="buffer_pct",
                y="total_cost",
                labels={
                    "buffer_pct": "Stock de sécurité (%)",
                    "total_cost": "Coût financier global (€)",
                },
                title="Courbe d'optimisation du coût en fonction du stock de sécurité",
            )
            fig_opt.add_vline(
                x=best_buf * 100,
                line_dash="dash",
                line_color="green",
                annotation_text=f"Buffer Optimal ({best_buf * 100:.1f}%)",
            )
            st.plotly_chart(fig_opt, use_container_width=True)
        else:
            st.info(
                "Veuillez d'abord entraîner le modèle dans l'Onglet 3 pour activer le simulateur."
            )