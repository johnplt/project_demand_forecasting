from typing import Tuple
import numpy as np
import pandas as pd


def load_raw_data(
    train_path: str, store_path: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Charge les données brutes des ventes et des caractéristiques des magasins."""
    df_sales = pd.read_csv(train_path, parse_dates=["Date"])
    df_stores = pd.read_csv(store_path)
    return df_sales, df_stores


def merge_and_clean_data(
    df_sales: pd.DataFrame, df_stores: pd.DataFrame
) -> pd.DataFrame:
    """Joint les jeux de données, nettoie les magasins fermés et traite les valeurs manquantes."""
    # 1. Ne garder que les jours où les magasins sont ouverts et avec des ventes > 0
    df = df_sales[(df_sales["Open"] == 1) & (df_sales["Sales"] > 0)].copy()

    # 2. Jointure avec la table magasins
    df = df.merge(df_stores, on="Store", how="left")

    # 3. Imputation des valeurs manquantes dans les métadonnées magasins
    # Si la distance au concurrent est inconnue, on met une valeur élevée (médiane/max)
    if "CompetitionDistance" in df.columns:
        df["CompetitionDistance"] = df["CompetitionDistance"].fillna(
            df["CompetitionDistance"].median()
        )

    # Si l'année/mois de début de concurrence/promo est inconnu -> 0
    cols_to_zero = [
        "CompetitionOpenSinceMonth",
        "CompetitionOpenSinceYear",
        "Promo2SinceWeek",
        "Promo2SinceYear",
    ]
    for c in cols_to_zero:
        if c in df.columns:
            df[c] = df[c].fillna(0).astype(int)

    # 4. Tri par magasin et par date (indispensable pour les séries temporelles)
    df = df.sort_values(by=["Store", "Date"]).reset_index(drop=True)

    return df