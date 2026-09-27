import os
import numpy as np
import pandas as pd


def generate_mock_dataset():
    os.makedirs("data/raw", exist_ok=True)

    dates = pd.date_range(start="2024-01-01", end="2025-12-31", freq="D")
    stores = [1, 2, 3, 4, 5]

    data = []
    for store in stores:
        for d in dates:
            # Simulation d'un chiffre de ventes avec saisonnalité hebdomadaire
            base_sales = 5000 + (store * 1000)
            day_effect = 1.5 if d.dayofweek in [4, 5] else 1.0  # Vendredi/Samedi
            noise = np.random.normal(0, 500)
            sales = max(0, int(base_sales * day_effect + noise))

            data.append(
                {
                    "Date": d,
                    "Store": store,
                    "Sales": sales,
                    "Open": 1 if d.dayofweek != 6 else 0,  # Fermé le dimanche
                    "Promo": np.random.choice([0, 1], p=[0.7, 0.3]),
                }
            )

    df_sales = pd.DataFrame(data)
    df_stores = pd.DataFrame(
        {
            "Store": stores,
            "StoreType": ["a", "b", "c", "a", "b"],
            "Assortment": ["a", "c", "a", "b", "c"],
            "CompetitionDistance": [1270, 570, 14130, 620, 4000],
        }
    )

    df_sales.to_csv("data/raw/train.csv", index=False)
    df_stores.to_csv("data/raw/store.csv", index=False)
    print("✅ Données de simulation générées dans data/raw/")


if __name__ == "__main__":
    generate_mock_dataset()