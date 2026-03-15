"""
02_feature_engineering.py
--------------------------
Lê data/processed.parquet e constrói features para previsão horária +24h.

Melhorias vs versão anterior (baseadas em análise de importância com Random Forest):
  - Novos lags: 96h e 120h para temperatura (importância demonstrada)
  - Removidos: lags de temperatura 1h/2h/3h/6h (importância < 0.001, redundantes)
  - Adicionados: pressure_trend_3h/6h/12h/24h (features físicas de frentes)
  - Adicionado: rain_roll168h (tendência semanal — top feature para chuva)
  - Adicionados: lags de cloud_cover_mid/highh a 12h/24h (precursor de chuva)
  - location_id correctamente codificado por ordem alfabética

Run: python 02_feature_engineering.py
"""

import os
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IN_FILE  = os.path.join(BASE_DIR, "data", "processed.parquet")
OUT_FILE = os.path.join(BASE_DIR, "data", "features.parquet")

RAW_TO_TARGET = {
    "temperature_2m": "target_temperature_2m",
    "rain":           "target_rain",
}

# Lags de temperatura: apenas múltiplos de 24h — lags curtos são redundantes (r>0.99)
TEMP_LAG_SHIFTS     = [24, 48, 72, 96, 120, 168]
# Lags de chuva: manter todos — nenhum é negligenciável
RAIN_LAG_SHIFTS     = [1, 12, 24, 48, 72, 96, 120, 168]
# Lags de pressão
PRESSURE_LAG_SHIFTS = [1, 6, 24, 48, 72]
# Lags de vento, humidade, nuvens
AUX_LAG_SHIFTS      = [1, 6, 24, 48, 72]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Todas as transformações são aplicadas por grupo de localização
    para evitar que lags atravessem fronteiras entre cidades.
    """
    locations_sorted = sorted(df["location"].unique())
    loc_to_id = {loc: i for i, loc in enumerate(locations_sorted)}

    groups = []

    for loc, grp in df.groupby("location", sort=False):
        grp = grp.copy().sort_values("time").reset_index(drop=True)

        # 1. Targets
        for raw_col, tgt_col in RAW_TO_TARGET.items():
            grp[tgt_col] = grp[raw_col].shift(-24)

        # 2. Lags de temperatura (só múltiplos de 24h)
        for s in TEMP_LAG_SHIFTS:
            grp[f"temperature_2m_lag_{s}h"] = grp["temperature_2m"].shift(s)

        # 3. Lags de chuva
        for s in RAIN_LAG_SHIFTS:
            grp[f"rain_lag_{s}h"] = grp["rain"].shift(s)

        # 4. Lags de pressão
        for s in PRESSURE_LAG_SHIFTS:
            grp[f"pressure_msl_lag_{s}h"] = grp["pressure_msl"].shift(s)

        # 5. Lags de vento, humidade e nuvens
        for s in AUX_LAG_SHIFTS:
            grp[f"wind_speed_10m_lag_{s}h"]    = grp["wind_speed_10m"].shift(s)
            grp[f"relative_humidity_lag_{s}h"] = grp["relative_humidity_2m"].shift(s)
            grp[f"cloud_cover_lag_{s}h"]       = grp["cloud_cover"].shift(s)

        # Nuvens altas/médias: precursores de sistemas de chuva 12-24h antes
        for s in [12, 24]:
            grp[f"cloud_cover_mid_lag_{s}h"]   = grp["cloud_cover_mid"].shift(s)
            grp[f"cloud_cover_highh_lag_{s}h"] = grp["cloud_cover_highh"].shift(s)

        # 6. Rolling features (sobre dados ≤ t — sem data leakage)
        grp["temp_roll_6h"]      = grp["temperature_2m"].rolling(6,   min_periods=1).mean()
        grp["temp_roll_24h"]     = grp["temperature_2m"].rolling(24,  min_periods=1).mean()
        grp["temp_roll_72h"]     = grp["temperature_2m"].rolling(72,  min_periods=1).mean()
        grp["rain_roll_24h"]     = grp["rain"].rolling(24,  min_periods=1).sum()
        grp["rain_roll_72h"]     = grp["rain"].rolling(72,  min_periods=1).sum()
        grp["rain_roll_168h"]    = grp["rain"].rolling(168, min_periods=1).sum()   # NEW
        grp["pressure_roll_24h"] = grp["pressure_msl"].rolling(24, min_periods=1).mean()
        grp["humidity_roll_3h"]  = grp["relative_humidity_2m"].rolling(3, min_periods=1).mean()
        grp["cloud_roll_3h"]     = grp["cloud_cover"].rolling(3, min_periods=1).mean()

        # 7. Tendência de pressão (NOVAS)
        # Queda rápida de pressão = frente a aproximar-se = chuva provável
        grp["pressure_trend_3h"]  = grp["pressure_msl"] - grp["pressure_msl"].shift(3)
        grp["pressure_trend_6h"]  = grp["pressure_msl"] - grp["pressure_msl"].shift(6)
        grp["pressure_trend_12h"] = grp["pressure_msl"] - grp["pressure_msl"].shift(12)
        grp["pressure_trend_24h"] = grp["pressure_msl"] - grp["pressure_msl"].shift(24)

        # 8. Interações físicas
        grp["temp_dew_spread"] = grp["temperature_2m"]  - grp["dew_point_2m"]
        grp["pressure_diff"]   = grp["pressure_msl"]    - grp["surface_pressure"]
        grp["wind_shear"]      = grp["wind_speed_100m"] - grp["wind_speed_10m"]

        # 9. Location ID
        grp["location_id"] = loc_to_id[loc]

        groups.append(grp)

    result = pd.concat(groups, ignore_index=True)

    target_cols = list(RAW_TO_TARGET.values())
    before = len(result)
    result = result.dropna(subset=target_cols).reset_index(drop=True)
    after  = len(result)
    print(f"  Dropped {before - after:,} rows com NaN targets → {after:,} rows restantes")

    return result


if __name__ == "__main__":
    print("Loading processed data …")
    df = pd.read_parquet(IN_FILE)
    print(f"  Shape: {df.shape}")

    print("Building features …")
    features_df = build_features(df)

    print(f"\nFinal feature set shape: {features_df.shape}")
    print(f"Colunas ({len(features_df.columns)}):")
    for c in features_df.columns:
        print(f"  {c}")

    tmp_out_file = OUT_FILE + ".tmp"
    if os.path.exists(tmp_out_file):
        os.remove(tmp_out_file)

    features_df.to_parquet(tmp_out_file, index=False)
    os.replace(tmp_out_file, OUT_FILE)
    print(f"\nSaved → {OUT_FILE}")