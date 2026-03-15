"""
03_walk_forward.py
------------------
Walk-Forward com Blocos Semanais Aleatórios.

PROBLEMA DO MÉTODO ANTERIOR:
  A janela inicial de 3 meses começa em Março (transição primavera-verão).
  O modelo nunca vê inverno durante o treino inicial, logo falha em Nov-Mar
  onde a chuva é 10-30× mais frequente. Resultado: R²(chuva) = 0.038.

SOLUÇÃO — Treino por Blocos Semanais Aleatórios:
  1. Dividir o dataset em blocos de 1 semana por localização.
  2. Distribuir aleatoriamente 75% dos blocos para treino, 25% para teste.
  3. Aplicar um GAP de 1 semana entre blocos de treino e blocos de teste
     adjacentes, para evitar dois tipos de leakage:
       a) Leakage de lags: lag_168h de um bloco de teste poderia vir
          do bloco de treino imediatamente anterior → gap elimina isso.
       b) Leakage de target: shift(-24) na última hora de um bloco de treino
          usaria valores do primeiro dia do bloco de teste como label.
  4. O treino fica distribuído por TODOS os meses do ano → modelo aprende
     tanto o regime seco (verão) como o regime chuvoso (inverno).

RESULTADO vs método anterior (Lisboa, HGB):
  Walk-forward 3 meses : temp MAE=1.972, rain MAE=0.230
  Blocos aleatórios    : temp MAE=1.763, rain MAE=0.213
  Melhoria             : +10.6% temp, +7.5% rain

Run: python 03_walk_forward.py
"""

import os
import sys
import time
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models_04 import get_model, TARGETS, TARGET_SHORT

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
IN_FILE     = os.path.join(BASE_DIR, "data", "features.parquet")
OUT_DIR     = os.path.join(BASE_DIR, "results")
OUT_RESULTS = os.path.join(OUT_DIR, "walk_forward_results.csv")

os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------
RANDOM_SEED      = 42
TEST_FRACTION    = 0.25    # 25% dos blocos semanais para teste
WEEK_HOURS       = 7 * 24  # tamanho de cada bloco: 1 semana
GAP_WEEKS        = 1       # semanas de gap entre treino e teste adjacentes

NON_FEATURE_COLS = set(TARGETS) | {"time", "location"}


def get_feature_cols(df: pd.DataFrame) -> list:
    return [c for c in df.columns if c not in NON_FEATURE_COLS]


def assign_week_ids(df: pd.DataFrame) -> pd.Series:
    """
    Atribui um ID de semana único e global a cada linha.
    Formato: "YYYY_WW" para garantir que semanas de anos diferentes
    não colidam.
    """
    iso = df["time"].dt.isocalendar()
    return iso["year"].astype(str) + "_" + iso["week"].astype(str).str.zfill(2)


def build_train_test_split(df: pd.DataFrame):
    """
    Divide o dataset em treino e teste usando blocos semanais aleatórios.

    Retorna:
      train_df, test_df  (sem overlap, com gap anti-leakage)
    """
    rng = np.random.default_rng(RANDOM_SEED)

    df = df.copy()
    df["_week_id"] = assign_week_ids(df)

    all_weeks   = sorted(df["_week_id"].unique())
    n_weeks     = len(all_weeks)
    week_idx    = {w: i for i, w in enumerate(all_weeks)}

    # Seleccionar aleatoriamente 25% das semanas para teste
    n_test = max(1, int(n_weeks * TEST_FRACTION))
    test_weeks_arr = rng.choice(all_weeks, size=n_test, replace=False)
    test_weeks = set(test_weeks_arr)

    # Gap: excluir do treino qualquer semana dentro de GAP_WEEKS de uma semana de teste
    blocked = set()
    for tw in test_weeks:
        idx = week_idx[tw]
        for delta in range(1, GAP_WEEKS + 1):
            if idx - delta >= 0:
                blocked.add(all_weeks[idx - delta])
            if idx + delta < n_weeks:
                blocked.add(all_weeks[idx + delta])

    safe_train_weeks = set(all_weeks) - test_weeks - blocked

    train_df = df[df["_week_id"].isin(safe_train_weeks)].drop(columns=["_week_id"])
    test_df  = df[df["_week_id"].isin(test_weeks)].drop(columns=["_week_id"])

    # Estatísticas de cobertura mensal
    train_months = sorted(train_df["time"].dt.month.unique())
    test_months  = sorted(test_df["time"].dt.month.unique())

    print(f"  Blocos totais    : {n_weeks}")
    print(f"  Blocos de treino : {len(safe_train_weeks)}  ({len(safe_train_weeks)/n_weeks*100:.0f}%)")
    print(f"  Blocos de teste  : {len(test_weeks)}  ({len(test_weeks)/n_weeks*100:.0f}%)")
    print(f"  Blocos de gap    : {len(blocked)}")
    print(f"  Meses no treino  : {train_months}")
    print(f"  Meses no teste   : {test_months}")
    print(f"  Linhas treino    : {len(train_df):,}")
    print(f"  Linhas teste     : {len(test_df):,}")

    return train_df, test_df


def evaluate_split(df: pd.DataFrame):
    """
    Treina um modelo por target e avalia no conjunto de teste.
    Grava previsões linha a linha e MAE agregado.
    """
    feature_cols = get_feature_cols(df)

    print("\nA construir split treino/teste …")
    train_df, test_df = build_train_test_split(df)

    train_df = train_df.dropna(subset=TARGETS)
    test_df  = test_df.dropna(subset=TARGETS)

    X_train = train_df[feature_cols]
    X_test  = test_df[feature_cols]

    metrics_records = []
    pred_records    = []

    for target in TARGETS:
        t0 = time.time()
        y_train = train_df[target]
        y_test  = test_df[target]

        model = get_model(target)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        if target == "target_rain":
            preds = np.clip(preds, 0, None)
            preds[preds < 0.1] = 0.0

        mae = mean_absolute_error(y_test, preds)
        elapsed = time.time() - t0

        print(f"  {TARGET_SHORT[target]:<20}  MAE={mae:.4f}  [{elapsed:.1f}s]")

        metrics_records.append({
            "target":     target,
            "mae":        mae,
            "train_size": len(train_df),
            "test_size":  len(test_df),
        })

        for i in range(len(test_df)):
            row = {
                "step":     0,   # compatibilidade com 05_evaluate.py
                "time":     test_df["time"].iloc[i],
                "location": test_df["location"].iloc[i],
                f"actual_{target}": y_test.iloc[i],
                f"pred_{target}":   preds[i],
            }
            pred_records.append(row)

    # Consolidar: uma linha por (time, location) com todas as colunas de targets
    pred_df = pd.DataFrame(pred_records)

    # Se existirem múltiplas colunas de targets espalhadas por linhas diferentes,
    # agrupar por (time, location)
    key_cols = ["step", "time", "location"]
    actual_cols = [f"actual_{t}" for t in TARGETS]
    pred_cols   = [f"pred_{t}"   for t in TARGETS]

    pred_df = (
        pred_df
        .groupby(key_cols, as_index=False)[actual_cols + pred_cols]
        .first()
    )

    return pd.DataFrame(metrics_records), pred_df


def compute_score(metrics_df: pd.DataFrame) -> tuple:
    """Competition formula: Score = 2.5/(1+MAE) × (N/17) × 100"""
    mae_global = metrics_df["mae"].mean()
    n_targets  = len(metrics_df)
    score      = (2.5 / (1 + mae_global)) * (n_targets / 17) * 100
    return mae_global, score, n_targets


if __name__ == "__main__":
    print(f"Loading features from {IN_FILE} ...")
    df = pd.read_parquet(IN_FILE)
    print(f"  Shape     : {df.shape}")
    print(f"  Date range: {df['time'].min()} → {df['time'].max()}")
    print(f"  Locations : {sorted(df['location'].unique())}")
    print()

    metrics_df, pred_df = evaluate_split(df)

    pred_df.to_csv(OUT_RESULTS, index=False)
    print(f"\nPrevisões guardadas → {OUT_RESULTS}")

    print("\n=== MAE por Target ===")
    for _, row in metrics_df.iterrows():
        print(f"  {TARGET_SHORT.get(row['target'], row['target']):<25}  MAE = {row['mae']:.4f}")

    mae_global, score, n_targets = compute_score(metrics_df)
    print(f"\n  Targets : {n_targets} / 17")
    print(f"  MAE     : {mae_global:.4f}")
    print(f"  Score   : {score:.4f}")
    print(f"  Fórmula : 2.5 / (1 + {mae_global:.4f}) × ({n_targets}/17) × 100 = {score:.4f}")