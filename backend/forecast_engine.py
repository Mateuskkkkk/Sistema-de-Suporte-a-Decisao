from __future__ import annotations

import math
import sqlite3
from functools import lru_cache
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from sklearn.feature_selection import f_regression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor


router = APIRouter(prefix="/api/previsao", tags=["previsao"])

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "banco_site.db"
CLIMATE_DIR = BASE_DIR / "data" / "indices_climaticos"

MONTHS = {
    "JAN": 1,
    "FEV": 2,
    "MAR": 3,
    "ABR": 4,
    "MAI": 5,
    "JUN": 6,
    "JUL": 7,
    "AGO": 8,
    "SET": 9,
    "OUT": 10,
    "NOV": 11,
    "DEZ": 12,
}

INDICATORS = {
    "nino34": {
        "label": "Niño 3.4",
        "description": "Anomalia de TSM do Pacífico equatorial central.",
    },
    "tna": {
        "label": "TNA",
        "description": "Anomalia de TSM do Atlântico Tropical Norte.",
    },
    "tsa": {
        "label": "TSA",
        "description": "Anomalia de TSM do Atlântico Tropical Sul.",
    },
    "dipolo": {
        "label": "Dipolo TNA−TSA",
        "description": "Gradiente meridional calculado como TNA menos TSA.",
    },
    "amm": {
        "label": "AMM",
        "description": "Modo Meridional do Atlântico.",
    },
    "amo": {
        "label": "AMO",
        "description": "Oscilação Multidecadal do Atlântico.",
    },
}

IMPORTANCE_METHODS = {
    "permutacao": "Permutação temporal do modelo",
    "select_k_best": "Select K Best (teste F)",
    "ganho_xgboost": "Ganho do XGBoost",
    "copeland": "Ranking unificado de Copeland",
}


class ForecastBaseRequest(BaseModel):
    reservatorio: str
    mes_inicial: int = 1
    ano_inicial: int = 1911
    mes_final: int = 12
    ano_final: int = 2021
    modelo: str = "knn"
    k: int = 5
    lags: int = 12
    horizonte: int = 3
    lag_climatico: int = 3
    teste_meses: int = 120


class ImportanceRequest(ForecastBaseRequest):
    metodo_importancia: str = "permutacao"


class CustomForecastRequest(ForecastBaseRequest):
    indicadores: List[str] = Field(default_factory=list)


def validate_request(request: ForecastBaseRequest) -> None:
    if request.modelo not in {"knn", "xgboost"}:
        raise HTTPException(status_code=400, detail="Modelo deve ser KNN ou XGBoost.")
    if not 1 <= int(request.horizonte) <= 12:
        raise HTTPException(status_code=400, detail="Horizonte deve estar entre 1 e 12 meses.")
    if not 1 <= int(request.lags) <= 24:
        raise HTTPException(status_code=400, detail="Memória de afluência deve estar entre 1 e 24 meses.")
    if not 1 <= int(request.lag_climatico) <= 12:
        raise HTTPException(status_code=400, detail="Lag climático deve estar entre 1 e 12 meses.")
    if not 1 <= int(request.k) <= 50:
        raise HTTPException(status_code=400, detail="K deve estar entre 1 e 50.")
    if int(request.teste_meses) < 12:
        raise HTTPException(status_code=400, detail="Use pelo menos 12 meses de validação.")


def load_flow(request: ForecastBaseRequest) -> pd.DataFrame:
    if not DB_PATH.exists():
        raise HTTPException(status_code=500, detail="Base de dados não encontrada.")
    with sqlite3.connect(DB_PATH) as connection:
        rows = connection.execute(
            """
            SELECT Ano, "Mês", "Vazão (m³/s)"
            FROM vazoes
            WHERE nome_reservatorio = ? OR nome_reservatorio LIKE ?
            ORDER BY Ano
            """,
            (request.reservatorio, f"%{request.reservatorio}%"),
        ).fetchall()
    if not rows:
        raise HTTPException(
            status_code=404,
            detail=f"Série de afluências não encontrada para {request.reservatorio}.",
        )

    records = []
    for year, month, value in rows:
        month_number = MONTHS.get(str(month).strip().upper()[:3])
        if month_number is None or value is None:
            continue
        records.append(
            {
                "data": pd.Timestamp(int(year), month_number, 1),
                "afluencia": float(value),
            }
        )
    frame = (
        pd.DataFrame(records)
        .groupby("data", as_index=False)["afluencia"]
        .mean()
        .sort_values("data")
    )
    start = pd.Timestamp(int(request.ano_inicial), int(request.mes_inicial), 1)
    end = pd.Timestamp(int(request.ano_final), int(request.mes_final), 1)
    frame = frame[frame["data"].between(start, end)].reset_index(drop=True)
    if frame.empty:
        raise HTTPException(status_code=404, detail="Não há afluências no período selecionado.")
    return frame


def load_noaa_csv(filename: str, column: str) -> pd.DataFrame:
    path = CLIMATE_DIR / filename
    if not path.exists():
        raise HTTPException(status_code=500, detail=f"Índice climático ausente: {filename}")
    raw = pd.read_csv(path)
    result = pd.DataFrame(
        {
            "data": pd.to_datetime(raw.iloc[:, 0], errors="coerce"),
            column: pd.to_numeric(raw.iloc[:, 1], errors="coerce"),
        }
    )
    result.loc[result[column] <= -90, column] = np.nan
    return result.dropna(subset=["data"]).drop_duplicates("data", keep="last")


def load_amo() -> pd.DataFrame:
    path = CLIMATE_DIR / "amo.data"
    rows = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        lines = handle.readlines()
    for line in lines[1:]:
        tokens = line.split()
        if len(tokens) != 13:
            continue
        try:
            year = int(tokens[0])
            values = [float(value) for value in tokens[1:]]
        except ValueError:
            continue
        for month, value in enumerate(values, start=1):
            rows.append(
                {
                    "data": pd.Timestamp(year, month, 1),
                    "amo": np.nan if value <= -90 else value,
                }
            )
    if not rows:
        raise HTTPException(status_code=500, detail="Não foi possível ler o índice AMO.")
    return pd.DataFrame(rows)


@lru_cache(maxsize=1)
def load_climate() -> pd.DataFrame:
    frames = [
        load_noaa_csv("nino34.csv", "nino34"),
        load_noaa_csv("tna.csv", "tna"),
        load_noaa_csv("tsa.csv", "tsa"),
        load_noaa_csv("amm.csv", "amm"),
        load_amo(),
    ]
    climate = frames[0]
    for frame in frames[1:]:
        climate = climate.merge(frame, on="data", how="outer")
    climate["dipolo"] = climate["tna"] - climate["tsa"]
    return climate.sort_values("data").reset_index(drop=True)


def build_dataset(
    flow: pd.DataFrame,
    horizon: int,
    flow_lags: int,
    climate_lag: int,
    indicators: list[str],
) -> tuple[pd.DataFrame, list[str], dict[str, str]]:
    frame = flow.merge(load_climate(), on="data", how="left").sort_values("data")
    features = []
    for offset in range(flow_lags):
        shift = horizon + offset
        name = f"afluencia_lag_{shift}"
        frame[name] = frame["afluencia"].shift(shift)
        features.append(name)

    month = frame["data"].dt.month.astype(float)
    frame["mes_seno"] = np.sin(2 * np.pi * month / 12.0)
    frame["mes_cosseno"] = np.cos(2 * np.pi * month / 12.0)
    features.extend(["mes_seno", "mes_cosseno"])

    effective_lag = max(int(horizon), int(climate_lag))
    climate_features = {}
    for indicator in indicators:
        name = f"{indicator}_lag_{effective_lag}"
        frame[name] = frame[indicator].shift(effective_lag)
        climate_features[indicator] = name
        features.append(name)

    frame = frame[["data", "afluencia", *features]].replace([np.inf, -np.inf], np.nan)
    return frame.dropna().reset_index(drop=True), features, climate_features


def temporal_split(frame: pd.DataFrame, test_months: int):
    available = len(frame)
    validation_size = min(int(test_months), max(12, available // 3))
    split = available - validation_size
    if split < 120:
        raise HTTPException(
            status_code=400,
            detail="A série efetiva precisa ter pelo menos 132 meses após as defasagens.",
        )
    return frame.iloc[:split].copy(), frame.iloc[split:].copy()


def make_model(model_name: str, k: int, samples: int):
    if model_name == "knn":
        neighbors = max(1, min(int(k), samples))
        return Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    KNeighborsRegressor(
                        n_neighbors=neighbors,
                        weights="distance",
                    ),
                ),
            ]
        )
    return XGBRegressor(
        n_estimators=220,
        max_depth=3,
        learning_rate=0.04,
        subsample=0.85,
        colsample_bytree=0.9,
        reg_lambda=2.0,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=1,
        tree_method="hist",
        importance_type="gain",
    )


def calculate_metrics(observed, predicted) -> dict:
    observed = np.asarray(observed, dtype=float)
    predicted = np.maximum(0.0, np.asarray(predicted, dtype=float))
    errors = predicted - observed
    denominator = float(np.sum((observed - observed.mean()) ** 2))
    nse = 1.0 - float(np.sum(errors**2)) / denominator if denominator > 0 else None
    correlation = (
        float(np.corrcoef(observed, predicted)[0, 1])
        if np.std(observed) > 0 and np.std(predicted) > 0
        else None
    )
    return {
        "mae_m3s": round(float(mean_absolute_error(observed, predicted)), 6),
        "rmse_m3s": round(float(math.sqrt(mean_squared_error(observed, predicted))), 6),
        "nse": round(nse, 6) if nse is not None and math.isfinite(nse) else None,
        "correlacao": (
            round(correlation, 6)
            if correlation is not None and math.isfinite(correlation)
            else None
        ),
        "vies_m3s": round(float(errors.mean()), 6),
        "amostras": int(len(observed)),
    }


def permutation_scores(
    model_name: str,
    k: int,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    features: list[str],
    climate_features: dict[str, str],
) -> tuple[dict[str, float], dict[str, float], dict]:
    model = make_model(model_name, k, len(train))
    model.fit(train[features], train["afluencia"])
    baseline_prediction = np.maximum(0.0, model.predict(validation[features]))
    baseline_mse = float(mean_squared_error(validation["afluencia"], baseline_prediction))
    rng = np.random.default_rng(42)
    increases = {}
    percent_increases = {}
    for indicator, feature in climate_features.items():
        values = validation[feature].to_numpy()
        scores = []
        for _ in range(20):
            permuted = validation[features].copy()
            permuted[feature] = rng.permutation(values)
            prediction = np.maximum(0.0, model.predict(permuted))
            scores.append(
                float(mean_squared_error(validation["afluencia"], prediction))
                - baseline_mse
            )
        increase = max(0.0, float(np.mean(scores)))
        increases[indicator] = increase
        percent_increases[indicator] = (
            increase / max(baseline_mse, 1e-12) * 100.0
        )
    return increases, percent_increases, calculate_metrics(
        validation["afluencia"], baseline_prediction
    )


def select_k_best_scores(
    train: pd.DataFrame,
    climate_features: dict[str, str],
) -> dict[str, float]:
    columns = list(climate_features.values())
    scores, _ = f_regression(train[columns], train["afluencia"])
    return {
        indicator: max(0.0, float(score)) if math.isfinite(score) else 0.0
        for indicator, score in zip(climate_features, scores)
    }


def xgboost_gain_scores(
    train: pd.DataFrame,
    features: list[str],
    climate_features: dict[str, str],
) -> dict[str, float]:
    model = make_model("xgboost", 5, len(train))
    model.fit(train[features], train["afluencia"])
    gains = dict(zip(features, model.feature_importances_))
    return {
        indicator: max(0.0, float(gains.get(feature, 0.0)))
        for indicator, feature in climate_features.items()
    }


def copeland_scores(rankings: list[list[str]]) -> dict[str, float]:
    indicators = list(INDICATORS)
    positions = [
        {indicator: position for position, indicator in enumerate(ranking)}
        for ranking in rankings
    ]
    scores = {indicator: 0.0 for indicator in indicators}
    for index, first in enumerate(indicators):
        for second in indicators[index + 1 :]:
            first_wins = sum(pos[first] < pos[second] for pos in positions)
            second_wins = sum(pos[second] < pos[first] for pos in positions)
            if first_wins > second_wins:
                scores[first] += 1
                scores[second] -= 1
            elif second_wins > first_wins:
                scores[second] += 1
                scores[first] -= 1
    return scores


def normalized_scores(scores: dict[str, float]) -> dict[str, float]:
    values = np.array(list(scores.values()), dtype=float)
    if len(values) and values.min() < 0:
        values = values - values.min()
    total = float(values.sum())
    if total <= 0:
        return {key: 0.0 for key in scores}
    return {
        key: float(value / total * 100.0)
        for key, value in zip(scores, values)
    }


def individual_r2(
    train: pd.DataFrame, climate_features: dict[str, str]
) -> dict[str, float]:
    result = {}
    target = train["afluencia"].to_numpy(dtype=float)
    for indicator, feature in climate_features.items():
        values = train[feature].to_numpy(dtype=float)
        correlation = (
            float(np.corrcoef(values, target)[0, 1])
            if np.std(values) > 0 and np.std(target) > 0
            else 0.0
        )
        result[indicator] = max(0.0, correlation**2 * 100.0)
    return result


def ranked_names(scores: dict[str, float]) -> list[str]:
    return sorted(scores, key=lambda indicator: (-scores[indicator], indicator))


@router.get("/indicadores")
def list_indicators():
    climate = load_climate()
    coverage = {}
    for indicator in INDICATORS:
        valid = climate.dropna(subset=[indicator])
        coverage[indicator] = {
            **INDICATORS[indicator],
            "inicio": valid["data"].min().strftime("%Y-%m"),
            "fim": valid["data"].max().strftime("%Y-%m"),
            "meses": int(len(valid)),
        }
    return {
        "indicadores": coverage,
        "metodos_importancia": IMPORTANCE_METHODS,
        "modelos": {"knn": "KNN", "xgboost": "XGBoost"},
    }


@router.post("/importancia")
def analyze_importance(request: ImportanceRequest):
    validate_request(request)
    if request.metodo_importancia not in IMPORTANCE_METHODS:
        raise HTTPException(status_code=400, detail="Método de importância desconhecido.")

    flow = load_flow(request)
    indicator_names = list(INDICATORS)
    frame, features, climate_features = build_dataset(
        flow,
        int(request.horizonte),
        int(request.lags),
        int(request.lag_climatico),
        indicator_names,
    )
    train, validation = temporal_split(frame, int(request.teste_meses))
    permutation, percent_increase, baseline_metrics = permutation_scores(
        request.modelo,
        int(request.k),
        train,
        validation,
        features,
        climate_features,
    )
    select_scores = select_k_best_scores(train, climate_features)
    gain_scores = xgboost_gain_scores(train, features, climate_features)

    if request.metodo_importancia == "permutacao":
        selected_scores = permutation
    elif request.metodo_importancia == "select_k_best":
        selected_scores = select_scores
    elif request.metodo_importancia == "ganho_xgboost":
        selected_scores = gain_scores
    else:
        selected_scores = copeland_scores(
            [
                ranked_names(permutation),
                ranked_names(select_scores),
                ranked_names(gain_scores),
            ]
        )

    relative = normalized_scores(selected_scores)
    explained = individual_r2(train, climate_features)
    ordered = ranked_names(selected_scores)
    results = []
    for rank, indicator in enumerate(ordered, start=1):
        results.append(
            {
                "id": indicator,
                **INDICATORS[indicator],
                "posicao": rank,
                "pontuacao": round(float(selected_scores[indicator]), 6),
                "contribuicao_relativa_percent": round(relative[indicator], 3),
                "aumento_mse_percent": round(percent_increase[indicator], 3),
                "variabilidade_individual_r2_percent": round(explained[indicator], 3),
                "recomendado": rank <= 3,
            }
        )

    return {
        "status": "sucesso",
        "reservatorio": request.reservatorio,
        "modelo": request.modelo,
        "metodo_importancia": request.metodo_importancia,
        "metodo_label": IMPORTANCE_METHODS[request.metodo_importancia],
        "parametros": {
            "lags": int(request.lags),
            "horizonte": int(request.horizonte),
            "lag_climatico_solicitado": int(request.lag_climatico),
            "lag_climatico_efetivo": max(
                int(request.horizonte), int(request.lag_climatico)
            ),
            "teste_meses": len(validation),
        },
        "periodo_efetivo": {
            "inicio": frame["data"].min().strftime("%Y-%m"),
            "fim": frame["data"].max().strftime("%Y-%m"),
            "meses": int(len(frame)),
        },
        "metricas_modelo_completo": baseline_metrics,
        "indicadores": results,
        "observacao": (
            "Contribuição relativa normaliza as importâncias para 100%. "
            "Variabilidade individual é o R² univariado e não deve ser somada "
            "quando os indicadores são correlacionados."
        ),
    }


def feature_for_future(
    flow: pd.DataFrame,
    target_date: pd.Timestamp,
    horizon: int,
    flow_lags: int,
    climate_lag: int,
    indicators: list[str],
    feature_names: list[str],
) -> pd.DataFrame:
    flow_lookup = flow.set_index("data")["afluencia"]
    climate_lookup = load_climate().set_index("data")
    values = {}
    for offset in range(flow_lags):
        source_date = target_date - pd.DateOffset(months=horizon + offset)
        if source_date not in flow_lookup.index:
            raise HTTPException(
                status_code=400,
                detail=f"Afluência histórica ausente em {source_date:%Y-%m}.",
            )
        values[f"afluencia_lag_{horizon + offset}"] = float(flow_lookup.loc[source_date])
    values["mes_seno"] = math.sin(2 * math.pi * target_date.month / 12.0)
    values["mes_cosseno"] = math.cos(2 * math.pi * target_date.month / 12.0)

    effective_lag = max(horizon, climate_lag)
    climate_date = target_date - pd.DateOffset(months=effective_lag)
    for indicator in indicators:
        if climate_date not in climate_lookup.index:
            raise HTTPException(
                status_code=400,
                detail=f"Índice climático ausente em {climate_date:%Y-%m}.",
            )
        value = climate_lookup.loc[climate_date, indicator]
        if pd.isna(value):
            raise HTTPException(
                status_code=400,
                detail=f"{INDICATORS[indicator]['label']} ausente em {climate_date:%Y-%m}.",
            )
        values[f"{indicator}_lag_{effective_lag}"] = float(value)
    return pd.DataFrame([[values[name] for name in feature_names]], columns=feature_names)


@router.post("/executar")
def run_custom_forecast(request: CustomForecastRequest):
    validate_request(request)
    indicators = list(dict.fromkeys(request.indicadores))
    invalid = [indicator for indicator in indicators if indicator not in INDICATORS]
    if invalid:
        raise HTTPException(
            status_code=400,
            detail="Indicadores desconhecidos: " + ", ".join(invalid),
        )

    flow = load_flow(request)
    horizon = int(request.horizonte)
    frame, features, _ = build_dataset(
        flow,
        horizon,
        int(request.lags),
        int(request.lag_climatico),
        indicators,
    )
    train, validation = temporal_split(frame, int(request.teste_meses))
    validation_model = make_model(request.modelo, int(request.k), len(train))
    validation_model.fit(train[features], train["afluencia"])
    validation_prediction = np.maximum(
        0.0, validation_model.predict(validation[features])
    )
    metrics = calculate_metrics(validation["afluencia"], validation_prediction)
    validation_rows = [
        {
            "data": date.strftime("%Y-%m"),
            "observado_m3s": round(float(observed), 6),
            "previsto_m3s": round(float(predicted), 6),
            "erro_m3s": round(float(predicted - observed), 6),
        }
        for date, observed, predicted in zip(
            validation["data"],
            validation["afluencia"],
            validation_prediction,
        )
    ]

    last_date = flow["data"].max()
    forecasts = []
    for step in range(1, horizon + 1):
        step_frame, step_features, _ = build_dataset(
            flow,
            step,
            int(request.lags),
            int(request.lag_climatico),
            indicators,
        )
        model = make_model(request.modelo, int(request.k), len(step_frame))
        model.fit(step_frame[step_features], step_frame["afluencia"])
        target_date = last_date + pd.DateOffset(months=step)
        target_features = feature_for_future(
            flow,
            target_date,
            step,
            int(request.lags),
            int(request.lag_climatico),
            indicators,
            step_features,
        )
        prediction = max(0.0, float(model.predict(target_features)[0]))
        forecasts.append(
            {
                "data": target_date.strftime("%Y-%m"),
                "horizonte": step,
                "vazao_m3s": round(prediction, 6),
                "afluencia_hm3_mes": round(prediction * 2.592, 6),
            }
        )

    historical = [
        {
            "data": row.data.strftime("%Y-%m"),
            "vazao_m3s": round(float(row.afluencia), 6),
            "afluencia_hm3_mes": round(float(row.afluencia) * 2.592, 6),
        }
        for row in flow.itertuples()
    ]
    return {
        "status": "sucesso",
        "metodo": (
            f"{request.modelo.upper()} direto por horizonte com histórico de "
            "afluências, sazonalidade e indicadores selecionados."
        ),
        "reservatorio": request.reservatorio,
        "modelo": request.modelo,
        "indicadores": [
            {"id": indicator, **INDICATORS[indicator]} for indicator in indicators
        ],
        "parametros": {
            "k": int(request.k) if request.modelo == "knn" else None,
            "lags": int(request.lags),
            "horizonte": horizon,
            "lag_climatico": int(request.lag_climatico),
            "teste_meses": len(validation),
        },
        "periodo": {
            "inicio": flow["data"].min().strftime("%Y-%m"),
            "fim": flow["data"].max().strftime("%Y-%m"),
            "meses": int(len(flow)),
            "inicio_efetivo_modelo": frame["data"].min().strftime("%Y-%m"),
        },
        "metricas": metrics,
        "historico": historical,
        "validacao": validation_rows,
        "previsao": forecasts,
    }
