"""
===============================================================================
🌸 App Streamlit: Predicción de Floración de Cerezos
===============================================================================

Dashboard interactivo que permite:
  1. Explorar datos históricos de floración (filtros por país, site, cultivar, año)
  2. Visualizar mapa de progreso de floración 2026 en tiempo real
  3. Consultar predicciones detalladas basadas en tendencia temporal
  4. Consultar la metodología y variables clave del pipeline

Diseño:
- Usa el dataset_final.csv limpiado y enriquecido por obtain_data.py
- Predice floración 2026 aplicando tendencia lineal sobre T crudas (T2M_MIN/T2M_MAX)
  y recalculando variables derivadas (chill, GDD, frost) desde las T ajustadas
- Muestra mapa interactivo con % de progreso hacia floración
- Opcionalmente carga un modelo ML serializado para predicciones avanzadas

Ejecución:
  streamlit run app.py

Autor: Bootcamp Ciencia de Datos 2026
===============================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime, timedelta
from pathlib import Path

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import requests
from sklearn.ensemble import RandomForestRegressor

# ============================================================================
# CONFIGURACIÓN GENERAL
# ============================================================================

st.set_page_config(
    page_title="🌸 Floración Cerezos",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded"
)

YEAR_PRED = 2026
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"

# Colores del tema
COLORS = {
    "pink_light": "#FFB7C5",
    "pink": "#FF69B4",
    "pink_dark": "#FF1493",
    "magenta": "#C71585",
    "blue": "#3498DB",
    "orange": "#F39C12",
    "red": "#E74C3C",
    "green": "#27AE60",
    "teal": "#1ABC9C",
    "purple": "#9B59B6",
}


# ============================================================================
# MODELO DINÁMICO DE FRÍO (replicado del pipeline para predicciones en vivo)
# ============================================================================

class DynamicChillModel:
    """
    Modelo Dinámico de acumulación de frío (Fishman et al., 1987).

    Calcula porciones de frío (chill portions) a partir de temperaturas
    horarias simuladas desde datos diarios Tmax/Tmin mediante un perfil
    sinusoidal (Linvill, 1990).
    """
    E0 = 4153.5
    E1 = 12888.8
    A0 = 1.395e5
    A1 = 2.567e18
    SLP = 1.28
    TETMLT = 277.0

    def __init__(self):
        self.inter_s = 0.0
        self.chill_portions = 0.0

    def reset(self):
        self.inter_s = 0.0
        self.chill_portions = 0.0

    def _process_hour(self, temp_c: float):
        TK = temp_c + 273.15
        TK = max(TK, 220.0)
        TK = min(TK, 330.0)

        try:
            xs = self.A0 / self.A1 * np.exp((self.E1 - self.E0) / TK)
        except (OverflowError, FloatingPointError):
            xs = 0.0

        try:
            ak1 = self.A1 * np.exp(-self.E1 / TK)
        except (OverflowError, FloatingPointError):
            ak1 = 0.0

        inter_e = xs - (xs - self.inter_s) * np.exp(-ak1)

        try:
            ftmprt = self.SLP * self.TETMLT * (TK - self.TETMLT) / TK
            st_val = np.exp(ftmprt)
            xi = st_val / (1.0 + st_val)
        except (OverflowError, FloatingPointError):
            xi = 1.0 if TK > self.TETMLT else 0.0

        if inter_e >= 1.0:
            delta_cp = xi * inter_e
            self.chill_portions += delta_cp
            self.inter_s = inter_e - delta_cp
        else:
            self.inter_s = inter_e

    def process_day(self, tmax: float, tmin: float) -> float:
        if pd.isna(tmax) or pd.isna(tmin) or tmax < tmin:
            return self.chill_portions
        t_mean = (tmax + tmin) / 2.0
        amplitude = (tmax - tmin) / 2.0
        for hour in range(24):
            temp_c = t_mean - amplitude * np.cos(2.0 * np.pi * hour / 24.0)
            self._process_hour(temp_c)
        return self.chill_portions

    def accumulate_series(self, tmax_series, tmin_series):
        self.reset()
        daily_cp = []
        for tmax, tmin in zip(tmax_series, tmin_series):
            self.process_day(tmax, tmin)
            daily_cp.append(self.chill_portions)
        return self.chill_portions, daily_cp


# ============================================================================
# FUNCIONES DE DATOS CLIMÁTICOS EN TIEMPO REAL (Open-Meteo)
# ============================================================================

@st.cache_data(ttl=3600, show_spinner=False)
def get_weather_daily(lat: float, lon: float, start_date: str, end_date: str):
    """
    Descarga datos diarios (Tmax, Tmin, precipitación, radiación) de
    Open-Meteo Archive API para un periodo histórico.
    """
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat, "longitude": lon,
        "start_date": start_date, "end_date": end_date,
        "daily": "temperature_2m_max,temperature_2m_min,temperature_2m_mean,"
                 "precipitation_sum,shortwave_radiation_sum",
        "timezone": "auto"
    }
    try:
        resp = requests.get(url, params=params, timeout=60)
        data = resp.json()
        if "daily" in data:
            df = pd.DataFrame({
                "date": pd.to_datetime(data["daily"]["time"]),
                "tmax": data["daily"]["temperature_2m_max"],
                "tmin": data["daily"]["temperature_2m_min"],
                "tmean": data["daily"]["temperature_2m_mean"],
                "precip": data["daily"]["precipitation_sum"],
                "radiation": data["daily"]["shortwave_radiation_sum"],
            })
            return df
    except Exception:
        pass
    return None


@st.cache_data(ttl=3600, show_spinner=False)
def get_forecast_daily(lat: float, lon: float):
    """
    Descarga forecast diario (16 días) de Open-Meteo Forecast API.
    """
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat, "longitude": lon,
        "daily": "temperature_2m_max,temperature_2m_min,temperature_2m_mean,"
                 "precipitation_sum,shortwave_radiation_sum",
        "forecast_days": 16, "timezone": "auto"
    }
    try:
        resp = requests.get(url, params=params, timeout=30)
        data = resp.json()
        if "daily" in data:
            df = pd.DataFrame({
                "date": pd.to_datetime(data["daily"]["time"]),
                "tmax": data["daily"]["temperature_2m_max"],
                "tmin": data["daily"]["temperature_2m_min"],
                "tmean": data["daily"]["temperature_2m_mean"],
                "precip": data["daily"]["precipitation_sum"],
                "radiation": data["daily"]["shortwave_radiation_sum"],
            })
            return df
    except Exception:
        pass
    return None


def compute_realtime_features(lat: float, lon: float, altitude: float,
                              year: int = YEAR_PRED) -> dict:
    """
    Calcula las features necesarias para el modelo a partir de datos
    climáticos en tiempo real (Open-Meteo).

    Retorna un diccionario con todas las features + datos intermedios.
    """
    today = datetime.now()

    # Descargar datos históricos: Oct año anterior hasta hace 5 días
    hist_start = f"{year-1}-10-01"
    hist_end = (today - timedelta(days=5)).strftime("%Y-%m-%d")
    df_hist = get_weather_daily(lat, lon, hist_start, hist_end)

    # Descargar forecast (16 días)
    df_forecast = get_forecast_daily(lat, lon)

    dfs = [df for df in [df_hist, df_forecast] if df is not None]
    if not dfs:
        return None

    df = pd.concat(dfs).drop_duplicates(subset="date").sort_values("date")
    df = df.dropna(subset=["tmax", "tmin"])

    if len(df) < 10:
        return None

    # ----- Dynamic Chill Portions -----
    model_chill = DynamicChillModel()
    total_cp, daily_cp = model_chill.accumulate_series(
        df["tmax"].values, df["tmin"].values
    )

    # ----- GDD desde 1 de enero -----
    df_gdd = df[df["date"] >= datetime(year, 1, 1)].copy()
    df_gdd["gdd"] = df_gdd.apply(
        lambda r: max(0, (r["tmax"] + r["tmin"]) / 2 - 5), axis=1
    )
    gdd_total = df_gdd["gdd"].sum()

    # ----- Frost days desde 1 enero -----
    frost_total = int((df_gdd["tmin"] < 0).sum())

    # ----- Temperaturas 30 días previos -----
    last_date = df["date"].max()
    df_30d = df[df["date"] > last_date - timedelta(days=30)]
    temp_media_30d = df_30d["tmean"].mean() if len(df_30d) > 0 else df["tmean"].mean()
    temp_max_30d = df_30d["tmax"].mean() if len(df_30d) > 0 else df["tmax"].mean()
    temp_min_30d = df_30d["tmin"].mean() if len(df_30d) > 0 else df["tmin"].mean()

    # ----- Precipitación total enero→hoy -----
    precip_total = df_gdd["precip"].sum() if "precip" in df_gdd.columns else 0

    # ----- Radiación media enero→hoy -----
    rad_media = df_gdd["radiation"].mean() if "radiation" in df_gdd.columns else 0
    # Convertir de Wh/m² a MJ/m²/día (Open-Meteo da Wh/m², NASA da MJ/m²/día)
    if rad_media and rad_media > 50:
        rad_media = rad_media * 0.0036

    return {
        "Latitude": lat,
        "Longitude": lon,
        "Altitude": altitude,
        "Year": year,
        "dynamic_chill_total": total_cp,
        "gdd_total": gdd_total,
        "frost_days_total": frost_total,
        "temp_media_30d": temp_media_30d,
        "temp_max_30d": temp_max_30d,
        "temp_min_30d": temp_min_30d,
        "precip_total": precip_total,
        "rad_media": rad_media,
        "last_date": last_date,
        "df_daily": df,
        "daily_cp": daily_cp,
    }


# ============================================================================
# UTILIDADES
# ============================================================================

def doy_to_date(doy, year=YEAR_PRED) -> str:
    """Convierte día del año a fecha legible."""
    if pd.isna(doy) or doy < 1:
        return "N/A"
    try:
        date = datetime(year, 1, 1) + timedelta(days=int(doy) - 1)
        return date.strftime("%d %B")
    except (ValueError, OverflowError):
        return "N/A"


# ============================================================================
# CARGA DE DATOS Y MODELO
# ============================================================================

@st.cache_data
def load_historical_data():
    """Carga dataset histórico con clima."""
    csv_path = DATA_DIR / "dataset_final.csv"
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().replace("\xa0", " ").strip() for c in df.columns]
    return df


@st.cache_resource
def load_trained_model():
    """
    Carga el modelo entrenado (pickle).

    El modelo se entrena en prediccion_floracion.ipynb y se serializa como
    'best_model.pkl'. Busca en data/ y en el directorio raíz.

    Si la versión de scikit-learn no coincide con la usada al serializar,
    devuelve None y se usará el modelo fallback auto-entrenado.
    """
    for path in [DATA_DIR / "best_model.pkl", BASE_DIR / "best_model.pkl"]:
        if path.exists():
            try:
                with open(path, "rb") as f:
                    bundle = pickle.load(f)
                return bundle
            except (AttributeError, ModuleNotFoundError, ImportError) as e:
                st.warning(
                    f"⚠️ No se pudo cargar `{path.name}`: incompatibilidad de "
                    f"versiones de scikit-learn ({e.__class__.__name__}). "
                    f"Se usará modelo auto-entrenado. Para solucionarlo, "
                    f"re-ejecuta el notebook `prediccion_floracion.ipynb` para "
                    f"re-serializar el modelo con la versión actual de sklearn."
                )
                return None
    return None


# ============================================================================
# PREDICCIÓN 2026: VARIABLES CLIMÁTICAS EXTRAPOLADAS → MODELO ML
# ============================================================================

# Variables climáticas derivadas (se recalculan desde T crudas ajustadas)
CLIMATE_FEATURES = [
    "dynamic_chill_total", "gdd_total", "frost_days_total",
    "temp_media_30d", "temp_max_30d", "temp_min_30d",
    "precip_total", "rad_media",
]

# Variables crudas de NASA POWER sobre las que se aplica la tendencia lineal
# El enfoque correcto: extrapolar T2M_MIN/T2M_MAX diarias y luego recalcular
# las derivadas no-lineales (chill, frost, GDD) desde las T ajustadas.
RAW_TREND_VARS = ["T2M_MIN", "T2M_MAX", "PRECTOTCORR", "ALLSKY_SFC_SW_DWN"]


@st.cache_data
def _load_daily_climate():
    """
    Carga los datos climáticos diarios de NASA POWER (clima_diario_nasa.csv).
    Retorna DataFrame indexado por Site + date, o None si no existe.
    """
    path = DATA_DIR / "clima_diario_nasa.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, parse_dates=["date"])
    return df


def _extrapolate_variable(series_years, series_values, year_pred):
    """
    Extrapola una variable al año year_pred usando regresión lineal.
    Si hay < 3 años con datos, devuelve la media.

    Retorna: (valor_extrapolado, pendiente)
    """
    mask = ~(np.isnan(series_values))
    years_clean = series_years[mask]
    vals_clean = series_values[mask]

    if len(vals_clean) < 3:
        return float(np.nanmean(vals_clean)) if len(vals_clean) > 0 else 0.0, 0.0

    try:
        coeffs = np.polyfit(years_clean, vals_clean, 1)
        predicted = np.polyval(coeffs, year_pred)
        return float(predicted), float(coeffs[0])
    except (np.linalg.LinAlgError, ValueError):
        return float(np.nanmean(vals_clean)), 0.0


def _compute_monthly_temp_trends(clima_site, year_pred):
    """
    Calcula tendencia lineal mensual de T2M_MIN, T2M_MAX, precipitación y
    radiación a partir de los datos diarios de un sitio.

    Para cada mes, ajusta: var_mensual(year) = a + b·year
    y devuelve el delta = b × (year_pred - year_medio_histórico)
    aplicable a la climatología media.

    Retorna
    -------
    dict : {mes: {"T2M_MIN": delta, "T2M_MAX": delta, ...}}
           con meses 1-12.
    dict : {variable: pendiente_media_anual} (para reportar).
    """
    trends_by_month = {}
    overall_slopes = {v: 0.0 for v in RAW_TREND_VARS}
    n_months = 0

    for month in range(1, 13):
        monthly = clima_site[clima_site["month"] == month].copy()
        if len(monthly) == 0:
            trends_by_month[month] = {v: 0.0 for v in RAW_TREND_VARS}
            continue

        # Promediar por año para obtener serie anual de cada variable
        yearly_means = monthly.groupby("year")[RAW_TREND_VARS].mean()
        years = yearly_means.index.values.astype(float)

        deltas = {}
        for var in RAW_TREND_VARS:
            vals = yearly_means[var].values.astype(float)
            mask = ~np.isnan(vals)
            yrs_clean = years[mask]
            vals_clean = vals[mask]

            if len(vals_clean) < 3:
                deltas[var] = 0.0
                continue

            try:
                coeffs = np.polyfit(yrs_clean, vals_clean, 1)
                slope = float(coeffs[0])  # °C/año o mm/año
                mean_year = float(np.mean(yrs_clean))
                # Delta: diferencia entre el año predicho y el año medio histórico
                deltas[var] = slope * (year_pred - mean_year)
                overall_slopes[var] += slope
                n_months += 1
            except (np.linalg.LinAlgError, ValueError):
                deltas[var] = 0.0

        trends_by_month[month] = deltas

    # Promedio anual de pendientes (para reporte)
    if n_months > 0:
        for v in RAW_TREND_VARS:
            overall_slopes[v] /= (n_months / len(RAW_TREND_VARS))

    return trends_by_month, overall_slopes


def _build_synthetic_year(clima_site, trends_by_month, year_pred,
                          mean_bloom_doy):
    """
    Construye un "año sintético" para year_pred:
      1. Toma la climatología media diaria (promedio por day_of_year)
      2. Aplica el delta de tendencia mensual a T2M_MIN y T2M_MAX
      3. Recalcula todas las variables derivadas desde las T ajustadas

    Periodo:
      - Chill: 1 Oct año anterior → fecha de floración
      - GDD, frost, precip, rad: 1 Ene → fecha de floración
      - Temp 30d: 30 días previos a floración media

    Retorna
    -------
    dict : Variables climáticas recalculadas.
    """
    bloom_doy = int(round(mean_bloom_doy))
    bloom_doy = max(30, min(180, bloom_doy))  # rango razonable

    # --- Climatología media diaria (promedio por day_of_year) ---
    clim_mean = clima_site.groupby("day_of_year").agg({
        "T2M_MAX": "mean",
        "T2M_MIN": "mean",
        "T2M": "mean",
        "PRECTOTCORR": "mean",
        "ALLSKY_SFC_SW_DWN": "mean",
    }).reset_index()

    # Necesitamos el mes para aplicar el delta mensual
    # Mapear day_of_year → month usando un año de referencia (no bisiesto)
    ref_dates = pd.date_range("2023-01-01", "2023-12-31", freq="D")
    doy_to_month = {d.timetuple().tm_yday: d.month for d in ref_dates}
    clim_mean["month"] = clim_mean["day_of_year"].map(doy_to_month)
    # DOY 366 → diciembre
    clim_mean["month"] = clim_mean["month"].fillna(12).astype(int)

    # --- Aplicar deltas de tendencia mensual a las T crudas ---
    clim_mean["T2M_MAX_adj"] = clim_mean.apply(
        lambda r: r["T2M_MAX"] + trends_by_month.get(r["month"], {}).get("T2M_MAX", 0.0),
        axis=1
    )
    clim_mean["T2M_MIN_adj"] = clim_mean.apply(
        lambda r: r["T2M_MIN"] + trends_by_month.get(r["month"], {}).get("T2M_MIN", 0.0),
        axis=1
    )
    # Asegurar T2M_MAX >= T2M_MIN
    clim_mean["T2M_MAX_adj"] = np.maximum(
        clim_mean["T2M_MAX_adj"], clim_mean["T2M_MIN_adj"]
    )
    clim_mean["T2M_mean_adj"] = (
        clim_mean["T2M_MAX_adj"] + clim_mean["T2M_MIN_adj"]
    ) / 2.0

    # Ajustar precipitación y radiación también
    clim_mean["PRECIP_adj"] = clim_mean.apply(
        lambda r: max(0, r["PRECTOTCORR"] +
                      trends_by_month.get(r["month"], {}).get("PRECTOTCORR", 0.0)),
        axis=1
    )
    clim_mean["RAD_adj"] = clim_mean.apply(
        lambda r: max(0, r["ALLSKY_SFC_SW_DWN"] +
                      trends_by_month.get(r["month"], {}).get("ALLSKY_SFC_SW_DWN", 0.0)),
        axis=1
    )

    # --- Periodo de CHILL: DOY 274 (1 Oct) año anterior → bloom_doy ---
    # Necesitamos DOY 274..365 + 1..bloom_doy
    chill_start_doy = 274  # 1 de octubre
    chill_days = clim_mean[
        (clim_mean["day_of_year"] >= chill_start_doy) |
        (clim_mean["day_of_year"] <= bloom_doy)
    ].sort_values("day_of_year")
    # Reordenar: Oct-Dic primero, luego Ene-bloom
    part_oct_dec = chill_days[chill_days["day_of_year"] >= chill_start_doy]
    part_jan_bloom = chill_days[chill_days["day_of_year"] <= bloom_doy]
    chill_days = pd.concat([part_oct_dec, part_jan_bloom])

    # Dynamic Chill Portions desde T ajustadas
    chill_model = DynamicChillModel()
    if len(chill_days) > 0:
        total_cp, _ = chill_model.accumulate_series(
            chill_days["T2M_MAX_adj"].values,
            chill_days["T2M_MIN_adj"].values
        )
    else:
        total_cp = 0.0

    # --- Periodo Ene → floración ---
    jan_bloom = clim_mean[
        (clim_mean["day_of_year"] >= 1) &
        (clim_mean["day_of_year"] <= bloom_doy)
    ].sort_values("day_of_year")

    # GDD desde T ajustadas
    if len(jan_bloom) > 0:
        gdd_total = sum(
            max(0, (tmax + tmin) / 2.0 - 5.0)
            for tmax, tmin in zip(jan_bloom["T2M_MAX_adj"], jan_bloom["T2M_MIN_adj"])
        )
    else:
        gdd_total = 0.0

    # Frost days desde T2M_MIN ajustada
    if len(jan_bloom) > 0:
        frost_total = int((jan_bloom["T2M_MIN_adj"] < 0).sum())
    else:
        frost_total = 0

    # Temperaturas 30 días previos a floración
    d30_start = max(1, bloom_doy - 30)
    days_30d = clim_mean[
        (clim_mean["day_of_year"] > d30_start) &
        (clim_mean["day_of_year"] <= bloom_doy)
    ]
    if len(days_30d) > 0:
        temp_media_30d = float(days_30d["T2M_mean_adj"].mean())
        temp_max_30d = float(days_30d["T2M_MAX_adj"].mean())
        temp_min_30d = float(days_30d["T2M_MIN_adj"].mean())
    else:
        temp_media_30d = temp_max_30d = temp_min_30d = 0.0

    # Precipitación total (Ene → floración)
    precip_total = float(jan_bloom["PRECIP_adj"].sum()) if len(jan_bloom) > 0 else 0.0

    # Radiación media (Ene → floración)
    rad_media = float(jan_bloom["RAD_adj"].mean()) if len(jan_bloom) > 0 else 0.0

    return {
        "dynamic_chill_total": total_cp,
        "gdd_total": gdd_total,
        "frost_days_total": frost_total,
        "temp_media_30d": temp_media_30d,
        "temp_max_30d": temp_max_30d,
        "temp_min_30d": temp_min_30d,
        "precip_total": precip_total,
        "rad_media": rad_media,
    }


@st.cache_data
def compute_trend_predictions_2026(historical, _model_bundle=None, year_pred=YEAR_PRED, model_name=""):
    """
    Calcula predicciones de floración para 2026 en tres pasos:

    Paso 1 – Tendencia sobre temperaturas crudas:
      Para cada sitio, calcula la tendencia lineal mensual de T2M_MIN y T2M_MAX
      (datos diarios de NASA POWER). La tendencia se aplica sobre las variables
      crudas, NO sobre las derivadas, porque dynamic_chill, frost_days y GDD
      son funciones no lineales de la temperatura.

    Paso 2 – Año sintético 2026:
      Genera un año de climatología media diaria + delta de tendencia.
      Recalcula dynamic_chill (Modelo Dinámico), GDD, frost_days, y demás
      variables derivadas desde las temperaturas ajustadas.

    Paso 3 – Predicción con modelo ML:
      Alimenta las variables climáticas recalculadas + features geográficas
      al modelo Random Forest para predecir el DOY de floración.

    Fallback: si no hay datos diarios o modelo ML, usa tendencia lineal
    directa sobre el DOY observado.

    Referencia:
      Menzel et al. (2006), Global Change Biology.
      Fishman et al. (1987), J. Theor. Biol. (Dynamic Chill Model).
    """
    target = "Beginning.of.flowering"
    results = []

    # --- Cargar datos diarios de NASA POWER ---
    clima_daily = _load_daily_climate()
    clima_by_site = {}
    if clima_daily is not None:
        for site_name, grp in clima_daily.groupby("Site"):
            clima_by_site[site_name] = grp

    # Preparar target encoding global
    global_mean_doy = historical[target].mean()
    cultivar_target_map = historical.groupby("Cultivar")[target].mean().to_dict()
    site_target_map = historical.groupby("Site")[target].mean().to_dict()
    cultivar_freq_map = historical["Cultivar"].value_counts().to_dict()

    cultivar_enc_map = {}
    if "Cultivar_enc" in historical.columns:
        cultivar_enc_map = historical.groupby("Cultivar")["Cultivar_enc"].first().to_dict()

    # Modelo ML y sus features
    model = None
    model_features = []
    if _model_bundle is not None:
        model = _model_bundle.get("model")
        model_features = _model_bundle.get("features", [])

    # MAE del modelo para IC (si disponible, si no ~1.6 días por defecto RF)
    model_mae = 1.6
    if _model_bundle is not None and "mae" in _model_bundle:
        model_mae = _model_bundle["mae"]

    # --- Precomputar tendencias mensuales por sitio (una vez) ---
    site_trends_cache = {}
    site_synthetic_cache = {}

    groups = historical.groupby(["Site", "Cultivar"])

    for (site, cultivar), group in groups:
        group_clean = group.dropna(subset=[target, "Year"])
        if len(group_clean) < 2:
            continue

        mean_doy = group_clean[target].mean()
        std_doy = group_clean[target].std()
        lat = group_clean["Latitude"].iloc[0]
        lon = group_clean["Longitude"].iloc[0]
        alt = group_clean["Altitude"].iloc[0] if not pd.isna(group_clean["Altitude"].iloc[0]) else 0
        country = group_clean["Country"].iloc[0]
        n_obs = len(group_clean)
        n_years = group_clean["Year"].nunique()
        year_min = int(group_clean["Year"].min())
        year_max = int(group_clean["Year"].max())
        year_range = f"{year_min}-{year_max}"
        years_arr = group_clean["Year"].values

        # =====================================================
        # PASO 1-2: Tendencia sobre T crudas → año sintético
        # =====================================================
        climate_projected = {}
        climate_trends = {}
        used_raw_trend = False

        if site in clima_by_site:
            # Calcular tendencias mensuales (cachear por sitio)
            if site not in site_trends_cache:
                trends_by_month, overall_slopes = _compute_monthly_temp_trends(
                    clima_by_site[site], year_pred
                )
                site_trends_cache[site] = (trends_by_month, overall_slopes)
            else:
                trends_by_month, overall_slopes = site_trends_cache[site]

            # Construir año sintético (cachear por sitio+DOY medio)
            # Cada cultivar puede tener distinto DOY medio → distinto bloom_doy
            cache_key = (site, int(round(mean_doy)))
            if cache_key not in site_synthetic_cache:
                climate_projected = _build_synthetic_year(
                    clima_by_site[site], trends_by_month, year_pred, mean_doy
                )
                site_synthetic_cache[cache_key] = climate_projected
            else:
                climate_projected = site_synthetic_cache[cache_key].copy()

            # Pendientes para reporte (traducir a variables derivadas)
            climate_trends = {
                "T2M_MIN": overall_slopes.get("T2M_MIN", 0.0),
                "T2M_MAX": overall_slopes.get("T2M_MAX", 0.0),
                "PRECTOTCORR": overall_slopes.get("PRECTOTCORR", 0.0),
                "ALLSKY_SFC_SW_DWN": overall_slopes.get("ALLSKY_SFC_SW_DWN", 0.0),
            }
            used_raw_trend = True

        # Fallback: si no hay datos diarios, extrapolar derivadas directamente
        if not used_raw_trend:
            for feat in CLIMATE_FEATURES:
                if feat in group_clean.columns:
                    vals = group_clean[feat].values.astype(float)
                    proj_val, slope = _extrapolate_variable(years_arr, vals, year_pred)
                    if feat in ("frost_days_total",):
                        proj_val = max(0, proj_val)
                    if feat in ("gdd_total", "precip_total", "rad_media"):
                        proj_val = max(0, proj_val)
                    climate_projected[feat] = proj_val
                    climate_trends[feat] = slope
                else:
                    climate_projected[feat] = 0.0
                    climate_trends[feat] = 0.0

        # =====================================================
        # PASO 3: Construir features para el modelo ML
        # =====================================================
        predicted_doy = None
        method = "Media histórica"

        if model is not None and len(model_features) > 0:
            feat_dict = {
                "Latitude": lat,
                "Longitude": lon,
                "Altitude": alt,
            }
            feat_dict.update(climate_projected)

            # Features engineered
            feat_dict["temp_range"] = (
                climate_projected.get("temp_max_30d", 0) -
                climate_projected.get("temp_min_30d", 0)
            )
            feat_dict["chill_gdd_ratio"] = (
                climate_projected.get("dynamic_chill_total", 0) /
                (climate_projected.get("gdd_total", 0) + 1)
            )
            feat_dict["lat_alt_interaction"] = lat * alt

            if "tree_age" in model_features:
                if "Plantation" in group_clean.columns:
                    mean_plantation = group_clean["Plantation"].mean()
                    feat_dict["tree_age"] = year_pred - mean_plantation if not pd.isna(mean_plantation) else 30
                else:
                    feat_dict["tree_age"] = 30

            feat_dict["Cultivar_enc"] = cultivar_enc_map.get(cultivar, 0)
            feat_dict["Cultivar_freq"] = cultivar_freq_map.get(cultivar, 1)
            feat_dict["Cultivar_target"] = cultivar_target_map.get(cultivar, global_mean_doy)
            feat_dict["Site_target"] = site_target_map.get(site, global_mean_doy)

            for f in model_features:
                if f not in feat_dict:
                    feat_dict[f] = 0

            try:
                X_pred = pd.DataFrame([feat_dict])[model_features]
                predicted_doy = float(model.predict(X_pred)[0])
                method = (
                    "ML +  tendencia climática"
                    if used_raw_trend
                    else "Modelo ML + tendencia climática"
                )
            except Exception:
                predicted_doy = None

        # Fallback: tendencia lineal directa sobre DOY
        if predicted_doy is None:
            doy_slope = 0.0
            if n_years >= 3:
                doys = group_clean[target].values
                try:
                    coeffs = np.polyfit(years_arr, doys, 1)
                    predicted_doy = float(np.polyval(coeffs, year_pred))
                    doy_slope = coeffs[0]
                    method = "Tendencia lineal DOY"
                except np.linalg.LinAlgError:
                    predicted_doy = mean_doy
            else:
                predicted_doy = mean_doy

        predicted_doy = float(np.clip(predicted_doy, 30, 180))

        # --- Intervalo de confianza (95%) ---
        # Combina incertidumbre del modelo (MAE) + variabilidad histórica
        hist_std = std_doy if (not pd.isna(std_doy) and std_doy > 0) else 5.0
        # Usar MAE como proxy de sigma del modelo (~68% de los errores)
        combined_std = np.sqrt(model_mae**2 + (hist_std / np.sqrt(max(n_obs, 1)))**2)
        ci_lower = float(np.clip(predicted_doy - 1.96 * combined_std, 30, 180))
        ci_upper = float(np.clip(predicted_doy + 1.96 * combined_std, 30, 180))

        # Tendencia del DOY (siempre calcularla para mostrar)
        doy_trend = 0.0
        if n_years >= 3:
            try:
                doy_coeffs = np.polyfit(years_arr, group_clean[target].values, 1)
                doy_trend = float(doy_coeffs[0])
            except (np.linalg.LinAlgError, ValueError):
                pass

        results.append({
            "Site": site,
            "Cultivar": cultivar,
            "Country": country,
            "Latitude": lat,
            "Longitude": lon,
            "Altitude": alt,
            "mean_historical_doy": round(mean_doy, 1),
            "std_historical_doy": round(std_doy, 1) if not pd.isna(std_doy) else 0,
            "trend_days_per_year": round(doy_trend, 3),
            "predicted_doy_2026": round(predicted_doy, 1),
            "ci_lower": round(ci_lower, 1),
            "ci_upper": round(ci_upper, 1),
            "n_observations": n_obs,
            "n_years": n_years,
            "year_range": year_range,
            "method": method,
            # Variables climáticas proyectadas (recalculadas desde T ajustadas)
            **{f"proj_{k}": round(v, 2) for k, v in climate_projected.items()},
            **{f"trend_{k}": round(v, 4) for k, v in climate_trends.items()},
        })

    df_pred = pd.DataFrame(results)
    return df_pred


@st.cache_data(ttl=3600)
def compute_site_map_data(df_predictions, year_pred=YEAR_PRED):
    """
    Agrega predicciones por sitio (promedio de cultivares) y calcula
    el porcentaje de progreso hacia la floración en el día actual.

    El progreso se define como:
      - Ventana de forzado: 1 de enero → DOY predicho de floración
      - Progreso = (DOY_actual / DOY_predicho) × 100, máximo 100%
      - Si DOY_actual >= DOY_predicho → 100% (ya floreció)
    """
    today = datetime.now()
    current_doy = today.timetuple().tm_yday

    # Agregar por sitio: media de predicciones de todos los cultivares
    site_agg = df_predictions.groupby("Site").agg({
        "Country": "first",
        "Latitude": "first",
        "Longitude": "first",
        "Altitude": "first",
        "predicted_doy_2026": "mean",
        "ci_lower": "mean",
        "ci_upper": "mean",
        "mean_historical_doy": "mean",
        "trend_days_per_year": "mean",
        "n_observations": "sum",
        "Cultivar": "count",  # número de cultivares
    }).reset_index()
    site_agg = site_agg.rename(columns={"Cultivar": "n_cultivars"})

    # Icono de aviso para sitios con pocas observaciones
    site_agg["fiabilidad"] = site_agg["n_observations"].apply(
        lambda n: "⚠️ Bajo" if n < 10 else "✅ Alto"
    )

    # Calcular progreso de floración
    site_agg["current_doy"] = current_doy
    site_agg["progress_pct"] = site_agg["predicted_doy_2026"].apply(
        lambda pred_doy: min(100.0, max(0.0, (current_doy / pred_doy) * 100))
    )

    # Días restantes
    site_agg["days_remaining"] = (site_agg["predicted_doy_2026"] - current_doy).clip(lower=0).astype(int)

    # Fecha estimada de floración
    site_agg["bloom_date"] = site_agg["predicted_doy_2026"].apply(
        lambda d: doy_to_date(d, year_pred)
    )

    # Estado fenológico descriptivo
    def get_phenological_stage(pct):
        if pct >= 100:
            return "🌸 Florecido"
        elif pct >= 75:
            return "🌷 Inminente"
        elif pct >= 50:
            return "🌿 Crecimiento activo"
        elif pct >= 25:
            return "❄️ Acumulando calor"
        else:
            return "🧊 Dormancia"

    site_agg["stage"] = site_agg["progress_pct"].apply(get_phenological_stage)

    return site_agg


def train_fallback_model(df):
    """
    Entrena un modelo rápido si no se encuentra uno serializado.
    Usa RandomForest con las features disponibles.
    """
    # Determinar nombre de columna de chill
    chill_col = "dynamic_chill_total" if "dynamic_chill_total" in df.columns \
                else "dynamic_chill_total"

    feature_cols = [
        "Latitude", "Longitude", "Altitude", "Year",
        chill_col, "gdd_total", "frost_days_total",
        "temp_media_30d", "temp_max_30d", "temp_min_30d",
        "precip_total", "rad_media"
    ]

    # Filtrar columnas disponibles
    available = [c for c in feature_cols if c in df.columns]

    target = "Beginning.of.flowering"
    if target not in df.columns:
        return None

    data = df.dropna(subset=available + [target])
    if len(data) < 100:
        return None

    X = data[available]
    y = data[target]

    model = RandomForestRegressor(
        n_estimators=500, max_depth=10,
        min_samples_split=10, min_samples_leaf=4,
        max_features='sqrt', bootstrap=True,
        n_jobs=-1, random_state=42
    )
    model.fit(X, y)

    # Calcular MAE en train como referencia
    from sklearn.metrics import mean_absolute_error
    y_pred = model.predict(X)
    mae = mean_absolute_error(y, y_pred)

    return {
        "model": model,
        "features": available,
        "name": "RandomForest (auto-entrenado)",
        "mae": mae,
        "chill_col": chill_col,
    }


# ============================================================================
# INTERFAZ PRINCIPAL
# ============================================================================

def main():
        # ---- Sidebar ----
    with st.sidebar:
        st.title("🌸 Cherry Blossom Predictor")
        
        st.markdown("""
        ¡Bienvenido! 🌱  
        Esta app predice **el inicio de floración** de cerezos (*Prunus avium*)  
        usando **ML** y datos climáticos históricos.
        """)
        
        st.markdown("---")
        if st.button("🔄 Limpiar caché"):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.rerun()
        
        st.subheader("🚀 Pipeline")
        
        with st.expander("1️⃣ Datos fenológicos (1978-2015)"):
            st.markdown("📂 Se usan registros históricos de floración de cerezos para entrenar el modelo.")
        
        with st.expander("2️⃣ Clima NASA POWER"):
            st.markdown("🌡️ Se integran datos de temperatura, precipitación y radiación solar por sitio y fecha.")
        
        with st.expander("3️⃣ Dynamic Chill + GDD"):
            st.markdown("❄️ Calculamos el enfriamiento dinámico y los grados-día acumulados para cada cultivar.")
        
        with st.expander("4️⃣ Tendencia de temperatura"):
            st.markdown("📈 Se procesan tendencias crudas y derivadas de temperatura para realizar la predicción de floración en 2026.")
        
        with st.expander("5️⃣ Visualización en tiempo real"):
            st.markdown("🗺️ Se muestra un mapa con el progreso de floración según el modelo y datos climáticos.")
        
        st.markdown("---")



    # ---- Carga de datos ----
    historical = load_historical_data()
    model_bundle = load_trained_model()

    if historical is None:
        st.error("❌ No se encontró `data/dataset_final.csv`. "
                "Ejecuta primero el pipeline: `python obtain_data.py`")
        st.stop()

    # Determinar qué columna de chill usa el dataset
    chill_col = "dynamic_chill_total" if "dynamic_chill_total" in historical.columns \
                else "dynamic_chill_total"

    # Entrenar modelo fallback si no hay uno serializado
    if model_bundle is None:
        with st.spinner("🔧 Entrenando modelo automático (no se encontró modelo serializado)..."):
            model_bundle = train_fallback_model(historical)
        if model_bundle is None:
            st.error("No se pudo entrenar un modelo. Verifica los datos.")
            st.stop()
        st.toast("ℹ️ Modelo auto-entrenado. Para mejores resultados, "
                "serializa un modelo desde prediccion_floracion.ipynb")

    # ---- Calcular predicciones 2026 (se usa en varios tabs) ----
    df_predictions_2026 = compute_trend_predictions_2026(
        historical, _model_bundle=model_bundle,
        model_name=model_bundle.get('name', 'unknown')
    )
    site_map_data = compute_site_map_data(df_predictions_2026)

    # ---- Tabs principales ----
    tab1, tab_map, tab2, tab3 = st.tabs([
        "📊 Datos Históricos",
        "🗺️ Mapa Floración 2026",
        "🔮 Predicciones Detalladas",
        "ℹ️ Metodología"
    ])

    # ==================================================================
    # TAB 1: DATOS HISTÓRICOS
    # ==================================================================
    with tab1:
        st.header("📊 Exploración de Datos Históricos")
        st.markdown("Datos fenológicos de cerezos (1978-2015) enriquecidos con "
                    "variables climáticas de NASA POWER.")

        # ---- Filtros ----
        col_f1, col_f2, col_f3, col_f4 = st.columns(4)

        with col_f1:
            countries = ["Todos"] + sorted(
                historical["Country"].dropna().unique().tolist()
            )
            sel_country = st.selectbox("🌍 País", countries)

        df_filt = historical.copy()
        if sel_country != "Todos":
            df_filt = df_filt[df_filt["Country"] == sel_country]

        with col_f2:
            site_list = ["Todos"] + sorted(df_filt["Site"].dropna().unique().tolist())
            sel_site = st.selectbox("📍 Sitio", site_list)

        if sel_site != "Todos":
            df_filt = df_filt[df_filt["Site"] == sel_site]

        with col_f3:
            cultivar_list = ["Todos"] + sorted(
                df_filt["Cultivar"].dropna().unique().tolist()
            )
            sel_cultivar = st.selectbox("🍒 Cultivar", cultivar_list)

        if sel_cultivar != "Todos":
            df_filt = df_filt[df_filt["Cultivar"] == sel_cultivar]

        with col_f4:
            year_min = int(df_filt["Year"].min()) if len(df_filt) > 0 else 1978
            year_max = int(df_filt["Year"].max()) if len(df_filt) > 0 else 2015
            sel_years = st.slider("📅 Período", year_min, year_max,
                                 (year_min, year_max))
            df_filt = df_filt[
                (df_filt["Year"] >= sel_years[0]) &
                (df_filt["Year"] <= sel_years[1])
            ]

        # ---- KPIs ----
        st.markdown("---")
        k1, k2, k3, k4, k5 = st.columns(5)

        with k1:
            st.metric("📊 Registros", f"{len(df_filt):,}")
        with k2:
            bloom_mean = df_filt["Beginning.of.flowering"].mean()
            st.metric("🌸 Floración media",
                     f"Día {bloom_mean:.0f}" if not pd.isna(bloom_mean) else "N/A",
                     help=f"≈ {doy_to_date(bloom_mean, 2026)}")
        with k3:
            st.metric("🌍 Países", df_filt["Country"].nunique())
        with k4:
            st.metric("📍 Sitios", df_filt["Site"].nunique())
        with k5:
            st.metric("🍒 Cultivares", df_filt["Cultivar"].nunique())

        # ---- Gráficos principales ----
        st.markdown("---")

        if len(df_filt) > 0:
            col_left, col_right = st.columns(2)

            with col_left:
                # Distribución de fechas de floración
                fig_hist = px.histogram(
                    df_filt, x="Beginning.of.flowering",
                    nbins=40,
                    title="Distribución de Fechas de Floración (DOY)",
                    labels={"Beginning.of.flowering": "Día del Año"},
                    color_discrete_sequence=[COLORS["pink_dark"]]
                )
                fig_hist.update_layout(
                    yaxis_title="Frecuencia",
                    bargap=0.05,
                    height=400
                )
                st.plotly_chart(fig_hist, use_container_width=True)

            with col_right:
                # Evolución temporal
                yearly = df_filt.groupby("Year")["Beginning.of.flowering"].agg(
                    ["mean", "std", "count"]
                ).reset_index()

                fig_trend = go.Figure()
                fig_trend.add_trace(go.Scatter(
                    x=yearly["Year"], y=yearly["mean"],
                    mode="lines+markers",
                    name="Media anual",
                    line=dict(color=COLORS["pink_dark"], width=2.5),
                    marker=dict(size=5),
                ))
                # Banda de ±1 std
                if "std" in yearly.columns and yearly["std"].notna().any():
                    upper = yearly["mean"] + yearly["std"]
                    lower = yearly["mean"] - yearly["std"]
                    fig_trend.add_trace(go.Scatter(
                        x=pd.concat([yearly["Year"], yearly["Year"][::-1]]),
                        y=pd.concat([upper, lower[::-1]]),
                        fill="toself",
                        fillcolor="rgba(255,20,147,0.15)",
                        line=dict(color="rgba(255,105,180,0)"),
                        name="± 1 Desv. Est.",
                    ))
                fig_trend.update_layout(
                    title="Evolución Temporal del Inicio de Floración",
                    xaxis_title="Año", yaxis_title="Día del Año (media)",
                    height=400, showlegend=True
                )
                st.plotly_chart(fig_trend, use_container_width=True)

            # -----------------------
            # NUEVO: Tendencia por Site o Cultivar
            # -----------------------
            st.markdown("---")  # separador

            # Selector de agrupación
            option_group = st.selectbox("Agrupar por:", ["Todos", "Site", "Cultivar"])

            # Preparar opciones con número de observaciones
            if option_group in ["Site", "Cultivar"]:
                # Excluir NaN para evitar errores
                df_nonan = df_filt[df_filt[option_group].notna()]
                counts = df_nonan[option_group].value_counts()
                options = [f"{val} ({counts[val]})" for val in counts.index]
                
                # Selección múltiple (hasta 5)
                selected_options = st.multiselect(
                    f"Selecciona hasta 5 {option_group}s:",
                    options=options,
                    default=options[:1],
                    max_selections=5
                )
                if not selected_options:
                    selected_options = [options[0]]  # asegurar al menos uno
            else:
                selected_options = ["Todos"]

            # Filtrar dataframe según la selección
            if option_group in ["Site", "Cultivar"]:
                df_plot = pd.DataFrame()
                for opt in selected_options:
                    name = opt.split(" (")[0]
                    df_plot = pd.concat([df_plot, df_filt[df_filt[option_group] == name]])
            else:
                df_plot = df_filt.copy()

            # Crear gráfico
            fig_trend_filt = go.Figure()

            for opt in selected_options:
                if opt == "Todos":
                    df_sub = df_plot.copy()
                    label = "Todos"
                else:
                    name = opt.split(" (")[0]
                    df_sub = df_plot[df_plot[option_group] == name]
                    label = opt

                # Agrupar por año
                yearly_filt = df_sub.groupby("Year")["Beginning.of.flowering"].agg(["mean", "std", "count"]).reset_index()

                # Línea media
                fig_trend_filt.add_trace(go.Scatter(
                    x=yearly_filt["Year"], y=yearly_filt["mean"],
                    mode="lines+markers",
                    name=f"{label}",
                    line=dict(width=2.5),
                    marker=dict(size=5),
                ))

                # Banda ±1 std
                if "std" in yearly_filt.columns and yearly_filt["std"].notna().any():
                    upper = yearly_filt["mean"] + yearly_filt["std"]
                    lower = yearly_filt["mean"] - yearly_filt["std"]
                    fig_trend_filt.add_trace(go.Scatter(
                        x=pd.concat([yearly_filt["Year"], yearly_filt["Year"][::-1]]),
                        y=pd.concat([upper, lower[::-1]]),
                        fill="toself",
                        fillcolor="rgba(199,21,133,0.15)" if opt != "Todos" else "rgba(100,100,100,0.1)",
                        line=dict(color="rgba(199,21,133,0)"),
                        showlegend=False
                    ))

            fig_trend_filt.update_layout(
                title=f"Evolución Temporal del Inicio de Floración ({option_group})",
                xaxis_title="Año",
                yaxis_title="Día del Año (media)",
                height=400,
                showlegend=True
            )

            st.plotly_chart(fig_trend_filt, use_container_width=True)



            # ---- Variables Climáticas vs Floración ----
            st.subheader("📈 Variables Climáticas vs Floración")

            climate_options = {
                chill_col: "❄️ Chill Acumulado",
                "gdd_total": "🌡️ GDD (Growing Degree Days)",
                "temp_media_30d": "🌡️ Temp. Media 30 días",
                "temp_max_30d": "🔥 Temp. Máxima 30 días",
                "temp_min_30d": "❄️ Temp. Mínima 30 días",
                "frost_days_total": "🥶 Días de Helada",
                "precip_total": "🌧️ Precipitación Total",
                "rad_media": "☀️ Radiación Solar Media",
                "Latitude": "🌍 Latitud",
                "Altitude": "🏔️ Altitud",
            }
            available_climate = {k: v for k, v in climate_options.items()
                                if k in df_filt.columns}

            sel_vars = st.multiselect(
                "Selecciona variables para visualizar:",
                options=list(available_climate.keys()),
                format_func=lambda x: available_climate[x],
                default=[chill_col, "gdd_total"] if chill_col in available_climate else []
            )

            if sel_vars:
                n_vars = len(sel_vars)
                n_cols_g = min(n_vars, 3)
                n_rows_g = (n_vars + n_cols_g - 1) // n_cols_g

                fig_scatter = make_subplots(
                    rows=n_rows_g, cols=n_cols_g,
                    subplot_titles=[available_climate[v] for v in sel_vars]
                )

                for i, var in enumerate(sel_vars):
                    row_g = i // n_cols_g + 1
                    col_g = i % n_cols_g + 1

                    mask = df_filt[[var, "Beginning.of.flowering"]].dropna()

                    fig_scatter.add_trace(
                        go.Scatter(
                            x=mask[var], y=mask["Beginning.of.flowering"],
                            mode="markers",
                            marker=dict(size=4, opacity=0.4,
                                       color=COLORS["pink_dark"]),
                            name=available_climate[var],
                            showlegend=False
                        ),
                        row=row_g, col=col_g
                    )

                    # Línea de tendencia
                    if len(mask) > 10:
                        try:
                            z = np.polyfit(mask[var].values,
                                          mask["Beginning.of.flowering"].values, 1)
                            p = np.poly1d(z)
                            x_sorted = np.sort(mask[var].values)
                            fig_scatter.add_trace(
                                go.Scatter(
                                    x=x_sorted, y=p(x_sorted),
                                    mode="lines",
                                    line=dict(color=COLORS["blue"], width=2.5,
                                             dash="dash"),
                                    name="Tendencia",
                                    showlegend=False
                                ),
                                row=row_g, col=col_g
                            )
                        except np.linalg.LinAlgError:
                            pass

                    fig_scatter.update_xaxes(title_text=var, row=row_g, col=col_g)
                    fig_scatter.update_yaxes(title_text="DOY", row=row_g, col=col_g)

                fig_scatter.update_layout(
                    height=350 * n_rows_g,
                    title_text="Relación Variables Climáticas vs Floración"
                )
                st.plotly_chart(fig_scatter, use_container_width=True)

                # Tabla de correlaciones
                st.markdown("**Correlaciones con la floración:**")
                corr_cols_ui = st.columns(min(len(sel_vars), 6))
                for i, var in enumerate(sel_vars[:6]):
                    corr_val = df_filt["Beginning.of.flowering"].corr(df_filt[var])
                    with corr_cols_ui[i]:
                        direction = "📈 Positiva" if corr_val > 0 else "📉 Negativa"
                        st.metric(
                            available_climate[var][:20],
                            f"r = {corr_val:.3f}",
                            delta=direction
                        )

            # ---- Floración por Cultivar ----
            if sel_cultivar == "Todos":
                st.subheader("🍒 Floración por Cultivar")
                top_n = st.slider("Top N cultivares", 5, 30, 15)

                top_cultivars = df_filt["Cultivar"].value_counts().head(top_n).index
                df_cult = df_filt[df_filt["Cultivar"].isin(top_cultivars)]

                if len(df_cult) > 0:
                    cult_order = (df_cult.groupby("Cultivar")[
                        "Beginning.of.flowering"
                    ].mean().sort_values().index.tolist())

                    fig_box = px.box(
                        df_cult, x="Cultivar", y="Beginning.of.flowering",
                        category_orders={"Cultivar": cult_order},
                        title=f"Distribución de Floración por Cultivar (Top {top_n})",
                        labels={"Beginning.of.flowering": "Día del Año"},
                        color_discrete_sequence=[COLORS["pink"]]
                    )
                    fig_box.update_layout(
                        xaxis_tickangle=-45, height=500,
                    )
                    st.plotly_chart(fig_box, use_container_width=True)

            # ---- Matriz de Correlación ----
            with st.expander("🔗 Matriz de Correlación Completa"):
                num_cols = ["Beginning.of.flowering", "Latitude", "Longitude",
                           "Altitude", chill_col, "gdd_total",
                           "frost_days_total", "temp_media_30d",
                           "temp_max_30d", "temp_min_30d",
                           "precip_total", "rad_media"]
                num_cols = [c for c in num_cols if c in df_filt.columns]

                if len(num_cols) > 2:
                    corr = df_filt[num_cols].corr()
                    fig_corr = px.imshow(
                        corr, text_auto=".2f",
                        color_continuous_scale="RdBu_r",
                        title="Matriz de Correlación",
                        aspect="auto"
                    )
                    fig_corr.update_layout(height=600)
                    st.plotly_chart(fig_corr, use_container_width=True)
        else:
            st.warning("No hay datos para los filtros seleccionados.")

    # ==================================================================
    # TAB MAPA: MAPA INTERACTIVO DE FLORACIÓN 2026
    # ==================================================================
    with tab_map:
        st.header("🗺️ Mapa de Progreso de Floración 2026")

        today = datetime.now()
        current_doy = today.timetuple().tm_yday

        # Verificar cuántas predicciones usan modelo ML
        n_ml = (df_predictions_2026["method"] == "Modelo ML + tendencia climática").sum()
        n_total = len(df_predictions_2026)
        st.markdown(
            f"**Fecha actual:** {today.strftime('%d %B %Y')} · "
            f"**Día del año:** {current_doy} · "
            f"Predicciones generadas con **variables climáticas extrapoladas** "
            f"alimentadas al **modelo ML** ({n_ml}/{n_total} combinaciones)."
        )

        # ---- KPIs globales ----
        st.markdown("---")
        km1, km2, km3, km4, km5 = st.columns(5)

        with km1:
            st.metric("📍 Sitios", len(site_map_data))
        with km2:
            avg_pred = site_map_data["predicted_doy_2026"].mean()
            st.metric("🌸 Floración media",
                     f"Día {avg_pred:.0f}",
                     help=f"≈ {doy_to_date(avg_pred)}")
        with km3:
            avg_progress = site_map_data["progress_pct"].mean()
            st.metric("📊 Progreso medio", f"{avg_progress:.0f}%")
        with km4:
            already_bloom = (site_map_data["progress_pct"] >= 100).sum()
            st.metric("🌸 Ya florecidos", f"{already_bloom}/{len(site_map_data)}")
        with km5:
            avg_trend = site_map_data["trend_days_per_year"].mean()
            st.metric("📉 Tendencia media",
                     f"{avg_trend:.2f} días/año",
                     delta="adelanto" if avg_trend < 0 else "retraso",
                     delta_color="normal")

        st.markdown("---")

        # ---- Mapa interactivo ----
        # Crear texto hover con información detallada
        site_map_data["hover_text"] = site_map_data.apply(
            lambda r: (
                f"<b>{r['Site']}</b> ({r['Country']})"
                f"{' ⚠️ Pocas obs.' if r['n_observations'] < 10 else ''}<br>"
                f"Progreso: {r['progress_pct']:.0f}%<br>"
                f"Estado: {r['stage']}<br>"
                f"Predicción: Día {r['predicted_doy_2026']:.0f} "
                f"IC 95%: [{r['ci_lower']:.0f}–{r['ci_upper']:.0f}] "
                f"({r['bloom_date']})<br>"
                f"Histórico: Día {r['mean_historical_doy']:.0f}<br>"
                f"Tendencia: {r['trend_days_per_year']:+.2f} días/año<br>"
                f"Días restantes: {r['days_remaining']}<br>"
                f"Cultivares: {r['n_cultivars']}"
            ), axis=1
        )

        # Color scale: azul (dormancia) → rosa (inminente) → magenta (florecido)
        fig_map = px.scatter_mapbox(
            site_map_data,
            lat="Latitude",
            lon="Longitude",
            color="progress_pct",
            size="n_cultivars",
            size_max=20,
            color_continuous_scale=[
                [0.0, "#1a237e"],     # azul oscuro - dormancia
                [0.25, "#42a5f5"],    # azul claro - acumulando calor
                [0.50, "#66bb6a"],    # verde - crecimiento activo
                [0.75, "#ffb74d"],    # naranja - cerca de floración
                [0.90, "#f06292"],    # rosa - inminente
                [1.0, "#c2185b"],     # magenta - florecido
            ],
            range_color=[0, 100],
            hover_name="Site",
            hover_data={
                "Latitude": False, "Longitude": False,
                "progress_pct": False, "n_cultivars": False,
            },
            custom_data=["hover_text"],
            zoom=3,
            center={"lat": site_map_data["Latitude"].mean(),
                    "lon": site_map_data["Longitude"].mean()},
            mapbox_style="carto-positron",
        )

        fig_map.update_traces(
            hovertemplate="%{customdata[0]}<extra></extra>"
        )

        fig_map.update_layout(
            height=600,
            margin={"r": 0, "t": 40, "l": 0, "b": 0},
            title="Progreso de Floración por Sitio (% completado)",
            coloraxis_colorbar=dict(
                title="Progreso (%)",
                tickvals=[0, 25, 50, 75, 100],
                ticktext=["0% Dormancia", "25%", "50%", "75%", "100% Florecido"],
            ),
        )

        st.plotly_chart(fig_map, use_container_width=True)

        # ---- Leyenda fenológica ----
        st.markdown("""        
        | Fase | Progreso | Descripción |
        |------|----------|-------------|
        | 🧊 Dormancia | 0-25% | Acumulación de frío invernal, el árbol está en reposo |
        | ❄️ Acumulando calor | 25-50% | Fin de la dormancia, inicio de acumulación térmica |
        | 🌿 Crecimiento activo | 50-75% | Yemas hinchándose, desarrollo vegetativo |
        | 🌷 Inminente | 75-99% | Botones florales visibles, floración muy próxima |
        | 🌸 Florecido | 100% | Inicio de floración alcanzado |
        """)

        # ---- Tabla de sitios con progreso ----
        st.markdown("---")
        st.subheader("📋 Estado de Floración por Sitio")

        # Filtro por país
        map_countries = ["Todos"] + sorted(
            site_map_data["Country"].dropna().unique().tolist()
        )
        sel_map_country = st.selectbox("Filtrar por país:", map_countries,
                                       key="map_country")

        display_data = site_map_data.copy()
        if sel_map_country != "Todos":
            display_data = display_data[
                display_data["Country"] == sel_map_country
            ]

        # Formatear para mostrar
        # Construir columna de IC formateada
        display_data["IC 95%"] = display_data.apply(
            lambda r: f"{r['ci_lower']:.0f}–{r['ci_upper']:.0f}", axis=1
        )

        display_df = display_data[[
            "Site", "Country", "progress_pct", "stage",
            "predicted_doy_2026", "IC 95%", "bloom_date", "days_remaining",
            "mean_historical_doy", "trend_days_per_year",
            "n_cultivars", "n_observations", "fiabilidad"
        ]].rename(columns={
            "Site": "Sitio",
            "Country": "País",
            "progress_pct": "Progreso (%)",
            "stage": "Estado",
            "predicted_doy_2026": "Pred. DOY",
            "bloom_date": "Fecha Estimada",
            "days_remaining": "Días Restantes",
            "mean_historical_doy": "Hist. DOY",
            "trend_days_per_year": "Tendencia (días/año)",
            "n_cultivars": "Cultivares",
            "n_observations": "Observaciones",
            "fiabilidad": "Fiabilidad",
        }).sort_values("Progreso (%)", ascending=False)

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Progreso (%)": st.column_config.ProgressColumn(
                    "Progreso (%)", min_value=0, max_value=100,
                    format="%.0f%%"
                ),
                "Tendencia (días/año)": st.column_config.NumberColumn(
                    format="%+.3f"
                ),
                "Pred. DOY": st.column_config.NumberColumn(format="%.0f"),
                "Hist. DOY": st.column_config.NumberColumn(format="%.0f"),
            }
        )

        # ---- Gráfico de barras: progreso por sitio ----
        st.markdown("---")
        st.subheader("📊 Progreso de Floración por Sitio")

        display_sorted = display_data.sort_values("progress_pct", ascending=True)

        fig_progress = go.Figure()
        fig_progress.add_trace(go.Bar(
            y=display_sorted["Site"],
            x=display_sorted["progress_pct"],
            orientation="h",
            marker=dict(
                color=display_sorted["progress_pct"],
                colorscale=[
                    [0.0, "#1a237e"],
                    [0.25, "#42a5f5"],
                    [0.50, "#66bb6a"],
                    [0.75, "#ffb74d"],
                    [1.0, "#c2185b"],
                ],
                cmin=0, cmax=100,
            ),
            text=display_sorted.apply(
                lambda r: f"{r['progress_pct']:.0f}% · {r['bloom_date']}", axis=1
            ),
            textposition="auto",
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Progreso: %{x:.0f}%<br>"
                "<extra></extra>"
            ),
        ))

        # Línea vertical en el 100%
        fig_progress.add_vline(
            x=100, line_dash="dash", line_color="red",
            annotation_text="🌸 Floración", annotation_position="top right"
        )

        fig_progress.update_layout(
            title=f"Progreso hacia la Floración (Día {current_doy}, {today.strftime('%d %B')})",
            xaxis_title="Progreso (%)",
            yaxis_title="",
            height=max(400, len(display_sorted) * 30),
            xaxis=dict(range=[0, 110]),
            showlegend=False,
        )
        st.plotly_chart(fig_progress, use_container_width=True)

        # ---- Análisis de tendencia temporal ----
        st.markdown("---")
        st.subheader("📉 Análisis de Tendencia Temporal (Cambio Climático)")

        st.markdown(
            "La tendencia lineal (días/año) indica cómo se ha desplazado "
            "la floración en el tiempo. **Valores negativos** significan que "
            "la floración se adelanta progresivamente (consistente con el "
            "calentamiento global)."
        )

        fig_trend_map = px.bar(
            display_data.sort_values("trend_days_per_year"),
            x="Site", y="trend_days_per_year",
            color="trend_days_per_year",
            color_continuous_scale="RdBu",
            color_continuous_midpoint=0,
            title="Tendencia Temporal por Sitio (días/año)",
            labels={"trend_days_per_year": "Tendencia (días/año)", "Site": "Sitio"},
        )
        fig_trend_map.update_layout(
            xaxis_tickangle=-45, height=450,
            coloraxis_colorbar_title="días/año"
        )
        st.plotly_chart(fig_trend_map, use_container_width=True)

    # ==================================================================
    # TAB 2: PREDICCIONES DETALLADAS 2026
    # ==================================================================
    with tab2:
        st.header(f"🔮 Predicciones Detalladas {YEAR_PRED}")
        st.markdown(
            f"Predicciones por **sitio y cultivar**: se aplica tendencia lineal "
            f"mensual sobre **T2M_MIN/T2M_MAX crudas** de NASA POWER, se genera "
            f"un año sintético {YEAR_PRED} y se **recalculan** chill, GDD, frost, "
            f"etc. desde las temperaturas ajustadas → **modelo ML**."
        )

        # ---- Filtros ----
        st.markdown("---")
        col_pf1, col_pf2, col_pf3 = st.columns(3)

        df_pred_filt = df_predictions_2026.copy()

        with col_pf1:
            pred_countries = ["Todos"] + sorted(
                df_pred_filt["Country"].dropna().unique().tolist()
            )
            pred_country = st.selectbox("🌍 País:", pred_countries,
                                        key="pred_detail_country")

        if pred_country != "Todos":
            df_pred_filt = df_pred_filt[
                df_pred_filt["Country"] == pred_country
            ]

        with col_pf2:
            pred_sites = ["Todos"] + sorted(
                df_pred_filt["Site"].dropna().unique().tolist()
            )
            pred_site_sel = st.selectbox("📍 Sitio:", pred_sites,
                                         key="pred_detail_site")

        if pred_site_sel != "Todos":
            df_pred_filt = df_pred_filt[
                df_pred_filt["Site"] == pred_site_sel
            ]

        with col_pf3:
            pred_cultivars = ["Todos"] + sorted(
                df_pred_filt["Cultivar"].dropna().unique().tolist()
            )
            pred_cult_sel = st.selectbox("🍒 Cultivar:", pred_cultivars,
                                         key="pred_detail_cult")

        if pred_cult_sel != "Todos":
            df_pred_filt = df_pred_filt[
                df_pred_filt["Cultivar"] == pred_cult_sel
            ]

        # ---- KPIs ----
        st.markdown("---")
        pk1, pk2, pk3, pk4 = st.columns(4)
        with pk1:
            st.metric("📊 Combinaciones", len(df_pred_filt))
        with pk2:
            if len(df_pred_filt) > 0:
                avg_p = df_pred_filt["predicted_doy_2026"].mean()
                st.metric("🌸 Predicción media",
                         f"Día {avg_p:.0f}",
                         help=doy_to_date(avg_p))
            else:
                st.metric("🌸 Predicción media", "N/A")
        with pk3:
            if len(df_pred_filt) > 0:
                diff_mean = (df_pred_filt["predicted_doy_2026"] -
                            df_pred_filt["mean_historical_doy"]).mean()
                st.metric("📉 Desplazamiento medio",
                         f"{diff_mean:+.1f} días",
                         delta="adelanto" if diff_mean < 0 else "retraso")
            else:
                st.metric("📉 Desplazamiento", "N/A")
        with pk4:
            if len(df_pred_filt) > 0:
                trend_m = df_pred_filt["trend_days_per_year"].mean()
                st.metric("📈 Tendencia media",
                         f"{trend_m:+.3f} días/año")
            else:
                st.metric("📈 Tendencia", "N/A")

        # ---- Tabla completa de predicciones ----
        st.markdown("---")
        st.subheader(f"📋 Predicciones {YEAR_PRED} por Sitio y Cultivar")

        if len(df_pred_filt) > 0:
            # Añadir fecha estimada
            df_show = df_pred_filt.copy()
            df_show["Fecha Estimada"] = df_show["predicted_doy_2026"].apply(
                lambda d: doy_to_date(d)
            )
            df_show["Diferencia (días)"] = (
                df_show["predicted_doy_2026"] - df_show["mean_historical_doy"]
            ).round(1)
            df_show["IC 95%"] = df_show.apply(
                lambda r: f"{r['ci_lower']:.0f}–{r['ci_upper']:.0f}", axis=1
            )
            df_show["Fiabilidad"] = df_show["n_observations"].apply(
                lambda n: "⚠️ Bajo" if n < 10 else "✅ Alto"
            )

            st.dataframe(
                df_show[[
                    "Country", "Site", "Cultivar",
                    "predicted_doy_2026", "IC 95%", "Fecha Estimada",
                    "mean_historical_doy", "Diferencia (días)",
                    "trend_days_per_year", "method",
                    "n_observations", "n_years", "year_range", "Fiabilidad"
                ]].rename(columns={
                    "Country": "País",
                    "Site": "Sitio",
                    "predicted_doy_2026": "Pred. DOY 2026",
                    "mean_historical_doy": "Media Hist.",
                    "trend_days_per_year": "Tendencia (días/año)",
                    "method": "Método",
                    "n_observations": "N Obs.",
                    "n_years": "N Años",
                    "year_range": "Rango Años",
                }).sort_values("Pred. DOY 2026"),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Tendencia (días/año)": st.column_config.NumberColumn(
                        format="%+.3f"
                    ),
                    "Pred. DOY 2026": st.column_config.NumberColumn(
                        format="%.0f"
                    ),
                    "Media Hist.": st.column_config.NumberColumn(
                        format="%.0f"
                    ),
                }
            )

            # ---- Gráfico: Predicción vs Histórico ----
            st.markdown("---")
            col_v1, col_v2 = st.columns(2)

            with col_v1:
                st.subheader("📊 Predicción vs Media Histórica")
                fig_vs = go.Figure()

                # Agregado por sitio
                site_agg_detail = df_pred_filt.groupby("Site").agg({
                    "predicted_doy_2026": "mean",
                    "mean_historical_doy": "mean",
                }).reset_index().sort_values("predicted_doy_2026")

                fig_vs.add_trace(go.Bar(
                    x=site_agg_detail["Site"],
                    y=site_agg_detail["predicted_doy_2026"],
                    name=f"Predicción {YEAR_PRED}",
                    marker_color=COLORS["pink_dark"]
                ))
                fig_vs.add_trace(go.Bar(
                    x=site_agg_detail["Site"],
                    y=site_agg_detail["mean_historical_doy"],
                    name="Media Histórica",
                    marker_color=COLORS["blue"],
                    opacity=0.6
                ))
                fig_vs.update_layout(
                    barmode="group",
                    xaxis_tickangle=-45,
                    yaxis_title="DOY",
                    height=400,
                )
                st.plotly_chart(fig_vs, use_container_width=True)

            with col_v2:
                st.subheader("📈 Dispersión Predicho vs Histórico")
                fig_scatter_pred = px.scatter(
                    df_pred_filt,
                    x="mean_historical_doy",
                    y="predicted_doy_2026",
                    color="Country",
                    hover_name="Site",
                    hover_data=["Cultivar", "trend_days_per_year"],
                    labels={
                        "mean_historical_doy": "Media Histórica (DOY)",
                        "predicted_doy_2026": f"Predicción {YEAR_PRED} (DOY)",
                    },
                )
                # Línea diagonal (predicción = histórico)
                doy_range = [df_pred_filt["mean_historical_doy"].min() - 5,
                            df_pred_filt["mean_historical_doy"].max() + 5]
                fig_scatter_pred.add_trace(go.Scatter(
                    x=doy_range, y=doy_range,
                    mode="lines",
                    line=dict(dash="dash", color="gray"),
                    name="Sin cambio",
                    showlegend=True
                ))
                fig_scatter_pred.update_layout(height=400)
                st.plotly_chart(fig_scatter_pred, use_container_width=True)

            # ---- Detalle por sitio seleccionado ----
            if pred_site_sel != "Todos":
                st.markdown("---")

                # ---- Variables climáticas proyectadas (expandible) ----
                with st.expander(f"🌡️ Variables Climáticas Proyectadas a {YEAR_PRED} — {pred_site_sel}"):
                    proj_cols = [c for c in df_pred_filt.columns if c.startswith("proj_")]
                    trend_cols = [c for c in df_pred_filt.columns if c.startswith("trend_")]

                    if proj_cols:
                        # Media por sitio de las proyecciones
                        site_proj = df_pred_filt[
                            df_pred_filt["Site"] == pred_site_sel
                        ]

                        if len(site_proj) > 0:
                            col_names_map = {
                                "proj_dynamic_chill_total": "❄️ Chill Dinámico",
                                "proj_gdd_total": "🌡️ GDD",
                                "proj_frost_days_total": "🥶 Días Helada",
                                "proj_temp_media_30d": "🌡️ T. Media",
                                "proj_temp_max_30d": "🔥 T. Máxima",
                                "proj_temp_min_30d": "❄️ T. Mínima",
                                "proj_precip_total": "🌧️ Precipitación",
                                "proj_rad_media": "☀️ Radiación",
                            }
                            trend_names_map = {
                                "trend_dynamic_chill_total": "❄️ Chill",
                                "trend_gdd_total": "🌡️ GDD",
                                "trend_frost_days_total": "🥶 Heladas",
                                "trend_temp_media_30d": "🌡️ T. Media",
                                "trend_temp_max_30d": "🔥 T. Máxima",
                                "trend_temp_min_30d": "❄️ T. Mínima",
                                "trend_precip_total": "🌧️ Precipitación",
                                "trend_rad_media": "☀️ Radiación",
                            }

                            # Métricas de proyección
                            pcols = st.columns(4)
                            for i, pc in enumerate(proj_cols):
                                label = col_names_map.get(pc, pc.replace("proj_", ""))
                                val = site_proj[pc].mean()
                                tc = pc.replace("proj_", "trend_")
                                trend_val = site_proj[tc].mean() if tc in site_proj.columns else 0
                                with pcols[i % 4]:
                                    st.metric(
                                        label,
                                        f"{val:.1f}",
                                        delta=f"{trend_val:+.3f}/año",
                                        delta_color="off"
                                    )

                            # Gráfico comparando histórico vs proyectado
                            st.markdown("**Comparación Histórico vs Proyectado:**")
                            site_hist_data = historical[
                                historical["Site"] == pred_site_sel
                            ]
                            if pred_cult_sel != "Todos":
                                site_hist_data = site_hist_data[
                                    site_hist_data["Cultivar"] == pred_cult_sel
                                ]

                            hist_means = {}
                            for feat in CLIMATE_FEATURES:
                                if feat in site_hist_data.columns:
                                    hist_means[feat] = site_hist_data[feat].mean()

                            compare_data = []
                            for feat in CLIMATE_FEATURES:
                                pc = f"proj_{feat}"
                                if pc in site_proj.columns and feat in hist_means:
                                    compare_data.append({
                                        "Variable": feat,
                                        "Histórico": round(hist_means[feat], 2),
                                        f"Proyectado {YEAR_PRED}": round(site_proj[pc].mean(), 2),
                                        "Cambio (%)": round(
                                            (site_proj[pc].mean() - hist_means[feat]) /
                                            (hist_means[feat] + 1e-6) * 100, 1
                                        ),
                                    })

                            if compare_data:
                                st.dataframe(
                                    pd.DataFrame(compare_data),
                                    use_container_width=True,
                                    hide_index=True,
                                )

                st.subheader(f"📈 Tendencia Histórica: {pred_site_sel}")

                site_hist = historical[
                    historical["Site"] == pred_site_sel
                ].copy()

                if pred_cult_sel != "Todos":
                    site_hist = site_hist[
                        site_hist["Cultivar"] == pred_cult_sel
                    ]

                if len(site_hist) > 2:
                    fig_site_trend = go.Figure()

                    # Datos históricos
                    yearly_site = site_hist.groupby("Year")[
                        "Beginning.of.flowering"
                    ].mean().reset_index()

                    fig_site_trend.add_trace(go.Scatter(
                        x=yearly_site["Year"],
                        y=yearly_site["Beginning.of.flowering"],
                        mode="markers+lines",
                        name="Observaciones",
                        marker=dict(color=COLORS["blue"], size=8),
                        line=dict(color=COLORS["blue"], width=1),
                    ))

                    # Línea de tendencia + extrapolación
                    years_fit = yearly_site["Year"].values
                    doys_fit = yearly_site[
                        "Beginning.of.flowering"
                    ].values

                    if len(years_fit) >= 3:
                        coeffs = np.polyfit(years_fit, doys_fit, 1)
                        p = np.poly1d(coeffs)
                        x_ext = np.append(years_fit, YEAR_PRED)
                        fig_site_trend.add_trace(go.Scatter(
                            x=x_ext, y=p(x_ext),
                            mode="lines",
                            name=f"Tendencia ({coeffs[0]:+.3f} días/año)",
                            line=dict(color=COLORS["pink_dark"],
                                     width=2.5, dash="dash"),
                        ))

                        # Punto predicción 2026
                        pred_val = df_pred_filt[
                            df_pred_filt["Site"] == pred_site_sel
                        ]["predicted_doy_2026"].mean()

                        fig_site_trend.add_trace(go.Scatter(
                            x=[YEAR_PRED], y=[pred_val],
                            mode="markers",
                            name=f"Predicción {YEAR_PRED}",
                            marker=dict(
                                color=COLORS["pink_dark"],
                                size=15, symbol="star",
                                line=dict(width=2, color="white")
                            ),
                        ))

                    fig_site_trend.update_layout(
                        title=f"Evolución y Extrapolación – {pred_site_sel}",
                        xaxis_title="Año",
                        yaxis_title="Día del Año (DOY)",
                        height=450,
                        hovermode="x unified",
                    )
                    st.plotly_chart(fig_site_trend,
                                  use_container_width=True)

                    st.caption(
                        f"La línea de tendencia extrapola la regresión "
                        f"lineal (DOY = a + b×Año) hasta {YEAR_PRED}. "
                        f"Pendiente: **{coeffs[0]:+.3f} días/año** "
                        f"({abs(coeffs[0]*10):.1f} días por década)."
                    )

        else:
            st.info("No hay predicciones con los filtros seleccionados.")

    # ==================================================================
    # TAB 3: METODOLOGÍA
    # ==================================================================
    with tab3:
        st.header("ℹ️ Metodología y Variables Clave")

        st.markdown("""
        ### Pipeline de Predicción

        Este proyecto predice el **día del año (DOY)** en el que comienza la
        floración de cerezos (*Prunus avium*) usando modelos de Machine Learning
        alimentados con datos fenológicos y climáticos.

        ---

        #### 1. Fuentes de Datos

        | Fuente | Datos | Cobertura |
        |--------|-------|-----------|
        | **Observaciones fenológicas** | Fechas de floración, cultivar, ubicación | 1978-2015 |
        | **NASA POWER** | Temperatura, precipitación, radiación (diarios) | 1981-actualidad |

        ---

        #### 2. Modelo Dinámico de Acumulación de Frío

        A diferencia del **Modelo Utah** (que asigna pesos fijos por rango de
        temperatura), el **Modelo Dinámico** (Fishman et al., 1987) simula un
        proceso bioquímico de dos etapas:
        """)

        st.latex(r"""
        \text{Etapa 1:} \quad
        x_s = \frac{A_0}{A_1} \cdot e^{(E_1 - E_0)/T_K}
        \quad \text{(equilibrio del intermediario)}
        """)

        st.latex(r"""
        \text{Etapa 2:} \quad
        \Delta CP = \xi \cdot x_e \quad \text{si } x_e \geq 1
        \quad \text{(conversión irreversible)}
        """)

        st.markdown("""
        **Ventajas del Modelo Dinámico sobre Utah:**
        - Modela la **destrucción parcial** del efecto de frío por calor
        - Tiene **"memoria térmica"** entre días consecutivos
        - Es el **estándar internacional** (FAO, IPGRI) para fenología frutal
        - Mejor predicción en climas mediterráneos y templados

        **Adaptación a datos diarios:**
        Se simulan 24 temperaturas horarias a partir de Tmax y Tmin diarias
        usando un perfil sinusoidal (Linvill, 1990):
        """)

        st.latex(r"""
        T(h) = \frac{T_{max} + T_{min}}{2}
        - \frac{T_{max} - T_{min}}{2} \cdot
        \cos\left(\frac{2\pi \cdot h}{24}\right)
        """)

        st.markdown("""
        ---

        #### 3. Variables del Modelo
        """)

        variables_info = pd.DataFrame([
            {"Variable": "dynamic_chill_total", "Descripción": "Porciones de frío acumuladas (Oct→floración)", "Tipo": "Climática"},
            {"Variable": "gdd_total", "Descripción": "Growing Degree Days acumulados (Ene→floración)", "Tipo": "Climática"},
            {"Variable": "frost_days_total", "Descripción": "Días con helada (Tmin < 0°C)", "Tipo": "Climática"},
            {"Variable": "temp_media_30d", "Descripción": "Temperatura media 30 días previos", "Tipo": "Climática"},
            {"Variable": "temp_max_30d", "Descripción": "Temp. máxima media 30 días previos", "Tipo": "Climática"},
            {"Variable": "temp_min_30d", "Descripción": "Temp. mínima media 30 días previos", "Tipo": "Climática"},
            {"Variable": "precip_total", "Descripción": "Precipitación total acumulada", "Tipo": "Climática"},
            {"Variable": "rad_media", "Descripción": "Radiación solar media", "Tipo": "Climática"},
            {"Variable": "Latitude", "Descripción": "Latitud del sitio", "Tipo": "Geográfica"},
            {"Variable": "Longitude", "Descripción": "Longitud del sitio", "Tipo": "Geográfica"},
            {"Variable": "Altitude", "Descripción": "Altitud del sitio (m)", "Tipo": "Geográfica"},
        ])
        st.dataframe(variables_info, use_container_width=True, hide_index=True)

        st.markdown("""
        ---

        #### 4. Modelos de Machine Learning

        Se evalúan múltiples algoritmos mediante pycaret y se selecciona el mejor por MAE y que no presente overfitting
        (Mean Absolute Error en días):

        | Modelo | Descripción |
        |--------|-------------|
        | Linear Regression | Baseline lineal |
        | Ridge / Lasso | Regularización L2 / L1 |
        | **Random Forest** | Ensemble de árboles con bagging |
        | Gradient Boosting | Boosting secuencial de árboles |
        | XGBoost | Gradient Boosting optimizado |
        | LightGBM | Gradient Boosting con histogramas |



        | Model               | MAE_Train | MAE_Test | MAE_Ratio | R2_Train | R2_Test | R2_Diff  | Diagnóstico               |
        |--------------------|-----------|----------|-----------|----------|---------|----------|---------------------------|
        | Linear Regression  | 2.447     | 2.431    | 0.994     | 0.927    | 0.929   | -0.002   | ✅ Sin Overfitting        |
        | Ridge              | 2.447     | 2.423    | 0.990     | 0.927    | 0.929   | -0.002   | ✅ Sin Overfitting        |
        | Lasso              | 2.661     | 2.598    | 0.976     | 0.915    | 0.920   | -0.005   | ✅ Sin Overfitting        |
        | Random Forest      | 1.415     | 1.600    | 1.131     | 0.970    | 0.962   | 0.008    | ✅ Sin Overfitting        |
        | Gradient Boosting  | 0.642     | 0.794    | 1.238     | 0.994    | 0.990   | 0.004    | ⚡ Overfitting Moderado   |
        | XGBoost            | 0.668     | 0.804    | 1.205     | 0.993    | 0.990   | 0.003    | ⚡ Overfitting Moderado   |
        | LightGBM           | 0.695     | 0.837    | 1.204     | 0.993    | 0.989   | 0.004    | ⚡ Overfitting Moderado   |

                    
        ---

        #### 5. Predicción 2026: Tendencia Climática + Modelo ML

        Para predecir la floración en **2026**, dado que no disponemos de
        datos climáticos futuros, se aplica un enfoque en dos pasos:

        **Paso 1 — Proyección climática:**
        Para cada ubicación, se ajusta una regresión
        lineal temporal sobre cada variable climática:

        $$\\text{var}(t) = a + b \\times t$$

        y se extrapola a 2026. La pendiente $b$ captura tendencias de
        cambio climático (e.g., aumento de Tmax).

        A partir de estas temperaturas proyectadas, se recalculan las variables climáticas clave.
                    
        - **Variables climáticas clave:**
            dynamic_chill_total, gdd_total, frost_days_total,
            temp_media, temp_max, temp_min, precipitación, radiación.

        **Paso 2 — Predicción con modelo ML:**
        Las variables climáticas proyectadas se combinan con las 
        demás features del modelo (e.g., geográficas (lat, lon, alt)) 
        para alimentar el modelo **Random Forest** entrenado, que predice el DOY.

        ---

        #### 6. Prevención de Data Leakage

        - ❌ Variables post-floración eliminadas (Full.Flowering, End.of.flowering, etc.)
        - ❌ Target encoding calculado **solo** en el conjunto de entrenamiento
        - ❌ Imputación KNN realizada sin usar la variable objetivo
        - ✅ Winsorizing aplicado excluyendo la variable objetivo
        - ✅ Train/Test split 80/20 antes de cualquier encoding basado en target

        ---
                    
        #### 7. Mejoras futuras

        - Incorporación de genotipos como feature
        - Uso de combinación de modelos mixtos + ML
        - Uso de datos predichos para temperaturas por modelos climáticos globales (CMIP6),para estimar el inicio de floración bajo escenarios futuros.


        ---


        #### 📚 Referencias

        1. Fishman, S. et al. (1987). *J. Theor. Biol.*, 124(4), 473-483.
        2. Linvill, D.E. (1990). *HortScience*, 25(1), 14-16.
        3. Erez, A. et al. (1988). *Acta Hort.*, 232, 76-89.
        4. McMaster, G.S. & Wilhelm, W.W. (1997). *Agric. Forest Meteorol.*,
           87(4), 291-300.

        ---
        """)

    # ---- Footer ----
    st.markdown("---")
    st.caption(
        "🌸 Predicción de Floración de Cerezos | "
        "Datos: NASA POWER | "
        f"Modelo: {model_bundle.get('name', 'ML')} | "
        "Dynamic Chill Model (Fishman et al., 1987)"
    )


if __name__ == "__main__":
    main()
