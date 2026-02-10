"""
===============================================================================
PIPELINE: Obtención de Datos Climáticos para Predicción de Floración de Cerezos
===============================================================================

Versión Python del pipeline obtain_data.R, reemplazando el Modelo Utah por el
Modelo Dinámico (Dynamic Model) para la acumulación de frío (chill portions).

MODELO DINÁMICO vs MODELO UTAH
-------------------------------
El Modelo Dinámico (Fishman et al., 1987) modela la dormancia como un proceso
bioquímico de dos etapas:
  1. Formación de un compuesto intermediario a temperaturas de frío
  2. Conversión irreversible del intermediario en "porciones de frío"
     (chill portions, CP) cuando se alcanza un umbral

Ventajas sobre el Modelo Utah:
  - Más fisiológicamente preciso (modela bioquímica real)
  - Considera la destrucción parcial del intermediario por calor
  - Mejor comportamiento en climas mediterráneos y templados
  - Es el estándar actual en fenología frutal (FAO, IPGRI)
  - Tiene en cuenta la "memoria térmica" entre días consecutivos

ADAPTACIÓN A DATOS DIARIOS
----------------------------
NASA POWER proporciona solo temperaturas diarias (Tmax, Tmin). Para aplicar el
Modelo Dinámico (que requiere datos horarios), se simulan 24 temperaturas por día
usando un perfil sinusoidal (Linvill, 1990):
    T(h) = (Tmax + Tmin)/2 - (Tmax - Tmin)/2 · cos(2π·h/24)

Esto ubica Tmin a medianoche (h=0) y Tmax al mediodía (h=12), lo cual es una
aproximación razonable para latitudes medias.

FLUJO DEL PIPELINE
--------------------
  1) Cargar datos de fenología desde archivo Excel
  2) Extraer sitios únicos con coordenadas y rangos temporales
  3) Descargar datos climáticos diarios de NASA POWER
  4) Calcular Porciones de Frío (Dynamic Model) y GDD por día
  5) Acumular variables climáticas hasta cada fecha de floración
  6) Generar dataset final listo para modelado

USO
----
  python obtain_data.py
  python obtain_data.py --input ruta/fenologia.xlsx --output ruta/salida/
  python obtain_data.py --help

DEPENDENCIAS
-------------
  pip install pandas numpy requests openpyxl tqdm

REFERENCIAS
------------
  [1] Fishman, S., Erez, A., & Couvillon, G.A. (1987a). "The temperature
      dependence of dormancy breaking in plants: Mathematical analysis of a
      two-step model involving a cooperative transition". J. Theor. Biol.,
      124(4), 473-483.
  [2] Fishman, S., Erez, A., & Couvillon, G.A. (1987b). "The temperature
      dependence of dormancy breaking in plants: Computer simulation of
      processes studied under controlled temperatures". J. Theor. Biol.,
      126(3), 309-321.
  [3] Linvill, D.E. (1990). "Calculating chilling hours and chill units from
      daily maximum and minimum temperature observations". HortScience,
      25(1), 14-16.
  [4] Erez, A., Fishman, S., Gat, Z., & Couvillon, G.A. (1988). "Evaluation
      of winter climate for breaking bud rest using the Dynamic Model".
      Acta Hort., 232, 76-89.

Autor: Pipeline Python - Bootcamp Ciencia de Datos 2026
===============================================================================
"""

import pandas as pd
import numpy as np
import requests
import time
import os
import argparse
from pathlib import Path
from datetime import datetime, timedelta

# Intentar importar tqdm; si no está disponible, usar un sustituto simple
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        """Sustituto simple de tqdm cuando no está instalado."""
        total = kwargs.get('total', None)
        desc = kwargs.get('desc', '')
        for i, item in enumerate(iterable):
            if total:
                print(f"\r  {desc}: {i+1}/{total}", end="", flush=True)
            yield item
        print()


# ============================================================================
# 1. MODELO DINÁMICO DE ACUMULACIÓN DE FRÍO (Dynamic Model)
# ============================================================================

class DynamicChillModel:
    """
    Implementación del Modelo Dinámico de acumulación de frío.
    
    El modelo simula la formación y destrucción de un compuesto intermediario
    que, al superar un umbral, se convierte irreversiblemente en una
    "porción de frío" (chill portion, CP).
    
    A diferencia del Modelo Utah (que evalúa cada hora independientemente),
    el Modelo Dinámico mantiene un ESTADO INTERMEDIO que se propaga entre
    horas consecutivas. Esto significa que:
    - La historia térmica completa influye en la acumulación
    - Periodos cálidos pueden destruir parcialmente el intermediario
    - No es posible calcular porciones de frío día a día de forma independiente
    
    Parámetros del modelo (Fishman et al., 1987):
    -------------------------------------------
    E0 (4153.5 K):    Energía de activación para la formación del intermediario.
                       Controla la sensibilidad a la temperatura fría.
    E1 (12888.8 K):   Energía de activación para la destrucción del intermediario.
                       Controla la destrucción por calor.
    A0 (1.395×10⁵):   Factor de frecuencia para formación.
    A1 (2.567×10¹⁸):  Factor de frecuencia para destrucción.
    SLP (1.28):        Pendiente de la curva sigmoidal que determina la "calidad"
                       de la porción de frío generada.
    TETMLT (277 K):    Temperatura crítica (~3.85°C). Es el punto de inflexión
                       de la curva sigmoidal de calidad.
    
    Ejemplo de uso:
    >>> model = DynamicChillModel()
    >>> total_cp, daily_cp = model.accumulate_series(tmax_array, tmin_array)
    >>> print(f"Total chill portions: {total_cp:.1f}")
    """
    
    # ---- Parámetros del modelo (Fishman et al., 1987) ----
    E0 = 4153.5        # Energía de activación formación (K)
    E1 = 12888.8       # Energía de activación destrucción (K)
    A0 = 1.395e5       # Factor pre-exponencial formación
    A1 = 2.567e18      # Factor pre-exponencial destrucción
    SLP = 1.28         # Pendiente de la curva sigmoidal de calidad
    TETMLT = 277.0     # Temperatura crítica en Kelvin (~3.85°C)
    
    def __init__(self):
        """Inicializa el modelo con estado intermediario en cero."""
        self.inter_s = 0.0         # Estado intermediario (persiste entre horas)
        self.chill_portions = 0.0  # Porciones de frío acumuladas
    
    def reset(self):
        """Reinicia el estado del modelo para una nueva serie temporal."""
        self.inter_s = 0.0
        self.chill_portions = 0.0
    
    def _process_hour(self, temp_c: float):
        """
        Procesa una hora de temperatura y actualiza el estado del modelo.
        
        Ecuaciones del Modelo Dinámico:
        
        1) Valor de equilibrio del intermediario (xs):
           xs = (A0/A1) · exp((E1-E0)/TK)
           → Resultado: a temperaturas de frío (~2-12°C), xs ≈ 1
                        a temperaturas cálidas (>15°C), xs → 0
        
        2) Constante de velocidad de reacción (ak1):
           ak1 = A1 · exp(-E1/TK)
           → Controla la velocidad a la que el intermediario alcanza xs
        
        3) Actualización del intermediario (inter_e):
           inter_e = xs - (xs - inter_s) · exp(-ak1)
           → El intermediario tiende exponencialmente hacia xs
        
        4) Factor de calidad sigmoidal (xi):
           xi = exp(SLP·TETMLT·(TK-TETMLT)/TK) / (1 + exp(...))
           → Determina qué fracción del intermediario se convierte en CP
        
        5) Si inter_e ≥ 1 → se genera una porción de frío:
           delta_cp = xi · inter_e
           chill_portions += delta_cp
           inter_s = inter_e - delta_cp
        
        Parámetros
        ----------
        temp_c : float
            Temperatura en grados Celsius para esta hora.
        """
        TK = temp_c + 273.15  # Convertir a Kelvin
        
        # Protección contra valores extremos (evitar overflow en exp)
        TK = max(TK, 220.0)  # -53°C mínimo
        TK = min(TK, 330.0)  # 57°C máximo
        
        # 1) Valor de equilibrio del intermediario
        try:
            xs = self.A0 / self.A1 * np.exp((self.E1 - self.E0) / TK)
        except (OverflowError, FloatingPointError):
            xs = 0.0
        
        # 2) Constante de velocidad (Arrhenius)
        try:
            ak1 = self.A1 * np.exp(-self.E1 / TK)
        except (OverflowError, FloatingPointError):
            ak1 = 0.0
        
        # 3) Actualización del intermediario
        inter_e = xs - (xs - self.inter_s) * np.exp(-ak1)
        
        # 4) Factor de calidad sigmoidal
        try:
            ftmprt = self.SLP * self.TETMLT * (TK - self.TETMLT) / TK
            st = np.exp(ftmprt)
            xi = st / (1.0 + st)
        except (OverflowError, FloatingPointError):
            xi = 1.0 if TK > self.TETMLT else 0.0
        
        # 5) Verificar umbral para generar porción de frío
        if inter_e >= 1.0:
            delta_cp = xi * inter_e
            self.chill_portions += delta_cp
            self.inter_s = inter_e - delta_cp
        else:
            self.inter_s = inter_e
    
    def process_day(self, tmax: float, tmin: float) -> float:
        """
        Procesa un día completo simulando 24 temperaturas horarias.
        
        Aproximación sinusoidal (Linvill, 1990):
            T(h) = T_media - amplitud · cos(2π · h / 24)
        
        Donde:
        - T_media = (Tmax + Tmin) / 2
        - amplitud = (Tmax - Tmin) / 2
        - Tmin ocurre a medianoche (h=0), Tmax a mediodía (h=12)
        
        Esta aproximación es adecuada para latitudes medias donde el ciclo
        diurno sigue un patrón sinusoidal. Para climas tropicales o polares,
        podrían necesitarse modelos más sofisticados.
        
        Parámetros
        ----------
        tmax : float - Temperatura máxima del día (°C)
        tmin : float - Temperatura mínima del día (°C)
            
        Retorna
        -------
        float : Porciones de frío acumuladas hasta el final de este día
        """
        if pd.isna(tmax) or pd.isna(tmin) or tmax < tmin:
            return self.chill_portions
        
        t_mean = (tmax + tmin) / 2.0
        amplitude = (tmax - tmin) / 2.0
        
        for hour in range(24):
            # Perfil sinusoidal: mínimo en h=0 (madrugada), máximo en h=12
            temp_c = t_mean - amplitude * np.cos(2.0 * np.pi * hour / 24.0)
            self._process_hour(temp_c)
        
        return self.chill_portions
    
    def accumulate_series(self, tmax_series, tmin_series):
        """
        Calcula porciones de frío acumuladas para una serie de días.
        
        IMPORTANTE: El estado intermediario se propaga entre días consecutivos.
        Esta es la ventaja fundamental del Modelo Dinámico sobre el Modelo Utah:
        la "memoria térmica" permite modelar correctamente situaciones como:
        - Un periodo frío seguido de uno cálido (destrucción parcial)
        - Oscilaciones térmicas diarias (efecto acumulativo no lineal)
        
        Parámetros
        ----------
        tmax_series : array-like
            Serie de temperaturas máximas diarias (°C)
        tmin_series : array-like
            Serie de temperaturas mínimas diarias (°C)
        
        Retorna
        -------
        float : Total de porciones de frío acumuladas
        list  : Porciones acumuladas día a día (para visualización)
        """
        self.reset()
        daily_cp = []
        
        for tmax, tmin in zip(tmax_series, tmin_series):
            self.process_day(tmax, tmin)
            daily_cp.append(self.chill_portions)
        
        return self.chill_portions, daily_cp


# ============================================================================
# 2. FUNCIONES AUXILIARES DE CÁLCULO CLIMÁTICO
# ============================================================================

def calculate_gdd(tmax: float, tmin: float, t_base: float = 5.0) -> float:
    """
    Calcula los Growing Degree Days (GDD) para un día.
    
    GDD = max(0, (Tmax + Tmin)/2 - T_base)
    
    Los GDD miden la acumulación de calor disponible para el crecimiento vegetal.
    Para cerezo (Prunus avium), la temperatura base típica es 5°C, por debajo
    de la cual no hay crecimiento vegetativo significativo.
    
    Parámetros
    ----------
    tmax : float - Temperatura máxima del día (°C)
    tmin : float - Temperatura mínima del día (°C)
    t_base : float - Temperatura base para crecimiento (°C). Default: 5°C
    
    Retorna
    -------
    float : GDD del día (siempre ≥ 0)
    
    Referencia
    ----------
    McMaster, G.S. & Wilhelm, W.W. (1997). "Growing degree-days: one equation,
    two interpretations". Agricultural and Forest Meteorology, 87(4), 291-300.
    """
    if pd.isna(tmax) or pd.isna(tmin):
        return np.nan
    t_mean = (tmax + tmin) / 2.0
    return max(0.0, t_mean - t_base)


def is_frost_day(tmin: float) -> int:
    """
    Determina si hubo helada (Tmin < 0°C).
    
    Las heladas tardías son un factor de riesgo para la floración del cerezo,
    pudiendo dañar yemas florales abiertas o en proceso de apertura.
    
    Retorna
    -------
    int : 1 si Tmin < 0°C (helada), 0 en caso contrario
    """
    if pd.isna(tmin):
        return 0
    return 1 if tmin < 0.0 else 0


# ============================================================================
# 3. DESCARGA DE DATOS DE NASA POWER
# ============================================================================

def download_nasa_power(lat: float, lon: float, start_year: int, end_year: int,
                        site_name: str = "", max_retries: int = 3) -> pd.DataFrame:
    """
    Descarga datos climáticos diarios desde la API de NASA POWER.
    
    NASA POWER (Prediction Of Worldwide Energy Resources) proporciona datos
    meteorológicos globales derivados de satélite y modelos de reanálisis,
    con resolución espacial de 0.5° × 0.5° y cobertura temporal desde 1981.
    
    Variables descargadas
    ---------------------
    - T2M:              Temperatura media a 2m (°C)
    - T2M_MAX:          Temperatura máxima a 2m (°C)
    - T2M_MIN:          Temperatura mínima a 2m (°C)
    - PRECTOTCORR:      Precipitación total corregida (mm/día)
    - ALLSKY_SFC_SW_DWN: Radiación solar de onda corta incidente (MJ/m²/día)
    - RH2M:             Humedad relativa a 2m (%)
    
    Parámetros
    ----------
    lat : float         Latitud del sitio (grados decimales)
    lon : float         Longitud del sitio (grados decimales)
    start_year : int    Año de inicio de la descarga
    end_year : int      Año de fin de la descarga
    site_name : str     Nombre descriptivo del sitio (para logs)
    max_retries : int   Número máximo de reintentos en caso de error
    
    Retorna
    -------
    pd.DataFrame : Datos diarios con columnas procesadas, o None si falla
    
    Notas
    -----
    - NASA POWER tiene datos desde 1981 (se ajusta automáticamente)
    - Los valores de relleno (-999) se reemplazan por NaN
    - Se incluye una pausa de cortesía entre reintentos
    """
    # NASA POWER solo tiene datos desde 1981
    start_year = max(start_year, 1981)
    
    # Endpoint de la API
    url = "https://power.larc.nasa.gov/api/temporal/daily/point"
    params = {
        "parameters": "T2M,T2M_MAX,T2M_MIN,PRECTOTCORR,ALLSKY_SFC_SW_DWN,RH2M",
        "community": "AG",
        "longitude": round(lon, 4),
        "latitude": round(lat, 4),
        "start": f"{start_year}0101",
        "end": f"{end_year}1231",
        "format": "JSON"
    }
    
    for attempt in range(max_retries):
        try:
            print(f"  📡 Descargando {site_name} ({lat:.2f}, {lon:.2f}) "
                  f"| {start_year}-{end_year} | Intento {attempt + 1}/{max_retries}")
            
            resp = requests.get(url, params=params, timeout=120)
            resp.raise_for_status()
            data = resp.json()
            
            # Verificar respuesta
            if "properties" not in data or "parameter" not in data["properties"]:
                print(f"  ⚠️  Respuesta inesperada de la API para {site_name}")
                continue
            
            # Extraer parámetros
            parameters = data["properties"]["parameter"]
            dates = sorted(parameters["T2M"].keys())
            
            # Construir DataFrame
            df = pd.DataFrame({
                "date": pd.to_datetime(dates, format="%Y%m%d"),
                "T2M": [parameters["T2M"][d] for d in dates],
                "T2M_MAX": [parameters["T2M_MAX"][d] for d in dates],
                "T2M_MIN": [parameters["T2M_MIN"][d] for d in dates],
                "PRECTOTCORR": [parameters["PRECTOTCORR"][d] for d in dates],
                "ALLSKY_SFC_SW_DWN": [parameters["ALLSKY_SFC_SW_DWN"][d] for d in dates],
                "RH2M": [parameters.get("RH2M", {d: np.nan for d in dates})[d] 
                         for d in dates],
            })
            
            # Reemplazar valores de relleno de NASA POWER (-999) por NaN
            fill_value = -999.0
            for col in ["T2M", "T2M_MAX", "T2M_MIN", "PRECTOTCORR",
                        "ALLSKY_SFC_SW_DWN", "RH2M"]:
                df[col] = df[col].replace(fill_value, np.nan)
            
            # Agregar columnas de metadata temporal
            df["Site"] = site_name
            df["year"] = df["date"].dt.year
            df["month"] = df["date"].dt.month
            df["day_of_year"] = df["date"].dt.dayofyear
            
            print(f"  ✅ {site_name}: {len(df)} días descargados correctamente")
            return df
            
        except requests.exceptions.Timeout:
            print(f"  ⏱️  Timeout en intento {attempt + 1}")
        except requests.exceptions.ConnectionError:
            print(f"  🔌 Error de conexión en intento {attempt + 1}")
        except requests.exceptions.HTTPError as e:
            print(f"  ❌ Error HTTP {e.response.status_code} en intento {attempt + 1}")
        except Exception as e:
            print(f"  ❌ Error inesperado en intento {attempt + 1}: {e}")
        
        if attempt < max_retries - 1:
            wait_time = 10 * (attempt + 1)
            print(f"  ⏳ Esperando {wait_time}s antes de reintentar...")
            time.sleep(wait_time)
    
    print(f"  ❌ No se pudieron descargar datos para {site_name} tras {max_retries} intentos")
    return None


# ============================================================================
# 4. CARGA Y PREPARACIÓN DE DATOS DE FENOLOGÍA
# ============================================================================

def load_phenology(filepath: str) -> pd.DataFrame:
    """
    Carga datos de fenología desde archivo Excel o CSV.
    
    El archivo debe contener datos de observaciones fenológicas de cerezos
    con columnas como: Country, Site/Station, Latitude, Longitude, Altitude,
    Year, Cultivar, Beginning.of.flowering, etc.
    
    Parámetros
    ----------
    filepath : str
        Ruta al archivo Excel (.xlsx/.xls) o CSV con datos de fenología.
    
    Retorna
    -------
    pd.DataFrame : Datos de fenología cargados y con nombres de columnas limpiados
    """
    print(f"\n{'='*70}")
    print(f"📂 PASO 1: Cargando datos de fenología")
    print(f"{'='*70}")
    print(f"  Archivo: {filepath}")
    
    ext = Path(filepath).suffix.lower()
    if ext in ['.xlsx', '.xls']:
        df = pd.read_excel(filepath)
    elif ext == '.csv':
        df = pd.read_csv(filepath, index_col=0)
    else:
        raise ValueError(f"Formato no soportado: {ext}. Use .xlsx, .xls o .csv")
    
    # Limpiar nombres de columnas (eliminar espacios extra, caracteres no-ASCII)
    df.columns = [c.strip().replace('\xa0', ' ').strip() for c in df.columns]
    
    # Crear columna Site si no existe
    if 'Site' not in df.columns:
        if 'Country' in df.columns and 'Station' in df.columns:
            df['Site'] = df['Country'].astype(str) + '_' + df['Station'].astype(str)
            print("  ℹ️  Columna 'Site' creada como Country_Station")
        else:
            df['Site'] = 'Unknown'
            print("  ⚠️  No se encontraron Country/Station, Site='Unknown'")
    
    print(f"  ✅ {len(df)} registros cargados")
    print(f"  📊 Columnas ({len(df.columns)}): {list(df.columns)}")
    
    return df


def extract_sites(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extrae sitios únicos con coordenadas geográficas y rangos temporales.
    
    Para cada sitio, determina:
    - Coordenadas (Latitude, Longitude)
    - Rango temporal (Start_Year, End_Year)
    - Número de observaciones (n_obs)
    
    El Start_Year se reduce en 1 para poder descargar datos de octubre previo,
    necesarios para la acumulación de frío invernal.
    
    Parámetros
    ----------
    df : pd.DataFrame
        Datos de fenología con columnas Site, Latitude, Longitude, Year.
    
    Retorna
    -------
    pd.DataFrame : Sitios únicos con metadatos
    """
    print(f"\n{'='*70}")
    print(f"📍 PASO 2: Extrayendo sitios únicos")
    print(f"{'='*70}")
    
    # Filtrar registros con coordenadas y año válidos
    valid = df.dropna(subset=['Latitude', 'Longitude', 'Year']).copy()
    valid['Latitude'] = pd.to_numeric(valid['Latitude'], errors='coerce')
    valid['Longitude'] = pd.to_numeric(valid['Longitude'], errors='coerce')
    valid['Year'] = pd.to_numeric(valid['Year'], errors='coerce').astype('Int64')
    
    valid = valid.dropna(subset=['Latitude', 'Longitude', 'Year'])
    
    sites = valid.groupby(['Site', 'Latitude', 'Longitude']).agg(
        Start_Year=('Year', 'min'),
        End_Year=('Year', 'max'),
        n_obs=('Year', 'count')
    ).reset_index()
    
    # Pedir un año extra antes para chill hours desde octubre
    sites['Start_Year'] = sites['Start_Year'] - 1
    sites = sites.sort_values('Site').reset_index(drop=True)
    
    print(f"  ✅ {len(sites)} sitios identificados")
    print(f"\n  {'Site':<30} {'Lat':>7} {'Lon':>8} {'Inicio':>6} {'Fin':>6} {'N':>5}")
    print(f"  {'─'*30} {'─'*7} {'─'*8} {'─'*6} {'─'*6} {'─'*5}")
    for _, row in sites.iterrows():
        print(f"  {row['Site']:<30} {row['Latitude']:>7.2f} {row['Longitude']:>8.2f} "
              f"{int(row['Start_Year']):>6} {int(row['End_Year']):>6} {int(row['n_obs']):>5}")
    
    return sites


# ============================================================================
# 5. DESCARGA MASIVA DE DATOS CLIMÁTICOS
# ============================================================================

def download_all_sites(sites: pd.DataFrame, cache_dir: str = None) -> pd.DataFrame:
    """
    Descarga datos climáticos diarios de NASA POWER para todos los sitios.
    
    Características:
    - Descarga secuencial con pausas de cortesía (2s entre sitios)
    - Sistema de caché: guarda/recupera CSVs individuales por sitio
    - Reintentos automáticos en caso de error
    - Barra de progreso con tqdm
    
    Parámetros
    ----------
    sites : pd.DataFrame
        DataFrame con columnas: Site, Latitude, Longitude, Start_Year, End_Year
    cache_dir : str, optional
        Directorio para cachear descargas individuales. Si un archivo ya existe,
        se carga desde caché sin descargar de nuevo. Muy recomendado para
        evitar re-descargas en caso de interrupción.
    
    Retorna
    -------
    pd.DataFrame : Todos los datos diarios combinados de todos los sitios
    """
    print(f"\n{'='*70}")
    print(f"📡 PASO 3: Descargando datos climáticos de NASA POWER")
    print(f"{'='*70}")
    print(f"  Total de sitios: {len(sites)}")
    print(f"  Caché: {cache_dir if cache_dir else 'Desactivado'}")
    
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
    
    all_data = []
    downloaded = 0
    cached = 0
    failed = 0
    
    for idx, row in tqdm(sites.iterrows(), total=len(sites), desc="  Progreso"):
        site_name = row['Site']
        
        # Verificar caché
        if cache_dir:
            safe_name = site_name.replace(' ', '_').replace('/', '_')
            cache_file = os.path.join(cache_dir, f"{safe_name}.csv")
            
            if os.path.exists(cache_file):
                try:
                    df_cached = pd.read_csv(cache_file, parse_dates=['date'])
                    all_data.append(df_cached)
                    cached += 1
                    continue
                except Exception:
                    pass  # Si falla la lectura del caché, re-descargar
        
        # Descargar de NASA POWER
        df_site = download_nasa_power(
            lat=row['Latitude'],
            lon=row['Longitude'],
            start_year=int(row['Start_Year']),
            end_year=int(row['End_Year']),
            site_name=site_name
        )
        
        if df_site is not None:
            all_data.append(df_site)
            downloaded += 1
            # Guardar en caché
            if cache_dir:
                safe_name = site_name.replace(' ', '_').replace('/', '_')
                cache_file = os.path.join(cache_dir, f"{safe_name}.csv")
                df_site.to_csv(cache_file, index=False)
        else:
            failed += 1
        
        # Pausa de cortesía para no sobrecargar la API
        time.sleep(2)
    
    if not all_data:
        raise RuntimeError("No se pudieron descargar datos para ningún sitio")
    
    clima_daily = pd.concat(all_data, ignore_index=True)
    
    print(f"\n  ✅ Descarga finalizada")
    print(f"     Descargados: {downloaded} sitios")
    print(f"     Desde caché: {cached} sitios")
    print(f"     Fallidos:    {failed} sitios")
    print(f"     Total días:  {len(clima_daily)}")
    
    return clima_daily


# ============================================================================
# 6. ACUMULACIÓN DE VARIABLES CLIMÁTICAS HASTA FLORACIÓN
# ============================================================================

def accumulate_climate(clima_daily: pd.DataFrame, 
                       fenologia: pd.DataFrame) -> pd.DataFrame:
    """
    Para cada observación de floración, acumula variables climáticas desde
    periodos específicos hasta la fecha de floración.
    
    Variables acumuladas
    --------------------
    dynamic_chill_total : Porciones de frío (Dynamic Model)
                          Período: 1 Oct año anterior → fecha de floración
                          El estado intermediario se propaga entre días.
    
    gdd_total :           Growing Degree Days acumulados
                          Período: 1 Ene → fecha de floración
    
    frost_days_total :    Número de días con helada (Tmin < 0°C)
                          Período: 1 Ene → fecha de floración
    
    temp_media_30d :      Temperatura media de los 30 días previos a floración
    temp_max_30d :        Temperatura máxima media de los 30 días previos
    temp_min_30d :        Temperatura mínima media de los 30 días previos
    
    precip_total :        Precipitación total acumulada
                          Período: 1 Ene → fecha de floración
    
    rad_media :           Radiación solar media
                          Período: 1 Ene → fecha de floración
    
    Parámetros
    ----------
    clima_daily : pd.DataFrame
        Datos diarios con: Site, date, T2M, T2M_MAX, T2M_MIN, PRECTOTCORR, etc.
    fenologia : pd.DataFrame
        Datos de fenología con: Site, Year, Beginning.of.flowering, etc.
    
    Retorna
    -------
    pd.DataFrame : Fenología enriquecida con variables climáticas acumuladas
    """
    print(f"\n{'='*70}")
    print(f"🔄 PASO 5: Acumulando variables climáticas hasta cada floración")
    print(f"{'='*70}")
    print(f"  Observaciones a procesar: {len(fenologia)}")
    print(f"  Modelo de frío: Dynamic Model (Fishman et al., 1987)")
    
    # Asegurar tipos correctos
    clima_daily = clima_daily.copy()
    clima_daily['date'] = pd.to_datetime(clima_daily['date'])
    
    # Pre-indexar datos climáticos por sitio para acceso eficiente
    clima_by_site = {}
    for site, group in clima_daily.groupby('Site'):
        clima_by_site[site] = group.sort_values('date').reset_index(drop=True)
    
    # Instanciar modelo dinámico (reutilizable)
    chill_model = DynamicChillModel()
    
    results = []
    errors = 0
    
    for idx, row in tqdm(fenologia.iterrows(), total=len(fenologia),
                         desc="  Acumulando"):
        site = row['Site']
        year = int(row['Year'])
        doy = row['Beginning.of.flowering']
        
        # Validar datos de entrada
        if pd.isna(doy) or site not in clima_by_site:
            errors += 1
            continue
        
        doy = int(doy)
        clima_site = clima_by_site[site]
        
        # ===== Período de CHILL: Oct año anterior → floración =====
        chill_start = pd.Timestamp(year=year-1, month=10, day=1)
        try:
            bloom_date = pd.Timestamp(year=year, month=1, day=1) + pd.Timedelta(days=doy-1)
        except (ValueError, OverflowError):
            errors += 1
            continue
        
        mask_chill = (clima_site['date'] >= chill_start) & \
                     (clima_site['date'] <= bloom_date)
        datos_chill = clima_site.loc[mask_chill].sort_values('date')
        
        # ===== Período de enero→floración (mismo año) =====
        year_start = pd.Timestamp(year=year, month=1, day=1)
        mask_year = (clima_site['date'] >= year_start) & \
                    (clima_site['date'] <= bloom_date)
        datos_year = clima_site.loc[mask_year]
        
        # ----- Dynamic Chill Portions -----
        if len(datos_chill) > 0:
            chill_model.reset()
            total_cp, _ = chill_model.accumulate_series(
                datos_chill['T2M_MAX'].values,
                datos_chill['T2M_MIN'].values
            )
        else:
            total_cp = np.nan
        
        # ----- GDD acumulado (enero→floración) -----
        if len(datos_year) > 0:
            gdd_values = [calculate_gdd(tmax, tmin) for tmax, tmin 
                         in zip(datos_year['T2M_MAX'], datos_year['T2M_MIN'])]
            gdd_total = np.nansum(gdd_values)
        else:
            gdd_total = np.nan
        
        # ----- Días de helada (enero→floración) -----
        if len(datos_year) > 0:
            frost_total = sum(is_frost_day(tmin) for tmin in datos_year['T2M_MIN'])
        else:
            frost_total = np.nan
        
        # ----- Temperaturas promedio, 30 días previos a floración -----
        mask_30d = (clima_site['date'] > bloom_date - pd.Timedelta(days=30)) & \
                   (clima_site['date'] <= bloom_date)
        datos_30d = clima_site.loc[mask_30d]
        
        if len(datos_30d) > 0:
            temp_media_30d = datos_30d['T2M'].mean()
            temp_max_30d = datos_30d['T2M_MAX'].mean()
            temp_min_30d = datos_30d['T2M_MIN'].mean()
        else:
            temp_media_30d = temp_max_30d = temp_min_30d = np.nan
        
        # ----- Precipitación total (enero→floración) -----
        precip_total = (datos_year['PRECTOTCORR'].sum() 
                       if len(datos_year) > 0 else np.nan)
        
        # ----- Radiación solar media (enero→floración) -----
        rad_media = (datos_year['ALLSKY_SFC_SW_DWN'].mean() 
                    if len(datos_year) > 0 else np.nan)
        
        # ----- Construir registro resultado -----
        result_row = row.to_dict()
        result_row.update({
            'dynamic_chill_total': total_cp,
            'gdd_total': gdd_total,
            'frost_days_total': frost_total,
            'temp_media_30d': temp_media_30d,
            'temp_max_30d': temp_max_30d,
            'temp_min_30d': temp_min_30d,
            'precip_total': precip_total,
            'rad_media': rad_media,
        })
        results.append(result_row)
    
    df_result = pd.DataFrame(results)
    
    print(f"\n  ✅ Acumulación completada")
    print(f"     Observaciones procesadas: {len(df_result)}")
    print(f"     Errores/omitidos: {errors}")
    if 'dynamic_chill_total' in df_result.columns:
        print(f"     NAs en dynamic_chill_total: "
              f"{df_result['dynamic_chill_total'].isna().sum()}")
        print(f"     Rango chill portions: "
              f"{df_result['dynamic_chill_total'].min():.1f} - "
              f"{df_result['dynamic_chill_total'].max():.1f}")
    
    return df_result


# ============================================================================
# 7. PREPARACIÓN DEL DATASET FINAL
# ============================================================================

def prepare_final_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepara el dataset final para modelado predictivo.
    
    Pasos realizados:
    1. Verificar columnas obligatorias
    2. Crear Cultivar_enc (frequency encoding)
    3. Reportar valores nulos por variable
    4. Eliminar filas con NAs en variables climáticas críticas
    5. Mostrar resumen estadístico
    
    Parámetros
    ----------
    df : pd.DataFrame
        Dataset con variables fenológicas y climáticas acumuladas
    
    Retorna
    -------
    pd.DataFrame : Dataset limpio y listo para modelado
    """
    print(f"\n{'='*70}")
    print(f"🔧 PASO 6: Preparando dataset final para modelado")
    print(f"{'='*70}")
    
    # Verificar columnas necesarias
    required = ['Beginning of Maturity', 'Latitude', 'Longitude', 
                'Year', 'Site', 'dynamic_chill_total', 'gdd_total']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"  ❌ Columnas faltantes: {missing}")
    
    df_final = df.copy()
    
    # Frequency encoding para Cultivar (si existe)
    if 'Cultivar' in df_final.columns:
        cultivar_freq = df_final['Cultivar'].value_counts().to_dict()
        df_final['Cultivar_enc'] = df_final['Cultivar'].map(cultivar_freq)
        print(f"  ✅ Cultivar codificado (frequency encoding): "
              f"{len(cultivar_freq)} categorías")
    
    # Resumen de NAs
    climate_cols = ['dynamic_chill_total', 'gdd_total', 'frost_days_total',
                    'temp_media_30d', 'temp_max_30d', 'temp_min_30d',
                    'precip_total', 'rad_media']
    climate_cols = [c for c in climate_cols if c in df_final.columns]
    
    print(f"\n  📊 Valores nulos por variable climática:")
    for col in climate_cols:
        na_count = df_final[col].isna().sum()
        na_pct = (na_count / len(df_final)) * 100
        status = "✅" if na_pct < 5 else "⚠️ " if na_pct < 20 else "❌"
        print(f"     {status} {col}: {na_count} ({na_pct:.1f}%)")
    
    # Eliminar filas con NAs en variables críticas
    n_before = len(df_final)
    critical = climate_cols + ['Beginning.of.flowering']
    df_final = df_final.dropna(subset=critical)
    n_after = len(df_final)
    n_dropped = n_before - n_after
    
    print(f"\n  📌 Filas eliminadas por NAs: {n_dropped} "
          f"({n_dropped/n_before*100:.1f}%)")
    print(f"  📌 Dataset final: {n_after} observaciones × "
          f"{len(df_final.columns)} variables")
    
    # Resumen estadístico de variables climáticas
    print(f"\n  📋 Resumen estadístico de variables climáticas:")
    stats = df_final[climate_cols].describe().round(2)
    print(stats.to_string())
    
    return df_final


# ============================================================================
# 8. EJECUCIÓN DEL PIPELINE COMPLETO
# ============================================================================

def run_pipeline(input_path: str, output_dir: str, cache_dir: str = None):
    """
    Ejecuta el pipeline completo de obtención y procesamiento de datos.
    
    Flujo:
    1. Carga fenología → 2. Extrae sitios → 3. Descarga NASA POWER →
    4. Prepara fenología → 5. Acumula clima → 6. Prepara dataset final
    
    Archivos de salida
    ------------------
    - sites_info.csv         : Información de sitios extraídos
    - clima_diario_nasa.csv  : Datos climáticos diarios crudos
    - dataset_con_clima.csv  : Dataset completo con clima acumulado
    - dataset_final.csv      : Dataset limpio listo para modelado
    
    Parámetros
    ----------
    input_path : str
        Ruta al archivo Excel/CSV con datos de fenología
    output_dir : str
        Directorio donde guardar los resultados
    cache_dir : str, optional
        Directorio para cachear descargas individuales de NASA POWER
    """
    start_time = time.time()
    
    print("\n" + "=" * 70)
    print("🍒 PIPELINE: OBTENCIÓN DE DATOS CLIMÁTICOS")
    print("   PREDICCIÓN DE FLORACIÓN DE CEREZOS")
    print("─" * 70)
    print(f"   Modelo de frío:    Dynamic Model (Fishman et al., 1987)")
    print(f"   Fuente climática: NASA POWER API")
    print(f"   Entrada:          {input_path}")
    print(f"   Salida:           {output_dir}")
    print("=" * 70)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # ─── PASO 1: Cargar fenología ───
    phenology = load_phenology(input_path)
    
    # ─── PASO 2: Extraer sitios ───
    sites = extract_sites(phenology)
    sites_path = os.path.join(output_dir, "sites_info.csv")
    sites.to_csv(sites_path, index=False)
    print(f"  💾 Guardado: {sites_path}")
    
    # ─── PASO 3: Descargar datos climáticos ───
    clima_daily = download_all_sites(sites, cache_dir=cache_dir)
    clima_path = os.path.join(output_dir, "clima_diario_nasa.csv")
    clima_daily.to_csv(clima_path, index=False)
    print(f"  💾 Guardado: {clima_path}")
    
    # ─── PASO 4: Preparar fenología para acumulación ───
    print(f"\n{'='*70}")
    print(f"📋 PASO 4: Preparando datos de fenología para acumulación")
    print(f"{'='*70}")
    
    fenologia_valid = phenology.dropna(
        subset=['Beginning of flowering', 'Latitude', 'Longitude']
    ).copy()
    
    # Convertir tipos numéricos
    for col in ['Latitude', 'Longitude', 'Altitude', 'Plantation']:
        if col in fenologia_valid.columns:
            fenologia_valid[col] = pd.to_numeric(fenologia_valid[col], errors='coerce')
    
    fenologia_valid['Beginning.of.flowering'] = pd.to_numeric(
        fenologia_valid['Beginning.of.flowering'], errors='coerce'
    )
    fenologia_valid = fenologia_valid.dropna(subset=['Beginning.of.flowering'])
    fenologia_valid['Beginning.of.flowering'] = fenologia_valid[
        'Beginning.of.flowering'
    ].astype(int)
    
    print(f"  ✅ Observaciones de fenología válidas: {len(fenologia_valid)}")
    
    # ─── PASO 5: Acumular clima hasta cada floración ───
    dataset_con_clima = accumulate_climate(clima_daily, fenologia_valid)
    
    # Guardar dataset con clima
    dataset_path = os.path.join(output_dir, "dataset_con_clima.csv")
    dataset_con_clima.to_csv(dataset_path, index=True)
    print(f"  💾 Guardado: {dataset_path}")
    
    # ─── PASO 6: Preparar dataset final ───
    dataset_final = prepare_final_dataset(dataset_con_clima)
    final_path = os.path.join(output_dir, "dataset_final.csv")
    dataset_final.to_csv(final_path, index=False)
    print(f"  💾 Guardado: {final_path}")
    
    # ─── RESUMEN FINAL ───
    elapsed = time.time() - start_time
    elapsed_str = str(timedelta(seconds=int(elapsed)))
    
    print(f"\n{'='*70}")
    print(f"🍒 PIPELINE COMPLETADO EXITOSAMENTE")
    print(f"{'='*70}")
    print(f"  ⏱️  Tiempo total: {elapsed_str}")
    print(f"  📁 Archivos generados en: {output_dir}/")
    print(f"     ├── sites_info.csv         ({len(sites)} sitios)")
    print(f"     ├── clima_diario_nasa.csv  ({len(clima_daily)} días)")
    print(f"     ├── dataset_con_clima.csv  ({len(dataset_con_clima)} obs.)")
    print(f"     └── dataset_final.csv      ({len(dataset_final)} obs.)")
    print(f"\n  ➡️  Siguiente paso: Entrenar modelos con el notebook")
    print(f"     prediccion_floracion.ipynb")
    print(f"{'='*70}\n")
    
    return dataset_final


# ============================================================================
# CONFIGURACIÓN Y PUNTO DE ENTRADA
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Pipeline de obtención de datos climáticos para predicción de "
            "floración de cerezos. Descarga datos de NASA POWER y calcula "
            "porciones de frío (Dynamic Model) y GDD para cada observación."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  python obtain_data.py
  python obtain_data.py --input fenologia.xlsx --output ./resultados/
  python obtain_data.py -i datos.csv -o ./salida/ -c ./cache/
        """
    )
    parser.add_argument(
        "--input", "-i",
        default=r"Sweet_cherry_phenology_data_1978-2015.xlsx",
        help="Ruta al archivo Excel/CSV con datos de fenología (default: Sweet_cherry...xlsx)"
    )
    parser.add_argument(
        "--output", "-o",
        default="./data",
        help="Directorio de salida para resultados (default: ./data)"
    )
    parser.add_argument(
        "--cache", "-c",
        default="./data/cache_nasa",
        help="Directorio para cachear descargas de NASA POWER (default: ./data/cache_nasa)"
    )
    
    args = parser.parse_args()
    
    # Ejecutar pipeline
    run_pipeline(
        input_path=args.input,
        output_dir=args.output,
        cache_dir=args.cache
    )
