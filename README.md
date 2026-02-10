# 🌸 Cherry Blossom Bloom Prediction
### Machine Learning para Predicción Fenológica de *Prunus avium*

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8-orange.svg)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.39-red.svg)](https://streamlit.io/)

> Predicción del día de floración de cerezos usando datos climáticos históricos, datos fenológicos
y AutoML.

## Accede a la web app aquí: [Cherry Blossom Predictor](https://bloappmcherry-u2t5pqymxqrljiplf9dpxr.streamlit.app/)


---

## 📋 Tabla de Contenidos

- [Descripción](#-descripción)
- [Características](#-características-principales)
- [Metodología](#-metodología)
- [Resultados](#-resultados)
- [Datos](#-datos)
- [Autores](#-autores)

---

## 🎯 Descripción

Este proyecto predice el **día del año (DOY)** en que comienza la floración de cerezos (*Prunus avium*) utilizando:

- ❄️ **Modelo Dinámico de Frío** (Fishman et al., 1987) - Estándar internacional para fenología frutal
- 🌡️ **Growing Degree Days (GDD)** - Acumulación de calor primaveral
- 🌍 **Variables geográficas** - Latitud, longitud, altitud
- 🍒 **Características genéticas** - Cultivar y edad del árbol
- 🤖 **Gradient Boosting** - Modelo de ensemble optimizado

### ¿Por qué es importante?

```
🚜 Agricultura de Precisión
   └─ Optimizar riego, fertilización y protección contra heladas
   
🌸 Turismo Estacional
   └─ Planificar eventos de Hanami (observación de flores)
   
🌡️ Cambio Climático
   └─ Monitorear adelanto fenológico (-0.17 días/año desde 1978)
   
📊 Investigación Científica
   └─ Validar modelos climáticos regionales
```

---
## Contexto biológico
Los árboles caducifolios, como los cerezos, presentan un ciclo anual característico, que incluye la floración, la maduración de frutos y la entrada en un estado de latencia durante el invierno. Este estado se conoce como dormancia. La salida de la dormancia depende de la acumulación de horas de frío durante la primera fase del invierno, seguida por la acumulación de calor en primavera. Para estimar estas variables existen distintos modelos fenológicos que permiten calcularlas.

Dado que estas variables son críticas para la floración, en nuestro trabajo hemos calculado parámetros de chill y GDD para cada sitio, incluyendo además la ubicación geográfica y el cultivar como factor genético. Para simplificar el análisis, no hemos utilizado datos genómicos como SNPs, enfocándonos únicamente en variables fenotípicas y ambientales.

![Contexto biológico del requerimiento de frío](https://cdn.portalfruticola.com/2016/12/fabbisogno_freddo_es_21.jpg)

## ✨ Características Principales

### 📊 **Dashboard Interactivo**
- Exploración de datos históricos (1978-2015)
- Filtros por país, sitio, cultivar y año
- Visualizaciones dinámicas con Plotly

### 🗺️ **Mapa de Progreso en Tiempo Real**
- Integración con Open-Meteo API
- Cálculo en vivo de porciones de frío y GDD
- Visualización del % de progreso hacia floración

### 🔮 **Predicciones 2026**
- Extrapolación de tendencias climáticas
- Predicciones por sitio y cultivar
- Análisis de impacto del cambio climático

### 🔬 **Modelo Científico Riguroso**
- Dynamic Chill Model (Fishman et al., 1987)
- Prevención estricta de data leakage
- Validación cruzada 5-fold
- Feature engineering con sentido agronómico


---

## 🔬 Metodología

### Pipeline General

```
┌─────────────────────────────────────────────────────────────┐
│  1. DATOS FENOLÓGICOS (1978-2015)                          │
│     └─ 10,961 observaciones de floración en Europa         │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  2. DATOS CLIMÁTICOS (NASA POWER)                           │
│     └─ Tmax, Tmin, Precipitación, Radiación (diarios)      │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  3. CÁLCULO DE VARIABLES FENOLÓGICAS                        │
│     ├─ Dynamic Chill (Fishman et al., 1987)                │
│     ├─ GDD (Growing Degree Days)                            │
│     └─ Días de helada, temperaturas 30d                     │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  4. FEATURE ENGINEERING                                     │
│     ├─ chill_gdd_ratio (balance frío/calor)                │
│     ├─ temp_range (amplitud térmica)                        │
│     ├─ tree_age (edad del árbol)                            │
│     └─ lat_alt_interaction (interacción geográfica)         │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  5. PREPROCESAMIENTO                                        │
│     ├─ Imputación KNN (k=5)                                 │
│     ├─ Winsorizing (P1-P99)                                 │
│     ├─ Target Encoding (SOLO en train)                      │
│     └─ RobustScaler                                         │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  6. MODELADO (Cross-Validation 5-Fold)                      │
│     ├─ Linear Regression                                    │
│     ├─ Ridge / Lasso                                        │
│     ├─ Random Forest                                        │
│     ├─ Gradient Boosting ✓ MEJOR                            │
│     ├─ XGBoost                                              │
│     └─ LightGBM                                             │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  7. PREDICCIONES 2026                                       │
│     ├─ Extrapolación lineal de variables climáticas        │
│     └─ Predicción con modelo GradientBoosting              │
└─────────────────────────────────────────────────────────────┘
```

### Modelo Dinámico de Frío

El **Dynamic Model** (Fishman et al., 1987) es superior al clásico **Utah Model** porque:

| Característica | Utah Model | Dynamic Model |
|----------------|------------|---------------|
| **Memoria térmica** | ❌ No | ✅ Sí |
| **Reversión por calor** | ❌ No | ✅ Sí |
| **Base bioquímica** | ❌ Empírico | ✅ Mecanístico |
| **Estándar FAO/IPGRI** | ❌ No | ✅ Sí |

**Ecuaciones clave:**

```
Etapa 1 (equilibrio del intermediario):
x_s = (A₀/A₁) × exp[(E₁ - E₀)/T_K]

Etapa 2 (conversión irreversible):
ΔCP = ξ × x_e  (si x_e ≥ 1)
```

**Simulación de temperaturas horarias** (Linvill, 1990):

```
T(h) = T_media - Amplitud × cos(2π × h/24)

donde:
  T_media = (T_max + T_min) / 2
  Amplitud = (T_max - T_min) / 2
```



## 📈 Resultados

### Performance del Modelo

| Métrica | Train | Test | Interpretación |
|---------|-------|------|----------------|
| **MAE** | 2.8 días | **3.4 días** | Error promedio ±3.4 días |
| **RMSE** | 4.1 días | 4.9 días | Penaliza errores grandes |
| **R²** | 0.91 | **0.87** | Explica 87% de la varianza |

**Validación Cruzada (5-Fold):**
- MAE promedio: 3.6 ± 0.4 días
- Modelo estable y generalizable

### Top 10 Variables Más Importantes

```
### Top 15 Features más importantes 

1. ❄️ **dynamic_chill_total** (15.2%) ━━━━━━━━━━━━━━━━━
2. 🧊 **frost_days_total** (14.7%) ━━━━━━━━━━━━━━━━
3. 🌍 **Latitude** (12.3%) ━━━━━━━━━━━━━
4. 🌍 **Site_target** (11.1%) ━━━━━━━━━━━
5. 💧 **precip_total** (10.0%) ━━━━━━━━━
6. 🌍 **Longitude** (4.9%) ━━
7. 🌡️ **temp_max_30d** (4.6%) ━━
8. 🌡️ **gdd_total** (4.4%) ━━
9. ☀️ **rad_media** (4.4%) ━━
10. ⚖️ **chill_gdd_ratio** (3.6%) ━
11. 🍒 **Cultivar_target** (3.0%) ━
12. 🌡️ **temp_media_30d** (2.7%) ━
13. 🌡️ **temp_min_30d** (2.4%) ━
14. 🌡️ **temp_range** (2.1%) ━
15. 🏔️ **Altitude** (2.0%) ━

```

### Tendencia Temporal (Cambio Climático)

**Adelanto fenológico observado (1978-2015):**

| Región | Tendencia | Adelanto Total |
|--------|-----------|----------------|
| Sur de Francia | **-0.22 días/año** | ≈8.4 días |
| Norte de Italia | **-0.19 días/año** | ≈7.2 días |
| Alemania Central | **-0.15 días/año** | ≈5.7 días |
| Reino Unido | **-0.12 días/año** | ≈4.6 días |

**Promedio global: -0.17 días/año** (≈1.7 días por década)

---

## 📊 Datos

### Fuentes

#### 1. Observaciones Fenológicas
- **Fuente**: Data from: A collection of European sweet cherry phenology data for assessing climate change
- **Periodo**: 1978-2015
- **Variables**: Fechas de floración, cultivar, ubicación, año de plantación
- **Cobertura**: Francia, Alemania, España, Italia, Reino Unido, Austria, Suiza, Países Bajos
- **Acceso**: [Acceso a datos](https://doi.org/10.5061/dryad.1d28m)

#### 2. Datos Climáticos
- **NASA POWER API**: Temperatura, precipitación, radiación (1981-actualidad)

### Estadísticas del Dataset

```
Total de observaciones:    10,961
Periodo temporal:          1978-2015 (38 años)
Número de países:          8
Número de sitios:          127
Número de cultivares:      52
Rango latitudinal:         41°N - 55°N
Rango altitudinal:         0 - 850 m
```

### Variables (19 features finales)

| Categoría | Variables |
|-----------|-----------|
| **Climáticas** (8) | dynamic_chill_total, gdd_total, frost_days_total, temp_media_30d, temp_max_30d, temp_min_30d, precip_total, rad_media |
| **Geográficas** (3) | Latitude, Longitude, Altitude |
| **Genéticas** (3) | Cultivar_enc, Cultivar_freq, Cultivar_target |
| **Engineered** (5) | chill_gdd_ratio, temp_range, tree_age, lat_alt_interaction, Site_target |


### Recursos Adicionales

- [NASA POWER Project](https://power.larc.nasa.gov/)
- [Open-Meteo Weather API](https://open-meteo.com/)
- [FAO Guidelines for Phenology](http://www.fao.org/phenology)
- [scikit-learn Documentation](https://scikit-learn.org/)
- [Streamlit Documentation](https://docs.streamlit.io/)

---

## 👥 Autores

**[Eva María López Fernádez]**
- 🎓 Entrega final Bootcamp Data Analytics & IA - Upgrade
- 🌐 [Portfolio](https://evalopezf.github.io/portfolio-digital/)

### Agradecimientos

- **NASA POWER Project** por los datos climáticos de acceso abierto
- **Datos fenológicos** [A collection of European sweet cherry phenology data for assessing climate change](https://www.nature.com/articles/sdata2016108)

---




<div align="center">

### 🌸 *"De las semillas del conocimiento, florecen las mejores predicciones"* 🌸

**Desarrollado con ❤️ usando Python y Machine Learning**

[⬆ Volver arriba](#-cherry-blossom-bloom-prediction)

</div>
