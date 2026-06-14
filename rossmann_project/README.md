# Tarea 3: Polars vs Pandas — Pipeline de Datos y Machine Learning

## Descripción del problema
Se construye un pipeline completo de análisis de datos y aprendizaje automático
sobre el dataset **Rossmann Store Sales**. El objetivo es un problema de **regresión**:
predecir las ventas diarias (`Sales`) de las tiendas a partir de variables de calendario,
promociones, competencia y características de cada tienda. Se compara sistemáticamente
el rendimiento de **Polars** y **Pandas** en cada etapa del procesamiento.

## Fuente del dataset
- Kaggle: *Rossmann Store Sales* (`rossmann-store-sales`).
- `train.csv`: ~1.017.000 registros (ventas diarias por tienda).
- `store.csv`: metadatos de 1.115 tiendas (se usa para el join requerido).
- Cumple los requisitos: >100.000 registros, problema de regresión, variables
  numéricas y categóricas, disponibilidad pública.

## Requisitos de software
Ver `requirements.txt`. Resumen: Python 3.10+, polars, pandas, numpy, scikit-learn,
matplotlib, psutil, pyarrow.

## Instrucciones de instalación
```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Obtención del dataset
Con la API de Kaggle configurada (`~/.kaggle/kaggle.json`):
```
python download_data.py
```
Esto descarga y descomprime `train.csv` y `store.csv` en `data/raw/`.
Alternativa de prueba sin Kaggle: `python make_synthetic.py` genera datos con el
mismo esquema (no usar para la entrega final, solo para verificar que el código corre).

## Instrucciones de ejecución
```
python src/main.py
```
Genera:
- Figuras en `figures/` (distribución del objetivo, correlación, tiempos por etapa,
  escalabilidad, speedup).
- Resultados en `results/` (system_info, eda_summary, ml_results, benchmark_stages,
  scalability, lazy_vs_eager).

El notebook `notebooks/analysis.ipynb` ejecuta el mismo flujo de forma interactiva.

## Estructura del repositorio
```
project/
  data/raw/             datasets originales
  data/processed/       datos transformados
  notebooks/            analysis.ipynb
  src/
    preprocessing.py        loaders polars/pandas
    feature_engineering.py  transformaciones en Polars
    polars_pipeline.py      pipeline Polars con tiempos por etapa + lazy
    pandas_pipeline.py      pipeline equivalente en Pandas
    train_models.py         entrenamiento de los 3 modelos
    benchmark.py            comparación Polars vs Pandas + info del sistema
    experiments.py          escalabilidad y lazy vs eager
    eda.py                  análisis exploratorio
    main.py                 orquestador
  figures/
  results/
  report/report.pdf
  requirements.txt
  README.md
```

## Pipeline implementado
1. Filtrado: `Open == 1` y `Sales > 0`.
2. Join: `train` ⨝ `store` por `Store`.
3. Manejo de faltantes: `CompetitionDistance` con la mediana; columnas de competencia
   y promo2 con 0; `PromoInterval` con `"None"`.
4. Transformación: `Date` → `Year/Month/Day/WeekOfYear`; codificación de
   `StateHoliday`, `StoreType`, `Assortment`.
5. Nuevas características: `CompetitionOpenMonths`, `IsPromoMonth`.
6. Agregación `group_by` + join de vuelta: `AvgStoreSales`, `AvgStoreCustomers` por tienda.

## Modelos
- Regresión Lineal (con `StandardScaler`).
- Random Forest.
- Gradient Boosting (`HistGradientBoostingRegressor`; se puede sustituir por XGBoost/LightGBM).
Métricas reportadas: RMSE y MAE.

## Resumen de resultados
Completar con los valores reales de tu corrida (ver `results/`):

| Modelo            | Tiempo entrenamiento | RMSE | MAE |
|-------------------|----------------------|------|-----|
| LinearRegression  |                      |      |     |
| RandomForest      |                      |      |     |
| GradientBoosting  |                      |      |     |

Benchmark global (de `results/benchmark_stages.json`): el mayor speedup de Polars
se observa en lectura y feature engineering; las agregaciones simples muestran
diferencias menores. Los tiempos exactos dependen del hardware (núcleos y RAM).

## Notas técnicas
- `AvgStoreSales`/`AvgStoreCustomers` se calculan sobre todo el conjunto; en un
  entrenamiento riguroso deberían calcularse solo sobre el fold de entrenamiento
  para evitar fuga de información.
- La medición de memoria (`lazy_vs_eager`) usa el RSS del proceso y es aproximada.
