# Tarea 3: Polars vs Pandas — Pipeline de Datos y Machine Learning

## Descripción del problema
Se construye un pipeline completo de análisis de datos y aprendizaje automático sobre el dataset **Rossmann Store Sales**. El objetivo es un problema de **regresión**: predecir las ventas diarias (`Sales`) de las tiendas a partir de variables de calendario, promociones, competencia y características de cada tienda. Se compara sistemáticamente el rendimiento de **Polars** y **Pandas** en cada etapa del procesamiento.

## Fuente del dataset
- Kaggle: *Rossmann Store Sales* (`rossmann-store-sales`).
- `train.csv`: 1.017.209 registros (ventas diarias por tienda).
- `store.csv`: metadatos de 1.115 tiendas (se usa para el join requerido).
- Cumple los requisitos: más de 100.000 registros, problema de regresión, variables numéricas y categóricas, disponibilidad pública.
- Por restricciones de acceso a la API de Kaggle, se descargó desde un repositorio espejo público que contiene los archivos originales sin modificación.

## Requisitos de software
Ver `requirements.txt`. Resumen: Python 3.10+, polars, pandas, numpy, scikit-learn, matplotlib, psutil, pyarrow.

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

## Instrucciones de ejecución
```
python src/main.py
```
Genera las figuras en `figures/` y los resultados en `results/`. El notebook `notebooks/analysis.ipynb` ejecuta el mismo flujo de forma interactiva.

## Estructura del repositorio
```
rossmann_project/
  data/raw/             datasets originales
  data/processed/       datos transformados
  notebooks/            analysis.ipynb
  src/
    preprocessing.py        loaders polars/pandas
    feature_engineering.py  transformaciones en Polars
    polars_pipeline.py      pipeline Polars con tiempos por etapa + lazy
    pandas_pipeline.py      pipeline equivalente en Pandas
    train_models.py         entrenamiento de los 3 modelos
    benchmark.py            comparacion Polars vs Pandas + info del sistema
    experiments.py          escalabilidad y lazy vs eager
    eda.py                  analisis exploratorio
    main.py                 orquestador
  figures/
  results/
  report/
  requirements.txt
  README.md
```

## Pipeline implementado
1. Filtrado: `Open == 1` y `Sales > 0`.
2. Join: `train` con `store` por `Store`.
3. Manejo de faltantes: `CompetitionDistance` con la mediana; columnas de competencia y promo2 con 0; `PromoInterval` con `"None"`.
4. Transformacion: `Date` a `Year/Month/Day/WeekOfYear`; codificacion de `StateHoliday`, `StoreType` y `Assortment`.
5. Nuevas caracteristicas: `CompetitionOpenMonths`, `IsPromoMonth`.
6. Agregacion `group_by` + join de vuelta: `AvgStoreSales`, `AvgStoreCustomers` por tienda.

## Modelos
- Regresion Lineal (con `StandardScaler`).
- Random Forest.
- Gradient Boosting (`HistGradientBoostingRegressor`).
Metricas reportadas: RMSE y MAE.

## Resumen de resultados

Información del sistema: 10 núcleos, 17.18 GB de RAM, 1.017.209 filas (38.06 MB).

### Modelos

| Modelo            | Tiempo entrenamiento (s) | RMSE    | MAE     |
|-------------------|--------------------------|---------|---------|
| LinearRegression  | 0.18                     | 1540.95 | 1102.87 |
| RandomForest      | 18.50                    | 835.83  | 577.57  |
| GradientBoosting  | 2.62                     | 966.64  | 683.39  |

El mejor modelo fue Random Forest (menor RMSE y MAE).

### Benchmark Polars vs Pandas (por etapa)

| Etapa               | Polars (s) | Pandas (s) | Speedup |
|---------------------|------------|------------|---------|
| read                | 0.0196     | 0.2698     | 13.74x  |
| filter              | 0.0031     | 0.0346     | 11.28x  |
| join                | 0.0081     | 0.0265     | 3.26x   |
| feature_engineering | 0.1072     | 0.6218     | 5.80x   |
| aggregation         | 0.0058     | 0.0216     | 3.75x   |
| total               | 0.1437     | 0.9742     | 6.78x   |

### Escalabilidad

| % del dataset | Polars (s) | Pandas (s) | Speedup |
|---------------|------------|------------|---------|
| 25%           | 0.0513     | 0.4661     | 9.09x   |
| 50%           | 0.0805     | 0.6498     | 8.07x   |
| 75%           | 0.1116     | 0.8189     | 7.34x   |
| 100%          | 0.1258     | 0.9186     | 7.30x   |

Polars fue 6.78x mas rapido que Pandas en el pipeline completo. El mayor speedup se observa en la lectura del CSV y el filtrado; las menores diferencias en el join y la agregacion.

## Notas técnicas
- `AvgStoreSales`/`AvgStoreCustomers` se calculan sobre todo el conjunto; en un entrenamiento riguroso deberian calcularse solo sobre el fold de entrenamiento para evitar fuga de informacion.
- La medicion de memoria (`lazy_vs_eager`) usa el RSS del proceso y es aproximada.