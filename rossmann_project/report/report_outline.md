# Tarea 3: Polars vs Pandas — Informe

> Plantilla. Llená cada sección con tus resultados reales (carpeta `results/` y `figures/`).
> Exportá a PDF (por ejemplo desde el notebook, VS Code, o `pandoc report_outline.md -o report.pdf`).

## 1. Descripción del dataset
- Fuente: Kaggle, Rossmann Store Sales.
- Tamaño: ___ filas, ___ columnas (train); ___ tiendas (store).
- Variable objetivo: `Sales` (regresión).
- Variables numéricas y categóricas: describir las principales.
- Valores faltantes detectados: (ver `results/eda_summary.json`).
- Decisiones de limpieza: filtro `Open==1 & Sales>0`, imputaciones aplicadas.

## 2. Pipeline implementado
Describir las 6 etapas: filtrado, join, manejo de faltantes, transformación,
nuevas características, agregación group_by. Incluir un fragmento corto de código
de cada etapa.

## 3. Resultados de Machine Learning
Tabla con los 3 modelos (de `results/ml_results.json`):

| Modelo            | Tiempo (s) | RMSE | MAE |
|-------------------|------------|------|-----|
| LinearRegression  |            |      |     |
| RandomForest      |            |      |     |
| GradientBoosting  |            |      |     |

Comentar cuál fue el mejor y por qué.

## 4. Tablas comparativas Polars vs Pandas
Tabla por etapa (de `results/benchmark_stages.json`):

| Etapa               | Polars (s) | Pandas (s) | Speedup |
|---------------------|------------|------------|---------|
| read                |            |            |         |
| filter              |            |            |         |
| join                |            |            |         |
| feature_engineering |            |            |         |
| aggregation         |            |            |         |
| total               |            |            |         |

Reportar: núcleos disponibles, RAM, tamaño del dataset (de `results/system_info.json`).

## 5. Gráficas
- `figures/stage_times.png` (tiempos por etapa).
- `figures/scalability.png` (escalabilidad 25/50/75/100%).
- `figures/speedup.png` (speedup vs tamaño).
- `figures/target_distribution.png`, `figures/correlation.png`.

## 6. Experimento de Lazy Execution
Comparar `read_csv` vs `scan_csv().collect()` (de `results/lazy_vs_eager.json`):
tiempo de ejecución, uso de memoria, complejidad del pipeline.

## 7. Discusión técnica y respuestas
Responder las 10 preguntas del enunciado (sección 8) con base en tus números:
1. Ventajas observadas con Polars.
2. Operaciones con mayor speedup.
3. Operaciones con diferencia pequeña.
4. Beneficios de Lazy Execution.
5. Limitaciones de Polars.
6. Ventajas que mantiene Pandas.
7. ¿Justifica migrar un proyecto existente?
8. Efecto del tamaño del dataset en el beneficio.
9. Mejor modelo predictivo.
10. Recomendaciones para proyectos futuros.

## 8. Conclusiones
Cierre con 3-4 ideas principales sustentadas en los datos.
