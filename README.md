# Predicción de churn en telecomunicaciones con modelos preentrenados de Hugging Face

Proyecto *Gestión de proyectos de inteligencia artificial* · Actividad 4 (Temas 9 y 10) · Continuación de la Fase I (*Sierra Móvil*).

## Título y descripción
**Objetivo.** Comparar al menos tres modelos preentrenados de Hugging Face (más modelos de referencia) para identificar cada mes
al 15 % de clientes con mayor riesgo de baja, y justificar con evidencia cuál conviene según precisión, latencia, costo y licencia.

**Contexto.** En la Fase I se propuso un modelo de churn con un KPI de **precisión ≥ 45 % en el 15 % de mayor riesgo**.
Esta actividad mide ese supuesto con el dataset *Telco Customer Churn* (IBM/Kaggle, 7,043 clientes) y prepara el prototipo para la Fase II (contenedor Docker + API).

**Enfoque de IA.** Clasificación supervisada binaria sobre datos tabulares. Horizonte de predicción: 1 mes.

## Modelos comparados
| Modelo | Id en Hugging Face | Enfoque |
|---|---|---|
| TabPFN-3 | `Prior-Labs/tabpfn_3` | Modelo fundacional tabular, sin entrenamiento (acceso restringido) |
| BERT-mini | `google/bert_uncased_L-4_H-256_A-4` | Encoder de texto + ajuste fino mínimo |
| DistilBERT | `distilbert-base-uncased` | Encoder de texto + ajuste fino mínimo |
| RoBERTa-base | `roberta-base` | Encoder de texto + ajuste fino mínimo |
| *Referencias* | – | Regla heurística de Fase I, regresión logística, HistGradientBoosting |

## Tecnologías utilizadas
Python 3.12+ · pandas · scikit-learn · PyTorch · Transformers · Datasets · huggingface_hub · TabPFN · matplotlib · Google Colab (GPU T4).
Versiones exactas de cada ejecución: `results/entorno.json`.

## Estructura del proyecto
```
├── README.md
├── requirements.txt
├── data/
│   ├── raw/WA_Fn-UseC_-Telco-Customer-Churn.csv   # original (Kaggle/IBM)
│   ├── processed/telco_churn_clean.csv            # limpio, Churn 0/1
│   ├── processed/split_ids.csv                    # customerID -> train/val/test
│   ├── diccionario_datos.csv                      # descripción de cada campo
│   └── DATA_CARD.md                               # origen, estructura y limitaciones
├── notebooks/PLN_Act4_Churn_HF_Comparativo.ipynb  # cuaderno de Colab (ejecutar de arriba a abajo)
├── src/
│   ├── prepare_dataset.py                         # limpieza + partición reproducible
│   └── churn_utils.py                             # métricas, latencia, auditoría de sesgo, modelos
└── results/                                       # métricas, decisión, gráficas y predicciones generadas
```

## Instrucciones de uso
1. Abre `notebooks/PLN_Act4_Churn_HF_Comparativo.ipynb` en Google Colab y activa **GPU T4**.
2. Crea un token **Read** en Hugging Face y guárdalo como Secret `HF_TOKEN` en Colab. Para TabPFN-3 acepta la licencia en
   https://huggingface.co/Prior-Labs/tabpfn_3 (su uso en producción requiere licencia comercial).
3. Sube el repositorio a Colab (o define `REPO_URL` en la celda 4) o sube el CSV cuando el cuaderno lo pida.
4. Ejecuta todas las celdas (≈ 15–25 min). Las tablas y figuras quedan en `results/`; se descarga `resultados_actividad4.zip`.
5. Solo la preparación de datos (sin GPU): `python src/prepare_dataset.py`.

Si un modelo no puede ejecutarse (p. ej. no se aceptó la licencia de TabPFN-3), el cuaderno lo registra en `results/estado_modelos.csv` y continúa.

## Resultados
Conjunto de prueba de 1,057 clientes (partición estratificada 70/15/15, semilla 42). KPI de la Fase I: precisión ≥ 45 % en el 15 % de clientes con mayor riesgo.

| Modelo | ROC-AUC (IC 95 %) | Precisión top 15 % | F1 (umbral en val.) | ms/cliente (lote) | Min. para 400,000 | Licencia |
|---|---|---|---|---|---|---|
| Regresión logística | 0.845 (0.819–0.868) | 69.2% | 0.619 | 0.009 | 0.06 | BSD-3 |
| HistGradientBoosting | 0.823 (0.794–0.849) | 67.3% | 0.597 | 0.019 | 0.12 | BSD-3 |
| TabPFN-3 | 0.850 (0.823–0.872) | 71.1% | 0.633 | 5.245 | 34.97 | Propia (producción requiere licencia comercial) |
| BERT-mini | 0.834 (0.806–0.859) | 67.9% | 0.617 | 0.532 | 3.55 | Apache-2.0 |
| DistilBERT-base | 0.840 (0.811–0.864) | 71.7% | 0.614 | 2.820 | 18.80 | Apache-2.0 |
| RoBERTa-base | 0.831 (0.803–0.855) | 66.0% | 0.609 | 1.728 | 11.52 | MIT |
| Regla heurística Fase I | 0.786 | 60.4% | — | — | — | — |

Detalle completo en `results/metricas_comparativas.csv` y figuras `results/fig_*.png`.

## Conclusiones
* Todos los modelos superan el KPI de la Fase I (precisión top 15 % ≥ 45 %) y el umbral de ROC-AUC ≥ 0.80.
* Ningún modelo preentrenado supera de forma significativa a la regresión logística: los intervalos de confianza se traslapan (ROC-AUC de 0.823 a 0.850). La elección se decide por costo, licencia y facilidad de despliegue.
* **Modelo preentrenado recomendado: BERT-mini** (`google/bert_uncased_L-4_H-256_A-4`, Apache-2.0): ROC-AUC 0.834, precisión top 15 % de 67.9 %, 0.53 ms por cliente en lote (≈ 3.5 min para 400,000 clientes) y 1.2 GB de GPU.
* TabPFN-3 tiene el mayor ROC-AUC (0.850) pero su licencia no permite producción sin licencia comercial y es el más lento (≈ 4.7 s por predicción individual); se conserva como referencia de evaluación.
* La regresión logística es la línea base de costo mínimo y se mantiene como referencia para la Fase II y el piloto.

## Limitaciones y riesgos
- Dataset de muestra: no sustituye datos propios de la operadora. Validar con un piloto y grupo de control (Fase I).
- Los encoders de texto no fueron diseñados para datos tabulares; se incluyen para contrastar popularidad vs. adecuación.
- Licencia de TabPFN-3: restringida para producción.
- Variables demográficas: `gender` se excluye de las variables predictoras; se recomienda auditar el desempeño por grupos antes de producción.

## Siguiente paso (Fase II)
Contenerizar con Docker el modelo seleccionado y exponerlo mediante una API (p. ej. FastAPI, endpoint `/predict`), reutilizando `src/churn_utils.py`.

## Contacto / créditos
Autor: Emilio · Asignatura: Gestión de Proyectos de Inteligencia Artificial (Tecmilenio) · Octubre 2026 · Dataset: IBM Sample Data Sets (Kaggle, `blastchar`).
