# Ficha del dataset: Telco Customer Churn

## Origen
- **Nombre:** Telco Customer Churn (IBM Sample Data Sets).
- **Publicación consultada:** Kaggle, `blastchar/telco-customer-churn`
  (https://www.kaggle.com/datasets/blastchar/telco-customer-churn).
- **Descripción oficial:** cada fila es un cliente; incluye clientes que se fueron en el último mes (`Churn`),
  servicios contratados, información de cuenta y datos demográficos.
- **Naturaleza:** conjunto de datos de muestra (no son clientes reales de *Sierra Móvil*). Revisa los términos de uso en la página de origen.
- **Archivo original:** `data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv` (7,043 filas × 21 columnas, sin modificar).

## Cómo se genera la versión de trabajo
```bash
python src/prepare_dataset.py
```
| Paso | Detalle |
|---|---|
| `TotalCharges` | Se convierte a numérico. 11 registros vacíos (todos con `tenure = 0`, clientes nuevos) se imputan con 0. |
| `Churn` | `Yes` → 1, `No` → 0. Tasa de churn: 26.54 % (1,869 de 7,043). |
| Partición | Estratificada por `Churn`, semilla 42: train 4,930 (70 %), val 1,056 (15 %), test 1,057 (15 %). |
| `gender` | Se excluye de las variables predictoras; solo se usa para auditar sesgo. |
| `customerID` | Identificador; no se usa como variable. Sirve para unir la partición (`split_ids.csv`). |

Salidas: `data/processed/telco_churn_clean.csv`, `data/processed/split_ids.csv`, `data/diccionario_datos.csv`.

## Estructura (resumen; ver `diccionario_datos.csv`)
| Grupo | Campos |
|---|---|
| Demografía | `gender`, `SeniorCitizen`, `Partner`, `Dependents` |
| Cuenta | `tenure`, `Contract`, `PaperlessBilling`, `PaymentMethod`, `MonthlyCharges`, `TotalCharges` |
| Servicios | `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies` |
| Objetivo | `Churn` |

## Limitaciones
- El **horizonte de predicción** es de un mes (el churn corresponde al último mes), consistente con la lista mensual de la Fase I.
- No hay fechas: no es posible una validación temporal ni medir deriva real.
- La tasa de churn de la muestra (26.5 %) no equivale al 22 % anual supuesto en la Fase I.
- Contiene variables demográficas (`gender`, `SeniorCitizen`): requieren auditoría de sesgo y revisión legal antes de cualquier uso productivo.
