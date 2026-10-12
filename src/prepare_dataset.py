"""Prepara el dataset Telco Customer Churn de forma reproducible.

Entrada : data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv  (Kaggle / IBM)
Salida  : data/processed/telco_churn_clean.csv   (dataset limpio, Churn 0/1)
          data/processed/split_ids.csv           (customerID -> train/val/test)
          data/diccionario_datos.csv             (descripcion de cada campo)

Uso: python src/prepare_dataset.py
"""
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
OUT = ROOT / "data" / "processed"

DICCIONARIO = [
    ("customerID", "texto", "Identificador unico del cliente (no se usa como variable predictora)."),
    ("gender", "categorica", "Genero (Female/Male). Se excluye del modelo y se usa solo para auditoria de sesgo."),
    ("SeniorCitizen", "binaria 0/1", "1 si el cliente es adulto mayor."),
    ("Partner", "Yes/No", "Tiene pareja."),
    ("Dependents", "Yes/No", "Tiene dependientes."),
    ("tenure", "numerica (meses)", "Meses de antiguedad como cliente."),
    ("PhoneService", "Yes/No", "Contrata servicio telefonico."),
    ("MultipleLines", "categorica", "Multiples lineas (Yes/No/No phone service)."),
    ("InternetService", "categorica", "Tipo de internet (DSL/Fiber optic/No)."),
    ("OnlineSecurity", "categorica", "Seguridad en linea (Yes/No/No internet service)."),
    ("OnlineBackup", "categorica", "Respaldo en linea."),
    ("DeviceProtection", "categorica", "Proteccion de dispositivo."),
    ("TechSupport", "categorica", "Soporte tecnico."),
    ("StreamingTV", "categorica", "Streaming de TV."),
    ("StreamingMovies", "categorica", "Streaming de peliculas."),
    ("Contract", "categorica", "Tipo de contrato (Month-to-month/One year/Two year)."),
    ("PaperlessBilling", "Yes/No", "Facturacion electronica."),
    ("PaymentMethod", "categorica", "Metodo de pago (4 categorias)."),
    ("MonthlyCharges", "numerica (USD)", "Cargo mensual."),
    ("TotalCharges", "numerica (USD)", "Cargo total acumulado. 11 registros vacios (tenure=0) se imputan con 0."),
    ("Churn", "objetivo 0/1", "1 = el cliente cancelo el servicio en el ultimo mes (Yes en el original)."),
]


def main():
    df = pd.read_csv(RAW)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].astype(str).str.strip(), errors="coerce")
    n_vacios = int(df["TotalCharges"].isna().sum())
    df["TotalCharges"] = df["TotalCharges"].fillna(0.0)  # clientes nuevos con tenure = 0
    df["Churn"] = (df["Churn"].str.strip() == "Yes").astype(int)
    assert df["customerID"].is_unique

    idx_tr, idx_tmp = train_test_split(df.index, test_size=0.30, stratify=df["Churn"], random_state=SEED)
    idx_va, idx_te = train_test_split(idx_tmp, test_size=0.50, stratify=df.loc[idx_tmp, "Churn"], random_state=SEED)
    split = pd.Series("train", index=df.index)
    split.loc[idx_va] = "val"
    split.loc[idx_te] = "test"

    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "telco_churn_clean.csv", index=False)
    pd.DataFrame({"customerID": df["customerID"], "split": split}).to_csv(OUT / "split_ids.csv", index=False)
    pd.DataFrame(DICCIONARIO, columns=["campo", "tipo", "descripcion"]).to_csv(
        ROOT / "data" / "diccionario_datos.csv", index=False, encoding="utf-8-sig")

    print(f"Filas: {len(df)} | TotalCharges vacios imputados: {n_vacios}")
    print(f"Tasa de churn: {df['Churn'].mean():.4f}")
    print(split.value_counts().to_string())
    print(df.groupby(split)["Churn"].mean().round(4).to_string())


if __name__ == "__main__":
    main()
