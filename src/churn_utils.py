"""Utilidades compartidas del proyecto de churn (Actividad 4, Temas 9 y 10).

Contiene: carga/split de datos, serializacion de filas a texto, metricas
(accuracy, precision, recall, F1, AUC, precision@k, IC bootstrap), medicion de
latencia, auditoria de sesgo y funciones para correr los modelos (baselines,
TabPFN y encoders de texto de Hugging Face). Los imports pesados (torch,
transformers, tabpfn) se hacen dentro de las funciones.
"""
import time
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, average_precision_score, precision_recall_curve)

SEED = 42
TARGET = "Churn"
ID_COL = "customerID"
SENSIBLES = ["gender", "SeniorCitizen"]          # solo auditoria (gender no se usa como predictora)
EXCLUIR = [ID_COL, TARGET, "gender"]
NUMERICAS = ["tenure", "MonthlyCharges", "TotalCharges"]
TOP_K = 0.15                                      # Fase I: campana sobre el 15% de mayor riesgo
UMBRAL_PRECISION_TOPK = 0.45                      # Fase I: precision minima en el top 15%


# ----------------------------------------------------------------------------- datos
def cargar_datos(ruta_clean, ruta_split):
    """Devuelve df limpio con columna 'split' (train/val/test)."""
    df = pd.read_csv(ruta_clean)
    sp = pd.read_csv(ruta_split)
    df = df.merge(sp, on=ID_COL, how="left", validate="one_to_one")
    assert df["split"].notna().all()
    return df


def columnas_modelo(df):
    feats = [c for c in df.columns if c not in EXCLUIR + ["split"]]
    cats = [c for c in feats if c not in NUMERICAS and c != "SeniorCitizen"]
    return feats, cats


def serializar_fila(r):
    """Convierte un cliente en una frase en ingles (los encoders son ingles/uncased)."""
    return (
        f"Customer with {int(r['tenure'])} months of tenure. "
        f"Monthly charges {r['MonthlyCharges']:.2f}, total charges {r['TotalCharges']:.2f}. "
        f"Contract: {r['Contract']}. Internet: {r['InternetService']}. Phone service: {r['PhoneService']}. "
        f"Multiple lines: {r['MultipleLines']}. Online security: {r['OnlineSecurity']}. "
        f"Online backup: {r['OnlineBackup']}. Device protection: {r['DeviceProtection']}. "
        f"Tech support: {r['TechSupport']}. Streaming TV: {r['StreamingTV']}. "
        f"Streaming movies: {r['StreamingMovies']}. Paperless billing: {r['PaperlessBilling']}. "
        f"Payment method: {r['PaymentMethod']}. Senior citizen: {'Yes' if r['SeniorCitizen'] == 1 else 'No'}. "
        f"Partner: {r['Partner']}. Dependents: {r['Dependents']}."
    )


# ----------------------------------------------------------------------------- metricas
def precision_recall_topk(y, p, frac=TOP_K):
    y = np.asarray(y); p = np.asarray(p)
    k = max(1, int(round(len(y) * frac)))
    idx = np.argsort(-p, kind="stable")[:k]
    prec = y[idx].mean()
    rec = y[idx].sum() / max(1, y.sum())
    lift = prec / y.mean()
    return float(prec), float(rec), float(lift)


def mejor_umbral_f1(y, p):
    """Umbral que maximiza F1 de la clase positiva (se calcula en VALIDACION)."""
    pr, rc, th = precision_recall_curve(y, p)
    f1 = 2 * pr[:-1] * rc[:-1] / np.clip(pr[:-1] + rc[:-1], 1e-12, None)
    return float(th[int(np.nanargmax(f1))])


def evaluar(y, p, umbral_val=None):
    """Metricas de la clase positiva (Churn=1) a umbral 0.5, mas ranking y top-k."""
    y = np.asarray(y); p = np.asarray(p)
    yhat = (p >= 0.5).astype(int)
    pk, rk, lk = precision_recall_topk(y, p)
    out = {
        "accuracy": accuracy_score(y, yhat),
        "precision": precision_score(y, yhat, zero_division=0),
        "recall": recall_score(y, yhat, zero_division=0),
        "f1": f1_score(y, yhat, zero_division=0),
        "roc_auc": roc_auc_score(y, p),
        "pr_auc": average_precision_score(y, p),
        "precision_top15": pk, "recall_top15": rk, "lift_top15": lk,
    }
    if umbral_val is not None:
        yv = (p >= umbral_val).astype(int)
        out["umbral_val"] = umbral_val
        out["f1_umbral_val"] = f1_score(y, yv, zero_division=0)
        out["recall_umbral_val"] = recall_score(y, yv, zero_division=0)
        out["precision_umbral_val"] = precision_score(y, yv, zero_division=0)
    return out


def ic_bootstrap(y, p, fn, n=1000, seed=SEED):
    """IC 95% por bootstrap para una metrica fn(y, p)."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y); p = np.asarray(p)
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        vals.append(fn(y[i], p[i]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


# ----------------------------------------------------------------------------- latencia
def medir_latencia(predict_fn, X_lote, X_unitarios, n_unit=30, sync=None, n_clientes_prod=400_000):
    """predict_fn(chunk)->probabilidades. Mide lote completo y cliente individual."""
    sync = sync or (lambda: None)
    predict_fn(X_unitarios[:1])  # calentamiento
    sync()
    t0 = time.perf_counter(); predict_fn(X_lote); sync()
    t_lote = time.perf_counter() - t0
    tiempos = []
    for i in range(min(n_unit, len(X_unitarios))):
        t0 = time.perf_counter(); predict_fn(X_unitarios[i:i + 1]); sync()
        tiempos.append((time.perf_counter() - t0) * 1000)
    ms_cliente_lote = t_lote / len(X_lote) * 1000
    return {
        "lote_seg": t_lote,
        "ms_por_cliente_lote": ms_cliente_lote,
        "ms_unitario_mediana": float(np.median(tiempos)),
        "ms_unitario_p95": float(np.percentile(tiempos, 95)),
        "min_estimados_400k": ms_cliente_lote * n_clientes_prod / 1000 / 60,
    }


# ----------------------------------------------------------------------------- auditoria de sesgo
def auditar_sesgo(df_test, p, col, frac=TOP_K):
    """Compara por grupo: tasa real de churn, tasa de seleccion en el top-k global, precision y recall."""
    d = df_test[[col, TARGET]].copy()
    d["p"] = np.asarray(p)
    k = int(round(len(d) * frac))
    d["sel"] = 0
    d.iloc[np.argsort(-d["p"].values, kind="stable")[:k], d.columns.get_loc("sel")] = 1
    filas = []
    for g, s in d.groupby(col):
        sel = s[s["sel"] == 1]
        filas.append({
            "variable": col, "grupo": g, "n": len(s), "churn_real": s[TARGET].mean(),
            "tasa_seleccion_top15": s["sel"].mean(),
            "precision_top15": sel[TARGET].mean() if len(sel) else np.nan,
            "recall_top15": sel[TARGET].sum() / max(1, s[TARGET].sum()),
        })
    return pd.DataFrame(filas)


# ----------------------------------------------------------------------------- modelos clasicos (referencia)
def construir_baselines(feats, cats):
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import HistGradientBoostingClassifier
    nums = [c for c in feats if c not in cats]
    prep_lr = ColumnTransformer([("n", StandardScaler(), nums),
                                 ("c", OneHotEncoder(handle_unknown="ignore"), cats)])
    prep_gb = ColumnTransformer([("n", "passthrough", nums),
                                 ("c", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cats)])
    return {
        "Regresión logística": Pipeline([("prep", prep_lr), ("clf", LogisticRegression(max_iter=1000, random_state=SEED))]),
        "HistGradientBoosting": Pipeline([("prep", prep_gb), ("clf", HistGradientBoostingClassifier(random_state=SEED))]),
    }


def regla_heuristica(df):
    """Linea base de Fase I (reglas por antiguedad y tipo de contrato). Devuelve un puntaje 0-1."""
    return 0.6 * (df["Contract"] == "Month-to-month").astype(float) + 0.4 * (1 - df["tenure"].clip(upper=72) / 72)


# ----------------------------------------------------------------------------- TabPFN (Hugging Face)
def correr_tabpfn(Xtr, ytr, feats, cats, model_path="auto"):
    """Entrena (en contexto) TabPFN. Requiere aceptar la licencia en el Hub y HF_TOKEN."""
    import torch
    from sklearn.preprocessing import OrdinalEncoder
    from tabpfn import TabPFNClassifier
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    def prep(X):
        Z = X[feats].copy()
        Z[cats] = enc.transform(Z[cats])
        return Z.astype(float).values
    enc.fit(Xtr[cats])
    cat_idx = [feats.index(c) for c in cats]
    clf = TabPFNClassifier(categorical_features_indices=cat_idx, random_state=SEED,
                           **({} if model_path == "auto" else {"model_path": model_path}))
    t0 = time.perf_counter()
    clf.fit(prep(Xtr), np.asarray(ytr))
    t_fit = time.perf_counter() - t0
    params = None
    try:
        m = getattr(clf, "model_", None) or getattr(clf, "models_", [None])[0]
        params = sum(p.numel() for p in m.parameters())
    except Exception:
        pass
    return (lambda X: clf.predict_proba(prep(X))[:, 1]), {"t_fit": t_fit, "params": params}


# ----------------------------------------------------------------------------- encoders de texto (Hugging Face)
def entrenar_encoder_texto(model_id, txt_tr, y_tr, txt_va, y_va, epochs=3, lr=3e-5, bs=32,
                           max_len=192, seed=SEED, verbose=True):
    """Fine-tuning minimo (bucle PyTorch simple, sin Trainer) de un encoder preentrenado.
    Conserva el epoch con mejor AUC en validacion. Devuelve (predict_fn, info)."""
    import torch, copy
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
    torch.manual_seed(seed); np.random.seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=2).to(dev)
    n_params = sum(p.numel() for p in model.parameters())

    def batches(txts, y=None, shuffle=False):
        idx = np.random.permutation(len(txts)) if shuffle else np.arange(len(txts))
        for i in range(0, len(idx), bs):
            b = idx[i:i + bs]
            enc = tok([txts[j] for j in b], truncation=True, max_length=max_len, padding=True, return_tensors="pt")
            enc = {k: v.to(dev) for k, v in enc.items()}
            yield enc, (torch.tensor(np.asarray(y)[b]).to(dev) if y is not None else None)

    @torch.no_grad()
    def predict(txts):
        model.eval(); out = []
        for enc, _ in batches(list(txts)):
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(dev == "cuda")):
                lg = model(**enc).logits
            out.append(torch.softmax(lg.float(), -1)[:, 1].cpu().numpy())
        return np.concatenate(out)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    steps = epochs * int(np.ceil(len(txt_tr) / bs))
    sch = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
    scaler = torch.amp.GradScaler("cuda", enabled=(dev == "cuda"))
    if dev == "cuda":
        torch.cuda.reset_peak_memory_stats()
    best_auc, best_state, hist = -1, None, []
    t0 = time.perf_counter()
    for ep in range(1, epochs + 1):
        model.train(); tot = 0
        for enc, yb in batches(list(txt_tr), y_tr, shuffle=True):
            opt.zero_grad()
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(dev == "cuda")):
                loss = torch.nn.functional.cross_entropy(model(**enc).logits.float(), yb)
            scaler.scale(loss).backward()
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sch.step()
            tot += loss.item() * len(yb)
        auc = roc_auc_score(y_va, predict(txt_va))
        hist.append({"epoch": ep, "loss_train": tot / len(txt_tr), "auc_val": auc})
        if verbose:
            print(f"  [{model_id}] epoch {ep}/{epochs} loss={tot/len(txt_tr):.4f} AUC val={auc:.4f}")
        if auc > best_auc:
            best_auc = auc; best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
    t_train = time.perf_counter() - t0
    model.load_state_dict(best_state)
    info = {"params": n_params, "t_train": t_train, "hist": hist, "auc_val_mejor": best_auc,
            "gpu_mem_mb": (torch.cuda.max_memory_allocated() / 2**20) if dev == "cuda" else None,
            "device": dev, "sync": (torch.cuda.synchronize if dev == "cuda" else (lambda: None)),
            "model": model, "tok": tok}
    return predict, info
