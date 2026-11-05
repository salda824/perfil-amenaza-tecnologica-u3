"""Métricas, incertidumbre, pruebas pareadas, esquemas de validación y control de deriva (PSI)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, brier_score_loss, confusion_matrix,
                             f1_score, matthews_corrcoef, precision_score, recall_score, roc_auc_score,
                             average_precision_score)
from sklearn.model_selection import GroupKFold

from . import config as C


def bce(y, p, eps=1e-7):
    y = np.asarray(y, float); p = np.clip(np.asarray(p, float), eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def ece(y, p, n_bins=5):
    """Error de calibración esperado con bins de igual frecuencia."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    cortes = np.quantile(p, np.linspace(0, 1, n_bins + 1)); cortes[-1] += 1e-9
    idx = np.clip(np.searchsorted(cortes, p, side="right") - 1, 0, n_bins - 1)
    return float(sum(np.abs(y[idx == b].mean() - p[idx == b].mean()) * np.mean(idx == b)
                     for b in range(n_bins) if np.any(idx == b)))


def metricas(y, p, umbral=0.5, transicion=None):
    """Batería para la clase positiva C (1) frente a D (0). `transicion` marca filas que cambian de perfil."""
    y = np.asarray(y).astype(int); p = np.asarray(p, float); yhat = (p >= umbral).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
    dos = len(np.unique(y)) == 2
    m = {"Exactitud": accuracy_score(y, yhat), "Exact. balanceada": balanced_accuracy_score(y, yhat),
         "Precisión C": precision_score(y, yhat, zero_division=0), "Sensibilidad C": recall_score(y, yhat, zero_division=0),
         "Especificidad D": tn / (tn + fp) if (tn + fp) else np.nan, "F1 C": f1_score(y, yhat, zero_division=0),
         "MCC": matthews_corrcoef(y, yhat), "ROC-AUC": roc_auc_score(y, p) if dos else np.nan,
         "PR-AUC": average_precision_score(y, p) if dos else np.nan, "Log-loss": bce(y, p),
         "Brier": brier_score_loss(y, p), "ECE": ece(y, p)}
    if transicion is not None:
        t = np.asarray(transicion, bool)
        m["Transiciones acertadas"] = int(((yhat == y) & t).sum())
        m["Transiciones"] = int(t.sum())
    return m


COLS_PRINCIPALES = ["ROC-AUC", "Log-loss", "Brier", "MCC", "Exact. balanceada", "ECE"]


def bootstrap_ic(y, p, fn, n=2000, semilla=C.SEED):
    """IC 95 % percentil por bootstrap estratificado por clase."""
    rng = np.random.default_rng(semilla); y = np.asarray(y).astype(int); p = np.asarray(p, float)
    i0, i1 = np.where(y == 0)[0], np.where(y == 1)[0]
    vals = [fn(y[idx], p[idx]) for idx in (np.concatenate([rng.choice(i0, len(i0)), rng.choice(i1, len(i1))])
                                           for _ in range(n))]
    return np.percentile(vals, [2.5, 97.5])


def comparar_pareado(y, pa, pb, n_boot=2000, semilla=C.SEED, grupos=None):
    """
    Diferencias pareadas A − B (ΔAUC, Δlog-loss, ΔBrier) con IC 95 % por bootstrap y McNemar exacta sobre aciertos.
    Si se pasan `grupos` (departamento), el bootstrap remuestrea departamentos completos (respeta la dependencia
    entre años de un mismo territorio).
    """
    y = np.asarray(y).astype(int); pa, pb = np.asarray(pa, float), np.asarray(pb, float)
    ok_a, ok_b = (pa >= .5) == y, (pb >= .5) == y
    n01, n10 = int(np.sum(ok_a & ~ok_b)), int(np.sum(~ok_a & ok_b))
    p_mc = stats.binomtest(min(n01, n10), n01 + n10, 0.5).pvalue if (n01 + n10) else 1.0
    rng = np.random.default_rng(semilla)
    if grupos is not None:
        g = np.asarray(grupos); unicos = np.unique(g); pos = {u: np.where(g == u)[0] for u in unicos}
    d = {"ΔAUC": [], "ΔLog-loss": [], "ΔBrier": []}
    for _ in range(n_boot):
        i = (np.concatenate([pos[u] for u in rng.choice(unicos, len(unicos))]) if grupos is not None
             else rng.integers(0, len(y), len(y)))
        if len(np.unique(y[i])) < 2:
            continue
        d["ΔAUC"].append(roc_auc_score(y[i], pa[i]) - roc_auc_score(y[i], pb[i]))
        d["ΔLog-loss"].append(bce(y[i], pa[i]) - bce(y[i], pb[i]))
        d["ΔBrier"].append(np.mean((pa[i] - y[i]) ** 2) - np.mean((pb[i] - y[i]) ** 2))
    punto = {"ΔAUC": roc_auc_score(y, pa) - roc_auc_score(y, pb), "ΔLog-loss": bce(y, pa) - bce(y, pb),
             "ΔBrier": np.mean((pa - y) ** 2) - np.mean((pb - y) ** 2)}
    out = {"n": len(y), "solo A acierta": n01, "solo B acierta": n10, "p McNemar": p_mc}
    for k, v in d.items():
        lo, hi = np.percentile(v, [2.5, 97.5])
        out[k] = punto[k]; out[f"IC95 {k}"] = f"[{lo:+.3f}; {hi:+.3f}]"
        out[f"P({k} mejora)"] = float(np.mean(np.array(v) > 0)) if k == "ΔAUC" else float(np.mean(np.array(v) < 0))
    return out


# ----------------------------------------------------------------------------------------------------------
# Esquemas de validación
# ----------------------------------------------------------------------------------------------------------
def particiones_origen_movil(datos: pd.DataFrame, anios=C.ANIOS_SELECCION):
    """Para cada año v: entrena con t < v, predice t = v. Nunca usa información posterior a v-1."""
    for v in anios:
        yield str(v), (datos["anio"] < v).values, (datos["anio"] == v).values


def particiones_espaciales(datos: pd.DataFrame, k=5, anios_max=max(C.ANIOS_SELECCION)):
    """GroupKFold por departamento sobre t <= anios_max: los departamentos de prueba nunca se ven."""
    sub = (datos["anio"] <= anios_max).values
    idx = np.where(sub)[0]
    for j, (a, b) in enumerate(GroupKFold(n_splits=k).split(idx, groups=datos.loc[sub, "depto"])):
        tr, te = np.zeros(len(datos), bool), np.zeros(len(datos), bool)
        tr[idx[a]], te[idx[b]] = True, True
        yield f"pliegue {j + 1}", tr, te


def validar(datos: pd.DataFrame, ajustar_predecir, particiones) -> pd.DataFrame:
    """
    Ejecuta `ajustar_predecir(tr, te) -> dict{nombre: probas}` en cada partición y devuelve las predicciones
    fuera de muestra en formato largo (una fila por departamento-año y columna por modelo).
    """
    bloques = []
    for nombre, tr, te in particiones:
        preds = ajustar_predecir(tr, te)
        b = datos.loc[te, ["depto", "departamento", "anio", "y", "s_C", "perfil", "perfil_prev", "transicion", "eventos"]].copy()
        b["particion"] = nombre
        for k, v in preds.items():
            b[k] = v
        bloques.append(b)
    return pd.concat(bloques).reset_index().rename(columns={"index": "fila"})


def resumir(oos: pd.DataFrame, modelos, umbral=0.5) -> pd.DataFrame:
    return pd.DataFrame({m: metricas(oos["y"], oos[m], umbral, oos["transicion"]) for m in modelos}).T


def por_particion(oos: pd.DataFrame, modelos, metrica="Log-loss") -> pd.DataFrame:
    return pd.DataFrame({m: oos.groupby("particion").apply(lambda g: metricas(g["y"], g[m])[metrica]) for m in modelos})


# ----------------------------------------------------------------------------------------------------------
# Deriva de datos
# ----------------------------------------------------------------------------------------------------------
def psi(referencia, actual, n_bins=5):
    """Índice de estabilidad poblacional con bins por cuantiles de la referencia. <0,1 estable; 0,1-0,25 vigilar; >0,25 deriva."""
    ref, act = np.asarray(referencia, float), np.asarray(actual, float)
    cortes = np.unique(np.quantile(ref, np.linspace(0, 1, n_bins + 1)))
    if len(cortes) < 3:
        return 0.0
    cortes[0], cortes[-1] = -np.inf, np.inf
    r = np.histogram(ref, cortes)[0] / len(ref); a = np.histogram(act, cortes)[0] / len(act)
    r, a = np.clip(r, 1e-4, None), np.clip(a, 1e-4, None)
    return float(np.sum((a - r) * np.log(a / r)))


def tabla_psi(datos_ref: pd.DataFrame, datos_act: pd.DataFrame, columnas) -> pd.DataFrame:
    filas = []
    for c in columnas:
        v = psi(datos_ref[c], datos_act[c])
        filas.append({"variable": c, "PSI": v, "estado": "estable" if v < .1 else ("vigilar" if v < .25 else "deriva")})
    return pd.DataFrame(filas).set_index("variable")
