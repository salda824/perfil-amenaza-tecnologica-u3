"""
Ingeniería de variables. Todas las predictoras de la fila (d, t) se calculan con información hasta el cierre de t-1.

Grupos (cada uno responde a un diagnóstico de la U2):
  BASE      x1-x7 de la U2 (6 de las 7 variables de la U1 + letalidad derivada).
  MEMORIA   I1: composición del perfil en t-1 y t-2 y su cambio (la inercia domina).
  DINAMICA  I2: proporción C del último semestre de t-1 y tendencias intra-anuales (solo había vista anual).
  VECINDAD  I4: composición y volumen de los departamentos contiguos en t-1 (no había información espacial).
  ACTORES   I3: actores armados no estatales activos en t-1 (UCDP GED; x3 pendiente desde la U1).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from . import datos as D

ALFA = 4.0    # contracción bayesiana de proporciones hacia 0,5 (equivale a 4 eventos "neutros" a priori)

BASE = ["x1_eventos", "x2_fatalidades", "x2b_letalidad", "x4_coca", "x5_bosque", "x6_distancia", "x7_ipm"]
MEMORIA = ["m1_sC_t1", "m2_sC_t2", "m3_delta_sC"]
DINAMICA = ["d1_sC_H2", "d2_tend_eventos", "d3_tend_sC"]
VECINDAD = ["v1_sC_vecinos", "v2_eventos_vecinos"]
ACTORES = ["x3_actores"]
COMPOSICION = ["s_A", "s_B", "s_C", "s_D"]
SECUENCIA = ["q_log_eventos", "q_log_fatalidades", "q_sA", "q_sB", "q_sC"]

ETIQ = {"x1_eventos": "x₁ eventos (log)", "x2_fatalidades": "x₂ fatalidades (log)", "x2b_letalidad": "x₂b letalidad",
        "x4_coca": "x₄ coca (log ha)", "x5_bosque": "x₅ bosque (%)", "x6_distancia": "x₆ distancia (km)",
        "x7_ipm": "x₇ IPM (%)", "x3_actores": "x₃ actores armados (log)",
        "m1_sC_t1": "m₁ s_C en t-1", "m2_sC_t2": "m₂ s_C en t-2", "m3_delta_sC": "m₃ cambio s_C (t-1 − t-2)",
        "d1_sC_H2": "d₁ s_C 2.º semestre t-1", "d2_tend_eventos": "d₂ tendencia de eventos", "d3_tend_sC": "d₃ tendencia de s_C",
        "v1_sC_vecinos": "v₁ s_C de vecinos t-1", "v2_eventos_vecinos": "v₂ eventos de vecinos (log)"}


def contraer(c, n, alfa=None, p0=0.5):
    """Proporción con contracción: (c + α·p0) / (n + α). Estabiliza celdas con pocos eventos."""
    alfa = ALFA if alfa is None else alfa
    return (np.asarray(c, float) + alfa * p0) / (np.asarray(n, float) + alfa)


def panel_extendido(panel, coca, bosque, estructura):
    p = (panel.merge(coca, on=["depto", "anio"], how="left")
              .merge(bosque[["depto", "anio", "bosque_pct"]], on=["depto", "anio"], how="left")
              .merge(estructura, on="depto", how="left")
              .sort_values(["depto", "anio"]))
    p["coca_ha"] = p.groupby("depto")["coca_ha"].ffill().fillna(0)
    p["bosque_pct"] = p.groupby("depto")["bosque_pct"].ffill()
    return p.reset_index(drop=True)


def _variables_de_un_anio(prev: pd.DataFrame, prev2: pd.DataFrame, bloques: pd.DataFrame, vec: dict,
                          actores: pd.DataFrame | None) -> pd.DataFrame:
    """Calcula todas las predictoras de las filas cuyo año de información es `prev.anio` (= t-1)."""
    f = pd.DataFrame(index=prev.index)
    anual = 52 / prev["semanas"]
    f["x1_eventos"] = np.log1p(prev["eventos"] * anual)
    f["x2_fatalidades"] = np.log1p(prev["fatalidades"] * anual)
    f["x2b_letalidad"] = (prev["fatalidades"] / prev["eventos"].where(prev["eventos"] > 0)).fillna(0.0)
    f["x4_coca"] = np.log1p(prev["coca_ha"])
    f["x5_bosque"] = prev["bosque_pct"]
    f["x6_distancia"] = prev["dist_km"]
    f["x7_ipm"] = prev["ipm"]
    # I1 memoria: composición contraída en t-1 y t-2 (si no hay t-2 en la ventana de datos, se usa t-1)
    sc1 = contraer(prev["C"], prev["eventos"])
    p2 = prev[["depto"]].merge(prev2[["depto", "C", "eventos"]], on="depto", how="left")
    sc2 = np.where(p2["eventos"].isna().values, sc1, contraer(p2["C"].fillna(0), p2["eventos"].fillna(0)))
    f["m1_sC_t1"], f["m2_sC_t2"], f["m3_delta_sC"] = sc1, sc2, sc1 - sc2
    # I2 dinámica intra-anual: semestre reciente (bloques 0-1) vs. anterior (bloques 2-3)
    b = bloques.assign(sem=np.where(bloques["bloque"] <= 1, "H2", "H1"))
    h = b.groupby(["depto", "sem"])[["C", "eventos"]].sum()
    get = lambda sem, col: prev["depto"].map(h.xs(sem, level="sem")[col]).fillna(0).values
    sC_H2 = contraer(get("H2", "C"), get("H2", "eventos"))
    sC_H1 = contraer(get("H1", "C"), get("H1", "eventos"))
    f["d1_sC_H2"] = sC_H2
    f["d2_tend_eventos"] = np.log1p(get("H2", "eventos")) - np.log1p(get("H1", "eventos"))
    f["d3_tend_sC"] = sC_H2 - sC_H1
    # I4 vecindad: composición agregada de los vecinos y su volumen medio anualizado
    tot = prev.set_index("depto")
    v_sc, v_ev = [], []
    for d in prev["depto"]:
        vs = [j for j in vec.get(d, []) if j in tot.index] or [j for j in tot.index if j != d]   # sin vecinos -> nacional
        c, n = tot.loc[vs, "C"].sum(), tot.loc[vs, "eventos"].sum()
        v_sc.append(float(contraer(c, n)))
        v_ev.append(float(np.log1p((tot.loc[vs, "eventos"] * 52 / tot.loc[vs, "semanas"]).mean())))
    f["v1_sC_vecinos"], f["v2_eventos_vecinos"] = v_sc, v_ev
    # I3 actores armados (UCDP), si existe la fuente
    if actores is not None:
        a = prev[["depto", "anio"]].merge(actores[["depto", "anio", "actores"]], on=["depto", "anio"], how="left")
        f["x3_actores"] = np.log1p(a["actores"].fillna(0).values)
    return f


def secuencias(prev: pd.DataFrame, bloques: pd.DataFrame) -> np.ndarray:
    """Tensor (n, 4, 5): los 4 bloques de 13 semanas de t-1, del más antiguo al más reciente."""
    b = bloques.copy()
    b["q_log_eventos"] = np.log1p(b["eventos"] * 13 / b["semanas"].clip(lower=1))
    b["q_log_fatalidades"] = np.log1p(b["fat"] * 13 / b["semanas"].clip(lower=1))
    for k in "ABC":
        b[f"q_s{k}"] = contraer(b[k], b["eventos"])
    piv = b.pivot_table(index="depto", columns="bloque", values=SECUENCIA)
    out = np.zeros((len(prev), 4, len(SECUENCIA)), dtype="float32")
    for i, d in enumerate(prev["depto"].values):
        for t, bloque in enumerate([3, 2, 1, 0]):
            out[i, t] = [piv.loc[d, (v, bloque)] for v in SECUENCIA]
    return out


def construir_dataset(acled: pd.DataFrame, mapa: dict | None = None, min_eventos: int = C.MIN_EVENTOS,
                      actores: pd.DataFrame | None = None, incluir_no_confiables: bool = False):
    """
    Devuelve (datos, SEQ, pext):
      datos  una fila por departamento-año etiquetable (t >= 2021, eventos >= min_eventos) con todas las predictoras
             en t-1, la etiqueta dura y, la composición en t (objetivos de I5 e I9) y llaves;
      SEQ    tensor de secuencias trimestrales de t-1 alineado con `datos`;
      pext   panel extendido completo (se usa para proyectar 2027 desde 2026).
    """
    mapa = mapa or C.MAPA_PERFIL_U1
    panel = D.construir_panel(acled, mapa)
    coca, bosque = D.fuentes_externas()
    pext = panel_extendido(panel, coca, bosque, D.estructura_estatica(acled))
    bloques = D.bloques_trimestrales(acled, mapa)
    vec = D.vecindad()
    filas, seqs = [], []
    for t in sorted(pext["anio"].unique()):
        if t < C.ANIO_INICIO + 1:
            continue
        prev = pext[pext["anio"] == t - 1].reset_index(drop=True)
        prev2 = pext[pext["anio"] == t - 2]
        bl = bloques[bloques["anio"] == t - 1]
        feats = _variables_de_un_anio(prev, prev2, bl, vec, actores)
        llave = pd.DataFrame({"depto": prev["depto"].values, "anio": t,
                              "s_C_prev": prev["s_C"].values, "perfil_prev": prev["perfil"].values,
                              "eventos_prev": prev["eventos"].values, "confiable_prev": prev["confiable"].values})
        filas.append(pd.concat([llave, feats.reset_index(drop=True)], axis=1))
        seqs.append(secuencias(prev, bl))
    X = pd.concat(filas, ignore_index=True)
    SEQ_all = np.concatenate(seqs)
    objetivo = pext[["depto", "anio", "A", "B", "C", "D", "eventos", "fatalidades", "semanas", "s_A", "s_B", "s_C", "s_D",
                     "perfil", "y", "confiable", "lat", "lon"]]
    datos = X.merge(objetivo, on=["depto", "anio"], how="inner")
    assert len(datos) == len(X), "La alineación predictoras-etiqueta perdió filas"
    filtro = (datos["eventos"] > 0) if incluir_no_confiables else (datos["eventos"] >= min_eventos)
    filtro &= datos["anio"] >= C.ANIO_INICIO + 1
    SEQ = SEQ_all[filtro.values]
    datos = datos[filtro].reset_index(drop=True)
    datos["departamento"] = datos["depto"].map(C.NOMBRE)
    datos["transicion"] = (datos["perfil"] != datos["perfil_prev"]) & datos["confiable_prev"]
    return datos, SEQ, pext


def filas_proyeccion(acled: pd.DataFrame, pext: pd.DataFrame, anio_info: int, mapa: dict | None = None,
                     actores: pd.DataFrame | None = None):
    """Predictoras para proyectar el año anio_info + 1 (sin etiqueta)."""
    mapa = mapa or C.MAPA_PERFIL_U1
    bloques = D.bloques_trimestrales(acled, mapa)
    prev = pext[pext["anio"] == anio_info].reset_index(drop=True)
    prev2 = pext[pext["anio"] == anio_info - 1]
    bl = bloques[bloques["anio"] == anio_info]
    feats = _variables_de_un_anio(prev, prev2, bl, D.vecindad(), actores)
    out = pd.concat([prev[["depto", "eventos", "semanas", "perfil", "confiable", "lat", "lon", "s_A", "s_B", "s_C", "A", "B"]]
                     .rename(columns={"eventos": "eventos_info", "perfil": "perfil_info", "confiable": "confiable_info",
                                      "s_C": "s_C_info"}), feats.reset_index(drop=True)], axis=1)
    out["departamento"] = out["depto"].map(C.NOMBRE)
    return out, secuencias(prev, bl)
