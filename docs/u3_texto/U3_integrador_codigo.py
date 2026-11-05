# Código de las celdas de 'Proyecto_U3_Integrador.ipynb' en orden (generado para lectura y para graphify).


# ======== Celda de código 1: 0.1 Librerías, determinismo y paquete del proyecto ========
# =============================================================================
# 0.1 Librerías, determinismo y paquete del proyecto
# =============================================================================
import os, sys, time, json, math, platform, subprocess, warnings
from pathlib import Path
from dataclasses import replace, asdict

RAIZ = Path.cwd()
sys.path.insert(0, str(RAIZ / "src"))
from perfil_amenaza import config as C            # fija TF_ENABLE_ONEDNN_OPTS=0 antes de importar TensorFlow
from perfil_amenaza import datos as D, variables as V, modelo as M, evaluacion as E, pipeline as P, graficos as G

import numpy as np, pandas as pd
import matplotlib as mpl, matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle, FancyBboxPatch
from matplotlib.lines import Line2D
import tensorflow as tf, keras, sklearn, scipy
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, confusion_matrix
from sklearn.calibration import calibration_curve
from IPython.display import display, Markdown

warnings.filterwarnings("ignore")
tf.get_logger().setLevel("ERROR")
keras.utils.set_random_seed(C.SEED)
tf.config.experimental.enable_op_determinism()
G.aplicar_estilo(); COL, TINTA, TINTA2, TENUE, GRID, EJE, FONDO = G.COL, G.TINTA, G.TINTA2, G.TENUE, G.GRID, G.EJE, G.FONDO
C.asegurar_carpetas()
pd.set_option("display.width", 200); pd.set_option("display.max_columns", 30)
tabla, guardar = G.tabla, G.guardar

display(pd.Series({
    "Python": platform.python_version(), "Sistema": platform.platform(terse=True),
    "NumPy / pandas / SciPy": f"{np.__version__} / {pd.__version__} / {scipy.__version__}",
    "scikit-learn": sklearn.__version__, "TensorFlow / Keras": f"{tf.__version__} / {keras.__version__}",
    "Dispositivos": ", ".join(sorted({d.device_type for d in tf.config.list_physical_devices()})),
    "Semilla global": C.SEED, "Paquete del proyecto": f"perfil_amenaza {__import__('perfil_amenaza').__version__}",
}, name="Valor").to_frame())
T0_CUADERNO = time.time()


# ======== Celda de código 2: 2.1 Carga, contrato de datos y auditoría ========
# =============================================================================
# 2.1 Carga, contrato de datos y auditoría
# =============================================================================
bruto = D.cargar_acled()
acled = D.filtrar_horizonte(bruto)
display(D.auditar(bruto).style.hide(axis="index"))
print(f"Horizonte de estudio: {acled['week'].min():%Y-%m-%d} → {acled['week'].max():%Y-%m-%d} | "
      f"{len(acled):,} registros semanales | {int(acled['events'].sum()):,} eventos | {acled['depto'].nunique()} departamentos")
print(f"Semanas de 2026 disponibles: {int(acled.loc[acled.anio == 2026, 'semana_anio'].max())} de 52")


# ======== Celda de código 3: 2.2 Panel, etiqueta y persistencia (réplica de la U2) ========
# =============================================================================
# 2.2 Panel, etiqueta y persistencia (réplica de la U2)
# =============================================================================
panel = D.construir_panel(acled)
etiq = panel[(panel.anio >= 2021) & panel.confiable]
pv = panel.pivot(index="depto", columns="anio", values="perfil").where(panel.pivot(index="depto", columns="anio", values="confiable"))
pares = pd.concat([pd.DataFrame({"t-1": pv[a - 1].values, "t": pv[a].values}) for a in range(2021, 2027)], ignore_index=True).dropna()
trans = pd.crosstab(pares["t-1"], pares["t"], rownames=["Perfil t-1"], colnames=["Perfil t"])
display(trans)
print(f"Celdas etiquetables 2021-2026: {len(etiq)} | perfil A o B dominante: {int(panel.perfil.isin(['A', 'B']).sum())} | "
      f"persistencia año a año: {np.trace(trans.values) / trans.values.sum():.1%} | cambios de perfil: {int(trans.values.sum() - np.trace(trans.values))}")
prev_anio = etiq.groupby("anio")["y"].agg(["size", "mean"]).rename(columns={"size": "Departamentos", "mean": "Prevalencia C"})
tabla(prev_anio.T, dec=2)


# ======== Celda de código 4: 2.3 Coincidencia entre la etiqueta a la semana 33 y la del año completo (2021-2025) ========
# =============================================================================
# 2.3 Coincidencia entre la etiqueta a la semana 33 y la del año completo (2021-2025)
# =============================================================================
ctrl, N_SEM_2026 = D.control_etiqueta_parcial(acled)
COINCIDENCIA_PARCIAL = ctrl["coincide"].mean()
por_anio = ctrl.groupby("anio")["coincide"].agg(["size", "mean"]).rename(columns={"size": "Celdas", "mean": "Coincidencia"})
tabla(por_anio.T, dec=3)
discrepa = ctrl[~ctrl.coincide].assign(departamento=lambda d: d.depto.map(C.NOMBRE))
cerca = ((discrepa["s_C"] - .5).abs() < .1).mean()
print(f"Coincidencia global: {COINCIDENCIA_PARCIAL:.1%} de {len(ctrl)} celdas | discrepancias: {len(discrepa)}, "
      f"de las cuales {cerca:.0%} tienen s_C del año completo a ±10 pp del umbral")
display(discrepa[["departamento", "anio", "s_C", "s_C_parcial", "perfil", "perfil_parcial", "eventos"]]
        .sort_values("anio").style.format({"s_C": "{:.2f}", "s_C_parcial": "{:.2f}"}).hide(axis="index"))

fig, ax = plt.subplots(figsize=(7.2, 5.4))
ax.scatter(ctrl["s_C_parcial"], ctrl["s_C"], s=26, c=np.where(ctrl.coincide, COL["mlp"], COL["A"]), edgecolor=FONDO, lw=.8, zorder=3)
ax.plot([0, 1], [0, 1], color=EJE, lw=1); ax.axvline(.5, color=TENUE, lw=.8, ls="--"); ax.axhline(.5, color=TENUE, lw=.8, ls="--")
ax.set_xlabel(f"$s^C$ con las primeras {N_SEM_2026} semanas"); ax.set_ylabel("$s^C$ con el año completo")
ax.set_title(f"Etiqueta parcial vs. completa 2021-2025: {COINCIDENCIA_PARCIAL:.1%} de coincidencia")
ax.legend(handles=[Line2D([], [], marker="o", ls="", color=COL["mlp"], label="misma etiqueta"),
                   Line2D([], [], marker="o", ls="", color=COL["A"], label="etiqueta distinta")], loc="upper left")
fig.tight_layout(); guardar(fig, "fig01_etiqueta_parcial"); plt.show()


# ======== Celda de código 5: 2.4 Dataset U3 y verificación contra el dataset de la U2 ========
# =============================================================================
# 2.4 Dataset U3 y verificación contra el dataset de la U2
# =============================================================================
ACTORES = D.cargar_ucdp()                     # None si UCDP GED no está en datos/externos/ (ver §6.3)
datos, SEQ, pext = V.construir_dataset(acled, actores=ACTORES)
datos["x8_memoria_C"] = datos["s_C_prev"]     # memoria cruda, tal como el experimento E1 de la U2
ref_u2 = pd.read_csv(RAIZ / "referencias_u2" / "dataset_modelado_departamento_anio.csv")
cmp_u2 = datos.merge(ref_u2, on=["depto", "anio"], suffixes=("", "_u2"))
dif_max = max(float(np.abs(cmp_u2[f] - cmp_u2[f + "_u2"]).max()) for f in V.BASE)
assert len(cmp_u2) == len(datos) == len(ref_u2) and (cmp_u2.y == cmp_u2.y_u2).all() and dif_max < 1e-9
print(f"Reproducción U2: {len(datos)} filas, {datos.depto.nunique()} departamentos, etiquetas idénticas, "
      f"diferencia máxima en las 7 predictoras = {dif_max:.1e}")
print(f"UCDP (x3): {'disponible, ' + str(len(ACTORES)) + ' filas departamento-año' if ACTORES is not None else 'no disponible en esta ejecución'}")
print(f"Tensor de secuencias trimestrales (comparativos GRU/CNN): {SEQ.shape}")
nuevas = V.MEMORIA + V.DINAMICA + V.VECINDAD
desc = datos[nuevas].describe().T[["mean", "std", "min", "50%", "max"]]
desc.columns = ["Media", "Desv. est.", "Mín.", "Mediana", "Máx."]
tabla(desc.rename(index=V.ETIQ), dec=3)


# ======== Celda de código 6: 2.5 EDA de las variables nuevas: ¿qué información agregan sobre la inercia? ========
# =============================================================================
# 2.5 EDA de las variables nuevas: ¿qué información agregan sobre la inercia?
# =============================================================================
from scipy import stats
from sklearn.feature_selection import mutual_info_classif
cand = V.BASE + nuevas
rho = pd.Series({f: stats.spearmanr(datos[f], datos["y"]).correlation for f in cand})
mi = pd.Series(mutual_info_classif(datos[cand], datos["y"], random_state=C.SEED, n_neighbors=5), index=cand)

# En las transiciones, ¿el semestre reciente ya apuntaba hacia el nuevo perfil?
tr_ = datos[datos.transicion]
hacia = np.sign(tr_["d1_sC_H2"] - tr_["m1_sC_t1"]) == np.where(tr_["y"] == 1, 1, -1)
ANTICIPA_D1 = hacia.mean()

fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), gridspec_kw={"width_ratios": [1.1, 1, 1]})
ax = axes[0]
orden = mi.sort_values().index
ax.barh(range(len(orden)), mi[orden], color=[COL["u3"] if f in nuevas else COL["D"] for f in orden])
ax.set_yticks(range(len(orden)), [V.ETIQ[f] for f in orden], fontsize=8)
ax.set_xlabel("Información mutua con y (nats)"); ax.set_title("(a) Dependencia con la etiqueta (verde = U3)")
ax = axes[1]
for clase, color in [(0, COL["D"]), (1, COL["C"])]:
    s = datos[datos.y == clase]
    ax.scatter(s["m1_sC_t1"], s["d1_sC_H2"], s=22, color=color, alpha=.8, edgecolor=FONDO, lw=.6, label=f"real {'C' if clase else 'D'} en t")
ax.scatter(tr_["m1_sC_t1"], tr_["d1_sC_H2"], s=90, facecolor="none", edgecolor=TINTA, lw=1.1, label="transición t-1 → t")
ax.plot([0, 1], [0, 1], color=EJE, lw=1); ax.axhline(.5, color=TENUE, lw=.8, ls="--"); ax.axvline(.5, color=TENUE, lw=.8, ls="--")
ax.set_xlabel("$m_1$: s_C de todo t-1"); ax.set_ylabel("$d_1$: s_C del 2.º semestre de t-1")
ax.set_title("(b) Memoria anual vs. semestre reciente"); ax.legend(fontsize=7.5, loc="upper left")
ax = axes[2]
corr = datos[["x2_fatalidades", "x2b_letalidad", "x4_coca", "x7_ipm"] + nuevas].corr(method="spearman")
im = ax.imshow(corr.values, cmap=G.CMAP_DIV, vmin=-1, vmax=1)
et = [V.ETIQ[c].split(" ")[0] for c in corr.columns]
ax.set_xticks(range(len(et)), et, fontsize=7.5, rotation=90); ax.set_yticks(range(len(et)), et, fontsize=7.5)
for i in range(len(et)):
    for j in range(len(et)):
        ax.text(j, i, f"{corr.values[i, j]:.1f}", ha="center", va="center", fontsize=6, color="white" if abs(corr.values[i, j]) > .6 else TINTA)
ax.grid(False); ax.set_title("(c) Correlación de Spearman")
fig.tight_layout(); guardar(fig, "fig02_eda_variables_nuevas"); plt.show()
print(f"Transiciones (cambio de perfil confiable): {len(tr_)} | en {ANTICIPA_D1:.0%} el 2.º semestre de t-1 ya se movía hacia el perfil de t")
tabla(pd.DataFrame({"ρ Spearman con y": rho, "Información mutua": mi}).loc[nuevas].rename(index=V.ETIQ), dec=3)


# ======== Celda de código 7: 3.2 Herramientas de evaluación: semillas pareadas, regla de retención y caché de resultados ========
# =============================================================================
# 3.2 Herramientas de evaluación: semillas pareadas, regla de retención y caché de resultados
# =============================================================================
SEM5 = (1, 2, 3, 4, 5)
RES = {}          # nombre -> {"oos": predicciones, "por_semilla": métricas, "P": matriz semillas x filas}

def evaluar(spec, semillas=SEM5, particiones="movil", base=None, seq=None, clave=None):
    # Entrena la especificación con cada semilla por separado en cada pliegue y guarda las predicciones OOS.
    base = datos if base is None else base; seq = SEQ if seq is None else seq
    clave = clave or spec.nombre
    semillas = semillas if spec.red is not None else (0,)
    Ps, filas = [], []
    for s in semillas:
        sp = replace(spec, semillas=(s,))
        part = E.particiones_origen_movil(base) if particiones == "movil" else E.particiones_espaciales(base)
        oos = E.validar(base, P.ajustar_predecir(base, seq, sp), part)
        Ps.append(oos[spec.nombre].values)
        filas.append({"semilla": s, **E.metricas(oos["y"], oos[spec.nombre], transicion=oos["transicion"])})
    oos = oos.drop(columns=[spec.nombre]); oos["p"] = np.mean(Ps, axis=0)
    RES[clave] = {"oos": oos, "por_semilla": pd.DataFrame(filas).set_index("semilla"), "P": np.array(Ps), "spec": spec}
    return RES[clave]

def resumen(claves, ensamble=False):
    # Tabla media ± desv. entre semillas (o métricas del ensamble de semillas si ensamble=True).
    filas = {}
    for k in claves:
        r = RES[k]
        if ensamble:
            filas[k] = E.metricas(r["oos"]["y"], r["oos"]["p"], transicion=r["oos"]["transicion"])
        else:
            ps = r["por_semilla"]
            filas[k] = {**{c: ps[c].mean() for c in E.COLS_PRINCIPALES}, "Log-loss (DE)": ps["Log-loss"].std(ddof=1) if len(ps) > 1 else 0.0,
                        "Transiciones acertadas": ps["Transiciones acertadas"].mean(), "Transiciones": ps["Transiciones"].iloc[0]}
    return pd.DataFrame(filas).T

def decidir(candidata, referencia, nombre=None):
    # Regla de retención (§3.1, regla 5) con diferencias pareadas por semilla.
    a, b = RES[candidata]["por_semilla"], RES[referencia]["por_semilla"]
    if len(a) != len(b):
        b = pd.concat([b] * len(a)).iloc[:len(a)].set_index(a.index)
    d_ll = a["Log-loss"].values - b["Log-loss"].values
    d_auc = a["ROC-AUC"].values - b["ROC-AUC"].values
    ee = d_ll.std(ddof=1) / np.sqrt(len(d_ll)) if len(d_ll) > 1 else 0.0
    retener = (d_ll.mean() + ee < 0) and (d_auc.mean() > -0.01)
    return {"Intervención": nombre or candidata, "Referencia": referencia, "ΔLog-loss": d_ll.mean(), "EE": ee,
            "ΔAUC": d_auc.mean(), "Decisión": "se retiene" if retener else "no se retiene"}

def persistencia_oos(particiones="movil", base=None):
    base = datos if base is None else base
    part = E.particiones_origen_movil(base) if particiones == "movil" else E.particiones_espaciales(base)
    return E.validar(base, lambda tr, te: {"p": P.persistencia(base, tr, te)}, part)

PERS = persistencia_oos()
M_PERS = E.metricas(PERS["y"], PERS["p"], transicion=PERS["transicion"])
print(f"Persistencia, origen móvil 2023-2025 (n = {len(PERS)}): AUC {M_PERS['ROC-AUC']:.3f} | log-loss {M_PERS['Log-loss']:.3f} | "
      f"Brier {M_PERS['Brier']:.3f} | transiciones acertadas {M_PERS['Transiciones acertadas']} de {M_PERS['Transiciones']}")

fig, ax = plt.subplots(figsize=(13, 2.9))
for fila, v in enumerate((2023, 2024, 2025, 2026)):
    for anio in range(2021, 2027):
        rol = ("Entrenamiento", COL["train"]) if anio < v else (("Validación" if v < 2026 else "Prueba final"), COL["val"] if v < 2026 else COL["test"]) if anio == v else ("", GRID)
        ax.add_patch(Rectangle((anio - .46, 3 - fila - .4), .92, .8, facecolor=rol[1], edgecolor=FONDO, lw=2))
        if rol[0]:
            ax.text(anio, 3 - fila, rol[0], ha="center", va="center", fontsize=7.5, color="white" if rol[1] != GRID else TINTA)
ax.set_yticks(range(4), ["Prueba 2026\n(una vez)", "Pliegue 2025", "Pliegue 2024", "Pliegue 2023"])
ax.set_xticks(range(2021, 2027)); ax.set_xlim(2020.5, 2026.5); ax.set_ylim(-.6, 3.6); ax.grid(False)
ax.set_title("Protocolo U3: selección con origen móvil 2023-2025; 2026 se evalúa una sola vez al final")
guardar(fig, "fig03_protocolo"); plt.show()


# ======== Celda de código 8: 4.1 Reproducción con el protocolo de la U2 (2023-2026) y con el de la U3 (2023-2025) ========
# =============================================================================
# 4.1 Reproducción con el protocolo de la U2 (2023-2026) y con el de la U3 (2023-2025)
# =============================================================================
CFG_U2 = M.ConfigRed()                                       # 7 -> 16 ReLU -> 1, L2 1e-3, Adam 1e-3, lote 16, 560 pasos
BASE_U2 = P.Especificacion("MLP base U2", V.BASE, CFG_U2)
LOG_BASE = P.Especificacion("Logística base", V.BASE, None)

def fn_u2(tr, te):
    out = {"Persistencia": P.persistencia(datos, tr, te)}
    for s in (replace(BASE_U2, semillas=(C.SEED,)), LOG_BASE):
        m_, e_ = P.entrenar_especificacion(datos, SEQ, s, tr)
        out[s.nombre] = P.predecir_especificacion(m_, e_, datos, SEQ, s, te)
    return out
OOS_U2 = E.validar(datos, fn_u2, E.particiones_origen_movil(datos, (2023, 2024, 2025, 2026)))
rep = E.resumir(OOS_U2, ["MLP base U2", "Logística base", "Persistencia"])[["ROC-AUC", "Log-loss", "Brier", "Exactitud", "MCC"]]
doc = pd.read_csv(RAIZ / "referencias_u2" / "experimento_E1_memoria.csv").query("Validación == 'Temporal'").set_index("Modelo")
doc = doc.rename(index={"MLP base (x1-x7)": "MLP base U2", "Logística base (x1-x7)": "Logística base"}).loc[rep.index, ["ROC-AUC", "Log-loss", "Brier", "Exactitud", "MCC"]]
comp_rep = pd.concat({"Documentado en la U2": doc, "Reproducido (U3)": rep}, axis=1)
tabla(comp_rep, dec=3)
DIF_REP = float((comp_rep["Reproducido (U3)"] - comp_rep["Documentado en la U2"]).abs().max().max())
print(f"Diferencia máxima entre lo documentado y lo reproducido: {DIF_REP:.4f}")

# Protocolo U3
evaluar(BASE_U2); evaluar(LOG_BASE)
evaluar(P.Especificacion("U2 E1 (memoria cruda x8)", V.BASE + ["x8_memoria_C"], CFG_U2))
tabla(pd.concat([resumen(["MLP base U2", "Logística base", "U2 E1 (memoria cruda x8)"]),
                 pd.DataFrame({**{c: M_PERS[c] for c in E.COLS_PRINCIPALES}, "Transiciones acertadas": 0,
                               "Transiciones": M_PERS["Transiciones"]}, index=["Persistencia"])]), dec=3)


# ======== Celda de código 9: 6.1 I1 memoria: m1 (s_C contraída de t-1) y la trayectoria m1, m2, m3 ========
# =============================================================================
# 6.1 I1 memoria: m1 (s_C contraída de t-1) y la trayectoria m1, m2, m3
# =============================================================================
DECISIONES, ESCALERA = [], [("Modelo base U2", "MLP base U2")]
evaluar(P.Especificacion("I1a +m1", V.BASE + ["m1_sC_t1"], CFG_U2))
evaluar(P.Especificacion("I1b +m1 m2 m3", V.BASE + V.MEMORIA, CFG_U2))
VARS, REF = list(V.BASE), "MLP base U2"
d = decidir("I1a +m1", REF, "I1 memoria m1"); DECISIONES.append(d)
if d["Decisión"] == "se retiene":
    VARS, REF = VARS + ["m1_sC_t1"], "I1a +m1"; ESCALERA.append(("+ I1 memoria m₁", REF))
d = decidir("I1b +m1 m2 m3", REF, "I1 trayectoria m2, m3"); DECISIONES.append(d)
if d["Decisión"] == "se retiene":
    VARS, REF = list(V.BASE) + V.MEMORIA, "I1b +m1 m2 m3"; ESCALERA.append(("+ I1 trayectoria", REF))
tabla(resumen(["MLP base U2", "U2 E1 (memoria cruda x8)", "I1a +m1", "I1b +m1 m2 m3"]), dec=3)
display(pd.DataFrame(DECISIONES).style.format({"ΔLog-loss": "{:+.4f}", "EE": "{:.4f}", "ΔAUC": "{:+.4f}"}).hide(axis="index"))
print("Variables retenidas tras I1:", [v for v in VARS if v not in V.BASE])


# ======== Celda de código 10: 6.2 I2 dinámica intra-anual sobre la especificación retenida ========
# =============================================================================
# 6.2 I2 dinámica intra-anual sobre la especificación retenida
# =============================================================================
evaluar(P.Especificacion("I2a +d1", VARS + ["d1_sC_H2"], CFG_U2))
evaluar(P.Especificacion("I2b +d1 d2 d3", VARS + V.DINAMICA, CFG_U2))
ref_i2 = REF
d = decidir("I2a +d1", REF, "I2 semestre reciente d1"); DECISIONES.append(d)
if d["Decisión"] == "se retiene":
    VARS, REF = VARS + ["d1_sC_H2"], "I2a +d1"; ESCALERA.append(("+ I2 semestre d₁", REF))
d = decidir("I2b +d1 d2 d3", REF, "I2 tendencias d2, d3"); DECISIONES.append(d)
if d["Decisión"] == "se retiene":
    VARS, REF = [v for v in VARS if v != "d1_sC_H2"] + V.DINAMICA, "I2b +d1 d2 d3"; ESCALERA.append(("+ I2 tendencias", REF))
tabla(resumen([ref_i2, "I2a +d1", "I2b +d1 d2 d3"]), dec=3)
display(pd.DataFrame(DECISIONES[-2:]).style.format({"ΔLog-loss": "{:+.4f}", "EE": "{:.4f}", "ΔAUC": "{:+.4f}"}).hide(axis="index"))

# ¿Dónde aporta? Log-loss por año y aciertos en transiciones
por_anio = pd.DataFrame({k: RES[k]["oos"].assign(p=RES[k]["P"].mean(0)).groupby("particion").apply(lambda g: E.bce(g.y, g.p))
                         for k in ["MLP base U2", "I1a +m1", "I2a +d1"]})
por_anio["Persistencia"] = PERS.groupby("particion").apply(lambda g: E.bce(g.y, g.p))
tabla(por_anio.T.rename(columns=str), dec=3)
print("Variables retenidas tras I2:", [v for v in VARS if v not in V.BASE])


# ======== Celda de código 11: 6.3 I3 actores armados (condicionada a la disponibilidad de UCDP GED) ========
# =============================================================================
# 6.3 I3 actores armados (condicionada a la disponibilidad de UCDP GED)
# =============================================================================
if ACTORES is not None and "x3_actores" in datos:
    evaluar(P.Especificacion("I3 +x3", VARS + ["x3_actores"], CFG_U2))
    d = decidir("I3 +x3", REF, "I3 actores armados x3"); DECISIONES.append(d)
    tabla(resumen([REF, "I3 +x3"]), dec=3)
    display(pd.DataFrame([d]).style.hide(axis="index"))
    ar = RES["I3 +x3"]["oos"].assign(p_x3=RES["I3 +x3"]["P"].mean(0), p_ref=RES[REF]["P"].mean(0)).query("depto == 'arauca'")
    display(ar[["departamento", "anio", "perfil", "p_ref", "p_x3"]])
    if d["Decisión"] == "se retiene":
        VARS, REF = VARS + ["x3_actores"], "I3 +x3"; ESCALERA.append(("+ I3 actores x₃", REF))
    I3_ESTADO = "evaluada"
else:
    I3_ESTADO = "pendiente de fuente"
    DECISIONES.append({"Intervención": "I3 actores armados x3", "Referencia": REF, "ΔLog-loss": np.nan, "EE": np.nan,
                       "ΔAUC": np.nan, "Decisión": "pendiente: UCDP GED no disponible en esta ejecución"})
    display(Markdown("**I3 no se ejecutó:** `GEDEvent_v26_1.csv` no está en `datos/externos/`. Descárguelo de "
                     "https://ucdp.uu.se/downloads/ged/ged261-csv.zip, cópielo en esa carpeta y vuelva a ejecutar el cuaderno."))


# ======== Celda de código 12: 6.4 I4 vecindad: validación temporal y espacial ========
# =============================================================================
# 6.4 I4 vecindad: validación temporal y espacial
# =============================================================================
spec_ref, ref_i4 = RES[REF]["spec"], REF
evaluar(P.Especificacion("I4 +v1 v2", VARS + V.VECINDAD, CFG_U2))
d_t = decidir("I4 +v1 v2", REF, "I4 vecindad (temporal)")
for k, s in [("esp: MLP base U2", BASE_U2), ("esp: referencia", spec_ref), ("esp: I4 +v1 v2", RES["I4 +v1 v2"]["spec"]),
             ("esp: logística base", LOG_BASE), ("esp: logística referencia", P.Especificacion("lr", VARS, None))]:
    evaluar(replace(s, nombre=k), particiones="espacial", clave=k)
PERS_E = persistencia_oos("espacial")
d_e = decidir("esp: I4 +v1 v2", "esp: referencia", "I4 vecindad (espacial)")
retener_i4 = d_e["Decisión"] == "se retiene" and d_t["ΔLog-loss"] < d_t["EE"]
d_t["Decisión"] = d_e["Decisión"] = "se retiene" if retener_i4 else "no se retiene"
DECISIONES += [d_t, d_e]
if retener_i4:
    VARS, REF = VARS + V.VECINDAD, "I4 +v1 v2"; ESCALERA.append(("+ I4 vecindad", REF))
esp = resumen(["esp: MLP base U2", "esp: logística base", "esp: referencia", "esp: logística referencia", "esp: I4 +v1 v2"])
M_PERS_E = E.metricas(PERS_E.y, PERS_E.p)
esp.loc["esp: persistencia", E.COLS_PRINCIPALES] = [M_PERS_E[c] for c in E.COLS_PRINCIPALES]
tabla(pd.concat({"Temporal 2023-2025": resumen([ref_i4, "I4 +v1 v2"])[E.COLS_PRINCIPALES],
                 "Espacial (deptos. nunca vistos)": esp[E.COLS_PRINCIPALES]}), dec=3)
display(pd.DataFrame([d_t, d_e]).style.format({"ΔLog-loss": "{:+.4f}", "EE": "{:.4f}", "ΔAUC": "{:+.4f}"}).hide(axis="index"))
VARS_FINAL, REF_VARS = list(VARS), REF
print(f"Especificación de variables fijada ({len(VARS_FINAL)}):", VARS_FINAL)


# ======== Celda de código 13: 6.5 I5 objetivo blando (con y sin ponderación por eventos), crudo y calibrado ========
# =============================================================================
# 6.5 I5 objetivo blando (con y sin ponderación por eventos), crudo y calibrado
# =============================================================================
def calibrar_adelante(oos, p, metodo="temperatura"):
    # Calibración con encadenamiento hacia adelante: el pliegue v se calibra con las predicciones de los pliegues < v.
    q = np.array(p, float).copy()
    for v in sorted(oos["particion"].unique())[1:]:
        m, prev = (oos["particion"] == v).values, (oos["particion"] < v).values
        q[m] = M.Calibrador(metodo).ajustar(p[prev], oos["y"].values[prev]).aplicar(p[m])
    return q

def registrar_calibrado(clave, metodo="temperatura"):
    r = RES[clave]; oos = r["oos"]
    Pc = np.array([calibrar_adelante(oos, p, metodo) for p in r["P"]])
    filas = [{"semilla": s, **E.metricas(oos.y, pc, transicion=oos.transicion)} for s, pc in zip(r["por_semilla"].index, Pc)]
    RES[clave + " · cal"] = {"oos": oos.assign(p=Pc.mean(0)), "por_semilla": pd.DataFrame(filas).set_index("semilla"), "P": Pc, "spec": r["spec"]}

CFG_REF = RES[REF_VARS]["spec"].red
evaluar(P.Especificacion("I5 blando", VARS_FINAL, replace(CFG_REF, objetivo="blando")))
evaluar(P.Especificacion("I5 blando + peso eventos", VARS_FINAL, replace(CFG_REF, objetivo="blando", peso_eventos="raiz")))
for k in (REF_VARS, "I5 blando", "I5 blando + peso eventos"):
    registrar_calibrado(k)
d_crudo = decidir("I5 blando", REF_VARS, "I5 objetivo blando (sin calibrar)")
d_cal = decidir("I5 blando · cal", REF_VARS + " · cal", "I5 objetivo blando (ambos calibrados)")
DECISIONES += [d_crudo, d_cal]
OBJETIVO = "blando" if d_cal["Decisión"] == "se retiene" else "duro"
tabla(resumen([REF_VARS, "I5 blando", "I5 blando + peso eventos", REF_VARS + " · cal", "I5 blando · cal", "I5 blando + peso eventos · cal"]), dec=3)
display(pd.DataFrame([d_crudo, d_cal]).style.format({"ΔLog-loss": "{:+.4f}", "EE": "{:.4f}", "ΔAUC": "{:+.4f}"}).hide(axis="index"))
print("Objetivo de entrenamiento elegido:", OBJETIVO)


# ======== Celda de código 14: 6.6 I9 multitarea: efecto sobre P(C) y calidad de la composición predicha ========
# =============================================================================
# 6.6 I9 multitarea: efecto sobre P(C) y calidad de la composición predicha
# =============================================================================
for lam in (0.5, 1.0):
    evaluar(P.Especificacion(f"I9 multitarea λ={lam:g}", VARS_FINAL, replace(CFG_REF, tipo="multitarea", lambda_comp=lam, objetivo=OBJETIVO)))
d9 = [decidir(f"I9 multitarea λ={lam:g}", REF_VARS, f"I9 multitarea λ={lam:g} (efecto en P(C))") for lam in (0.5, 1.0)]

# Composición fuera de muestra (ensamble de 5 semillas por pliegue)
spec9 = replace(RES["I9 multitarea λ=0.5"]["spec"], semillas=SEM5)
comp_oos = []
for v, tr, te in E.particiones_origen_movil(datos):
    m_, e_ = P.entrenar_especificacion(datos, SEQ, spec9, tr)
    comp_oos.append(P.predecir_especificacion(m_, e_, datos, SEQ, spec9, te, composicion=True))
comp_oos = np.concatenate(comp_oos)
te_ = datos.anio.isin(C.ANIOS_SELECCION).values
real = datos.loc[te_, V.COMPOSICION].values
prev_ab = datos.loc[te_, ["depto", "anio"]].merge(pext[["depto", "anio", "s_A", "s_B"]].assign(anio=lambda d: d.anio + 1), on=["depto", "anio"], how="left")
sAB_real, sAB_pers, sAB_mod = real[:, 0] + real[:, 1], (prev_ab.s_A + prev_ab.s_B).fillna(0).values, comp_oos[:, 0] + comp_oos[:, 1]
umbral_ab = pd.Series(sAB_real).groupby(datos.loc[te_, "anio"].values).transform(lambda s: 2 * s.mean()).values
alto_ab = sAB_real >= umbral_ab
calidad = pd.DataFrame({
    "Modelo multitarea (λ=0,5)": [np.mean(np.abs(sAB_mod - sAB_real)), stats.spearmanr(sAB_mod, sAB_real).correlation, roc_auc_score(alto_ab, sAB_mod)],
    "Persistencia de la composición": [np.mean(np.abs(sAB_pers - sAB_real)), stats.spearmanr(sAB_pers, sAB_real).correlation, roc_auc_score(alto_ab, sAB_pers)]},
    index=["MAE de s_A + s_B", "ρ de Spearman con s_A + s_B real", "AUC para detectar alerta A/B (≥ 2× media del año)"])
I9_COMPOSICION_GANA = calidad.iloc[2, 0] > calidad.iloc[2, 1] and calidad.iloc[0, 0] < calidad.iloc[0, 1]
for x in d9:
    if x["Decisión"] == "se retiene" and not I9_COMPOSICION_GANA:
        x["Decisión"] = "no se retiene (no aporta en P(C) ni en A/B)"
    elif x["Decisión"] != "se retiene" and I9_COMPOSICION_GANA and x["ΔLog-loss"] < x["EE"]:
        x["Decisión"] = "se retiene (no empeora P(C) y anticipa A/B)"
DECISIONES += d9
TIPO = "multitarea" if any(x["Decisión"].startswith("se retiene") for x in d9) else "mlp"
CFG_ACTUAL = RES["I9 multitarea λ=0.5"]["spec"].red if TIPO == "multitarea" else replace(CFG_REF, objetivo=OBJETIVO)
tabla(resumen([REF_VARS, "I9 multitarea λ=0.5", "I9 multitarea λ=1"]), dec=3)
tabla(calidad, dec=3)
display(pd.DataFrame(d9).style.format({"ΔLog-loss": "{:+.4f}", "EE": "{:.4f}", "ΔAUC": "{:+.4f}"}).hide(axis="index"))
print(f"Filas con alerta A/B real en 2023-2025: {alto_ab.sum()} de {len(alto_ab)} | tipo de red elegido: {TIPO}")


# ======== Celda de código 15: 6.7 Comparativos: GRU, CNN 1D, embedding, MLP profundo y logística con las mismas variables ========
# =============================================================================
# 6.7 Comparativos: GRU, CNN 1D, embedding, MLP profundo y logística con las mismas variables
# =============================================================================
comparativos = {
    "GRU trimestral": replace(CFG_REF, tipo="gru", unidades_seq=8),
    "CNN 1D trimestral": replace(CFG_REF, tipo="cnn", unidades_seq=8),
    "Embedding de departamento": replace(CFG_REF, tipo="embedding", emb_dim=2),
    "MLP profundo (16, 8)": replace(CFG_REF, capas=(16, 8)),
}
for k, cfg_k in comparativos.items():
    evaluar(P.Especificacion(k, VARS_FINAL, cfg_k))
LOG_F = P.Especificacion("Logística (mismas variables)", VARS_FINAL, None)
evaluar(LOG_F)
d7 = [decidir(k, REF_VARS, k) for k in [*comparativos, "Logística (mismas variables)"]]
emb = next(x for x in d7 if x["Intervención"] == "Embedding de departamento")
TR_EMB = RES["Embedding de departamento"]["por_semilla"]["Transiciones acertadas"].mean()
TR_REF = RES[REF_VARS]["por_semilla"]["Transiciones acertadas"].mean()
if emb["Decisión"] == "se retiene" and TR_EMB < TR_REF:
    emb["Decisión"] = (f"cumple la regla, no se adopta: acierta {TR_EMB:.1f} de {int(M_PERS['Transiciones'])} transiciones "
                       f"(referencia {TR_REF:.1f}) y no transfiere a territorios nuevos")
DECISIONES += d7
evaluar(P.Especificacion("esp: embedding", VARS_FINAL, comparativos["Embedding de departamento"]), particiones="espacial", clave="esp: embedding")
tab7 = resumen([REF_VARS, *comparativos, "Logística (mismas variables)"])
tab7["Parámetros"] = [RES[k]["spec"].red and M.n_parametros(M.construir_red(len(VARS_FINAL), RES[k]["spec"].red, SEQ.shape[1:] if RES[k]["spec"].red.tipo in ("gru", "cnn") else None)) or len(VARS_FINAL) + 1 for k in tab7.index]
tabla(tab7, dec=3)
display(pd.DataFrame(d7).style.format({"ΔLog-loss": "{:+.4f}", "EE": "{:.4f}", "ΔAUC": "{:+.4f}"}).hide(axis="index"))
print(f"Embedding en validación espacial: log-loss {RES['esp: embedding']['por_semilla']['Log-loss'].mean():.3f} "
      f"vs. MLP sin embedding {RES['esp: referencia']['por_semilla']['Log-loss'].mean():.3f}")


# ======== Celda de código 16: 6.8 I6 rejilla de capacidad x L2 x activación x presupuesto (3 semillas, 3 pliegues) ========
# =============================================================================
# 6.8 I6 rejilla de capacidad x L2 x activación x presupuesto (3 semillas, 3 pliegues)
# =============================================================================
t0 = time.time(); filas = []
REJILLA = [(c, l2, act, pasos) for c in [(4,), (8,), (16,), (32,)] for l2 in [1e-3, 1e-2] for act in ["relu", "tanh"] for pasos in [280, 560, 1120]]
for capas, l2, act, pasos in REJILLA:
    cfg_g = replace(CFG_ACTUAL, capas=capas, l2=l2, activacion=act, pasos=pasos)
    for s in (1, 2, 3):
        sp = P.Especificacion("g", VARS_FINAL, cfg_g, semillas=(s,))
        oos = E.validar(datos, P.ajustar_predecir(datos, SEQ, sp), E.particiones_origen_movil(datos))
        for v, g in oos.groupby("particion"):
            filas.append({"capas": str(capas), "L2": l2, "activación": act, "pasos": pasos, "semilla": s, "año": int(v), "LL": E.bce(g.y, g.g)})
REJ = pd.DataFrame(filas)
print(f"{len(REJILLA)} configuraciones × 3 semillas × 3 pliegues = {len(REJILLA) * 9} entrenamientos en {time.time() - t0:.0f} s")
K_ = ["capas", "L2", "activación", "pasos"]
N_PAR = {c: M.n_parametros(M.construir_red(len(VARS_FINAL), replace(CFG_ACTUAL, capas=eval(c)))) for c in REJ.capas.unique()}

def seleccionar(df):
    a = df.groupby(K_)["LL"].agg(media="mean", ee=lambda s: s.std(ddof=1) / np.sqrt(len(s))).reset_index()
    a["parámetros"] = a.capas.map(N_PAR)
    mejor = a.loc[a.media.idxmin()]
    cand = a[a.media <= mejor.media + mejor.ee].sort_values(["parámetros", "pasos", "media"])
    return a.sort_values("media"), mejor, cand.iloc[0], len(cand)

TAB_REJ, MEJOR, UNO_EE, N_CAND = seleccionar(REJ)
DEF = (str(CFG_ACTUAL.capas), CFG_ACTUAL.l2, CFG_ACTUAL.activacion, CFG_ACTUAL.pasos)
ll_def = TAB_REJ.set_index(K_).loc[DEF]
print(f"Mejor media: {tuple(MEJOR[K_])} LL {MEJOR.media:.3f} ± {MEJOR.ee:.3f} | configuraciones dentro de 1 EE: {N_CAND} de {len(TAB_REJ)}")
print(f"Configuración vigente {DEF}: LL {ll_def.media:.3f} (puesto {int((TAB_REJ.media < ll_def.media).sum()) + 1} de {len(TAB_REJ)}; "
      f"{'dentro' if ll_def.media <= MEJOR.media + MEJOR.ee else 'fuera'} de 1 EE del mejor)")

# Validación anidada: seleccionar con 2023-2024 y medir en 2025
_, m_in, u_in, _ = seleccionar(REJ[REJ["año"] < 2025])
ll25 = REJ[REJ["año"] == 2025].groupby(K_)["LL"].agg(["mean", "std"])
ANIDADA = pd.DataFrame({
    "Configuración": [str(tuple(m_in[K_])), str(tuple(u_in[K_])), str(DEF), str(tuple(ll25["mean"].idxmin()))],
    "LL 2023-2024 (selección)": [m_in.media, u_in.media, REJ[REJ["año"] < 2025].groupby(K_)["LL"].mean().loc[DEF], np.nan],
    "LL 2025 (fuera de la selección)": [ll25.loc[tuple(m_in[K_]), "mean"], ll25.loc[tuple(u_in[K_]), "mean"], ll25.loc[DEF, "mean"], ll25["mean"].min()],
}, index=["Mínimo en 2023-2024", "Regla 1 EE en 2023-2024", "Vigente (U2)", "Oráculo (mejor en 2025, inalcanzable)"])
tabla(ANIDADA, dec=3)
SELECCION_TRANSFIERE = min(ANIDADA.iloc[0, 2], ANIDADA.iloc[1, 2]) < ANIDADA.iloc[2, 2]
CFG_I6 = CFG_ACTUAL if not SELECCION_TRANSFIERE else replace(CFG_ACTUAL, capas=eval(UNO_EE.capas), l2=float(UNO_EE.L2), activacion=UNO_EE["activación"], pasos=int(UNO_EE.pasos))
DECISIONES.append({"Intervención": "I6 capacidad / activación / L2 / presupuesto", "Referencia": str(DEF), "ΔLog-loss": ANIDADA.iloc[1, 2] - ANIDADA.iloc[2, 2],
                   "EE": np.nan, "ΔAUC": np.nan, "Decisión": "se retiene" if SELECCION_TRANSFIERE else "no se retiene (la selección no transfiere a 2025)"})

# Tasa de aprendizaje (con el presupuesto vigente)
for lr in (3e-4, 3e-3):
    evaluar(P.Especificacion(f"I6 lr={lr:g}", VARS_FINAL, replace(CFG_I6, lr=lr)))
tabla(resumen([REF_VARS, "I6 lr=0.0003", "I6 lr=0.003"])[["ROC-AUC", "Log-loss", "Log-loss (DE)", "Brier"]], dec=3)

fig, axes = plt.subplots(1, 3, figsize=(16, 4.4), gridspec_kw={"width_ratios": [1, 1, 1.15]})
for ax, act in zip(axes[:2], ["relu", "tanh"]):
    m_ = REJ[REJ["activación"] == act].groupby(["capas", "pasos"])["LL"].mean().unstack().loc[["(4,)", "(8,)", "(16,)", "(32,)"]]
    im = ax.imshow(m_.values, cmap=G.CMAP_P.reversed(), vmin=TAB_REJ.media.min(), vmax=.42, aspect="auto")
    for i in range(m_.shape[0]):
        for j in range(m_.shape[1]):
            ax.text(j, i, f"{m_.values[i, j]:.3f}", ha="center", va="center", fontsize=9, color="white" if m_.values[i, j] < .31 else TINTA)
    ax.set_xticks(range(3), [f"{p} pasos" for p in m_.columns]); ax.set_yticks(range(4), [f"{c} · {N_PAR[c]} par." for c in m_.index])
    ax.grid(False); ax.set_title(f"({'ab'[act == 'tanh']}) Log-loss media · {act.upper()} (promedio de L2)")
ax = axes[2]
x = np.arange(len(ANIDADA) - 1)
ax.bar(x - .18, ANIDADA.iloc[:3, 1], width=.34, color=COL["D"], label="2023-2024 (donde se elige)")
ax.bar(x + .18, ANIDADA.iloc[:3, 2], width=.34, color=COL["val"], label="2025 (fuera de la elección)")
ax.axhline(ANIDADA.iloc[3, 2], color=TINTA, lw=1, ls=":"); ax.text(2.45, ANIDADA.iloc[3, 2], " oráculo 2025", fontsize=8, va="bottom", ha="right")
ax.set_xticks(x, ["Mínimo\n2023-24", "Regla 1 EE\n2023-24", "Vigente\n(U2)"]); ax.set_ylabel("Log-loss")
ax.set_title("(c) Validación anidada del proceso de selección"); ax.legend(fontsize=8, loc="upper left")
fig.tight_layout(); guardar(fig, "fig04_rejilla_anidada"); plt.show()


# ======== Celda de código 17: 6.9 I7 ensamble: 20 semillas de la especificación vigente y curva de tamaño del ensamble ========
# =============================================================================
# 6.9 I7 ensamble: 20 semillas de la especificación vigente y curva de tamaño del ensamble
# =============================================================================
SPEC_I7 = P.Especificacion("I7 base de semillas", VARS_FINAL, CFG_I6)
r20 = evaluar(SPEC_I7, semillas=tuple(range(1, 21)), clave="I7 20 semillas")
P20, y_oos = r20["P"], r20["oos"]["y"].values
rng_k = np.random.default_rng(C.SEED); curva = []
for k in (1, 2, 3, 5, 7, 10, 15, 20):
    ll = [E.bce(y_oos, P20[rng_k.choice(20, k, replace=False)].mean(0)) for _ in range(200 if k < 20 else 1)]
    au = [roc_auc_score(y_oos, P20[rng_k.choice(20, k, replace=False)].mean(0)) for _ in range(200 if k < 20 else 1)]
    curva.append({"K": k, "LL media": np.mean(ll), "LL p5": np.percentile(ll, 5), "LL p95": np.percentile(ll, 95), "AUC media": np.mean(au), "LL DE": np.std(ll)})
CURVA_K = pd.DataFrame(curva).set_index("K")
K_ENS = 10
desv_ind = P20.std(0)
unan = np.mean(((P20 >= .5).mean(0) == 0) | ((P20 >= .5).mean(0) == 1))
tabla(CURVA_K, dec=4)
print(f"Dispersión de P(C) entre semillas por fila: mediana {np.median(desv_ind):.3f}, máx. {desv_ind.max():.3f} | "
      f"filas con predicción unánime entre 20 redes: {unan:.0%}")
RANGO_K = (CURVA_K.loc[1, "LL p95"] - CURVA_K.loc[1, "LL p5"], CURVA_K.loc[K_ENS, "LL p95"] - CURVA_K.loc[K_ENS, "LL p5"])
DECISIONES.append({"Intervención": f"I7 ensamble de {K_ENS} semillas", "Referencia": "1 red", "ΔLog-loss": CURVA_K.loc[K_ENS, "LL media"] - CURVA_K.loc[1, "LL media"],
                   "EE": np.nan, "ΔAUC": CURVA_K.loc[K_ENS, "AUC media"] - CURVA_K.loc[1, "AUC media"], "Decisión": "se retiene (estabilidad)"})

fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
ax = axes[0]
ax.fill_between(CURVA_K.index, CURVA_K["LL p5"], CURVA_K["LL p95"], color=COL["u3"], alpha=.18, lw=0, label="banda 5-95 % entre sorteos")
ax.plot(CURVA_K.index, CURVA_K["LL media"], color=COL["u3"], marker="o", label="log-loss media")
ax.axhline(M_PERS["Log-loss"], color=COL["pers"], ls="--", lw=1.2); ax.text(20, M_PERS["Log-loss"], "persistencia ", ha="right", va="bottom", fontsize=8)
ax.axvline(K_ENS, color=TINTA, lw=1, ls=":"); ax.set_xlabel("Redes en el ensamble (K)"); ax.set_ylabel("Log-loss origen móvil 2023-2025")
ax.set_title("(a) El ensamble reduce sobre todo la dispersión"); ax.legend(fontsize=8)
ax = axes[1]
orden = np.argsort(r20["oos"]["p"].values)
ax.errorbar(np.arange(len(orden)), P20.mean(0)[orden], yerr=desv_ind[orden], fmt="none", ecolor=EJE, lw=1)
ax.scatter(np.arange(len(orden)), P20.mean(0)[orden], s=14, c=np.where(y_oos[orden] == 1, COL["C"], COL["D"]), zorder=3)
ax.axhline(.5, color=TINTA, lw=.8, ls="--"); ax.set_xlabel("Departamento-año (ordenado por P(C) del ensamble)"); ax.set_ylabel("P(C) ± DE entre 20 redes")
ax.set_title("(b) Incertidumbre por fila (color = perfil real)")
fig.tight_layout(); guardar(fig, "fig05_ensamble"); plt.show()


# ======== Celda de código 18: 6.10 I8 calibración (Platt / temperatura) y estabilidad del umbral óptimo entre años ========
# =============================================================================
# 6.10 I8 calibración (Platt / temperatura) y estabilidad del umbral óptimo entre años
# =============================================================================
oos7 = r20["oos"]; p_ens = P20[:K_ENS].mean(0); yv = oos7["y"].values
def calibrar_lofo(p, metodo):
    q = p.copy()
    for v in oos7["particion"].unique():
        m = (oos7["particion"] == v).values
        q[m] = M.Calibrador(metodo).ajustar(p[~m], yv[~m]).aplicar(p[m])
    return q
mask_fwd = oos7["particion"].isin(["2024", "2025"]).values           # el encadenamiento no calibra el primer pliegue
filas = []
for nombre, q in [("Sin calibrar", p_ens), ("Platt · adelante", calibrar_adelante(oos7, p_ens, "platt")),
                  ("Temperatura · adelante", calibrar_adelante(oos7, p_ens, "temperatura")),
                  ("Platt · dejar un año fuera", calibrar_lofo(p_ens, "platt")), ("Temperatura · dejar un año fuera", calibrar_lofo(p_ens, "temperatura"))]:
    filas.append({"Calibración": nombre, "LL 2024-2025": E.bce(yv[mask_fwd], q[mask_fwd]), "Brier 2024-2025": np.mean((q[mask_fwd] - yv[mask_fwd]) ** 2),
                  "ECE 2024-2025": E.ece(yv[mask_fwd], q[mask_fwd]), "LL 2023-2025": E.bce(yv, q), "ECE 2023-2025": E.ece(yv, q)})
CAL = pd.DataFrame(filas).set_index("Calibración")
tabla(CAL, dec=3)
par_anio = pd.DataFrame({v: (lambda c: {"a (escala)": c.a, "b (sesgo)": c.b, "Prevalencia C": yv[(oos7.particion == v).values].mean()})(
    M.Calibrador("platt").ajustar(p_ens[(oos7.particion == v).values], yv[(oos7.particion == v).values])) for v in sorted(oos7.particion.unique())})
print("Parámetros de Platt ajustados en cada año por separado (muestran si el sesgo es estable):")
tabla(par_anio, dec=3)
mejor_cal = CAL.iloc[1:3]["LL 2024-2025"].idxmin()
CALIBRACION = ("platt" if "Platt" in mejor_cal else "temperatura") if CAL.loc[mejor_cal, "LL 2024-2025"] < CAL.loc["Sin calibrar", "LL 2024-2025"] - 0.005 else "ninguna"
DECISIONES.append({"Intervención": "I8 calibración", "Referencia": "sin calibrar", "ΔLog-loss": CAL.loc[mejor_cal, "LL 2024-2025"] - CAL.loc["Sin calibrar", "LL 2024-2025"],
                   "EE": np.nan, "ΔAUC": 0.0, "Decisión": "se retiene" if CALIBRACION != "ninguna" else "no se retiene (no mejora sin mirar el futuro)"})

# Umbral: ¿el umbral óptimo es estable entre años?
from sklearn.metrics import matthews_corrcoef
umbrales = np.round(np.linspace(.1, .9, 81), 2)
U = pd.DataFrame({v: [matthews_corrcoef(yv[(oos7.particion == v).values], p_ens[(oos7.particion == v).values] >= u) for u in umbrales]
                  for v in sorted(oos7.particion.unique())}, index=umbrales)
U["Agregado 2023-2025"] = [matthews_corrcoef(yv, p_ens >= u) for u in umbrales]
U_OPT = U.idxmax()
print("Umbral que maximiza el MCC en cada año:", U_OPT.round(2).to_dict())
U_AGR = float(U_OPT["Agregado 2023-2025"]); UMBRAL = 0.5
print(f"MCC con umbral 0,5: {U.loc[0.5, 'Agregado 2023-2025']:.3f} | con el óptimo agregado {U_AGR:.2f}: {U['Agregado 2023-2025'].max():.3f}")

fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
ax = axes[0]; ax.plot([0, 1], [0, 1], color=EJE, lw=1)
for nombre, q, color in [("Sin calibrar", p_ens, COL["u3"]), ("Platt · adelante", calibrar_adelante(oos7, p_ens, "platt"), COL["logit"]), ("Persistencia", PERS["p"].values, COL["pers"])]:
    yy_ = yv if nombre != "Persistencia" else PERS["y"].values
    fr, mp = calibration_curve(yy_, q, n_bins=5, strategy="quantile")
    ax.plot(mp, fr, marker="o", color=color, mec=FONDO, mew=1.5, label=f"{nombre} (ECE {E.ece(yy_, q):.3f})")
ax.set_xlabel("P(C) media predicha"); ax.set_ylabel("Fracción observada de C"); ax.set_title("(a) Confiabilidad · origen móvil 2023-2025"); ax.legend(fontsize=8)
ax = axes[1]
for v, color in zip(sorted(oos7.particion.unique()), [COL["mlp"], COL["logit"], COL["test"]]):
    ax.plot(U.index, U[v], color=color, lw=1.4, label=f"{v} (óptimo {U_OPT[v]:.2f})")
ax.plot(U.index, U["Agregado 2023-2025"], color=TINTA, lw=2, label="agregado"); ax.axvline(.5, color=EJE, lw=1, ls="--")
ax.set_xlabel("Umbral sobre P(C)"); ax.set_ylabel("MCC"); ax.set_title("(b) El umbral óptimo cambia con el año"); ax.legend(fontsize=8)
fig.tight_layout(); guardar(fig, "fig06_calibracion_umbral"); plt.show()
print("Calibración adoptada:", CALIBRACION, "| umbral operativo:", UMBRAL)


# ======== Celda de código 19: 6.11 Tabla de decisiones y escalera de mejora ========
# =============================================================================
# 6.11 Tabla de decisiones y escalera de mejora
# =============================================================================
DEC = pd.DataFrame(DECISIONES)
display(DEC.style.format({"ΔLog-loss": "{:+.4f}", "EE": "{:.4f}", "ΔAUC": "{:+.4f}"}, na_rep="–").hide(axis="index")
        .map(lambda v: f"color: {COL['u3']}; font-weight: 600" if isinstance(v, str) and v.startswith("se retiene") else "", subset=["Decisión"]))
ESCALERA_T = [(e, k) for e, k in ESCALERA]
filas = []
for etiqueta, k in ESCALERA_T:
    ps = RES[k]["por_semilla"]
    filas.append({"Paso": etiqueta, "LL media": ps["Log-loss"].mean(), "LL DE": ps["Log-loss"].std(ddof=1), "AUC": ps["ROC-AUC"].mean(), "Brier": ps["Brier"].mean(),
                  "Transiciones acertadas": ps["Transiciones acertadas"].mean()})
m10 = E.metricas(yv, p_ens, transicion=oos7["transicion"])
filas.append({"Paso": f"+ I7 ensamble ({K_ENS})", "LL media": m10["Log-loss"], "LL DE": 0, "AUC": m10["ROC-AUC"], "Brier": m10["Brier"], "Transiciones acertadas": m10["Transiciones acertadas"]})
ESC = pd.DataFrame(filas).drop_duplicates("Paso").set_index("Paso")
tabla(ESC, dec=3)

fig, axes = plt.subplots(1, 2, figsize=(14, 4.4))
for ax, col, et in [(axes[0], "LL media", "Log-loss (↓ mejor)"), (axes[1], "AUC", "ROC-AUC (↑ mejor)")]:
    x = np.arange(len(ESC))
    ax.bar(x, ESC[col], color=[COL["mlp"]] + [COL["u3"]] * (len(ESC) - 1), yerr=ESC["LL DE"] if col == "LL media" else None, ecolor=TINTA2, capsize=3, width=.62)
    for xi, v in zip(x, ESC[col]):
        ax.text(xi, v + (0.006 if col == "LL media" else 0.003), f"{v:.3f}", ha="center", fontsize=8.5)
    ref_p = M_PERS["Log-loss" if col == "LL media" else "ROC-AUC"]
    ax.axhline(ref_p, color=COL["pers"], ls="--", lw=1.2); ax.text(len(ESC) - .5, ref_p, "persistencia", ha="right", va="bottom", fontsize=8)
    ax.set_xticks(x, ESC.index, fontsize=8.5, rotation=15); ax.set_ylabel(et)
    ax.set_ylim((0.25, max(ESC[col].max(), ref_p) * 1.12) if col == "LL media" else (0.88, 1.0))
axes[0].set_title("(a) Escalera de mejora · log-loss (media ± DE entre semillas)"); axes[1].set_title("(b) ROC-AUC")
fig.tight_layout(); guardar(fig, "fig07_escalera_mejora"); plt.show()


# ======== Celda de código 20: 7.1 Ablaciones de preprocesamiento (5 semillas, origen móvil 2023-2025) ========
# =============================================================================
# 7.1 Ablaciones de preprocesamiento (5 semillas, origen móvil 2023-2025)
# =============================================================================
t0 = time.time()
REF_CTRL = P.Especificacion("ctrl: referencia", VARS_FINAL, CFG_I6)
evaluar(REF_CTRL)
evaluar(replace(REF_CTRL, nombre="ctrl: sin estandarización", escalar=False))
evaluar(replace(REF_CTRL, nombre="ctrl: escalador con fuga", fuga_escalador=True))
evaluar(replace(REF_CTRL, nombre="ctrl: sin pesos de clase", red=replace(CFG_I6, pesos_clase=False)))
d_nolog = datos.copy()
for x_ in ["x1_eventos", "x2_fatalidades", "x4_coca"]:
    d_nolog[x_] = np.expm1(datos[x_])
evaluar(replace(REF_CTRL, nombre="ctrl: sin logaritmos"), base=d_nolog)
V.ALFA = 1e-9
d_cruda, S_cruda, _ = V.construir_dataset(acled, actores=ACTORES)
V.ALFA = 4.0
evaluar(replace(REF_CTRL, nombre="ctrl: sin contracción"), base=d_cruda, seq=S_cruda)
d_alt, S_alt, _ = V.construir_dataset(acled, mapa=C.MAPA_ALTERNATIVO, actores=ACTORES)
evaluar(replace(REF_CTRL, nombre="ctrl: mapeo alternativo"), base=d_alt, seq=S_alt)

# Etiquetas barajadas: se entrena con y permutada y se evalúa contra la etiqueta real
rng_p = np.random.default_rng(0); idx_p = rng_p.permutation(len(datos))
d_perm = datos.copy(); d_perm["y"] = datos["y"].values[idx_p]; d_perm["s_C"] = datos["s_C"].values[idx_p]
filas = []
for s in SEM5:
    sp = replace(REF_CTRL, semillas=(s,)); ps, ys, ts = [], [], []
    for v, tr, te in E.particiones_origen_movil(datos):
        m_, e_ = P.entrenar_especificacion(d_perm, SEQ, sp, tr)
        ps.append(P.predecir_especificacion(m_, e_, datos, SEQ, sp, te)); ys.append(datos.loc[te, "y"].values); ts.append(datos.loc[te, "transicion"].values)
    filas.append({"semilla": s, **E.metricas(np.concatenate(ys), np.concatenate(ps), transicion=np.concatenate(ts))})
RES["ctrl: etiquetas barajadas"] = {"por_semilla": pd.DataFrame(filas).set_index("semilla"), "spec": REF_CTRL}

# Umbral de confiabilidad: se entrena con MIN_EVENTOS = 5, 10, 20 y se evalúa en las mismas filas (>= 20 eventos)
d5, S5, _ = V.construir_dataset(acled, min_eventos=5, actores=ACTORES)
MINEV = []
for m_ev in (5, 10, 20):
    for s in SEM5:
        sp = replace(REF_CTRL, semillas=(s,)); ps, ys = [], []
        for v in C.ANIOS_SELECCION:
            tr = ((d5.anio < v) & (d5.eventos >= m_ev)).values; te = ((d5.anio == v) & (d5.eventos >= 20)).values
            m_, e_ = P.entrenar_especificacion(d5, S5, sp, tr)
            ps.append(P.predecir_especificacion(m_, e_, d5, S5, sp, te)); ys.append(d5.loc[te, "y"].values)
        MINEV.append({"MIN_EVENTOS": m_ev, "semilla": s, "filas de entrenamiento (último pliegue)": int(tr.sum()),
                      "LL": E.bce(np.concatenate(ys), np.concatenate(ps)), "AUC": roc_auc_score(np.concatenate(ys), np.concatenate(ps))})
MINEV = pd.DataFrame(MINEV)
print(f"Controles en {time.time() - t0:.0f} s")

ctrl_k = ["ctrl: sin estandarización", "ctrl: sin logaritmos", "ctrl: sin pesos de clase", "ctrl: sin contracción",
          "ctrl: escalador con fuga", "ctrl: mapeo alternativo", "ctrl: etiquetas barajadas"]
CTRL = pd.DataFrame([decidir(k, "ctrl: referencia") for k in ctrl_k]).set_index("Intervención")[["ΔLog-loss", "EE", "ΔAUC"]]
CTRL["LL media"] = [RES[k]["por_semilla"]["Log-loss"].mean() for k in ctrl_k]
CTRL["AUC media"] = [RES[k]["por_semilla"]["ROC-AUC"].mean() for k in ctrl_k]
ll_ref = RES["ctrl: referencia"]["por_semilla"]["Log-loss"].mean()
mv = MINEV.groupby("MIN_EVENTOS")[["LL", "AUC", "filas de entrenamiento (último pliegue)"]].mean()
print(f"Referencia: LL {ll_ref:.3f} | AUC {RES['ctrl: referencia']['por_semilla']['ROC-AUC'].mean():.3f}")
tabla(CTRL, dec=4)
print("Umbral de confiabilidad (evaluación común: filas con >= 20 eventos):")
tabla(mv, dec=3)


# ======== Celda de código 21: 7.2 Impacto en el entrenamiento: curvas de pérdida con y sin cada transformación (pliegue 2025) ========
# =============================================================================
# 7.2 Impacto en el entrenamiento: curvas de pérdida con y sin cada transformación (pliegue 2025)
# =============================================================================
def curva_entrenamiento(base, spec, epocas=160, semilla=1):
    tr, va = (base.anio < 2025).values, (base.anio == 2025).values
    esc = M.Escalador().ajustar(base.loc[tr, spec.variables].values)
    if not spec.escalar:
        esc.mu, esc.sd = np.zeros_like(esc.mu), np.ones_like(esc.sd)
    Xa, Xv = esc.transformar(base.loc[tr, spec.variables].values), esc.transformar(base.loc[va, spec.variables].values)
    y_obj = base.loc[tr, "s_C"].values if spec.red.objetivo == "blando" else base.loc[tr, "y"].values
    r = M.entrenar(Xa, y_obj, spec.red, semilla=semilla, y_duro=base.loc[tr, "y"].values, eventos=base.loc[tr, "eventos"].values,
                   comp=base.loc[tr, V.COMPOSICION].values, val=(Xv, base.loc[va, "y"].values), diagnostico=True, epocas=epocas)
    return r.historial
curvas_c = {"Referencia (estandarizado, log, contraído)": curva_entrenamiento(datos, REF_CTRL),
            "Sin estandarización": curva_entrenamiento(datos, replace(REF_CTRL, escalar=False)),
            "Sin logaritmos": curva_entrenamiento(d_nolog, REF_CTRL),
            "Sin contracción": curva_entrenamiento(d_cruda, REF_CTRL)}
EP_560 = math.ceil(560 / math.ceil((datos.anio < 2025).sum() / CFG_I6.batch))
fig, axes = plt.subplots(1, 3, figsize=(16, 4.3), gridspec_kw={"width_ratios": [1, 1, 1.1]})
colores_c = [COL["u3"], COL["A"], COL["logit"], COL["B"]]
for (nombre, H), color in zip(curvas_c.items(), colores_c):
    axes[0].plot(H["epoca"], H["bce_tr"], color=color, lw=1.6, label=nombre)
    axes[1].plot(H["epoca"], H["bce_va"], color=color, lw=1.6, label=nombre)
for ax, t in zip(axes[:2], ["(a) BCE de entrenamiento (convergencia)", "(b) BCE de validación 2025"]):
    ax.axvline(EP_560, color=TINTA, lw=1, ls=":"); ax.text(EP_560, ax.get_ylim()[1] * .98, " 560 pasos", fontsize=8, va="top")
    ax.set_xlabel("Época"); ax.set_title(t); ax.set_ylabel("BCE")
axes[0].legend(fontsize=7.5)
ax = axes[2]
orden = CTRL["ΔLog-loss"].sort_values().index
ax.barh(range(len(orden)), CTRL.loc[orden, "ΔLog-loss"], xerr=CTRL.loc[orden, "EE"], color=[COL["A"] if v > 0 else COL["u3"] for v in CTRL.loc[orden, "ΔLog-loss"]],
        error_kw=dict(ecolor=TINTA2, lw=1), height=.6)
ax.axvline(0, color=TINTA, lw=1); ax.set_yticks(range(len(orden)), [o.replace("ctrl: ", "") for o in orden], fontsize=8.5)
ax.set_xlabel("Δ log-loss frente a la referencia (+ = peor)"); ax.set_title("(c) Efecto de cada control (media ± EE, 5 semillas)")
fig.tight_layout(); guardar(fig, "fig08_controles_datos"); plt.show()
conv = pd.DataFrame({k: {"BCE train época 10": H["bce_tr"][9], f"BCE train época {EP_560}": H["bce_tr"][EP_560 - 1],
                         f"BCE val época {EP_560}": H["bce_va"][EP_560 - 1], "BCE val mínima": min(H["bce_va"])} for k, H in curvas_c.items()}).T
tabla(conv, dec=3)


# ======== Celda de código 22: 8.1 Especificación final, resumen de Keras y configuración exportada ========
# =============================================================================
# 8.1 Especificación final, resumen de Keras y configuración exportada
# =============================================================================
FINAL = P.Especificacion("MLP U3", VARS_FINAL, CFG_I6, semillas=tuple(range(1, K_ENS + 1)))
miembro = M.construir_red(len(VARS_FINAL), FINAL.red)
miembro.summary()
N_PAR_FINAL = M.n_parametros(miembro)
n_tr_2025 = int((datos.anio <= 2025).sum())
print(f"\nParámetros por red: {N_PAR_FINAL} | redes en el ensamble: {K_ENS} | filas 2021-2025: {n_tr_2025} | "
      f"parámetros por observación (una red): {N_PAR_FINAL / n_tr_2025:.2f}")
config_final = pd.DataFrame([
    ("Variables de entrada", f"{len(VARS_FINAL)}: " + ", ".join(VARS_FINAL)),
    ("Arquitectura", f"{len(VARS_FINAL)} → {' → '.join(map(str, FINAL.red.capas))} {FINAL.red.activacion.upper()} → 1 sigmoide" + (" + softmax(4)" if FINAL.red.tipo == "multitarea" else "")),
    ("Inicialización", "He normal (ReLU) / Glorot (tanh)"), ("Regularización", f"L2 = {FINAL.red.l2:g}; dropout = {FINAL.red.dropout:g}"),
    ("Pérdida", "BCE ponderada por clase" + (" con objetivo blando" if FINAL.red.objetivo == "blando" else "") + (f" + {FINAL.red.lambda_comp:g}·KL" if FINAL.red.tipo == "multitarea" else "")),
    ("Optimizador", f"Adam, η = {FINAL.red.lr:g}, lote {FINAL.red.batch}"), ("Presupuesto", f"{FINAL.red.pasos} pasos de gradiente por red"),
    ("Ensamble", f"{K_ENS} semillas (1-{K_ENS}), promedio de probabilidades"), ("Calibración", CALIBRACION), ("Umbral operativo", f"{UMBRAL}"),
    ("Escalado", "Estandarización ajustada solo con el entrenamiento de cada pliegue"),
], columns=["Componente", "Elección"]).set_index("Componente")
display(config_final)

CONFIG = {
    "descripcion": "Configuración del modelo final de la U3 (generada por la §8.1 del cuaderno).",
    "especificacion_final": FINAL.a_dict(),
    "especificacion_base_u2": replace(BASE_U2, semillas=(C.SEED,)).a_dict(),
    "calibracion": CALIBRACION, "umbral": UMBRAL,
    "protocolo": {"seleccion_origen_movil": list(C.ANIOS_SELECCION), "prueba": C.ANIO_PRUEBA, "min_eventos": C.MIN_EVENTOS,
                  "alfa_contraccion": V.ALFA, "semillas_intervenciones": list(SEM5)},
    "decisiones": [{k: (None if (isinstance(v, float) and np.isnan(v)) else v) for k, v in d.items()} for d in DECISIONES],
}
C.RUTA_CONFIG.write_text(json.dumps(CONFIG, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
print(f"Configuración escrita en {C.RUTA_CONFIG.name}")


# ======== Celda de código 23: 8.2 Esquema de la arquitectura final ========
# =============================================================================
# 8.2 Esquema de la arquitectura final
# =============================================================================
def dibujar_arquitectura(cfg, nombres):
    tam = [len(nombres), *cfg.capas, 1]
    fig, ax = plt.subplots(figsize=(13, 5.8)); ax.axis("off"); ax.grid(False)
    xs = np.arange(len(tam)) * 3.0
    pos = [[(x, y) for y in np.linspace(-(n - 1) / 2, (n - 1) / 2, n) * min(.62, 7 / max(n, 1))] for x, n in zip(xs, tam)]
    for a, b in zip(pos[:-1], pos[1:]):
        for (x1, y1) in a:
            for (x2, y2) in b:
                ax.plot([x1, x2], [y1, y2], color=GRID, lw=.5, zorder=1)
    nuevas_ = set(V.MEMORIA + V.DINAMICA + V.VECINDAD + V.ACTORES)
    for k, capa in enumerate(pos):
        for j, (x, y) in enumerate(capa):
            color = (COL["u3"] if nombres[j] in nuevas_ else COL["D"]) if k == 0 else (COL["C"] if k == len(pos) - 1 else COL["mlp"])
            ax.scatter(x, y, s=140, color=color, edgecolor=FONDO, lw=2, zorder=3)
    for (x, y), n in zip(pos[0], nombres):
        ax.text(x - .35, y, V.ETIQ.get(n, n), ha="right", va="center", fontsize=8.5, color=TINTA2)
    ax.text(pos[-1][0][0] + .35, 0, f"P(perfil C | x$_{{t-1}}$)\nensamble de {K_ENS} redes", ha="left", va="center", fontsize=9.5)
    top = max(max(abs(y) for _, y in c) for c in pos)
    desc = [f"Entrada\n{len(nombres)} variables\nestandarizadas"] + [f"Oculta {i}\n{u} {cfg.activacion.upper()}\nL2={cfg.l2:g}" for i, u in enumerate(cfg.capas, 1)] + ["Salida\n1 sigmoide"]
    for x, d_ in zip(xs, desc):
        ax.text(x, -top - .55, d_, ha="center", va="top", fontsize=8.5, color=TINTA2)
    ax.set_title(f"Arquitectura final U3 · {N_PAR_FINAL} parámetros por red (verde = variables nuevas de la U3)")
    ax.set_xlim(xs[0] - 3.2, xs[-1] + 3.2); ax.set_ylim(-top - 2.1, top + .6)
    return fig
fig = dibujar_arquitectura(FINAL.red, VARS_FINAL); guardar(fig, "fig09_arquitectura_final"); plt.show()


# ======== Celda de código 24: 8.3 Curvas de aprendizaje (2021-2024 -> validación 2025), 5 semillas, 3x el presupuesto ========
# =============================================================================
# 8.3 Curvas de aprendizaje (2021-2024 -> validación 2025), 5 semillas, 3x el presupuesto
# =============================================================================
CURVAS = {}
for nombre, spec_c in [("Base U2", BASE_U2), ("Final U3", FINAL)]:
    CURVAS[nombre] = [curva_entrenamiento(datos, replace(spec_c, semillas=(s,)), epocas=3 * EP_560, semilla=s) for s in SEM5]
fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
for ax, (nombre, Hs), color in zip(axes, CURVAS.items(), [COL["mlp"], COL["u3"]]):
    tr_ = np.array([h["bce_tr"] for h in Hs]); va_ = np.array([h["bce_va"] for h in Hs]); ep_ = Hs[0]["epoca"]
    ax.fill_between(ep_, tr_.min(0), tr_.max(0), color=color, alpha=.12, lw=0); ax.plot(ep_, tr_.mean(0), color=color, lw=1.6, ls=":", label="entrenamiento 2021-2024")
    ax.fill_between(ep_, va_.min(0), va_.max(0), color=COL["val"], alpha=.12, lw=0); ax.plot(ep_, va_.mean(0), color=COL["val"], lw=1.8, label="validación 2025")
    k_min = int(np.argmin(va_.mean(0))); ax.scatter(ep_[k_min], va_.mean(0)[k_min], color=TINTA, zorder=4, s=25)
    ax.axvline(EP_560, color=TINTA, lw=1, ls="--"); ax.text(EP_560, .62, " 560 pasos", fontsize=8)
    ax.set_ylim(.05, .7); ax.set_xlabel("Época"); ax.set_ylabel("BCE"); ax.legend(fontsize=8.5)
    ax.set_title(f"({'ab'[nombre == 'Final U3']}) {nombre}: BCE val. en 560 pasos = {va_.mean(0)[EP_560 - 1]:.3f} · mínimo {va_.mean(0)[k_min]:.3f} (época {ep_[k_min]})")
fig.tight_layout(); guardar(fig, "fig10_curvas_aprendizaje"); plt.show()
resumen_curvas = pd.DataFrame({n: {"BCE train (560 pasos)": np.mean([h["bce_tr"][EP_560 - 1] for h in Hs]), "BCE val (560 pasos)": np.mean([h["bce_va"][EP_560 - 1] for h in Hs]),
                                   "Brecha val − train": np.mean([h["bce_va"][EP_560 - 1] - h["bce_tr"][EP_560 - 1] for h in Hs]),
                                   "BCE val mínima (media)": float(np.min(np.array([h["bce_va"] for h in Hs]).mean(0))),
                                   "BCE val al final (3x)": np.mean([h["bce_va"][-1] for h in Hs])} for n, Hs in CURVAS.items()}).T
tabla(resumen_curvas, dec=3)


# ======== Celda de código 25: 9.1 Antes vs. después en el origen móvil 2023-2025, con IC y pruebas pareadas ========
# =============================================================================
# 9.1 Antes vs. después en el origen móvil 2023-2025, con IC y pruebas pareadas
# =============================================================================
evaluar(replace(BASE_U2, nombre="MLP base U2 (semilla 42)"), semillas=(C.SEED,))
r_fin = RES["I7 20 semillas"]; oosF = r_fin["oos"].copy(); oosF["final"] = P20[:K_ENS].mean(0)
if CALIBRACION != "ninguna":
    oosF["final"] = calibrar_adelante(oosF, oosF["final"].values, CALIBRACION)
oosF["antes"] = RES["MLP base U2 (semilla 42)"]["oos"]["p"].values
oosF["logistica"] = RES["Logística (mismas variables)"]["oos"]["p"].values
oosF["pers"] = PERS["p"].values
assert (oosF["y"].values == PERS["y"].values).all()
MODELOS_9 = {"Persistencia": "pers", "Antes: MLP base U2 (semilla 42)": "antes", "Logística (mismas variables)": "logistica", "Después: MLP U3 (ensamble)": "final"}
T91 = pd.DataFrame({k: E.metricas(oosF.y, oosF[c], transicion=oosF.transicion) for k, c in MODELOS_9.items()}).T
tabla(T91[["ROC-AUC", "Log-loss", "Brier", "ECE", "MCC", "Exact. balanceada", "Sensibilidad C", "Especificidad D", "Transiciones acertadas", "Transiciones"]], dec=3)
ic = {k: {m: "[{:.3f} – {:.3f}]".format(*E.bootstrap_ic(oosF.y, oosF[c], fn)) for m, fn in [("ROC-AUC", roc_auc_score), ("Log-loss", E.bce)]} for k, c in MODELOS_9.items()}
display(pd.DataFrame(ic).T.rename(columns=lambda c: f"IC95 {c}"))
PAR_91 = pd.DataFrame({f"Después vs. {k}": E.comparar_pareado(oosF.y, oosF.final, oosF[c], grupos=oosF.depto)
                       for k, c in [("persistencia", "pers"), ("antes (U2)", "antes"), ("logística", "logistica")]}).T
display(PAR_91[["n", "ΔLog-loss", "IC95 ΔLog-loss", "P(ΔLog-loss mejora)", "ΔBrier", "IC95 ΔBrier", "ΔAUC", "IC95 ΔAUC", "solo A acierta", "solo B acierta", "p McNemar"]]
        .style.format({"ΔLog-loss": "{:+.3f}", "ΔBrier": "{:+.3f}", "ΔAUC": "{:+.3f}", "P(ΔLog-loss mejora)": "{:.2f}", "p McNemar": "{:.3f}"}))


# ======== Celda de código 26: 9.2 Validación espacial (5 pliegues por departamento, años <= 2025) ========
# =============================================================================
# 9.2 Validación espacial (5 pliegues por departamento, años <= 2025)
# =============================================================================
evaluar(replace(FINAL, nombre="esp: final U3"), semillas=tuple(range(1, K_ENS + 1)), particiones="espacial", clave="esp: final U3")
eF = RES["esp: final U3"]["oos"]
esp_tab = pd.DataFrame({
    "Persistencia": E.metricas(PERS_E.y, PERS_E.p),
    "Antes: MLP base U2 (media 5 semillas)": {c: RES["esp: MLP base U2"]["por_semilla"][c].mean() for c in E.COLS_PRINCIPALES},
    "Logística (mismas variables)": E.metricas(RES["esp: logística referencia"]["oos"].y, RES["esp: logística referencia"]["oos"].p),
    "Después: MLP U3 (red individual, media)": {c: RES["esp: final U3"]["por_semilla"][c].mean() for c in E.COLS_PRINCIPALES},
    "Después: MLP U3 (ensamble)": E.metricas(eF.y, eF.p)}).T
tabla(esp_tab[E.COLS_PRINCIPALES], dec=3)
pers_e_al = PERS_E.set_index("fila").loc[eF["fila"], "p"].values
display(pd.DataFrame({"Después vs. persistencia (espacial)": E.comparar_pareado(eF.y, eF.p, pers_e_al, grupos=eF.depto)}).T[["n", "ΔLog-loss", "IC95 ΔLog-loss", "ΔAUC", "IC95 ΔAUC"]])


# ======== Celda de código 27: 9.3 Prueba 2026: entrenamiento con 2021-2025, una sola evaluación ========
# =============================================================================
# 9.3 Prueba 2026: entrenamiento con 2021-2025, una sola evaluación
# =============================================================================
tr26, te26 = (datos.anio <= 2025).values, (datos.anio == 2026).values
T26 = datos.loc[te26, ["depto", "departamento", "anio", "y", "s_C", "perfil", "perfil_prev", "transicion", "eventos"]].copy()
T26["pers"] = P.persistencia(datos, tr26, te26)
for nombre, s in [("antes", replace(BASE_U2, semillas=(C.SEED,))), ("logistica", LOG_F), ("final", FINAL)]:
    m_, e_ = P.entrenar_especificacion(datos, SEQ, s, tr26)
    T26[nombre] = P.predecir_especificacion(m_, e_, datos, SEQ, s, te26)
    if nombre == "final":
        MODELOS_2025, ESC_2025 = m_, e_
if CALIBRACION != "ninguna":
    cal_26 = M.Calibrador(CALIBRACION).ajustar(P20[:K_ENS].mean(0), oosF.y.values)   # calibrador con 2023-2025 (sin calibrar)
    T26["final"] = cal_26.aplicar(T26["final"])
T93 = pd.DataFrame({k: E.metricas(T26.y, T26[c], transicion=T26.transicion) for k, c in MODELOS_9.items()}).T
spec_emb = replace(RES["Embedding de departamento"]["spec"], semillas=tuple(range(1, K_ENS + 1)))
m_, e_ = P.entrenar_especificacion(datos, SEQ, spec_emb, tr26)
T26["emb"] = P.predecir_especificacion(m_, e_, datos, SEQ, spec_emb, te26)
T93.loc["(alternativa no adoptada) MLP + embedding"] = pd.Series(E.metricas(T26.y, T26.emb, transicion=T26.transicion))
doc26 = pd.read_csv(RAIZ / "referencias_u2" / "metricas_modelos.csv").set_index(["Modelo", "Conjunto"])
T93.loc["(referencia) U2 fase 2 documentado", ["ROC-AUC", "Log-loss", "Brier", "MCC", "Exactitud"]] = doc26.loc[("MLP fase 2", "Prueba"), ["ROC-AUC", "Log-loss", "Brier", "MCC", "Exactitud"]].values
tabla(T93[["ROC-AUC", "Log-loss", "Brier", "ECE", "MCC", "Exactitud", "Sensibilidad C", "Especificidad D", "Transiciones acertadas", "Transiciones"]], dec=3)
ic26 = {k: {m: "{:.3f} [{:.3f} – {:.3f}]".format(fn(T26.y, T26[c]), *E.bootstrap_ic(T26.y, T26[c], fn)) for m, fn in [("ROC-AUC", roc_auc_score), ("Log-loss", E.bce)]} for k, c in MODELOS_9.items()}
display(pd.DataFrame(ic26).T)
PAR_26 = pd.DataFrame({f"Después vs. {k}": E.comparar_pareado(T26.y, T26.final, T26[c]) for k, c in [("persistencia", "pers"), ("antes (U2)", "antes"), ("logística", "logistica")]}).T
display(PAR_26[["n", "ΔLog-loss", "IC95 ΔLog-loss", "ΔAUC", "IC95 ΔAUC", "solo A acierta", "solo B acierta", "p McNemar"]]
        .style.format({"ΔLog-loss": "{:+.3f}", "ΔAUC": "{:+.3f}", "p McNemar": "{:.3f}"}))
print("Transiciones de 2026:")
display(T26[T26.transicion][["departamento", "perfil_prev", "perfil", "s_C", "pers", "antes", "final"]].round(2).style.hide(axis="index"))


# ======== Celda de código 28: 9.4 Matrices de confusión, ROC y desempeño por año (antes vs. después vs. persistencia) ========
# =============================================================================
# 9.4 Matrices de confusión, ROC y desempeño por año (antes vs. después vs. persistencia)
# =============================================================================
fig = plt.figure(figsize=(16, 8.4)); gs = fig.add_gridspec(2, 4, hspace=.42, wspace=.32)
casos = [("Antes · 2023-2025", oosF.y, oosF.antes), ("Después · 2023-2025", oosF.y, oosF.final), ("Antes · 2026", T26.y, T26.antes), ("Después · 2026", T26.y, T26.final)]
for k, (tit, y_, p_) in enumerate(casos):
    ax = fig.add_subplot(gs[0, k]); cm = confusion_matrix(np.asarray(y_).astype(int), (np.asarray(p_) >= UMBRAL).astype(int), labels=[0, 1])
    ax.imshow(cm, cmap=G.CMAP_P, vmin=0, vmax=cm.max() * 1.3); ax.grid(False)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]}\n({cm[i, j] / max(cm[i].sum(), 1):.0%})", ha="center", va="center", fontsize=10, color="white" if cm[i, j] > cm.max() * .65 else TINTA)
    ax.set_xticks([0, 1], ["D", "C"]); ax.set_yticks([0, 1], ["D", "C"]); ax.set_xlabel("Predicho"); ax.set_ylabel("Real"); ax.set_title(tit, fontsize=9.5)
ax = fig.add_subplot(gs[1, :2])
for nombre, c, color, ls in [("Persistencia", "pers", COL["pers"], "--"), ("Antes (U2)", "antes", COL["mlp"], ":"), ("Logística", "logistica", COL["logit"], "-"), ("Después (U3)", "final", COL["u3"], "-")]:
    fpr, tpr, _ = roc_curve(oosF.y, oosF[c]); ax.plot(fpr, tpr, color=color, ls=ls, label=f"{nombre}: 2023-25 AUC {roc_auc_score(oosF.y, oosF[c]):.3f} · 2026 AUC {roc_auc_score(T26.y, T26[c]):.3f}")
ax.plot([0, 1], [0, 1], color=EJE, lw=1); ax.set_xlabel("1 − especificidad"); ax.set_ylabel("Sensibilidad C"); ax.set_title("(e) Curvas ROC · origen móvil 2023-2025"); ax.legend(fontsize=7.8, loc="lower right")
ax = fig.add_subplot(gs[1, 2:])
anual = pd.concat([oosF.assign(particion=oosF.particion.astype(str)), T26.assign(particion="2026")])
for nombre, c, color, ls in [("Persistencia", "pers", COL["pers"], "--"), ("Antes (U2)", "antes", COL["mlp"], ":"), ("Después (U3)", "final", COL["u3"], "-")]:
    s_ = anual.groupby("particion").apply(lambda g: E.bce(g.y, g[c]))
    ax.plot(s_.index, s_.values, color=color, ls=ls, marker="o", mec=FONDO, mew=1.5, ms=7, label=nombre)
ax.axvspan(2.6, 3.4, color=COL["test"], alpha=.08, lw=0); ax.text(3, ax.get_ylim()[0] + .005, "prueba única", ha="center", va="bottom", fontsize=8.5)
ax.set_ylabel("Log-loss"); ax.set_title("(f) Log-loss por año (2023-2025 origen móvil, 2026 prueba)"); ax.legend(fontsize=8.5)
guardar(fig, "fig11_antes_despues"); plt.show()


# ======== Celda de código 29: 10.1 Análisis de errores: 2023-2025 (origen móvil) + 2026 (prueba) ========
# =============================================================================
# 10.1 Análisis de errores: 2023-2025 (origen móvil) + 2026 (prueba)
# =============================================================================
fuera = pd.concat([oosF[["departamento", "depto", "anio", "y", "s_C", "perfil", "perfil_prev", "transicion", "eventos", "pers", "antes", "final"]],
                   T26[["departamento", "depto", "anio", "y", "s_C", "perfil", "perfil_prev", "transicion", "eventos", "pers", "antes", "final"]]], ignore_index=True)
fuera["acierto antes"] = (fuera.antes >= UMBRAL) == (fuera.y == 1)
fuera["acierto después"] = (fuera.final >= UMBRAL) == (fuera.y == 1)
fuera["cerca del umbral"] = (fuera.s_C - .5).abs() < .1
err = fuera[~fuera["acierto después"]].sort_values(["anio", "final"])
display(err[["departamento", "anio", "perfil_prev", "perfil", "s_C", "eventos", "pers", "antes", "final", "transicion", "cerca del umbral"]]
        .rename(columns={"pers": "P persist.", "antes": "P antes", "final": "P después"})
        .style.format({"s_C": "{:.2f}", "P persist.": "{:.2f}", "P antes": "{:.2f}", "P después": "{:.2f}"}).hide(axis="index"))
perfil_err = pd.DataFrame({
    "Errores después": [err.transicion.mean(), err["cerca del umbral"].mean(), err.eventos.median(), len(err)],
    "Aciertos después": [fuera.loc[fuera["acierto después"], "transicion"].mean(), fuera.loc[fuera["acierto después"], "cerca del umbral"].mean(),
                         fuera.loc[fuera["acierto después"], "eventos"].median(), int(fuera["acierto después"].sum())]},
    index=["% transiciones", "% a ±10 pp del umbral", "Mediana de eventos", "Filas"])
tabla(perfil_err, dec=2)
cruce = pd.crosstab(fuera["acierto antes"].map({True: "antes acierta", False: "antes falla"}),
                    fuera["acierto después"].map({True: "después acierta", False: "después falla"}))
display(cruce)
tr_f = fuera[fuera.transicion]
print(f"Transiciones 2023-2026: {len(tr_f)} | acertadas por la persistencia: 0 | antes: {int(tr_f['acierto antes'].sum())} | después: {int(tr_f['acierto después'].sum())}")
seg = fuera[fuera.depto.isin(["arauca", "casanare", "meta", "cesar", "valle del cauca", "bolivar", "choco"])].pivot_table(index="departamento", columns="anio", values="final").round(2)
obs = fuera.pivot_table(index="departamento", columns="anio", values="perfil", aggfunc="first").reindex(seg.index)
display(pd.concat({"P(C) después": seg, "Perfil observado": obs}, axis=1))

fig, ax = plt.subplots(figsize=(10.5, 6))
for anio_, color in zip(sorted(fuera.anio.unique()), [COL["mlp"], COL["logit"], COL["val"], COL["test"]]):
    s_ = fuera[fuera.anio == anio_]
    ax.scatter(s_.s_C, s_.final, s=40, color=color, edgecolor=FONDO, lw=1.2, label=str(anio_), zorder=3)
for _, r in err.iterrows():
    ax.annotate(f"{r.departamento} {r.anio}", (r.s_C, r.final), xytext=(4, 3), textcoords="offset points", fontsize=7.2, color=TINTA2)
ax.axvline(.5, color=EJE, lw=1); ax.axhline(UMBRAL, color=EJE, lw=1)
ax.axvspan(.4, .6, color=GRID, alpha=.5, lw=0, zorder=0)
ax.text(.02, .97, "falso positivo", fontsize=8.5, color=TENUE, va="top"); ax.text(.98, .03, "falso negativo", fontsize=8.5, color=TENUE, ha="right")
ax.set_xlabel("Proporción observada de eventos C en t ($s^C$); banda gris: ±10 pp del umbral"); ax.set_ylabel("P(C) del modelo final (información de t-1)")
ax.set_title("Errores del modelo final, 2023-2026 (etiquetados)"); ax.legend(fontsize=8.5, title="año", loc="center right")
fig.tight_layout(); guardar(fig, "fig12_errores"); plt.show()


# ======== Celda de código 30: 10.2 Importancia por permutación fuera de muestra (por pliegue) y dependencia parcial 2D ========
# =============================================================================
# 10.2 Importancia por permutación fuera de muestra (por pliegue) y dependencia parcial 2D
# =============================================================================
rng_i = np.random.default_rng(C.SEED); imp = {f: [] for f in VARS_FINAL}
spec_imp = replace(FINAL, semillas=SEM5)
for v, tr, te in E.particiones_origen_movil(datos):
    m_, e_ = P.entrenar_especificacion(datos, SEQ, spec_imp, tr)
    X_te = datos.loc[te, VARS_FINAL].copy(); y_te = datos.loc[te, "y"].values
    base_ll = E.bce(y_te, M.predecir_ensamble(m_, e_.transformar(X_te.values)))
    for f in VARS_FINAL:
        for _ in range(30):
            Xp = X_te.copy(); Xp[f] = rng_i.permutation(Xp[f].values)
            imp[f].append(E.bce(y_te, M.predecir_ensamble(m_, e_.transformar(Xp.values))) - base_ll)
IMP = pd.DataFrame({"ΔLog-loss media": {f: np.mean(v) for f, v in imp.items()}, "Desv. est.": {f: np.std(v) for f, v in imp.items()}}).sort_values("ΔLog-loss media")

def pdp_2d(modelos, esc, fx, fy, n=35):
    base_ = datos.loc[tr26, VARS_FINAL].reset_index(drop=True)
    gx = np.linspace(*datos[fx].quantile([.02, .98]), n); gy = np.linspace(*datos[fy].quantile([.02, .98]), n)
    GX, GY = np.meshgrid(gx, gy); rep_ = pd.concat([base_] * (n * n), ignore_index=True)
    rep_[fx] = np.repeat(GX.ravel(), len(base_)); rep_[fy] = np.repeat(GY.ravel(), len(base_))
    if isinstance(modelos, list) and hasattr(modelos[0], "predecir"):
        z = M.predecir_ensamble(modelos, esc.transformar(rep_[VARS_FINAL].values))
    else:
        z = modelos[0].predict_proba(esc.transformar(rep_[VARS_FINAL].values)[0])[:, 1]
    return gx, gy, z.reshape(n * n, len(base_)).mean(1).reshape(n, n)
LOG_MODS, LOG_ESC = P.entrenar_especificacion(datos, SEQ, LOG_F, tr26)
fig = plt.figure(figsize=(16, 5.2)); gs_ = fig.add_gridspec(1, 3, width_ratios=[1, 1.05, 1.05], wspace=.35)
ax = fig.add_subplot(gs_[0])
ax.barh(range(len(IMP)), IMP["ΔLog-loss media"], xerr=IMP["Desv. est."], color=[COL["u3"] if f not in V.BASE else COL["D"] for f in IMP.index], error_kw=dict(ecolor=TENUE))
ax.set_yticks(range(len(IMP)), [V.ETIQ[f] for f in IMP.index], fontsize=8.5); ax.axvline(0, color=EJE, lw=1)
ax.set_xlabel("Aumento de la log-loss al permutar"); ax.set_title("(a) Importancia fuera de muestra (verde = U3)")
fy_ = IMP.index[-2] if IMP.index[-1] == "m1_sC_t1" else IMP.index[-1]
fy_ = "x2b_letalidad" if fy_ in ("m1_sC_t1", "d1_sC_H2") else fy_
for k, (nombre, mods_, esc_) in enumerate([("MLP U3 (entrenado 2021-2025)", MODELOS_2025, ESC_2025), ("Logística, mismas variables", LOG_MODS, LOG_ESC)]):
    ax = fig.add_subplot(gs_[k + 1])
    gx, gy, Z = pdp_2d(mods_, esc_, "m1_sC_t1", fy_)
    cs = ax.contourf(gx, gy, Z, levels=np.linspace(0, 1, 11), cmap=G.CMAP_P, vmin=0, vmax=1); ax.contour(gx, gy, Z, levels=[.5], colors=TINTA, linewidths=1.6)
    for clase, color in [(0, COL["D"]), (1, COL["C"])]:
        s_ = datos[tr26 & (datos.y == clase).values]; ax.scatter(s_["m1_sC_t1"], s_[fy_], s=16, color=color, edgecolor=TINTA, lw=.4, zorder=3)
    ax.grid(False); ax.set_xlabel(V.ETIQ["m1_sC_t1"]); ax.set_ylabel(V.ETIQ[fy_]); ax.set_title(f"({'bc'[k]}) {nombre}")
cb = fig.colorbar(cs, ax=fig.axes[1:], fraction=.03, pad=.02); cb.set_label("P(C) parcial promedio"); cb.outline.set_visible(False)
guardar(fig, "fig13_importancia_pdp"); plt.show()
tabla(IMP.iloc[::-1].rename(index=V.ETIQ), dec=4)
print("Interacción mostrada en (b)-(c): memoria m1 ×", V.ETIQ[fy_])


# ======== Celda de código 31: 11.1 Diagrama del pipeline integrado ========
# =============================================================================
# 11.1 Diagrama del pipeline integrado
# =============================================================================
etapas = [("1 · Datos", "contrato + auditoría\npanel depto-año\netiqueta y control 2026", "dataset_modelado_u3.csv"),
          ("2 · Variables", f"{len(VARS_FINAL)} predictoras en t-1\nescalador solo con train", "configuracion.json"),
          ("3 · Selección", "origen móvil 2023-2025\nvs. persistencia y U2", "metricas_seleccion.csv"),
          ("4 · Calibración", f"método: {CALIBRACION}\numbral {UMBRAL}", "parámetros en el artefacto"),
          ("5 · Prueba", "2026, una sola vez\nmodelo hasta 2025", "metricas_prueba_2026.csv"),
          ("6 · Despliegue", f"ensamble {K_ENS} redes\n2021-2026 → 1 archivo", "perfil_amenaza_u3.keras"),
          ("7 · Proyección", "P(C) 2027, categoría,\nalerta A/B, PSI", "proyeccion_2027_u3.csv"),
          ("8 · Tarjeta", "uso previsto, límites,\nmétricas, deriva", "TARJETA_MODELO.md")]
fig, ax = plt.subplots(figsize=(16, 4.6)); ax.axis("off"); ax.grid(False)
for i, (t, d_, a) in enumerate(etapas):
    x = i * 2.0
    ax.add_patch(FancyBboxPatch((x - .05, 1.25), 1.8, 1.55, boxstyle="round,pad=0.03,rounding_size=0.12", facecolor="#eef4fc" if i not in (2, 4) else "#e8f6ef", edgecolor=COL["mlp"] if i not in (2, 4) else COL["u3"], lw=1.4))
    ax.text(x + .85, 2.55, t, ha="center", va="center", fontsize=10, fontweight="semibold")
    ax.text(x + .85, 1.85, d_, ha="center", va="center", fontsize=7, color=TINTA2)
    ax.add_patch(FancyBboxPatch((x - .02, .25), 1.74, .55, boxstyle="round,pad=0.02,rounding_size=0.08", facecolor=FONDO, edgecolor=EJE, lw=1, ls="--"))
    ax.text(x + .85, .52, a, ha="center", va="center", fontsize=6.2, color=TINTA2, family="monospace")
    ax.annotate("", xy=(x + .85, .82), xytext=(x + .85, 1.23), arrowprops=dict(arrowstyle="->", color=TENUE, lw=1))
    if i < len(etapas) - 1:
        ax.annotate("", xy=(x + 1.98, 2.03), xytext=(x + 1.72, 2.03), arrowprops=dict(arrowstyle="->", color=TINTA, lw=1.4))
ax.annotate("", xy=(4 * 2 + .85, 3.05), xytext=(2 * 2 + .85, 3.05), arrowprops=dict(arrowstyle="<->", color=COL["u3"], lw=1, connectionstyle="arc3,rad=-0.25"))
ax.text(3 * 2 + .85, 3.5, "las decisiones se congelan antes de la prueba", ha="center", fontsize=8, color=COL["u3"])
ax.text(0, -.25, "python -m perfil_amenaza.pipeline todo      ·      python -m perfil_amenaza.pipeline predecir --anio 2026      ·      python -m pytest -q tests",
        fontsize=8.5, family="monospace", color=TINTA)
ax.set_xlim(-.2, 16); ax.set_ylim(-.5, 3.7); ax.set_title("Pipeline integrado del modelo final (U3): cada etapa deja un artefacto verificable")
guardar(fig, "fig14_pipeline"); plt.show()


# ======== Celda de código 32: 11.2 Pruebas automáticas y ejecución de extremo a extremo por línea de comandos ========
# =============================================================================
# 11.2 Pruebas automáticas y ejecución de extremo a extremo por línea de comandos
# =============================================================================
entorno = {**os.environ, "PYTHONPATH": str(RAIZ / "src"), "TF_CPP_MIN_LOG_LEVEL": "3"}
t0 = time.time()
r_test = subprocess.run([sys.executable, "-m", "pytest", "-q", "tests"], cwd=RAIZ, env=entorno, capture_output=True, text=True)
print("$ python -m pytest -q tests"); print("\n".join(l for l in r_test.stdout.strip().splitlines() if "passed" in l or "failed" in l or "PASS" in l or "FAIL" in l or "::" in l))
print(f"({time.time() - t0:.0f} s)\n")
t0 = time.time()
r_cli = subprocess.run([sys.executable, "-m", "perfil_amenaza.pipeline", "todo"], cwd=RAIZ, env=entorno, capture_output=True, text=True)
print("$ python -m perfil_amenaza.pipeline todo"); print(r_cli.stdout.strip())
if r_cli.returncode != 0:
    print(r_cli.stderr[-3000:])
assert r_test.returncode == 0 and r_cli.returncode == 0


# ======== Celda de código 33: 11.3 El artefacto único: carga, estructura y consistencia con lo evaluado ========
# =============================================================================
# 11.3 El artefacto único: carga, estructura y consistencia con lo evaluado
# =============================================================================
ART, META = P.cargar_artefacto()
ruta_art = C.DIR_MODELOS / "perfil_amenaza_u3.keras"
print(f"{ruta_art.name}: {ruta_art.stat().st_size / 1024:.0f} KB | entrada: {tuple(ART.inputs[0].shape)} (variables crudas: {', '.join(META['variables'])}) | "
      f"salidas: {[tuple(o.shape) for o in ART.outputs]} | calibración: {META['calibracion']}")
capas_art = pd.DataFrame([(l.name, type(l).__name__, M.n_parametros(l) if hasattr(l, "trainable_variables") else 0) for l in ART.layers], columns=["Capa", "Tipo", "Parámetros entrenables"])
display(capas_art.style.hide(axis="index"))
met_cli_sel = pd.read_csv(C.DIR_SALIDA / "metricas_seleccion.csv", index_col=0)
met_cli_te = pd.read_csv(C.DIR_SALIDA / "metricas_prueba_2026.csv", index_col=0)
fin_cli = met_cli_sel.index[-1]
consist = pd.DataFrame({"Cuaderno (§9)": [T91.loc["Después: MLP U3 (ensamble)", "Log-loss"], T93.loc["Después: MLP U3 (ensamble)", "Log-loss"]],
                        "Pipeline CLI": [met_cli_sel.loc[fin_cli, "Log-loss"], met_cli_te.loc[fin_cli, "Log-loss"]]}, index=["Log-loss origen móvil 2023-2025", "Log-loss prueba 2026"])
consist["Diferencia"] = consist["Pipeline CLI"] - consist["Cuaderno (§9)"]
tabla(consist, dec=4)
PSI = pd.read_csv(C.DIR_SALIDA / "deriva_psi_2026.csv", index_col=0)
print("Deriva de las predictoras de la proyección (información 2026) frente al entrenamiento 2021-2026:")
tabla(PSI.rename(index=V.ETIQ), dec=3)
display(Markdown((C.DIR_SALIDA / "TARJETA_MODELO.md").read_text(encoding="utf-8")))


# ======== Celda de código 34: 12.1 Proyección 2027 con el artefacto, incertidumbre entre redes y cambio frente a la U2 ========
# =============================================================================
# 12.1 Proyección 2027 con el artefacto, incertidumbre entre redes y cambio frente a la U2
# =============================================================================
proy = pd.read_csv(C.DIR_SALIDA / "proyeccion_2027_u3.csv")
filas27, _ = V.filas_proyeccion(acled, pext, 2026, actores=ACTORES)
x27 = filas27.set_index("depto").loc[proy.depto, META["variables"]].values.astype("float32")
norm = ART.get_layer("estandarizacion")(x27)
miembros = [l for l in ART.layers if l.name.startswith("miembro_")]
P_m = np.array([np.asarray((lambda o: o[0] if isinstance(o, (list, tuple)) else o)(m(norm))).ravel() for m in miembros])
proy["DE entre redes"] = P_m.std(0)
u2p = pd.read_csv(RAIZ / "referencias_u2" / "proyeccion_2027.csv")[["depto", "P(C) 2027", "Categoría 2027"]].rename(columns={"P(C) 2027": "P(C) U2", "Categoría 2027": "Categoría U2"})
proy = proy.merge(u2p, on="depto", how="left")
proy["¿Coincide con el perfil 2026?"] = np.where(proy["perfil_info"].isna(), "–", np.where((proy["P(C) 2027"] >= .5) == (proy["perfil_info"] == "C"), "sí", "revisar"))
cols_p = ["departamento", "perfil_info", "s_C_info", "P(C) 2027", "DE entre redes", "Categoría 2027", "Categoría U2", "¿Coincide con el perfil 2026?", "s_AB observada 2025-26", "Énfasis sugerido", "confiable_info"]
display(proy[cols_p].rename(columns={"perfil_info": "Perfil 2026", "s_C_info": "s_C 2026", "confiable_info": "Datos suficientes"})
        .style.format({"s_C 2026": "{:.2f}", "P(C) 2027": "{:.2f}", "DE entre redes": "{:.3f}", "s_AB observada 2025-26": "{:.1%}"})
        .background_gradient(subset=["P(C) 2027"], cmap=G.CMAP_P, vmin=0, vmax=1).hide(axis="index"))
CAT = proy["Categoría 2027"].value_counts()
print("Categorías 2027 (U3):", CAT.to_dict(), "| U2:", proy["Categoría U2"].value_counts().to_dict())
display(pd.crosstab(proy["Categoría U2"], proy["Categoría 2027"], rownames=["U2"], colnames=["U3"]))
REVISAR = proy.loc[proy["¿Coincide con el perfil 2026?"] == "revisar", "departamento"].tolist()
ALERTA = proy.loc[proy["alerta_AB"].fillna(False), "departamento"].tolist()
print(f"Discrepancias con el perfil observado en 2026: {len(REVISAR)} ({', '.join(REVISAR)}) | alerta A/B: {', '.join(ALERTA)}")


# ======== Celda de código 35: 12.2 Mapa esquemático de la proyección 2027 (centroides) y transición U2 -> U3 ========
# =============================================================================
# 12.2 Mapa esquemático de la proyección 2027 (centroides) y transición U2 -> U3
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 9.5), gridspec_kw={"width_ratios": [1.05, 1]})
ax = axes[0]
tam = 40 + 12 * np.sqrt((proy["eventos_info"] / proy["semanas"] * 52).clip(lower=1))
sc = ax.scatter(proy.lon, proy.lat, s=tam, c=proy["P(C) 2027"], cmap=G.CMAP_P, vmin=0, vmax=1, edgecolor=FONDO, lw=2, zorder=3)
al = proy[proy["alerta_AB"].fillna(False)]
ax.scatter(al.lon, al.lat, s=tam[al.index] + 260, facecolor="none", edgecolor="#d03b3b", lw=2, zorder=2)
izq = {"valle del cauca", "risaralda", "quindio", "sucre", "cordoba", "choco"}
for _, r in proy.iterrows():
    i_ = r.depto in izq
    ax.annotate(f"{r.departamento}\n{r['P(C) 2027']:.2f}", (r.lon, r.lat), xytext=(-9 if i_ else 9, -3), textcoords="offset points",
                ha="right" if i_ else "left", fontsize=7.2, color=TINTA if r.confiable_info else TENUE)
cb = fig.colorbar(sc, ax=ax, fraction=.035, pad=.02); cb.set_label("P(perfil C) 2027"); cb.outline.set_visible(False)
ax.set_xlabel("Longitud"); ax.set_ylabel("Latitud"); ax.set_aspect("equal")
ax.set_title("(a) P(C) 2027 por departamento (aro rojo = alerta A/B; gris = pocos datos)")
ax = axes[1]
orden = proy.sort_values("P(C) 2027")
y_ = np.arange(len(orden))
ax.hlines(y_, orden["P(C) U2"], orden["P(C) 2027"], color=EJE, lw=1.5)
ax.scatter(orden["P(C) U2"], y_, color=COL["mlp"], s=30, label="U2 (modelo base, ensamble 5)", zorder=3)
ax.scatter(orden["P(C) 2027"], y_, color=COL["u3"], s=40, label="U3 (modelo final, ensamble 10)", zorder=3)
ax.errorbar(orden["P(C) 2027"], y_, xerr=orden["DE entre redes"], fmt="none", ecolor=COL["u3"], lw=1, alpha=.6)
ax.set_yticks(y_, [f"{d}  [{p if isinstance(p, str) else '–'}]" for d, p in zip(orden.departamento, orden.perfil_info)], fontsize=7.5)
for u in (.3, .5, .7):
    ax.axvline(u, color=TENUE if u != .5 else TINTA, lw=.8, ls="--")
ax.set_xlabel("P(C) 2027  [perfil observado 2026 entre corchetes]"); ax.set_title("(b) Cambio de la proyección U2 → U3"); ax.legend(fontsize=8.5, loc="lower right")
fig.tight_layout(); guardar(fig, "fig15_proyeccion_2027"); plt.show()


# ======== Celda de código 36: 16.1 Artefactos producidos (cuaderno + pipeline) ========
# =============================================================================
# 16.1 Artefactos producidos (cuaderno + pipeline)
# =============================================================================
pd.DataFrame({"tiempo total del cuaderno (min)": [round((time.time() - T0_CUADERNO) / 60, 1)]}).pipe(display)
dec_tab = pd.DataFrame(DECISIONES); dec_tab.to_csv(C.DIR_SALIDA / "decisiones_intervenciones.csv", index=False)
ESC.to_csv(C.DIR_SALIDA / "escalera_mejora.csv"); CTRL.to_csv(C.DIR_SALIDA / "controles_datos.csv"); REJ.to_csv(C.DIR_SALIDA / "rejilla_I6.csv", index=False)
T91.to_csv(C.DIR_SALIDA / "antes_despues_origen_movil.csv"); T93.to_csv(C.DIR_SALIDA / "antes_despues_prueba_2026.csv"); esp_tab.to_csv(C.DIR_SALIDA / "validacion_espacial.csv")
archivos = sorted(p for p in C.DIR_SALIDA.rglob("*") if p.is_file())
display(pd.DataFrame({"Archivo": [str(p.relative_to(C.DIR_SALIDA)) for p in archivos], "KB": [round(p.stat().st_size / 1024, 1) for p in archivos]}).style.hide(axis="index"))


# ======== Celda de código 37: Anexo A. Figuras R1-R5 para el informe (carta, 6,6" de ancho, 300 dpi) ========
# =============================================================================
# Anexo A. Figuras R1-R5 para el informe (carta, 6,6" de ancho, 300 dpi)
# =============================================================================
DIR_INF = C.DIR_SALIDA / "figuras_informe"
with mpl.rc_context({"font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5, "xtick.labelsize": 6.8, "ytick.labelsize": 6.8, "legend.fontsize": 6.8}):
    # R1 escalera
    fig, ax = plt.subplots(figsize=(6.6, 2.6)); x = np.arange(len(ESC))
    ax.bar(x, ESC["LL media"], yerr=ESC["LL DE"], color=[COL["mlp"]] + [COL["u3"]] * (len(ESC) - 1), width=.6, ecolor=TINTA2, capsize=2)
    for xi, v in zip(x, ESC["LL media"]):
        ax.text(xi, v + .005, f"{v:.3f}", ha="center", fontsize=6.5)
    ax.axhline(M_PERS["Log-loss"], color=COL["pers"], ls="--", lw=1); ax.text(len(ESC) - .5, M_PERS["Log-loss"], "persistencia", ha="right", va="bottom", fontsize=6.5)
    ax.set_xticks(x, ESC.index, rotation=12); ax.set_ylabel("Log-loss 2023-2025"); ax.set_ylim(.25, ESC["LL media"].max() * 1.1)
    ax.set_title("Escalera de mejora (media ± DE entre semillas)"); fig.tight_layout(); guardar(fig, "R1_escalera", DIR_INF, 300); plt.close(fig)
    # R2 antes vs después por año
    fig, ax = plt.subplots(figsize=(6.6, 2.5))
    for nombre, c, color, ls in [("Persistencia", "pers", COL["pers"], "--"), ("Antes (U2)", "antes", COL["mlp"], ":"), ("Después (U3)", "final", COL["u3"], "-")]:
        s_ = anual.groupby("particion").apply(lambda g: E.bce(g.y, g[c])); ax.plot(s_.index, s_.values, color=color, ls=ls, marker="o", ms=4, label=nombre)
    ax.set_ylabel("Log-loss"); ax.set_title("Log-loss por año (2026 = prueba única)"); ax.legend(ncol=3, loc="upper left")
    fig.tight_layout(); guardar(fig, "R2_antes_despues_anual", DIR_INF, 300); plt.close(fig)
    # R3 controles
    fig, ax = plt.subplots(figsize=(6.6, 2.4)); o = CTRL["ΔLog-loss"].sort_values().index
    ax.barh(range(len(o)), CTRL.loc[o, "ΔLog-loss"], xerr=CTRL.loc[o, "EE"], color=[COL["A"] if v > 0 else COL["u3"] for v in CTRL.loc[o, "ΔLog-loss"]], height=.6)
    ax.axvline(0, color=TINTA, lw=.8); ax.set_yticks(range(len(o)), [s.replace("ctrl: ", "") for s in o]); ax.set_xlabel("Δ log-loss vs. referencia (+ = peor)")
    ax.set_title("Controles de preparación de datos"); fig.tight_layout(); guardar(fig, "R3_controles", DIR_INF, 300); plt.close(fig)
    # R4 rejilla anidada
    fig, ax = plt.subplots(figsize=(3.3, 2.4)); x = np.arange(3)
    ax.bar(x - .18, ANIDADA.iloc[:3, 1], width=.34, color=COL["D"], label="2023-24 (elección)"); ax.bar(x + .18, ANIDADA.iloc[:3, 2], width=.34, color=COL["val"], label="2025 (fuera)")
    ax.set_xticks(x, ["Mínimo", "1 EE", "Vigente"]); ax.set_ylabel("Log-loss"); ax.set_title("Validación anidada (I6)"); ax.legend(loc="upper left")
    fig.tight_layout(); guardar(fig, "R4_anidada", DIR_INF, 300); plt.close(fig)
    # R5 proyección
    fig, ax = plt.subplots(figsize=(6.6, 4.2)); o = proy.sort_values("P(C) 2027"); yy = np.arange(len(o))
    ax.hlines(yy, o["P(C) U2"], o["P(C) 2027"], color=EJE, lw=1); ax.scatter(o["P(C) U2"], yy, s=10, color=COL["mlp"], label="U2"); ax.scatter(o["P(C) 2027"], yy, s=14, color=COL["u3"], label="U3")
    ax.set_yticks(yy, o.departamento, fontsize=5.6); [ax.axvline(u, color=TENUE, lw=.6, ls="--") for u in (.3, .5, .7)]
    ax.set_xlabel("P(C) 2027"); ax.set_title("Proyección 2027: U2 vs. U3"); ax.legend(loc="lower right"); fig.tight_layout(); guardar(fig, "R5_proyeccion", DIR_INF, 300); plt.close(fig)
print("Figuras del informe:", sorted(p.name for p in DIR_INF.glob("*.png")))
print(f"Figuras del cuaderno ({len(G.FIGURAS)}):", ", ".join(G.FIGURAS))
