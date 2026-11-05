"""
Figuras del informe técnico de la U3 en tamaño de impresión (carta, ancho útil 6,6", 300 dpi).

    python informe/generar_figuras.py

Usa las salidas ya guardadas en salidas_u3/ (pipeline y cuaderno) y reentrena solo lo necesario para las curvas de
aprendizaje y la dependencia parcial (mismas semillas y configuración: resultados idénticos a los del cuaderno).
"""
import sys
from dataclasses import replace
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

import math
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter, FuncFormatter
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix

from perfil_amenaza import config as C, datos as D, variables as V, modelo as M, evaluacion as E, pipeline as P, graficos as G

SAL = RAIZ / "salidas_u3"
OUT = RAIZ / "informe" / "figuras"
OUT.mkdir(parents=True, exist_ok=True)
G.aplicar_estilo()
mpl.rcParams.update({"font.size": 7.2, "axes.titlesize": 7.6, "axes.labelsize": 7, "xtick.labelsize": 6.5,
                     "ytick.labelsize": 6.5, "legend.fontsize": 6.3, "lines.linewidth": 1.4, "figure.dpi": 150})
COL, TINTA, TINTA2, TENUE, GRID, EJE, FONDO = G.COL, G.TINTA, G.TINTA2, G.TENUE, G.GRID, G.EJE, G.FONDO


def _coma(v, _pos=None):
    return f"{v:g}".replace("-", "−").replace(".", ",")


def guardar(fig, nombre):
    # Coma decimal en los ejes numéricos (los ejes con rótulos fijos no se tocan)
    for ax in fig.axes:
        for eje in (ax.xaxis, ax.yaxis):
            if isinstance(eje.get_major_formatter(), ScalarFormatter):
                eje.set_major_formatter(FuncFormatter(_coma))
    fig.savefig(OUT / f"{nombre}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  ", nombre)


cfg = C.cargar_config()
FINAL = P.Especificacion.de_dict(cfg["especificacion_final"])
BASE = P.Especificacion.de_dict(cfg["especificacion_base_u2"])
acled = D.filtrar_horizonte(D.cargar_acled())
datos, SEQ, pext = V.construir_dataset(acled)
oos = pd.read_csv(SAL / "predicciones_origen_movil.csv")
t26 = pd.read_csv(SAL / "predicciones_prueba_2026.csv")
esc = pd.read_csv(SAL / "escalera_mejora.csv", index_col=0)
ctrl = pd.read_csv(SAL / "controles_datos.csv", index_col=0)
rej = pd.read_csv(SAL / "rejilla_I6.csv")
proy = pd.read_csv(SAL / "proyeccion_2027_u3.csv")
u2p = pd.read_csv(RAIZ / "referencias_u2" / "proyeccion_2027.csv")[["depto", "P(C) 2027"]].rename(columns={"P(C) 2027": "P(C) U2"})
proy = proy.merge(u2p, on="depto", how="left")
print("Figuras del informe:")

# ---------------------------------------------------------------- R01 datos: variables nuevas y etiqueta parcial
ctrl_p, n26 = D.control_etiqueta_parcial(acled)
fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.55))
ax = axes[0]
for clase, color in [(0, COL["D"]), (1, COL["C"])]:
    s = datos[datos.y == clase]
    ax.scatter(s.m1_sC_t1, s.d1_sC_H2, s=9, color=color, edgecolor=FONDO, lw=.3, label=f"perfil {'C' if clase else 'D'} en t")
tr = datos[datos.transicion & (datos.anio <= 2025)]          # sin la prueba 2026
ax.scatter(tr.m1_sC_t1, tr.d1_sC_H2, s=34, facecolor="none", edgecolor=TINTA, lw=.7, label="transición t−1 → t (2021–2025)")
ax.plot([0, 1], [0, 1], color=EJE, lw=.8); ax.axhline(.5, color=TENUE, lw=.6, ls="--"); ax.axvline(.5, color=TENUE, lw=.6, ls="--")
ax.set_xlabel("$m_1$: proporción C de todo t−1"); ax.set_ylabel("$d_1$: proporción C 2.º semestre t−1")
ax.set_title("(a) Memoria anual vs. semestre reciente"); ax.legend(loc="upper left", handletextpad=.3)
ax = axes[1]
ax.scatter(ctrl_p.s_C_parcial, ctrl_p.s_C, s=9, c=np.where(ctrl_p.coincide, COL["mlp"], COL["A"]), edgecolor=FONDO, lw=.3)
ax.plot([0, 1], [0, 1], color=EJE, lw=.8); ax.axvline(.5, color=TENUE, lw=.6, ls="--"); ax.axhline(.5, color=TENUE, lw=.6, ls="--")
ax.set_xlabel(f"$s^C$ con las primeras {n26} semanas"); ax.set_ylabel("$s^C$ con el año completo")
ax.set_title(f"(b) Etiqueta parcial vs. completa: {ctrl_p.coincide.mean():.1%} coincide".replace(".", ","))
ax.legend(handles=[Line2D([], [], marker="o", ls="", ms=3.5, color=COL["mlp"], label="misma etiqueta"),
                   Line2D([], [], marker="o", ls="", ms=3.5, color=COL["A"], label="distinta")], loc="upper left")
fig.tight_layout(); guardar(fig, "R01_datos")

# ---------------------------------------------------------------- R02 arquitectura final
nombres = FINAL.variables; tam = [len(nombres), *FINAL.red.capas, 1]
fig, ax = plt.subplots(figsize=(3.3, 2.45)); ax.axis("off"); ax.grid(False)
xs = np.arange(len(tam)) * 2.2
pos = [[(x, y) for y in np.linspace(-(n - 1) / 2, (n - 1) / 2, n) * min(.5, 6.5 / max(n, 1))] for x, n in zip(xs, tam)]
for a, b in zip(pos[:-1], pos[1:]):
    for (x1, y1) in a:
        for (x2, y2) in b:
            ax.plot([x1, x2], [y1, y2], color=GRID, lw=.3, zorder=1)
for k, capa in enumerate(pos):
    for j, (x, y) in enumerate(capa):
        c = (COL["u3"] if nombres[j] in V.MEMORIA + V.DINAMICA else COL["D"]) if k == 0 else (COL["C"] if k == len(pos) - 1 else COL["mlp"])
        ax.scatter(x, y, s=16, color=c, edgecolor=FONDO, lw=.6, zorder=3)
for (x, y), n in zip(pos[0], nombres):
    ax.text(x - .25, y, V.ETIQ[n], ha="right", va="center", fontsize=5.3, color=TINTA2)
ax.text(pos[-1][0][0] + .25, 0, "P(C | x$_{t-1}$)\npromedio de\n10 redes", ha="left", va="center", fontsize=5.6)
top = max(abs(y) for c in pos for _, y in c)
for x, d_ in zip(xs, [f"Entrada\n{len(nombres)} variables", f"Oculta\n16 ReLU · L2", "Salida\nsigmoide"]):
    ax.text(x, -top - .35, d_, ha="center", va="top", fontsize=5.6, color=TINTA2)
ax.set_xlim(xs[0] - 3.3, xs[-1] + 1.9); ax.set_ylim(-top - 1.4, top + .3)
guardar(fig, "R02_arquitectura")

# ---------------------------------------------------------------- R03 curvas de aprendizaje base vs. final
def curva(spec, semilla, epocas):
    tr, va = (datos.anio < 2025).values, (datos.anio == 2025).values
    e = M.Escalador().ajustar(datos.loc[tr, spec.variables].values)
    Xa, Xv = e.transformar(datos.loc[tr, spec.variables].values), e.transformar(datos.loc[va, spec.variables].values)
    return M.entrenar(Xa, datos.loc[tr, "y"].values, spec.red, semilla=semilla, val=(Xv, datos.loc[va, "y"].values),
                      diagnostico=True, epocas=epocas).historial
ep560 = math.ceil(560 / math.ceil((datos.anio < 2025).sum() / 16))
fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.25), sharey=True)
for ax, (nombre, spec, color) in zip(axes, [("Modelo base U2", BASE, COL["mlp"]), ("Modelo final U3", FINAL, COL["u3"])]):
    H = [curva(spec, s, 3 * ep560) for s in (1, 2, 3, 4, 5)]
    trn, val = np.array([h["bce_tr"] for h in H]), np.array([h["bce_va"] for h in H]); ep = H[0]["epoca"]
    ax.fill_between(ep, trn.min(0), trn.max(0), color=color, alpha=.15, lw=0); ax.plot(ep, trn.mean(0), color=color, ls=":", label="entrenamiento 2021-2024")
    ax.fill_between(ep, val.min(0), val.max(0), color=COL["val"], alpha=.15, lw=0); ax.plot(ep, val.mean(0), color=COL["val"], label="validación 2025")
    ax.axvline(ep560, color=TINTA, lw=.8, ls="--")
    ax.text(ep560 + 2, .64, "560 pasos", fontsize=6)
    ax.set_title(f"({'ab'[spec is FINAL]}) {nombre}: BCE validación = " + f"{val.mean(0)[ep560 - 1]:.3f}".replace(".", ","))
    ax.set_xlabel("Época (560 pasos = 70 épocas)"); ax.set_ylim(.05, .7); ax.legend(loc="lower left")
axes[0].set_ylabel("BCE")
fig.tight_layout(); guardar(fig, "R03_curvas")

# ---------------------------------------------------------------- R04 escalera de mejora + validación anidada
K = ["capas", "L2", "activación", "pasos"]
def sel(df):
    a = df.groupby(K)["LL"].agg(media="mean", ee=lambda s: s.std(ddof=1) / np.sqrt(len(s))).reset_index()
    par = {"(4,)": 45, "(8,)": 89, "(16,)": 177, "(32,)": 353}; a["par"] = a.capas.map(par)
    m = a.loc[a.media.idxmin()]; c = a[a.media <= m.media + m.ee].sort_values(["par", "pasos", "media"]).iloc[0]
    return m, c
m_in, u_in = sel(rej[rej["año"] < 2025]); r25 = rej[rej["año"] == 2025].groupby(K)["LL"].mean(); r24 = rej[rej["año"] < 2025].groupby(K)["LL"].mean()
DEF = ("(16,)", 0.001, "relu", 560)
fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.35), gridspec_kw={"width_ratios": [1.6, 1]})
ax = axes[0]; x = np.arange(len(esc))
ax.bar(x, esc["LL media"], yerr=esc["LL DE"], color=[COL["mlp"]] + [COL["u3"]] * (len(esc) - 1), width=.6, ecolor=TINTA2, capsize=2, error_kw=dict(lw=.8))
for xi, v in zip(x, esc["LL media"]):
    ax.text(xi, v + .007, f"{v:.3f}".replace(".", ","), ha="center", fontsize=6.2)
pers = E.bce(oos.y, oos.Persistencia)
ax.axhline(pers, color=COL["pers"], ls="--", lw=.9); ax.text(len(esc) - .45, pers + .002, "persistencia", ha="right", va="bottom", fontsize=6)
ax.set_xticks(x, [s.replace("+ ", "+\n") for s in esc.index]); ax.set_ylim(.25, .41); ax.set_ylabel("Log-loss 2023-2025")
ax.set_title("(a) Escalera de mejora (media ± DE, 5 semillas)")
ax = axes[1]; x = np.arange(3)
vals_in = [m_in.media, u_in.media, r24.loc[DEF]]; vals_25 = [r25.loc[tuple(m_in[K])], r25.loc[tuple(u_in[K])], r25.loc[DEF]]
ax.bar(x - .18, vals_in, width=.34, color=COL["D"], label="2023-24 (donde se elige)"); ax.bar(x + .18, vals_25, width=.34, color=COL["val"], label="2025 (fuera)")
for xi, v in zip(x + .18, vals_25):
    ax.text(xi, v + .005, f"{v:.3f}".replace(".", ","), ha="center", fontsize=6)
ax.set_xticks(x, ["Mínimo\n2023-24", "Regla 1 EE\n2023-24", "Vigente\n(U2)"]); ax.set_ylabel("Log-loss"); ax.set_ylim(0, .62)
ax.set_title("(b) Validación anidada de la rejilla (I6)"); ax.legend(loc="upper center", ncol=2, columnspacing=.8, handlelength=1)
fig.tight_layout(); guardar(fig, "R04_optimizacion")

# ---------------------------------------------------------------- R05 controles de datos
d_nolog = datos.copy()
for x_ in ["x1_eventos", "x2_fatalidades", "x4_coca"]:
    d_nolog[x_] = np.expm1(datos[x_])
def curva_base(base, spec, escalar=True):
    tr, va = (base.anio < 2025).values, (base.anio == 2025).values
    e = M.Escalador().ajustar(base.loc[tr, spec.variables].values)
    if not escalar:
        e.mu, e.sd = np.zeros_like(e.mu), np.ones_like(e.sd)
    Xa, Xv = e.transformar(base.loc[tr, spec.variables].values), e.transformar(base.loc[va, spec.variables].values)
    return M.entrenar(Xa, base.loc[tr, "y"].values, spec.red, semilla=1, val=(Xv, base.loc[va, "y"].values), diagnostico=True, epocas=160).historial
fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.4), gridspec_kw={"width_ratios": [1, 1.25]})
ax = axes[0]
for nombre, H, color in [("Referencia", curva_base(datos, FINAL), COL["u3"]), ("Sin estandarización", curva_base(datos, FINAL, False), COL["A"]),
                         ("Sin logaritmos", curva_base(d_nolog, FINAL), COL["logit"])]:
    ax.plot(H["epoca"], np.clip(H["bce_tr"], 0, 1.2), color=color, label=nombre)
    ax.plot(H["epoca"], np.clip(H["bce_va"], 0, 1.2), color=color, ls=":", lw=1)
ax.set_ylim(0, 1.2); ax.set_xlabel("Época"); ax.set_ylabel("BCE (recortada en 1,2)")
ax.set_title("(a) BCE por época (···: validación 2025)"); ax.legend(loc="center right")
ax = axes[1]
o = ctrl["ΔLog-loss"].sort_values().index
v = ctrl.loc[o, "ΔLog-loss"].clip(upper=.6)
ax.barh(range(len(o)), v, xerr=ctrl.loc[o, "EE"].where(ctrl.loc[o, "ΔLog-loss"] < .6, 0), height=.6,
        color=[COL["A"] if x_ > 0 else COL["u3"] for x_ in ctrl.loc[o, "ΔLog-loss"]], error_kw=dict(ecolor=TINTA2, lw=.8))
for i, k in enumerate(o):
    val = ctrl.loc[k, "ΔLog-loss"]
    ax.text(min(val, .6) + (.01 if val >= 0 else -.01), i, f"{val:+.3f}".replace(".", ","), va="center", ha="left" if val >= 0 else "right", fontsize=5.8)
ax.axvline(0, color=TINTA, lw=.8); ax.set_yticks(range(len(o)), [s.replace("ctrl: ", "") for s in o]); ax.set_xlim(-.12, .72)
ax.set_xlabel("Δ log-loss vs. referencia (+ = peor; eje recortado)"); ax.set_title("(b) Efecto de cada control (5 semillas)")
fig.tight_layout(); guardar(fig, "R05_controles")

# ---------------------------------------------------------------- R06 antes vs. después
fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.3), gridspec_kw={"width_ratios": [1.1, 1.1, .8]})
ax = axes[0]
for nombre, c, color, ls in [("Persistencia", "Persistencia", COL["pers"], "--"), ("Persist. calibrada", "Persistencia calibrada", COL["A"], "-."),
                             ("Antes (U2)", "MLP base U2", COL["mlp"], ":"), ("Después (U3)", "MLP U3", COL["u3"], "-")]:
    f, t, _ = roc_curve(oos.y, oos[c]); ax.plot(f, t, color=color, ls=ls, label=f"{nombre} {roc_auc_score(oos.y, oos[c]):.3f}".replace(".", ","))
ax.plot([0, 1], [0, 1], color=EJE, lw=.7); ax.set_xlabel("1 − especificidad"); ax.set_ylabel("Sensibilidad C")
ax.set_title("(a) ROC 2023-2025 (AUC)"); ax.legend(loc="lower right")
ax = axes[1]
an = pd.concat([oos.assign(p_=oos.particion.astype(str)), t26.assign(p_="2026")])
for nombre, c, color, ls in [("Persistencia", "Persistencia", COL["pers"], "--"), ("Persist. calibrada", "Persistencia calibrada", COL["A"], "-."),
                             ("Antes (U2)", "MLP base U2", COL["mlp"], ":"), ("Después (U3)", "MLP U3", COL["u3"], "-")]:
    s = an.groupby("p_").apply(lambda g: E.bce(g.y, g[c])); ax.plot(s.index, s.values, color=color, ls=ls, marker="o", ms=3, label=nombre)
ax.axvspan(2.6, 3.4, color=COL["test"], alpha=.1, lw=0); ax.text(3, .215, "prueba", ha="center", fontsize=6)
ax.set_ylabel("Log-loss"); ax.set_ylim(.2, .56); ax.set_title("(b) Log-loss por año"); ax.legend(loc="upper left", fontsize=5.6)
ax = axes[2]
cm = confusion_matrix(t26.y.astype(int), (t26["MLP U3"] >= .5).astype(int), labels=[0, 1])
ax.imshow(cm, cmap=G.CMAP_P, vmin=0, vmax=cm.max() * 1.3); ax.grid(False)
for i in range(2):
    for j in range(2):
        ax.text(j, i, f"{cm[i, j]}\n({cm[i, j] / cm[i].sum():.0%})", ha="center", va="center", fontsize=6.5, color="white" if cm[i, j] > cm.max() * .65 else TINTA)
ax.set_xticks([0, 1], ["D", "C"]); ax.set_yticks([0, 1], ["D", "C"]); ax.set_xlabel("Predicho"); ax.set_ylabel("Real")
ax.set_title("(c) Después · prueba 2026")
fig.tight_layout(); guardar(fig, "R06_antes_despues")

# ---------------------------------------------------------------- R07 errores + dependencia parcial
tr26 = (datos.anio <= 2025).values
mods, esc_ = P.entrenar_especificacion(datos, SEQ, FINAL, tr26)
fx, fy = "m1_sC_t1", "x2b_letalidad"
base_ = datos.loc[tr26, FINAL.variables].reset_index(drop=True); n = 30
gx = np.linspace(*datos[fx].quantile([.02, .98]), n); gy = np.linspace(*datos[fy].quantile([.02, .98]), n)
GX, GY = np.meshgrid(gx, gy); rep_ = pd.concat([base_] * (n * n), ignore_index=True)
rep_[fx] = np.repeat(GX.ravel(), len(base_)); rep_[fy] = np.repeat(GY.ravel(), len(base_))
Z = M.predecir_ensamble(mods, esc_.transformar(rep_[FINAL.variables].values)).reshape(n * n, len(base_)).mean(1).reshape(n, n)
fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.55), gridspec_kw={"width_ratios": [1.15, 1]})
ax = axes[0]
fu = pd.concat([oos, t26.assign(particion="2026")])
for anio, color in zip(["2023", "2024", "2025", "2026"], [COL["mlp"], COL["logit"], COL["val"], COL["test"]]):
    s = fu[fu.particion.astype(str) == anio]; ax.scatter(s.s_C, s["MLP U3"], s=10, color=color, edgecolor=FONDO, lw=.3, label=anio, zorder=3)
err = fu[(fu["MLP U3"] >= .5) != (fu.y == 1)]
CORTO = {"Valle del Cauca": "Valle", "Norte de Santander": "N. Santander"}
for _, r in err.iterrows():
    if r.transicion or r.departamento in ("Casanare", "Huila"):
        ax.annotate(f"{CORTO.get(r.departamento, r.departamento)} {r.anio}", (r.s_C, r["MLP U3"]), xytext=(2, 1.5),
                    textcoords="offset points", fontsize=4.6, color=TINTA2)
ax.axvspan(.4, .6, color=GRID, alpha=.6, lw=0, zorder=0); ax.axvline(.5, color=EJE, lw=.7); ax.axhline(.5, color=EJE, lw=.7)
ax.set_xlabel("Proporción observada de eventos C en t ($s^C$)"); ax.set_ylabel("P(C) del modelo final")
ax.set_title("(a) Errores 2023-2026 (banda: ±10 pp del umbral)"); ax.legend(loc="upper left", ncol=2, columnspacing=.6)
ax = axes[1]
cs = ax.contourf(gx, gy, Z, levels=np.linspace(0, 1, 11), cmap=G.CMAP_P, vmin=0, vmax=1); ax.contour(gx, gy, Z, levels=[.5], colors=TINTA, linewidths=1)
for clase, color in [(0, COL["D"]), (1, COL["C"])]:
    s = datos[tr26 & (datos.y == clase).values]; ax.scatter(s[fx], s[fy], s=6, color=color, edgecolor=TINTA, lw=.25, zorder=3)
ax.grid(False); ax.set_xlabel("$m_1$: proporción C en t−1"); ax.set_ylabel("$x_{2b}$: letalidad (fatalidades/evento)")
ax.set_title("(b) Dependencia parcial de P(C): memoria × letalidad")
cb = fig.colorbar(cs, ax=ax, fraction=.05, pad=.02); cb.outline.set_visible(False); cb.ax.tick_params(labelsize=5.5)
fig.tight_layout(); guardar(fig, "R07_errores_pdp")

# ---------------------------------------------------------------- R08 pipeline + lazo de entrenamiento
etapas = [("1 · Datos", "contrato, auditoría,\npanel y etiqueta"), ("2 · Variables", "9 predictoras en t−1,\nescalador por pliegue"),
          ("3 · Evaluación", "origen móvil 2023-25\nvs. persistencias y U2"), ("4 · Calibración", "solo si mejora\nsin mirar el futuro"),
          ("5 · Prueba", "2026, una sola vez"), ("6 · Despliegue", "10 redes 2021-2026\n→ un archivo .keras"),
          ("7 · Proyección", "P(C) 2027, bandas,\nalerta A/B, PSI"), ("8 · Tarjeta", "uso, límites,\nmétricas, deriva")]
artef = ["dataset_modelado_u3.csv", "configuracion.json", "metricas_seleccion.csv", "(en el artefacto)", "metricas_prueba_2026.csv",
         "perfil_amenaza_u3.keras", "proyeccion_2027_u3.csv", "TARJETA_MODELO.md"]
fig, ax = plt.subplots(figsize=(6.6, 3.0)); ax.axis("off"); ax.grid(False)
for i, ((t, d_), a) in enumerate(zip(etapas, artef)):
    fila, col = divmod(i, 4); x, y = col * 1.65, (1 - fila) * 1.15
    verde = i in (2, 4)
    ax.add_patch(FancyBboxPatch((x, y + .32), 1.45, .62, boxstyle="round,pad=0.02,rounding_size=0.06", facecolor="#e8f6ef" if verde else "#eef4fc",
                                edgecolor=COL["u3"] if verde else COL["mlp"], lw=.9))
    ax.text(x + .725, y + .8, t, ha="center", va="center", fontsize=6.6, fontweight="bold")
    ax.text(x + .725, y + .55, d_, ha="center", va="center", fontsize=5.3, color=TINTA2)
    ax.text(x + .725, y + .19, a, ha="center", va="center", fontsize=4.8, color=TINTA2, family="monospace")
    if col < 3:
        ax.annotate("", xy=(x + 1.64, y + .63), xytext=(x + 1.46, y + .63), arrowprops=dict(arrowstyle="->", color=TINTA, lw=.8))
xr = 3 * 1.65 + 1.49
ax.plot([3 * 1.65 + 1.45, xr, xr, -.04, -.04], [1.78, 1.78, 1.07, 1.07, .63], color=TINTA, lw=.8)
ax.annotate("", xy=(0, .63), xytext=(-.04, .63), arrowprops=dict(arrowstyle="->", color=TINTA, lw=.8))
# Lazo de entrenamiento de cada red (etapas 3, 5 y 6): error -> gradiente -> actualización de pesos
ax.text(3.3, -.12, "Entrenamiento de cada red en las etapas 3, 5 y 6 (bucle tf.GradientTape)", ha="center", fontsize=5.9, fontweight="bold", color=TINTA)
lazo = ["x(t−1)\nestandarizado", "red 9 → 16 → 1\npropagación", "p̂ = P(C)", "pérdida: BCE\nponderada + L2", "∂L/∂θ por\nretropropagación", "Adam actualiza\npesos y sesgos"]
yl = -.72
for j, txt in enumerate(lazo):
    xj = .05 + j * 1.1
    ax.add_patch(FancyBboxPatch((xj, yl), .95, .42, boxstyle="round,pad=0.02,rounding_size=0.05", facecolor="#fdf6e6", edgecolor=COL["C"], lw=.8))
    ax.text(xj + .475, yl + .21, txt, ha="center", va="center", fontsize=4.9, color=TINTA)
    if j < len(lazo) - 1:
        ax.annotate("", xy=(xj + 1.09, yl + .21), xytext=(xj + .96, yl + .21), arrowprops=dict(arrowstyle="->", color=TINTA, lw=.7))
ax.annotate("", xy=(.05 + 1 * 1.1 + .475, yl - .01), xytext=(.05 + 5 * 1.1 + .475, yl - .01),
            arrowprops=dict(arrowstyle="->", color=COL["C"], lw=.9, connectionstyle="arc3,rad=-0.18"))
ax.text(.05 + 3 * 1.1, yl - .2, "se repite 560 pasos (lotes de 16 filas)", ha="center", va="center", fontsize=5.1, color=TINTA2)
ax.text(3.3, -1.42, "python -m perfil_amenaza.pipeline todo  ·  python -m perfil_amenaza.pipeline predecir  ·  python -m pytest -q tests (10 pruebas)",
        ha="center", fontsize=5.2, family="monospace")
ax.set_xlim(-.1, 6.65); ax.set_ylim(-1.55, 2.15)
guardar(fig, "R08_pipeline")

# ---------------------------------------------------------------- R09 proyección 2027
fig, ax = plt.subplots(figsize=(3.4, 3.9))
tam = 8 + 3.2 * np.sqrt((proy["eventos_info"] / proy["semanas"] * 52).clip(lower=1))
sc = ax.scatter(proy.lon, proy.lat, s=tam, c=proy["P(C) 2027"], cmap=G.CMAP_P, vmin=0, vmax=1, edgecolor=FONDO, lw=.8, zorder=3)
al = proy[proy["alerta_AB"].fillna(False)]
ax.scatter(al.lon, al.lat, s=tam[al.index] + 70, facecolor="none", edgecolor="#d03b3b", lw=1.1, zorder=2)
izq = {"valle del cauca", "sucre", "cordoba", "choco", "guainia", "amazonas"}
SIN_ROTULO = {"bogota", "boyaca", "caldas", "cundinamarca", "quindio", "risaralda", "santander"}   # zona andina, P < 0,1
DESPL = {"valle del cauca": (-5, -5), "arauca": (6, -1.5), "tolima": (5, -2.5)}
for _, r in proy.iterrows():
    if r.depto in SIN_ROTULO:
        continue
    i_ = r.depto in izq
    dx, dy = DESPL.get(r.depto, (-4 if i_ else 4, -1.5))
    ax.annotate(f"{r.departamento} " + f"{r['P(C) 2027']:.2f}".replace(".", ","), (r.lon, r.lat), xytext=(dx, dy), textcoords="offset points",
                ha="right" if dx < 0 else "left", fontsize=4.3, color=TINTA if r.confiable_info else TENUE)
ax.text(-75.0, 5.0, "zona andina\nP < 0,1", ha="center", va="center", fontsize=3.9, color=TENUE, style="italic")
cb = fig.colorbar(sc, ax=ax, fraction=.04, pad=.02); cb.set_label("P(perfil C) 2027", fontsize=6); cb.outline.set_visible(False); cb.ax.tick_params(labelsize=5.5)
ax.set_xlabel("Longitud"); ax.set_ylabel("Latitud"); ax.set_aspect("equal"); ax.tick_params(labelsize=5.5); ax.set_xlim(-82.5, -66.3)
ax.legend(handles=[Line2D([], [], marker="o", ls="", mfc="none", mec="#d03b3b", ms=6, label="alerta A/B"),
                   Line2D([], [], marker="o", ls="", color=TENUE, ms=3, label="rótulo gris: pocos datos")], loc="lower left", fontsize=5.2)
guardar(fig, "R09_proyeccion")
print("Listo:", OUT)
