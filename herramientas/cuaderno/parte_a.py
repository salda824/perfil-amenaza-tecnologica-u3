from nbhelp import md, code, lectura

md(r"""
# Perfil de amenaza tecnológica por departamento: del modelo base al modelo integrado
## Proyecto Integrador · Diseño, implementación y evaluación de un modelo neuronal aplicado (Unidad 3)

**Redes Neuronales · Universidad de La Sabana · Sector elegido: Seguridad y defensa**

| | |
|---|---|
| **Integrantes** | Laura Alejandra Díaz Sastoque (0000288870) · Juan Diego Saldarriaga Muñoz (0000287391) |
| **Actividad** | Actividad 5: Proyecto Integrador (evaluativa, grupal) |
| **Entregas anteriores** | Etapa 1: formulación, variables y dataset (U1) · Etapa 2: modelo base entrenado (U2) |
| **Fecha** | Noviembre de 2026 |
| **Herramientas** | Python · pandas · scikit-learn · TensorFlow / Keras 3 · paquete propio `perfil_amenaza` (este repositorio) |
""")

lectura("resumen")

md(r"""
## Cómo leer este notebook

La guía de la Unidad 3 pide un informe con ocho contenidos y la rúbrica evalúa cinco criterios. Cada fila indica dónde
está la evidencia. La matriz completa, con las cifras, está en §15.

| Contenido exigido (Producto 1) / criterio de la rúbrica | Sección |
|---|---|
| Planteamiento del problema y contexto · **Coherencia metodológica integral** | §1 (trazabilidad U1 → U2 → U3), §13 |
| Descripción del dataset y preprocesamiento · **Preparación y representación de datos** | §2 (datos y variables nuevas), §7 (controles experimentales) |
| Arquitectura del modelo y justificación · **Calidad técnica del modelo y optimización** | §6.6–§6.8, §8 |
| Proceso de entrenamiento y evaluación | §3 (protocolo), §8 (curvas), §9 |
| Optimización realizada y comparación de resultados | §5 (diagnóstico), §6 (intervenciones I1–I9), §9 (antes vs. después) |
| Análisis crítico del desempeño · **Evaluación del desempeño e interpretación** | §9, §10 |
| Integración final del modelo en un pipeline estructurado | §11 (paquete, línea de comandos, pruebas, artefacto único, deriva) |
| Conclusiones técnicas y recomendaciones futuras | §12 (decisión 2027), §13 |
| **Entregables finales y comunicación ejecutiva** (roles, bitácora, IA) | §14, §15, §16 |

> **Reproducibilidad.** El cuaderno se ejecuta de principio a fin desde la carpeta `Proyecto_Unidad_3/` (en local o en
> Colab, subiendo la carpeta). Todo el cálculo está en el paquete `src/perfil_amenaza/`, el mismo que usa la línea de
> comandos de §11. El cuaderno no duplica la lógica: la importa. Las cifras del texto corresponden a la ejecución
> guardada en este archivo (CPU, semillas fijas, determinismo de TensorFlow activado). En otro hardware pueden
> variar en el tercer decimal.
""")

md(r"""
---
# 0. Entorno, semillas y paquete del proyecto
""")

code(r"""
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
""")

md(r"""
---
# 1. Problema, contexto y trazabilidad U1 → U2 → U3

**Pregunta orientadora (Unidad 1, textual):**

> *¿Cómo orientar la adquisición, transferencia de tecnología y fabricación de sistemas de seguridad y defensa de las
> Fuerzas de Colombia en función del perfil de amenaza tecnológica predominante de los municipios colombianos según
> las variables de actividad armada histórica, geografía operacional y economías ilícitas, de modo que se optimice la
> inversión de recursos en búsqueda del bienestar de los colombianos y el desarrollo del país?*

La pregunta habla de municipios, pero el export gratuito de ACLED solo desagrega a departamento (hallazgo de la
Etapa 1). Por eso la respondemos a escala departamental y dejamos la migración a municipio como primera recomendación
(R1, §13).

**Formulación operativa (igual a la U2).** Para cada departamento $d$ y año $t$, la etiqueta es

$$
s^{C}_{d,t} = \frac{\#\{\text{eventos de subtipos C en } d,t\}}{\#\{\text{eventos en } d,t\}},\qquad
y_{d,t} = \mathbb{1}\left[s^{C}_{d,t} \ge 0.5\right],\qquad
\hat p_{d,t} = f_{\boldsymbol\theta}\left(\mathbf{x}_{d,t-1}\right) = P\left(y_{d,t} = 1 \mid \mathbf{x}_{d,t-1}\right).
$$

El modelo responde la pregunta que enfrenta quien planea la inversión: *con lo observado hasta el cierre del año,
¿qué perfil tendrá cada territorio el año siguiente?* El rezago $t-1 \rightarrow t$ evita la circularidad entre
predictoras y etiqueta (§1.1 de la U2).

## 1.1 Qué dejó cada etapa y qué cambia en esta

| Etapa | Resultado | Hallazgo que condiciona la siguiente |
|---|---|---|
| **U1** Formulación | Perfiles A/B/C/D con la regla del 50 %; 7 variables; ACLED agregado a departamento | Ningún departamento es A o B dominante → clasificación **C vs. D**; 33 filas no bastan → panel departamento–año |
| **U2** Modelo base | MLP 7 → 16 → 1 (145 parámetros); AUC temporal 0,89; equivalente a la logística | **La inercia domina**: la persistencia (AUC 0,965; log-loss 0,380) supera a la red. La memoria del perfil (E1) es la palanca más fuerte |
| **U3** Integración | Intervenciones I1–I9 derivadas del diagnóstico, controles de datos, pipeline reproducible | La referencia a superar es la persistencia (cruda y, tras la revisión, calibrada). Un modelo aporta sobre ella cuando anticipa transiciones |

## 1.2 Trazabilidad de variables (actualizada)

| ID | Variable de la U1 | Operacionalización | Fuente | U2 | U3 |
|---|---|---|---|---|---|
| $y$ | Perfil predominante | $\mathbb{1}[s^C_{d,t} \ge 0{,}5]$, mapeo de subtipos de la U1 | ACLED | ✅ | ✅ sin cambios + control de la etiqueta parcial de 2026 (§2.3) |
| $x_1$, $x_2$, $x_{2b}$ | Eventos, fatalidades, letalidad | $\log(1+\cdot)$ anualizado en $t-1$; fatalidades/evento | ACLED | ✅ | ✅ |
| $x_3$ | Grupos armados activos | $\log(1+$actores no estatales distintos en $t-1)$ | UCDP GED 26.1 (sustituye a ACLED por evento) | ⏳ | ✅ implementada y evaluada (**I3**, §6.3) |
| $x_4$, $x_5$, $x_6$, $x_7$ | Coca, bosque, distancia, IPM | Igual que la U2 | SIMCI-ODC, GFW, centroides, DANE | ✅ | ✅ |
| $m_1$–$m_3$ | *(derivada de $x_1$: "intensidad histórica")* | Composición C contraída en $t-1$, $t-2$ y su cambio | ACLED | E1 (cruda) | **I1** (§6.1) |
| $d_1$–$d_3$ | *(nueva)* dinámica intra-anual | $s^C$ del 2.º semestre de $t-1$, tendencias semestrales | ACLED semanal | – | **I2** (§6.2) |
| $v_1$, $v_2$ | *(nueva)* contexto espacial | Composición y volumen de los departamentos contiguos en $t-1$ | ACLED + contigüidad IGAC | – | **I4** (§6.4) |

## 1.3 De la predicción a la decisión (sin cambios frente a la U2)

| Perfil proyectado | Lectura operacional | Énfasis tecnológico |
|---|---|---|
| **C consolidado** ($\hat p \ge 0{,}7$) | Domina la violencia directa entre actores armados | ISR, detección temprana, movilidad protegida |
| **Transición** ($0{,}3 \le \hat p < 0{,}7$) | Perfil inestable | Portafolio flexible y monitoreo trimestral |
| **D consolidado** ($\hat p < 0{,}3$) | Protesta, disturbio y violencia mixta | Orden público y capacidades mixtas |
| **Alerta A/B** | Explosivos y drones: poco volumen, alto impacto | C-UAS y detección de AEI, cualquiera sea el perfil |
""")

md(r"""
---
# 2. Datos y preprocesamiento

## 2.1 Contrato y auditoría del export de ACLED

El export es el mismo de la U2 (semana × departamento × subtipo, hasta el 15 de agosto de 2026). La función
`datos.cargar_acled` valida el **contrato de datos**: si falta una columna, el pipeline se detiene con un mensaje
explícito en lugar de producir un modelo silenciosamente distinto.
""")

code(r"""
# =============================================================================
# 2.1 Carga, contrato de datos y auditoría
# =============================================================================
bruto = D.cargar_acled()
acled = D.filtrar_horizonte(bruto)
display(D.auditar(bruto).style.hide(axis="index"))
print(f"Horizonte de estudio: {acled['week'].min():%Y-%m-%d} → {acled['week'].max():%Y-%m-%d} | "
      f"{len(acled):,} registros semanales | {int(acled['events'].sum()):,} eventos | {acled['depto'].nunique()} departamentos")
print(f"Semanas de 2026 disponibles: {int(acled.loc[acled.anio == 2026, 'semana_anio'].max())} de 52")
""")

md(r"""
## 2.2 Panel departamento–año y variable objetivo

Se reutiliza sin cambios la construcción de la U2: mapeo de subtipos de la U1, regla del 50 % y umbral de
confiabilidad de 10 eventos (D3). Comprobamos que el panel reproduce los hallazgos de la U2 antes de agregar nada.
""")

code(r"""
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
""")

md(r"""
## 2.3 Control nuevo: ¿es confiable la etiqueta parcial de 2026?

La prueba de 2026 solo tiene **33 semanas** (enero–agosto). Si la composición de la segunda mitad del año fuera
distinta, la etiqueta parcial no representaría el perfil del año, y la prueba mediría otra cosa. Lo verificamos
con datos que sí están completos: recalculamos la etiqueta de 2021–2025 usando **solo las primeras 33 semanas** de
cada año y la comparamos con la del año completo.
""")

code(r"""
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
ax.set_title(f"Etiqueta parcial vs. completa 2021–2025: {COINCIDENCIA_PARCIAL:.1%} de coincidencia")
ax.legend(handles=[Line2D([], [], marker="o", ls="", color=COL["mlp"], label="misma etiqueta"),
                   Line2D([], [], marker="o", ls="", color=COL["A"], label="etiqueta distinta")], loc="upper left")
fig.tight_layout(); guardar(fig, "fig01_etiqueta_parcial"); plt.show()
""")

lectura("etiqueta_parcial")

md(r"""
## 2.4 Dataset de modelado: reproducción de la U2 y variables nuevas

`variables.construir_dataset` arma una fila por departamento–año etiquetable, con todas las predictoras medidas en
$t-1$. Primero comprobamos que las 7 predictoras y la etiqueta de la U2 se reproducen **exactamente**, para que toda
diferencia posterior se deba a las intervenciones y no a la reconstrucción de los datos.

**Variables nuevas.** Cada grupo responde a un hallazgo del diagnóstico de la U2:

| Grupo | Variable | Definición (todas en $t-1$) | Diagnóstico U2 que la motiva |
|---|---|---|---|
| I1 memoria | $m_1$ | $\tilde s^C_{d,t-1} = \dfrac{C_{d,t-1} + \alpha\,p_0}{n_{d,t-1} + \alpha}$, con $\alpha = 4$, $p_0 = 0{,}5$ | La persistencia acierta el 86 %; 57 % de los errores son transiciones |
| | $m_2$, $m_3$ | $\tilde s^C_{d,t-2}$ y $m_1 - m_2$ | ¿Importa la trayectoria, no solo el último año? |
| I2 dinámica | $d_1$ | $\tilde s^C$ de las últimas 26 semanas de $t-1$ | La vista anual promedia cambios de régimen recientes (p. ej., 2025) |
| | $d_2$, $d_3$ | $\log(1+n_{H2}) - \log(1+n_{H1})$ y $\tilde s^C_{H2} - \tilde s^C_{H1}$ | Tendencia de volumen y de composición dentro del año |
| I4 vecindad | $v_1$, $v_2$ | $\tilde s^C$ agregada y $\log$ de eventos medios de los departamentos contiguos | Validación espacial débil (AUC 0,80) |

**¿Por qué contraer las proporciones?** Con $n$ eventos, $s^C$ tiene un error estándar de hasta $\sqrt{0{,}25/n}$.
La etiqueta exige $n \ge 10$ en $t$, pero las predictoras de $t-1$ o de un semestre pueden venir de pocos eventos.
La contracción bayesiana (un prior Beta(2, 2), equivalente a 4 eventos "neutros") acerca a 0,5 las proporciones
basadas en pocos datos y casi no toca las bien medidas: con 100 eventos, el ajuste máximo es de 2 pp. En §7 se
mide qué pasa sin ella.
""")

code(r"""
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
""")

code(r"""
# =============================================================================
# 2.5 EDA de las variables nuevas: ¿qué información agregan sobre la inercia?
# =============================================================================
from scipy import stats
from sklearn.feature_selection import mutual_info_classif
cand = V.BASE + nuevas
rho = pd.Series({f: stats.spearmanr(datos[f], datos["y"]).correlation for f in cand})
mi = pd.Series(mutual_info_classif(datos[cand], datos["y"], random_state=C.SEED, n_neighbors=5), index=cand)

# En las transiciones, ¿el semestre reciente ya apuntaba hacia el nuevo perfil? Solo 2021-2025: 2026 es la prueba
tr_ = datos[datos.transicion & (datos.anio <= 2025)]
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
ax.scatter(tr_["m1_sC_t1"], tr_["d1_sC_H2"], s=90, facecolor="none", edgecolor=TINTA, lw=1.1, label="transición t−1 → t (2021–2025)")
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
print(f"Transiciones 2021-2025 (cambio de perfil confiable, sin la prueba 2026): {len(tr_)} | en {ANTICIPA_D1:.0%} el 2.º semestre "
      f"de t-1 ya se movía hacia el perfil de t")
tabla(pd.DataFrame({"ρ Spearman con y": rho, "Información mutua": mi}).loc[nuevas].rename(index=V.ETIQ), dec=3)
""")

lectura("eda")

md(r"""
---
# 3. Protocolo experimental de la Unidad 3

## 3.1 Reglas del protocolo

Las reglas 1 a 7 se fijaron antes de ejecutar las intervenciones. La regla 8 la agregamos durante las intervenciones,
al encontrar el caso del embedding de departamento (§6.7), y lo declaramos así. La persistencia calibrada de la regla 3
se agregó después de la revisión del informe y no cambió el modelo adoptado.

1. **Selección solo con origen móvil 2023–2025.** Para cada año $v \in \{2023, 2024, 2025\}$ se entrena con
   $t < v$ y se predice $v$ (87 predicciones fuera de muestra). El escalador se ajusta en cada pliegue solo con su
   entrenamiento (prueba automática en §11.3).
2. **2026 es la prueba final y se usa una sola vez**, con el modelo ya elegido. Hay que declararlo: la U2 ya reportó
   métricas sobre 2026, así que no es un conjunto virgen. Ninguna decisión de esta unidad se tomó mirándolo.
3. **La referencia es la persistencia** ($\hat p_{d,t} = s^C_{d,t-1}$; AUC 0,973 y log-loss 0,363 en 2023–2025).
   Como usa una proporción como si fuera una probabilidad, tras la revisión agregamos su versión calibrada: una
   logística de dos parámetros sobre $\mathrm{logit}(m_1)$, entrenada solo con los años anteriores al validado.
   Se reporta además a qué transiciones acierta cada modelo, porque la persistencia acierta 0 por construcción.
4. **Métrica de decisión: log-loss** (calidad probabilística completa), con AUC, Brier, MCC y ECE como métricas de
   apoyo. La exactitud con umbral fijo es la métrica menos estable con 29 departamentos por año (U2 §11.2).
5. **Regla de retención de una intervención.** Cada variante se entrena con las **mismas 5 semillas** que su
   referencia. Se conserva si la mejora media de la log-loss supera **un error estándar** de la diferencia pareada
   entre semillas, $\overline{\Delta LL} + \mathrm{EE}(\Delta LL) < 0$, y no baja el AUC medio más de 0,01. Si no se
   cumple, se prefiere la variante más simple (espíritu de la regla de 1 EE de la U2, D7). El EE mide el ruido del
   entrenamiento entre semillas, no el del muestreo de departamentos. En I4, cuya hipótesis es espacial, la regla se
   aplica a la validación espacial y se exige no empeorar la temporal. La regla está en `evaluacion.regla_retencion`.
   Las tablas de decisión agregan, como información y no como criterio, el IC 95 % de la misma diferencia
   remuestreando departamentos completos (`evaluacion.ic_pareado_semillas`), que sí refleja el muestreo de territorios.
6. **Presupuesto de entrenamiento fijo** de 560 pasos de gradiente (D10 de la U2), salvo cuando el presupuesto es
   la variable estudiada (§6.8). Así ninguna partición usa sus propios datos de prueba para detener el
   entrenamiento.
7. **Los controles de §7 verifican, no seleccionan.** Ponen a prueba decisiones de preprocesamiento heredadas de la
   U1 y la U2. Si un control favorece quitar una transformación, se reporta y queda como tarea para la siguiente
   iteración, pero no cambia el modelo dentro de esta. Si lo cambiara, se sumarían grados de libertad a la
   selección sobre las mismas 87 predicciones.
8. **Objetivo del proyecto por encima de la métrica (agregada durante las intervenciones, §6.7).** Una variante que gana en log-loss solo porque se parece más a
   la persistencia (menos transiciones acertadas) o porque no puede aplicarse a territorios nuevos se documenta como
   alternativa, pero no se adopta. El valor diferencial de un modelo frente a la inercia está en los cambios de
   perfil y en la migración a municipio (M3 de la U2).
""")

code(r"""
# =============================================================================
# 3.2 Herramientas de evaluación: semillas pareadas, regla de retención y caché de resultados
# =============================================================================
SEM5 = (1, 2, 3, 4, 5)
RES = {}          # nombre -> {"oos": predicciones, "por_semilla": métricas, "P": matriz semillas x filas}

def evaluar(spec, semillas=SEM5, particiones="movil", base=None, seq=None, clave=None):
    # La lógica está en pipeline.evaluar_semillas (cada semilla por separado en cada pliegue); aquí solo se guarda
    # el resultado en la caché RES para compararlo después.
    base = datos if base is None else base; seq = SEQ if seq is None else seq
    RES[clave or spec.nombre] = P.evaluar_semillas(spec, base, seq, semillas, particiones)
    return RES[clave or spec.nombre]

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
    # Regla de retención (§3.1, regla 5) con diferencias pareadas por semilla; la regla vive en el paquete.
    a, b = RES[candidata]["por_semilla"], RES[referencia]["por_semilla"]
    r = E.regla_retencion(a["Log-loss"].values, b["Log-loss"].values, a["ROC-AUC"].values, b["ROC-AUC"].values)
    # IC 95 % de la misma diferencia remuestreando departamentos (muestreo de territorios, no solo de semillas)
    oa, ob = RES[candidata].get("oos"), RES[referencia].get("oos")
    ic = "–"
    if oa is not None and ob is not None and "P" in RES[candidata] and "P" in RES[referencia] \
            and len(oa) == len(ob) and (oa["fila"].values == ob["fila"].values).all():
        c = E.ic_pareado_semillas(oa["y"], RES[candidata]["P"], RES[referencia]["P"], oa["depto"])
        ic = f"[{c['IC95 inf']:+.3f}; {c['IC95 sup']:+.3f}]".replace(".", ",")
    return {"Intervención": nombre or candidata, "Referencia": referencia, "ΔLog-loss": r["ΔLog-loss"], "EE": r["EE"],
            "IC95 Δ (depto)": ic, "ΔAUC": r["ΔAUC"], "Decisión": "se retiene" if r["retiene"] else "no se retiene"}

def persistencia_oos(particiones="movil", base=None, calibrada=False):
    base = datos if base is None else base
    fn_p = P.persistencia_calibrada if calibrada else P.persistencia
    return E.validar(base, lambda tr, te: {"p": fn_p(base, tr, te)}, E.particiones(particiones, base))

PERS = persistencia_oos()
M_PERS = E.metricas(PERS["y"], PERS["p"], transicion=PERS["transicion"])
PERS_CAL = persistencia_oos(calibrada=True)      # rival agregado tras la revisión (no interviene en la selección)
M_PERS_CAL = E.metricas(PERS_CAL["y"], PERS_CAL["p"], transicion=PERS_CAL["transicion"])
print(f"Persistencia, origen móvil 2023-2025 (n = {len(PERS)}): AUC {M_PERS['ROC-AUC']:.3f} | log-loss {M_PERS['Log-loss']:.3f} | "
      f"Brier {M_PERS['Brier']:.3f} | transiciones acertadas {M_PERS['Transiciones acertadas']} de {M_PERS['Transiciones']}")
print(f"Persistencia calibrada (logística sobre logit(m1), años anteriores): AUC {M_PERS_CAL['ROC-AUC']:.3f} | "
      f"log-loss {M_PERS_CAL['Log-loss']:.3f} | Brier {M_PERS_CAL['Brier']:.3f} | ECE {M_PERS_CAL['ECE']:.3f}")

fig, ax = plt.subplots(figsize=(13, 2.9))
for fila, v in enumerate((2023, 2024, 2025, 2026)):
    for anio in range(2021, 2027):
        rol = ("Entrenamiento", COL["train"]) if anio < v else (("Validación" if v < 2026 else "Prueba final"), COL["val"] if v < 2026 else COL["test"]) if anio == v else ("", GRID)
        ax.add_patch(Rectangle((anio - .46, 3 - fila - .4), .92, .8, facecolor=rol[1], edgecolor=FONDO, lw=2))
        if rol[0]:
            ax.text(anio, 3 - fila, rol[0], ha="center", va="center", fontsize=7.5, color="white" if rol[1] != GRID else TINTA)
ax.set_yticks(range(4), ["Prueba 2026\n(una vez)", "Pliegue 2025", "Pliegue 2024", "Pliegue 2023"])
ax.set_xticks(range(2021, 2027)); ax.set_xlim(2020.5, 2026.5); ax.set_ylim(-.6, 3.6); ax.grid(False)
ax.set_title("Protocolo U3: selección con origen móvil 2023–2025; 2026 se evalúa una sola vez al final")
guardar(fig, "fig03_protocolo"); plt.show()
""")

md(r"""
---
# 4. Línea base de la U2 reproducida

Antes de intervenir comprobamos que el modelo base de la U2 se reproduce con el paquete nuevo. Primero con el mismo
protocolo de la U2 (origen móvil 2023–2026, semilla 42, 560 pasos), contra las cifras guardadas en
`referencias_u2/`. Después, con el protocolo de esta unidad (2023–2025, 5 semillas), que es la línea "antes" de §9.
""")

code(r"""
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
""")

lectura("reproduccion")

md(r"""
---
# 5. Diagnóstico de la U2 → plan de intervención

Cada intervención responde a un hallazgo medido en la U2 y tiene una hipótesis que puede fallar. La columna "qué
la refutaría" se escribió antes de ejecutar los experimentos.

| # | Diagnóstico U2 (evidencia) | Intervención | Hipótesis | Qué la refutaría |
|---|---|---|---|---|
| I1 | La inercia domina: persistencia AUC 0,965; 57 % de los errores son transiciones | Memoria $m_1$–$m_3$ | Se acerca a la persistencia y la supera en log-loss | ΔLL dentro del ruido entre semillas |
| I2 | Solo vista anual; los errores de 2025 son escaladas recientes | Dinámica intra-anual $d_1$–$d_3$ | Mejora en transiciones | Sin ganancia en transiciones ni en LL |
| I3 | Arauca: C los seis años con P ≈ 0,36; falta $x_3$ | Actores armados (UCDP) | Corrige departamentos con actor dominante | Arauca sigue mal clasificada |
| I4 | Sin información espacial; validación espacial débil | Vecindad $v_1$, $v_2$ | Mejora la validación espacial | Sin mejora espacial |
| I5 | 20 % de las celdas a ±10 pp del umbral | Objetivo blando ($s^C$) ponderado por eventos | Menos ruido de etiqueta, mejor LL | LL peor incluso tras calibrar |
| I6 | 4 neuronas mejor que 16 a largo plazo; brecha sin cerrar | Capacidad, activación, L2 y presupuesto, con validación anidada | Baja la varianza | Lo elegido no transfiere al año siguiente |
| I7 | AUC 0,848 ± 0,024 entre semillas | Ensamble de semillas | Estabilidad | Sin reducción de dispersión |
| I8 | Subestima P(C); el umbral F2 no se transfiere | Calibración (Platt / temperatura) y umbral multianual | Mejor Brier y umbral estable | La calibración no mejora fuera de muestra |
| I9 | A/B nunca dominan: la etiqueta binaria las diluye | Salida multitarea de composición A/B/C/D | Alerta A/B modelada | No supera a la persistencia de la composición |
| – | Arquitectura MLP sin estructura temporal | Comparativos: GRU, CNN 1D, embedding de departamento | Una arquitectura secuencial aprovecha los trimestres | No ganan al MLP con las mismas variables |

Las intervenciones sobre **variables** (I1–I4) se aplican en cascada: cada una se evalúa sobre la mejor
especificación retenida hasta ese momento. Las de **modelo** (I5–I9) se evalúan después sobre el conjunto de
variables ya fijado, para no mezclar efectos.
""")
