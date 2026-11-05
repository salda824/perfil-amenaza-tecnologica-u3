# Código de las celdas de 'Proyecto_U2_Modelo_Base.ipynb' extraído en orden (generado para graphify).

# ======== Celda de código 1 (índice 4): 0.1 Librerías, semilla global y rutas de trabajo ========
# =============================================================================
# 0.1 Librerías, semilla global y rutas de trabajo
# =============================================================================
import os, sys, time, json, re, math, platform, unicodedata, warnings
from pathlib import Path
from dataclasses import dataclass, asdict, replace

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")    # silencia mensajes informativos de TensorFlow
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")   # resultados numéricos repetibles en CPU
os.environ.setdefault("KERAS_BACKEND", "tensorflow")  # el bucle de entrenamiento (§8) usa tf.GradientTape

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)   # oculta avisos de plataforma y de retrazado
import tensorflow as tf
import keras
from keras import layers
import sklearn
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.feature_selection import mutual_info_classif
from sklearn.calibration import calibration_curve
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, precision_score, recall_score,
                             f1_score, fbeta_score, matthews_corrcoef, roc_auc_score,
                             average_precision_score, log_loss, brier_score_loss, confusion_matrix,
                             roc_curve, precision_recall_curve)
from IPython.display import display, Markdown

warnings.filterwarnings("ignore")
SEED = 42
keras.utils.set_random_seed(SEED)                  # fija las semillas de Python, NumPy y TensorFlow
tf.config.experimental.enable_op_determinism()     # operaciones deterministas en TensorFlow
tf.get_logger().setLevel("ERROR"); tf.autograph.set_verbosity(0)

EN_COLAB = "google.colab" in sys.modules
RUTA_ACLED = Path("acled-data-export.csv")
DIR_EXT = Path("datos_externos")                    # caché de fuentes externas (coca, bosque)
DIR_SALIDA = Path("salidas_u2")
DIR_FIG = DIR_SALIDA / "figuras"
for carpeta in (DIR_EXT, DIR_SALIDA, DIR_FIG):
    carpeta.mkdir(parents=True, exist_ok=True)

if not RUTA_ACLED.exists() and EN_COLAB:            # en Colab se sube el CSV si no está en el entorno
    from google.colab import files
    print("Sube el archivo acled-data-export.csv exportado de ACLED:")
    files.upload()
assert RUTA_ACLED.exists(), "No se encontró acled-data-export.csv en la carpeta de trabajo."

entorno = pd.Series({
    "Python": platform.python_version(),
    "Sistema": platform.platform(terse=True),
    "NumPy / pandas": f"{np.__version__} / {pd.__version__}",
    "scikit-learn": sklearn.__version__,
    "TensorFlow / Keras": f"{tf.__version__} / {keras.__version__}",
    "Backend de Keras": keras.backend.backend(),
    "Dispositivos": ", ".join(sorted({d.device_type for d in tf.config.list_physical_devices()})),
    "Semilla global": SEED,
}, name="Valor")
display(entorno.to_frame())

# ======== Celda de código 2 (índice 5): 0.2 Estilo gráfico, paleta y funciones auxiliares ========
# =============================================================================
# 0.2 Estilo gráfico, paleta y funciones auxiliares
# =============================================================================
# Se conservan los colores de perfil de la Etapa 1 (C = ámbar, D = gris, A = rojo, B = azul).
COL = {"C": "#eda100", "D": "#888780", "A": "#e34948", "B": "#2a78d6",
       "mlp": "#2a78d6", "logit": "#eb6834", "pers": "#52514e", "mayor": "#b9b8b2",
       "train": "#2a78d6", "val": "#eb6834", "test": "#1baf7a"}
TINTA, TINTA2, TENUE, GRID, EJE, FONDO = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
mpl.rcParams.update({
    "figure.facecolor": FONDO, "axes.facecolor": FONDO, "savefig.facecolor": FONDO,
    "font.family": "sans-serif", "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial", "Liberation Sans"],
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "semibold", "axes.titlelocation": "left",
    "axes.labelcolor": TINTA2, "axes.edgecolor": EJE, "axes.linewidth": 0.8, "axes.axisbelow": True,
    "xtick.color": TENUE, "ytick.color": TENUE, "xtick.labelcolor": TINTA2, "ytick.labelcolor": TINTA2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.7, "grid.linestyle": "-",
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "lines.linewidth": 2, "lines.solid_capstyle": "round", "figure.dpi": 110,
})
CMAP_C = LinearSegmentedColormap.from_list("prop_C", ["#fdf6e6", "#f6d58a", "#eda100", "#a86f00", "#4d3300"])
CMAP_P = LinearSegmentedColormap.from_list("prob_C", ["#eef4fc", "#9ec5f4", "#3987e5", "#1c5cab", "#0d366b"])
CMAP_DIV = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#f0efec", "#e34948"])

FIGURAS = []
def guardar(fig, nombre):
    """Guarda la figura en salidas_u2/figuras (PNG, 200 dpi) para reutilizarla en el informe PDF."""
    ruta = DIR_FIG / f"{nombre}.png"
    fig.savefig(ruta, dpi=200, bbox_inches="tight")
    if ruta.name not in FIGURAS:
        FIGURAS.append(ruta.name)

def normalizar(nombre):
    """Llave de cruce entre fuentes: minúsculas, sin tildes ni signos ('Bogotá, D.C.' -> 'bogota')."""
    s = unicodedata.normalize("NFKD", str(nombre)).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"\s+", " ", re.sub(r"[^a-z ]", " ", s)).strip()
    if s.startswith("bogota"):
        return "bogota"
    if s.startswith("san andres") or s.startswith("archipielago"):
        return "san andres"
    return s

NOMBRE = {  # nombres con tildes para tablas y figuras
    "amazonas": "Amazonas", "antioquia": "Antioquia", "arauca": "Arauca", "atlantico": "Atlántico",
    "bogota": "Bogotá D.C.", "bolivar": "Bolívar", "boyaca": "Boyacá", "caldas": "Caldas", "caqueta": "Caquetá",
    "casanare": "Casanare", "cauca": "Cauca", "cesar": "Cesar", "choco": "Chocó", "cordoba": "Córdoba",
    "cundinamarca": "Cundinamarca", "guainia": "Guainía", "guaviare": "Guaviare", "huila": "Huila",
    "la guajira": "La Guajira", "magdalena": "Magdalena", "meta": "Meta", "narino": "Nariño",
    "norte de santander": "Norte de Santander", "putumayo": "Putumayo", "quindio": "Quindío",
    "risaralda": "Risaralda", "san andres": "San Andrés", "santander": "Santander", "sucre": "Sucre",
    "tolima": "Tolima", "valle del cauca": "Valle del Cauca", "vaupes": "Vaupés", "vichada": "Vichada"}

def haversine_km(lat1, lon1, lat2, lon2):
    """Distancia geodésica (km) sobre una esfera de radio 6.371 km."""
    la1, lo1, la2, lo2 = map(np.radians, (lat1, lon1, lat2, lon2))
    h = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(h))

def bce(y, p, eps=1e-7):
    """Entropía cruzada binaria media (log-loss) sin ponderar."""
    y = np.asarray(y, float); p = np.clip(np.asarray(p, float), eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

def metricas(y, p, umbral=0.5):
    """Batería de métricas para la clase positiva C (1) frente a D (0)."""
    y = np.asarray(y).astype(int); p = np.asarray(p, float); yhat = (p >= umbral).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
    dos = len(np.unique(y)) == 2
    return {"Exactitud": accuracy_score(y, yhat),
            "Exact. balanceada": balanced_accuracy_score(y, yhat),
            "Precisión C": precision_score(y, yhat, zero_division=0),
            "Sensibilidad C": recall_score(y, yhat, zero_division=0),
            "Especificidad D": tn / (tn + fp) if (tn + fp) else np.nan,
            "F1 C": f1_score(y, yhat, zero_division=0),
            "MCC": matthews_corrcoef(y, yhat),
            "ROC-AUC": roc_auc_score(y, p) if dos else np.nan,
            "PR-AUC": average_precision_score(y, p) if dos else np.nan,
            "Log-loss": bce(y, p),
            "Brier": brier_score_loss(y, p)}

def bootstrap_ic(y, p, fn, n=2000, semilla=SEED):
    """IC 95 % percentil por bootstrap estratificado por clase (conserva la prevalencia en cada réplica)."""
    rng = np.random.default_rng(semilla); y = np.asarray(y).astype(int); p = np.asarray(p, float)
    i0, i1 = np.where(y == 0)[0], np.where(y == 1)[0]
    vals = [fn(y[idx], p[idx]) for idx in (np.concatenate([rng.choice(i0, len(i0)), rng.choice(i1, len(i1))])
                                           for _ in range(n))]
    return np.percentile(vals, [2.5, 97.5])

def tabla(df, dec=3, **kw):
    """Muestra un DataFrame con formato numérico uniforme."""
    return display(df.style.format(precision=dec, thousands=".", decimal=",", na_rep="–", **kw))

print("Estilo y utilidades listos.")

# ======== Celda de código 3 (índice 9): 2.1 Carga del archivo original ========
# =============================================================================
# 2.1 Carga del archivo original
# =============================================================================
acled_bruto = pd.read_csv(RUTA_ACLED, encoding="utf-8-sig")
acled_bruto["week"] = pd.to_datetime(acled_bruto["week"])
print(f"Archivo: {RUTA_ACLED.name} | {RUTA_ACLED.stat().st_size / 1e6:.2f} MB | "
      f"{acled_bruto.shape[0]:,} filas × {acled_bruto.shape[1]} columnas")
display(acled_bruto.head(4))

# ======== Celda de código 4 (índice 10): 2.2 Auditoría de calidad: cada verificación con su implicación para el modelo ========
# =============================================================================
# 2.2 Auditoría de calidad: cada verificación con su implicación para el modelo
# =============================================================================
def auditar(df):
    semanas = np.sort(df["week"].unique())
    saltos = np.diff(semanas).astype("timedelta64[D]").astype(int)
    fuera = df.loc[df["country"] != "Colombia", "admin1"].unique()
    pob = pd.to_numeric(df["population_exposure"].astype(str).str.replace(",", "", regex=False), errors="coerce")
    col = df[df["country"] == "Colombia"]
    filas = [
        ("Registros / columnas", f"{len(df):,} / {df.shape[1]}", "Archivo ligero (≈2 MB), carga directa con pandas"),
        ("Registros fuera de Colombia", f"{(df['country'] != 'Colombia').sum()} ({', '.join(fuera)})",
         "Se excluyen: aguas internacionales, no son territorio de decisión"),
        ("Cobertura temporal", f"{df['week'].min():%Y-%m-%d} → {df['week'].max():%Y-%m-%d}",
         "Se descarta la semana 2019-12-28 (horizonte 2020+ de la U1)"),
        ("Semanas distintas / saltos ≠ 7 días", f"{len(semanas)} / {(saltos != 7).sum()}", "Serie semanal continua, sin huecos"),
        ("Valores faltantes (todas las columnas)", f"{int(df.isna().sum().sum())}", "No se requiere imputación"),
        ("Duplicados exactos", f"{int(df.duplicated().sum())}", "Sin doble conteo"),
        ("Duplicados por llave semana–depto–subtipo", f"{int(df.duplicated(['week', 'admin1', 'sub_event_type']).sum())}",
         "La llave es única: agregación segura"),
        ("Rangos: events ≥ 1 y fatalities ≥ 0", f"{bool((df['events'] >= 1).all() and (df['fatalities'] >= 0).all())}",
         f"events máx = {df['events'].max()}, fatalities máx = {df['fatalities'].max()}"),
        ("Subtipos asignados a más de un tipo", f"{int((df.groupby('sub_event_type')['event_type'].nunique() > 1).sum())}",
         "Taxonomía consistente con el codebook"),
        ("Centroides distintos por departamento (máx.)",
         f"{int(df.groupby('admin1')[['centroid_latitude', 'centroid_longitude']].nunique().max().max())}",
         "Un centroide fijo por departamento"),
        ("population_exposure no convertible a número", f"{int(pob.isna().sum())}", "Texto con separador de miles: se limpia"),
        ("Departamentos de Colombia (admin1)", f"{col['admin1'].nunique()}", "32 departamentos + Bogotá D.C."),
    ]
    return pd.DataFrame(filas, columns=["Verificación", "Resultado", "Implicación para el modelo"])

display(auditar(acled_bruto).style.hide(axis="index"))

# ======== Celda de código 5 (índice 11): 2.3 Filtrado al horizonte de estudio y conciliación con las cifras de la Etapa 1 ========
# =============================================================================
# 2.3 Filtrado al horizonte de estudio y conciliación con las cifras de la Etapa 1
# =============================================================================
acled = acled_bruto[(acled_bruto["country"] == "Colombia") & (acled_bruto["week"] >= "2020-01-01")].copy()
acled["anio"] = acled["week"].dt.year
acled["depto"] = acled["admin1"].map(normalizar)
acled["population_exposure"] = pd.to_numeric(acled["population_exposure"].astype(str).str.replace(",", "", regex=False))

corte_u1 = acled[acled["week"] <= "2026-08-01"]
conciliacion = pd.DataFrame({
    "Etapa 1 (reportado)": [17491, 33336, 13732, "2026-08-01"],
    "Export actual, mismo corte": [len(corte_u1), int(corte_u1["events"].sum()), int(corte_u1["fatalities"].sum()), "2026-08-01"],
    "Export actual, completo": [len(acled), int(acled["events"].sum()), int(acled["fatalities"].sum()),
                                f"{acled['week'].max():%Y-%m-%d}"],
}, index=["Registros semanales", "Eventos", "Fatalidades", "Última semana"])
display(conciliacion)
dif = 100 * (len(corte_u1) - 17491) / 17491
print(f"Diferencia de registros frente a la Etapa 1 con el mismo corte: {dif:+.2f} %")

# ======== Celda de código 6 (índice 13): 2.4 Cobertura de los datos: dinámica nacional y densidad departamento–año ========
# =============================================================================
# 2.4 Cobertura de los datos: dinámica nacional y densidad departamento–año
# =============================================================================
semanal = acled.groupby("week")["events"].sum()
eventos_da = acled.pivot_table(index="depto", columns="anio", values="events", aggfunc="sum", fill_value=0)
eventos_da = eventos_da.loc[eventos_da.sum(axis=1).sort_values(ascending=False).index]

fig, axes = plt.subplots(1, 2, figsize=(15, 6.2), gridspec_kw={"width_ratios": [1.25, 1]})
ax = axes[0]
ax.plot(semanal.index, semanal.values, color=COL["mlp"], lw=0.9, alpha=0.35, label="Eventos por semana")
ax.plot(semanal.index, semanal.rolling(8, center=True).mean(), color=COL["mlp"], lw=2, label="Media móvil 8 semanas")
ax.axvspan(pd.Timestamp("2021-04-28"), pd.Timestamp("2021-06-30"), color=COL["D"], alpha=0.15, lw=0)
ax.annotate("Paro Nacional 2021", xy=(pd.Timestamp("2021-05-20"), semanal.max() * 0.93), ha="left",
            xytext=(pd.Timestamp("2021-08-01"), semanal.max() * 0.93), color=TINTA2, fontsize=9,
            arrowprops=dict(arrowstyle="-", color=TENUE, lw=0.8))
ax.set_title("(a) Eventos semanales en Colombia, 2020-2026")
ax.set_ylabel("Eventos por semana"); ax.legend(loc="upper right")

ax = axes[1]
im = ax.imshow(np.log10(eventos_da.values + 1), aspect="auto", cmap=CMAP_P)
ax.set_xticks(range(eventos_da.shape[1]), eventos_da.columns)
ax.set_yticks(range(len(eventos_da)), [NOMBRE[d] for d in eventos_da.index], fontsize=7.5)
for i in range(eventos_da.shape[0]):
    for j in range(eventos_da.shape[1]):
        v = eventos_da.values[i, j]
        ax.text(j, i, f"{v:,}".replace(",", "."), ha="center", va="center", fontsize=6.2,
                color="white" if np.log10(v + 1) > 0.42 * np.log10(eventos_da.values.max() + 1) else TINTA)
ax.grid(False); ax.set_title("(b) Eventos por departamento y año")
cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02); cb.set_label("log₁₀(eventos + 1)")
cb.outline.set_visible(False)
fig.tight_layout(); guardar(fig, "fig01_cobertura_datos"); plt.show()

# ======== Celda de código 7 (índice 16): 3.1 Mapeo subtipo -> perfil (idéntico al notebook de la Etapa 1) ========
# =============================================================================
# 3.1 Mapeo subtipo -> perfil (idéntico al notebook de la Etapa 1)
# =============================================================================
MAPA_PERFIL_U1 = {
    "Remote explosive/landmine/IED": "A", "Grenade": "A", "Suicide bomb": "A", "Chemical weapon": "A",
    "Air/drone strike": "B",
    "Armed clash": "C", "Attack": "C", "Shooting": "C", "Abduction/forced disappearance": "C",
    "Looting/property destruction": "C", "Disrupted weapons use": "C",
    "Mob violence": "D", "Peaceful protest": "D", "Excessive force against protesters": "D", "Arrests": "D",
    "Government regains territory": "D", "Non-state actor overtakes territory": "D",
}
subtipos = (acled.groupby(["event_type", "sub_event_type"])[["events", "fatalities"]].sum()
            .reset_index().sort_values("events", ascending=False))
subtipos["perfil"] = subtipos["sub_event_type"].map(MAPA_PERFIL_U1).fillna("D")
subtipos["asignación"] = np.where(subtipos["sub_event_type"].isin(MAPA_PERFIL_U1), "explícita (U1)", "por defecto → D (U1)")
subtipos["% eventos"] = 100 * subtipos["events"] / subtipos["events"].sum()
subtipos["fatalidades/evento"] = subtipos["fatalities"] / subtipos["events"]
tabla(subtipos.rename(columns={"event_type": "Tipo", "sub_event_type": "Subtipo", "events": "Eventos",
                               "fatalities": "Fatalidades"}).reset_index(drop=True), dec=2)
print(subtipos.groupby("perfil")["% eventos"].sum().round(1).rename("% de eventos por perfil").to_string())

# ======== Celda de código 8 (índice 18): 3.2 Panel departamento–año y réplica del resultado de la Etapa 1 ========
# =============================================================================
# 3.2 Panel departamento–año y réplica del resultado de la Etapa 1
# =============================================================================
def construir_panel(df, mapa):
    """Agrega los eventos semanales al nivel departamento–año y calcula proporciones y etiqueta por perfil."""
    d = df.assign(perfil=df["sub_event_type"].map(mapa).fillna("D"))
    semanas = d.groupby("anio")["week"].nunique()
    conteo = d.pivot_table(index=["depto", "anio"], columns="perfil", values="events", aggfunc="sum", fill_value=0)
    conteo = conteo.reindex(columns=list("ABCD"), fill_value=0)
    base = pd.MultiIndex.from_product([sorted(d["depto"].unique()), sorted(d["anio"].unique())], names=["depto", "anio"])
    panel = conteo.reindex(base, fill_value=0)
    panel["eventos"] = panel[list("ABCD")].sum(axis=1)
    panel["fatalidades"] = d.groupby(["depto", "anio"])["fatalities"].sum().reindex(base, fill_value=0)
    panel = panel.reset_index()
    panel["semanas"] = panel["anio"].map(semanas)
    for k in "ABCD":
        panel[f"s_{k}"] = panel[k] / panel["eventos"].where(panel["eventos"] > 0)
    panel["perfil"] = np.select([panel["s_C"] >= .5, panel["s_A"] >= .5, panel["s_B"] >= .5], ["C", "A", "B"], "D")
    panel.loc[panel["eventos"] == 0, "perfil"] = None
    panel["y"] = np.where(panel["eventos"] > 0, (panel["perfil"] == "C").astype(float), np.nan)
    return panel

panel = construir_panel(acled, MAPA_PERFIL_U1)
print(f"Panel: {panel['depto'].nunique()} departamentos × {panel['anio'].nunique()} años = {len(panel)} celdas "
      f"({panel['anio'].min()}-{panel['anio'].max()}; 2026 con {panel.loc[panel.anio == 2026, 'semanas'].iat[0]} semanas)")
print("Celdas con perfil A o B dominante:", int(panel["perfil"].isin(["A", "B"]).sum()))

# Réplica de la Etapa 1: un perfil por departamento con todo el periodo agregado
total_depto = panel.groupby("depto")[list("ABCD")].sum()
perfil_global = np.where(total_depto["C"] / total_depto.sum(axis=1) >= .5, "C", "D")
print("\nRéplica Etapa 1 (perfil agregado 2020-2026):",
      pd.Series(perfil_global).value_counts().to_dict(), "| Etapa 1 reportó {'D': 17, 'C': 16}")

# ======== Celda de código 9 (índice 19): 3.3 Confiabilidad de la etiqueta y dinámica temporal de los perfiles ========
# =============================================================================
# 3.3 Confiabilidad de la etiqueta y dinámica temporal de los perfiles
# =============================================================================
MIN_EVENTOS = 10   # umbral de confiabilidad de la etiqueta (justificación en el texto)
panel["confiable"] = panel["eventos"] >= MIN_EVENTOS
etiquetables = panel[(panel["anio"] >= 2021) & panel["confiable"]]
print(f"Celdas 2021-2026 con eventos: {int(((panel.anio >= 2021) & (panel.eventos > 0)).sum())} | "
      f"con ≥ {MIN_EVENTOS} eventos (etiqueta confiable): {len(etiquetables)}")
print(f"Error estándar máximo de s_C con n = {MIN_EVENTOS} eventos: ±{np.sqrt(.25 / MIN_EVENTOS):.3f}")

# Matriz de transición entre años consecutivos (celdas confiables en t-1 y en t)
pv = panel.pivot(index="depto", columns="anio", values="perfil").where(panel.pivot(index="depto", columns="anio", values="confiable"))
pares = pd.concat([pd.DataFrame({"t-1": pv[a - 1].values, "t": pv[a].values}) for a in range(2021, 2027)]).dropna()
trans = pd.crosstab(pares["t-1"], pares["t"], rownames=["Perfil t-1"], colnames=["Perfil t"])
display(trans)
persistencia = np.trace(trans.values) / trans.values.sum()
print(f"Persistencia año a año: {persistencia:.1%} de los departamentos repite perfil")

fig = plt.figure(figsize=(15, 8.2))
gs = fig.add_gridspec(2, 2, width_ratios=[1.35, 1], hspace=0.45, wspace=0.25)
ax = fig.add_subplot(gs[:, 0])
sC = panel.pivot(index="depto", columns="anio", values="s_C")
conf = panel.pivot(index="depto", columns="anio", values="confiable")
orden = sC.mean(axis=1).sort_values(ascending=False).index
sC, conf, pvo = sC.loc[orden], conf.loc[orden], panel.pivot(index="depto", columns="anio", values="perfil").loc[orden]
im = ax.imshow(sC.values, cmap=CMAP_C, vmin=0, vmax=1, aspect="auto")
for i in range(sC.shape[0]):
    for j in range(sC.shape[1]):
        if not conf.values[i, j]:
            ax.add_patch(Rectangle((j - .5, i - .5), 1, 1, facecolor=FONDO, edgecolor=GRID, hatch="////", lw=0))
            continue
        ax.text(j, i, pvo.values[i, j], ha="center", va="center", fontsize=7.5, fontweight="semibold",
                color="white" if sC.values[i, j] > .62 else TINTA)
ax.set_xticks(range(sC.shape[1]), sC.columns); ax.set_yticks(range(len(orden)), [NOMBRE[d] for d in orden], fontsize=8)
ax.grid(False); ax.set_title("(a) Proporción de eventos C y perfil por departamento–año")
cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02); cb.set_label("$s^C_{d,t}$ (proporción de eventos C)")
cb.ax.axhline(0.5, color=TINTA, lw=1.2); cb.outline.set_visible(False)
ax.set_xlabel(f"Celdas rayadas: menos de {MIN_EVENTOS} eventos (etiqueta no confiable, excluidas del modelado)",
              fontsize=8.5, color=TINTA2, labelpad=8)

ax = fig.add_subplot(gs[0, 1])
cnt = etiquetables.groupby(["anio", "perfil"]).size().unstack(fill_value=0)[["C", "D"]]
ax.bar(cnt.index, cnt["D"], color=COL["D"], width=0.55, label="D (mixto)")
ax.bar(cnt.index, cnt["C"], bottom=cnt["D"], color=COL["C"], width=0.55, label="C (convencional)",
       edgecolor=FONDO, linewidth=2)
for a in cnt.index:
    ax.text(a, cnt.loc[a].sum() + 0.4, f"{cnt.loc[a, 'C'] / cnt.loc[a].sum():.0%} C", ha="center", fontsize=8.5, color=TINTA2)
ax.set_title("(b) Departamentos por perfil y año"); ax.set_ylabel("Departamentos (etiqueta confiable)")
ax.set_ylim(0, cnt.sum(axis=1).max() * 1.18); ax.legend(ncol=2, loc="upper left", fontsize=8.5)

ax = fig.add_subplot(gs[1, 1])
ax.hist(etiquetables["s_C"], bins=np.linspace(0, 1, 21), color=COL["C"], edgecolor=FONDO, linewidth=1.5)
ax.axvline(0.5, color=TINTA, lw=1.2, ls="--"); ax.text(0.49, ax.get_ylim()[1] * 0.96, "umbral 50 % ", fontsize=8.5, color=TINTA2, ha="right", va="top")
ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
cerca = ((etiquetables["s_C"] - .5).abs() < .1).mean()
ax.set_title(f"(c) Distribución de $s^C$: {cerca:.0%} de celdas a ±10 pp del umbral")
ax.set_xlabel("$s^C_{d,t}$"); ax.set_ylabel("Celdas departamento–año")
guardar(fig, "fig02_variable_objetivo"); plt.show()

# ======== Celda de código 10 (índice 21): 3.4 Sensibilidad de la etiqueta a los subtipos asignados por defecto ========
# =============================================================================
# 3.4 Sensibilidad de la etiqueta a los subtipos asignados por defecto
# =============================================================================
MAPA_ALTERNATIVO = {**MAPA_PERFIL_U1, "Shelling/artillery/missile attack": "B", "Sexual violence": "C"}
panel_alt = construir_panel(acled, MAPA_ALTERNATIVO)
comp = panel[["depto", "anio", "perfil", "confiable"]].merge(panel_alt[["depto", "anio", "perfil"]],
                                                            on=["depto", "anio"], suffixes=("_u1", "_alt"))
comp = comp[(comp["anio"] >= 2021) & comp["confiable"]]
cambios = comp[comp["perfil_u1"] != comp["perfil_alt"]]
print(f"Etiquetas que cambian al reasignar Shelling→B y Sexual violence→C: {len(cambios)} de {len(comp)} "
      f"({len(cambios) / len(comp):.1%})")
if len(cambios):
    display(cambios.assign(departamento=cambios["depto"].map(NOMBRE))[["departamento", "anio", "perfil_u1", "perfil_alt"]])

# ======== Celda de código 11 (índice 24): 4.1 Fuentes externas con descarga automática y caché local ========
# =============================================================================
# 4.1 Fuentes externas con descarga automática y caché local
# =============================================================================
URL_COCA = "https://www.datos.gov.co/resource/acs4-3wgp.csv?$limit=5000"   # MinJusticia/ODC – SIMCI municipal
URL_GFW = "https://gfw2-data.s3.amazonaws.com/country-pages/country_stats/download/2025/COL.xlsx"
CABECERA = {"User-Agent": "Mozilla/5.0 (proyecto academico Universidad de La Sabana)"}
RUTA_COCA, RUTA_BOSQUE = DIR_EXT / "coca_departamento_anio.csv", DIR_EXT / "bosque_departamento_anio.csv"

def descargar_coca():
    m = pd.read_csv(URL_COCA, dtype=str, storage_options=CABECERA)
    anios = [c for c in m.columns if re.fullmatch(r"_\d{4}", c)]
    m[anios] = m[anios].apply(pd.to_numeric, errors="coerce").fillna(0)
    m["depto"] = m["departamento"].map(normalizar)
    d = m.groupby("depto")[anios].sum()
    d.columns = [int(c[1:]) for c in d.columns]
    return d.stack().rename("coca_ha").rename_axis(["depto", "anio"]).reset_index()

def descargar_bosque():
    g = pd.read_excel(URL_GFW, sheet_name="Subnational 1 tree cover loss", storage_options=CABECERA)
    g = g[g["threshold"] == 30].copy()                       # definición estándar de GFW: dosel ≥ 30 %
    g["depto"] = g["subnational1"].map(normalizar)
    filas = []
    for _, r in g.iterrows():
        acumulada = 0.0
        for anio in range(2011, 2026):
            acumulada += r[f"tc_loss_ha_{anio}"]
            filas.append((r["depto"], anio, 100 * (r["extent_2010_ha"] - acumulada) / r["area_ha"],
                          r[f"tc_loss_ha_{anio}"], r["area_ha"]))
    return pd.DataFrame(filas, columns=["depto", "anio", "bosque_pct", "perdida_ha", "area_ha"])

def cargar_o_descargar(ruta, funcion, etiqueta):
    if ruta.exists():
        print(f"✓ {etiqueta}: caché local ({ruta})")
        return pd.read_csv(ruta)
    print(f"↓ {etiqueta}: descargando de la fuente oficial…")
    df = funcion(); df.to_csv(ruta, index=False)
    return df

coca = cargar_o_descargar(RUTA_COCA, descargar_coca, "Coca SIMCI/ODC (datos.gov.co)")
bosque = cargar_o_descargar(RUTA_BOSQUE, descargar_bosque, "Cobertura arbórea (Global Forest Watch)")

# Control de calidad: totales nacionales frente a las cifras oficiales publicadas por SIMCI–UNODC
oficial = {2020: 143000, 2021: 204000, 2022: 230000, 2023: 253000}
control = coca.groupby("anio")["coca_ha"].sum().loc[2019:].round(0).to_frame("Total nacional (ha, este dataset)")
control["Cifra oficial publicada (ha, redondeada)"] = control.index.map(oficial)
display(control)

# ======== Celda de código 12 (índice 25): 4.2 Variables estructurales estáticas: IPM (DANE) y distancia a Bogotá ========
# =============================================================================
# 4.2 Variables estructurales estáticas: IPM (DANE) y distancia a Bogotá
# =============================================================================
# DANE (2019). Pobreza multidimensional por departamentos 2018, Encuesta de Calidad de Vida. Incidencia (%) total.
IPM_2018 = {
    "guainia": 65.0, "vaupes": 59.4, "vichada": 55.0, "la guajira": 51.4, "choco": 45.1, "sucre": 39.7,
    "magdalena": 38.6, "cordoba": 36.7, "amazonas": 34.9, "narino": 33.5, "guaviare": 33.5, "cesar": 33.2,
    "bolivar": 32.4, "arauca": 31.8, "norte de santander": 31.5, "caqueta": 28.7, "cauca": 28.7,
    "putumayo": 25.1, "tolima": 23.5, "atlantico": 20.1, "huila": 19.2, "casanare": 19.1, "antioquia": 17.1,
    "boyaca": 16.6, "quindio": 16.2, "meta": 15.6, "caldas": 15.3, "valle del cauca": 13.6, "santander": 12.9,
    "risaralda": 12.5, "cundinamarca": 11.5, "san andres": 8.9, "bogota": 4.4}
BOGOTA = (4.5981, -74.0760)   # Plaza de Bolívar

estructura = acled.groupby("depto")[["centroid_latitude", "centroid_longitude"]].first()
estructura.columns = ["lat", "lon"]
estructura["dist_km"] = haversine_km(estructura["lat"], estructura["lon"], *BOGOTA)
estructura["ipm"] = estructura.index.map(IPM_2018)
estructura = estructura.reset_index()

faltan = {"IPM": set(estructura.depto) - set(IPM_2018), "coca (sin registro = 0 ha)": set(estructura.depto) - set(coca.depto),
          "bosque": set(estructura.depto) - set(bosque.depto)}
print("Cruce de llaves entre fuentes (departamentos sin dato):", {k: sorted(v) for k, v in faltan.items()})
assert not faltan["IPM"] and not faltan["bosque"], "Hay departamentos sin cruce en IPM o bosque"

# ======== Celda de código 13 (índice 26): 4.3 Dataset de modelado: etiqueta en t, predictoras en t-1 ========
# =============================================================================
# 4.3 Dataset de modelado: etiqueta en t, predictoras en t-1
# =============================================================================
FEATURES = ["x1_eventos", "x2_fatalidades", "x2b_letalidad", "x4_coca", "x5_bosque", "x6_distancia", "x7_ipm"]
ETIQ = {"x1_eventos": "x₁ eventos (log)", "x2_fatalidades": "x₂ fatalidades (log)", "x2b_letalidad": "x₂b letalidad",
        "x4_coca": "x₄ coca (log ha)", "x5_bosque": "x₅ bosque (%)", "x6_distancia": "x₆ distancia (km)",
        "x7_ipm": "x₇ IPM (%)"}

def panel_extendido(panel):
    """Añade al panel las variables externas por departamento–año (con arrastre del último dato disponible)."""
    p = (panel.merge(coca, on=["depto", "anio"], how="left")
              .merge(bosque[["depto", "anio", "bosque_pct"]], on=["depto", "anio"], how="left")
              .merge(estructura, on="depto", how="left")
              .sort_values(["depto", "anio"]))
    p["coca_ha"] = p.groupby("depto")["coca_ha"].ffill().fillna(0)     # sin registro SIMCI -> 0 ha
    p["bosque_pct"] = p.groupby("depto")["bosque_pct"].ffill()         # 2026 -> último año GFW (2025)
    return p

def calcular_features(prev):
    """Predictoras a partir de las filas del año t-1 (una por departamento)."""
    return pd.DataFrame({
        "x1_eventos": np.log1p(prev["eventos"] / prev["semanas"] * 52),
        "x2_fatalidades": np.log1p(prev["fatalidades"] / prev["semanas"] * 52),
        "x2b_letalidad": (prev["fatalidades"] / prev["eventos"].where(prev["eventos"] > 0)).fillna(0.0),
        "x4_coca": np.log1p(prev["coca_ha"]),
        "x5_bosque": prev["bosque_pct"],
        "x6_distancia": prev["dist_km"],
        "x7_ipm": prev["ipm"],
    }, index=prev.index)

pext = panel_extendido(panel)
previo = pext.copy(); previo["anio"] += 1                               # la fila t-1 se alinea con el año t
previo = pd.concat([previo[["depto", "anio"]], calcular_features(previo),
                    previo[["s_C", "perfil", "eventos", "confiable"]].add_suffix("_prev")], axis=1)
datos = (pext.merge(previo, on=["depto", "anio"], how="inner")
             .query("anio >= 2021 and eventos >= @MIN_EVENTOS").reset_index(drop=True))
datos["departamento"] = datos["depto"].map(NOMBRE)
assert datos[FEATURES].notna().all().all(), "Hay predictoras faltantes"

resumen = pd.DataFrame({"Filas": [len(datos)], "Departamentos": [datos.depto.nunique()],
                        "Años etiqueta": [f"{datos.anio.min()}-{datos.anio.max()}"],
                        "Perfil C": [int(datos.y.sum())], "Perfil D": [int((1 - datos.y).sum())],
                        "Predictoras": [len(FEATURES)], "Faltantes": [int(datos[FEATURES].isna().sum().sum())]})
display(resumen.style.hide(axis="index"))
desc = datos[FEATURES].describe().T[["mean", "std", "min", "50%", "max"]].rename(index=ETIQ)
desc.columns = ["Media", "Desv. est.", "Mín.", "Mediana", "Máx."]
tabla(desc, dec=2)

# ======== Celda de código 14 (índice 29): 5.1 Distribución de cada predictora por perfil ========
# =============================================================================
# 5.1 Distribución de cada predictora por perfil
# =============================================================================
fig, axes = plt.subplots(2, 4, figsize=(15, 6.6))
rng_j = np.random.default_rng(SEED)
for ax, f in zip(axes.ravel(), FEATURES):
    for k, (clase, color) in enumerate([(0, COL["D"]), (1, COL["C"])]):
        v = datos.loc[datos.y == clase, f].values
        ax.boxplot(v, positions=[k], widths=0.45, showfliers=False, patch_artist=True,
                   boxprops=dict(facecolor=color, alpha=0.25, edgecolor=color), medianprops=dict(color=color, lw=2),
                   whiskerprops=dict(color=color), capprops=dict(color=color))
        ax.scatter(k + rng_j.uniform(-.16, .16, len(v)), v, s=11, color=color, alpha=0.8, edgecolor=FONDO, lw=0.5)
    u = stats.mannwhitneyu(datos.loc[datos.y == 1, f], datos.loc[datos.y == 0, f])
    ax.set_xticks([0, 1], ["D", "C"]); ax.set_title(f"{ETIQ[f]}   ·   Mann-Whitney p = {u.pvalue:.1e}", fontsize=9.5)
ax = axes.ravel()[-1]; ax.axis("off")
ax.legend(handles=[Patch(color=COL["C"], label=f"C convencional (n={int(datos.y.sum())})"),
                   Patch(color=COL["D"], label=f"D mixto (n={int((1 - datos.y).sum())})")], loc="center", fontsize=10)
fig.suptitle("Predictoras en t-1 según el perfil observado en t", x=0.01, ha="left", fontweight="semibold")
fig.tight_layout(); guardar(fig, "fig03_predictoras_por_clase"); plt.show()

# ======== Celda de código 15 (índice 30): 5.2 Dependencia lineal/monótona vs. dependencia general, y correlación entre predictoras ========
# =============================================================================
# 5.2 Dependencia lineal/monótona vs. dependencia general, y correlación entre predictoras
# =============================================================================
rho = pd.Series({f: stats.spearmanr(datos[f], datos["y"]).correlation for f in FEATURES})
mi = pd.Series(mutual_info_classif(datos[FEATURES], datos["y"], random_state=SEED, n_neighbors=5), index=FEATURES)
corr = datos[FEATURES].corr(method="spearman")

fig, axes = plt.subplots(1, 2, figsize=(14, 5.4), gridspec_kw={"width_ratios": [1, 1.1]})
ax = axes[0]
ax.scatter(rho.abs(), mi, s=70, color=COL["mlp"], edgecolor=FONDO, lw=2, zorder=3)
for f in FEATURES:
    ax.annotate(ETIQ[f], (abs(rho[f]), mi[f]), xytext=(6, 4), textcoords="offset points", fontsize=8.5, color=TINTA2)
ax.set_xlabel("|ρ de Spearman| con la etiqueta (asociación monótona)")
ax.set_ylabel("Información mutua (nats, dependencia general)")
ax.set_title("(a) ¿Cuánta información no captura una relación monótona?")
ax.set_xlim(0, max(rho.abs()) * 1.3); ax.set_ylim(0, max(mi) * 1.25)

ax = axes[1]
im = ax.imshow(corr.values, cmap=CMAP_DIV, norm=TwoSlopeNorm(0, -1, 1))
ax.set_xticks(range(len(FEATURES)), [ETIQ[f].split(" ")[0] for f in FEATURES])
ax.set_yticks(range(len(FEATURES)), [ETIQ[f] for f in FEATURES])
for i in range(len(FEATURES)):
    for j in range(len(FEATURES)):
        ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=8,
                color="white" if abs(corr.values[i, j]) > .6 else TINTA)
ax.grid(False); ax.set_title("(b) Correlación de Spearman entre predictoras")
cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02); cb.outline.set_visible(False)
fig.tight_layout(); guardar(fig, "fig04_dependencia_correlacion"); plt.show()
display(pd.DataFrame({"ρ Spearman con y": rho, "Información mutua": mi}).rename(index=ETIQ).round(3))

# ======== Celda de código 16 (índice 31): 5.3 Hipótesis de interacción de la Etapa 1 (§6.2): bosque × distancia y coca × pobreza ========
# =============================================================================
# 5.3 Hipótesis de interacción de la Etapa 1 (§6.2): bosque × distancia y coca × pobreza
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
pares = [("x5_bosque", "x6_distancia"), ("x4_coca", "x7_ipm")]
for ax, (fx, fy) in zip(axes, pares):
    for clase, color, et in [(0, COL["D"], "D mixto"), (1, COL["C"], "C convencional")]:
        s = datos[datos.y == clase]
        ax.scatter(s[fx], s[fy], s=40, color=color, edgecolor="none", alpha=0.75, label=et, zorder=3)
    ax.set_xlabel(ETIQ[fx]); ax.set_ylabel(ETIQ[fy]); ax.legend(loc="upper left", fontsize=8.5)
axes[0].set_title("(a) Bosque × distancia: ¿la selva lejana es C?")
axes[1].set_title("(b) Coca × pobreza: ¿economía ilícita en territorios pobres es C?")
fig.tight_layout(); guardar(fig, "fig05_interacciones_hipotesis"); plt.show()

# ======== Celda de código 17 (índice 34): 6.1 Partición temporal y estandarización ajustada solo con entrenamiento ========
# =============================================================================
# 6.1 Partición temporal y estandarización ajustada solo con entrenamiento
# =============================================================================
ANIOS_TRAIN, ANIO_VAL, ANIO_TEST = [2021, 2022, 2023, 2024], 2025, 2026
m_tr = datos["anio"].isin(ANIOS_TRAIN).values
m_va = (datos["anio"] == ANIO_VAL).values
m_te = (datos["anio"] == ANIO_TEST).values

escalador = StandardScaler().fit(datos.loc[m_tr, FEATURES])
def matriz(mascara, esc=escalador):
    return esc.transform(datos.loc[mascara, FEATURES]).astype("float32")

Xtr, Xva, Xte = matriz(m_tr), matriz(m_va), matriz(m_te)
ytr, yva, yte = (datos.loc[m, "y"].values.astype("float32") for m in (m_tr, m_va, m_te))

particiones = pd.DataFrame({
    "Años (t)": ["2021-2024", "2025", "2026 (ene-ago)"],
    "Filas": [len(ytr), len(yva), len(yte)],
    "Perfil C": [int(ytr.sum()), int(yva.sum()), int(yte.sum())],
    "Prevalencia C": [ytr.mean(), yva.mean(), yte.mean()],
    "Departamentos": [datos.loc[m, "depto"].nunique() for m in (m_tr, m_va, m_te)],
}, index=["Entrenamiento", "Validación", "Prueba"])
tabla(particiones, dec=2)

fig, ax = plt.subplots(figsize=(13, 2.6))
roles = {2020: ("Solo predictoras (t-1)", GRID), 2021: ("Entrenamiento", COL["train"]), 2022: ("Entrenamiento", COL["train"]),
         2023: ("Entrenamiento", COL["train"]), 2024: ("Entrenamiento", COL["train"]), 2025: ("Validación", COL["val"]),
         2026: ("Prueba", COL["test"]), 2027: ("Proyección (§12)", "#d6e6f9")}
for anio, (rol, color) in roles.items():
    ax.add_patch(Rectangle((anio - .46, 0), .92, 1, facecolor=color, edgecolor=FONDO, lw=2))
    sub = datos[datos.anio == anio]
    txt = f"{rol}\nn = {len(sub)} · {sub.y.mean():.0%} C" if len(sub) else rol
    ax.text(anio, .5, txt, ha="center", va="center", fontsize=8.5,
            color="white" if color in (COL["train"], COL["test"]) else TINTA)
ax.set_xlim(2019.5, 2027.5); ax.set_ylim(0, 1); ax.set_yticks([]); ax.grid(False)
ax.set_xticks(range(2020, 2028)); ax.spines["left"].set_visible(False)
ax.set_title("Diseño temporal: cada año se predice con la información del año anterior")
guardar(fig, "fig06_particion_temporal"); plt.show()

# ======== Celda de código 18 (índice 36): 6.2 Líneas base: mayoritaria, persistencia y regresión logística (una neurona) ========
# =============================================================================
# 6.2 Líneas base: mayoritaria, persistencia y regresión logística (una neurona)
# =============================================================================
prev_train = float(ytr.mean())

def score_persistencia(mascara):
    """Probabilidad de C = proporción de eventos C del año anterior (prevalencia de entrenamiento si no hubo eventos)."""
    s = datos.loc[mascara, "s_C_prev"].values.astype(float)
    return np.where(np.isnan(s), prev_train, s)

def folds_temporales(anios_val=(2023, 2024, 2025), columnas=None):
    """Validación cruzada de origen móvil: entrena con años < v y valida en v (escalador por pliegue)."""
    columnas = columnas or FEATURES
    for v in anios_val:
        tr, va = (datos["anio"] < v).values, (datos["anio"] == v).values
        esc = StandardScaler().fit(datos.loc[tr, columnas])
        yield (v, esc.transform(datos.loc[tr, columnas]).astype("float32"), datos.loc[tr, "y"].values.astype("float32"),
               esc.transform(datos.loc[va, columnas]).astype("float32"), datos.loc[va, "y"].values.astype("float32"))

FOLDS_CV = list(folds_temporales())

# Selección de C (inverso de la regularización L2) de la logística con la misma validación temporal
C_GRID = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
cv_logit = pd.DataFrame([{"C": c, "anio_val": v,
                          "BCE": bce(yv, LogisticRegression(C=c, class_weight="balanced", max_iter=5000)
                                     .fit(Xa, ya).predict_proba(Xv)[:, 1])}
                         for c in C_GRID for v, Xa, ya, Xv, yv in FOLDS_CV])
C_LR = float(cv_logit.groupby("C")["BCE"].mean().idxmin())
logit = LogisticRegression(C=C_LR, class_weight="balanced", max_iter=5000).fit(Xtr, ytr)
print(f"C elegido para la logística (menor BCE media en validación temporal): {C_LR}")
print(cv_logit.groupby("C")["BCE"].mean().round(4).to_string())

signo_u1 = {"x1_eventos": "±", "x2_fatalidades": "+", "x2b_letalidad": "+", "x4_coca": "+",
            "x5_bosque": "+", "x6_distancia": "+", "x7_ipm": "+"}
coef = pd.DataFrame({"Coeficiente (por 1 DE)": logit.coef_.ravel(), "Razón de odds": np.exp(logit.coef_.ravel()),
                     "Signo esperado (U1)": [signo_u1[f] for f in FEATURES]}, index=[ETIQ[f] for f in FEATURES])
coef["¿Coincide?"] = [("—" if e == "±" else ("sí" if np.sign(c) == (1 if e == "+" else -1) else "no"))
                      for c, e in zip(coef.iloc[:, 0], coef.iloc[:, 2])]
tabla(coef, dec=3)

# ======== Celda de código 19 (índice 39): 7.1 Configuración y constructor del modelo (Keras, API funcional) ========
# =============================================================================
# 7.1 Configuración y constructor del modelo (Keras, API funcional)
# =============================================================================
@dataclass(frozen=True)
class ConfigMLP:
    capas: tuple = (16, 8)       # neuronas por capa oculta
    l2: float = 1e-3             # penalización L2 (λ)
    dropout: float = 0.2         # tasa de dropout tras cada capa oculta
    lr: float = 1e-3             # tasa de aprendizaje inicial de Adam
    batch: int = 16              # tamaño de lote
    epocas_max: int = 800        # tope de épocas (la parada temprana actúa antes)
    paciencia: int = 80          # épocas sin mejora en validación antes de detener
    paciencia_lr: int = 30       # épocas sin mejora antes de reducir la tasa de aprendizaje
    factor_lr: float = 0.5       # factor de reducción de la tasa de aprendizaje
    lr_min: float = 1e-5         # tasa de aprendizaje mínima
    min_delta: float = 1e-4      # mejora mínima que cuenta como mejora

def construir_mlp(n_x, cfg):
    """MLP: entrada -> [Dense(ReLU, He, L2) -> Dropout] x L -> Dense(1, sigmoide)."""
    reg = keras.regularizers.L2(cfg.l2) if cfg.l2 > 0 else None
    entrada = keras.Input(shape=(n_x,), name="entrada_x_t_1")
    h = entrada
    for i, unidades in enumerate(cfg.capas, 1):
        h = layers.Dense(unidades, activation="relu", kernel_initializer="he_normal",
                         kernel_regularizer=reg, name=f"oculta_{i}")(h)
        if cfg.dropout > 0:
            h = layers.Dropout(cfg.dropout, name=f"dropout_{i}")(h)
    salida = layers.Dense(1, activation="sigmoid", name="salida_P_C")(h)
    return keras.Model(entrada, salida, name="MLP_perfil_amenaza")

def n_parametros(capas, n_x=len(FEATURES)):
    dims = [n_x, *capas, 1]
    return int(sum(a * b + b for a, b in zip(dims[:-1], dims[1:])))

# ======== Celda de código 20 (índice 41): 7.2 Bucle de entrenamiento (se define aquí porque la búsqueda lo usa; se explica en §8) ========
# =============================================================================
# 7.2 Bucle de entrenamiento (se define aquí porque la búsqueda lo usa; se explica en §8)
# =============================================================================
def entrenar_mlp(Xtr_, ytr_, Xva_=None, yva_=None, cfg=None, semilla=SEED, epocas=None,
                 calendario_lr=None, pesos_clase=True, diagnostico=True):
    """
    Entrena el MLP con un bucle explícito (tf.GradientTape); cada época completa corre dentro de un tf.function.
    - Con validación y sin `epocas`: parada temprana + reducción de LR en meseta; restaura los mejores pesos.
    - Con `epocas` (y opcionalmente `calendario_lr`): número fijo de épocas (reajustes y validaciones cruzadas).
    Devuelve: modelo, historial por época, resumen y función de predicción (inferencia, sin dropout).
    """
    cfg = cfg or CFG
    keras.utils.set_random_seed(semilla)
    rng = np.random.default_rng(semilla)
    Xtr_ = np.asarray(Xtr_, "float32"); ytr_ = np.asarray(ytr_, "float32").reshape(-1, 1)
    n, d, B = len(ytr_), Xtr_.shape[1], cfg.batch
    n_lotes = math.ceil(n / B)
    modelo = construir_mlp(d, cfg)
    opt = keras.optimizers.Adam(learning_rate=cfg.lr)
    opt.build(modelo.trainable_variables)                     # variables de Adam creadas antes del grafo
    perdida_bce = keras.losses.BinaryCrossentropy()
    n1 = float(ytr_.sum())
    w = (np.where(ytr_.ravel() == 1, n / (2 * n1), n / (2 * (n - n1))) if pesos_clase else np.ones(n)).astype("float32")

    @tf.function(input_signature=(tf.TensorSpec([None, d], tf.float32), tf.TensorSpec([None, 1], tf.float32),
                                  tf.TensorSpec([None], tf.float32)))
    def epoca(Xp, yp, wp):
        """Una época: recorre los lotes de 16 y aplica un paso de Adam por lote."""
        perdidas = tf.TensorArray(tf.float32, size=n_lotes)
        normas = tf.TensorArray(tf.float32, size=n_lotes)
        for i in tf.range(n_lotes):
            xb, yb, wb = Xp[i * B:(i + 1) * B], yp[i * B:(i + 1) * B], wp[i * B:(i + 1) * B]
            with tf.GradientTape() as cinta:
                p = modelo(xb, training=True)                             # propagación hacia adelante (dropout activo)
                perdida = perdida_bce(yb, p, sample_weight=wb)            # BCE ponderada por clase
                if modelo.losses:
                    perdida = perdida + tf.add_n(modelo.losses)           # + λ·||W||²
            grads = cinta.gradient(perdida, modelo.trainable_variables)   # retropropagación
            opt.apply_gradients(zip(grads, modelo.trainable_variables))   # actualización Adam
            tam = tf.cast(tf.shape(xb)[0], tf.float32)                      # promedio ponderado por tamaño de lote
            perdidas = perdidas.write(i, perdida * tam)
            normas = normas.write(i, tf.linalg.global_norm(grads) * tam)
        return tf.reduce_sum(perdidas.stack()) / n, tf.reduce_sum(normas.stack()) / n

    @tf.function(input_signature=(tf.TensorSpec([None, d], tf.float32),))
    def predecir(x):
        return modelo(x, training=False)                                  # inferencia (sin dropout)

    con_val = Xva_ is not None
    parada = con_val and epocas is None
    if con_val:
        Xva_ = np.asarray(Xva_, "float32"); yv = np.asarray(yva_, "float32").ravel()
    kernels = [c.kernel for c in modelo.layers if hasattr(c, "kernel")]
    H = {k: [] for k in ["epoca", "lr", "perdida_obj", "norma_grad", "bce_tr", "bce_va", "acc_tr", "acc_va",
                         "auc_tr", "auc_va", "ratio_act"]}
    mejor, mejor_ep, mejores_pesos, espera, espera_lr = np.inf, 0, None, 0, 0
    for ep in range(1, (epocas or cfg.epocas_max) + 1):
        if calendario_lr is not None:
            opt.learning_rate.assign(float(calendario_lr[min(ep - 1, len(calendario_lr) - 1)]))
        antes = [k.numpy().copy() for k in kernels] if diagnostico else None
        orden = rng.permutation(n)                                        # barajado por época
        l, g = epoca(Xtr_[orden], ytr_[orden], w[orden])
        if not (diagnostico or parada):
            continue
        H["epoca"].append(ep); H["lr"].append(float(keras.ops.convert_to_numpy(opt.learning_rate)))
        H["perdida_obj"].append(float(l)); H["norma_grad"].append(float(g))
        if diagnostico:
            p_tr = predecir(Xtr_).numpy().ravel()
            H["bce_tr"].append(bce(ytr_.ravel(), p_tr)); H["acc_tr"].append(float(((p_tr >= .5) == ytr_.ravel()).mean()))
            H["auc_tr"].append(roc_auc_score(ytr_.ravel(), p_tr))
            H["ratio_act"].append([float(np.linalg.norm(k.numpy() - a) / (np.linalg.norm(a) + 1e-12))
                                   for k, a in zip(kernels, antes)])
        if con_val:
            p_va = predecir(Xva_).numpy().ravel()
            H["bce_va"].append(bce(yv, p_va))
            if diagnostico:
                H["acc_va"].append(float(((p_va >= .5) == yv).mean()))
                H["auc_va"].append(roc_auc_score(yv, p_va) if len(np.unique(yv)) > 1 else np.nan)
        if parada:
            if H["bce_va"][-1] < mejor - cfg.min_delta:
                mejor, mejor_ep, mejores_pesos, espera, espera_lr = H["bce_va"][-1], ep, modelo.get_weights(), 0, 0
            else:
                espera += 1; espera_lr += 1
                if espera_lr >= cfg.paciencia_lr and calendario_lr is None:
                    opt.learning_rate.assign(max(H["lr"][-1] * cfg.factor_lr, cfg.lr_min)); espera_lr = 0
                if espera >= cfg.paciencia:
                    break
    if parada and mejores_pesos is not None:
        modelo.set_weights(mejores_pesos)
    info = {"mejor_epoca": mejor_ep if parada else ep, "epocas_ejecutadas": ep,
            "mejor_bce_val": mejor if parada else np.nan,
            "calendario_lr": H["lr"][:mejor_ep] if parada else H["lr"]}
    return modelo, H, info, lambda X: predecir(np.asarray(X, "float32")).numpy().ravel()

# ======== Celda de código 21 (índice 42): 7.3 Búsqueda de capacidad y regularización (3 pliegues temporales × 2 semillas) ========
# =============================================================================
# 7.3 Búsqueda de capacidad y regularización (3 pliegues temporales × 2 semillas)
# =============================================================================
REJILLA = [(capas, l2, do) for capas in [(8,), (16,), (16, 8), (32, 16)] for l2 in [1e-3, 1e-2] for do in [0.0, 0.2]]
SEMILLAS_BUSQUEDA = [1, 2]
t0, filas = time.time(), []
for capas, l2, do in REJILLA:
    cfg_i = ConfigMLP(capas=capas, l2=l2, dropout=do)
    for v, Xa, ya, Xv, yv in FOLDS_CV:
        for s in SEMILLAS_BUSQUEDA:
            _, _, inf, pred = entrenar_mlp(Xa, ya, Xv, yv, cfg=cfg_i, semilla=s, diagnostico=False)
            pv = pred(Xv)
            filas.append({"capas": str(capas), "L2": l2, "dropout": do, "anio_val": v, "semilla": s,
                          "BCE_val": bce(yv, pv), "AUC_val": roc_auc_score(yv, pv), "mejor_epoca": inf["mejor_epoca"]})
busqueda = pd.DataFrame(filas)
print(f"{len(busqueda)} entrenamientos en {time.time() - t0:.0f} s")

res_busq = (busqueda.groupby(["capas", "L2", "dropout"])
            .agg(BCE_media=("BCE_val", "mean"), BCE_ee=("BCE_val", lambda s: s.std(ddof=1) / np.sqrt(len(s))),
                 AUC_media=("AUC_val", "mean"), epoca_mediana=("mejor_epoca", "median"))
            .reset_index())
res_busq["parámetros"] = res_busq["capas"].map(lambda c: n_parametros(eval(c)))
res_busq = res_busq.sort_values("BCE_media").reset_index(drop=True)
mejor = res_busq.iloc[0]
candidatas = res_busq[res_busq["BCE_media"] <= mejor["BCE_media"] + mejor["BCE_ee"]]
elegida = candidatas.sort_values(["parámetros", "BCE_media"]).iloc[0]
CFG = ConfigMLP(capas=eval(elegida["capas"]), l2=float(elegida["L2"]), dropout=float(elegida["dropout"]))
tabla(res_busq, dec=4)
print(f"\nMejor BCE media: {mejor['capas']} L2={mejor['L2']} dropout={mejor['dropout']} -> {mejor['BCE_media']:.4f} ± {mejor['BCE_ee']:.4f}")
print(f"Configuraciones dentro de 1 EE del mejor: {len(candidatas)}")
print(f"ELEGIDA (regla 1-EE, menos parámetros): {CFG}")

# ======== Celda de código 22 (índice 43): 7.4 Visualización de la búsqueda: capacidad vs. regularización ========
# =============================================================================
# 7.4 Visualización de la búsqueda: capacidad vs. regularización
# =============================================================================
fig, ax = plt.subplots(figsize=(11, 4.6))
etiquetas = [f"{c}\nL2={l:g}, do={d:g}" for c, l, d in zip(res_busq.capas, res_busq.L2, res_busq.dropout)]
orden = np.argsort(res_busq["parámetros"].values * 10 + res_busq["BCE_media"].values)
ys = res_busq["BCE_media"].values[orden]; es = res_busq["BCE_ee"].values[orden]
colores = [COL["mlp"] if (res_busq.capas.values[i] == str(CFG.capas) and res_busq.L2.values[i] == CFG.l2
                          and res_busq.dropout.values[i] == CFG.dropout) else COL["D"] for i in orden]
ax.errorbar(range(len(ys)), ys, yerr=es, fmt="none", ecolor=EJE, elinewidth=1.2, capsize=0)
ax.scatter(range(len(ys)), ys, s=60, c=colores, edgecolor=FONDO, lw=2, zorder=3)
ax.axhline(mejor["BCE_media"] + mejor["BCE_ee"], color=TINTA2, lw=1, ls="--")
ax.text(len(ys) - .5, mejor["BCE_media"] + mejor["BCE_ee"], "  mejor + 1 EE", va="bottom", ha="right", fontsize=8, color=TINTA2)
ax.set_xticks(range(len(ys)), [etiquetas[i] for i in orden], fontsize=7, rotation=90)
ax.set_ylabel("BCE de validación temporal (media ± EE)")
ax.set_title("Búsqueda de arquitectura: configuraciones ordenadas por número de parámetros (azul = elegida)")
fig.tight_layout(); guardar(fig, "fig07_busqueda_hiperparametros"); plt.show()

# ======== Celda de código 23 (índice 45): 7.5 Modelo base final: resumen de Keras y conteo de parámetros ========
# =============================================================================
# 7.5 Modelo base final: resumen de Keras y conteo de parámetros
# =============================================================================
modelo_demo = construir_mlp(len(FEATURES), CFG)
modelo_demo.summary()
n_par = n_parametros(CFG.capas)
print(f"\nParámetros entrenables: {n_par} | Filas de entrenamiento: {len(ytr)} | "
      f"Parámetros por observación: {n_par / len(ytr):.2f}")
tabla(pd.DataFrame(asdict(CFG).items(), columns=["Hiperparámetro", "Valor"]).astype(str).set_index("Hiperparámetro"))

# ======== Celda de código 24 (índice 46): 7.6 Esquema de la arquitectura ========
# =============================================================================
# 7.6 Esquema de la arquitectura
# =============================================================================
def dibujar_arquitectura(cfg, nombres_entrada):
    tam = [len(nombres_entrada), *cfg.capas, 1]
    fig, ax = plt.subplots(figsize=(12.5, 5.6)); ax.axis("off"); ax.grid(False)
    xs = np.arange(len(tam)) * 3.0
    pos = [[(x, y) for y in np.linspace(-(n - 1) / 2, (n - 1) / 2, n) * min(0.62, 7.0 / max(n, 1))] for x, n in zip(xs, tam)]
    for a, b in zip(pos[:-1], pos[1:]):
        for (x1, y1) in a:
            for (x2, y2) in b:
                ax.plot([x1, x2], [y1, y2], color=GRID, lw=0.6, zorder=1)
    col_capa = [COL["D"]] + [COL["mlp"]] * len(cfg.capas) + [COL["C"]]
    for k, (capa, color) in enumerate(zip(pos, col_capa)):
        for (x, y) in capa:
            ax.scatter(x, y, s=150 if len(capa) <= 16 else 90, color=color, edgecolor=FONDO, lw=2, zorder=3)
    for (x, y), nombre in zip(pos[0], nombres_entrada):
        ax.text(x - 0.35, y, nombre, ha="right", va="center", fontsize=8.5, color=TINTA2)
    ax.text(pos[-1][0][0] + 0.35, 0, "P(perfil C | x$_{t-1}$)", ha="left", va="center", fontsize=9.5, color=TINTA)
    desc = ["Entrada\n7 variables\nestandarizadas"]
    for i, u in enumerate(cfg.capas, 1):
        desc.append(f"Oculta {i}\n{u} ReLU · He\nL2={cfg.l2:g}" + (f"\nDropout {cfg.dropout:g}" if cfg.dropout else ""))
    desc.append("Salida\n1 sigmoide")
    top = max(max(abs(y) for _, y in c) for c in pos)
    for x, d in zip(xs, desc):
        ax.text(x, -top - 0.55, d, ha="center", va="top", fontsize=8.5, color=TINTA2)
    ax.set_title(f"Arquitectura del modelo base · {n_parametros(cfg.capas)} parámetros entrenables", loc="left")
    ax.set_xlim(xs[0] - 2.8, xs[-1] + 2.8); ax.set_ylim(-top - 2.0, top + 0.6)
    return fig

fig = dibujar_arquitectura(CFG, [ETIQ[f] for f in FEATURES]); guardar(fig, "fig08_arquitectura"); plt.show()

# ======== Celda de código 25 (índice 48): 8.2 Verificación: bucle explícito vs. model.fit con lote completo (sin aleatoriedad de barajado) ========
# =============================================================================
# 8.2 Verificación: bucle explícito vs. model.fit con lote completo (sin aleatoriedad de barajado)
# =============================================================================
EPOCAS_CHK = 150
cfg_chk = replace(CFG, batch=len(ytr), dropout=0.0)          # lote completo: ambas vías ven exactamente los mismos datos
pesos_clase_dict = {0: len(ytr) / (2 * (ytr == 0).sum()), 1: len(ytr) / (2 * (ytr == 1).sum())}
keras.utils.set_random_seed(SEED)
modelo_fit = construir_mlp(len(FEATURES), cfg_chk)             # mismos pesos iniciales que el bucle (misma semilla)
modelo_fit.compile(optimizer=keras.optimizers.Adam(cfg_chk.lr), loss="binary_crossentropy")
t0 = time.time()
h_fit = modelo_fit.fit(Xtr, ytr, epochs=EPOCAS_CHK, batch_size=len(ytr), class_weight=pesos_clase_dict,
                       shuffle=False, verbose=0)
t_fit = time.time() - t0
t0 = time.time()
_, H_chk, _, pred_chk = entrenar_mlp(Xtr, ytr, Xva, yva, cfg=cfg_chk, semilla=SEED, epocas=EPOCAS_CHK)
t_bucle = time.time() - t0

perd_fit, perd_bucle = np.array(h_fit.history["loss"]), np.array(H_chk["perdida_obj"])
p_fit, p_bucle = modelo_fit.predict(Xva, verbose=0).ravel(), pred_chk(Xva)
verif = pd.DataFrame({"model.fit": [perd_fit[-1], bce(yva, p_fit), roc_auc_score(yva, p_fit), 1000 * t_fit / EPOCAS_CHK],
                      "Bucle explícito": [perd_bucle[-1], bce(yva, p_bucle), roc_auc_score(yva, p_bucle), 1000 * t_bucle / EPOCAS_CHK]},
                     index=["Pérdida optimizada en la última época", "BCE validación", "AUC validación", "Milisegundos por época"])
tabla(verif, dec=4)
print(f"Diferencia máxima entre las dos curvas de pérdida (150 épocas): {np.abs(perd_fit - perd_bucle).max():.1e}")
print(f"Diferencia máxima entre las probabilidades predichas en validación: {np.abs(p_fit - p_bucle).max():.1e}")

fig, ax = plt.subplots(figsize=(9, 3.6))
ax.plot(range(1, EPOCAS_CHK + 1), perd_fit, color=COL["logit"], lw=4, alpha=0.5, label="model.fit (Keras)")
ax.plot(H_chk["epoca"], perd_bucle, color=COL["mlp"], lw=1.4, ls="--", label="Bucle explícito (tf.GradientTape)")
ax.set_xlabel("Época"); ax.set_ylabel("Pérdida optimizada (BCE ponderada + L2)")
ax.set_title("Con los mismos datos por paso, ambas vías producen la misma trayectoria"); ax.legend()
fig.tight_layout(); guardar(fig, "fig09_verificacion_bucle"); plt.show()

# ======== Celda de código 26 (índice 50): 8.3 Entrenamiento del modelo base (fase 1: entrenamiento 2021-2024, validación 2025) ========
# =============================================================================
# 8.3 Entrenamiento del modelo base (fase 1: entrenamiento 2021-2024, validación 2025)
# =============================================================================
t0 = time.time()
modelo, HIST, INFO, predecir = entrenar_mlp(Xtr, ytr, Xva, yva, cfg=CFG, semilla=SEED)
T_ENTRENAMIENTO = time.time() - t0
reducciones_lr = int(np.sum(np.diff(HIST["lr"]) < 0))
info_tabla = pd.Series({
    "Épocas ejecutadas": INFO["epocas_ejecutadas"],
    "Mejor época (mínima BCE de validación)": INFO["mejor_epoca"],
    "BCE de validación en la mejor época": round(INFO["mejor_bce_val"], 4),
    "Reducciones de la tasa de aprendizaje": reducciones_lr,
    "Tasa de aprendizaje final": f"{HIST['lr'][-1]:.2e}",
    "Actualizaciones de pesos (pasos de gradiente)": INFO["epocas_ejecutadas"] * math.ceil(len(ytr) / CFG.batch),
    "Tiempo de entrenamiento (s)": round(T_ENTRENAMIENTO, 1),
}, name="Valor")
display(info_tabla.to_frame())

# ======== Celda de código 27 (índice 52): 9.1 Curvas de pérdida, exactitud, AUC y tasa de aprendizaje por época ========
# =============================================================================
# 9.1 Curvas de pérdida, exactitud, AUC y tasa de aprendizaje por época
# =============================================================================
ep = np.array(HIST["epoca"]); be = INFO["mejor_epoca"]
H0 = bce(ytr, np.full_like(ytr, prev_train))   # BCE de un predictor constante (no aprende nada)
fig, axes = plt.subplots(2, 2, figsize=(14, 8.4))

ax = axes[0, 0]
ax.plot(ep, HIST["perdida_obj"], color=COL["train"], lw=1, alpha=0.35,
        label="Objetivo optimizado (train, BCE ponderada + L2" + (" con dropout)" if CFG.dropout else ")"))
ax.plot(ep, HIST["bce_tr"], color=COL["train"], label="BCE entrenamiento (inferencia)")
ax.plot(ep, HIST["bce_va"], color=COL["val"], label="BCE validación 2025")
ax.axhline(H0, color=TENUE, lw=1, ls="--"); ax.text(ep[-1], H0, "predictor constante ", ha="right", va="bottom", fontsize=8, color=TINTA2)
ax.axvline(be, color=TINTA, lw=1, ls=":"); ax.text(be, ax.get_ylim()[1] * 0.97, f" mejor época = {be}", fontsize=8.5, va="top", color=TINTA)
ax.set_title("(a) Entropía cruzada binaria"); ax.set_ylabel("BCE"); ax.legend(fontsize=8, loc="upper right")

ax = axes[0, 1]
ax.plot(ep, HIST["acc_tr"], color=COL["train"], label="Entrenamiento")
ax.plot(ep, HIST["acc_va"], color=COL["val"], label="Validación 2025")
ax.axhline(max(prev_train, 1 - prev_train), color=TENUE, lw=1, ls="--")
ax.text(ep[-1], max(prev_train, 1 - prev_train), "clase mayoritaria ", ha="right", va="bottom", fontsize=8, color=TINTA2)
ax.axvline(be, color=TINTA, lw=1, ls=":")
ax.set_title("(b) Exactitud (umbral 0,5)"); ax.set_ylabel("Exactitud"); ax.legend(fontsize=8, loc="lower right")

ax = axes[1, 0]
ax.plot(ep, HIST["auc_tr"], color=COL["train"], label="Entrenamiento")
ax.plot(ep, HIST["auc_va"], color=COL["val"], label="Validación 2025")
ax.axhline(0.5, color=TENUE, lw=1, ls="--"); ax.axvline(be, color=TINTA, lw=1, ls=":")
ax.set_title("(c) ROC-AUC (capacidad de ordenamiento)"); ax.set_ylabel("AUC"); ax.set_xlabel("Época"); ax.legend(fontsize=8, loc="lower right")

ax = axes[1, 1]
ax.plot(ep, HIST["lr"], color=COL["mlp"]); ax.set_yscale("log"); ax.axvline(be, color=TINTA, lw=1, ls=":")
ax.set_title("(d) Tasa de aprendizaje (reducción en meseta)"); ax.set_ylabel("η (escala log)"); ax.set_xlabel("Época")
fig.tight_layout(); guardar(fig, "fig10_curvas_aprendizaje"); plt.show()

gap = pd.DataFrame({"Entrenamiento": [HIST["bce_tr"][be - 1], HIST["acc_tr"][be - 1], HIST["auc_tr"][be - 1]],
                    "Validación": [HIST["bce_va"][be - 1], HIST["acc_va"][be - 1], HIST["auc_va"][be - 1]]},
                   index=["BCE", "Exactitud", "ROC-AUC"])
gap["Brecha (train − val)"] = gap["Entrenamiento"] - gap["Validación"]
print(f"BCE de referencia (predictor constante con prevalencia de entrenamiento): {H0:.4f}")
tabla(gap, dec=3)

# ======== Celda de código 28 (índice 55): 10.1 Predicciones de todos los modelos y reajuste con 2021-2025 (fase 2) ========
# =============================================================================
# 10.1 Predicciones de todos los modelos y reajuste con 2021-2025 (fase 2)
# =============================================================================
X_de = {"Entrenamiento": Xtr, "Validación": Xva, "Prueba": Xte}
y_de = {"Entrenamiento": ytr, "Validación": yva, "Prueba": yte}
m_de = {"Entrenamiento": m_tr, "Validación": m_va, "Prueba": m_te}
P = {("MLP fase 1", k): predecir(X) for k, X in X_de.items()}
P.update({("Logística fase 1", k): logit.predict_proba(X)[:, 1] for k, X in X_de.items()})
P.update({("Persistencia", k): score_persistencia(m) for k, m in m_de.items()})
P.update({("Clase mayoritaria", k): np.full(len(y_de[k]), prev_train) for k in X_de})

# Fase 2: reajuste con entrenamiento + validación (2021-2025), mismas épocas y calendario de LR
m_trva = m_tr | m_va
escalador_f2 = StandardScaler().fit(datos.loc[m_trva, FEATURES])
Xtrva_f2, Xte_f2 = matriz(m_trva, escalador_f2), matriz(m_te, escalador_f2)
ytrva = datos.loc[m_trva, "y"].values.astype("float32")
modelo_f2, _, _, predecir_f2 = entrenar_mlp(Xtrva_f2, ytrva, cfg=CFG, semilla=SEED, epocas=INFO["mejor_epoca"],
                                            calendario_lr=INFO["calendario_lr"], diagnostico=False)
logit_f2 = LogisticRegression(C=C_LR, class_weight="balanced", max_iter=5000).fit(Xtrva_f2, ytrva)
P[("MLP fase 2", "Prueba")] = predecir_f2(Xte_f2)
P[("Logística fase 2", "Prueba")] = logit_f2.predict_proba(Xte_f2)[:, 1]
P[("MLP fase 2", "Entrenamiento")] = predecir_f2(Xtrva_f2)

filas = []
for (modelo_n, conjunto), p in P.items():
    y_ = ytrva if (modelo_n == "MLP fase 2" and conjunto == "Entrenamiento") else y_de[conjunto]
    filas.append({"Modelo": modelo_n, "Conjunto": conjunto if not (modelo_n == "MLP fase 2" and conjunto == "Entrenamiento")
                  else "Entrenamiento (2021-25)", **metricas(y_, p)})
MET = pd.DataFrame(filas).set_index(["Modelo", "Conjunto"])
orden_m = ["Clase mayoritaria", "Persistencia", "Logística fase 1", "MLP fase 1", "Logística fase 2", "MLP fase 2"]
orden_c = ["Entrenamiento", "Entrenamiento (2021-25)", "Validación", "Prueba"]
MET = MET.reindex([(m, c) for m in orden_m for c in orden_c if (m, c) in MET.index])
cols = ["Exactitud", "Exact. balanceada", "Sensibilidad C", "Especificidad D", "F1 C", "MCC", "ROC-AUC", "PR-AUC", "Log-loss", "Brier"]
display(MET[cols].style.format(precision=3, decimal=",", na_rep="–")
        .background_gradient(subset=["ROC-AUC", "Exact. balanceada", "MCC"], cmap=CMAP_P, vmin=0.4, vmax=1.0)
        .background_gradient(subset=["Log-loss"], cmap=CMAP_P.reversed(), vmin=0.2, vmax=1.0))

# ======== Celda de código 29 (índice 56): 10.2 Incertidumbre de la prueba: intervalos de confianza bootstrap al 95 % ========
# =============================================================================
# 10.2 Incertidumbre de la prueba: intervalos de confianza bootstrap al 95 %
# =============================================================================
fns = {"Exactitud": lambda y, p: accuracy_score(y, p >= .5), "ROC-AUC": roc_auc_score, "Log-loss": bce}
filas = []
for nombre in ["Persistencia", "Logística fase 1", "MLP fase 1", "Logística fase 2", "MLP fase 2"]:
    p = P[(nombre, "Prueba")]
    fila = {"Modelo": nombre}
    for met, fn in fns.items():
        lo, hi = bootstrap_ic(yte, p, fn)
        fila[met] = f"{fn(yte, p):.3f}  [{lo:.3f} – {hi:.3f}]"
    filas.append(fila)
IC_TEST = pd.DataFrame(filas).set_index("Modelo")
display(IC_TEST)
print(f"n prueba = {len(yte)} departamentos (C = {int(yte.sum())}, D = {int(len(yte) - yte.sum())}); "
      f"cada error equivale a {1 / len(yte):.1%} de exactitud.")

# ======== Celda de código 30 (índice 58): 10.3 Matrices de confusión ========
# =============================================================================
# 10.3 Matrices de confusión
# =============================================================================
casos = [("MLP fase 1 · Validación 2025", yva, P[("MLP fase 1", "Validación")]),
         ("MLP fase 1 · Prueba 2026", yte, P[("MLP fase 1", "Prueba")]),
         ("MLP fase 2 · Prueba 2026", yte, P[("MLP fase 2", "Prueba")]),
         ("Persistencia · Prueba 2026", yte, P[("Persistencia", "Prueba")])]
fig, axes = plt.subplots(1, 4, figsize=(16, 3.9))
for ax, (tit, y_, p_) in zip(axes, casos):
    cm = confusion_matrix(y_.astype(int), (p_ >= .5).astype(int), labels=[0, 1])
    ax.imshow(cm, cmap=CMAP_P, vmin=0, vmax=cm.max() * 1.3); ax.grid(False)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]}\n({cm[i, j] / cm[i].sum():.0%})", ha="center", va="center", fontsize=10,
                    color="white" if cm[i, j] > cm.max() * 0.65 else TINTA)
    ax.set_xticks([0, 1], ["D", "C"]); ax.set_yticks([0, 1], ["D", "C"])
    ax.set_xlabel("Predicho"); ax.set_ylabel("Real"); ax.set_title(tit, fontsize=9.5)
fig.tight_layout(); guardar(fig, "fig11_matrices_confusion"); plt.show()

# ======== Celda de código 31 (índice 59): 10.4 Curvas ROC y precisión–sensibilidad en la prueba (2026) ========
# =============================================================================
# 10.4 Curvas ROC y precisión–sensibilidad en la prueba (2026)
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
curvas = [("MLP fase 2", COL["mlp"], "-"), ("MLP fase 1", COL["mlp"], ":"), ("Logística fase 2", COL["logit"], "-"),
          ("Persistencia", COL["pers"], "--")]
for nombre, color, ls in curvas:
    p = P[(nombre, "Prueba")]
    fpr, tpr, _ = roc_curve(yte, p); prec, rec, _ = precision_recall_curve(yte, p)
    axes[0].plot(fpr, tpr, color=color, ls=ls, label=f"{nombre} (AUC = {roc_auc_score(yte, p):.3f})")
    axes[1].plot(rec, prec, color=color, ls=ls, label=f"{nombre} (AP = {average_precision_score(yte, p):.3f})")
axes[0].plot([0, 1], [0, 1], color=EJE, lw=1); axes[1].axhline(yte.mean(), color=EJE, lw=1)
axes[0].set_xlabel("Tasa de falsos positivos (1 − especificidad)"); axes[0].set_ylabel("Sensibilidad (C)")
axes[1].set_xlabel("Sensibilidad (C)"); axes[1].set_ylabel("Precisión (C)")
axes[0].set_title("(a) Curva ROC · prueba 2026"); axes[1].set_title("(b) Curva precisión–sensibilidad · prueba 2026")
for ax in axes:
    ax.legend(fontsize=8.5, loc="lower right" if ax is axes[0] else "lower left")
fig.tight_layout(); guardar(fig, "fig12_roc_pr"); plt.show()

# ======== Celda de código 32 (índice 60): 10.5 Calibración: ¿una probabilidad de 0,8 significa 80 % de departamentos C? ========
# =============================================================================
# 10.5 Calibración: ¿una probabilidad de 0,8 significa 80 % de departamentos C?
# =============================================================================
y_fuera = np.concatenate([yva, yte])
fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [1, 1.1]})
ax = axes[0]
ax.plot([0, 1], [0, 1], color=EJE, lw=1)
for nombre, color in [("MLP fase 1", COL["mlp"]), ("Logística fase 1", COL["logit"])]:
    p = np.concatenate([P[(nombre, "Validación")], P[(nombre, "Prueba")]])
    fr, mp = calibration_curve(y_fuera, p, n_bins=5, strategy="quantile")
    ax.plot(mp, fr, marker="o", ms=7, color=color, mec=FONDO, mew=2,
            label=f"{nombre} (Brier = {brier_score_loss(y_fuera, p):.3f})")
ax.set_xlabel("Probabilidad media predicha"); ax.set_ylabel("Fracción observada de C")
ax.set_title("(a) Diagrama de confiabilidad · validación + prueba"); ax.legend(fontsize=8.5, loc="upper left")
ax = axes[1]
p = np.concatenate([P[("MLP fase 1", "Validación")], P[("MLP fase 1", "Prueba")]])
ax.hist(p[y_fuera == 0], bins=np.linspace(0, 1, 11), color=COL["D"], alpha=0.85, label="Real D", edgecolor=FONDO, lw=1.5)
ax.hist(p[y_fuera == 1], bins=np.linspace(0, 1, 11), color=COL["C"], alpha=0.75, label="Real C", edgecolor=FONDO, lw=1.5)
ax.axvline(0.5, color=TINTA, lw=1, ls="--")
ax.set_xlabel("P(C) predicha por el MLP (fase 1)"); ax.set_ylabel("Departamento–año")
ax.set_title("(b) Separación de probabilidades por clase real"); ax.legend(fontsize=8.5)
fig.tight_layout(); guardar(fig, "fig13_calibracion"); plt.show()

# ======== Celda de código 33 (índice 61): 10.6 Umbral de decisión: costo asimétrico de los errores ========
# =============================================================================
# 10.6 Umbral de decisión: costo asimétrico de los errores
# =============================================================================
# Un falso negativo (C clasificado como D) deja sin capacidades ISR/detección un territorio de violencia
# armada directa; un falso positivo sobreasigna recursos. Priorizamos la sensibilidad con F2 (β = 2).
umbrales = np.linspace(0.05, 0.95, 91)
p_va = P[("MLP fase 1", "Validación")]
curva_u = pd.DataFrame({"umbral": umbrales,
                        "Sensibilidad C": [recall_score(yva, p_va >= u, zero_division=0) for u in umbrales],
                        "Especificidad D": [recall_score(1 - yva, p_va < u, zero_division=0) for u in umbrales],
                        "F1 C": [f1_score(yva, p_va >= u, zero_division=0) for u in umbrales],
                        "F2 C": [fbeta_score(yva, p_va >= u, beta=2, zero_division=0) for u in umbrales]})
U_OPT = float(curva_u.loc[curva_u["F2 C"].idxmax(), "umbral"])

fig, ax = plt.subplots(figsize=(10, 4.4))
for col, color, ls in [("Sensibilidad C", COL["C"], "-"), ("Especificidad D", COL["D"], "-"), ("F1 C", COL["mlp"], "--"), ("F2 C", COL["mlp"], "-")]:
    ax.plot(curva_u["umbral"], curva_u[col], color=color, ls=ls, label=col)
ax.axvline(0.5, color=EJE, lw=1); ax.axvline(U_OPT, color=TINTA, lw=1.2, ls=":")
ax.text(U_OPT, 0.03, f" umbral F2 = {U_OPT:.2f}", fontsize=9, color=TINTA)
ax.set_xlabel("Umbral sobre P(C)"); ax.set_ylabel("Métrica en validación 2025")
ax.set_title("Selección del umbral en validación (el conjunto de prueba no se usa)"); ax.legend(fontsize=8.5, ncol=2, loc="lower left")
fig.tight_layout(); guardar(fig, "fig14_umbral"); plt.show()

comp_u = pd.DataFrame({f"Umbral 0,50": metricas(yte, P[("MLP fase 1", "Prueba")], 0.5),
                       f"Umbral {U_OPT:.2f} (F2 en validación)": metricas(yte, P[("MLP fase 1", "Prueba")], U_OPT)}).T
tabla(comp_u[["Exactitud", "Exact. balanceada", "Sensibilidad C", "Especificidad D", "F1 C", "MCC"]], dec=3)

# ======== Celda de código 34 (índice 65): 11.1 Señales de convergencia: gradiente, razón de actualización y pérdida ========
# =============================================================================
# 11.1 Señales de convergencia: gradiente, razón de actualización y pérdida
# =============================================================================
ratios = np.array(HIST["ratio_act"])
capas_k = [c.name for c in modelo.layers if hasattr(c, "kernel")]
cortes_lr = [e for e, a, b in zip(ep[1:], HIST["lr"][:-1], HIST["lr"][1:]) if b < a]

fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
ax = axes[0]
ax.plot(ep, HIST["perdida_obj"], color=COL["train"], lw=1.4, label="Objetivo (train)")
ax.plot(ep, HIST["bce_va"], color=COL["val"], lw=1.4, label="BCE validación")
ax.set_yscale("log"); ax.set_title("(a) Pérdida (escala log)"); ax.set_xlabel("Época"); ax.legend(fontsize=8.5)
ax = axes[1]
ax.plot(ep, HIST["norma_grad"], color=COL["mlp"], lw=1.2)
ax.plot(ep, pd.Series(HIST["norma_grad"]).rolling(15, min_periods=1).mean(), color=TINTA, lw=1.4, label="media móvil 15 épocas")
ax.set_yscale("log"); ax.set_title("(b) Norma global del gradiente"); ax.set_xlabel("Época"); ax.legend(fontsize=8.5)
ax = axes[2]
paleta_capas = [COL["mlp"], COL["logit"], COL["test"], COL["C"]]
for j, nombre in enumerate(capas_k):
    ax.plot(ep, ratios[:, j], color=paleta_capas[j % 4], lw=1.2, label=nombre)
ax.axhspan(1e-4, 1e-2, color=GRID, alpha=0.5, lw=0, label="rango de referencia 10⁻⁴–10⁻²"); ax.set_yscale("log")
ax.set_title("(c) Razón de actualización ‖ΔW‖/‖W‖ por capa"); ax.set_xlabel("Época"); ax.legend(fontsize=8.5)
for a in axes:
    a.axvline(INFO["mejor_epoca"], color=TINTA, lw=1, ls=":")
    for c in cortes_lr:
        a.axvline(c, color=TENUE, lw=0.8, ls="--")
fig.tight_layout(); guardar(fig, "fig15_convergencia"); plt.show()

bce_tr = np.array(HIST["bce_tr"]); reduccion = bce_tr[0] - bce_tr.min()
ep90 = int(ep[np.argmax(bce_tr <= bce_tr[0] - 0.9 * reduccion)])
conv = pd.Series({
    "BCE train: inicial → mínima": f"{bce_tr[0]:.3f} → {bce_tr.min():.3f}",
    "Época en que se logra el 90 % de la reducción de la BCE de train": ep90,
    "Pendiente de la BCE de train en las últimas 20 épocas (por época)": f"{np.polyfit(ep[-20:], bce_tr[-20:], 1)[0]:+.2e}",
    "Norma del gradiente: primeras 5 → últimas 5 épocas": f"{np.mean(HIST['norma_grad'][:5]):.3f} → {np.mean(HIST['norma_grad'][-5:]):.3f}",
    "Razón de actualización mediana (últimas 20 épocas)": ", ".join(f"{n}: {np.median(ratios[-20:, j]):.1e}" for j, n in enumerate(capas_k)),
    "Reducciones de η (épocas)": ", ".join(map(str, cortes_lr)) or "ninguna",
    "Mejor época / épocas ejecutadas": f"{INFO['mejor_epoca']} / {INFO['epocas_ejecutadas']}",
}, name="Valor")
display(conv.to_frame())

# ======== Celda de código 35 (índice 68): 11.2 Reentrenamiento con 10 semillas ========
# =============================================================================
# 11.2 Reentrenamiento con 10 semillas
# =============================================================================
SEMILLAS = list(range(101, 111))
filas, curvas_va, probs_te, probs_va = [], [], [], []
t0 = time.time()
for s in SEMILLAS:
    _, H_s, inf_s, pred_s = entrenar_mlp(Xtr, ytr, Xva, yva, cfg=CFG, semilla=s, diagnostico=False)
    pv, pt = pred_s(Xva), pred_s(Xte)
    curvas_va.append(H_s["bce_va"]); probs_te.append(pt); probs_va.append(pv)
    filas.append({"Semilla": s, "Mejor época": inf_s["mejor_epoca"], "BCE val": bce(yva, pv), "AUC val": roc_auc_score(yva, pv),
                  "BCE prueba": bce(yte, pt), "AUC prueba": roc_auc_score(yte, pt), "Exactitud prueba": accuracy_score(yte, pt >= .5)})
ESTAB = pd.DataFrame(filas).set_index("Semilla")
print(f"10 entrenamientos en {time.time() - t0:.0f} s")
resumen_estab = ESTAB.agg(["mean", "std", "min", "max"]).T
resumen_estab["CV (%)"] = 100 * resumen_estab["std"] / resumen_estab["mean"]
resumen_estab.columns = ["Media", "Desv. est.", "Mín.", "Máx.", "CV (%)"]
tabla(resumen_estab, dec=3)

# Presupuesto de entrenamiento para los reentrenamientos sin validación (§11.3, §12, §13):
# mediana de la mejor época entre semillas, expresada en pasos de gradiente para que no dependa del tamaño del conjunto.
EPOCAS_REF = int(np.median(ESTAB["Mejor época"]))
PASOS_REF = EPOCAS_REF * math.ceil(len(ytr) / CFG.batch)
print(f"Presupuesto de referencia: mediana de la mejor época = {EPOCAS_REF} épocas ≈ {PASOS_REF} pasos de gradiente "
      f"(la semilla 42 del modelo base se detuvo en la época {INFO['mejor_epoca']})")

probs_te = np.array(probs_te); p_ens = probs_te.mean(axis=0)
acuerdo = (probs_te >= .5).mean(axis=0)
unanimes = np.mean((acuerdo == 0) | (acuerdo == 1))
ens = pd.DataFrame({"Semilla individual (media)": [ESTAB["BCE prueba"].mean(), ESTAB["AUC prueba"].mean(), ESTAB["Exactitud prueba"].mean()],
                    "Ensamble de 10 semillas": [bce(yte, p_ens), roc_auc_score(yte, p_ens), accuracy_score(yte, p_ens >= .5)]},
                   index=["BCE prueba", "AUC prueba", "Exactitud prueba"])
tabla(ens, dec=3)
print(f"Departamentos de prueba con predicción unánime entre las 10 semillas: {unanimes:.0%}")

fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), gridspec_kw={"width_ratios": [1.2, 0.8, 1.3]})
ax = axes[0]
for c, s in zip(curvas_va, SEMILLAS):
    ax.plot(range(1, len(c) + 1), c, color=COL["mlp"], lw=0.9, alpha=0.45)
    k = int(np.argmin(c)); ax.scatter(k + 1, c[k], s=22, color=COL["val"], edgecolor=FONDO, lw=1, zorder=3)
ax.set_title("(a) BCE de validación por semilla (● = mejor época)"); ax.set_xlabel("Época"); ax.set_ylabel("BCE validación")
ax = axes[1]
datos_box = [ESTAB["BCE val"], ESTAB["BCE prueba"]]
ax.boxplot(datos_box, widths=0.45, showfliers=False, medianprops=dict(color=TINTA))
for k, v in enumerate(datos_box, 1):
    ax.scatter(np.full(len(v), k) + np.random.default_rng(k).uniform(-.1, .1, len(v)), v, s=18, color=[COL["val"], COL["test"]][k - 1], zorder=3)
ax.set_xticks([1, 2], ["Validación 2025", "Prueba 2026"]); ax.set_ylabel("BCE"); ax.set_title("(b) Dispersión entre semillas")
ax = axes[2]
orden_t = np.argsort(acuerdo)
nombres_te = datos.loc[m_te, "departamento"].values
ax.barh(range(len(orden_t)), acuerdo[orden_t], color=[COL["C"] if yte[i] == 1 else COL["D"] for i in orden_t], height=0.7)
ax.set_yticks(range(len(orden_t)), nombres_te[orden_t], fontsize=7); ax.axvline(0.5, color=TINTA, lw=1, ls="--")
ax.set_xlabel("Fracción de semillas que predicen C"); ax.set_title("(c) Acuerdo entre semillas · prueba 2026")
ax.legend(handles=[Patch(color=COL["C"], label="Real C"), Patch(color=COL["D"], label="Real D")], fontsize=8, loc="lower right")
fig.tight_layout(); guardar(fig, "fig16_estabilidad_semillas"); plt.show()

# ======== Celda de código 36 (índice 71): 11.3a-b Validación temporal de origen móvil y validación espacial por departamentos ========
# =============================================================================
# 11.3a-b Validación temporal de origen móvil y validación espacial por departamentos
# =============================================================================
def entrenar_fijo(Xa, ya, semilla=SEED, cfg=None):
    """Entrenamiento sin validación: presupuesto fijo de PASOS_REF pasos de gradiente (η constante)."""
    cfg = cfg or CFG
    epocas = max(1, math.ceil(PASOS_REF / math.ceil(len(ya) / cfg.batch)))
    return entrenar_mlp(Xa, ya, cfg=cfg, semilla=semilla, epocas=epocas, diagnostico=False)[3]

def evaluar_particion(tr, te_, columnas, semillas=(SEED,)):
    esc = StandardScaler().fit(datos.loc[tr, columnas])
    Xa, Xb = esc.transform(datos.loc[tr, columnas]), esc.transform(datos.loc[te_, columnas])
    ya, yb = datos.loc[tr, "y"].values.astype("float32"), datos.loc[te_, "y"].values
    s = datos.loc[te_, "s_C_prev"].values.astype(float)
    return yb, {"MLP": np.mean([entrenar_fijo(Xa, ya, semilla=k)(Xb) for k in semillas], axis=0),
                "Logística": LogisticRegression(C=C_LR, class_weight="balanced", max_iter=5000).fit(Xa, ya).predict_proba(Xb)[:, 1],
                "Persistencia": np.where(np.isnan(s), ya.mean(), s)}

def validacion(tipo="temporal", columnas=None, semillas=(SEED,)):
    columnas = columnas or FEATURES
    if tipo == "temporal":
        particiones_ = [(f"{v}", (datos.anio < v).values, (datos.anio == v).values) for v in (2023, 2024, 2025, 2026)]
    else:
        gkf = GroupKFold(n_splits=5)
        particiones_ = [(f"pliegue {k + 1}", np.isin(np.arange(len(datos)), a), np.isin(np.arange(len(datos)), b))
                        for k, (a, b) in enumerate(gkf.split(datos, datos["y"], groups=datos["depto"]))]
    filas, oos = [], []
    for nombre, tr, te_ in particiones_:
        yb, preds = evaluar_particion(tr, te_, columnas, semillas)
        for mod, p in preds.items():
            filas.append({"Partición": nombre, "Modelo": mod, "n": len(yb), **metricas(yb, p)})
        oos.append(pd.DataFrame({"particion": nombre, "fila": np.where(te_)[0], "y": yb, **preds}))
    return pd.DataFrame(filas), pd.concat(oos, ignore_index=True)

t0 = time.time()
VT, OOS_T = validacion("temporal")
VE, OOS_E = validacion("espacial")
print(f"Validaciones temporal y espacial en {time.time() - t0:.0f} s")

def agregado(oos):
    return pd.DataFrame({m: metricas(oos["y"], oos[m]) for m in ["MLP", "Logística", "Persistencia"]}).T

cols_g = ["Exactitud", "Exact. balanceada", "F1 C", "MCC", "ROC-AUC", "Log-loss", "Brier"]
print("Validación temporal (predicciones fuera de muestra 2023-2026, agregadas):")
tabla(agregado(OOS_T)[cols_g], dec=3)
print("Validación espacial (departamentos nunca vistos, agregadas):")
tabla(agregado(OOS_E)[cols_g], dec=3)

fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
for ax, met in zip(axes[:2], ["ROC-AUC", "Log-loss"]):
    for mod, color, ls in [("MLP", COL["mlp"], "-"), ("Logística", COL["logit"], "-"), ("Persistencia", COL["pers"], "--")]:
        s = VT[VT.Modelo == mod]
        ax.plot(s["Partición"], s[met], color=color, ls=ls, marker="o", ms=7, mec=FONDO, mew=2, label=mod)
    ax.set_xlabel("Año predicho (entrenando con los años anteriores)"); ax.set_title(f"({'ab'[axes.tolist().index(ax)]}) {met} · validación temporal")
    ax.legend(fontsize=8.5)
ax = axes[2]
agT, agE = agregado(OOS_T), agregado(OOS_E)
x = np.arange(3); ancho = 0.36
for k, (ag, nombre, alpha) in enumerate([(agT, "Temporal", 1.0), (agE, "Espacial", 0.55)]):
    ax.bar(x + (k - .5) * ancho, ag.loc[["MLP", "Logística", "Persistencia"], "ROC-AUC"], width=ancho * 0.92,
           color=[COL["mlp"], COL["logit"], COL["pers"]], alpha=alpha, label=nombre)
    for xi, v in zip(x + (k - .5) * ancho, ag.loc[["MLP", "Logística", "Persistencia"], "ROC-AUC"]):
        ax.text(xi, v + 0.01, f"{v:.2f}", ha="center", fontsize=8, color=TINTA2)
ax.set_xticks(x, ["MLP", "Logística", "Persistencia"]); ax.set_ylim(0.5, 1.02); ax.set_ylabel("ROC-AUC agregado")
ax.set_title("(c) Temporal (sólido) vs. espacial (claro)")
ax.legend(handles=[Patch(color=TENUE, alpha=1, label="Temporal: años nuevos"), Patch(color=TENUE, alpha=0.55, label="Espacial: departamentos nuevos")], fontsize=8, loc="lower left")
fig.tight_layout(); guardar(fig, "fig17_generalizacion_temporal_espacial"); plt.show()

# ======== Celda de código 37 (índice 73): 11.3c-d Diagnóstico sesgo–varianza: tamaño de muestra y capacidad ========
# =============================================================================
# 11.3c-d Diagnóstico sesgo–varianza: tamaño de muestra y capacidad
# =============================================================================
t0 = time.time()
# (c) Curva de aprendizaje por tamaño de muestra: submuestras estratificadas del entrenamiento 2021-2024
FRACCIONES, REPETICIONES = [0.3, 0.45, 0.6, 0.8, 1.0], 3
idx0, idx1 = np.where(ytr == 0)[0], np.where(ytr == 1)[0]
filas = []
for f in FRACCIONES:
    for r in range(REPETICIONES):
        rg = np.random.default_rng(1000 + r)
        sub = np.concatenate([rg.choice(idx0, int(round(f * len(idx0))), replace=False),
                              rg.choice(idx1, int(round(f * len(idx1))), replace=False)])
        _, _, _, pr = entrenar_mlp(Xtr[sub], ytr[sub], Xva, yva, cfg=CFG, semilla=SEED + r, diagnostico=False)
        lr_ = LogisticRegression(C=C_LR, class_weight="balanced", max_iter=5000).fit(Xtr[sub], ytr[sub])
        filas.append({"n": len(sub), "MLP train": bce(ytr[sub], pr(Xtr[sub])), "MLP prueba": bce(yte, pr(Xte)),
                      "Logística train": bce(ytr[sub], lr_.predict_proba(Xtr[sub])[:, 1]),
                      "Logística prueba": bce(yte, lr_.predict_proba(Xte)[:, 1]), "AUC MLP prueba": roc_auc_score(yte, pr(Xte)),
                      "AUC Logística prueba": roc_auc_score(yte, lr_.predict_proba(Xte)[:, 1])})
CURVA_N = pd.DataFrame(filas).groupby("n").agg(["mean", "std"])

# (d) Capacidad: una capa oculta de 2 a 128 neuronas, con y sin regularización, épocas fijas
UNIDADES, EPOCAS_CAP = [2, 4, 8, 16, 32, 64, 128], 400
filas = []
for u in UNIDADES:
    for regimen, cfg_r in [("Sin regularización", ConfigMLP(capas=(u,), l2=0.0, dropout=0.0)),
                           ("Con L2 del modelo base", ConfigMLP(capas=(u,), l2=CFG.l2, dropout=CFG.dropout))]:
        for s in (1, 2):
            _, _, _, pr = entrenar_mlp(Xtr, ytr, cfg=cfg_r, semilla=s, epocas=EPOCAS_CAP, diagnostico=False)
            filas.append({"unidades": u, "régimen": regimen, "semilla": s, "BCE train": bce(ytr, pr(Xtr)), "BCE validación": bce(yva, pr(Xva))})
CAPACIDAD = pd.DataFrame(filas).groupby(["régimen", "unidades"])[["BCE train", "BCE validación"]].mean().reset_index()
print(f"Curvas de tamaño de muestra y capacidad en {time.time() - t0:.0f} s")

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
ax = axes[0]; nn = CURVA_N.index.values
for col, color, ls in [("MLP train", COL["mlp"], ":"), ("MLP prueba", COL["mlp"], "-"), ("Logística train", COL["logit"], ":"), ("Logística prueba", COL["logit"], "-")]:
    m_, s_ = CURVA_N[(col, "mean")].values, CURVA_N[(col, "std")].fillna(0).values
    ax.plot(nn, m_, color=color, ls=ls, marker="o", ms=5, label=col); ax.fill_between(nn, m_ - s_, m_ + s_, color=color, alpha=0.1, lw=0)
ax.set_xlabel("Filas de entrenamiento"); ax.set_ylabel("BCE"); ax.set_title("(a) Pérdida vs. tamaño de muestra"); ax.legend(fontsize=8)
ax = axes[1]
for col, color in [("AUC MLP prueba", COL["mlp"]), ("AUC Logística prueba", COL["logit"])]:
    m_, s_ = CURVA_N[(col, "mean")].values, CURVA_N[(col, "std")].fillna(0).values
    ax.plot(nn, m_, color=color, marker="o", ms=5, label=col.replace("AUC ", "")); ax.fill_between(nn, m_ - s_, m_ + s_, color=color, alpha=0.1, lw=0)
ax.set_xlabel("Filas de entrenamiento"); ax.set_ylabel("ROC-AUC prueba 2026"); ax.set_title("(b) AUC de prueba vs. tamaño de muestra"); ax.legend(fontsize=8)
ax = axes[2]
for regimen, color in [("Sin regularización", COL["logit"]), ("Con L2 del modelo base", COL["mlp"])]:
    s = CAPACIDAD[CAPACIDAD["régimen"] == regimen]
    ax.plot(s["unidades"], s["BCE train"], color=color, ls=":", marker="o", ms=4)
    ax.plot(s["unidades"], s["BCE validación"], color=color, marker="o", ms=5, label=regimen)
ax.axvline(CFG.capas[0], color=TINTA, lw=1, ls=":"); ax.set_xscale("log", base=2); ax.set_xticks(UNIDADES, UNIDADES)
ax.set_xlabel("Neuronas en la capa oculta"); ax.set_ylabel("BCE (línea continua = validación, punteada = train)")
ax.set_title(f"(c) Capacidad vs. sobreajuste ({EPOCAS_CAP} épocas fijas)"); ax.legend(fontsize=8)
fig.tight_layout(); guardar(fig, "fig18_sesgo_varianza"); plt.show()

# ======== Celda de código 38 (índice 76): 11.4a Pruebas pareadas MLP vs. logística sobre predicciones fuera de muestra ========
# =============================================================================
# 11.4a Pruebas pareadas MLP vs. logística sobre predicciones fuera de muestra
# =============================================================================
def comparar_pareado(oos, a="MLP", b="Logística", n_boot=2000, semilla=SEED):
    y = oos["y"].values.astype(int); pa, pb = oos[a].values, oos[b].values
    ok_a, ok_b = (pa >= .5) == y, (pb >= .5) == y
    n01, n10 = int(np.sum(ok_a & ~ok_b)), int(np.sum(~ok_a & ok_b))
    p_mcnemar = stats.binomtest(min(n01, n10), n01 + n10, 0.5).pvalue if (n01 + n10) else 1.0
    rng = np.random.default_rng(semilla); d_auc, d_bce = [], []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) < 2:
            continue
        d_auc.append(roc_auc_score(y[i], pa[i]) - roc_auc_score(y[i], pb[i])); d_bce.append(bce(y[i], pa[i]) - bce(y[i], pb[i]))
    return {"n": len(y), f"solo {a} acierta": n01, f"solo {b} acierta": n10, "p McNemar (exacta)": p_mcnemar,
            "ΔAUC (MLP − Log.)": roc_auc_score(y, pa) - roc_auc_score(y, pb), "IC95 ΔAUC": np.percentile(d_auc, [2.5, 97.5]).round(3).tolist(),
            "ΔBCE (MLP − Log.)": bce(y, pa) - bce(y, pb), "IC95 ΔBCE": np.percentile(d_bce, [2.5, 97.5]).round(3).tolist()}

PAREADO = pd.DataFrame({"Temporal 2023-2026": comparar_pareado(OOS_T), "Espacial (5 pliegues)": comparar_pareado(OOS_E)}).T
display(PAREADO)

# ======== Celda de código 39 (índice 77): 11.4b Superficies de dependencia parcial (PDP 2D): MLP vs. logística ========
# =============================================================================
# 11.4b Superficies de dependencia parcial (PDP 2D): MLP vs. logística
# =============================================================================
def pdp_2d(pred_fn, esc, fx, fy, n=40):
    """PDP: promedio de P(C) sobre las filas de entrenamiento, fijando (fx, fy) en cada punto de la rejilla."""
    base = datos.loc[m_tr, FEATURES].reset_index(drop=True)
    gx = np.linspace(*datos[fx].quantile([.02, .98]), n); gy = np.linspace(*datos[fy].quantile([.02, .98]), n)
    GX, GY = np.meshgrid(gx, gy)
    rep = pd.concat([base] * (n * n), ignore_index=True)
    rep[fx] = np.repeat(GX.ravel(), len(base)); rep[fy] = np.repeat(GY.ravel(), len(base))
    Z = pred_fn(esc.transform(rep[FEATURES]).astype("float32")).reshape(n * n, len(base)).mean(axis=1).reshape(n, n)
    return gx, gy, Z

pred_lr_fn = lambda X: logit.predict_proba(X)[:, 1]
fig, axes = plt.subplots(2, 2, figsize=(13, 10))
for fila, (fx, fy) in enumerate([("x5_bosque", "x6_distancia"), ("x4_coca", "x7_ipm")]):
    for colm, (nombre, fn) in enumerate([("MLP (modelo base)", predecir), ("Regresión logística", pred_lr_fn)]):
        ax = axes[fila, colm]
        gx, gy, Z = pdp_2d(fn, escalador, fx, fy)
        cs = ax.contourf(gx, gy, Z, levels=np.linspace(0, 1, 11), cmap=CMAP_P, vmin=0, vmax=1)
        ax.contour(gx, gy, Z, levels=[0.5], colors=TINTA, linewidths=1.6)
        for clase, color in [(0, COL["D"]), (1, COL["C"])]:
            s = datos[m_tr & (datos.y == clase).values]
            ax.scatter(s[fx], s[fy], s=26, color=color, edgecolor=TINTA, lw=0.6, zorder=3)
        ax.grid(False); ax.set_xlabel(ETIQ[fx]); ax.set_ylabel(ETIQ[fy])
        ax.set_title(f"{nombre} · {ETIQ[fx].split(' ')[0]} × {ETIQ[fy].split(' ')[0]}")
cb = fig.colorbar(cs, ax=axes, fraction=0.025, pad=0.02); cb.set_label("P(C) parcial promedio"); cb.outline.set_visible(False)
fig.legend(handles=[Patch(color=COL["C"], label="Entrenamiento: real C"), Patch(color=COL["D"], label="Entrenamiento: real D"),
                    Line2D([], [], color=TINTA, lw=1.6, label="Frontera P = 0,5")], loc="lower center", ncol=3, fontsize=9)
guardar(fig, "fig19_dependencia_parcial"); plt.show()

# ======== Celda de código 40 (índice 80): 11.5a Importancia por permutación (aumento de la BCE al romper cada variable) en validación + prueba ========
# =============================================================================
# 11.5a Importancia por permutación (aumento de la BCE al romper cada variable) en validación + prueba
# =============================================================================
def importancia_perm(pred_fn, X, y, n_rep=30, semilla=SEED):
    rng = np.random.default_rng(semilla); base = bce(y, pred_fn(X)); res = {}
    for j, f in enumerate(FEATURES):
        d = []
        for _ in range(n_rep):
            Xp = X.copy(); Xp[:, j] = rng.permutation(Xp[:, j]); d.append(bce(y, pred_fn(Xp)) - base)
        res[ETIQ[f]] = (np.mean(d), np.std(d))
    return pd.DataFrame(res, index=["ΔBCE media", "Desv. est."]).T

X_fuera = np.vstack([Xva, Xte])
IMP = pd.concat({"MLP": importancia_perm(predecir, X_fuera, y_fuera), "Logística": importancia_perm(pred_lr_fn, X_fuera, y_fuera)}, axis=1)
orden_i = IMP[("MLP", "ΔBCE media")].sort_values().index
fig, ax = plt.subplots(figsize=(10, 4.8))
yy = np.arange(len(orden_i)); h = 0.38
ax.barh(yy + h / 2, IMP.loc[orden_i, ("MLP", "ΔBCE media")], xerr=IMP.loc[orden_i, ("MLP", "Desv. est.")], height=h * 0.9,
        color=COL["mlp"], label="MLP", error_kw=dict(ecolor=TENUE, lw=1))
ax.barh(yy - h / 2, IMP.loc[orden_i, ("Logística", "ΔBCE media")], xerr=IMP.loc[orden_i, ("Logística", "Desv. est.")], height=h * 0.9,
        color=COL["logit"], label="Logística", error_kw=dict(ecolor=TENUE, lw=1))
ax.axvline(0, color=EJE, lw=1); ax.set_yticks(yy, orden_i)
ax.set_xlabel("Aumento de la BCE al permutar la variable (mayor = más importante)")
ax.set_title("Importancia por permutación · validación 2025 + prueba 2026 (30 permutaciones)"); ax.legend(fontsize=9, loc="lower right")
fig.tight_layout(); guardar(fig, "fig20_importancia_permutacion"); plt.show()
tabla(IMP, dec=4)

# ======== Celda de código 41 (índice 81): 11.5b Análisis de errores: ¿dónde se equivoca el modelo? ========
# =============================================================================
# 11.5b Análisis de errores: ¿dónde se equivoca el modelo?
# =============================================================================
cols_err = ["departamento", "anio", "perfil", "s_C", "perfil_prev", "s_C_prev", "eventos"]
fuera = pd.concat([datos.loc[m_va, cols_err], datos.loc[m_te, cols_err]])      # mismo orden que [val, prueba]
fuera["P(C) MLP"] = np.concatenate([P[("MLP fase 1", "Validación")], P[("MLP fase 1", "Prueba")]])
fuera["predicho"] = np.where(fuera["P(C) MLP"] >= .5, "C", "D")
fuera["acierto"] = fuera["predicho"] == fuera["perfil"]
fuera["transición"] = fuera["perfil"] != fuera["perfil_prev"]
fuera["cerca del umbral"] = (fuera["s_C"] - .5).abs() < .10
errores = fuera[~fuera["acierto"]].sort_values(["anio", "P(C) MLP"])
display(errores.style.format({"s_C": "{:.2f}", "s_C_prev": "{:.2f}", "P(C) MLP": "{:.2f}"}).hide(axis="index"))
perfil_err = pd.DataFrame({"Errores": [errores["transición"].mean(), errores["cerca del umbral"].mean(), errores["eventos"].median()],
                           "Aciertos": [fuera.loc[fuera.acierto, "transición"].mean(), fuera.loc[fuera.acierto, "cerca del umbral"].mean(),
                                        fuera.loc[fuera.acierto, "eventos"].median()]},
                          index=["% que son transición de perfil (t-1 ≠ t)", "% con s_C a ±10 pp del umbral", "Mediana de eventos en t"])
tabla(perfil_err, dec=2)
trans_f = fuera[fuera["transición"]]
print(f"Transiciones de perfil en validación + prueba: {len(trans_f)} | acertadas por el MLP: {int(trans_f['acierto'].sum())} "
      f"| acertadas por la persistencia: 0 (por construcción)")

fig, ax = plt.subplots(figsize=(10, 6))
for anio, color, et in [(ANIO_VAL, COL["val"], "Validación 2025"), (ANIO_TEST, COL["test"], "Prueba 2026")]:
    s = fuera[fuera.anio == anio]
    ax.scatter(s["s_C"], s["P(C) MLP"], s=46, color=color, edgecolor=FONDO, lw=1.5, label=et, zorder=3)
ax.axvline(.5, color=EJE, lw=1); ax.axhline(.5, color=EJE, lw=1)
for _, r in errores.iterrows():
    ax.annotate(f"{r['departamento']} {r['anio']}", (r["s_C"], r["P(C) MLP"]), xytext=(5, 4), textcoords="offset points", fontsize=7.5, color=TINTA2)
ax.text(0.02, 0.97, "falso positivo", fontsize=8.5, color=TENUE, va="top"); ax.text(0.98, 0.03, "falso negativo", fontsize=8.5, color=TENUE, ha="right")
ax.set_xlabel("Proporción observada de eventos C en t ($s^C$)"); ax.set_ylabel("P(C) predicha con información de t-1")
ax.set_title("Probabilidad predicha vs. proporción observada (errores etiquetados)"); ax.legend(fontsize=9, loc="center right")
fig.tight_layout(); guardar(fig, "fig21_analisis_errores"); plt.show()

# ======== Celda de código 42 (índice 85): 12.1 Modelo de despliegue (ensamble de 5 semillas, 2021-2026) y proyección 2027 ========
# =============================================================================
# 12.1 Modelo de despliegue (ensamble de 5 semillas, 2021-2026) y proyección 2027
# =============================================================================
escalador_all = StandardScaler().fit(datos[FEATURES])
X_all, y_all = escalador_all.transform(datos[FEATURES]).astype("float32"), datos["y"].values.astype("float32")
SEMILLAS_DESPLIEGUE = [SEED, 7, 11, 23, 31]
ensamble = [entrenar_fijo(X_all, y_all, semilla=s) for s in SEMILLAS_DESPLIEGUE]

base27 = pext[pext["anio"] == 2026].copy()
X27 = escalador_all.transform(calcular_features(base27)[FEATURES]).astype("float32")
P27 = np.array([f(X27) for f in ensamble])

ab = panel[panel["anio"] >= 2025].groupby("depto")[["A", "B", "eventos"]].sum()
ab["s_AB"] = (ab["A"] + ab["B"]) / ab["eventos"]
nacional_AB = (ab["A"].sum() + ab["B"].sum()) / ab["eventos"].sum()
ab["alerta_AB"] = (ab["s_AB"] >= 2 * nacional_AB) & ((ab["A"] + ab["B"]) >= 5)

proy = pd.DataFrame({"depto": base27["depto"].values, "Departamento": base27["depto"].map(NOMBRE).values,
                     "lat": base27["lat"].values, "lon": base27["lon"].values,
                     "Eventos 2026 (anualizados)": (base27["eventos"] / base27["semanas"] * 52).round(0).values,
                     "Perfil 2026 observado": base27["perfil"].fillna("–").values,
                     "P(C) 2027": P27.mean(axis=0), "Desv. entre semillas": P27.std(axis=0)})
proy = proy.merge(ab[["s_AB", "alerta_AB"]], left_on="depto", right_index=True, how="left")
proy["Datos suficientes"] = base27["confiable"].values

def categoria(p):
    return ("C consolidado" if p >= 0.7 else "Transición" if p >= 0.3 else "D consolidado")
ENFASIS = {"C consolidado": "ISR, detección temprana y movilidad protegida",
           "Transición": "Portafolio flexible y monitoreo trimestral del perfil",
           "D consolidado": "Orden público y capacidades mixtas"}
proy["Categoría 2027"] = proy["P(C) 2027"].map(categoria)
proy["¿Coincide con la inercia?"] = np.where(proy["Perfil 2026 observado"] == "–", "–",
                                            np.where((proy["P(C) 2027"] >= .5) == (proy["Perfil 2026 observado"] == "C"), "sí", "revisar"))
proy["Énfasis sugerido"] = proy["Categoría 2027"].map(ENFASIS) + np.where(proy["alerta_AB"], " + evaluar C-UAS / detección de AEI", "")
proy = proy.sort_values("P(C) 2027", ascending=False).reset_index(drop=True)
print(f"Proporción nacional de eventos A+B (2025-2026): {nacional_AB:.1%} → umbral de alerta: {2 * nacional_AB:.1%}")
display(proy[["Departamento", "Perfil 2026 observado", "P(C) 2027", "Desv. entre semillas", "Categoría 2027",
              "¿Coincide con la inercia?", "s_AB", "Énfasis sugerido", "Datos suficientes"]]
        .style.format({"P(C) 2027": "{:.2f}", "Desv. entre semillas": "{:.3f}", "s_AB": "{:.1%}"})
        .background_gradient(subset=["P(C) 2027"], cmap=CMAP_P, vmin=0, vmax=1).hide(axis="index"))

# ======== Celda de código 43 (índice 86): 12.2 Mapa esquemático (centroides departamentales) ========
# =============================================================================
# 12.2 Mapa esquemático (centroides departamentales)
# =============================================================================
fig, ax = plt.subplots(figsize=(10, 11))
tam = 40 + 1.1 * np.sqrt(proy["Eventos 2026 (anualizados)"].clip(lower=1)) * 12
sc = ax.scatter(proy["lon"], proy["lat"], s=tam, c=proy["P(C) 2027"], cmap=CMAP_P, vmin=0, vmax=1, edgecolor=FONDO, lw=2, zorder=3)
alerta = proy[proy["alerta_AB"].fillna(False)]
ax.scatter(alerta["lon"], alerta["lat"], s=tam[alerta.index] + 260, facecolor="none", edgecolor="#d03b3b", lw=2, zorder=2)
a_la_izquierda = {"valle del cauca", "risaralda", "quindio", "sucre", "cordoba"}
for _, r in proy.iterrows():
    izq = r["depto"] in a_la_izquierda
    ax.annotate(f"{r['Departamento']}\n{r['P(C) 2027']:.2f}", (r["lon"], r["lat"]), xytext=(-9 if izq else 9, -3),
                textcoords="offset points", ha="right" if izq else "left",
                fontsize=7.4, color=TINTA if r["Datos suficientes"] else TENUE)
ax.set_aspect("equal"); ax.set_xlabel("Longitud"); ax.set_ylabel("Latitud")
cb = fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.02); cb.set_label("P(perfil C) proyectada 2027"); cb.outline.set_visible(False)
ax.legend(handles=[Line2D([], [], marker="o", ls="", mfc="none", mec="#d03b3b", mew=2, ms=12, label="Alerta A/B (IED/drones ≥ 2 × promedio nacional)"),
                   Line2D([], [], marker="o", ls="", color=TENUE, ms=6, label="Tamaño proporcional a los eventos 2026 · rótulo gris = datos insuficientes")],
          loc="lower left", fontsize=8.5)
ax.set_title("Perfil de amenaza proyectado 2027 por departamento (ensamble de 5 semillas)")
fig.tight_layout(); guardar(fig, "fig22_mapa_proyeccion_2027"); plt.show()
print(proy["Categoría 2027"].value_counts().to_string())
print(f"Departamentos donde el modelo estructural contradice el perfil observado en 2026: "
      f"{', '.join(proy.loc[proy['¿Coincide con la inercia?'] == 'revisar', 'Departamento'])}")

# ======== Celda de código 44 (índice 89): 13.1 Experimentos E1 (memoria del perfil, x8 = s_C del año anterior) y E2 (ensamble de semillas) ========
# =============================================================================
# 13.1 Experimentos E1 (memoria del perfil, x8 = s_C del año anterior) y E2 (ensamble de semillas)
# =============================================================================
datos["x8_memoria_C"] = datos["s_C_prev"].fillna(prev_train)
ETIQ["x8_memoria_C"] = "x₈ memoria s_C(t-1)"
t0 = time.time()
SEMILLAS_E2 = (SEED, 1, 2, 3, 4)
VT_E1, OOS_T_E1 = validacion("temporal", columnas=FEATURES + ["x8_memoria_C"])
VE_E1, OOS_E_E1 = validacion("espacial", columnas=FEATURES + ["x8_memoria_C"])
_, OOS_T_E2 = validacion("temporal", semillas=SEMILLAS_E2)
_, OOS_E_E2 = validacion("espacial", semillas=SEMILLAS_E2)
_, OOS_T_E12 = validacion("temporal", columnas=FEATURES + ["x8_memoria_C"], semillas=SEMILLAS_E2)
_, OOS_E_E12 = validacion("espacial", columnas=FEATURES + ["x8_memoria_C"], semillas=SEMILLAS_E2)
print(f"Experimentos E1 y E2 en {time.time() - t0:.0f} s")
filas = []
for nombre_v, base_oos, e1_oos, e2_oos, e12_oos in [("Temporal", OOS_T, OOS_T_E1, OOS_T_E2, OOS_T_E12),
                                                   ("Espacial", OOS_E, OOS_E_E1, OOS_E_E2, OOS_E_E12)]:
    for etiqueta, oos, mod in [("Persistencia", base_oos, "Persistencia"), ("Logística base (x1-x7)", base_oos, "Logística"),
                               ("MLP base (x1-x7)", base_oos, "MLP"), ("MLP base, ensamble 5 semillas (E2)", e2_oos, "MLP"),
                               ("Logística + x8", e1_oos, "Logística"), ("MLP + x8 (E1)", e1_oos, "MLP"),
                               ("MLP + x8, ensamble 5 semillas (E1+E2)", e12_oos, "MLP")]:
        filas.append({"Validación": nombre_v, "Modelo": etiqueta, **{k: v for k, v in metricas(oos["y"], oos[mod]).items() if k in cols_g}})
E1 = pd.DataFrame(filas).set_index(["Validación", "Modelo"])
display(E1.style.format(precision=3, decimal=",").background_gradient(subset=["ROC-AUC", "MCC"], cmap=CMAP_P, vmin=0.4, vmax=1)
        .background_gradient(subset=["Log-loss"], cmap=CMAP_P.reversed(), vmin=0.2, vmax=0.8))

# ======== Celda de código 45 (índice 91): 13.2 Efecto de la memoria en la proyección 2027 (ensamble de 5 semillas, 2021-2026) ========
# =============================================================================
# 13.2 Efecto de la memoria en la proyección 2027 (ensamble de 5 semillas, 2021-2026)
# =============================================================================
cols_e1 = FEATURES + ["x8_memoria_C"]
esc_e1 = StandardScaler().fit(datos[cols_e1])
ens_e1 = [entrenar_fijo(esc_e1.transform(datos[cols_e1]).astype("float32"), y_all, semilla=s) for s in SEMILLAS_DESPLIEGUE]
F27 = calcular_features(base27)
F27["x8_memoria_C"] = base27["s_C"].fillna(prev_train).values
P27_e1 = np.mean([f(esc_e1.transform(F27[cols_e1]).astype("float32")) for f in ens_e1], axis=0)
comp27 = (proy[["depto", "Departamento", "Perfil 2026 observado", "P(C) 2027", "¿Coincide con la inercia?", "Datos suficientes"]]
          .merge(pd.DataFrame({"depto": base27["depto"].values, "P(C) 2027 con memoria (E1)": P27_e1}), on="depto")
          .rename(columns={"P(C) 2027": "P(C) 2027 base"}))
revisar = comp27[comp27["¿Coincide con la inercia?"] == "revisar"]
resueltos = int(((revisar["P(C) 2027 con memoria (E1)"] >= .5) == (revisar["Perfil 2026 observado"] == "C")).sum())
print(f"Discrepancias base vs. inercia: {len(revisar)} | resueltas al agregar la memoria: {resueltos}")
display(revisar.drop(columns=["depto", "¿Coincide con la inercia?"]).style.format(
    {"P(C) 2027 base": "{:.2f}", "P(C) 2027 con memoria (E1)": "{:.2f}"}).hide(axis="index"))

orden_c = comp27.sort_values("P(C) 2027 con memoria (E1)").reset_index(drop=True)
fig, ax = plt.subplots(figsize=(10, 9))
yy = np.arange(len(orden_c))
ax.hlines(yy, orden_c["P(C) 2027 base"], orden_c["P(C) 2027 con memoria (E1)"], color=EJE, lw=1.5)
ax.scatter(orden_c["P(C) 2027 base"], yy, s=40, color=COL["D"], edgecolor=FONDO, lw=1.5, zorder=3, label="Modelo base (x₁–x₇)")
ax.scatter(orden_c["P(C) 2027 con memoria (E1)"], yy, s=48, color=COL["mlp"], edgecolor=FONDO, lw=1.5, zorder=3,
           label="Base + memoria del perfil (E1)")
ax.scatter(np.full(len(orden_c), -0.06), yy, marker="s", s=36, clip_on=False, zorder=3,
           color=[COL["C"] if p == "C" else (COL["D"] if p == "D" else GRID) for p in orden_c["Perfil 2026 observado"]])
ax.axvline(0.5, color=TINTA, lw=1, ls="--")
ax.set_yticks(yy, orden_c["Departamento"], fontsize=8); ax.set_xlim(-0.08, 1.02)
ax.set_xlabel("P(perfil C) proyectada para 2027")
ax.set_title("Proyección 2027: modelo base vs. modelo con memoria del perfil")
ax.legend(handles=[*ax.get_legend_handles_labels()[0], Patch(color=COL["C"], label="Perfil 2026 observado: C"),
                   Patch(color=COL["D"], label="Perfil 2026 observado: D")], fontsize=8.5, loc="lower right")
fig.tight_layout(); guardar(fig, "fig23_proyeccion_con_memoria"); plt.show()

# ======== Celda de código 46 (índice 96): 16.1 Artefactos para el informe y para la siguiente iteración ========
# =============================================================================
# 16.1 Artefactos para el informe y para la siguiente iteración
# =============================================================================
datos.to_csv(DIR_SALIDA / "dataset_modelado_departamento_anio.csv", index=False)
MET.to_csv(DIR_SALIDA / "metricas_modelos.csv")
IC_TEST.to_csv(DIR_SALIDA / "intervalos_confianza_prueba.csv")
res_busq.to_csv(DIR_SALIDA / "busqueda_hiperparametros.csv", index=False)
ESTAB.to_csv(DIR_SALIDA / "estabilidad_semillas.csv")
pd.concat({"temporal": VT, "espacial": VE}).to_csv(DIR_SALIDA / "validacion_temporal_espacial.csv")
E1.to_csv(DIR_SALIDA / "experimento_E1_memoria.csv")
proy.drop(columns=["lat", "lon"]).merge(comp27[["depto", "P(C) 2027 con memoria (E1)"]], on="depto").to_csv(
    DIR_SALIDA / "proyeccion_2027.csv", index=False)
pd.DataFrame({k: v for k, v in HIST.items() if k != "ratio_act" and len(v) == len(HIST["epoca"])}).to_csv(
    DIR_SALIDA / "historial_entrenamiento.csv", index=False)
modelo.save(DIR_SALIDA / "modelo_base_fase1.keras")
modelo_f2.save(DIR_SALIDA / "modelo_reajustado_fase2.keras")
with open(DIR_SALIDA / "configuracion.json", "w", encoding="utf-8") as fh:
    json.dump({"config_mlp": {**asdict(CFG), "capas": list(CFG.capas)}, "C_logistica": C_LR, "mejor_epoca": INFO["mejor_epoca"],
               "umbral_F2_validacion": U_OPT, "min_eventos": MIN_EVENTOS, "features": FEATURES, "semilla": SEED,
               "particion": {"train": ANIOS_TRAIN, "val": ANIO_VAL, "test": ANIO_TEST}}, fh, ensure_ascii=False, indent=2)

archivos = sorted(p for p in DIR_SALIDA.rglob("*") if p.is_file())
display(pd.DataFrame({"Archivo": [str(p.relative_to(DIR_SALIDA)) for p in archivos],
                      "KB": [round(p.stat().st_size / 1024, 1) for p in archivos]}).style.hide(axis="index"))

# ======== Celda de código 47 (índice 99): Anexo B. Figuras del informe (carta, 6,6" de ancho útil, 300 dpi) ========
# =============================================================================
# Anexo B. Figuras del informe (carta, 6,6" de ancho útil, 300 dpi)
# =============================================================================
DIR_INF = DIR_SALIDA / "figuras_informe"
DIR_INF.mkdir(parents=True, exist_ok=True)
ESTILO_INFORME = {"font.size": 7.2, "axes.titlesize": 7.6, "axes.labelsize": 7, "xtick.labelsize": 6.5,
                  "ytick.labelsize": 6.5, "legend.fontsize": 6.2, "lines.linewidth": 1.25, "axes.linewidth": 0.6,
                  "grid.linewidth": 0.45, "lines.markersize": 3.2, "axes.titlepad": 4}
FIG_INFORME = []

def guardar_inf(fig, nombre):
    fig.savefig(DIR_INF / f"{nombre}.png", dpi=300, bbox_inches="tight", pad_inches=0.03)
    FIG_INFORME.append(nombre)
    plt.close(fig)

with mpl.rc_context(ESTILO_INFORME):
    # ---------- R1. Variable objetivo ----------
    sC_i = panel.pivot(index="depto", columns="anio", values="s_C")
    cf_i = panel.pivot(index="depto", columns="anio", values="confiable")
    pf_i = panel.pivot(index="depto", columns="anio", values="perfil")
    ord_i = sC_i.mean(axis=1).sort_values(ascending=False).index
    sC_i, cf_i, pf_i = sC_i.loc[ord_i], cf_i.loc[ord_i], pf_i.loc[ord_i]
    fig = plt.figure(figsize=(6.6, 3.75))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.45, 1], hspace=0.6, wspace=0.32)
    ax = fig.add_subplot(gs[:, 0])
    im = ax.imshow(sC_i.values, cmap=CMAP_C, vmin=0, vmax=1, aspect="auto")
    for i in range(sC_i.shape[0]):
        for j in range(sC_i.shape[1]):
            if not cf_i.values[i, j]:
                ax.add_patch(Rectangle((j - .5, i - .5), 1, 1, facecolor=FONDO, edgecolor=GRID, hatch="////", lw=0))
                continue
            ax.text(j, i, pf_i.values[i, j], ha="center", va="center", fontsize=4.8,
                    color="white" if sC_i.values[i, j] > .62 else TINTA)
    ax.set_xticks(range(sC_i.shape[1]), sC_i.columns)
    ax.set_yticks(range(len(ord_i)), [NOMBRE[d] for d in ord_i], fontsize=5.4)
    ax.tick_params(length=0); ax.grid(False)
    ax.set_title("(a) Proporción de eventos C ($s^C$) y perfil")
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    cb.ax.axhline(0.5, color=TINTA, lw=1); cb.outline.set_visible(False); cb.ax.tick_params(labelsize=5.5)
    ax.set_xlabel("Celdas rayadas: < 10 eventos (excluidas)", fontsize=5.8, color=TINTA2)
    ax = fig.add_subplot(gs[0, 1])
    cnt = etiquetables.groupby(["anio", "perfil"]).size().unstack(fill_value=0)[["C", "D"]]
    ax.bar(cnt.index, cnt["D"], color=COL["D"], width=0.6, label="D (mixto)")
    ax.bar(cnt.index, cnt["C"], bottom=cnt["D"], color=COL["C"], width=0.6, label="C (convencional)", edgecolor=FONDO, lw=1)
    for a in cnt.index:
        ax.text(a, cnt.loc[a].sum() + 0.5, f"{cnt.loc[a, 'C'] / cnt.loc[a].sum():.0%}", ha="center", fontsize=5.8, color=TINTA2)
    ax.set_ylim(0, 37); ax.set_title("(b) Departamentos por perfil (% C)"); ax.legend(ncol=2, loc="upper left", fontsize=5.6)
    ax.set_xticks(cnt.index, cnt.index, fontsize=6)
    ax = fig.add_subplot(gs[1, 1])
    ax.hist(etiquetables["s_C"], bins=np.linspace(0, 1, 21), color=COL["C"], edgecolor=FONDO, linewidth=0.8)
    ax.axvline(0.5, color=TINTA, lw=1, ls="--")
    ax.set_title("(c) Distribución de $s^C$ (20 % a ±10 pp del umbral)"); ax.set_xlabel("$s^C_{d,t}$")
    ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
    guardar_inf(fig, "R1_variable_objetivo")

    # ---------- R2. Arquitectura ----------
    fig, ax = plt.subplots(figsize=(3.0, 2.25)); ax.axis("off"); ax.grid(False)
    tam = [len(FEATURES), CFG.capas[0], 1]; xs = [0, 2.2, 4.0]
    pos = [[(x, y) for y in np.linspace(-(n - 1) / 2, (n - 1) / 2, n) * (0.42 if n > 8 else 0.62)] for x, n in zip(xs, tam)]
    for a_, b_ in zip(pos[:-1], pos[1:]):
        for (x1, y1) in a_:
            for (x2, y2) in b_:
                ax.plot([x1, x2], [y1, y2], color=GRID, lw=0.35, zorder=1)
    for capa, color in zip(pos, [COL["D"], COL["mlp"], COL["C"]]):
        for (x, y) in capa:
            ax.scatter(x, y, s=22, color=color, edgecolor=FONDO, lw=0.8, zorder=3)
    for (x, y), f in zip(pos[0], FEATURES):
        ax.text(x - 0.18, y, ETIQ[f], ha="right", va="center", fontsize=5.3, color=TINTA2)
    ax.text(xs[2] + 0.15, 0, "P(C | x$_{t-1}$)", ha="left", va="center", fontsize=6.2)
    tope = max(abs(y) for _, y in pos[1])
    for x, t in zip(xs, ["Entrada\n7 variables", f"Oculta\n{CFG.capas[0]} ReLU · He\nL2 = {CFG.l2:g}", "Salida\nsigmoide"]):
        ax.text(x, -tope - 0.45, t, ha="center", va="top", fontsize=5.6, color=TINTA2)
    ax.set_xlim(-2.2, 5.3); ax.set_ylim(-tope - 1.6, tope + 0.3)
    guardar_inf(fig, "R2_arquitectura")

    # ---------- R3. Curvas de aprendizaje ----------
    ep_i = np.array(HIST["epoca"]); be_i = INFO["mejor_epoca"]
    cortes_i = [e for e, a, b in zip(ep_i[1:], HIST["lr"][:-1], HIST["lr"][1:]) if b < a]
    fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.05))
    ax = axes[0]
    ax.plot(ep_i, HIST["perdida_obj"], color=COL["train"], lw=0.8, alpha=0.35, label="Objetivo (BCE pond. + L2)")
    ax.plot(ep_i, HIST["bce_tr"], color=COL["train"], label="Entrenamiento")
    ax.plot(ep_i, HIST["bce_va"], color=COL["val"], label="Validación 2025")
    ax.axhline(H0, color=TENUE, lw=0.8, ls="--", label="Predictor constante")
    ax.set_title("(a) Entropía cruzada (BCE)"); ax.set_xlabel("Época")
    ax.legend(loc="upper right", fontsize=5.4, frameon=True, facecolor=FONDO, edgecolor="none", framealpha=1)
    ax = axes[1]
    ax.plot(ep_i, HIST["acc_tr"], color=COL["train"], label="Entrenamiento"); ax.plot(ep_i, HIST["acc_va"], color=COL["val"], label="Validación 2025")
    ax.axhline(1 - prev_train, color=TENUE, lw=0.8, ls="--"); ax.text(ep_i[-1], 1 - prev_train + 0.006, "clase mayoritaria", ha="right", va="bottom", fontsize=5.4, color=TINTA2)
    ax.set_title("(b) Exactitud (umbral 0,5)"); ax.set_xlabel("Época"); ax.legend(loc="lower right", fontsize=5.6)
    ax = axes[2]
    ax.plot(ep_i, HIST["auc_tr"], color=COL["train"], label="Entrenamiento"); ax.plot(ep_i, HIST["auc_va"], color=COL["val"], label="Validación 2025")
    ax.set_title("(c) ROC-AUC"); ax.set_xlabel("Época"); ax.legend(loc="lower right", fontsize=5.6)
    for a_ in axes:
        a_.axvline(be_i, color=TINTA, lw=0.8, ls=":")
        for c in cortes_i:
            a_.axvline(c, color=TENUE, lw=0.5, ls="--")
    axes[0].text(be_i + 2, axes[0].get_ylim()[0] + 0.02, f"mejor época = {be_i}", fontsize=5.6)
    fig.tight_layout(w_pad=1.2); guardar_inf(fig, "R3_curvas_aprendizaje")

    # ---------- R4. Métricas en prueba: ROC, calibración y matriz de confusión ----------
    fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.1), gridspec_kw={"width_ratios": [1, 1, 0.8]})
    ax = axes[0]
    for nombre, color, ls in [("MLP fase 2", COL["mlp"], "-"), ("MLP fase 1", COL["mlp"], ":"), ("Logística fase 2", COL["logit"], "-"), ("Persistencia", COL["pers"], "--")]:
        p = P[(nombre, "Prueba")]; fpr, tpr, _ = roc_curve(yte, p)
        ax.plot(fpr, tpr, color=color, ls=ls, label=f"{nombre} ({roc_auc_score(yte, p):.2f})")
    ax.plot([0, 1], [0, 1], color=EJE, lw=0.7); ax.set_xlabel("1 − especificidad"); ax.set_ylabel("Sensibilidad C")
    ax.set_title("(a) ROC en prueba 2026 (AUC)"); ax.legend(fontsize=5.3, loc="lower right")
    ax = axes[1]; ax.plot([0, 1], [0, 1], color=EJE, lw=0.7)
    for nombre, color in [("MLP fase 1", COL["mlp"]), ("Logística fase 1", COL["logit"])]:
        p = np.concatenate([P[(nombre, "Validación")], P[(nombre, "Prueba")]])
        fr, mp = calibration_curve(y_fuera, p, n_bins=5, strategy="quantile")
        ax.plot(mp, fr, marker="o", color=color, mec=FONDO, mew=0.8, label=f"{nombre} (Brier {brier_score_loss(y_fuera, p):.3f})")
    ax.set_xlabel("P(C) media predicha"); ax.set_ylabel("Fracción observada de C")
    ax.set_title("(b) Calibración (validación + prueba)"); ax.legend(fontsize=5.3, loc="upper left")
    ax = axes[2]
    cm = confusion_matrix(yte.astype(int), (P[("MLP fase 2", "Prueba")] >= .5).astype(int), labels=[0, 1])
    ax.imshow(cm, cmap=CMAP_P, vmin=0, vmax=cm.max() * 1.3); ax.grid(False)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]}\n({cm[i, j] / cm[i].sum():.0%})", ha="center", va="center", fontsize=6.5,
                    color="white" if cm[i, j] > cm.max() * 0.65 else TINTA)
    ax.set_xticks([0, 1], ["D", "C"]); ax.set_yticks([0, 1], ["D", "C"]); ax.set_xlabel("Predicho"); ax.set_ylabel("Real")
    ax.set_title("(c) MLP fase 2 · prueba")
    fig.tight_layout(w_pad=1.2); guardar_inf(fig, "R4_metricas_prueba")

    # ---------- R5. Convergencia y estabilidad ----------
    rat_i = np.array(HIST["ratio_act"]); capas_i = [c.name for c in modelo.layers if hasattr(c, "kernel")]
    fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.0))
    ax = axes[0]
    ax.plot(ep_i, HIST["norma_grad"], color=COL["mlp"], lw=0.8, alpha=0.6)
    ax.plot(ep_i, pd.Series(HIST["norma_grad"]).rolling(15, min_periods=1).mean(), color=TINTA, lw=1.1, label="media móvil 15 ép.")
    ax.set_yscale("log"); ax.set_title("(a) Norma global del gradiente"); ax.set_xlabel("Época")
    ax.legend(fontsize=5.6, frameon=True, facecolor=FONDO, edgecolor="none", framealpha=1)
    ax = axes[1]
    for j, (nombre, color) in enumerate(zip(capas_i, [COL["mlp"], COL["logit"]])):
        ax.plot(ep_i, rat_i[:, j], color=color, lw=0.9, label=nombre)
    ax.axhspan(1e-4, 1e-2, color=GRID, alpha=0.5, lw=0); ax.set_yscale("log")
    ax.set_title("(b) Razón de actualización ‖ΔW‖/‖W‖"); ax.set_xlabel("Época")
    ax.legend(fontsize=5.6, loc="lower left", frameon=True, facecolor=FONDO, edgecolor="none", framealpha=1)
    for a_ in axes[:2]:
        a_.axvline(be_i, color=TINTA, lw=0.8, ls=":")
        for c in cortes_i:
            a_.axvline(c, color=TENUE, lw=0.5, ls="--")
    ax = axes[2]
    for c in curvas_va:
        ax.plot(range(1, len(c) + 1), c, color=COL["mlp"], lw=0.6, alpha=0.5)
        k = int(np.argmin(c)); ax.scatter(k + 1, c[k], s=9, color=COL["val"], zorder=3)
    ax.set_ylim(0.3, 1.0); ax.set_title("(c) BCE de validación, 10 semillas"); ax.set_xlabel("Época")
    fig.tight_layout(w_pad=1.2); guardar_inf(fig, "R5_convergencia_estabilidad")

    # ---------- R6. Generalización y sesgo–varianza ----------
    fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.1))
    ax = axes[0]
    for mod, color, ls in [("MLP", COL["mlp"], "-"), ("Logística", COL["logit"], "-"), ("Persistencia", COL["pers"], "--")]:
        s = VT[VT.Modelo == mod]
        ax.plot(s["Partición"], s["ROC-AUC"], color=color, ls=ls, marker="o", mec=FONDO, mew=0.8, label=mod)
    ax.set_title("(a) AUC por año predicho (temporal)"); ax.set_xlabel("Año (entrenando con los anteriores)"); ax.legend(fontsize=5.6, loc="lower left")
    ax = axes[1]; nn_i = CURVA_N.index.values
    for col, color, ls, et in [("MLP prueba", COL["mlp"], "-", "MLP prueba"), ("MLP train", COL["mlp"], ":", "MLP entren."),
                               ("Logística prueba", COL["logit"], "-", "Log. prueba"), ("Logística train", COL["logit"], ":", "Log. entren.")]:
        ax.plot(nn_i, CURVA_N[(col, "mean")].values, color=color, ls=ls, marker="o", label=et)
    ax.set_ylim(0.25, 0.92); ax.set_title("(b) BCE vs. tamaño de muestra"); ax.set_xlabel("Filas de entrenamiento")
    ax.legend(fontsize=5.2, loc="upper right", ncol=2, columnspacing=0.8, handlelength=1.8, frameon=True, facecolor=FONDO, edgecolor="none", framealpha=1)
    ax = axes[2]
    for regimen, color, et in [("Sin regularización", COL["logit"], "Sin regularización"), ("Con L2 del modelo base", COL["mlp"], "Con L2 = 10⁻³")]:
        s = CAPACIDAD[CAPACIDAD["régimen"] == regimen]
        ax.plot(s["unidades"], s["BCE train"], color=color, ls=":", marker="o", ms=2.5)
        ax.plot(s["unidades"], s["BCE validación"], color=color, marker="o", label=et)
    ax.axvline(CFG.capas[0], color=TINTA, lw=0.8, ls=":"); ax.set_xscale("log", base=2); ax.set_xticks(UNIDADES, UNIDADES)
    ax.set_title("(c) Capacidad, 400 épocas fijas"); ax.set_xlabel("Neuronas en la capa oculta"); ax.set_ylim(0.05, 0.98)
    manejadores = ax.get_legend_handles_labels()[0] + [Line2D([], [], color=TENUE, ls="-", label="validación"),
                                                        Line2D([], [], color=TENUE, ls=":", label="entrenamiento")]
    ax.legend(handles=manejadores, fontsize=5.2, loc="upper left", ncol=2, columnspacing=0.8, handlelength=1.8,
              frameon=True, facecolor=FONDO, edgecolor="none", framealpha=1)
    fig.tight_layout(w_pad=1.2); guardar_inf(fig, "R6_generalizacion")

    # ---------- R7. Superficies de dependencia parcial ----------
    def pdp_completo(pred_fn, fx, fy, n=35):
        base = datos.loc[m_tr, FEATURES].reset_index(drop=True)
        gx = np.linspace(datos[fx].min(), datos[fx].max(), n); gy = np.linspace(datos[fy].min(), datos[fy].max(), n)
        GX, GY = np.meshgrid(gx, gy)
        rep = pd.concat([base] * (n * n), ignore_index=True)
        rep[fx] = np.repeat(GX.ravel(), len(base)); rep[fy] = np.repeat(GY.ravel(), len(base))
        Z = pred_fn(escalador.transform(rep[FEATURES]).astype("float32")).reshape(n * n, len(base)).mean(axis=1).reshape(n, n)
        return gx, gy, Z
    fig, axes = plt.subplots(1, 4, figsize=(6.6, 1.85))
    for k, ((fx, fy), (nombre, fn)) in enumerate([(("x5_bosque", "x6_distancia"), ("MLP", predecir)), (("x5_bosque", "x6_distancia"), ("Logística", pred_lr_fn)),
                                                (("x4_coca", "x7_ipm"), ("MLP", predecir)), (("x4_coca", "x7_ipm"), ("Logística", pred_lr_fn))]):
        ax = axes[k]; gx, gy, Z = pdp_completo(fn, fx, fy)
        cs = ax.contourf(gx, gy, Z, levels=np.linspace(0, 1, 11), cmap=CMAP_P, vmin=0, vmax=1)
        ax.contour(gx, gy, Z, levels=[0.5], colors=TINTA, linewidths=1)
        for clase, color in [(0, COL["D"]), (1, COL["C"])]:
            s = datos[m_tr & (datos.y == clase).values]
            ax.scatter(s[fx], s[fy], s=6, color=color, edgecolor=TINTA, lw=0.25, zorder=3)
        ax.grid(False); ax.set_xlabel(ETIQ[fx], fontsize=6); ax.set_ylabel(ETIQ[fy], fontsize=6)
        ax.set_title(f"({'abcd'[k]}) {nombre}", fontsize=7)
    fig.subplots_adjust(wspace=0.6, right=0.9)
    cax = fig.add_axes([0.925, 0.18, 0.012, 0.68])
    cb = fig.colorbar(cs, cax=cax); cb.set_label("P(C)", fontsize=6); cb.outline.set_visible(False); cb.ax.tick_params(labelsize=5.5)
    guardar_inf(fig, "R7_dependencia_parcial")

    # ---------- R8. Mapa esquemático de la proyección 2027 ----------
    fig, ax = plt.subplots(figsize=(3.25, 3.75))
    tam_i = 8 + 2.2 * np.sqrt(proy["Eventos 2026 (anualizados)"].clip(lower=1))
    sc_i = ax.scatter(proy["lon"], proy["lat"], s=tam_i, c=proy["P(C) 2027"], cmap=CMAP_P, vmin=0, vmax=1,
                      edgecolor=FONDO, lw=0.8, zorder=3)
    al_i = proy[proy["alerta_AB"].fillna(False)]
    ax.scatter(al_i["lon"], al_i["lat"], s=tam_i[al_i.index] + 70, facecolor="none", edgecolor="#d03b3b", lw=1.1, zorder=2)
    izq_i = {"valle del cauca", "risaralda", "quindio", "sucre", "cordoba", "choco", "narino", "cauca"}
    for _, r in proy.iterrows():
        izq = r["depto"] in izq_i
        dx = 8 if bool(r["alerta_AB"]) else 4
        ax.annotate(f"{r['Departamento']} {r['P(C) 2027']:.2f}".replace(".", ","), (r["lon"], r["lat"]),
                    xytext=(-dx if izq else dx, -1.5), textcoords="offset points", ha="right" if izq else "left",
                    fontsize=4.6, color=TINTA if r["Datos suficientes"] else TENUE)
    ax.set_aspect("equal"); ax.set_xlim(-82.2, -66.2); ax.set_ylim(-4.6, 14.2)
    ax.tick_params(labelsize=5); ax.set_xlabel("Longitud", fontsize=5.5); ax.set_ylabel("Latitud", fontsize=5.5)
    cb = fig.colorbar(sc_i, ax=ax, fraction=0.035, pad=0.02); cb.set_label("P(perfil C) 2027", fontsize=5.5)
    cb.outline.set_visible(False); cb.ax.tick_params(labelsize=5)
    ax.legend(handles=[Line2D([], [], marker="o", ls="", mfc="none", mec="#d03b3b", mew=1.1, ms=6, label="Alerta A/B (IED/drones)"),
                       Line2D([], [], marker="o", ls="", color=TENUE, ms=3, label="Rótulo gris: pocos datos")],
              loc="lower left", fontsize=4.6, frameon=True, facecolor=FONDO, edgecolor="none", framealpha=1)
    guardar_inf(fig, "R8_proyeccion_2027")

print(f"Figuras del informe guardadas en {DIR_INF}/:", ", ".join(FIG_INFORME))

# ======== Celda de código 48 (índice 101):  ========
print(f"Ejecución completa. Figuras exportadas: {len(FIGURAS)} en {DIR_FIG}/")
