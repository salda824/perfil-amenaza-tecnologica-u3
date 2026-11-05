"""Estilo gráfico común (paleta de la U1/U2) y utilidades de presentación."""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from IPython.display import display

from . import config as C

COL = {"C": "#eda100", "D": "#888780", "A": "#e34948", "B": "#2a78d6",
       "mlp": "#2a78d6", "u3": "#1baf7a", "logit": "#eb6834", "pers": "#52514e", "mayor": "#b9b8b2",
       "train": "#2a78d6", "val": "#eb6834", "test": "#1baf7a", "rechazo": "#c3c2b7"}
TINTA, TINTA2, TENUE, GRID, EJE, FONDO = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
CMAP_C = LinearSegmentedColormap.from_list("prop_C", ["#fdf6e6", "#f6d58a", "#eda100", "#a86f00", "#4d3300"])
CMAP_P = LinearSegmentedColormap.from_list("prob_C", ["#eef4fc", "#9ec5f4", "#3987e5", "#1c5cab", "#0d366b"])
CMAP_DIV = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#f0efec", "#e34948"])

FIGURAS: list[str] = []


def aplicar_estilo():
    mpl.rcParams.update({
        "figure.facecolor": FONDO, "axes.facecolor": FONDO, "savefig.facecolor": FONDO,
        "font.family": "sans-serif", "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial", "Liberation Sans"],
        "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "semibold", "axes.titlelocation": "left",
        "axes.labelcolor": TINTA2, "axes.edgecolor": EJE, "axes.linewidth": 0.8, "axes.axisbelow": True,
        "xtick.color": TENUE, "ytick.color": TENUE, "xtick.labelcolor": TINTA2, "ytick.labelcolor": TINTA2,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.7, "axes.spines.top": False,
        "axes.spines.right": False, "legend.frameon": False, "lines.linewidth": 2, "figure.dpi": 110,
    })


def guardar(fig, nombre, carpeta=None, dpi=200):
    carpeta = carpeta or C.DIR_FIG
    carpeta.mkdir(parents=True, exist_ok=True)
    ruta = carpeta / f"{nombre}.png"
    fig.savefig(ruta, dpi=dpi, bbox_inches="tight")
    if ruta.name not in FIGURAS:
        FIGURAS.append(ruta.name)
    return ruta


def _fmt(dec):
    def f(v):
        try:
            if v is None or (isinstance(v, float) and v != v):
                return "–"
            if isinstance(v, (bool,)) or not isinstance(v, (int, float)) and not hasattr(v, "dtype"):
                return str(v)
            v = float(v)
            if v.is_integer() and abs(v) >= 1:
                return f"{v:,.0f}".replace(",", ".")
            return f"{v:.{dec}f}".replace(".", ",")
        except (TypeError, ValueError):
            return str(v)
    return f


def tabla(df, dec=3, **kw):
    """Tabla con coma decimal; los enteros se muestran sin decimales."""
    return display(df.style.format(_fmt(dec), na_rep="–", **kw))
