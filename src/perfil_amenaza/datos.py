"""Carga y auditoría de ACLED, panel departamento-año, etiqueta y fuentes externas."""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C

COLUMNAS_ACLED = ["week", "region", "country", "admin1", "event_type", "sub_event_type", "events",
                  "fatalities", "population_exposure", "disorder_type", "centroid_latitude", "centroid_longitude"]


# ----------------------------------------------------------------------------------------------------------
# Utilidades
# ----------------------------------------------------------------------------------------------------------
def normalizar(nombre) -> str:
    """Llave de cruce entre fuentes: minúsculas, sin tildes ni signos ('Bogotá, D.C.' -> 'bogota')."""
    s = unicodedata.normalize("NFKD", str(nombre)).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"\b(department|departamento|district|distrito capital|province)\b", " ", s)
    s = re.sub(r"\s+", " ", re.sub(r"[^a-z ]", " ", s)).strip()
    if s.startswith("bogota"):
        return "bogota"
    if s.startswith("san andres") or s.startswith("archipielago"):
        return "san andres"
    if s.startswith("valle"):
        return "valle del cauca"
    if s in ("guajira",):
        return "la guajira"
    return s


def haversine_km(lat1, lon1, lat2, lon2):
    la1, lo1, la2, lo2 = map(np.radians, (lat1, lon1, lat2, lon2))
    h = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(h))


# ----------------------------------------------------------------------------------------------------------
# ACLED
# ----------------------------------------------------------------------------------------------------------
def cargar_acled(ruta: Path | str = C.RUTA_ACLED) -> pd.DataFrame:
    """Lee el export semanal y valida el contrato de datos (columnas y tipos)."""
    bruto = pd.read_csv(ruta, encoding="utf-8-sig")
    faltan = set(COLUMNAS_ACLED) - set(bruto.columns)
    if faltan:
        raise ValueError(f"El export de ACLED no cumple el contrato: faltan columnas {sorted(faltan)}")
    bruto["week"] = pd.to_datetime(bruto["week"])
    return bruto


def filtrar_horizonte(bruto: pd.DataFrame) -> pd.DataFrame:
    """Colombia, semanas desde 2020; llave de departamento normalizada y semana del año (1..53)."""
    d = bruto[(bruto["country"] == "Colombia") & (bruto["week"] >= f"{C.ANIO_INICIO}-01-01")].copy()
    d["anio"] = d["week"].dt.year
    d["depto"] = d["admin1"].map(normalizar)
    semanas = d[["anio", "week"]].drop_duplicates().sort_values("week")
    semanas["semana_anio"] = semanas.groupby("anio").cumcount() + 1
    return d.merge(semanas, on=["anio", "week"], how="left")


def auditar(bruto: pd.DataFrame) -> pd.DataFrame:
    """Auditoría de calidad: cada verificación con su resultado."""
    semanas = np.sort(bruto["week"].unique())
    saltos = np.diff(semanas).astype("timedelta64[D]").astype(int)
    col = bruto[bruto["country"] == "Colombia"]
    filas = [
        ("Registros / columnas", f"{len(bruto):,} / {bruto.shape[1]}"),
        ("Registros fuera de Colombia", f"{(bruto['country'] != 'Colombia').sum()}"),
        ("Cobertura temporal", f"{bruto['week'].min():%Y-%m-%d} → {bruto['week'].max():%Y-%m-%d}"),
        ("Semanas distintas / saltos ≠ 7 días", f"{len(semanas)} / {(saltos != 7).sum()}"),
        ("Valores faltantes", f"{int(bruto.isna().sum().sum())}"),
        ("Duplicados por llave semana–depto–subtipo",
         f"{int(bruto.duplicated(['week', 'admin1', 'sub_event_type']).sum())}"),
        ("Rangos válidos (events ≥ 1, fatalities ≥ 0)",
         f"{bool((bruto['events'] >= 1).all() and (bruto['fatalities'] >= 0).all())}"),
        ("Departamentos de Colombia", f"{col['admin1'].nunique()}"),
    ]
    return pd.DataFrame(filas, columns=["Verificación", "Resultado"])


def construir_panel(acled: pd.DataFrame, mapa: dict | None = None, semanas_max: int | None = None) -> pd.DataFrame:
    """
    Agrega a departamento-año: conteos por perfil (A-D), eventos, fatalidades, proporciones y etiqueta.
    `semanas_max` recorta cada año a sus primeras N semanas (control de la etiqueta parcial de 2026).
    """
    mapa = mapa or C.MAPA_PERFIL_U1
    d = acled if semanas_max is None else acled[acled["semana_anio"] <= semanas_max]
    d = d.assign(perfil=d["sub_event_type"].map(mapa).fillna("D"))
    semanas = d.groupby("anio")["week"].nunique()
    conteo = d.pivot_table(index=["depto", "anio"], columns="perfil", values="events", aggfunc="sum", fill_value=0)
    conteo = conteo.reindex(columns=list("ABCD"), fill_value=0)
    base = pd.MultiIndex.from_product([sorted(acled["depto"].unique()), sorted(acled["anio"].unique())],
                                      names=["depto", "anio"])
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
    panel["confiable"] = panel["eventos"] >= C.MIN_EVENTOS
    return panel


def bloques_trimestrales(acled: pd.DataFrame, mapa: dict | None = None, n_bloques: int = 4,
                         semanas_bloque: int = 13) -> pd.DataFrame:
    """
    Para cada departamento y año de corte `a`, agrega los eventos en bloques de 13 semanas contados hacia atrás
    desde la última semana disponible de `a` (bloque 0 = el más reciente). Con años completos los cuatro bloques
    son los trimestres de `a`; con el año parcial (2026) la ventana se extiende al año anterior, de modo que cada
    bloque siempre resume 13 semanas reales de información disponible al momento de decidir.
    """
    mapa = mapa or C.MAPA_PERFIL_U1
    d = acled.assign(perfil=acled["sub_event_type"].map(mapa).fillna("D"))
    sem = (d.pivot_table(index=["depto", "week"], columns="perfil", values="events", aggfunc="sum", fill_value=0)
             .reindex(columns=list("ABCD"), fill_value=0))
    sem["fat"] = d.groupby(["depto", "week"])["fatalities"].sum()
    sem = sem.reset_index()
    todas = np.sort(d["week"].unique())
    deptos = sorted(d["depto"].unique())
    filas = []
    for a in sorted(d["anio"].unique()):
        fin = d.loc[d["anio"] == a, "week"].max()
        pos = int(np.searchsorted(todas, np.datetime64(fin), side="right"))
        for b in range(n_bloques):
            hi, lo = pos - b * semanas_bloque, max(0, pos - (b + 1) * semanas_bloque)
            ventana = todas[lo:hi]
            s = sem[sem["week"].isin(ventana)].groupby("depto")[["A", "B", "C", "D", "fat"]].sum().reindex(deptos, fill_value=0)
            s = s.assign(anio=a, bloque=b, semanas=len(ventana)).reset_index()
            filas.append(s)
    out = pd.concat(filas, ignore_index=True)
    out["eventos"] = out[list("ABCD")].sum(axis=1)
    return out


# ----------------------------------------------------------------------------------------------------------
# Fuentes externas (caché local; descarga desde la fuente oficial solo si falta)
# ----------------------------------------------------------------------------------------------------------
URL_COCA = "https://www.datos.gov.co/resource/acs4-3wgp.csv?$limit=5000"
URL_GFW = "https://gfw2-data.s3.amazonaws.com/country-pages/country_stats/download/2025/COL.xlsx"
URL_UCDP = "https://ucdp.uu.se/downloads/ged/ged261-csv.zip"
CABECERA = {"User-Agent": "Mozilla/5.0 (proyecto academico Universidad de La Sabana)"}


def _descargar_coca() -> pd.DataFrame:
    m = pd.read_csv(URL_COCA, dtype=str, storage_options=CABECERA)
    anios = [c for c in m.columns if re.fullmatch(r"_\d{4}", c)]
    m[anios] = m[anios].apply(pd.to_numeric, errors="coerce").fillna(0)
    m["depto"] = m["departamento"].map(normalizar)
    d = m.groupby("depto")[anios].sum()
    d.columns = [int(c[1:]) for c in d.columns]
    return d.stack().rename("coca_ha").rename_axis(["depto", "anio"]).reset_index()


def _descargar_bosque() -> pd.DataFrame:
    g = pd.read_excel(URL_GFW, sheet_name="Subnational 1 tree cover loss", storage_options=CABECERA)
    g = g[g["threshold"] == 30].copy()
    g["depto"] = g["subnational1"].map(normalizar)
    filas = []
    for _, r in g.iterrows():
        acumulada = 0.0
        for anio in range(2011, 2026):
            acumulada += r[f"tc_loss_ha_{anio}"]
            filas.append((r["depto"], anio, 100 * (r["extent_2010_ha"] - acumulada) / r["area_ha"],
                          r[f"tc_loss_ha_{anio}"], r["area_ha"]))
    return pd.DataFrame(filas, columns=["depto", "anio", "bosque_pct", "perdida_ha", "area_ha"])


def cargar_o_descargar(ruta: Path, funcion) -> pd.DataFrame:
    if ruta.exists():
        return pd.read_csv(ruta)
    df = funcion()
    ruta.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(ruta, index=False)
    return df


def fuentes_externas() -> tuple[pd.DataFrame, pd.DataFrame]:
    coca = cargar_o_descargar(C.DIR_EXT / "coca_departamento_anio.csv", _descargar_coca)
    bosque = cargar_o_descargar(C.DIR_EXT / "bosque_departamento_anio.csv", _descargar_bosque)
    return coca, bosque


def estructura_estatica(acled: pd.DataFrame) -> pd.DataFrame:
    """Centroides, distancia a Bogotá (x6, proxy) e IPM 2018 (x7)."""
    e = acled.groupby("depto")[["centroid_latitude", "centroid_longitude"]].first()
    e.columns = ["lat", "lon"]
    e["dist_km"] = haversine_km(e["lat"], e["lon"], *C.BOGOTA)
    e["ipm"] = e.index.map(C.IPM_2018)
    return e.reset_index()


def vecindad() -> dict[str, list[str]]:
    """Lista de contigüidad simetrizada (si A es vecino de B, B es vecino de A)."""
    v = {k: set(vs) for k, vs in C.VECINOS.items()}
    for k, vs in list(v.items()):
        for j in vs:
            v.setdefault(j, set()).add(k)
    return {k: sorted(vs) for k, vs in v.items()}


# ----------------------------------------------------------------------------------------------------------
# UCDP GED (x3: actores armados activos). Opcional: solo si el archivo está disponible.
# ----------------------------------------------------------------------------------------------------------
RUTAS_UCDP = [C.DIR_EXT / "ucdp_colombia.csv", C.DIR_EXT / "GEDEvent_v26_1.csv", C.DIR_EXT / "ged261-csv.zip"]
COLUMNAS_UCDP = ["country", "year", "adm_1", "side_a", "side_b", "best", "type_of_violence"]


def _separar_actores(s: str) -> list[str]:
    """'ELN, FARC-EMC' -> ['ELN', 'FARC-EMC']; se excluyen gobiernos y civiles."""
    partes = [p.strip() for p in re.split(r",| and ", str(s)) if p.strip()]
    return [p for p in partes if not (p.lower().startswith("government of") or p.lower() == "civilians")]


def cargar_ucdp() -> pd.DataFrame | None:
    """
    Devuelve actores armados no estatales distintos por departamento-año (UCDP GED 26.1, Colombia) o None si el
    archivo no está en datos/externos/. Solo cubre violencia organizada letal: es un indicador aproximado de x3.
    """
    cache = C.DIR_EXT / "ucdp_actores_departamento_anio.csv"
    if cache.exists():
        return pd.read_csv(cache)
    origen = next((r for r in RUTAS_UCDP if r.exists()), None)
    if origen is None:
        return None
    g = pd.read_csv(origen, low_memory=False, usecols=lambda c: c in COLUMNAS_UCDP)
    g = g[(g["country"] == "Colombia") & g["adm_1"].notna()].copy()
    g["depto"] = g["adm_1"].map(normalizar)
    g = g[g["depto"].isin(C.NOMBRE)]
    filas = []
    for (depto, anio), grupo in g.groupby(["depto", "year"]):
        actores = set()
        for col in ("side_a", "side_b"):
            for s in grupo[col].dropna():
                actores.update(_separar_actores(s))
        filas.append({"depto": depto, "anio": int(anio), "actores": len(actores), "eventos_ucdp": len(grupo),
                      "muertes_ucdp": int(grupo.get("best", pd.Series(0)).sum())})
    out = pd.DataFrame(filas)
    out.to_csv(cache, index=False)
    return out


def control_etiqueta_parcial(acled: pd.DataFrame, anios=range(2021, 2026)) -> pd.DataFrame:
    """
    ¿Coincide la etiqueta calculada con las primeras N semanas (N = semanas disponibles de 2026) con la del año
    completo? Se mide en los años completos 2021-2025, con celdas confiables en ambos cortes.
    """
    n_2026 = int(acled.loc[acled["anio"] == C.ANIO_PRUEBA, "semana_anio"].max())
    completo = construir_panel(acled)
    parcial = construir_panel(acled, semanas_max=n_2026)
    m = completo.merge(parcial[["depto", "anio", "perfil", "confiable", "s_C"]], on=["depto", "anio"],
                       suffixes=("", "_parcial"))
    m = m[m["anio"].isin(list(anios)) & m["confiable"] & m["confiable_parcial"]]
    m["coincide"] = m["perfil"] == m["perfil_parcial"]
    return m, n_2026
