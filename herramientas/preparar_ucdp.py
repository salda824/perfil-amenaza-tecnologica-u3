"""
Reduce UCDP GED 26.1 (39 MB) a lo que usa el proyecto. Se ejecuta en el computador donde se descargó el archivo:

    python herramientas/preparar_ucdp.py ruta/a/ged261-csv.zip      (o el .csv, o la carpeta descomprimida)

Escribe dos archivos en datos/externos/:
  ucdp_colombia.csv                     solo eventos de Colombia y las 7 columnas necesarias (~100-200 KB)
  ucdp_actores_departamento_anio.csv    x3 ya agregado por departamento y año (~5 KB)

Basta con subir cualquiera de los dos; el pipeline los detecta. No necesita TensorFlow.
"""
import sys
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
from perfil_amenaza import datos as D  # noqa: E402

if len(sys.argv) != 2:
    sys.exit(__doc__)
origen = Path(sys.argv[1].strip('"'))
if not origen.exists() and Path(str(origen) + ".zip").exists():      # 'ged261-csv' sin la extensión .zip
    origen = Path(str(origen) + ".zip")
if origen.is_dir():                                                    # carpeta descomprimida
    candidatos = sorted(origen.rglob("GEDEvent*.csv")) or sorted(origen.rglob("*.csv")) or sorted(origen.rglob("*.zip"))
    if not candidatos:
        sys.exit(f"No encontré el CSV de UCDP dentro de {origen}")
    origen = candidatos[0]
if not origen.exists():
    sys.exit(f"No existe {origen}")
print(f"Leyendo {origen} ...")
g = pd.read_csv(origen, low_memory=False, usecols=lambda c: c in D.COLUMNAS_UCDP)
col = g[g["country"] == "Colombia"]
destino = D.C.DIR_EXT
destino.mkdir(parents=True, exist_ok=True)
col.to_csv(destino / "ucdp_colombia.csv", index=False)
cache = destino / "ucdp_actores_departamento_anio.csv"
cache.unlink(missing_ok=True)
actores = D.cargar_ucdp()
print(f"Colombia: {len(col):,} eventos ({col['year'].min()}-{col['year'].max()}) -> {destino / 'ucdp_colombia.csv'}")
print(f"x3 agregado: {len(actores)} filas departamento-año -> {cache}")
print(actores.groupby("anio")["actores"].describe()[["mean", "max"]].round(1))
