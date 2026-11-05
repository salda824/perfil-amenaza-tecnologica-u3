"""
Convierte los límites departamentales del DANE (Marco Geoestadístico Nacional 2018, GeoJSON) en trazos SVG livianos
para el mapa interactivo de las presentaciones. Se ejecuta una sola vez; el resultado (colombia_dptos.json) queda en
el repositorio, así que las presentaciones no dependen de internet.

    python presentaciones/fuentes/preparar_mapa.py ruta/a/co_2018_MGN_DPTO_POLITICO.geojson

Fuente del GeoJSON: DANE, MGN 2018 (copia pública en github.com/caticoa3/colombia_mapa).
Proyección: equirectangular con corrección cos(4°) (Colombia está cerca del ecuador). San Andrés va en un recuadro.
"""
import json
import math
import sys
import unicodedata
from pathlib import Path

K = 62.0                       # píxeles por grado
LAT0 = math.radians(4.0)
LON_MIN, LAT_MAX = -79.2, 12.6
TOL = 0.012                    # tolerancia de Douglas-Peucker en grados
INSET = {"x": 18, "y": 18, "w": 120, "h": 150}   # recuadro de San Andrés (px)


def clave(nombre):
    n = unicodedata.normalize("NFKD", nombre).encode("ascii", "ignore").decode().lower()
    if n.startswith("archipielago"):
        return "san andres"
    if n.startswith("bogota"):
        return "bogota"
    return n


def dp(puntos, tol):
    """Douglas-Peucker iterativo sobre una lista de (lon, lat)."""
    if len(puntos) < 4:
        return puntos
    mantener = [False] * len(puntos); mantener[0] = mantener[-1] = True
    pila = [(0, len(puntos) - 1)]
    while pila:
        a, b = pila.pop()
        (x1, y1), (x2, y2) = puntos[a], puntos[b]
        dx, dy = x2 - x1, y2 - y1; L = math.hypot(dx, dy) or 1e-12
        idx, dmax = -1, 0.0
        for i in range(a + 1, b):
            x0, y0 = puntos[i]
            d = abs(dy * x0 - dx * y0 + x2 * y1 - y2 * x1) / L
            if d > dmax:
                idx, dmax = i, d
        if dmax > tol and idx > 0:
            mantener[idx] = True; pila += [(a, idx), (idx, b)]
    return [p for p, m in zip(puntos, mantener) if m]


def area(anillo):
    return 0.5 * sum(x1 * y2 - x2 * y1 for (x1, y1), (x2, y2) in zip(anillo, anillo[1:] + anillo[:1]))


def centroide(anillo):
    A = area(anillo) or 1e-12
    cx = sum((x1 + x2) * (x1 * y2 - x2 * y1) for (x1, y1), (x2, y2) in zip(anillo, anillo[1:] + anillo[:1])) / (6 * A)
    cy = sum((y1 + y2) * (x1 * y2 - x2 * y1) for (x1, y1), (x2, y2) in zip(anillo, anillo[1:] + anillo[:1])) / (6 * A)
    return cx, cy


def main(ruta):
    g = json.loads(Path(ruta).read_text(encoding="utf-8"))
    proy = lambda lon, lat: ((lon - LON_MIN) * math.cos(LAT0) * K, (LAT_MAX - lat) * K)
    salida, xmax, ymax = {}, 0, 0
    for f in g["features"]:
        k = clave(f["properties"]["DPTO_CNMBR"])
        geom = f["geometry"]
        polis = geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]
        anillos = [[tuple(p[:2]) for p in poli[0]] for poli in polis]      # solo el anillo exterior
        anillos.sort(key=lambda r: -abs(area(r)))
        mayor = abs(area(anillos[0]))
        if k == "san andres":
            # recuadro ampliado: se proyecta la caja del archipiélago dentro de INSET
            pts = [p for r in anillos for p in r]
            lo0, lo1 = min(p[0] for p in pts), max(p[0] for p in pts)
            la0, la1 = min(p[1] for p in pts), max(p[1] for p in pts)
            esc = min((INSET["w"] - 20) / (lo1 - lo0), (INSET["h"] - 30) / (la1 - la0))
            proy_k = lambda lon, lat: (INSET["x"] + 10 + (lon - lo0) * esc, INSET["y"] + 20 + (la1 - lat) * esc)
            conservar = [r for r in anillos if abs(area(r)) > mayor * 0.002]
            tol = TOL / 20
        else:
            proy_k = proy
            conservar = [r for r in anillos if abs(area(r)) > max(mayor * 0.01, 0.0015)] or anillos[:1]
            tol = TOL
        trazos = []
        for r in conservar:
            r = r[:-1] if r[0] == r[-1] else r
            # anillo cerrado: se parte en el vértice más lejano al primero y se simplifica cada mitad
            j = max(range(len(r)), key=lambda i: (r[i][0] - r[0][0]) ** 2 + (r[i][1] - r[0][1]) ** 2)
            s = dp(r[:j + 1], tol)[:-1] + dp(r[j:] + [r[0]], tol)
            xy = [proy_k(*p) for p in s]
            if len(xy) < 4:
                continue
            xmax = max(xmax, *(x for x, _ in xy)); ymax = max(ymax, *(y for _, y in xy))
            trazos.append("M" + "L".join(f"{x:.1f},{y:.1f}" for x, y in xy) + "Z")
        cx, cy = proy_k(*centroide(anillos[0]))
        salida[k] = {"d": "".join(trazos), "cx": round(cx, 1), "cy": round(cy, 1)}
    res = {"viewBox": f"0 0 {math.ceil(xmax) + 10} {math.ceil(ymax) + 10}", "inset": INSET, "deptos": salida,
           "fuente": "DANE, Marco Geoestadístico Nacional 2018 (límites departamentales), simplificado"}
    destino = Path(__file__).with_name("colombia_dptos.json")
    destino.write_text(json.dumps(res, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(destino, f"{destino.stat().st_size // 1024} KB", len(salida), "departamentos", res["viewBox"])


if __name__ == "__main__":
    main(sys.argv[1])
