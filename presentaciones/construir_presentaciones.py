"""
Construye las dos presentaciones HTML autocontenidas del proyecto (un solo archivo cada una, sin dependencias locales):

    presentaciones/Presentacion_Video_U3.html          video ejecutivo de 8-10 min (Producto 2 de la U3)
    presentaciones/Presentacion_Clase_Proyecto.html    presentación en clase de 20-30 min (U1 + U2 + U3)

Uso (desde la raíz del repositorio):

    python presentaciones/construir_presentaciones.py

Todas las cifras salen de los archivos de resultados del proyecto (salidas_u3/, referencias_u2/, el export de ACLED y
la salida real del pipeline guardada en fuentes/salida_pipeline.txt). Ninguna cifra se escribe a mano en las
diapositivas que tienen gráfico: el JavaScript las lee del objeto DATOS que este script incrusta. Las figuras de la U1 y
la U2 se extraen de los cuadernos entregados (docs/entregas/) y las de la U3 de salidas_u3/figuras e informe/figuras.
"""
from __future__ import annotations

import base64
import io
import json
import re
import sys
from pathlib import Path

import pandas as pd
from PIL import Image

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parent
FUENTES = AQUI / "fuentes"
SAL, U2 = RAIZ / "salidas_u3", RAIZ / "referencias_u2"
sys.path.insert(0, str(RAIZ / "src"))
from perfil_amenaza import config as C, datos as D  # noqa: E402


def num(x, n=4):
    return None if pd.isna(x) else round(float(x), n)


# ------------------------------------------------------------------------------------------------ datos
def nombre_modelo(etq: str) -> str:
    e = etq.lower()
    if "embedding" in e:
        return "Embedding (no adoptado)"
    if "u2 fase 2" in e:
        return None
    if "calibrada" in e:
        return "Persistencia calibrada"
    if e.startswith("persistencia"):
        return "Persistencia"
    if "logística" in e:
        return "Logística (9 variables)"
    if "base u2" in e and "ensamble" in e:
        return "Base U2 · ensamble 10"
    if "base u2" in e and ("semilla 42" in e or "individual" in e):
        return "Base U2 · 1 red" if "individual" in e else "Base U2 · semilla 42"
    if "u3" in e and "individual" in e:
        return "MLP U3 · 1 red"
    if "u3" in e:
        return "MLP U3"
    return etq


def comparacion(archivo):
    df = pd.read_csv(SAL / archivo, index_col=0)
    filas = []
    for etq, r in df.iterrows():
        m = nombre_modelo(str(etq))
        if m is None:
            continue
        filas.append({"modelo": m, "ll": num(r.get("Log-loss")), "auc": num(r.get("ROC-AUC")), "ece": num(r.get("ECE")),
                      "brier": num(r.get("Brier")), "mcc": num(r.get("MCC")), "exact": num(r.get("Exactitud")),
                      "trans": num(r.get("Transiciones acertadas"), 1), "ntrans": num(r.get("Transiciones"), 0)})
    return filas


def datos_departamentos():
    proy = pd.read_csv(SAL / "proyeccion_2027_u3.csv")
    panel = pd.read_csv(SAL / "dataset_modelado_u3.csv")
    om = pd.read_csv(SAL / "predicciones_origen_movil.csv")
    t26 = pd.read_csv(SAL / "predicciones_prueba_2026.csv")
    pred = pd.concat([om, t26])[["depto", "anio", "MLP U3", "Persistencia calibrada"]]
    panel = panel.merge(pred, on=["depto", "anio"], how="left")
    salida = {}
    for _, r in proy.iterrows():
        h = panel[panel.depto == r.depto].sort_values("anio")
        salida[r.depto] = {
            "n": r.departamento, "p27": num(r["P(C) 2027"], 4), "cat": r["Categoría 2027"],
            "alerta": bool(r["alerta_AB"]), "sab": num(r["s_AB observada 2025-26"], 4),
            "perf26": r.perfil_info, "sc26": num(r.s_C_info, 4), "ev26": int(r.eventos_info),
            "conf26": bool(r.confiable_info), "enf": r["Énfasis sugerido"],
            "hist": [[int(x.anio), num(x.s_C, 4), x.perfil, bool(x.transicion), int(x.eventos),
                      num(x["MLP U3"], 4)] for _, x in h.iterrows()],
        }
    return salida


def datos_acled():
    a = D.filtrar_horizonte(D.cargar_acled())
    a["perfil"] = a["sub_event_type"].map(C.MAPA_PERFIL_U1).fillna("D")
    por_anio = a.pivot_table(index="anio", columns="perfil", values="events", aggfunc="sum", fill_value=0)
    sub = (a.groupby("sub_event_type")["events"].sum().sort_values(ascending=False).head(10))
    return {
        "filas_export": int(len(D.cargar_acled())), "eventos": int(a["events"].sum()),
        "fatalidades": int(a["fatalities"].sum()), "deptos": int(a["depto"].nunique()),
        "anios": [int(x) for x in por_anio.index],
        "por_anio": {p: [int(v) for v in por_anio.get(p, pd.Series(0, index=por_anio.index))] for p in "ABCD"},
        "subtipos": [[s, int(v), C.MAPA_PERFIL_U1.get(s, "D")] for s, v in sub.items()],
    }


def decisiones():
    d = pd.read_csv(SAL / "decisiones_intervenciones.csv")
    filas = []
    for _, r in d.iterrows():
        ic = None
        if isinstance(r["IC95 Δ (depto)"], str) and r["IC95 Δ (depto)"].startswith("["):
            lo, hi = r["IC95 Δ (depto)"].strip("[]").replace(",", ".").split(";")
            ic = [float(lo), float(hi)]
        m = re.match(r"(I\d)", str(r["Intervención"]))
        filas.append({"id": m.group(1) if m else "Arq.", "nombre": r["Intervención"], "dll": num(r["ΔLog-loss"]),
                      "ee": num(r["EE"]), "ic": ic, "dauc": num(r["ΔAUC"]), "decision": r["Decisión"],
                      "retiene": str(r["Decisión"]).startswith("se retiene")})
    return filas


def importancia():
    """Importancia por permutación tal como la imprimió el cuaderno ejecutado (docs/u3_texto)."""
    txt = (RAIZ / "docs/u3_texto/U3_integrador_texto.md").read_text(encoding="utf-8")
    bloque = txt.split("ΔLog-loss media  Desv. est.", 1)[1].split("Interacción", 1)[0]
    filas = []
    for linea in bloque.strip().splitlines():
        m = re.match(r"\s*(.+?)\s+(-?\d+,\d+)\s+(\d+,\d+)\s*$", linea)
        if m:
            filas.append([m.group(1), float(m.group(2).replace(",", ".")), float(m.group(3).replace(",", "."))])
    return filas


def datos_u2():
    e1 = pd.read_csv(U2 / "experimento_E1_memoria.csv")
    est = pd.read_csv(U2 / "estabilidad_semillas.csv")
    return {
        "e1": [{"val": r["Validación"], "modelo": r.Modelo, "auc": num(r["ROC-AUC"]), "ll": num(r["Log-loss"])}
               for _, r in e1.iterrows()],
        "semillas_auc_prueba": [num(x, 3) for x in est["AUC prueba"]],
        "semillas_bce_prueba": [num(x, 3) for x in est["BCE prueba"]],
    }


def pareadas():
    d = pd.read_csv(SAL / "comparaciones_pareadas.csv")
    d.columns = ["conj", "comp"] + list(d.columns[2:])
    ic = lambda t: [float(x) for x in str(t).strip("[]").split(";")]
    return [{"conj": r.conj, "comp": r.comp, "n": int(r.n), "dll": num(r["ΔLog-loss"]), "ic": ic(r["IC95 ΔLog-loss"]),
             "pm": num(r["P(ΔLog-loss mejora)"], 3), "solo_a": int(r["solo A acierta"]), "solo_b": int(r["solo B acierta"]),
             "p_mc": num(r["p McNemar"], 3)} for _, r in d.iterrows()]


def transiciones_13():
    """Transiciones acertadas 2023-2026 (origen móvil + prueba) por modelo."""
    om, t26 = pd.read_csv(SAL / "predicciones_origen_movil.csv"), pd.read_csv(SAL / "predicciones_prueba_2026.csv")
    todo = pd.concat([om, t26]); tr = todo[todo.transicion]
    return {m: [int(((tr[m] >= .5) == (tr.y == 1)).sum()), int(len(tr))]
            for m in ["MLP U3", "Persistencia", "Persistencia calibrada", "MLP base U2", "MLP base U2 (ensamble 10)"]}


def terminal():
    bloques, actual = {}, None
    for linea in (FUENTES / "salida_pipeline.txt").read_text(encoding="utf-8").splitlines():
        if linea.startswith("### "):
            actual = linea[4:].strip(); bloques[actual] = []
        elif actual:
            bloques[actual].append(linea.rstrip())
    return bloques


def construir_datos():
    esc = pd.read_csv(SAL / "escalera_mejora.csv")
    ctrl = pd.read_csv(SAL / "controles_datos.csv")
    psi = pd.read_csv(SAL / "deriva_psi_2026.csv")
    bandas = pd.read_csv(SAL / "bandas_decision.csv")
    cfg = json.loads((RAIZ / "configuracion.json").read_text(encoding="utf-8"))
    return {
        "mapa": json.loads((FUENTES / "colombia_dptos.json").read_text(encoding="utf-8")),
        "deptos": datos_departamentos(),
        "acled": datos_acled(),
        "escalera": [{"paso": r.Paso, "ll": num(r["LL media"]), "de": num(r["LL DE"]), "auc": num(r.AUC),
                      "trans": num(r["Transiciones acertadas"], 1)} for _, r in esc.iterrows()],
        "comparaciones": {"sel": comparacion("antes_despues_origen_movil.csv"),
                          "t26": comparacion("antes_despues_prueba_2026.csv"),
                          "et": comparacion("validacion_espacio_temporal.csv"),
                          "esp": comparacion("validacion_espacial.csv")},
        "decisiones": decisiones(),
        "controles": [{"nombre": r["Intervención"].replace("ctrl: ", ""), "dll": num(r["ΔLog-loss"]),
                       "ll": num(r["LL media"]), "auc": num(r["AUC media"])} for _, r in ctrl.iterrows()],
        "psi": [{"var": r.variable, "psi": num(r.PSI), "p95": num(r["p95 sin deriva"]), "estado": r.estado,
                 "supera": bool(r["supera el ruido"])} for _, r in psi.iterrows()],
        "bandas": [{"conj": r.Conjunto, "modelo": r.Modelo, "filas": int(r.Filas), "banda": int(r["En banda consolidada"]),
                    "aciertos": int(r["Aciertos en banda consolidada"]), "trans": int(r["En transición"])}
                   for _, r in bandas.iterrows()],
        "importancia": importancia(),
        "u2": datos_u2(),
        "terminal": terminal(),
        "pareadas": pareadas(),
        "trans13": transiciones_13(),
        "resumen27": pd.read_csv(SAL / "proyeccion_2027_u3.csv")["Categoría 2027"].value_counts().to_dict(),
        "panel": {"filas": int(len(pd.read_csv(SAL / "dataset_modelado_u3.csv"))),
                  "repite": num(1 - pd.read_csv(SAL / "dataset_modelado_u3.csv")["transicion"].mean(), 3)},
        "modelo": {"variables": cfg["especificacion_final"].get("variables")},
    }


# ------------------------------------------------------------------------------------------------ figuras
def figuras_cuaderno(ruta):
    nb = json.loads(ruta.read_text(encoding="utf-8"))
    return [base64.b64decode(o["data"]["image/png"]) for c in nb["cells"] if c["cell_type"] == "code"
            for o in c.get("outputs", []) if "image/png" in o.get("data", {})]


def a_webp(png: bytes, ancho_max=2000, calidad=90) -> str:
    im = Image.open(io.BytesIO(png)).convert("RGB")
    if im.width > ancho_max:
        im = im.resize((ancho_max, round(im.height * ancho_max / im.width)), Image.LANCZOS)
    buf = io.BytesIO(); im.save(buf, "WEBP", quality=calidad, method=6)
    return "data:image/webp;base64," + base64.b64encode(buf.getvalue()).decode()


def figuras():
    u1 = figuras_cuaderno(RAIZ / "docs/entregas/U1/Proyecto_U1_Exploracion.ipynb")
    u2 = figuras_cuaderno(RAIZ / "docs/entregas/U2/Proyecto_U2_Modelo_Base.ipynb")
    leer = lambda p: (RAIZ / p).read_bytes()
    fuentes = {
        "u1_subtipos": u1[0], "u1_evolucion": u1[1], "u1_top": u1[2],
        "u2_disenio": u2[5], "u2_arquitectura": u2[7], "u2_curvas": u2[9], "u2_temporal_espacial": u2[16],
        "u2_fronteras": u2[18], "u2_errores": u2[20],
        "u3_datos": leer("informe/figuras/R01_datos.png"), "u3_curvas": leer("informe/figuras/R03_curvas.png"),
        "u3_optimizacion": leer("informe/figuras/R04_optimizacion.png"),
        "u3_controles": leer("informe/figuras/R05_controles.png"),
        "u3_errores_pdp": leer("informe/figuras/R07_errores_pdp.png"),
        "u3_ensamble": leer("salidas_u3/figuras/fig05_ensamble.png"),
        "u3_calibracion": leer("salidas_u3/figuras/fig06_calibracion_umbral.png"),
        "u3_rejilla": leer("salidas_u3/figuras/fig04_rejilla_anidada.png"),
    }
    return {k: a_webp(v) for k, v in fuentes.items()}


# ------------------------------------------------------------------------------------------------ ensamblado
def ensamblar(plantilla: str, destino: str, titulo: str, datos: dict, figs: dict):
    html = (FUENTES / plantilla).read_text(encoding="utf-8")
    usadas = set(re.findall(r"\{\{FIG:(\w+)\}\}", html))
    faltan = usadas - figs.keys()
    if faltan:
        raise KeyError(f"figuras inexistentes en {plantilla}: {faltan}")
    for k in usadas:
        html = html.replace("{{FIG:" + k + "}}", figs[k])
    css = (FUENTES / "motor.css").read_text(encoding="utf-8")
    js = (FUENTES / "motor.js").read_text(encoding="utf-8") + "\n" + (FUENTES / "graficos.js").read_text(encoding="utf-8")
    datos_js = json.dumps(datos, ensure_ascii=False, separators=(",", ":"))
    pagina = f"""<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{titulo}</title>
<meta name="description" content="Perfil de amenaza tecnológica por departamento · Redes Neuronales · Universidad de La Sabana">
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=JetBrains+Mono:wght@400;500;700&family=Space+Grotesk:wght@500;600;700&display=swap" rel="stylesheet">
<style>
{css}
</style>
</head>
<body>
{html}
<script>window.DATOS = {datos_js};</script>
<script>
{js}
</script>
</body>
</html>
"""
    ruta = AQUI / destino
    ruta.write_text(pagina, encoding="utf-8")
    print(f"{ruta.name}: {ruta.stat().st_size / 1e6:.1f} MB, {html.count('class=\"slide')} diapositivas")


def main():
    datos, figs = construir_datos(), figuras()
    (FUENTES / "datos_presentaciones.json").write_text(
        json.dumps({k: v for k, v in datos.items() if k != "mapa"}, ensure_ascii=False, indent=1), encoding="utf-8")
    ensamblar("video.html", "Presentacion_Video_U3.html",
              "Perfil de amenaza · Video ejecutivo U3", datos, figs)
    ensamblar("clase.html", "Presentacion_Clase_Proyecto.html",
              "Perfil de amenaza · Presentación del proyecto", datos, figs)


if __name__ == "__main__":
    main()
