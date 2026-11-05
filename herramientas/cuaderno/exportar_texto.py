"""
Exporta el cuaderno ejecutado a texto legible (para revisión, búsqueda y graphify):

    python herramientas/cuaderno/exportar_texto.py            # desde la raíz del proyecto

Escribe docs/u3_texto/U3_integrador_texto.md (markdown y salidas, sin imágenes) y docs/u3_texto/U3_integrador_codigo.py
(código de las celdas en orden). Las tablas HTML se copian con el texto de cada celda tal como se muestra (coma decimal
en las tablas de `graficos.tabla`), sin convertir números.
"""
import io
import json
import re
from pathlib import Path


RAIZ = Path(__file__).resolve().parents[2]
NB = RAIZ / "Proyecto_U3_Integrador.ipynb"
DEST = RAIZ / "docs" / "u3_texto"


def _titulo(fuente: str) -> str:
    for linea in fuente.splitlines():
        t = linea.strip()
        if t.startswith("#") and not set(t) <= set("#= "):
            return t.lstrip("# ").strip()
    return ""


def _tabla_html(html: str) -> str:
    """
    Copia cada tabla HTML con el texto de sus celdas tal como se muestra en el cuaderno (sin convertir números), para
    no confundir la coma decimal de `graficos.tabla` con el punto de las tablas formateadas con Styler.
    """
    import lxml.html
    try:
        raiz = lxml.html.fromstring(html)
    except Exception:
        return re.sub(r"<[^>]+>", " ", html).strip()
    salida = []
    for tabla in raiz.iter("table"):
        filas = [[c.text_content().strip() for c in tr if c.tag in ("th", "td")] for tr in tabla.iter("tr")]
        filas = [f for f in filas if f]
        if not filas:
            continue
        n = max(len(f) for f in filas)
        filas = [f + [""] * (n - len(f)) for f in filas]
        anchos = [max(len(f[j]) for f in filas) for j in range(n)]
        salida.append("\n".join("  ".join(c.rjust(a) for c, a in zip(f, anchos)).rstrip() for f in filas))
    return "\n".join(salida) if salida else re.sub(r"<[^>]+>", " ", html).strip()


def _salidas(celda) -> str:
    partes = []
    for o in celda.get("outputs", []):
        tipo = o.get("output_type")
        if tipo == "stream":
            partes.append("".join(o.get("text", "")).rstrip("\n"))
        elif tipo in ("display_data", "execute_result"):
            d = o.get("data", {})
            if "image/png" in d:
                partes.append("[figura]")
            elif "text/markdown" in d:
                partes.append("".join(d["text/markdown"]).rstrip("\n"))
            elif "text/html" in d:
                partes.append(_tabla_html("".join(d["text/html"])))
            elif "text/plain" in d:
                partes.append("".join(d["text/plain"]).rstrip("\n"))
        elif tipo == "error":
            partes.append(f"ERROR: {o.get('ename')}: {o.get('evalue')}")
    return re.sub(r"\x1b\[[0-9;]*m", "", "\n".join(p for p in partes if p))


def main():
    nb = json.loads(NB.read_text(encoding="utf-8"))
    md = ["> Texto extraído de **Proyecto_U3_Integrador.ipynb** (markdown y salidas de texto, sin imágenes). Código de cada "
          "celda en `U3_integrador_codigo.py`; la lógica completa vive en `src/perfil_amenaza/`.", ""]
    py = ["# Código de las celdas de 'Proyecto_U3_Integrador.ipynb' en orden (generado para lectura y para graphify).", "", ""]
    k = 0
    for c in nb["cells"]:
        fuente = "".join(c["source"])
        if c["cell_type"] == "markdown":
            md += [fuente, ""]
            continue
        k += 1
        tit = _titulo(fuente)
        md += [f"#### Celda de código {k}: {tit}", "Salida de la celda:", "```text", _salidas(c), "```", ""]
        py += [f"# ======== Celda de código {k}: {tit} ========", fuente, "", ""]
    DEST.mkdir(parents=True, exist_ok=True)
    (DEST / "U3_integrador_texto.md").write_text("\n".join(md), encoding="utf-8")
    (DEST / "U3_integrador_codigo.py").write_text("\n".join(py), encoding="utf-8")
    print(f"{k} celdas de código exportadas a {DEST}")


if __name__ == "__main__":
    main()
