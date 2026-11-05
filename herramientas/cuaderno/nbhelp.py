import json, os
CELLS = []
LECTURAS = {}
_path = os.path.join(os.path.dirname(__file__), "lecturas.json")
if os.path.exists(_path):
    LECTURAS = json.load(open(_path, encoding="utf-8"))

def md(t):
    CELLS.append(("markdown", t.strip("\n")))

def code(t):
    CELLS.append(("code", t.strip("\n")))

def lectura(clave):
    txt = LECTURAS.get(clave)
    if txt:
        CELLS.append(("markdown", txt.strip("\n")))
    else:
        CELLS.append(("markdown", f"**Lectura [{clave}].** *(pendiente)*"))
