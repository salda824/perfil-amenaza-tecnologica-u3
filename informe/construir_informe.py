"""
Construye el informe técnico final de la U3 (Producto 1) en Word y PDF, con el mismo diseño de los informes de la
U1 y la U2 (Times, borde de página, títulos de tablas y figuras en azul, numeración inferior derecha).

    python informe/generar_figuras.py      # primero, si cambiaron los resultados
    python informe/construir_informe.py    # escribe informe/Informe_U3_Proyecto_Integrador.docx y .pdf

Todas las cifras provienen de la ejecución guardada del cuaderno y del pipeline (salidas_u3/).
"""
import re
import subprocess
from pathlib import Path

from docx import Document
from docx.enum.section import WD_ORIENT
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Pt, Cm, RGBColor, Inches

RAIZ = Path(__file__).resolve().parents[1]
FIG = RAIZ / "informe" / "figuras"
SALIDA = RAIZ / "informe" / "Informe_U3_Proyecto_Integrador.docx"
AZUL = RGBColor(0x1F, 0x6F, 0xB5)
FUENTE, TAM = "Times New Roman", 10.5

doc = Document()
sec = doc.sections[0]
sec.page_width, sec.page_height = Inches(8.5), Inches(11)
sec.left_margin = sec.right_margin = Cm(2.0)
sec.top_margin, sec.bottom_margin = Cm(1.9), Cm(1.6)
sec.footer_distance = Cm(1.0)

# Borde de página (como los informes anteriores)
pg = OxmlElement("w:pgBorders"); pg.set(qn("w:offsetFrom"), "page")
for lado in ("top", "left", "bottom", "right"):
    b = OxmlElement(f"w:{lado}"); b.set(qn("w:val"), "single"); b.set(qn("w:sz"), "4"); b.set(qn("w:space"), "24"); b.set(qn("w:color"), "000000")
    pg.append(b)
sec._sectPr.append(pg)

estilo = doc.styles["Normal"]
estilo.font.name, estilo.font.size = FUENTE, Pt(TAM)
estilo.element.rPr.rFonts.set(qn("w:eastAsia"), FUENTE)
estilo.paragraph_format.space_after = Pt(2.5); estilo.paragraph_format.space_before = Pt(0)
estilo.paragraph_format.line_spacing = 1.0

# Número de página abajo a la derecha
fp = sec.footer.paragraphs[0]; fp.alignment = WD_ALIGN_PARAGRAPH.RIGHT
for tipo, txt in (("begin", None), (None, "PAGE"), ("end", None)):
    r = fp.add_run()
    if tipo:
        e = OxmlElement("w:fldChar"); e.set(qn("w:fldCharType"), tipo); r._r.append(e)
    else:
        e = OxmlElement("w:instrText"); e.set(qn("xml:space"), "preserve"); e.text = txt; r._r.append(e)
    r.font.size = Pt(9); r.font.name = FUENTE

TOKEN = re.compile(r"(\*\*.+?\*\*|\*.+?\*|_\{.+?\}|\^\{.+?\}|`.+?`)")


def runs(par, texto, tam=None, color=None, negrita=False, cursiva=False):
    """Mini marcado: **negrita**, *cursiva*, _{subíndice}, ^{superíndice}."""
    for trozo in TOKEN.split(texto):
        if not trozo:
            continue
        b, i, sub, sup, mono = negrita, cursiva, False, False, False
        if trozo.startswith("`") and trozo.endswith("`") and len(trozo) > 2:
            trozo, mono = trozo[1:-1], True
        elif trozo.startswith("**"):
            trozo, b = trozo[2:-2], True
        elif trozo.startswith("*") and len(trozo) > 2:
            trozo, i = trozo[1:-1], True
        elif trozo.startswith("_{"):
            trozo, sub = trozo[2:-1], True
        elif trozo.startswith("^{"):
            trozo, sup = trozo[2:-1], True
        r = par.add_run(trozo); r.bold, r.italic = b, i
        r.font.subscript, r.font.superscript = sub, sup
        r.font.name = "Consolas" if mono else FUENTE
        if tam: r.font.size = Pt(tam - (1 if mono else 0))
        elif mono: r.font.size = Pt(TAM - 1)
        if color: r.font.color.rgb = color
    return par


def p(texto, alin="j", tam=None, antes=0, despues=2.5, sangria=None):
    par = doc.add_paragraph()
    par.alignment = {"j": WD_ALIGN_PARAGRAPH.JUSTIFY, "c": WD_ALIGN_PARAGRAPH.CENTER, "l": WD_ALIGN_PARAGRAPH.LEFT}[alin]
    par.paragraph_format.space_before, par.paragraph_format.space_after = Pt(antes), Pt(despues)
    if sangria is not None:
        par.paragraph_format.left_indent = Cm(sangria)
    return runs(par, texto, tam)


def viñeta(texto, nivel=0, numero=None):
    par = doc.add_paragraph()
    par.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    par.paragraph_format.left_indent = Cm(0.55 + 0.5 * nivel); par.paragraph_format.first_line_indent = Cm(-0.4)
    par.paragraph_format.space_after = Pt(1.5)
    runs(par, (f"{numero}. " if numero else "•  ") + texto)
    return par


def h1(texto):
    par = doc.add_paragraph(); par.paragraph_format.space_before = Pt(5); par.paragraph_format.space_after = Pt(2)
    par.paragraph_format.left_indent = Cm(0.9); par.paragraph_format.keep_with_next = True
    runs(par, texto, tam=12.5, negrita=True)


def h2(texto):
    par = doc.add_paragraph(); par.paragraph_format.space_before = Pt(3); par.paragraph_format.space_after = Pt(1.5)
    par.paragraph_format.keep_with_next = True
    runs(par, texto, tam=11, negrita=True, cursiva=True)


def leyenda(texto):
    par = doc.add_paragraph(); par.paragraph_format.space_before = Pt(1); par.paragraph_format.space_after = Pt(3)
    par.paragraph_format.keep_with_next = texto.startswith("Tabla")
    r = par.add_run(texto); r.italic = True; r.font.size = Pt(8.3); r.font.color.rgb = AZUL; r.font.name = "Calibri"
    return par


def _bordes(tabla):
    tblPr = tabla._tbl.tblPr
    bd = OxmlElement("w:tblBorders")
    for lado in ("top", "left", "bottom", "right", "insideH", "insideV"):
        e = OxmlElement(f"w:{lado}"); e.set(qn("w:val"), "single"); e.set(qn("w:sz"), "4"); e.set(qn("w:color"), "000000"); bd.append(e)
    tblPr.append(bd)


def _margen_celdas(tabla, cm=0.08):
    tblPr = tabla._tbl.tblPr; m = OxmlElement("w:tblCellMar")
    for lado in ("left", "right"):
        e = OxmlElement(f"w:{lado}"); e.set(qn("w:w"), str(int(cm * 567))); e.set(qn("w:type"), "dxa"); m.append(e)
    tblPr.append(m)


def _fijar(t, anchos):
    """Anchos de columna fijos (LibreOffice y Word respetan tblLayout fixed + tblGrid)."""
    t.autofit = False
    tblPr = t._tbl.tblPr
    lay = OxmlElement("w:tblLayout"); lay.set(qn("w:type"), "fixed"); tblPr.append(lay)
    w = OxmlElement("w:tblW"); w.set(qn("w:w"), str(int(sum(anchos) * 567))); w.set(qn("w:type"), "dxa")
    for e in tblPr.findall(qn("w:tblW")):
        tblPr.remove(e)
    tblPr.append(w)
    grid = t._tbl.tblGrid
    for gc, a in zip(grid.findall(qn("w:gridCol")), anchos):
        gc.set(qn("w:w"), str(int(a * 567)))


def tabla(titulo, filas, anchos, tam=8.3, negrita_col0=False, centrar_desde=None, centrar=None):
    leyenda(titulo)
    t = doc.add_table(rows=len(filas), cols=len(filas[0])); t.alignment = WD_TABLE_ALIGNMENT.CENTER
    _bordes(t); _margen_celdas(t); _fijar(t, anchos)
    for i, fila in enumerate(filas):
        for j, val in enumerate(fila):
            c = t.cell(i, j); c.width = Cm(anchos[j]); c.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            par = c.paragraphs[0]; par.paragraph_format.space_after = Pt(0)
            centro = i == 0 or (centrar_desde is not None and j >= centrar_desde) or (centrar is not None and j in centrar)
            par.paragraph_format.keep_with_next = i < len(filas) - 1        # la tabla no se parte entre páginas
            par.alignment = WD_ALIGN_PARAGRAPH.CENTER if centro else WD_ALIGN_PARAGRAPH.LEFT
            runs(par, str(val), tam=tam, negrita=(i == 0) or (negrita_col0 and j == 0))
    for j, w in enumerate(anchos):
        for c in t.columns[j].cells:
            c.width = Cm(w)
    if titulo:
        pass
    doc.add_paragraph().paragraph_format.space_after = Pt(0)
    return t


def figura(nombre, ancho_cm, texto):
    par = doc.add_paragraph(); par.alignment = WD_ALIGN_PARAGRAPH.CENTER
    par.paragraph_format.space_after = Pt(0); par.paragraph_format.keep_with_next = True
    par.add_run().add_picture(str(FIG / f"{nombre}.png"), width=Cm(ancho_cm))
    leyenda(texto)


def caja(texto):
    t = doc.add_table(rows=1, cols=1); _bordes(t)
    par = t.cell(0, 0).paragraphs[0]; par.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    runs(par, texto, negrita=True, cursiva=True)
    doc.add_paragraph().paragraph_format.space_after = Pt(0)


def lado_a_lado(parrafos, nombre, ancho_fig, texto_fig, ancho_texto=10.0):
    """Texto a la izquierda y figura a la derecha (como la §3.2 y la §6.5 de la U2)."""
    t = doc.add_table(rows=1, cols=2); _fijar(t, [ancho_texto, 17.5 - ancho_texto])
    izq, der = t.cell(0, 0), t.cell(0, 1)
    izq.width, der.width = Cm(ancho_texto), Cm(17.5 - ancho_texto)
    primero = True
    for txt in parrafos:
        par = izq.paragraphs[0] if primero else izq.add_paragraph(); primero = False
        par.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY; par.paragraph_format.space_after = Pt(2.5)
        runs(par, txt)
    pf = der.paragraphs[0]; pf.alignment = WD_ALIGN_PARAGRAPH.CENTER
    pf.add_run().add_picture(str(FIG / f"{nombre}.png"), width=Cm(ancho_fig))
    lg = der.add_paragraph(); r = lg.add_run(texto_fig); r.italic = True; r.font.size = Pt(8.3); r.font.color.rgb = AZUL; r.font.name = "Calibri"


# =====================================================================================================================
# Portada (encabezado del informe)
# =====================================================================================================================
par = doc.add_paragraph(); par.alignment = WD_ALIGN_PARAGRAPH.CENTER; par.paragraph_format.space_before = Pt(8)
runs(par, "Modelo neuronal integrado para anticipar el perfil de amenaza tecnológica de los territorios colombianos y orientar la adopción de tecnologías militares", tam=13.5, negrita=True)
for txt in ("Sector Elegido: Seguridad y defensa",
            "Proyecto Integrador, Etapa 3 - Laura Alejandra Díaz Sastoque (0000288870) & Juan Diego Saldarriaga Muñoz (0000287391)",
            "Código: https://github.com/salda824/perfil-amenaza-tecnologica-u3"):
    q = doc.add_paragraph(); q.alignment = WD_ALIGN_PARAGRAPH.CENTER; q.paragraph_format.space_after = Pt(0)
    r = q.add_run(txt); r.font.size = Pt(10); r.font.color.rgb = RGBColor(0x59, 0x59, 0x59); r.font.name = FUENTE

# =====================================================================================================================
h1("1.   Planteamiento del problema y contexto")
p("Colombia ejecuta en 2026 la mayor inversión de modernización militar de su historia sin un modelo basado en datos que "
  "oriente qué tecnología desplegar en cada territorio (Ministerio de Defensa de Colombia, 2025). En la Etapa 1 formulamos "
  "la pregunta orientadora:")
caja("¿Cómo orientar la adquisición, transferencia de tecnología y fabricación de sistemas de seguridad y defensa de las "
     "Fuerzas de Colombia en función del perfil de amenaza tecnológica predominante de los territorios, según su actividad "
     "armada histórica, geografía operacional y economías ilícitas, de modo que se optimice la inversión?")
p("El problema se mantiene igual que en las etapas anteriores. Para cada departamento *d* y año *t*, la red estima, con lo "
  "observado hasta el cierre de *t*−1, la probabilidad de que el perfil sea **C** (armas convencionales y emboscadas) o "
  "**D** (amenaza mixta): s^{C}_{d,t} = eventos C / eventos totales, y_{d,t} = 1[s^{C}_{d,t} ≥ 0,5], "
  "p̂_{d,t} = f_{θ}(x_{d,t−1}). El rezago evita la circularidad y reproduce el momento real de la decisión de inversión. "
  "La Tabla 1 resume qué dejó cada etapa y por qué la tercera se planteó como lo hicimos.")
tabla("Tabla 1: Trayectoria del proyecto y hallazgo que condiciona cada etapa", [
    ["Etapa", "Resultado", "Hallazgo que condiciona la siguiente"],
    ["U1 · Formulación", "Perfiles A/B/C/D con la regla del 50 %, 7 variables, ACLED agregado a departamento",
     "Ningún departamento es A o B dominante → clasificación C vs. D; 33 filas no bastan → panel departamento–año"],
    ["U2 · Modelo base", "MLP 7 → 16 → 1 (145 parámetros); AUC temporal 0,89; equivalente a la logística",
     "La persistencia del perfil (AUC 0,965; log-loss 0,380) supera a la red; la memoria (E1) es la palanca más fuerte"],
    ["U3 · Integración", "10 frentes de intervención, controles de datos, pipeline reproducible, prueba 2026 única",
     "El rival es la persistencia, no el MLP base; el valor propio de un modelo está en anticipar cambios de perfil"],
], [2.8, 6.6, 8.0])

# =====================================================================================================================
h1("2.   Descripción del dataset y preprocesamiento")
h2("2.1 Fuentes, panel y variables")
p("**Fuentes.** El núcleo es el export semanal de ACLED para Colombia (Raleigh et al., 2010): 17.573 registros semana × "
  "departamento × subtipo hasta el 15 de agosto de 2026. Se integran la coca del censo SIMCI (Observatorio de Drogas de "
  "Colombia, 2026), la cobertura arbórea de Global Forest Watch (Hansen et al., 2013), el IPM de DANE (2019) y, en esta "
  "etapa, UCDP GED 26.1 (Sundberg y Melander, 2013) para la variable de actores armados. **Unidad.** El panel tiene 171 "
  "filas departamento–año etiquetables (30 departamentos, 2021–2026, al menos 10 eventos por celda). El paquete de la U3 "
  "reconstruye el dataset de la U2 **al dígito**: las mismas etiquetas y una diferencia máxima de 1,7×10^{−13} en las 7 "
  "predictoras, de modo que todo cambio posterior se debe a las intervenciones. La Tabla 2 muestra el estado de cada "
  "variable al final de la U3.")
tabla("Tabla 2: Variables del modelo (todas medidas en t−1) y su estado al cierre de la U3", [
    ["ID", "Variable", "Operacionalización", "Fuente", "Estado U3"],
    ["x_{1}, x_{2}, x_{2b}", "Eventos, fatalidades, letalidad", "log(1 + anualizado); fatalidades/evento", "ACLED", "En el modelo (U2)"],
    ["x_{4}–x_{7}", "Coca, bosque, distancia, IPM", "Igual que la U2", "SIMCI, GFW, centroides, DANE", "En el modelo (U2)"],
    ["**m₁**", "Memoria del perfil", "s̃^{C} contraída del año t−1", "ACLED", "**Nueva, retenida (I1)**"],
    ["**d₁**", "Semestre reciente", "s̃^{C} de las últimas 26 semanas de t−1", "ACLED semanal", "**Nueva, retenida (I2)**"],
    ["m_{2}, m_{3}; d_{2}, d_{3}", "Trayectoria y tendencias", "s̃^{C} en t−2 y cambio; tendencias semestrales", "ACLED", "Evaluadas, no retenidas"],
    ["x_{3}", "Grupos armados activos", "log(1 + actores no estatales distintos)", "UCDP GED 26.1", "Evaluada, no retenida (I3)"],
    ["v_{1}, v_{2}", "Vecindad", "s̃^{C} y eventos de los departamentos contiguos", "ACLED + IGAC", "Evaluada, no retenida (I4)"],
], [2.6, 3.6, 5.2, 3.0, 3.0])
p("**Representación.** Las proporciones nuevas se calculan con **contracción bayesiana**, s̃^{C} = (C + 2)/(n + 4), "
  "equivalente a un prior Beta(2, 2). Así, un semestre con pocos eventos no produce proporciones extremas, y con 100 "
  "eventos el ajuste no pasa de 2 puntos. Todas las variables se estandarizan con media y desviación calculadas **solo con "
  "el entrenamiento de cada pliegue**. Los conteos y las hectáreas mantienen el logaritmo de la U2.")
p("**Control de la etiqueta parcial.** La prueba de 2026 tiene solo 33 semanas. Recalculamos la etiqueta de 2021–2025 con "
  "las primeras 33 semanas de cada año y coincide con la del año completo en el **94,9 %** de 136 celdas. Las 7 "
  "discrepancias tienen todas s^{C} a menos de 10 puntos del umbral (Figura 1b). En 2026 se esperan entonces 1 o 2 "
  "etiquetas cambiadas, y todas en la franja ambigua.")
figura("R01_datos", 16.2, "Figura 1: (a) En el 70 % de las 23 transiciones, el segundo semestre del año anterior (d₁) ya se movía hacia el nuevo "
       "perfil. (b) La etiqueta a la semana 33 coincide con la anual en el 94,9 % de las celdas 2021-2025")
h2("2.2 Controles experimentales de la preparación de datos")
p("Cada decisión de preprocesamiento se puso a prueba quitándola, o introduciendo a propósito un error conocido, con el "
  "mismo protocolo y las mismas 5 semillas (Tabla 3, Figura 2).")
tabla("Tabla 3: Controles de preparación de datos (Δ log-loss frente a la referencia de 0,310; 5 semillas, origen móvil 2023-2025)", [
    ["Control", "Δ log-loss (EE)", "Lectura"],
    ["Sin estandarización", "+5,780 (1,445)", "No converge: distancias y porcentajes saturan ReLU y sigmoide; BCE de entrenamiento fija en 7,1"],
    ["Etiquetas barajadas (cordura)", "+0,506 (0,012)", "AUC 0,32: el desempeño medido viene de señal real, no del protocolo"],
    ["Sin pesos de clase", "+0,015 (0,002)", "Con prevalencias de C entre 41 % y 62 %, la red se inclina hacia D"],
    ["Mapeo alternativo de subtipos", "+0,007 (0,001)", "La regla de la U1 se sostiene"],
    ["Sin contracción (α ≈ 0)", "+0,002 (0,001)", "Aporte pequeño, en la dirección esperada"],
    ["Escalador con fuga (control negativo)", "−0,002 (0,001)", "Cuantifica la mejora artificial que daría una fuga; las pruebas automáticas la impiden"],
    ["Sin logaritmos", "−0,012 (0,003)", "No anticipado: levemente mejor. No se cambia en esta iteración (regla 7, §4.1); en el pliegue 2025 es peor"],
    ["MIN_EVENTOS = 5 / 10 / 20", "0,300 / 0,307 / 0,338", "Log-loss en las mismas filas (≥ 20 eventos): 5 ≈ 10 y 20 empeora; se mantiene 10"],
], [4.6, 2.8, 10.0], centrar={1})
figura("R05_controles", 16.2, "Figura 2: (a) Curvas de entrenamiento con y sin cada transformación (pliegue 2025; la versión sin estandarizar no "
       "baja de 7,1). (b) Efecto de cada control sobre la log-loss fuera de muestra")

# =====================================================================================================================
h1("3.   Arquitectura del modelo y justificación")
lado_a_lado([
    "El modelo final es un **ensamble de 10 perceptrones multicapa** idénticos en estructura (API funcional de Keras): 9 "
    "entradas estandarizadas, una capa oculta de 16 neuronas ReLU y una salida sigmoide, con 177 parámetros por red (1,25 por "
    "observación). Cada red se entrena con una semilla distinta y la predicción es el promedio de las 10 probabilidades. "
    "Es la misma arquitectura de la U2 con dos entradas más (m₁ y d₁, en verde en la Figura 3).",
    "La arquitectura no se mantuvo por inercia. Se compararon una **GRU** y una **CNN 1D** sobre los cuatro trimestres de "
    "*t*−1 (log-loss 0,319 y 0,324, frente a 0,310), un **MLP profundo** (16, 8; 0,309, empate), un **embedding de "
    "departamento** (0,284, §6) y la **logística** con las mismas variables (0,306). Las redes secuenciales pagan en "
    "varianza sus 393–665 parámetros sin encontrar en los trimestres más información que la que ya resume d₁.",
], "R02_arquitectura", 6.8, "Figura 3: Arquitectura final (9 → 16 → 1, 177 parámetros por red; verde = variables nuevas)", ancho_texto=10.4)
tabla("Tabla 4: Configuración del modelo final y su justificación", [
    ["Componente", "Elección", "Justificación técnica"],
    ["Entrada", "9 variables en t−1, estandarizadas con el entrenamiento", "7 de la U2 + m₁, d₁ (únicas que superaron la regla de retención, §5)"],
    ["Capa oculta", "16 neuronas ReLU, He normal", "La rejilla de 48 configuraciones fue plana y la validación anidada mostró que elegir otra no transfiere (§5)"],
    ["Salida y pérdida", "Sigmoide; BCE ponderada por clase + L2 (λ = 10⁻³)", "Log-verosimilitud Bernoulli (Goodfellow et al., 2016); sin pesos de clase la log-loss sube 0,015"],
    ["Optimizador", "Adam, η = 10⁻³, lote 16, 560 pasos fijos", "Presupuesto mediano de la U2 (D10); las curvas confirman que sigue cerca del óptimo (§4)"],
    ["Ensamble", "10 semillas, promedio de probabilidades", "Reduce 6 veces la dispersión entre semillas (Lakshminarayanan et al., 2017)"],
    ["Calibración y umbral", "Sin recalibración; umbral 0,5 y bandas 0,3/0,7", "Platt y temperatura no mejoraron sin mirar el futuro; el umbral óptimo cambia cada año"],
], [3.0, 6.0, 8.4])

# =====================================================================================================================
h1("4.   Proceso de entrenamiento y evaluación")
h2("4.1 Protocolo fijado antes de ver resultados")
p("**Selección.** Validación de origen móvil (Bergmeir y Benítez, 2012): para *v* = 2023, 2024 y 2025 se entrena con *t* < *v* "
  "y se predice *v*, lo que da 87 predicciones fuera de muestra. **Prueba.** 2026 se evalúa una sola vez, con todo congelado "
  "en `configuracion.json`. Lo declaramos: la U2 ya había reportado 2026. **Rival.** La persistencia, p̂_{d,t} = "
  "s^{C}_{d,t−1} (AUC 0,973; log-loss 0,363), que acierta 0 transiciones por construcción. **Métrica de decisión.** "
  "Log-loss, una regla de puntuación propia (Gneiting y Raftery, 2007), con AUC, Brier, ECE, MCC y aciertos en transiciones "
  "como apoyo. **Regla de retención.** Cada variante se entrena con las mismas 5 semillas que su referencia y se retiene "
  "si la mejora media de la log-loss supera un error estándar de la diferencia pareada, sin bajar el AUC más de 0,01. "
  "**Reglas 7 y 8.** Los controles de datos verifican, no seleccionan. Una variante que gana solo porque se parece más a "
  "la persistencia, o porque no puede aplicarse a territorios nuevos, se documenta pero no se adopta. Las comparaciones "
  "pareadas usan bootstrap por departamento y McNemar (Dietterich, 1998).")
h2("4.2 Entrenamiento y curvas de desempeño")
p("Cada red se entrena con el bucle explícito de tf.GradientTape verificado en la U2 frente a model.fit (diferencia máxima "
  "2×10^{−6}), generalizado en el paquete para objetivo blando, salida multitarea y entradas secuenciales. La Figura 4 "
  "compara las curvas del modelo base y del final, entrenando con 2021–2024 y validando en 2025, con 5 semillas y tres "
  "veces el presupuesto. En el modelo final, la BCE de validación en el paso 560 es **0,377**, frente a un mínimo de "
  "0,360: el presupuesto sigue cerca del óptimo, y entrenar el triple la sube a 0,441 (sobreajuste). El modelo final baja "
  "a la vez la BCE de entrenamiento (0,341 → 0,279) y la de validación (0,448 → 0,377), y reduce la brecha entre ambas "
  "(0,107 → 0,097). No es una red más flexible que memoriza más, es una red mejor informada.")
figura("R03_curvas", 16.2, "Figura 4: Curvas de aprendizaje con validación 2025 (media y rango de 5 semillas). Línea discontinua: presupuesto de 560 pasos")

# =====================================================================================================================
h1("5.   Optimización realizada y comparación de resultados")
p("Cada intervención responde a un hallazgo medido en la U2 y tiene una hipótesis refutable. Las de variables se "
  "aplicaron en cascada (cada una sobre la mejor especificación retenida) y las de modelo, sobre el conjunto de variables ya "
  "fijado (Tabla 5).")
tabla("Tabla 5: Del diagnóstico de la U2 a la decisión (Δ log-loss media entre 5 semillas pareadas; EE entre paréntesis)", [
    ["#", "Diagnóstico U2", "Intervención", "Resultado (2023-2025)", "Decisión"],
    ["I1", "La inercia domina (persistencia AUC 0,965)", "Memoria m₁ (y m₂, m₃)", "m₁: −0,036 (0,009); AUC +0,017. m₂, m₃: +0,002", "**Retiene m₁**"],
    ["I2", "Solo vista anual; escaladas de 2025", "Semestre reciente d₁ (y d₂, d₃)", "d₁: −0,037 (0,007); AUC 0,951. d₂, d₃: +0,013", "**Retiene d₁**"],
    ["I3", "Arauca mal clasificada; falta x₃", "Actores armados (UCDP)", "+0,025 (0,006); AUC −0,014", "No retiene"],
    ["I4", "Validación espacial débil", "Vecindad v₁, v₂", "Temporal −0,004 (0,002); espacial +0,009", "No retiene"],
    ["I5", "20 % de celdas cerca del umbral", "Objetivo blando s^{C}", "+0,119 sin calibrar; +0,068 calibrado", "No retiene"],
    ["I9", "A/B diluidas en la etiqueta", "Salida multitarea A/B/C/D", "P(C): +0,001; A/B peor que la persistencia", "No retiene"],
    ["–", "MLP sin estructura temporal", "GRU, CNN 1D, profundo, embedding", "+0,010; +0,014; 0,000; −0,026 (§6)", "No adopta"],
    ["I6", "Capacidad inestable entre análisis", "Rejilla de 48 config. + anidada", "Elegido en 2023-24 empeora 2025 (0,41–0,42 vs. 0,36)", "Mantiene U2"],
    ["I7", "AUC 0,848 ± 0,024 entre semillas", "Ensamble de 10 redes", "LL −0,011; banda 5-95 % de 0,053 a 0,009", "**Retiene**"],
    ["I8", "Subestima P(C); umbral no transfiere", "Platt / temperatura; umbral", "LL 0,298 → 0,302 (Platt hacia adelante)", "No retiene"],
], [0.8, 4.0, 3.9, 5.7, 2.4], tam=7.8)
p("**La lección de I6.** La rejilla de capacidad (4–32 neuronas), L2, activación (ReLU, tanh) y presupuesto (280–1.120 pasos) "
  "es plana: 31 de 48 configuraciones quedan dentro de un error estándar de la mejor. La **validación anidada** pone a "
  "prueba el propio proceso de selección (Cawley y Talbot, 2010). La configuración con mejor log-loss en 2023–2024 "
  "obtiene 0,423 en 2025, la elegida por la regla de 1 EE obtiene 0,411 y la vigente de la U2 obtiene **0,359** (Figura "
  "5b). Con 58 filas de validación, optimizar hiperparámetros ajusta ruido. **I8.** Ajustado año por año, Platt da una "
  "escala de 1,31 en 2023 y 3,38 en 2024, y diverge en 2025, cuando la red ordena sin errores. El umbral que maximiza el "
  "MCC es 0,20, 0,51 y 0,15 según el año. La incertidumbre se gestiona en la decisión, con las bandas 0,3/0,7 de la Etapa "
  "1, no con un umbral ajustado.")
figura("R04_optimizacion", 16.2, "Figura 5: (a) Escalera de mejora: solo se retuvieron m₁, d₁ y el ensamble. (b) Validación anidada: lo que se elige "
       "con 2023-2024 es peor en 2025 que la configuración vigente")
tabla("Tabla 6: Antes (modelo base U2, semilla 42) vs. después (modelo final U3) frente a la persistencia y la logística con las mismas variables", [
    ["Conjunto", "Modelo", "ROC-AUC", "Log-loss", "Brier", "ECE", "MCC", "Transiciones"],
    ["Origen móvil 2023-2025 (n = 87)", "Persistencia", "0,973", "0,363", "0,106", "0,173", "0,795", "0 de 9"],
    ["", "Antes: MLP base U2", "0,913", "0,407", "0,132", "0,084", "0,659", "2 de 9"],
    ["", "Logística (mismas variables)", "0,949", "0,306", "0,097", "0,068", "0,737", "1 de 9"],
    ["", "**Después: MLP U3 (ensamble)**", "**0,956**", "**0,304**", "**0,095**", "0,124", "0,691", "2 de 9"],
    ["Prueba 2026 (n = 29), una sola vez", "Persistencia", "0,948", "0,429", "0,131", "0,207", "0,755", "0 de 4"],
    ["", "Antes: MLP base U2", "0,852", "0,524", "0,166", "0,104", "0,517", "1 de 4"],
    ["", "Logística (mismas variables)", "0,929", "0,383", "0,129", "0,114", "0,531", "0 de 4"],
    ["", "**Después: MLP U3 (ensamble)**", "0,900", "**0,380**", "**0,121**", "**0,074**", "0,672", "0 de 4"],
], [4.2, 4.4, 1.5, 1.5, 1.3, 1.3, 1.3, 1.9], tam=8, centrar_desde=2)
p("**Antes vs. después.** En selección, la log-loss baja de 0,407 a 0,304 (Δ −0,103; IC95 por departamento [−0,185; "
  "−0,033]), el Brier de 0,132 a 0,095 (IC [−0,068; −0,009]) y el AUC sube de 0,913 a 0,956 (IC [+0,005; +0,105]). En la "
  "prueba 2026, la log-loss baja de 0,524 a 0,380 (Δ −0,144; IC [−0,270; −0,033]), y el modelo queda mejor calibrado que "
  "las alternativas (ECE 0,074). La mejora es significativa frente al modelo de la U2 en los dos conjuntos.")

# =====================================================================================================================
h1("6.   Análisis crítico del desempeño")
figura("R06_antes_despues", 16.2, "Figura 6: (a) Curvas ROC en 2023-2025. (b) Log-loss por año: el modelo final supera a la persistencia en los cuatro "
       "años. (c) Matriz de confusión del modelo final en la prueba 2026")
viñeta("**Frente a la inercia, la mejora es probable pero no concluyente.** En selección, Δ log-loss = −0,059 frente a la "
       "persistencia (IC95 [−0,142; +0,028]; el modelo gana en el 92 % de los remuestreos). En 2026, Δ = −0,049 (IC "
       "[−0,162; +0,092]). La persistencia conserva el mejor AUC (0,973 y 0,948): ordena mejor, pero sus probabilidades "
       "son peores (ECE 0,173 y 0,207), porque usa s^{C} como si fuera una probabilidad. Con 30 departamentos, la "
       "conclusión honesta es que el modelo supera a la inercia en calidad probabilística y no en ordenamiento.")
viñeta("**La debilidad espacial de la U2 quedó resuelta.** Con validación agrupada por departamento (Roberts et al., 2017), "
       "la log-loss en territorios nunca vistos pasa de 0,475 (modelo base) a **0,351**, mejor que la persistencia (0,418; Δ "
       "−0,067, IC [−0,135; +0,003]). Las variables estructurales casi constantes funcionaban como identificadores. Con "
       "información de composición, la red aprende una relación que se transfiere: es el argumento cuantitativo para "
       "escalar a municipio.")
viñeta("**Errores sistemáticos.** De 19 errores en 2023–2026, el 58 % son transiciones (frente al 2 % entre los aciertos) "
       "y el 58 % tiene s^{C} a ±10 puntos del umbral (Figura 7a). Se corrigió el error estructural de la U2: Arauca recibe "
       "P(C) entre 0,77 y 0,85 en 2023–2026, todos C. Persisten las escaladas D → C de 2025 (Córdoba, Magdalena, Cesar, "
       "Meta, Bolívar y Valle), las desescaladas C → D de 2026 (Bolívar, Chocó, Magdalena y Valle, las 4 transiciones de "
       "la prueba, que ningún modelo acierta) y Casanare, con muy pocos eventos. El modelo acierta 2 de 13 transiciones y la "
       "persistencia 0: anticipar cambios de régimen sigue siendo la limitación principal.")
viñeta("**¿Aporta la red neuronal?** Con las mismas variables, la logística empata (Δ −0,002 en selección; −0,003 en 2026). "
       "La no linealidad aparece en la forma (Figura 7b): la frontera entre memoria y letalidad es curva, y con 0,6 "
       "fatalidades por evento basta una memoria de ≈ 0,45 para esperar C (sin letalidad hacen falta ≈ 0,7). Pero no aparece "
       "en el desempeño. Se mantiene el MLP por coherencia con la U1 y porque es la arquitectura que debe escalar.")
viñeta("**Un resultado incómodo que reportamos tal cual.** El embedding de departamento cumplió la regla numérica, pero no "
       "se adoptó por la regla 8: acertaba 0,8 de 9 transiciones (frente a 2,0) y no transfiere a territorios nuevos. En "
       "2026 obtuvo log-loss 0,314 y AUC 0,933, mejor que el modelo adoptado. No cambiamos la decisión, porque elegir con "
       "la prueba la invalidaría. La lectura es que los efectos fijos territoriales tienen valor predictivo que las "
       "variables actuales no capturan (R7).")
figura("R07_errores_pdp", 16.2, "Figura 7: (a) P(C) predicha frente a la proporción observada, 2023-2026 (rótulos: errores en transiciones y Casanare). "
       "(b) Dependencia parcial de P(C) entre memoria y letalidad (línea negra: P = 0,5)")

# =====================================================================================================================
h1("7.   Integración final del modelo en un pipeline estructurado")
p("El vacío principal frente a la guía era la integración: en la U2 el cuaderno lo hacía todo. En la U3, todo el proceso "
  "vive en el paquete `perfil_amenaza` del repositorio, con siete módulos: `config`, `datos`, `variables`, `modelo`, "
  "`evaluacion`, `pipeline` y `graficos`. El cuaderno importa ese paquete en lugar de duplicar la lógica. Una sola orden "
  "ejecuta las 8 etapas de la Figura 8 en unos 30 segundos, y cada etapa deja un artefacto verificable.")
figura("R08_pipeline", 15.5, "Figura 8: Pipeline integrado. Las decisiones se congelan en configuracion.json antes de la prueba 2026")
viñeta("**Artefacto único.** Un archivo `perfil_amenaza_u3.keras` (152 KB) contiene la estandarización (capa Normalization con "
       "las medias y desviaciones del entrenamiento), las 10 redes, el promedio y la capa de calibración. Recibe las 9 "
       "variables en su escala original y devuelve P(C), sin depender del cuaderno.")
viñeta("**Pruebas automáticas (pytest).** Contrato de datos, ausencia de fuga temporal (multiplicar por 5 los eventos de 2024 "
       "no cambia ninguna predictora de las filas de 2024 y sí las de 2025), escalador ajustado solo con años anteriores, "
       "determinismo y reproducción del dataset de la U2. Las 5 pasan.")
viñeta("**Consistencia.** Las log-loss del pipeline (0,304 en selección y 0,380 en prueba) coinciden al cuarto decimal con las "
       "del cuaderno. El repositorio incluye `ejecutar_todo.bat`, que reproduce todo en Windows con un solo comando.")
viñeta("**Operación.** La tarjeta del modelo (Mitchell et al., 2019) fija el uso previsto, los límites y el mantenimiento. El "
       "índice de estabilidad poblacional (PSI) de la información de 2026 marca deriva en m₁ (0,29) y en la letalidad (0,25), "
       "así que la proyección 2027 debe leerse con más cautela que la de un año estable.")

# =====================================================================================================================
h1("8.   Del modelo a la decisión: priorización territorial 2027")
lado_a_lado([
    "El ensamble se reentrenó con todas las etiquetas 2021–2026 y se aplicó, a través del artefacto, a la información de "
    "2026. Clasifica **11 departamentos como C consolidado** (P ≥ 0,7: Guaviare, Nariño, Cauca, Caquetá, Putumayo, "
    "Antioquia, Norte de Santander, Vichada y Cesar, además de Amazonas y Vaupés, que tienen pocos datos), **11 en "
    "transición** y **11 como D consolidado**.",
    "La traducción a capacidades sigue la Etapa 1: ISR, detección temprana y movilidad protegida donde domina C; portafolio "
    "flexible y monitoreo trimestral en transición; orden público y capacidades mixtas en D. **Guaviare (14,4 %) y Cauca "
    "(13,4 %)** mantienen la **alerta A/B**, por su proporción de explosivos y drones: allí conviene sumar C-UAS y detección "
    "de AEI.",
    "Frente a la U2, 25 de 33 departamentos conservan su categoría. Las discrepancias con el perfil observado en 2026 bajan "
    "de 10 a 3 (Chocó, Valle del Cauca y Meta). Chocó y Valle son desescaladas recientes con s^{C} de 2026 entre 0,45 y "
    "0,50, la franja ambigua. La dispersión entre las 10 redes crece donde el modelo duda (Chocó 0,16; Córdoba 0,12), y "
    "debe leerse junto con la probabilidad. Es un insumo para priorizar, no un reemplazo del análisis de inteligencia.",
], "R09_proyeccion", 6.9, "Figura 9: P(perfil C) proyectada para 2027 (centroides; tamaño proporcional a los eventos de 2026)", ancho_texto=10.2)

# =====================================================================================================================
h1("9.   Conclusiones técnicas y recomendaciones futuras")
for k, txt in enumerate([
    "**Coherencia.** El modelo final responde la misma pregunta de la U1, con la misma etiqueta, el mismo diseño *t*−1 → *t* "
    "y las mismas variables estructurales, y agrega dos variables que operacionalizan la \"intensidad histórica\" de la U1 "
    "como composición reciente.",
    "**Optimizar fue intervenir los datos, no la arquitectura.** De 10 frentes evaluados se retuvieron tres (m₁, d₁ y el "
    "ensamble). La capacidad, la profundidad, las redes secuenciales, la salida multitarea, el objetivo blando, la vecindad, "
    "los actores y la calibración no superaron el ruido entre semillas. La validación anidada mostró por qué no conviene "
    "ajustar hiperparámetros con tan pocos datos.",
    "**Impacto real.** La log-loss baja de 0,407 a 0,304 en selección y de 0,524 a 0,380 en la prueba 2026. Frente a la "
    "persistencia, el modelo es mejor en calidad probabilística (selección, territorios nuevos y 2026), con intervalos que "
    "aún tocan el cero.",
    "**Límites.** Anticipa 2 de 13 transiciones. No supera a una logística con las mismas variables. Los efectos fijos "
    "territoriales (embedding) contienen información que las variables no explican.",
], 1):
    viñeta(txt, numero=k)
tabla("Tabla 7: Recomendaciones priorizadas para la siguiente iteración", [
    ["#", "Recomendación", "Evidencia", "Acción concreta"],
    ["R1", "Migrar a municipio (M3)", "Generalización espacial resuelta (§6)", "Acceso académico al registro por evento de ACLED; el pipeline solo cambia `datos.py`"],
    ["R2", "x₃ a escala municipal", "El conteo anual por departamento empeoró (I3)", "Medir entradas y salidas de grupos por municipio, no el conteo anual"],
    ["R3", "Señales adelantadas de transición", "11 de 19 errores son transiciones", "Variables mensuales del último trimestre, ceses al fuego y negociaciones"],
    ["R4", "Reentrenar cada año y vigilar el PSI", "PSI de m₁ = 0,29; prevalencia 41-62 %", "Ejecutar el pipeline al cierre de cada año; decidir con bandas, no con un umbral fijo"],
    ["R5", "Revisar logaritmos y MIN_EVENTOS", "Controles de §2.2", "Repetir la batería de controles con el dataset municipal"],
    ["R7", "Reevaluar el embedding con 2026 cerrado", "Mejor en la prueba 2026 (0,314)", "Candidato en la selección 2024-2026; a escala municipal, reemplazarlo por x₃ y distancia"],
], [0.8, 4.2, 5.0, 7.4], tam=7.9)

# =====================================================================================================================
h1("10.   Trabajo en equipo y organización")
p("Mantuvimos la división de las etapas anteriores y la llevamos a los cuatro roles que pide la guía. Las decisiones "
  "metodológicas se tomaron en conjunto y quedaron en la bitácora del cuaderno (D11–D20, con alternativas y evidencia).")
tabla("Tabla 8: Roles técnicos de la Unidad 3", [
    ["Rol", "Responsable", "Entregables"],
    ["Modelado", "Juan Diego Saldarriaga", "Paquete perfil_amenaza, intervenciones I1–I9, rejilla y validación anidada, pipeline y pruebas"],
    ["Evaluación", "Conjunta (cálculo: Juan Diego; interpretación: Laura)", "Protocolo y regla de retención, controles, antes vs. después, análisis de errores"],
    ["Documentación", "Laura Díaz Sastoque", "Interpretaciones, tarjeta del modelo, informe y README"],
    ["Presentación", "Laura (guion) · Juan Diego (demostración)", "Video ejecutivo de 8 a 10 minutos y sustentación"],
], [2.6, 5.4, 9.4])

# =====================================================================================================================
h1("11.   Referencias bibliográficas en APA 7")
p("Nota: declaramos el uso de inteligencia artificial (Claude, de Anthropic) como apoyo en la programación del pipeline y "
  "del cuaderno. La propuesta del proyecto, la búsqueda de datos, el análisis, las decisiones metodológicas y las "
  "conclusiones son del equipo, que ejecutó, revisó y verificó todos los resultados.", tam=9.5)
REFS = [
    "Bergmeir, C., & Benítez, J. M. (2012). On the use of cross-validation for time series predictor evaluation. *Information Sciences, 191*, 192–213. https://doi.org/10.1016/j.ins.2011.12.028",
    "Cawley, G. C., & Talbot, N. L. C. (2010). On over-fitting in model selection and subsequent selection bias in performance evaluation. *Journal of Machine Learning Research, 11*, 2079–2107.",
    "Departamento Administrativo Nacional de Estadística (DANE). (2019). *Pobreza multidimensional por departamentos 2018: Resultados*. DANE.",
    "Dietterich, T. G. (1998). Approximate statistical tests for comparing supervised classification learning algorithms. *Neural Computation, 10*(7), 1895–1923. https://doi.org/10.1162/089976698300017197",
    "Géron, A. (2017). *Hands-on machine learning with Scikit-Learn and TensorFlow*. O'Reilly Media.",
    "Gneiting, T., & Raftery, A. E. (2007). Strictly proper scoring rules, prediction, and estimation. *Journal of the American Statistical Association, 102*(477), 359–378. https://doi.org/10.1198/016214506000001437",
    "Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep learning*. MIT Press.",
    "Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On calibration of modern neural networks. En *Proceedings of the 34th International Conference on Machine Learning* (pp. 1321–1330).",
    "Hansen, M. C., Potapov, P. V., Moore, R., Hancher, M., Turubanova, S. A., Tyukavina, A., … Townshend, J. R. G. (2013). High-resolution global maps of 21st-century forest cover change. *Science, 342*(6160), 850–853. https://doi.org/10.1126/science.1244693",
    "Kingma, D. P., & Ba, J. (2015). Adam: A method for stochastic optimization. En *International Conference on Learning Representations*. https://arxiv.org/abs/1412.6980",
    "Lakshminarayanan, B., Pritzel, A., & Blundell, C. (2017). Simple and scalable predictive uncertainty estimation using deep ensembles. En *Advances in Neural Information Processing Systems 30*.",
    "Ministerio de Defensa de Colombia. (2025). *Informe de gestión: modernización y transformación de las Fuerzas Militares*. Ministerio de Defensa Nacional.",
    "Ministerio de Justicia y del Derecho – Observatorio de Drogas de Colombia. (2026). *Detección de cultivos de coca (hectáreas)* [Conjunto de datos]. Datos Abiertos Colombia. https://www.datos.gov.co/d/acs4-3wgp",
    "Mitchell, M., Wu, S., Zaldivar, A., Barnes, P., Vasserman, L., Hutchinson, B., … Gebru, T. (2019). Model cards for model reporting. En *Proceedings of the Conference on Fairness, Accountability, and Transparency* (pp. 220–229). https://doi.org/10.1145/3287560.3287596",
    "Platt, J. (1999). Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods. En A. Smola et al. (Eds.), *Advances in large margin classifiers* (pp. 61–74). MIT Press.",
    "Raleigh, C., Linke, A., Hegre, H., & Karlsen, J. (2010). Introducing ACLED: An armed conflict location and event dataset. *Journal of Peace Research, 47*(5), 651–660. https://doi.org/10.1177/0022343310378914",
    "Roberts, D. R., Bahn, V., Ciuti, S., Boyce, M. S., Elith, J., Guillera-Arroita, G., … Dormann, C. F. (2017). Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure. *Ecography, 40*(8), 913–929. https://doi.org/10.1111/ecog.02881",
    "Sundberg, R., & Melander, E. (2013). Introducing the UCDP Georeferenced Event Dataset. *Journal of Peace Research, 50*(4), 523–532. https://doi.org/10.1177/0022343313484347",
]
for ref in REFS:
    par = doc.add_paragraph(); par.alignment = WD_ALIGN_PARAGRAPH.LEFT
    par.paragraph_format.left_indent = Cm(0.9); par.paragraph_format.first_line_indent = Cm(-0.9); par.paragraph_format.space_after = Pt(1)
    runs(par, ref, tam=9)

doc.save(SALIDA)
print("Word:", SALIDA)
r = subprocess.run(["soffice", "--headless", "--convert-to", "pdf", "--outdir", str(SALIDA.parent), str(SALIDA)], capture_output=True, text=True)
print(r.stdout.strip() or r.stderr.strip())
