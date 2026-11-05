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


def recuadro(texto, tam=8.6):
    """Recuadro de texto normal (notación y siglas)."""
    t = doc.add_table(rows=1, cols=1); _bordes(t)
    par = t.cell(0, 0).paragraphs[0]; par.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    par.paragraph_format.space_after = Pt(0)
    runs(par, texto, tam=tam)
    doc.add_paragraph().paragraph_format.space_after = Pt(0)


def hipervinculo(par, url, texto, tam=10):
    """Hipervínculo real (w:hyperlink), para que la URL sea clicable en el PDF."""
    r_id = par.part.relate_to(url, "http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink", is_external=True)
    h = OxmlElement("w:hyperlink"); h.set(qn("r:id"), r_id)
    r = OxmlElement("w:r"); rPr = OxmlElement("w:rPr")
    fuentes = OxmlElement("w:rFonts"); fuentes.set(qn("w:ascii"), FUENTE); fuentes.set(qn("w:hAnsi"), FUENTE)
    color = OxmlElement("w:color"); color.set(qn("w:val"), "1F6FB5")
    sz = OxmlElement("w:sz"); sz.set(qn("w:val"), str(int(tam * 2)))
    u = OxmlElement("w:u"); u.set(qn("w:val"), "single")
    for e in (fuentes, color, sz, u):
        rPr.append(e)
    r.append(rPr); t = OxmlElement("w:t"); t.text = texto; r.append(t); h.append(r)
    par._p.append(h)


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
# Portada (encabezado del informe) y metadatos
# =====================================================================================================================
import pandas as pd
SAL = RAIZ / "salidas_u3"
URL_REPO = "https://github.com/salda824/perfil-amenaza-tecnologica-u3"
doc.core_properties.title = "Modelo neuronal integrado para anticipar el perfil de amenaza tecnológica de los territorios colombianos"
doc.core_properties.author = "Laura Alejandra Díaz Sastoque y Juan Diego Saldarriaga Muñoz"
doc.core_properties.subject = "Proyecto Integrador, Unidad 3 · Redes Neuronales · Universidad de La Sabana · noviembre de 2026"
doc.core_properties.keywords = "redes neuronales; perfil de amenaza; ACLED; Colombia"

par = doc.add_paragraph(); par.alignment = WD_ALIGN_PARAGRAPH.CENTER; par.paragraph_format.space_before = Pt(8)
runs(par, "Modelo neuronal integrado para anticipar el perfil de amenaza tecnológica de los territorios colombianos y orientar la adopción de tecnologías militares", tam=13.5, negrita=True)
for txt in ("Redes Neuronales · Universidad de La Sabana · noviembre de 2026 · Sector elegido: seguridad y defensa",
            "Proyecto Integrador, Etapa 3 · Laura Alejandra Díaz Sastoque (0000288870) y Juan Diego Saldarriaga Muñoz (0000287391)"):
    q = doc.add_paragraph(); q.alignment = WD_ALIGN_PARAGRAPH.CENTER; q.paragraph_format.space_after = Pt(0)
    r = q.add_run(txt); r.font.size = Pt(10); r.font.color.rgb = RGBColor(0x59, 0x59, 0x59); r.font.name = FUENTE
q = doc.add_paragraph(); q.alignment = WD_ALIGN_PARAGRAPH.CENTER; q.paragraph_format.space_after = Pt(0)
r = q.add_run("Código: "); r.font.size = Pt(10); r.font.color.rgb = RGBColor(0x59, 0x59, 0x59); r.font.name = FUENTE
hipervinculo(q, URL_REPO, URL_REPO)
q = doc.add_paragraph(); q.alignment = WD_ALIGN_PARAGRAPH.CENTER; q.paragraph_format.space_after = Pt(0)
r = q.add_run("Repositorio privado por la licencia académica de ACLED; se da acceso al profesor como colaborador.")
r.font.size = Pt(9); r.italic = True; r.font.color.rgb = RGBColor(0x59, 0x59, 0x59); r.font.name = FUENTE

# =====================================================================================================================
h1("1.   Planteamiento del problema y contexto")
p("Colombia ejecuta en 2026 la mayor inversión de modernización militar de su historia sin un modelo basado en datos que "
  "indique qué tecnología desplegar en cada territorio (Ministerio de Defensa de Colombia, 2025). En la Etapa 1 (U1) "
  "formulamos la pregunta orientadora:")
caja("¿Cómo orientar la adquisición, transferencia de tecnología y fabricación de sistemas de seguridad y defensa de las "
     "Fuerzas de Colombia en función del perfil de amenaza tecnológica predominante de los municipios colombianos según las "
     "variables de actividad armada histórica, geografía operacional y economías ilícitas, de modo que se optimice la "
     "inversión de recursos en búsqueda del bienestar de los colombianos y el desarrollo del país?")
p("La pregunta habla de municipios, pero el export gratuito de ACLED solo desagrega a departamento (hallazgo de la Etapa 1). "
  "Por eso la respondemos a esa escala y dejamos la migración a municipio como primera recomendación (R1). La formulación es "
  "la misma de la U2. Para cada departamento *d* y año *t*, la red estima con lo observado hasta el cierre de *t*−1 la "
  "probabilidad de que el perfil sea C (armas convencionales y emboscadas) o D (amenaza mixta): s^{C}_{d,t} = eventos C / "
  "eventos totales, y_{d,t} = 1[s^{C}_{d,t} ≥ 0,5] (al menos la mitad de los eventos son C, igual que en la U2) y "
  "p̂_{d,t} = f_{θ}(x_{d,t−1}). Usamos el año anterior por dos motivos: las fatalidades de *t* salen de los mismos ataques "
  "que definen el perfil C, y el presupuesto de un año se asigna con lo observado hasta el cierre del anterior. La Tabla 1 "
  "resume lo que dejó cada etapa.")
tabla("Tabla 1: Trayectoria del proyecto y hallazgo que condiciona cada etapa", [
    ["Etapa", "Resultado", "Hallazgo que condiciona la siguiente"],
    ["U1 · Formulación", "Perfiles A/B/C/D con la regla del 50 %, 7 variables, ACLED agregado a departamento",
     "Ningún departamento tiene perfil A o B dominante, así que la clasificación es C vs. D; 33 filas no bastan, así que se arma un panel departamento–año"],
    ["U2 · Modelo base", "MLP 7 → 16 → 1 (145 parámetros); AUC temporal 0,89; empata con la logística",
     "Repetir el perfil del año anterior (persistencia: AUC 0,965 y log-loss 0,380 en la validación 2023–2026 de la U2) supera a la red; agregar ese perfil como variable (experimento E1) fue la mejora más grande"],
    ["U3 · Integración", "10 frentes de intervención, controles de datos, pipeline reproducible, prueba 2026 única",
     "La referencia a superar pasa a ser la persistencia, cruda y calibrada; un modelo aporta sobre ella cuando anticipa cambios de perfil"],
], [2.8, 6.6, 8.0])
recuadro("Notación y siglas. ACLED: Armed Conflict Location & Event Data. UCDP GED: Uppsala Conflict Data Program, "
         "Georeferenced Event Dataset. SIMCI: Sistema Integrado de Monitoreo de Cultivos Ilícitos. GFW: Global Forest Watch. "
         "IGAC: Instituto Geográfico Agustín Codazzi. IPM: índice de pobreza multidimensional del DANE. ISR: inteligencia, "
         "vigilancia y reconocimiento. C-UAS: sistemas contra aeronaves no tripuladas. AEI: artefactos explosivos improvisados. "
         "Perfiles según el mapeo de subtipos de ACLED de la U1: A agrupa IED/landmine, Grenade, Suicide bomb y Chemical weapon; "
         "B, Air/drone strike; C, Armed clash, Attack, Shooting, Abduction, Looting/property destruction y Disrupted weapons use; "
         "D, protestas, disturbios, arrestos y el resto de subtipos. Un territorio D es, por tanto, uno donde la violencia armada "
         "directa no llega a la mitad de los eventos. Transición: cambio de perfil entre *t*−1 y *t*. Log-loss o BCE: entropía "
         "cruzada binaria media, −media[y log p̂ + (1 − y) log(1 − p̂)]. Brier: error cuadrático medio de p̂. ECE: error de "
         "calibración esperado con 5 grupos de igual tamaño. MCC: coeficiente de correlación de Matthews. AUC: área bajo la "
         "curva ROC. EE: error estándar. IC95: intervalo de confianza del 95 %. PSI: índice de estabilidad poblacional. "
         "I1–I9: intervenciones de la U3 (Tabla 5). Dk: decisiones numeradas de la bitácora del cuaderno (D1–D10 de la U2, "
         "D11–D20 de la U3).")

# =====================================================================================================================
h1("2.   Descripción del dataset y preprocesamiento")
h2("2.1 Fuentes, panel y variables")
p("La base es el export semanal de ACLED para Colombia (ACLED, 2026; Raleigh et al., 2010): 17.573 registros semana × "
  "departamento × subtipo de evento, hasta el 15 de agosto de 2026. Le sumamos la coca del censo SIMCI (Ministerio de "
  "Justicia y del Derecho, Observatorio de Drogas de Colombia, 2026), la cobertura arbórea de GFW (Hansen et al., 2013; "
  "World Resources Institute, 2025), el IPM del DANE (2019) y, en esta etapa, UCDP GED 26.1 (Sundberg y Melander, 2013; "
  "Uppsala Conflict Data Program, 2026) para contar actores armados. El panel tiene 171 filas departamento–año etiquetables: "
  "30 departamentos, años 2021–2026 y al menos 10 eventos por celda (Guainía, San Andrés y Vaupés no tienen ninguna celda "
  "con 10 eventos). Hay 82 filas C (48 %) y 89 D; la prevalencia de C va de 32 % en 2021 a 62 % en 2025 y es 48 % en 2026. "
  "El código de la U3 reconstruye el dataset de la U2 con las mismas etiquetas y una diferencia máxima de 1,7×10^{−13} en "
  "las 7 predictoras, así que cualquier cambio posterior en las métricas viene de las intervenciones. La Tabla 2 muestra "
  "el estado de cada variable al cierre de la U3.")
tabla("Tabla 2: Variables del modelo (todas medidas en t−1) y su estado al cierre de la U3", [
    ["ID", "Variable", "Operacionalización", "Fuente", "Estado U3"],
    ["x_{1}, x_{2}, x_{2b}", "Eventos, fatalidades, letalidad", "log(1 + anualizado); fatalidades/evento", "ACLED", "En el modelo (U2)"],
    ["x_{4}–x_{7}", "Coca, bosque, distancia, IPM", "log(1 + ha); % de dosel ≥ 30 %; km del centroide a Bogotá (proxy de presencia estatal); IPM 2018 (%)", "SIMCI, GFW, centroides ACLED, DANE", "En el modelo (U2)"],
    ["m_{1}", "Memoria del perfil", "s̃^{C} contraída del año t−1", "ACLED", "Nueva, retenida (I1)"],
    ["d_{1}", "Semestre reciente", "s̃^{C} de las últimas 26 semanas de t−1", "ACLED semanal", "Nueva, retenida (I2)"],
    ["m_{2}, m_{3}; d_{2}, d_{3}", "Trayectoria y tendencias", "s̃^{C} en t−2 y su cambio; tendencias semestrales", "ACLED", "Evaluadas, no retenidas"],
    ["x_{3}", "Grupos armados activos", "log(1 + actores no estatales distintos)", "UCDP GED 26.1", "Evaluada, no retenida (I3)"],
    ["v_{1}, v_{2}", "Vecindad", "s̃^{C} y eventos de los departamentos contiguos", "ACLED + IGAC", "Evaluada, no retenida (I4)"],
], [2.6, 3.4, 5.6, 3.0, 2.8])
p("Las proporciones nuevas usan contracción bayesiana, s̃^{C} = (C + 2)/(n + 4), que equivale a un prior Beta(2, 2): un "
  "semestre con pocos eventos queda cerca de 0,5 en vez de saltar a 0 o 1, y con 100 eventos el ajuste no pasa de 2 puntos. "
  "Todas las variables se estandarizan, z = (x − media)/desviación, con la media y la desviación del entrenamiento de cada "
  "pliegue. Preferimos estandarizar a normalizar al rango [0, 1] porque con 113–142 filas un solo valor extremo nuevo (la "
  "coca de Nariño, por ejemplo) movería el mínimo o el máximo y desplazaría toda la escala. Los conteos y las hectáreas "
  "conservan el logaritmo de la U2. La coca y el bosque del año *t*−1 se publican durante el año *t*; como cambian lento, "
  "tomamos ese rezago de publicación como un límite menor. La Figura 1a muestra la razón para agregar d₁: en el 74 % de "
  "las 19 transiciones de 2021–2025, el segundo semestre del año anterior ya se movía hacia el perfil nuevo.")
p("La prueba de 2026 tiene solo 33 semanas. Recalculamos la etiqueta de 2021–2025 con las primeras 33 semanas de cada año: "
  "coincide con la del año completo en el 94,9 % de 136 celdas, y las 7 discrepancias tienen s^{C} a menos de 10 puntos del "
  "umbral (Figura 1b). Para 2026 esperamos entonces 1 o 2 etiquetas distintas de las del año cerrado, todas en la franja "
  "ambigua.")
figura("R01_datos", 16.2, "Figura 1: (a) En el 74 % de las 19 transiciones de 2021–2025 (sin la prueba 2026), el segundo semestre del año anterior "
       "(d₁) ya se movía hacia el nuevo perfil. (b) La etiqueta a la semana 33 coincide con la anual en el 94,9 % de las celdas 2021–2025")
h2("2.2 Controles experimentales de la preparación de datos")
p("Pusimos a prueba cada decisión de preprocesamiento quitándola, o metiendo a propósito un error conocido, con el mismo "
  "protocolo y las mismas 5 semillas (Tabla 3, Figura 2).")
tabla("Tabla 3: Controles de preparación de datos (Δ log-loss frente a la referencia de 0,310 y EE entre paréntesis; 5 semillas, origen móvil 2023–2025)", [
    ["Control", "Δ log-loss (EE)", "Lectura"],
    ["Sin estandarización", "+5,780 (1,445)", "La red no converge: distancias y porcentajes saturan ReLU y sigmoide, y la BCE de entrenamiento se queda en 7,1"],
    ["Etiquetas barajadas (control negativo)", "+0,506 (0,012)", "AUC 0,32: con etiquetas al azar no hay señal que aprender, así que el desempeño medido viene de los datos"],
    ["Sin pesos de clase", "+0,015 (0,002)", "En entrenamiento la prevalencia de C es 44 % y en los años validados llega a 62 %; sin pesos, la red se inclina hacia D"],
    ["Mapeo alternativo de subtipos", "+0,007 (0,001)", "La regla de la U1 da mejor resultado"],
    ["Sin contracción (α ≈ 0)", "+0,002 (0,001)", "Aporte pequeño, en la dirección esperada"],
    ["Escalador con fuga (fuga deliberada)", "−0,002 (0,001)", "Mide la mejora artificial que daría una fuga; las pruebas automáticas la detectarían"],
    ["Sin logaritmos", "−0,012 (0,003)", "No lo esperábamos: sale levemente mejor. No se cambia en esta iteración (regla 7 de §4.1); en el pliegue 2025 es peor"],
    ["MIN_EVENTOS (umbral de eventos por celda) = 5 / 10 / 20", "LL 0,300 / 0,307 / 0,338 (absolutas)", "Log-loss en las mismas filas (≥ 20 eventos): 5 y 10 empatan, 20 empeora; se mantiene 10"],
], [4.6, 2.8, 10.0], centrar={1})
figura("R05_controles", 16.2, "Figura 2: (a) Curvas de entrenamiento (—) y de validación 2025 (···) con y sin estandarización y logaritmos (semilla 1; "
       "la versión sin estandarizar no baja de 7,1). (b) Efecto de cada control sobre la log-loss fuera de muestra (5 semillas)")

# =====================================================================================================================
h1("3.   Arquitectura del modelo y justificación")
lado_a_lado([
    "El modelo final es un ensamble de 10 perceptrones multicapa con la misma estructura (API funcional de Keras): 9 "
    "entradas estandarizadas, una capa oculta de 16 neuronas ReLU y una salida sigmoide, con 177 parámetros por red. Son 1,25 "
    "parámetros por fila de entrenamiento (142 filas en 2021–2025); hay más parámetros que datos, y la L2 y el presupuesto "
    "fijo de pasos controlan el sobreajuste. Cada red se entrena con una semilla distinta y la predicción es el promedio de "
    "las 10 probabilidades. Frente a la U2, la arquitectura solo gana dos entradas, m₁ y d₁ (en verde en la Figura 3). La "
    "Tabla 4 resume la configuración.",
    "También probamos una GRU y una CNN 1D sobre los cuatro trimestres de *t*−1 (log-loss 0,319 y 0,324, frente a 0,310), "
    "un MLP más profundo (16, 8; 0,309), un embedding de departamento, que codifica cada departamento como un entero y aprende "
    "un vector de 2 dimensiones (0,284; §6.6), y la regresión logística con las mismas variables (0,306). Las redes "
    "secuenciales tienen entre 393 y 665 parámetros y no encuentran en los trimestres más información de la que ya resume "
    "d₁, así que pierden por varianza.",
], "R02_arquitectura", 6.8, "Figura 3: Arquitectura final (9 → 16 → 1, 177 parámetros por red; verde = variables nuevas)", ancho_texto=10.4)
tabla("Tabla 4: Configuración del modelo final y su justificación", [
    ["Componente", "Elección", "Justificación técnica"],
    ["Entrada", "9 variables en t−1, estandarizadas con el entrenamiento", "Las 7 de la U2 más m₁ y d₁, las únicas que pasaron la regla de retención (§5)"],
    ["Capa oculta", "16 neuronas ReLU, He normal", "La rejilla de 48 configuraciones fue plana y ninguna alternativa elegida con 2023–2024 mejoró en 2025 (§5)"],
    ["Salida y pérdida", "Sigmoide; BCE ponderada por clase + L2 (λ = 10⁻³)", "Log-verosimilitud de Bernoulli (Goodfellow et al., 2016); sin pesos de clase la log-loss sube 0,015"],
    ["Optimizador", "Adam (Kingma y Ba, 2015), η = 10⁻³, lote 16, 560 pasos fijos", "Presupuesto mediano de la U2 (decisión D10, elegido con parada temprana sobre 2025); las curvas muestran que sigue cerca del óptimo (§4.2)"],
    ["Ensamble", "10 semillas, promedio de probabilidades", "Reduce unas 6 veces la dispersión entre semillas (Lakshminarayanan et al., 2017)"],
    ["Calibración y umbral", "Sin recalibración; umbral 0,5 y bandas 0,3/0,7", "Platt (1999) y temperatura (Guo et al., 2017) no mejoraron sin mirar el futuro, y el umbral óptimo cambia cada año"],
], [3.0, 6.0, 8.4])

# =====================================================================================================================
h1("4.   Proceso de entrenamiento y evaluación")
h2("4.1 Protocolo de evaluación")
p("Las reglas 1 a 7 se fijaron antes de ejecutar las intervenciones. La regla 8 la agregamos durante ellas, al encontrar el "
  "caso del embedding de departamento (§6.6), y lo declaramos. La persistencia calibrada de la regla 3 se agregó después de "
  "la revisión del informe y no cambió el modelo adoptado.")
for k, txt in enumerate([
    "Selección con validación de origen móvil (Bergmeir y Benítez, 2012): para *v* = 2023, 2024 y 2025 entrenamos con *t* < *v* "
    "y predecimos *v*, lo que da 87 predicciones fuera de muestra. El escalador se ajusta en cada pliegue solo con su entrenamiento.",
    "Prueba: 2026 se evalúa una sola vez, con las decisiones congeladas en `configuracion.json`. La U2 ya había reportado "
    "métricas sobre ese año, y lo dejamos dicho.",
    "Referencia: la persistencia, p̂_{d,t} = s^{C}_{d,t−1} (AUC 0,973; log-loss 0,363), que por construcción no acierta "
    "ninguna transición. Como usa una proporción como si fuera una probabilidad, la comparamos también con su versión "
    "calibrada: una logística de dos parámetros sobre logit(m₁), entrenada solo con los años anteriores al predicho.",
    "Métrica de decisión: la log-loss, que es una regla de puntuación propia (Gneiting y Raftery, 2007); AUC, Brier, ECE, "
    "MCC, exactitud y aciertos en transiciones sirven de apoyo.",
    "Retención: cada variante se entrena con las mismas 5 semillas que su referencia y se queda si la mejora media de la "
    "log-loss supera un EE de la diferencia pareada, sin bajar el AUC más de 0,01. El EE es entre semillas, así que mide el "
    "ruido del entrenamiento y no el del muestreo de departamentos. En I4, cuya hipótesis es espacial, la regla se aplica a "
    "la validación espacial y se exige no empeorar la temporal.",
    "Presupuesto fijo de 560 pasos de gradiente (D10), salvo cuando el presupuesto es lo que se estudia.",
    "Los controles de datos de la §2.2 solo verifican decisiones heredadas: si uno favorece quitar una transformación, se "
    "reporta y queda para la siguiente iteración.",
    "Una variante que mejora la log-loss solo porque se parece más a la persistencia (acierta menos transiciones) o porque "
    "no puede aplicarse a territorios nuevos queda documentada y no se adopta.",
], 1):
    viñeta(txt, numero=k)
p("Las comparaciones pareadas usan bootstrap por departamento (2.000 remuestreos de departamentos completos) y la prueba de "
  "McNemar sobre los aciertos con umbral 0,5 (Dietterich, 1998).")
h2("4.2 Entrenamiento y curvas de desempeño")
p("Cada red se entrena con el bucle explícito de tf.GradientTape de la U2, verificado allí contra model.fit (diferencia máxima "
  "de 2×10^{−6}; Géron, 2017). En cada paso, la red calcula p̂ para un lote de 16 filas (propagación hacia adelante), evalúa "
  "la pérdida L(θ) = −(1/n) Σ_{i} ω(y_{i})[y_{i} log p̂_{i} + (1 − y_{i}) log(1 − p̂_{i})] + λ‖W‖², con ω(c) = n/(2n_{c}) y "
  "λ = 10⁻³, obtiene ∂L/∂θ por retropropagación y Adam actualiza pesos y sesgos con η = 10⁻³ (franja inferior de la "
  "Figura 8). Con 113 filas de entrenamiento, 560 pasos son 70 épocas. En la U3 el bucle acepta además objetivo blando, "
  "salida multitarea y entradas secuenciales. La Figura 4 compara las curvas del modelo base y del final, entrenando con "
  "2021–2024 y validando en 2025, con 5 semillas y tres veces el presupuesto. En el modelo final la BCE de validación en el "
  "paso 560 es 0,377 y su mínimo es 0,360, así que el presupuesto sigue cerca del óptimo; entrenar el triple la sube a 0,441 "
  "por sobreajuste. El modelo final baja tanto la BCE de entrenamiento (0,341 → 0,279) como la de validación "
  "(0,448 → 0,377), y la brecha entre ambas pasa de 0,107 a 0,097. Como la capacidad es la misma, la mejora viene de la "
  "información de entrada.")
figura("R03_curvas", 16.2, "Figura 4: Curvas de aprendizaje con validación 2025 (media y rango de 5 semillas). Línea discontinua: presupuesto de 560 pasos (70 épocas)")

# =====================================================================================================================
h1("5.   Optimización realizada y comparación de resultados")
p("Cada intervención responde a un hallazgo medido en la U2 y tiene una hipótesis que podía fallar (Tabla 5). Las de "
  "variables se aplicaron en cascada, cada una sobre la mejor especificación retenida hasta ese momento; las de modelo, sobre "
  "el conjunto de variables ya fijado. I9 tiene el último número porque la planteamos después de I1–I8, pero se evaluó junto "
  "con las de modelo.")
tabla("Tabla 5: Del diagnóstico de la U2 a la decisión (Δ log-loss media entre 5 semillas pareadas, EE entre paréntesis; LL = log-loss)", [
    ["#", "Diagnóstico U2", "Intervención", "Resultado (2023–2025)", "Decisión"],
    ["I1", "La inercia domina (persistencia AUC 0,965 en la U2)", "Memoria m₁ (y m₂, m₃)", "m₁: −0,036 (0,009); ΔAUC +0,017. m₂, m₃: +0,002", "Retiene m₁"],
    ["I2", "Solo vista anual; escaladas de 2025", "Semestre reciente d₁ (y d₂, d₃)", "d₁: −0,037 (0,007); ΔAUC +0,013. d₂, d₃: +0,013", "Retiene d₁"],
    ["I3", "Arauca mal clasificada; falta x₃", "Actores armados (UCDP)", "+0,025 (0,006); ΔAUC −0,014", "No retiene"],
    ["I4", "Validación espacial débil", "Vecindad v₁, v₂", "Espacial +0,009 (0,004); temporal −0,004 (0,002)", "No retiene (criterio espacial)"],
    ["I5", "20 % de celdas cerca del umbral", "Objetivo blando s^{C}", "+0,119 sin calibrar; +0,068 calibrado", "No retiene"],
    ["I9", "A/B diluidas en la etiqueta", "Salida multitarea A/B/C/D", "P(C): +0,001; A/B peor que la persistencia", "No retiene"],
    ["–", "MLP sin estructura temporal", "GRU, CNN 1D, profundo, embedding, logística", "+0,010; +0,014; 0,000; −0,026; −0,003 (§6)", "No adopta (embedding: regla 8)"],
    ["I6", "Capacidad inestable entre análisis", "Rejilla de 48 config. + anidada", "Lo elegido en 2023–2024 empeora 2025 (0,41–0,42 vs. 0,36; ver texto)", "Mantiene U2"],
    ["I7", "AUC 0,848 ± 0,024 entre semillas", "Ensamble de 10 redes", "LL −0,011 sin EE; banda 5–95 % de 0,053 a 0,009", "Retiene (estabilidad)"],
    ["I8", "Subestima P(C); umbral no transfiere", "Platt / temperatura; umbral", "LL 0,298 → 0,302 en 2024–2025 (Platt hacia adelante)", "No retiene"],
], [0.8, 3.8, 3.8, 5.8, 2.6], tam=7.8)
p("La rejilla de capacidad (4–32 neuronas), L2, activación (ReLU, tanh) y presupuesto (280–1.120 pasos) es plana (I6): 31 de "
  "48 configuraciones quedan a menos de un EE de la mejor, y la vigente (puesto 18) está en esa banda. Para ver si elegir en "
  "la rejilla ayudaría, seleccionamos con 2023–2024 y medimos en 2025, una validación anidada de un solo pliegue externo. La "
  "mejor de 2023–2024 obtiene 0,423 y la elegida por la regla de 1 EE obtiene 0,411; la vigente obtiene 0,359 (Figura 5b; "
  "media de 3 semillas, frente a 0,377 con las 5 de la Figura 4). "
  "La comparación favorece a la vigente, porque su capacidad y su presupuesto se eligieron en la U2 con una validación que "
  "incluía 2025. Por eso no la leemos como prueba de que la vigente sea mejor: con 58 filas, la rejilla no ofrece una "
  "alternativa que se sostenga al año siguiente (Cawley y Talbot, 2010), y mantenemos la configuración de la U2 por "
  "parsimonia. En la calibración (I8), el parámetro de escala de Platt vale 1,31 en 2023 y 3,38 en 2024, y en 2025 no "
  "converge porque la red separa perfectamente las clases ese año (AUC = 1,0), aunque con umbral 0,5 clasifica mal seis "
  "escaladas. El umbral que maximiza el MCC es 0,20, 0,51 o 0,15 según el año. Por eso la incertidumbre se maneja en la "
  "decisión con las bandas 0,3/0,7 de la Etapa 2 y el umbral se deja en 0,5.")
figura("R04_optimizacion", 16.2, "Figura 5: (a) Escalera de mejora: log-loss de una red individual (media ± DE de 5 semillas) al agregar m₁ y d₁, y del "
       "ensamble de 10; el modelo entregado en la U2 (semilla 42) tenía 0,407. (b) Validación anidada: lo que se elige con 2023–2024 es "
       "peor en 2025 que la configuración vigente, elegida en la U2 con una validación que incluía 2025")

# Tabla 6 generada desde las salidas del cuaderno (antes_despues_*.csv)
def _f(v, d=3):
    return "–" if pd.isna(v) else f"{v:.{d}f}".replace(".", ",")
FILAS6 = [("Persistencia", "Persistencia"), ("Persistencia calibrada*", "Persistencia calibrada"),
          ("Antes: base U2, semilla 42", "Antes: MLP base U2 (semilla 42"), ("Antes: base U2, ensamble de 10", "Antes: MLP base U2 (ensamble"),
          ("Logística (mismas variables)", "Logística"), ("Después: MLP U3 (ensamble de 10)", "Después: MLP U3")]
filas6 = [["Conjunto", "Modelo", "AUC", "Log-loss", "Brier", "ECE", "MCC", "Exactitud", "Sens. C / Espec. D", "Transiciones"]]
for conj, arch in [("Origen móvil 2023–2025 (n = 87)", "antes_despues_origen_movil.csv"), ("Prueba 2026 (n = 29), una sola vez", "antes_despues_prueba_2026.csv")]:
    t6 = pd.read_csv(SAL / arch, index_col=0)
    for k, (etq, clave) in enumerate(FILAS6):
        m = t6.loc[next(i for i in t6.index if i.startswith(clave))]
        filas6.append([conj if k == 0 else "", etq, _f(m["ROC-AUC"]), _f(m["Log-loss"]), _f(m["Brier"]), _f(m["ECE"]), _f(m["MCC"]),
                       _f(m["Exactitud"]), f"{_f(m['Sensibilidad C'], 2)} / {_f(m['Especificidad D'], 2)}",
                       f"{int(m['Transiciones acertadas'])} de {int(m['Transiciones'])}"])
tabla("Tabla 6: Antes y después frente a los rivales (* rival agregado tras la revisión; no cambió el modelo adoptado)",
      filas6, [2.7, 4.3, 1.1, 1.3, 1.1, 1.1, 1.1, 1.4, 2.0, 1.6], tam=7.6, centrar_desde=2)
p("Frente al modelo de la U2 (una red, semilla 42, reentrenada con los 560 pasos fijos de la U3; la fase 2 de la U2 "
  "reportó 0,547 en 2026), la log-loss baja de 0,407 a 0,304 en selección "
  "(Δ −0,103; IC95 por departamento [−0,185; −0,033]) y de 0,524 a 0,380 en 2026 (Δ −0,144; IC95 [−0,270; −0,033]). La "
  "U2 había documentado que esa semilla quedó en la cola desfavorable, así que la comparamos también con la misma "
  "arquitectura base promediada sobre las mismas 10 semillas del modelo final (Tabla 6). Con ensamble en los dos lados, la "
  "log-loss baja de 0,376 a 0,304 en selección (Δ −0,072; IC95 [−0,114; −0,033]) y de 0,470 a 0,380 en 2026 (Δ −0,090; "
  "IC95 [−0,162; −0,024]); la mejora atribuible a m₁ y d₁ se mantiene con los dos criterios. Las cifras de selección salen "
  "de una cascada de unas 20 comparaciones sobre las mismas 87 predicciones y por eso son algo optimistas (Cawley y Talbot, "
  "2010); 2026 es la estimación que no pasó por la selección.")

# =====================================================================================================================
h1("6.   Análisis crítico del desempeño")
figura("R06_antes_despues", 16.2, "Figura 6: (a) Curvas ROC en 2023–2025. (b) Log-loss por año: el modelo final queda por debajo de la persistencia cruda "
       "en los cuatro años y de la calibrada en 2023–2025; en 2026 la calibrada es mejor. (c) Matriz de confusión del modelo final en la prueba 2026")
h2("6.1 Frente a la inercia")
p("Frente a la persistencia cruda, el modelo baja la log-loss en 0,059 en selección (IC95 [−0,142; +0,028]; gana en el 92 % "
  "de los remuestreos) y en 0,049 en 2026 (IC95 [−0,162; +0,092]). La persistencia conserva el mejor AUC (0,973 y 0,948), "
  "así que ordena mejor los departamentos, y con umbral 0,5 también clasifica mejor (exactitud 0,897 frente a 0,839 en "
  "selección; acierta 7 casos que el modelo falla y el modelo 2 que ella falla, McNemar p = 0,18). La ventaja del modelo "
  "está en la calibración: la persistencia usa s^{C} como si fuera una probabilidad (ECE 0,173). Calibrada con una logística "
  "entrenada solo con años anteriores, la inercia obtiene 0,328 en selección y 0,351 en 2026 (Tabla 6). El modelo la supera "
  "en selección (Δ −0,024; IC95 [−0,096; +0,052]) y queda por encima en 2026 (Δ +0,029; IC95 [−0,055; +0,125]). La "
  "información útil está en la composición reciente (m₁ y d₁): a escala departamental, ni la red ni las variables "
  "estructurales agregan una ventaja demostrable sobre una inercia bien calibrada.")
h2("6.2 Matriz de confusión y regla de decisión")
p("En 2026 el modelo clasifica bien 13 de 14 departamentos C y 11 de 15 D (Figura 6c; sensibilidad 0,93, especificidad "
  "0,73, exactitud 0,83). Los 4 falsos positivos son exactamente las 4 desescaladas C → D (Bolívar, Chocó, Magdalena y Valle "
  "del Cauca): el modelo sigue la inercia y sobrestima C cuando el conflicto baja, con el riesgo de asignar ISR y movilidad "
  "protegida donde el perfil ya cambió. El único falso negativo es Cesar (P = 0,45), que cae en la banda de transición y "
  "recibiría portafolio flexible. La decisión usa bandas y no el umbral 0,5, así que las evaluamos contra el perfil "
  "observado. En 2023–2025, 71 de 87 filas caen en una banda consolidada y 66 de ellas aciertan (31 de 32 en C y 35 de 39 "
  "en D); en 2026, 24 de 29, con 21 aciertos (10 de 13 en C y 11 de 11 en D). Con la persistencia cruda, 42 de las 87 "
  "filas quedan en transición y la banda informa poco.")
h2("6.3 Departamentos nunca vistos")
p("Con 5 grupos de departamentos (años ≤ 2025; Roberts et al., 2017), la log-loss del ensamble pasa de 0,445 en el modelo "
  "base a 0,351 en el final (0,476 → 0,373 si se compara red individual con red individual), por debajo de la persistencia "
  "(0,418). La U2 había reportado dejar un departamento fuera porque GroupKFold desempataba distinto según la versión de "
  "scikit-learn; desde la versión 1.6 el desempate es estable, y usamos 5 grupos por costo. Ese esquema, sin embargo, deja "
  "que la red aprenda de otros departamentos en el mismo año que predice, ventaja que la persistencia no tiene. Si además "
  "se entrena solo con años anteriores, que es el escenario real de un territorio nuevo al año siguiente, el modelo obtiene "
  "0,361, la persistencia 0,363 (Δ −0,002; IC95 [−0,114; +0,130]), la persistencia calibrada 0,332 (Δ +0,029; IC95 "
  "[−0,075; +0,154]) y la logística 0,416. Las variables de composición transfieren mucho mejor que las estructurales de la "
  "U2, pero la ventaja frente a la inercia en territorios nuevos no está demostrada: la persistencia calibrada es la mejor "
  "de los cinco modelos.")
h2("6.4 Errores sistemáticos")
p("De 19 errores en 2023–2026, el 58 % son transiciones (frente al 2 % entre los aciertos) y el 58 % tiene s^{C} a ±10 puntos "
  "del umbral (Figura 7a). El error estructural de la U2 se corrigió: Arauca recibe P(C) entre 0,77 y 0,85 en 2023–2026 y "
  "queda bien clasificada todos los años. Siguen fallando las escaladas D → C de 2025 (Córdoba, Magdalena, Cesar, Meta, "
  "Bolívar y Valle), las 4 desescaladas C → D de 2026, que ni el modelo final ni la persistencia aciertan (el modelo de la U2 "
  "acertó Magdalena, con P = 0,40), Huila, que es C en 2023 y 2024 y recibe 0,36 y 0,47, y Casanare, que tiene muy pocos "
  "eventos. En 2023–2026 el modelo final acierta 2 de 13 transiciones, el de la U2 3 (4 con ensamble), la persistencia "
  "calibrada 1 y la persistencia cruda ninguna: la mejora probabilística vino de los territorios estables y anticipar cambios "
  "de régimen sigue siendo la limitación principal.")
h2("6.5 Red neuronal frente a logística")
p("Con las mismas variables, la logística empata en el eje temporal (Δ −0,002 en selección y −0,003 en 2026). La red aprende "
  "una frontera curva entre memoria y letalidad (Figura 7b): con 0,6 fatalidades por evento basta una memoria de ≈ 0,45 "
  "para esperar C, y sin letalidad hace falta ≈ 0,7. Como m₁ y la letalidad están correlacionadas (ρ = 0,84), la dependencia "
  "parcial extrapola fuera de la zona con datos, y la leemos solo donde hay puntos. Mantenemos el MLP por dos razones, "
  "ninguna concluyente: en departamentos no vistos el ensamble obtiene menos log-loss que la logística (0,351 frente a 0,373 "
  "con 5 grupos; 0,361 frente a 0,416 si además se respeta el tiempo, con IC95 de la diferencia [−0,236; +0,067]), y su "
  "frontera curva coincide con la hipótesis de no linealidad de la Etapa 1. La logística queda como control permanente.")
h2("6.6 Embedding de departamento")
p("Cumplió la regla de retención (−0,026; EE 0,010), pero no lo adoptamos por la regla 8, que agregamos al ver este "
  "resultado: acertaba 0,8 de 9 transiciones (frente a 2,0) y en departamentos nuevos se reduce al MLP sin embedding (0,369 "
  "frente a 0,379 en el esquema espacial), así que su ganancia viene de efectos fijos memorizados. En 2026 obtuvo log-loss "
  "0,314 y AUC 0,933, mejor que el modelo adoptado. No cambiamos la decisión, porque elegir con la prueba la dejaría sin "
  "valor como estimación. El resultado indica que los efectos fijos de cada departamento contienen información que las "
  "variables actuales no capturan (R7).")
figura("R07_errores_pdp", 16.2, "Figura 7: (a) P(C) predicha frente a la proporción observada, 2023–2026 (rótulos: errores en transiciones, Casanare y Huila). "
       "(b) Dependencia parcial de P(C) entre memoria y letalidad (línea negra: P = 0,5)")

# =====================================================================================================================
h1("7.   Integración final del modelo en un pipeline estructurado")
p("En la U2 el cuaderno hacía todo el trabajo, y ese era el vacío principal frente a la guía. En la U3 el proceso está en el "
  "paquete `perfil_amenaza` del repositorio, con siete módulos: `config`, `datos`, `variables`, `modelo`, `evaluacion`, "
  "`pipeline` y `graficos`. El cuaderno importa el paquete, incluidas la regla de retención, la persistencia calibrada y "
  "las particiones de validación. Una sola orden ejecuta las 8 etapas de la Figura 8 en menos de un minuto, y cada etapa "
  "escribe un archivo que se puede revisar. La etapa 3 vuelve a evaluar la especificación congelada; la cascada de "
  "intervenciones se corrió en el cuaderno con la misma partición y la misma regla.")
figura("R08_pipeline", 15.5, "Figura 8: Pipeline integrado. Arriba, las 8 etapas y el archivo que deja cada una; las decisiones se congelan en "
       "configuracion.json antes de la prueba 2026. Abajo, el lazo de entrenamiento de cada red: error, gradiente y actualización de pesos")
viñeta("El archivo `perfil_amenaza_u3.keras` (152 KB) es el artefacto de despliegue: contiene la estandarización (una capa "
       "Normalization con las medias y desviaciones del entrenamiento), las 10 redes, el promedio y la capa de calibración. "
       "Recibe las 9 variables en su escala original y devuelve P(C) sin necesitar el cuaderno.")
viñeta("Las 10 pruebas automáticas (pytest) revisan el contrato de datos y su caso negativo, la ausencia de fuga temporal "
       "(multiplicar por 5 los eventos de 2024 no cambia ninguna predictora de las filas de 2024 y sí las de 2025), que el "
       "escalador y la capa Normalization del artefacto usen solo el entrenamiento, que la persistencia calibrada y la "
       "validación espacio-temporal no miren el año predicho, la regla de retención, el determinismo y la reproducción del "
       "dataset de la U2. Las 10 pasan.")
viñeta("Las log-loss del pipeline (0,304 en selección y 0,380 en prueba) coinciden al cuarto decimal con las del cuaderno. "
       "El repositorio incluye `ejecutar_todo.bat`, que reproduce todo en Windows con un solo comando.")
viñeta("La tarjeta del modelo (Mitchell et al., 2019) fija el uso previsto, los límites y el mantenimiento. El PSI compara la "
       "información de 2026 de los 30 departamentos del entrenamiento con las filas de entrenamiento (umbrales usuales: "
       "< 0,1 estable, 0,1–0,25 vigilar, > 0,25 deriva). Con 30 filas, el PSI tiene un piso de ruido alto: sin deriva, su percentil 95 está entre 0,21 y 0,39 según la "
       "variable. Solo la letalidad lo supera (PSI 0,45 frente a 0,37), así que la distribución de fatalidades por evento de 2026 "
       "difiere de la del entrenamiento. m₁ (0,23) y d₁ (0,17) caen en la franja de vigilancia, pero dentro del ruido de muestreo.")

# =====================================================================================================================
h1("8.   Del modelo a la decisión: priorización territorial 2027")
lado_a_lado([
    "Reentrenamos el ensamble con todas las etiquetas 2021–2026 y lo aplicamos, a través del artefacto, a la información de "
    "2026. Las bandas de la Etapa 2 asignan C consolidado si P ≥ 0,7, transición si 0,3 ≤ P < 0,7 y D consolidado si "
    "P < 0,3 (Figura 9). Quedan 9 departamentos con datos suficientes como C consolidado (Guaviare, Nariño, Cauca, Caquetá, "
    "Putumayo, Antioquia, Norte de Santander, Vichada y Cesar), más Amazonas y Vaupés, que tienen menos de 10 eventos en "
    "2026 y requieren revisión experta. Otros 11 quedan en transición (entre ellos San Andrés y Guainía, también con pocos "
    "datos) y 11 como D consolidado.",
    "Las capacidades siguen la Etapa 1: ISR, detección temprana y movilidad protegida donde domina C; portafolio flexible y "
    "monitoreo trimestral en transición; orden público y capacidades mixtas en D. La alerta A/B se activa cuando la "
    "proporción de eventos con explosivos o drones de 2025–2026 es al menos el doble de la nacional (6,4 %, es decir, "
    "12,7 %) y suma al menos 5 eventos. La cumplen Guaviare (14,4 %) y Cauca (13,4 %), donde conviene sumar C-UAS y "
    "detección de AEI.",
    "Frente a la U2, 25 de 33 departamentos conservan su categoría. La proyección contradice el perfil observado en 2026 en "
    "3 departamentos (Chocó, Valle del Cauca y Meta; en la U2 eran 10). Es una medida de cercanía a la inercia, no de "
    "acierto, y señala los casos que conviene revisar con inteligencia de campo. Chocó y Valle son desescaladas recientes con "
    "s^{C} de 2026 entre 0,45 y 0,50. La desviación entre las 10 redes crece donde el modelo duda (Chocó 0,16; Córdoba 0,12).",
    "La información de 2026 cubre 33 semanas, así que d₁ resume febrero a agosto y no el segundo semestre. En 2020–2025, d₁ "
    "medido en esa ventana difiere del semestral en 0,084 en promedio (correlación 0,88). Meta (0,50), Huila (0,51) y "
    "Casanare (0,52) están en el umbral, así que la proyección debe repetirse al cerrar 2026.",
], "R09_proyeccion", 6.9, "Figura 9: P(perfil C) proyectada para 2027 (centroides; tamaño según la raíz de los eventos anualizados de 2026; aro rojo: "
   "alerta A/B; rótulo gris: menos de 10 eventos). Sin rótulo, la zona andina con P < 0,1: Bogotá 0,01, Boyacá 0,02, Caldas 0,07, "
   "Cundinamarca 0,04, Quindío 0,06, Risaralda 0,07 y Santander 0,04", ancho_texto=10.2)

# =====================================================================================================================
h1("9.   Conclusiones técnicas y recomendaciones futuras")
for k, txt in enumerate([
    "El modelo final responde la pregunta de la U1 a escala departamental, con la misma etiqueta, el mismo diseño *t*−1 → *t* "
    "y las mismas variables estructurales. Las dos variables nuevas llevan la \"intensidad histórica\" de la U1 a la "
    "composición reciente del conflicto.",
    "De 10 frentes evaluados se retuvieron tres: m₁, d₁ y el ensamble. Las demás intervenciones no cumplieron la regla de "
    "retención, salvo la vecindad, que mejoró solo en el eje que no era el de su hipótesis, y el embedding, que la cumplió y "
    "se descartó por la regla 8. Con 58 filas de validación, la rejilla de hiperparámetros no ofreció una alternativa que se "
    "sostuviera al año siguiente.",
    "Frente al modelo de la U2 la mejora es clara: con ensamble en los dos lados, la log-loss baja de 0,376 a 0,304 en "
    "selección y de 0,470 a 0,380 en 2026. Frente a la inercia cruda, la ventaja es de calibración. Frente a una inercia "
    "calibrada no está demostrada a escala departamental: el modelo gana por 0,024 en selección, pierde por 0,029 en 2026 y "
    "empata en territorios nuevos cuando se respeta el tiempo (0,361 frente a 0,363 de la persistencia y 0,332 de la "
    "persistencia calibrada).",
    "El modelo anticipa 2 de 13 transiciones (el de la U2, 3) y no supera a una logística con las mismas variables en el eje "
    "temporal. Los efectos fijos de cada departamento (embedding) tienen información que las variables actuales no explican.",
    "Para la pregunta de inversión, la evidencia sostiene para 2027 priorizar ISR, detección temprana y movilidad protegida "
    "en los 9 departamentos C consolidados con datos suficientes, sumar C-UAS y detección de AEI en Guaviare y Cauca, "
    "mantener portafolio flexible y monitoreo trimestral en los 11 de transición, y revisar con inteligencia de campo Chocó, "
    "Valle del Cauca y Meta. Como una inercia calibrada da casi la misma probabilidad a esta escala, el modelo aporta hoy un "
    "procedimiento reproducible que integra la composición reciente, las variables estructurales y la incertidumbre entre "
    "redes; su ventaja predictiva queda por probar a escala municipal (R1).",
], 1):
    viñeta(txt, numero=k)
p("La Tabla 7 ordena las recomendaciones para la siguiente iteración.")
tabla("Tabla 7: Recomendaciones priorizadas para la siguiente iteración", [
    ["#", "Recomendación", "Evidencia", "Acción concreta"],
    ["R1", "Migrar a municipio (registro por evento de ACLED)", "Con 30 departamentos el modelo está limitado por varianza (U2) y en territorios nuevos empata con la inercia (§6.3)", "Acceso académico al registro por evento; en el pipeline cambian `datos.py` y las constantes departamentales de `config.py`"],
    ["R2", "x₃ a escala municipal", "El conteo anual por departamento empeoró el modelo (I3)", "Medir entradas y salidas de grupos por municipio en lugar del conteo anual"],
    ["R3", "Señales adelantadas de transición", "11 de 19 errores son transiciones", "Variables mensuales del último trimestre, ceses al fuego y negociaciones"],
    ["R4", "Reentrenar cada año y vigilar el PSI con su piso de ruido", "Prevalencia de C entre 41 % y 62 % en los años validados; en 2026 la letalidad supera el piso de ruido del PSI (§7)", "Ejecutar el pipeline al cierre de cada año y decidir con bandas, sin umbral fijo"],
    ["R5", "Revisar logaritmos y MIN_EVENTOS", "Controles de §2.2", "Repetir la batería de controles con el dataset municipal"],
    ["R6", "Persistencia calibrada y logística como rivales permanentes", "Empatan con el modelo dentro del ruido (§6)", "Reportarlas en cada ejecución del pipeline y en la tarjeta del modelo"],
    ["R7", "Reevaluar el embedding cuando cierre 2026", "Mejor en la prueba 2026 (0,314)", "Candidato en la selección 2024–2026, con prueba en 2027; a escala municipal, reemplazarlo por x₃ y distancia"],
], [0.8, 4.2, 5.4, 7.0], tam=7.9)

# =====================================================================================================================
h1("10.   Trabajo en equipo y organización")
p("Mantuvimos la división de las etapas anteriores y la llevamos a los cuatro roles que pide la guía (Tabla 8). Juan Diego "
  "propuso e implementó las intervenciones a partir del diagnóstico de la U2 y escribió el paquete; Laura revisó cada "
  "resultado contra las salidas del cuaderno, redactó las lecturas y propuso la lectura para la decisión (bandas, alerta A/B "
  "y énfasis tecnológico). Las decisiones de protocolo las tomamos juntos antes de ejecutar: seleccionar solo con origen móvil "
  "y usar 2026 una vez, medir contra la persistencia con log-loss, la regla de retención y mantener la capacidad de la U2 si "
  "la rejilla no ofrecía una alternativa estable. Dos decisiones salieron de discutir resultados que no esperábamos: no "
  "adoptar el embedding (regla 8) y no cambiar los logaritmos aunque el control los favoreciera (regla 7). Después de la "
  "revisión del informe agregamos la persistencia calibrada y la validación espacio-temporal, sin cambiar el modelo. Las 20 "
  "decisiones (D1–D20) quedaron en la bitácora del cuaderno con su alternativa y su evidencia.")
tabla("Tabla 8: Roles técnicos de la Unidad 3", [
    ["Rol", "Responsable", "Entregables"],
    ["Modelado", "Juan Diego Saldarriaga Muñoz", "Paquete perfil_amenaza, intervenciones I1–I9, rejilla y validación anidada, pipeline y pruebas"],
    ["Evaluación", "Conjunta (cálculo: Juan Diego; interpretación: Laura)", "Protocolo y regla de retención, controles, antes vs. después, análisis de errores"],
    ["Documentación", "Laura Alejandra Díaz Sastoque", "Interpretaciones, tarjeta del modelo, informe y README"],
    ["Presentación", "Laura (guion) y Juan Diego (demostración)", "Video ejecutivo de 8 a 10 minutos y sustentación"],
], [2.6, 5.4, 9.4])

# =====================================================================================================================
h1("11.   Referencias bibliográficas en APA 7")
p("Uso de IA. Usamos Claude (Anthropic) como apoyo en la programación del pipeline y del cuaderno y en la revisión de la "
  "redacción y la organización del informe. La propuesta del proyecto, la búsqueda y selección de datos, el análisis, las "
  "decisiones metodológicas y las conclusiones son del equipo, que ejecutó, revisó y verificó todos los resultados.", tam=9.5)
REFS = [
    "Armed Conflict Location & Event Data Project (ACLED). (2026). *ACLED data export: Colombia, agregado semanal por departamento y subtipo de evento (datos hasta el 15 de agosto de 2026)* [Conjunto de datos]. https://acleddata.com",
    "Bergmeir, C., & Benítez, J. M. (2012). On the use of cross-validation for time series predictor evaluation. *Information Sciences, 191*, 192–213. https://doi.org/10.1016/j.ins.2011.12.028",
    "Cawley, G. C., & Talbot, N. L. C. (2010). On over-fitting in model selection and subsequent selection bias in performance evaluation. *Journal of Machine Learning Research, 11*, 2079–2107. https://www.jmlr.org/papers/v11/cawley10a.html",
    "Departamento Administrativo Nacional de Estadística (DANE). (2019). *Pobreza multidimensional por departamentos 2018: Resultados*. DANE.",
    "Dietterich, T. G. (1998). Approximate statistical tests for comparing supervised classification learning algorithms. *Neural Computation, 10*(7), 1895–1923. https://doi.org/10.1162/089976698300017197",
    "Géron, A. (2017). *Hands-on machine learning with Scikit-Learn and TensorFlow*. O'Reilly Media.",
    "Gneiting, T., & Raftery, A. E. (2007). Strictly proper scoring rules, prediction, and estimation. *Journal of the American Statistical Association, 102*(477), 359–378. https://doi.org/10.1198/016214506000001437",
    "Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep learning*. MIT Press.",
    "Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On calibration of modern neural networks. En *Proceedings of the 34th International Conference on Machine Learning* (PMLR 70, pp. 1321–1330). https://proceedings.mlr.press/v70/guo17a.html",
    "Hansen, M. C., Potapov, P. V., Moore, R., Hancher, M., Turubanova, S. A., Tyukavina, A., Thau, D., Stehman, S. V., Goetz, S. J., Loveland, T. R., Kommareddy, A., Egorov, A., Chini, L., Justice, C. O., & Townshend, J. R. G. (2013). High-resolution global maps of 21st-century forest cover change. *Science, 342*(6160), 850–853. https://doi.org/10.1126/science.1244693",
    "Kingma, D. P., & Ba, J. (2015). Adam: A method for stochastic optimization. En *International Conference on Learning Representations*. https://arxiv.org/abs/1412.6980",
    "Lakshminarayanan, B., Pritzel, A., & Blundell, C. (2017). Simple and scalable predictive uncertainty estimation using deep ensembles. En *Advances in Neural Information Processing Systems 30* (pp. 6402–6413). Curran Associates.",
    "Ministerio de Defensa de Colombia. (2025). *Informe de gestión: modernización y transformación de las Fuerzas Militares*. Ministerio de Defensa Nacional.",
    "Ministerio de Justicia y del Derecho, Observatorio de Drogas de Colombia. (2026). *Detección de cultivos de coca (hectáreas)* [Conjunto de datos]. Datos Abiertos Colombia. https://www.datos.gov.co/d/acs4-3wgp",
    "Mitchell, M., Wu, S., Zaldivar, A., Barnes, P., Vasserman, L., Hutchinson, B., Spitzer, E., Raji, I. D., & Gebru, T. (2019). Model cards for model reporting. En *Proceedings of the Conference on Fairness, Accountability, and Transparency* (pp. 220–229). https://doi.org/10.1145/3287560.3287596",
    "Platt, J. (1999). Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods. En A. J. Smola, P. Bartlett, B. Schölkopf, & D. Schuurmans (Eds.), *Advances in large margin classifiers* (pp. 61–74). MIT Press.",
    "Raleigh, C., Linke, A., Hegre, H., & Karlsen, J. (2010). Introducing ACLED: An armed conflict location and event dataset. *Journal of Peace Research, 47*(5), 651–660. https://doi.org/10.1177/0022343310378914",
    "Roberts, D. R., Bahn, V., Ciuti, S., Boyce, M. S., Elith, J., Guillera-Arroita, G., Hauenstein, S., Lahoz-Monfort, J. J., Schröder, B., Thuiller, W., Warton, D. I., Wintle, B. A., Hartig, F., & Dormann, C. F. (2017). Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure. *Ecography, 40*(8), 913–929. https://doi.org/10.1111/ecog.02881",
    "Sundberg, R., & Melander, E. (2013). Introducing the UCDP Georeferenced Event Dataset. *Journal of Peace Research, 50*(4), 523–532. https://doi.org/10.1177/0022343313484347",
    "Uppsala Conflict Data Program. (2026). *UCDP Georeferenced Event Dataset (GED) Global, versión 26.1* [Conjunto de datos]. Universidad de Uppsala. https://ucdp.uu.se/downloads/",
    "World Resources Institute. (2025). *Global Forest Watch: Colombia tree cover loss by subnational region* [Conjunto de datos]. https://www.globalforestwatch.org",
]
for ref in REFS:
    par = doc.add_paragraph(); par.alignment = WD_ALIGN_PARAGRAPH.LEFT
    par.paragraph_format.left_indent = Cm(0.9); par.paragraph_format.first_line_indent = Cm(-0.9); par.paragraph_format.space_after = Pt(1)
    runs(par, ref, tam=9)

doc.save(SALIDA)
print("Word:", SALIDA)
r = subprocess.run(["soffice", "--headless", "--convert-to", "pdf", "--outdir", str(SALIDA.parent), str(SALIDA)], capture_output=True, text=True)
print(r.stdout.strip() or r.stderr.strip())
