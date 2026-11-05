"""
Genera los guiones en Word para Laura y Juan Diego:

    presentaciones/Guion_Video_U3.docx                video ejecutivo (8–10 min)
    presentaciones/Guion_Presentacion_Clase.docx      presentación en clase (20–30 min)

Uso (desde la raíz del repositorio):  python presentaciones/construir_guiones.py

El texto está en presentaciones/guiones.py. Antes de escribir, el script comprueba que cada diapositiva del guion coincida
con la del HTML (título, orador y segundos meta de fuentes/video.html y fuentes/clase.html) y estima la duración a
140 palabras por minuto.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
import guiones as G  # noqa: E402

PPM = 140                                       # palabras por minuto (ritmo de exposición en español)
FUENTE = "Arial"
COLOR = {"laura": RGBColor(0xC2, 0x30, 0x6A), "juan": RGBColor(0x0B, 0x7F, 0xA6), "ambos": RGBColor(0x55, 0x4B, 0x8F)}
NOMBRE = {"laura": "Laura Díaz", "juan": "Juan Diego Saldarriaga", "ambos": "Laura y Juan Diego"}
TINTA, GRIS, ACENTO = RGBColor(0x10, 0x15, 0x1F), RGBColor(0x6D, 0x75, 0x86), RGBColor(0xB9, 0x78, 0x00)


def mmss(seg):
    return f"{int(seg // 60)}:{int(round(seg % 60)):02d}"


def palabras(texto):
    return len(re.findall(r"\w+", texto))


def validar(guion, plantilla):
    html = (AQUI / "fuentes" / plantilla).read_text(encoding="utf-8")
    sl = re.findall(r'<section class="slide[^"]*" data-titulo="([^"]+)" data-orador="(\w+)" data-seg="(\d+)"', html)
    if len(sl) != len(guion):
        raise ValueError(f"{plantilla}: {len(sl)} diapositivas en el HTML y {len(guion)} en el guion")
    for i, ((t, o, s), g) in enumerate(zip(sl, guion), 1):
        if (t, o, int(s)) != (g[0], g[1], g[2]):
            raise ValueError(f"{plantilla}, diapositiva {i}: HTML {(t, o, s)} ≠ guion {g[:3]}")


# ------------------------------------------------------------------------------------------------ utilidades docx
def sombrear(celda, hexa):
    tcPr = celda._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd"); shd.set(qn("w:val"), "clear"); shd.set(qn("w:color"), "auto"); shd.set(qn("w:fill"), hexa)
    tcPr.append(shd)


def bordes_tabla(tabla, color="D9D6CC"):
    tblPr = tabla._tbl.tblPr
    b = OxmlElement("w:tblBorders")
    for lado in ("top", "left", "bottom", "right", "insideH", "insideV"):
        e = OxmlElement(f"w:{lado}"); e.set(qn("w:val"), "single"); e.set(qn("w:sz"), "4"); e.set(qn("w:color"), color)
        b.append(e)
    tblPr.append(b)


def fuente_fija(rpr_padre):
    """Fija Arial en ascii, hAnsi y eastAsia y quita las fuentes del tema (que Word aplica a los títulos)."""
    rf = rpr_padre.find(qn("w:rFonts"))
    if rf is None:
        rf = OxmlElement("w:rFonts"); rpr_padre.insert(0, rf)
    for k in ("w:asciiTheme", "w:hAnsiTheme", "w:eastAsiaTheme", "w:cstheme"):
        if rf.get(qn(k)) is not None:
            del rf.attrib[qn(k)]
    for k in ("w:ascii", "w:hAnsi", "w:eastAsia", "w:cs"):
        rf.set(qn(k), FUENTE)


def run(p, texto, negrita=False, cursiva=False, color=None, tam=None):
    r = p.add_run(texto); r.bold = negrita; r.italic = cursiva
    fuente_fija(r._r.get_or_add_rPr())
    if color is not None:
        r.font.color.rgb = color
    if tam:
        r.font.size = Pt(tam)
    return r


def parrafo(doc, texto="", tam=10.5, color=TINTA, antes=0, despues=6, negrita=False, cursiva=False, sangria=None):
    p = doc.add_paragraph(); p.paragraph_format.space_before = Pt(antes); p.paragraph_format.space_after = Pt(despues)
    if sangria is not None:
        p.paragraph_format.left_indent = Cm(sangria)
    if texto:
        run(p, texto, negrita, cursiva, color, tam)
    return p


def titulo_seccion(doc, texto):
    p = doc.add_heading(texto, level=1)
    p.paragraph_format.space_before = Pt(16); p.paragraph_format.space_after = Pt(6); p.paragraph_format.keep_with_next = True
    pPr = p._p.get_or_add_pPr(); bd = OxmlElement("w:pBdr"); bot = OxmlElement("w:bottom")
    bot.set(qn("w:val"), "single"); bot.set(qn("w:sz"), "8"); bot.set(qn("w:color"), "B97800"); bd.append(bot); pPr.append(bd)
    return p


def vineta(doc, texto, negrita_ini=None):
    p = doc.add_paragraph(style="List Bullet"); p.paragraph_format.space_after = Pt(3)
    if negrita_ini:
        run(p, negrita_ini, negrita=True)
    run(p, texto)
    return p


def tabla(doc, encabezado, filas, anchos, colores=None):
    t = doc.add_table(rows=1, cols=len(encabezado)); t.alignment = WD_TABLE_ALIGNMENT.CENTER; bordes_tabla(t)
    t.autofit = False
    lay = OxmlElement("w:tblLayout"); lay.set(qn("w:type"), "fixed"); t._tbl.tblPr.append(lay)
    for j, w in enumerate(anchos):
        t.columns[j].width = Cm(w)
    for j, h in enumerate(encabezado):
        c = t.rows[0].cells[j]; c.width = Cm(anchos[j]); sombrear(c, "1E2A44")
        c.paragraphs[0].text = ""; run(c.paragraphs[0], h, negrita=True, color=RGBColor(0xFF, 0xFF, 0xFF), tam=9.5)
    for i, f in enumerate(filas):
        cells = t.add_row().cells
        for j, v in enumerate(f):
            c = cells[j]; c.width = Cm(anchos[j])
            if i % 2:
                sombrear(c, "F4F2EC")
            col = colores[i][j] if colores and colores[i] and colores[i][j] else TINTA
            c.paragraphs[0].text = ""; run(c.paragraphs[0], str(v), color=col, tam=9.5, negrita=bool(colores and colores[i] and colores[i][j]))
    return t


# ------------------------------------------------------------------------------------------------ documento
def documento(guion, plantilla, destino, titulo, subtitulo, html, uso, cobertura, cierre_titulo, cierre, recorte):
    validar(guion, plantilla)
    doc = Document()
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Cm(21), Cm(29.7)
    for m in ("left_margin", "right_margin"):
        setattr(sec, m, Cm(2))
    sec.top_margin, sec.bottom_margin = Cm(1.8), Cm(1.8)
    st = doc.styles["Normal"]; st.font.name = FUENTE; st.font.size = Pt(10.5)
    st.element.rPr.rFonts.set(qn("w:eastAsia"), FUENTE)
    for n, tam in (("Heading 1", 15), ("Heading 2", 12.5)):
        h = doc.styles[n]; h.font.name = FUENTE; h.font.size = Pt(tam); h.font.bold = True; h.font.color.rgb = TINTA
        fuente_fija(h.element.rPr)
    doc.core_properties.title = titulo
    doc.core_properties.author = "Laura Alejandra Díaz Sastoque y Juan Diego Saldarriaga Muñoz"

    # pie con número de página
    pie = sec.footer.paragraphs[0]; pie.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    run(pie, f"{titulo} · página ", color=GRIS, tam=8)
    for tipo, txt in (("begin", None), (None, "PAGE"), ("end", None)):
        r = pie.add_run(); r.font.size = Pt(8); r.font.color.rgb = GRIS
        if tipo:
            fc = OxmlElement("w:fldChar"); fc.set(qn("w:fldCharType"), tipo); r._r.append(fc)
        else:
            it = OxmlElement("w:instrText"); it.set(qn("xml:space"), "preserve"); it.text = txt; r._r.append(it)

    # ---------------------------------------------------------------- portada breve
    p = parrafo(doc, "REDES NEURONALES · UNIVERSIDAD DE LA SABANA", tam=9, color=ACENTO, negrita=True, despues=2)
    p = parrafo(doc, titulo, tam=24, negrita=True, despues=2)
    parrafo(doc, subtitulo, tam=12, color=GRIS, despues=10)
    tiempos = {"laura": 0.0, "juan": 0.0}
    estim, acum, filas, cols = [], 0.0, [], []
    for i, (t, o, s, pant, partes) in enumerate(guion, 1):
        seg_est = sum(palabras(x) for _, x in partes) / PPM * 60
        for quien, x in partes:
            tiempos[quien] += palabras(x) / PPM * 60
        acum += s; estim.append(seg_est)
        filas.append([f"{i:02d}", t, NOMBRE[o], mmss(s), mmss(seg_est), mmss(acum)])
        cols.append([None, None, COLOR[o], None, None, None])
    meta_total, est_total = sum(g[2] for g in guion), sum(estim)
    tabla(doc, ["Dato", "Valor"], [
        ["Presentación", html],
        ["Diapositivas", str(len(guion))],
        ["Duración meta", f"{mmss(meta_total)} min (estimada por palabras: {mmss(est_total)})"],
        ["Laura Díaz", f"{mmss(tiempos['laura'])} min de texto"],
        ["Juan Diego Saldarriaga", f"{mmss(tiempos['juan'])} min de texto"],
        ["Ritmo usado para estimar", f"{PPM} palabras por minuto"],
    ], [5, 12])

    titulo_seccion(doc, "Cómo usar la presentación")
    for a, b in uso:
        vineta(doc, b, a)

    if cobertura:
        titulo_seccion(doc, cobertura[0])
        tabla(doc, cobertura[1], cobertura[2], cobertura[3])

    titulo_seccion(doc, "Reparto y tiempos")
    parrafo(doc, "«Meta» es el tiempo que marca el cronómetro de la presentación (tecla C). «Estimado» sale de contar las palabras "
                 "del guion. «Acumulado» es la meta sumada hasta esa diapositiva.", tam=9.5, color=GRIS)
    tabla(doc, ["N.º", "Diapositiva", "Orador", "Meta", "Estimado", "Acumulado"], filas, [1.0, 5.8, 4.4, 1.6, 2.0, 2.2], cols)

    # ---------------------------------------------------------------- guion
    titulo_seccion(doc, "Guion por diapositiva")
    acum = 0.0
    for i, (t, o, s, pant, partes) in enumerate(guion, 1):
        acum += s
        h = doc.add_heading(level=2); h.paragraph_format.space_before = Pt(12); h.paragraph_format.keep_with_next = True
        run(h, f"{i:02d} · ", color=ACENTO); run(h, t)
        p = parrafo(doc, despues=3); p.paragraph_format.keep_with_next = True
        run(p, NOMBRE[o], negrita=True, color=COLOR[o], tam=9.5)
        run(p, f"   ·   meta {mmss(s)}   ·   estimado {mmss(estim[i - 1])}   ·   acumulado {mmss(acum)}", color=GRIS, tam=9.5)
        p = parrafo(doc, despues=5, sangria=0.3); p.paragraph_format.keep_with_next = True
        run(p, "En pantalla: ", negrita=True, color=GRIS, tam=9.5); run(p, pant, cursiva=True, color=GRIS, tam=9.5)
        for quien, texto in partes:
            p = parrafo(doc, despues=6, sangria=0.3)
            p.paragraph_format.line_spacing = 1.25
            run(p, NOMBRE[quien].split()[0].upper() + ":  ", negrita=True, color=COLOR[quien], tam=10.5)
            run(p, texto, tam=11)

    titulo_seccion(doc, cierre_titulo)
    for item in cierre:
        if len(item) == 3:                       # pregunta, quién responde, respuesta
            q, quien, r = item
            p = parrafo(doc, antes=6, despues=2); p.paragraph_format.keep_with_next = True
            run(p, q, negrita=True); run(p, f"   responde {NOMBRE[quien]}", color=COLOR[quien], tam=9)
            parrafo(doc, r, sangria=0.3, despues=4)
        else:
            vineta(doc, item[1], item[0])

    titulo_seccion(doc, "Si el tiempo no alcanza")
    for a, b in recorte:
        vineta(doc, b, a)

    ruta = AQUI / destino
    doc.save(ruta)
    print(f"{ruta.name}: {len(guion)} diapositivas · meta {mmss(meta_total)} · estimado {mmss(est_total)} · "
          f"Laura {mmss(tiempos['laura'])} · Juan Diego {mmss(tiempos['juan'])}")
    for i, ((t, o, s, *_), e) in enumerate(zip(guion, estim), 1):
        if abs(e - s) > 0.25 * s:
            print(f"   aviso: diapositiva {i} «{t}» meta {mmss(s)} y estimado {mmss(e)}")


USO = [
    ("Abrir: ", "doble clic en el archivo HTML (Chrome o Edge). Funciona sin internet; con internet carga las fuentes tipográficas."),
    ("Avanzar: ", "flecha → o barra espaciadora; retroceder con ←. En las diapositivas con tarjetas que aparecen por pasos, cada → muestra una."),
    ("Pantalla completa: ", "tecla F. Vista general de todas las diapositivas: tecla O (clic en una para saltar)."),
    ("Ensayo: ", "tecla C muestra el cronómetro con la meta de cada diapositiva (verde si van a tiempo, rojo si van tarde); R lo reinicia."),
    ("Orador: ", "el recuadro de arriba a la derecha indica quién habla. La tecla S lo oculta junto con el pie, por si estorba a la cámara."),
    ("Proyector: ", "si el proyector lava los colores oscuros, la tecla T cambia a tema claro."),
    ("Interacción: ", "pasar el cursor muestra detalles en mapas y gráficos; los botones cambian el conjunto de datos o la métrica."),
]

COBERTURA_VIDEO = ("Qué pide la guía (Producto 2) y dónde se cubre", ["Requisito de la guía", "Diapositivas", "Quién"], [
    ["Síntesis del problema", "01 a 04", "Laura"],
    ["Principales decisiones técnicas", "05 a 08 (protocolo, intervenciones, modelo, pipeline)", "Juan Diego"],
    ["Resultados más relevantes", "09 y 10 (antes y después, lectura crítica)", "Juan Diego y Laura"],
    ["Conclusiones estratégicas", "11 y 12 (mapa 2027, bandas, alerta A/B, siguiente paso)", "Laura y Juan Diego"],
    ["Audiencia técnica con enfoque aplicado", "Todas: cifras de evaluación y traducción a tecnología", "Ambos"],
], [5.6, 8.0, 3.6])

BLOQUES_CLASE = [("Apertura y recorrido", 1, 2), ("Unidad 1 · Formulación", 3, 7), ("Unidad 2 · Modelo base", 8, 12),
                 ("Unidad 3 · Proyecto integrador", 13, 24), ("Cierre: conclusiones, recomendaciones y equipo", 25, 27)]
COBERTURA_CLASE = ("Estructura de la presentación", ["Bloque", "Diapositivas", "Meta"], [
    [n, f"{a:02d} a {b:02d}", mmss(sum(g[2] for g in G.CLASE[a - 1:b]))] for n, a, b in BLOQUES_CLASE], [8.0, 5.0, 3.6])

GRABACION = [
    ("Formato: ", "MP4, 1920 × 1080, entre 8 y 10 minutos (la meta es 9:15)."),
    ("Herramienta: ", "OBS Studio (gratuito) o una reunión de Teams/Zoom grabada compartiendo la pantalla completa del navegador."),
    ("Antes de grabar: ", "abrir el HTML, presionar F, ensayar una vez con C y ocultar el cronómetro (C otra vez) antes de empezar."),
    ("Cámaras: ", "si salen las dos cámaras, ubicarlas abajo a la derecha; si tapan algo, ocultar el orador con S."),
    ("Audio: ", "un micrófono por persona o grabar en un cuarto sin eco; hacer una prueba de 20 segundos."),
    ("Ritmo: ", "las diapositivas 9 y 11 tienen botones: hacer los clics que indica el guion mientras se habla, sin pausas largas."),
    ("Revisión: ", "ver el video completo y confirmar la duración antes de subirlo."),
]

RECORTE_VIDEO = [
    ("Diapositiva 6: ", "omitir la lista de intervenciones que no mejoraron (ahorra unos 15 segundos)."),
    ("Diapositiva 8: ", "omitir la frase sobre la fuga de información (unos 8 segundos)."),
    ("Diapositiva 11: ", "nombrar solo un departamento por categoría (unos 10 segundos)."),
]
RECORTE_CLASE = [
    ("Diapositiva 15: ", "mostrar solo la pestaña I3 en lugar de tres (unos 25 segundos)."),
    ("Diapositiva 18: ", "saltarla y mencionar el ensamble en la diapositiva 17 (unos 45 segundos)."),
    ("Diapositiva 22: ", "decir solo la frase del semestre reciente (unos 25 segundos)."),
    ("Si sobra tiempo: ", "en la diapositiva 23 usar el botón «Histórico» y recorrer 2021 a 2026 mostrando las transiciones."),
]


def main():
    documento(G.VIDEO, "video.html", "Guion_Video_U3.docx", "Guion del video ejecutivo · Unidad 3",
              "Perfil de amenaza tecnológica por departamento · Producto 2 del Proyecto Integrador",
              "presentaciones/Presentacion_Video_U3.html", USO, COBERTURA_VIDEO, "Lista de verificación para grabar",
              GRABACION, RECORTE_VIDEO)
    documento(G.CLASE, "clase.html", "Guion_Presentacion_Clase.docx", "Guion de la presentación en clase",
              "Perfil de amenaza tecnológica por departamento · Proyecto completo (U1, U2 y U3) · 25 minutos",
              "presentaciones/Presentacion_Clase_Proyecto.html", USO, COBERTURA_CLASE,
              "Preguntas probables y cómo responderlas", G.PREGUNTAS, RECORTE_CLASE)


if __name__ == "__main__":
    main()
