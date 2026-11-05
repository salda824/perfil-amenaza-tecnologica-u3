# Presentaciones del proyecto (video U3 y clase)

| Archivo | Para qué | Duración |
|---|---|---|
| `Presentacion_Video_U3.html` | Apoyo visual del **video ejecutivo** (Producto 2 de la U3): problema, decisiones técnicas, resultados y conclusiones estratégicas | 8–10 min (meta 9:15), 12 diapositivas |
| `Guion_Video_U3.docx` | Guion del video: qué dice Laura y qué dice Juan Diego en cada diapositiva, qué hacer en pantalla, tiempos y lista para grabar | |
| `Presentacion_Clase_Proyecto.html` | Presentación en clase de **todo el proyecto** (U1, U2 y U3) | 20–30 min (meta 25:00), 27 diapositivas |
| `Guion_Presentacion_Clase.docx` | Guion de la clase, tiempos y **preguntas probables de la sustentación** con su respuesta | |

## Cómo se usan

Los HTML son autocontenidos (un solo archivo, sin dependencias locales): basta abrirlos en Chrome o Edge. Sin internet
funcionan igual; con internet cargan las fuentes Space Grotesk, IBM Plex Sans y JetBrains Mono.

| Tecla | Acción |
|---|---|
| `→` / `Espacio` · `←` | Avanzar (también revela tarjetas por pasos) · retroceder |
| `F` | Pantalla completa (para grabar) |
| `O` / `Esc` | Vista general de todas las diapositivas |
| `C` · `R` | Cronómetro de ensayo con la meta de cada diapositiva · reiniciar |
| `S` | Ocultar el orador y el pie |
| `T` | Tema claro (proyectores con poco contraste) u oscuro |
| `H` | Ayuda · `número` + `Enter` salta a esa diapositiva |

Elementos interactivos: mapa de Colombia por departamento (proyección 2027, perfil 2026 y recorrido histórico 2021–2026,
con detalle al hacer clic), mapa de calor departamento × año, comparador de modelos (selección, prueba 2026, departamentos
nuevos; log-loss, AUC, ECE, exactitud), gráfico de bosque de las intervenciones, red neuronal animada (propagación y
retropropagación), pipeline de 8 etapas, terminal con la salida real del pipeline, simulador de bandas de decisión,
tarjetas giratorias de los perfiles y pestañas con el detalle de I1–I9.

## Cómo se regeneran

```bash
python presentaciones/construir_presentaciones.py   # los dos HTML (lee salidas_u3/, referencias_u2/, ACLED y las figuras)
python presentaciones/construir_guiones.py          # los dos .docx (valida títulos, oradores y tiempos contra los HTML)
```

Si cambian los resultados del proyecto, basta volver a ejecutar el cuaderno o el pipeline y luego estos dos scripts: las
cifras de los gráficos y de los textos con `data-v` se leen de los CSV. Las cifras dichas en voz alta están escritas en
`guiones.py` y deben revisarse a mano.

## Estructura

```
presentaciones/
├── construir_presentaciones.py   arma cada HTML: datos (DATOS) + figuras WebP en base64 + motor + plantilla
├── construir_guiones.py          arma los .docx con python-docx y estima la duración a 140 palabras por minuto
├── guiones.py                    texto de los guiones y preguntas de la sustentación
└── fuentes/
    ├── video.html · clase.html   marcado de las diapositivas (data-titulo, data-orador, data-seg)
    ├── motor.css · motor.js      motor de diapositivas propio (escenario 1920 × 1080 escalado, HUD, atajos, temas)
    ├── graficos.js               componentes interactivos en SVG/HTML (sin librerías externas)
    ├── colombia_dptos.json       límites departamentales del DANE (MGN 2018) simplificados a trazos SVG
    ├── preparar_mapa.py          genera colombia_dptos.json desde el GeoJSON del DANE
    ├── salida_pipeline.txt       salida real de pytest y del pipeline (la muestra la terminal animada)
    └── datos_presentaciones.json copia legible de los datos incrustados (para revisar cifras)
```

Fuente del mapa: DANE, Marco Geoestadístico Nacional 2018 (límites departamentales), copia pública en
github.com/caticoa3/colombia_mapa. Las figuras de la U1 y la U2 se extraen de los cuadernos entregados en `docs/entregas/`.
