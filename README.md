# Perfil de amenaza tecnológica por departamento (Proyecto Integrador, Unidad 3)

Redes Neuronales · Universidad de La Sabana · Laura Alejandra Díaz Sastoque y Juan Diego Saldarriaga Muñoz

> **¿Llegas sin contexto (persona o IA)?** Lee [`CONTEXTO_PROYECTO.md`](CONTEXTO_PROYECTO.md) y abre el grafo
> `graphify-out/graph.html`. El informe final está en `informe/Informe_U3_Proyecto_Integrador.pdf`.

El modelo estima, con la información disponible al cierre de un año, la probabilidad de que cada departamento de
Colombia tenga el año siguiente perfil de amenaza **C (armas convencionales y emboscadas)** o **D (mixto)**. Esa
probabilidad orienta el énfasis tecnológico (ISR, C-UAS, detección de AEI). Este repositorio consolida las Unidades 1
(formulación) y 2 (modelo base) en un pipeline reproducible.

## Estructura

```
CONTEXTO_PROYECTO.md      documento maestro del proyecto (U1, U2, U3, resultados, pendientes)
CLAUDE.md / AGENTS.md     instrucciones para asistentes de IA
configuracion.json        modelo final y protocolo (lo escribe la §8.1 del cuaderno)
Proyecto_U3_Integrador.ipynb   cuaderno de la entrega: diagnóstico, intervenciones, evaluación e interpretación
informe/                  informe técnico final (PDF/DOCX) y scripts que lo generan
src/perfil_amenaza/       paquete: config, datos, variables, modelo, evaluacion, pipeline, graficos
tests/                    10 pruebas automáticas (contrato y su caso negativo, fuga temporal, escalador, artefacto,
                          persistencia calibrada, partición espacio-temporal, regla de retención, determinismo, réplica U2)
herramientas/             preparar_ucdp.py (reduce UCDP a Colombia) y cuaderno/ (generador del cuaderno y exportar_texto.py)
revision/                 revisión estricta del informe y correcciones aplicadas
datos/                    export semanal de ACLED + datos/externos/ (coca, bosque, UCDP)
salidas_u3/               métricas, predicciones, figuras, modelos/perfil_amenaza_u3.keras, TARJETA_MODELO.md
referencias_u2/           resultados guardados de la Unidad 2 (para comparar "antes vs. después")
docs/                     guías y rúbricas, entregas U1 y U2, planeación U3 y el cuaderno U3 en texto
graphify-out/             grafo de conocimiento del proyecto (graph.html, GRAPH_REPORT.md, graph.json)
```

## Uso rápido en Windows (cmd)

```bat
ejecutar_todo.bat "C:\ruta\a\ged261-csv"
```

Crea el entorno virtual (Python 3.11 a 3.13), instala las dependencias, reduce UCDP (opcional), corre las pruebas,
ejecuta el cuaderno completo en una copia (`Proyecto_U3_Integrador_ejecutado.ipynb`, para no sobrescribir el entregado)
y muestra la proyección 2027. Al ejecutarse, el cuaderno reescribe `configuracion.json` y `salidas_u3/`.

## Uso

```bash
pip install -r requirements.txt
python -m pytest -q tests                                       # 10 pruebas
PYTHONPATH=src python -m perfil_amenaza.pipeline todo           # datos -> evaluación -> artefacto -> proyección 2027
PYTHONPATH=src python -m perfil_amenaza.pipeline predecir --anio 2026   # solo con el artefacto guardado
jupyter nbconvert --to notebook --execute Proyecto_U3_Integrador.ipynb  # cuaderno completo (~20 min en CPU)
python herramientas/cuaderno/exportar_texto.py                  # versión en texto del cuaderno (docs/u3_texto/)
python informe/generar_figuras.py && python informe/construir_informe.py   # informe (requiere LibreOffice para el PDF)
```

En Windows (PowerShell), use `$env:PYTHONPATH="src"` antes de los comandos de `perfil_amenaza`.

## Variable x3 (actores armados, opcional)

Descargue `https://ucdp.uu.se/downloads/ged/ged261-csv.zip` (39 MB) y redúzcalo en su computador:

```bash
python herramientas/preparar_ucdp.py ruta/a/ged261-csv.zip
```

El script deja en `datos/externos/` `ucdp_colombia.csv` (solo Colombia, unos cientos de KB) y
`ucdp_actores_departamento_anio.csv` (x3 agregado, unos pocos KB); basta con cualquiera de los dos. El cuaderno los
detecta y evalúa la intervención I3 (§6.3). Sin ellos, I3 queda declarada como pendiente y el resto del análisis no
cambia.

## Protocolo

* Selección con validación de origen móvil 2023–2025: cada año se predice con un modelo entrenado solo con años
  anteriores.
* La prueba 2026 (33 semanas) se evalúa una sola vez, con todas las decisiones congeladas en `configuracion.json`.
* La referencia es la persistencia del perfil. Tras la revisión se agregó su versión calibrada (logística sobre
  logit(m1) entrenada con años anteriores), que no cambió el modelo adoptado.
* Una intervención se retiene solo si mejora la log-loss más que un error estándar entre semillas pareadas
  (`evaluacion.regla_retencion`).
* Además del origen móvil, la generalización a departamentos nuevos se mide con bloqueo espacial y temporal
  (`evaluacion.particiones_espacio_temporales`).

## Reproducibilidad

Semillas fijas, `TF_ENABLE_ONEDNN_OPTS=0` y `tf.config.experimental.enable_op_determinism()`. Ejecución de referencia:
Python 3.13, TensorFlow 2.21.0 y Keras 3.15.1 (las mismas de la Unidad 2), NumPy 2.5.3, pandas 3.0.5, SciPy 1.18.1 y
scikit-learn 1.9.1, en CPU (`requirements.txt` fija estas versiones). En otro hardware las cifras pueden variar en el
tercer decimal.

## Datos y licencias

ACLED (uso académico no comercial), SIMCI-UNODC / Observatorio de Drogas de Colombia (Datos Abiertos Colombia),
Global Forest Watch (CC BY 4.0), DANE y UCDP GED (CC BY 4.0). Las salidas del modelo son un insumo analítico, no una
recomendación operativa.
