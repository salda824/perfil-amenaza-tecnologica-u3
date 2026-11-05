# Perfil de amenaza tecnológica por departamento (Proyecto Integrador, Unidad 3)

Redes Neuronales · Universidad de La Sabana · Laura Alejandra Díaz Sastoque y Juan Diego Saldarriaga Muñoz

El modelo estima, con la información disponible al cierre de un año, la probabilidad de que cada departamento de
Colombia tenga el año siguiente perfil de amenaza **C (armas convencionales y emboscadas)** o **D (mixto)**. Esa
probabilidad orienta el énfasis tecnológico (ISR, C-UAS, detección de AEI). Este repositorio consolida las Unidades 1
(formulación) y 2 (modelo base) en un pipeline reproducible.

## Estructura

```
configuracion.json        modelo final y protocolo (lo escribe la §8.1 del cuaderno)
datos/                    export semanal de ACLED + datos/externos/ (coca SIMCI-ODC, bosque GFW; UCDP opcional)
src/perfil_amenaza/       paquete: config, datos, variables, modelo, evaluacion, pipeline, graficos
herramientas/             preparar_ucdp.py (reduce UCDP GED a Colombia)
tests/                    pruebas automáticas (contrato de datos, fuga temporal, escalador, determinismo, réplica U2)
referencias_u2/           resultados guardados de la Unidad 2 (para comparar "antes vs. después")
Proyecto_U3_Integrador.ipynb   cuaderno de la entrega: diagnóstico, intervenciones, evaluación e interpretación
salidas_u3/               métricas, predicciones, figuras, modelos/perfil_amenaza_u3.keras, TARJETA_MODELO.md
```

## Uso rápido en Windows (cmd)

```bat
ejecutar_todo.bat "C:\ruta\a\ged261-csv"
```

Crea el entorno virtual, instala las dependencias, reduce UCDP (opcional), corre las pruebas, ejecuta el cuaderno
completo y muestra la proyección 2027.

## Uso

```bash
pip install -r requirements.txt
python -m pytest -q tests                                       # 5 pruebas
PYTHONPATH=src python -m perfil_amenaza.pipeline todo           # datos -> evaluación -> artefacto -> proyección 2027
PYTHONPATH=src python -m perfil_amenaza.pipeline predecir --anio 2026   # solo con el artefacto guardado
jupyter nbconvert --to notebook --execute Proyecto_U3_Integrador.ipynb  # cuaderno completo (~20 min en CPU)
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
* El rival a vencer es la persistencia del perfil, no el modelo base.
* Una intervención se retiene solo si mejora la log-loss más que un error estándar entre semillas pareadas.

## Reproducibilidad

Semillas fijas, `TF_ENABLE_ONEDNN_OPTS=0` y `tf.config.experimental.enable_op_determinism()`. Ejecución de referencia:
Python 3.13, TensorFlow 2.21.0 y Keras 3.15.1 (las mismas versiones de TensorFlow y Keras de la Unidad 2), en CPU.

## Datos y licencias

ACLED (uso académico no comercial), SIMCI-UNODC / Observatorio de Drogas de Colombia (Datos Abiertos Colombia),
Global Forest Watch (CC BY 4.0), DANE y UCDP GED (CC BY 4.0). Las salidas del modelo son un insumo analítico, no una
recomendación operativa.
