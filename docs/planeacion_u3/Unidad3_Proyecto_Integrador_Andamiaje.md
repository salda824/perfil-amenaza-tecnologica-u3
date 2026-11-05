> **Documento histórico (4-nov-2026), escrito antes de construir la U3.** El estado final y los resultados están en `CONTEXTO_PROYECTO.md` (raíz del repositorio).

# Unidad 3 — Proyecto Integrador (andamiaje, por llenar)

**Actividad:** Proyecto Integrador: Diseño, implementación y evaluación de un modelo neuronal aplicado (Redes Neuronales, Universidad de La Sabana).
**Modalidad:** grupal. **Equipo:** Laura Alejandra Díaz Sastoque (0000288870) y Juan Diego Saldarriaga Muñoz (0000287391).
**Fuente de los requisitos:** Guía de actividad de la Unidad 3, páginas 11 a 13 (archivo `Guia_Proyecto_U3_Integrador_pp11-13.pdf`).
**Proyecto que se consolida:** el Proyecto de Aplicación sobre perfil de amenaza tecnológica con ACLED (Unidad 1: formulación y datos; Unidad 2: modelo base entrenado).
**Estado general:** por iniciar. Fecha de entrega y de sustentación: por confirmar (no figuran en las páginas 11 a 13).

Este documento no contiene resultados de la Unidad 3. Reúne lo que la guía exige, lo que ya existe de las Unidades 1 y 2 para cada exigencia y lo que falta, para que el grafo muestre las brechas y se vayan llenando.

## 1. Qué debe demostrar el equipo (según la guía)

1. Coherencia entre problema, datos y modelo.
2. Aplicación rigurosa del ciclo de vida de la Ciencia de Datos.
3. Capacidad de análisis interpretativo de métricas.
4. Implementación de mejoras fundamentadas.
5. Integración estructurada del modelo final.

La sustentación sincrónica evalúa claridad comunicativa, dominio técnico y solidez argumentativa.

## 2. Producto 1: informe técnico final

- **Formato:** PDF, máximo 12 páginas, sin anexos de código.
- **Código:** se incluye como enlace a un repositorio (GitHub o similar). Estado del repositorio: por crear.
- **Estado del informe:** por escribir.

Contenido exigido, con la evidencia que ya existe y lo pendiente:

### S1. Planteamiento del problema y contexto
- **Evidencia heredada:** Unidad 1, secciones 1 (identificación del problema) y 2 (formulación y pregunta orientadora); informe de la Unidad 2, sección 1 (recordatorio sintético y trazabilidad de variables).
- **Estado:** por llenar. **Pendiente:** decidir si el alcance de la Unidad 3 se mantiene a nivel departamento–año o migra a municipio (mejora M3).

### S2. Descripción del dataset y preprocesamiento
- **Evidencia heredada:** Unidad 1, secciones 3 (variables), 4 (selección de datos) y 5 (calidad y EDA); cuaderno de la Unidad 2, secciones 2 (auditoría de ACLED), 3 (variable objetivo y panel departamento–año), 4 (fuentes externas: coca, bosque, IPM) y 5 (EDA orientado al modelado).
- **Estado:** por llenar. **Pendientes:** registro de evento individual de ACLED con municipio y actores (M3, M4), variable x3 (grupos armados activos), distancia a cabecera municipal con cartografía IGAC y refinar el mapeo de subtipos (M8).

### S3. Arquitectura del modelo y justificación
- **Evidencia heredada:** Unidad 1, sección 6 (justificación del uso de redes neuronales frente a modelos lineales); cuaderno de la Unidad 2, sección 7 (MLP 7 → 16 → 1 con 145 parámetros, elegido por validación cruzada temporal y regla de 1 error estándar).
- **Estado:** por llenar. **Pendiente:** definir la arquitectura final de la Unidad 3.

### S4. Proceso de entrenamiento y evaluación
- **Evidencia heredada:** cuaderno de la Unidad 2, secciones 6 (protocolo temporal y líneas base), 8 (bucle de entrenamiento con tf.GradientTape), 9 (curvas de aprendizaje) y 10 (métricas, intervalos bootstrap, calibración y umbral).
- **Estado:** por llenar. **Pendiente:** reentrenar con los datos y variables que resulten de S2.

### S5. Optimización realizada y comparación de resultados
- **Evidencia heredada:** cuaderno de la Unidad 2, sección 13: experimento E1 (memoria del perfil como variable x8) y E2 (ensamble de 5 semillas), y la hoja de ruta M1 a M8 (sección 13.3).
- **Estado:** por llenar. **Pendiente:** aplicar las mejoras elegidas y comparar antes y después con métricas pertinentes.

### S6. Análisis crítico del desempeño
- **Evidencia heredada:** cuaderno de la Unidad 2, sección 11 (convergencia, estabilidad con 10 semillas, generalización temporal y espacial, sesgo–varianza, aporte de la no linealidad, análisis de errores y síntesis).
- **Estado:** por llenar.

### S7. Integración final del modelo en un pipeline estructurado
- **Evidencia heredada:** el cuaderno de la Unidad 2 ejecuta en secuencia la descarga de fuentes, el panel, el entrenamiento, el ensamble y la proyección 2027, y exporta artefactos (modelos .keras, configuracion.json, proyeccion_2027.csv), pero no está empaquetado como pipeline reproducible.
- **Estado:** por llenar. Es el vacío principal respecto a la guía. **Pendiente:** decidir formato (módulos, repositorio, orden de etapas).

### S8. Conclusiones técnicas y recomendaciones futuras
- **Evidencia heredada:** cuaderno e informe de la Unidad 2 (conclusiones y hoja de ruta).
- **Estado:** por llenar.

## 3. Producto 2: video de presentación ejecutiva

- **Formato:** MP4, duración de 8 a 10 minutos.
- **Debe incluir:** síntesis del problema, principales decisiones técnicas, resultados más relevantes y conclusiones estratégicas.
- **Audiencia:** técnica, con enfoque aplicado.
- **Estado:** por producir.

## 4. Recomendaciones y reglas de la guía

- Asegurar coherencia narrativa entre todas las secciones del informe.
- Evitar resultados sin interpretación técnica.
- Priorizar claridad conceptual sobre complejidad innecesaria.
- Garantizar reproducibilidad con un código bien organizado.
- Distribuir roles técnicos: modelado, evaluación, documentación y presentación.
- **Uso de IA:** permitido como apoyo para revisión de redacción, organización del informe o contraste conceptual. No se aceptan documentos o análisis generados automáticamente sin comprensión técnica explícita. Todos los integrantes deben poder explicar y defender las decisiones y reproducir el proceso. El uso no responsable o no declarado de IA constituye falta a la integridad académica.

## 5. Roles

- **Heredados:** en la Unidad 1 y la Unidad 2, Juan Diego Saldarriaga lidera la implementación técnica (preprocesamiento, arquitectura en Keras, hiperparámetros) y Laura Díaz Sastoque la interpretación estratégica y la comunicación; las decisiones metodológicas son conjuntas.
- **Roles de la Unidad 3** (modelado, evaluación, documentación, presentación): por definir.

## 6. Rúbrica de la actividad (5 criterios, 1,0 punto cada uno; total 5,0)

Niveles: Excelente 1,0 · Bueno 0,8 · Necesita mejorar 0,7 · Deficiente 0,5 · No cumple 0,0.

| Criterio | Descriptor de nivel Excelente |
|---|---|
| Coherencia metodológica integral (problema, datos, modelo, evaluación) | Integra de forma consistente problema, datos, modelo y conclusiones con trazabilidad clara y evidencia cuantitativa. |
| Preparación y representación de datos | Evalúa y justifica técnicamente transformaciones, partición y controles experimentales, evidenciando impacto en el entrenamiento. |
| Calidad técnica del modelo y optimización | Diseña y optimiza la arquitectura con configuración documentada, curvas de desempeño e intervención estratégica basada en diagnóstico. |
| Evaluación del desempeño e interpretación | Evalúa métricas pertinentes interpretando resultados, errores y comparación experimental con rigor técnico. |
| Entregables finales y comunicación ejecutiva | Integra informe y sustentación con estructura profesional, coherencia técnica y articulación efectiva del equipo. |

### Mapeo propuesto criterio → secciones (planeación del equipo, no proviene de la guía)

- Coherencia metodológica integral: S1, S2, S3 y S8, con trazabilidad de variables de la Unidad 1 al modelo final.
- Preparación y representación de datos: S2.
- Calidad técnica del modelo y optimización: S3, S4 y S5.
- Evaluación del desempeño e interpretación: S4, S5 y S6.
- Entregables finales y comunicación ejecutiva: S7, informe completo, video y sustentación.
