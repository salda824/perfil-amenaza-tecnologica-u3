> Texto extraído del cuaderno **Proyecto_U2_Modelo_Base.ipynb** (markdown y salidas de texto; sin imágenes). Código completo en `U2_modelo_base_codigo.py`.
> Parte 3 de 3.

---
# 14. Trabajo colaborativo y bitácora de decisiones

## 14.1 Distribución de responsabilidades en la Etapa 2

Mantuvimos la división acordada al cierre de la Etapa 1: Juan Diego lidera la implementación técnica y Laura la
interpretación estratégica y la comunicación. Las decisiones metodológicas se toman en conjunto.

| Frente de trabajo | Responsable | Entregable en este notebook |
|---|---|---|
| Auditoría del export actualizado y conciliación con la Etapa 1 | Juan Diego | §2 |
| Construcción del panel, regla de etiqueta y umbral de confiabilidad | Conjunta | §3 |
| Búsqueda e integración de fuentes externas (SIMCI/ODC, GFW, DANE) | Laura | §4 |
| Diseño del protocolo temporal y de las líneas base | Conjunta | §6 |
| Implementación del MLP, del bucle de entrenamiento y de la búsqueda de hiperparámetros | Juan Diego | §7–§8 |
| Métricas, curvas y diagnóstico técnico | Juan Diego (cálculo) · Laura (interpretación) | §9–§11 |
| Traducción a decisión, alerta A/B y énfasis tecnológico | Laura | §12 |
| Priorización de mejoras y revisión final de coherencia | Conjunta | §13–§15 |

## 14.2 Bitácora de decisiones técnicas

| # | Decisión | Alternativas consideradas | Justificación | Evidencia |
|---|---|---|---|---|
| D1 | Panel departamento–año (171 filas) | Municipio (sin datos gratuitos); departamento–mes (etiquetas muy ruidosas) | Compromiso declarado en la Etapa 1; equilibra tamaño muestral y confiabilidad de la etiqueta | §3 |
| D2 | Predecir $t$ con información de $t-1$ | Predictoras del mismo año | Evita la circularidad con la etiqueta y replica el momento real de la decisión | §1.1 |
| D3 | Umbral de 10 eventos para etiquetar | 5 o 20 eventos | Error estándar de $s^C$ ≤ 0,16; conserva el 88 % de las celdas con eventos | §3.3 |
| D4 | Regla y mapeo de subtipos idénticos a la Etapa 1 | Reasignar *Shelling* y *Sexual violence* | Trazabilidad; la sensibilidad mostró impacto mínimo | §3.4 |
| D5 | $x_3$ pendiente; $x_5$ y $x_6$ con fuente sustituta/proxy | Excluir el modelo hasta tener todas las fuentes | Avanzar con 6 de 7 variables definidas y declarar el faltante | §1.2, §4 |
| D6 | Partición temporal + validación cruzada de origen móvil | Partición aleatoria estratificada | La aleatoria filtra información del futuro y sobreestima el desempeño | §6 |
| D7 | Arquitectura por regla de 1 error estándar | Elegir el mínimo absoluto | Con $n$ pequeño se prefiere el modelo más simple estadísticamente equivalente | §7.2 |
| D8 | Bucle explícito con `tf.GradientTape` | `model.fit` | Diagnósticos por época y ≈6× menos tiempo; equivalencia verificada (diferencia máx. 2×10⁻⁶) | §8 |
| D9 | Reajuste final con 2021–2025 (fase 2) | Usar solo la fase 1 | Incorpora el año más reciente ante el cambio de prevalencia | §10 |
| D10 | Reentrenamientos sin validación con presupuesto fijo de 560 pasos (mediana de 10 semillas) | Usar las 45 épocas de la semilla 42 | La semilla 42 resultó atípica (se detuvo antes que casi todas); la mediana es un criterio más robusto y no usa la prueba | §11.2 |

---
# 15. Conclusiones técnicas

1. **Coherencia con el problema.** El modelo base responde a la pregunta de la Etapa 1 con la misma variable objetivo
   (regla del 50 %, C vs. D), 6 de sus 7 predictoras más la letalidad derivada, y en la unidad de análisis que la Etapa
   1 comprometió (panel departamento–año). El diseño $t-1 \rightarrow t$ reproduce el momento real de la decisión de
   inversión y evita la circularidad entre predictoras y etiqueta.
2. **Viabilidad técnica demostrada.** Un MLP de una capa oculta (145 parámetros), elegido por validación cruzada
   temporal, aprende una señal real: AUC fuera de muestra de 0,83–0,89 frente a 0,50 del azar. Además converge de forma
   estable, es reproducible entre semillas (CV 2,8 %) y está verificado contra la implementación de Keras.
3. **El límite es la generalización, no la optimización.** La red llega al óptimo de validación antes de la meseta de
   entrenamiento, las curvas por tamaño de muestra muestran un modelo limitado por varianza, y la validación espacial
   revela que con 30 departamentos la red memoriza firmas territoriales. Mejorar el optimizador no ayudaría; más
   unidades y mejores predictoras, sí.
4. **La hipótesis de no linealidad sigue abierta.** La red representa interacciones coherentes con la Etapa 1
   (coca × pobreza, bosque × distancia) que la logística no puede trazar, pero a esta escala no se traducen en una
   ventaja predictiva significativa. Es la conclusión honesta con n = 171, y un argumento cuantitativo para migrar a
   municipio.
5. **El hallazgo central es la inercia.** El perfil se repite el 86 % de las veces y la persistencia supera a
   cualquier modelo sin memoria. Las variables de la Etapa 1 explican *dónde* es probable cada perfil, pero no *cuándo
   cambia*. Con la memoria del perfil (E1) los modelos superan a la persistencia en calidad probabilística, y el valor
   diferencial de un modelo estructural pasa a estar en anticipar transiciones. La decisión útil no es repetir el mapa
   del año pasado, sino detectar dónde va a cambiar.
6. **Decisión.** La proyección 2027 separa 10 departamentos C consolidados, 13 en transición y 10 D. Suma una alerta
   A/B para Guaviare y Cauca, que rescata el vector de explosivos y drones que la etiqueta binaria diluye, y marca 10
   discrepancias con la inercia como casos de revisión. Es un insumo para priorizar, no un reemplazo del análisis de
   inteligencia.

---
# 16. Exportación de artefactos

#### Celda de código 46: 16.1 Artefactos para el informe y para la siguiente iteración  (fuente completa en U2_modelo_base_codigo.py)
Salida de la celda:
```text
Archivo | KB
busqueda_hiperparametros.csv | 1.400000
configuracion.json | 0.700000
dataset_modelado_departamento_anio.csv | 55.000000
estabilidad_semillas.csv | 1.100000
experimento_E1_memoria.csv | 2.400000
figuras\fig01_cobertura_datos.png | 381.000000
figuras\fig02_variable_objetivo.png | 290.700000
figuras\fig03_predictoras_por_clase.png | 298.200000
figuras\fig04_dependencia_correlacion.png | 202.500000
figuras\fig05_interacciones_hipotesis.png | 149.500000
figuras\fig06_particion_temporal.png | 56.800000
figuras\fig07_busqueda_hiperparametros.png | 96.800000
figuras\fig08_arquitectura.png | 369.200000
figuras\fig09_verificacion_bucle.png | 108.400000
figuras\fig10_curvas_aprendizaje.png | 283.100000
figuras\fig11_matrices_confusion.png | 70.200000
figuras\fig12_roc_pr.png | 182.700000
figuras\fig13_calibracion.png | 148.900000
figuras\fig14_umbral.png | 114.000000
figuras\fig15_convergencia.png | 277.700000
figuras\fig16_estabilidad_semillas.png | 239.000000
figuras\fig17_generalizacion_temporal_espacial.png | 219.900000
figuras\fig18_sesgo_varianza.png | 261.100000
figuras\fig19_dependencia_parcial.png | 233.400000
figuras\fig20_importancia_permutacion.png | 87.500000
figuras\fig21_analisis_errores.png | 133.500000
figuras\fig22_mapa_proyeccion_2027.png | 227.400000
figuras\fig23_proyeccion_con_memoria.png | 176.900000
figuras_informe\R1_variable_objetivo.png | 222.000000
figuras_informe\R2_arquitectura.png | 160.500000
figuras_informe\R3_curvas_aprendizaje.png | 133.200000
figuras_informe\R4_metricas_prueba.png | 148.500000
figuras_informe\R5_convergencia_estabilidad.png | 186.800000
figuras_informe\R6_generalizacion.png | 172.800000
figuras_informe\R7_dependencia_parcial.png | 125.600000
figuras_informe\R8_proyeccion_2027.png | 127.000000
historial_entrenamiento.csv | 21.900000
intervalos_confianza_prueba.csv | 0.500000
metricas_modelos.csv | 3.000000
modelo_base_fase1.keras | 17.500000
modelo_reajustado_fase2.keras | 17.500000
proyeccion_2027.csv | 5.200000
validacion_temporal_espacial.csv | 5.800000
```

---
# Anexo A. Material sugerido para el informe PDF (máximo 8 páginas)

El informe entregado (`Informe_U2_Modelo_Base.pdf`, 8 páginas, con su versión editable `.docx`) sigue el orden de la
guía de la actividad. Sus figuras son las del **Anexo B** (`salidas_u2/figuras_informe/`), redibujadas a tamaño de
impresión con los mismos resultados de este notebook.

| Sección del informe | Figuras (Anexo B) | Tablas | Origen en el notebook |
|---|---|---|---|
| 1. Recordatorio sintético del problema | – | Tabla 1: trazabilidad U1 → U2 | §1 |
| 2. Descripción del dataset | Figura 1 (`R1_variable_objetivo`) | Tabla 2: calidad | §2–§5 |
| 3. Arquitectura y configuración | Figura 2 (`R2_arquitectura`) | Tabla 3: configuración | §6–§8 |
| 4. Métricas de entrenamiento y validación | Figura 3 (`R4_metricas_prueba`) | Tabla 4: métricas | §10 |
| 5. Curvas de aprendizaje | Figura 4 (`R3_curvas_aprendizaje`) | – | §9 |
| 6. Diagnóstico técnico | Figuras 5–7 (`R5`, `R6`, `R7`) y 8 (`R8_proyeccion_2027`) | Tabla 5: síntesis | §11–§12 |
| 7. Oportunidades de mejora | – | Tablas 6 (E1/E2) y 7 (hoja de ruta) | §13 |
| 8. Trabajo colaborativo | – | Tabla 8: responsabilidades y decisiones | §14 |

---
# Anexo B. Figuras en tamaño de impresión para el informe

Las figuras del notebook están pensadas para pantalla (14–16 pulgadas de ancho). Al llevarlas a una página carta,
su texto quedaría en 4–5 pt. Esta celda redibuja las figuras que usa el informe con el ancho útil de la página
(6,6 pulgadas) y texto de 7–8 pt, a partir de **los mismos resultados calculados arriba**, sin reentrenar nada.
Se guardan en `salidas_u2/figuras_informe/` a 300 dpi.

#### Celda de código 47: Anexo B. Figuras del informe (carta, 6,6" de ancho útil, 300 dpi)  (fuente completa en U2_modelo_base_codigo.py)
Salida de la celda:
```text
Figuras del informe guardadas en salidas_u2\figuras_informe/: R1_variable_objetivo, R2_arquitectura, R3_curvas_aprendizaje, R4_metricas_prueba, R5_convergencia_estabilidad, R6_generalizacion, R7_dependencia_parcial, R8_proyeccion_2027
```
Figuras generadas: R1_variable_objetivo, R2_arquitectura, R3_curvas_aprendizaje, R4_metricas_prueba, R5_convergencia_estabilidad, R6_generalizacion, R7_dependencia_parcial, R8_proyeccion_2027 (imagen omitida en este texto).

---
# Referencias

ACLED. (2024). *ACLED codebook*. Armed Conflict Location & Event Data Project. https://acleddata.com

Bergmeir, C., & Benítez, J. M. (2012). On the use of cross-validation for time series predictor evaluation. *Information Sciences, 191*, 192–213. https://doi.org/10.1016/j.ins.2011.12.028

Bishop, C. M., & Bishop, H. (2024). *Deep learning: Foundations and concepts*. Springer.

Centro Nacional de Memoria Histórica. (2022). *Presencia diferenciada del Estado colombiano*. CNMH.

Departamento Administrativo Nacional de Estadística. (2019). *Pobreza multidimensional por departamentos 2018: Resultados* [Presentación]. DANE. https://www.dane.gov.co

Dietterich, T. G. (1998). Approximate statistical tests for comparing supervised classification learning algorithms. *Neural Computation, 10*(7), 1895–1923. https://doi.org/10.1162/089976698300017197

Friedman, J. H. (2001). Greedy function approximation: A gradient boosting machine. *The Annals of Statistics, 29*(5), 1189–1232. https://doi.org/10.1214/aos/1013203451

Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep learning*. MIT Press.

Hansen, M. C., Potapov, P. V., Moore, R., Hancher, M., Turubanova, S. A., Tyukavina, A., … Townshend, J. R. G. (2013). High-resolution global maps of 21st-century forest cover change. *Science, 342*(6160), 850–853. https://doi.org/10.1126/science.1244693

Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The elements of statistical learning* (2.ª ed.). Springer.

He, K., Zhang, X., Ren, S., & Sun, J. (2015). Delving deep into rectifiers: Surpassing human-level performance on ImageNet classification. En *Proceedings of the IEEE International Conference on Computer Vision* (pp. 1026–1034).

Hornik, K. (1991). Approximation capabilities of multilayer feedforward networks. *Neural Networks, 4*(2), 251–257. https://doi.org/10.1016/0893-6080(91)90009-T

Kingma, D. P., & Ba, J. (2015). Adam: A method for stochastic optimization. En *International Conference on Learning Representations*. https://arxiv.org/abs/1412.6980

Ministerio de Justicia y del Derecho – Observatorio de Drogas de Colombia. (2026). *Detección de cultivos de coca (hectáreas)* [Conjunto de datos]. Datos Abiertos Colombia. https://www.datos.gov.co/d/acs4-3wgp

Prechelt, L. (1998). Early stopping — but when? En G. B. Orr & K.-R. Müller (Eds.), *Neural networks: Tricks of the trade* (pp. 55–69). Springer. https://doi.org/10.1007/3-540-49430-8_3

Raleigh, C., Linke, A., Hegre, H., & Karlsen, J. (2010). Introducing ACLED: An armed conflict location and event dataset. *Journal of Peace Research, 47*(5), 651–660. https://doi.org/10.1177/0022343310378914

Roberts, D. R., Bahn, V., Ciuti, S., Boyce, M. S., Elith, J., Guillera-Arroita, G., … Dormann, C. F. (2017). Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure. *Ecography, 40*(8), 913–929. https://doi.org/10.1111/ecog.02881

Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., & Salakhutdinov, R. (2014). Dropout: A simple way to prevent neural networks from overfitting. *Journal of Machine Learning Research, 15*, 1929–1958.

United Nations Office on Drugs and Crime. (2024). *Colombia: Monitoreo de territorios afectados por cultivos ilícitos 2023*. UNODC–SIMCI.

World Resources Institute. (2025). *Global Forest Watch: Colombia tree cover loss by subnational region* [Conjunto de datos]. https://www.globalforestwatch.org

---

**Declaración de uso de inteligencia artificial.** *[El equipo debe completar esta declaración describiendo con
honestidad qué herramientas de IA usó y para qué, en el mismo espíritu de la declaración de la Etapa 1, y confirmar
que ejecutó, revisó y comprende cada sección antes de entregar.]*

#### Celda de código 48  (fuente completa en U2_modelo_base_codigo.py)
Salida de la celda:
```text
Ejecución completa. Figuras exportadas: 23 en salidas_u2\figuras/
```
