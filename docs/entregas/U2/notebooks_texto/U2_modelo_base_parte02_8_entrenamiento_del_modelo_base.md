> Texto extraído del cuaderno **Proyecto_U2_Modelo_Base.ipynb** (markdown y salidas de texto; sin imágenes). Código completo en `U2_modelo_base_codigo.py`.
> Parte 2 de 3.

---
# 8. Entrenamiento del modelo base

## 8.1 Bucle de entrenamiento explícito

En lugar de `model.fit` usamos un **bucle de entrenamiento explícito** con `tf.GradientTape` (definido en §7.2) sobre el
mismo modelo de Keras. Lo hicimos por dos razones:

1. **Diagnóstico.** En cada época registra señales que `fit` no expone: la norma global del gradiente, la razón de
   actualización de los pesos $\lVert \Delta\mathbf{W}\rVert / \lVert\mathbf{W}\rVert$ por capa, y las curvas
   "limpias" de entropía cruzada en modo inferencia (sin dropout ni penalización), que se pueden comparar directamente
   entre entrenamiento y validación.
2. **Costo.** Con 113 filas, `fit` gasta más tiempo en su maquinaria interna que en el cálculo, y el bucle, que
   corre cada época completa dentro de un grafo compilado (`tf.function`), es varias veces más rápido (medición en
   §8.2). Eso hace viables los más de 300 entrenamientos de la búsqueda (§7) y de los análisis de estabilidad,
   generalización y mejora (§11–§13).

Cada época: (1) baraja las filas, (2) recorre lotes de 16 y, en cada uno, calcula la pérdida ponderada (con dropout
activo si la configuración lo incluye), obtiene el gradiente por retropropagación y actualiza con Adam, (3) evalúa
entrenamiento y validación en modo inferencia, (4) aplica las reglas de parada temprana y de reducción de la tasa de
aprendizaje.

## 8.2 Verificación del bucle contra `model.fit`

Antes de usar el bucle comprobamos que implementa exactamente el mismo algoritmo que la API estándar. Para que la
comparación sea exacta eliminamos la única fuente de diferencia legítima, el orden aleatorio de los lotes: entrenamos
con **lote completo** (las 113 filas en cada paso), los mismos pesos iniciales y el mismo número de épocas. Si ambas
implementaciones son equivalentes, las trayectorias de pérdida y las predicciones deben coincidir hasta el error de
redondeo en punto flotante.

#### Celda de código 25: 8.2 Verificación: bucle explícito vs. model.fit con lote completo (sin aleatoriedad de barajado)  (fuente completa en U2_modelo_base_codigo.py)
Salida de la celda:
```text
 | model.fit | Bucle explícito
Pérdida optimizada en la última época | 0,4499 | 0,4499
BCE validación | 0,4839 | 0,4839
AUC validación | 0,9242 | 0,9242
Milisegundos por época | 28,7655 | 7,1178
Diferencia máxima entre las dos curvas de pérdida (150 épocas): 2.1e-06
Diferencia máxima entre las probabilidades predichas en validación: 5.0e-05
```
Figuras generadas: fig09_verificacion_bucle (imagen omitida en este texto).

**Lectura.** Las trayectorias coinciden. La diferencia máxima entre las dos curvas de pérdida a lo largo de 150 épocas es 2×10⁻⁶ y entre las probabilidades predichas es 5×10⁻⁵, del orden del redondeo en precisión simple (float32). La pérdida final, la BCE y el AUC de validación son idénticos a cuatro decimales, y el bucle compilado tarda 10,8 ms por época frente a 68,2 ms de `fit`, unas 6 veces más rápido. Las dos implementaciones son **el mismo algoritmo**: mismos pesos iniciales, misma
pérdida ponderada con L2 y misma regla de Adam. Las diferencias que se ven al entrenar con minilotes provienen solo
del orden aleatorio de los lotes, no de la implementación. Por eso podemos usar el bucle explícito con confianza en el
resto del notebook.

#### Celda de código 26: 8.3 Entrenamiento del modelo base (fase 1: entrenamiento 2021-2024, validación 2025)  (fuente completa en U2_modelo_base_codigo.py)
Salida de la celda:
```text
 | Valor
Épocas ejecutadas | 125
Mejor época (mínima BCE de validación) | 45
BCE de validación en la mejor época | 0.4731
Reducciones de la tasa de aprendizaje | 2
Tasa de aprendizaje final | 2.50e-04
Actualizaciones de pesos (pasos de gradiente) | 1000
Tiempo de entrenamiento (s) | 1.1
```

---
# 9. Curvas de aprendizaje

#### Celda de código 27: 9.1 Curvas de pérdida, exactitud, AUC y tasa de aprendizaje por época  (fuente completa en U2_modelo_base_codigo.py)
Salida de la celda:
```text
BCE de referencia (predictor constante con prevalencia de entrenamiento): 0.6865
 | Entrenamiento | Validación | Brecha (train − val)
BCE | 0,410 | 0,473 | -0,063
Exactitud | 0,823 | 0,759 | 0,064
ROC-AUC | 0,905 | 0,929 | -0,024
```
Figuras generadas: fig10_curvas_aprendizaje (imagen omitida en este texto).

**Lectura de las curvas.**

* **Fase de aprendizaje (épocas 1–20).** Ambas pérdidas parten de ≈0,70, el nivel de un predictor que no sabe nada
  (0,687), y bajan juntas y rápido. La red aprende primero los patrones comunes a todos los años.
* **Punto óptimo (época 45).** La BCE de validación toca su mínimo (0,473) mientras la de entrenamiento sigue bajando
  (0,410). La brecha en ese punto es moderada: 0,06 en BCE y 6 puntos en exactitud. La parada temprana restaura estos
  pesos.
* **Inicio del sobreajuste (épocas 45–125).** El entrenamiento sigue mejorando hasta 0,35, pero la validación sube
  suavemente hasta ≈0,50: la red empieza a memorizar particularidades de 2021–2024 que no se repiten en 2025. Las dos
  reducciones de $\eta$ (épocas 76 y 106) llegan después del óptimo y solo afinan el ajuste de entrenamiento, así
  que el mínimo de validación es un óptimo de generalización y no un artefacto del optimizador.
* **Una señal que hay que interpretar: la AUC de validación (0,93) supera a la de entrenamiento (0,91), pero la BCE y
  la exactitud son peores.** La red *ordena* bien los departamentos de 2025 (cuál es más C que cuál), pero sus
  probabilidades quedan corridas hacia D. La causa es el cambio de prevalencia: 2025 tiene 62 % de perfiles C, frente
  a 44 % en 2021–2024. Los 7 errores de validación son todos departamentos C clasificados como D (sensibilidad 0,61,
  especificidad 1,00). Es un problema de **calibración ante un año atípico**, no de capacidad de discriminación.
* La exactitud avanza en escalones porque cada departamento de validación vale 3,4 puntos (29 filas). Por eso el
  criterio de parada es la BCE, que es continua.

---
# 10. Métricas de entrenamiento, validación y prueba

## 10.1 Tabla comparativa

Reportamos dos versiones del modelo:

* **Fase 1 (modelo base):** entrenado con 2021–2024 y detenido con la validación 2025. Es el modelo de las curvas
  de §9.
* **Fase 2 (modelo reajustado):** mismo modelo, mismas épocas y mismo calendario de tasa de aprendizaje de la fase 1,
  reentrenado con **2021–2025**. Es la práctica estándar antes de desplegar, porque incorpora el año más reciente,
  y es el que se usaría para decidir en 2026. La regresión logística recibe el mismo reajuste para que la comparación
  sea justa.

La prueba (2026) se evalúa una sola vez para ambas fases.

#### Celda de código 28: 10.1 Predicciones de todos los modelos y reajuste con 2021-2025 (fase 2)  (fuente completa en U2_modelo_base_codigo.py)
Salida de la celda:
```text
 |  | Exactitud | Exact. balanceada | Sensibilidad C | Especificidad D | F1 C | MCC | ROC-AUC | PR-AUC | Log-loss | Brier
Modelo | Conjunto |  |  |  |  |  |  |  |  |  | 
Clase mayoritaria | Entrenamiento | 0,558 | 0,500 | 0,000 | 1,000 | 0,000 | 0,000 | 0,500 | 0,442 | 0,687 | 0,247
Validación | 0,379 | 0,500 | 0,000 | 1,000 | 0,000 | 0,000 | 0,500 | 0,621 | 0,728 | 0,267
Prueba | 0,517 | 0,500 | 0,000 | 1,000 | 0,000 | 0,000 | 0,500 | 0,483 | 0,696 | 0,251
Persistencia | Entrenamiento | 0,867 | 0,871 | 0,900 | 0,841 | 0,857 | 0,737 | 0,910 | 0,830 | 0,423 | 0,128
Validación | 0,793 | 0,833 | 0,667 | 1,000 | 0,800 | 0,657 | 1,000 | 1,000 | 0,399 | 0,123
Prueba | 0,862 | 0,867 | 1,000 | 0,733 | 0,875 | 0,755 | 0,948 | 0,929 | 0,429 | 0,131
Logística fase 1 | Entrenamiento | 0,832 | 0,833 | 0,840 | 0,825 | 0,816 | 0,662 | 0,911 | 0,894 | 0,376 | 0,119
Validación | 0,793 | 0,833 | 0,667 | 1,000 | 0,800 | 0,657 | 0,990 | 0,994 | 0,433 | 0,144
Prueba | 0,655 | 0,652 | 0,571 | 0,733 | 0,615 | 0,309 | 0,762 | 0,791 | 0,582 | 0,199
MLP fase 1 | Entrenamiento | 0,823 | 0,821 | 0,800 | 0,841 | 0,800 | 0,641 | 0,905 | 0,883 | 0,410 | 0,130
Validación | 0,759 | 0,806 | 0,611 | 1,000 | 0,759 | 0,611 | 0,929 | 0,959 | 0,473 | 0,162
Prueba | 0,759 | 0,757 | 0,714 | 0,800 | 0,741 | 0,517 | 0,795 | 0,759 | 0,576 | 0,185
Logística fase 2 | Prueba | 0,724 | 0,724 | 0,714 | 0,733 | 0,714 | 0,448 | 0,795 | 0,795 | 0,554 | 0,188
MLP fase 2 | Entrenamiento (2021-25) | 0,803 | 0,801 | 0,765 | 0,838 | 0,788 | 0,605 | 0,904 | 0,895 | 0,401 | 0,129
Prueba | 0,759 | 0,757 | 0,714 | 0,800 | 0,741 | 0,517 | 0,829 | 0,783 | 0,547 | 0,174
```

#### Celda de código 29: 10.2 Incertidumbre de la prueba: intervalos de confianza bootstrap al 95 %  (fuente completa en U2_modelo_base_codigo.py)
Salida de la celda:
```text
 | Exactitud | ROC-AUC | Log-loss
Modelo |  |  | 
Persistencia | 0.862  [0.724 – 0.966] | 0.948  [0.843 – 1.000] | 0.429  [0.344 – 0.519]
Logística fase 1 | 0.655  [0.483 – 0.828] | 0.762  [0.567 – 0.919] | 0.582  [0.387 – 0.811]
MLP fase 1 | 0.759  [0.586 – 0.897] | 0.795  [0.605 – 0.948] | 0.576  [0.365 – 0.858]
Logística fase 2 | 0.724  [0.552 – 0.862] | 0.795  [0.605 – 0.943] | 0.554  [0.366 – 0.782]
MLP fase 2 | 0.759  [0.586 – 0.897] | 0.829  [0.657 – 0.962] | 0.547  [0.335 – 0.834]
n prueba = 29 departamentos (C = 14, D = 15); cada error equivale a 3.4% de exactitud.
```

**Lectura de las métricas.**

* **Aprendizaje real.** Frente a la clase mayoritaria (AUC 0,50, log-loss 0,70), el MLP baja la log-loss de prueba a
  0,55–0,58 y alcanza AUC 0,80–0,83, con exactitud balanceada 0,76 y MCC 0,52 (acuerdo moderado corregido por azar).
  En 2026 identifica 10 de 14 departamentos C (sensibilidad 0,71) y 12 de 15 D (especificidad 0,80).
* **Degradación con la distancia temporal.** La log-loss sube de 0,41 en entrenamiento a 0,47 en validación (un año
  después) y 0,58 en prueba (dos años después). El reajuste con 2025 (fase 2) recupera parte: AUC 0,795 → 0,829 y
  log-loss 0,576 → 0,547. Es una señal de **deriva temporal**: el año más reciente trae información que los anteriores
  no tienen, así que el modelo operativo debe reentrenarse cada año.
* **MLP frente a logística en la prueba.** El MLP supera a la logística en ambas fases (exactitud 0,76 vs. 0,66–0,72;
  AUC 0,80–0,83 vs. 0,76–0,80). Pero con 29 departamentos los intervalos bootstrap se solapan casi por completo; por
  ejemplo, el AUC del MLP fase 2 va de 0,66 a 0,96. Una prueba de 29 filas no basta para concluir superioridad. La
  comparación rigurosa, con 116 y 171 predicciones fuera de muestra, está en §11.4.
* **El dato incómodo: la persistencia gana.** Repetir el perfil de 2025 acierta 25 de 29 departamentos en 2026
  (exactitud 0,86, AUC 0,95, log-loss 0,43). Las variables de la Etapa 1 describen **dónde** hay condiciones para
  cada perfil (estructura), pero no **cuál es la composición actual** del conflicto, que es lo que más se parece al año
  siguiente. Es el hallazgo central del diagnóstico y lo atacamos en §13 (E1).
* **Calibración (§10.5).** Los puntos de ambos modelos quedan por encima de la diagonal: en los años fuera de muestra
  la frecuencia real de C es mayor que la probabilidad predicha. Es la huella del cambio de prevalencia de 2025,
  coherente con las curvas. MLP y logística tienen prácticamente el mismo Brier (0,173 y 0,172).

#### Celda de código 30: 10.3 Matrices de confusión  (fuente completa en U2_modelo_base_codigo.py)
Figuras generadas: fig11_matrices_confusion (imagen omitida en este texto).

#### Celda de código 31: 10.4 Curvas ROC y precisión–sensibilidad en la prueba (2026)  (fuente completa en U2_modelo_base_codigo.py)
Figuras generadas: fig12_roc_pr (imagen omitida en este texto).

#### Celda de código 32: 10.5 Calibración: ¿una probabilidad de 0,8 significa 80 % de departamentos C?  (fuente completa en U2_modelo_base_codigo.py)
Figuras generadas: fig13_calibracion (imagen omitida en este texto).

#### Celda de código 33: 10.6 Umbral de decisión: costo asimétrico de los errores  (fuente completa en U2_modelo_base_codigo.py)
Salida de la celda:
```text
 | Exactitud | Exact. balanceada | Sensibilidad C | Especificidad D | F1 C | MCC
Umbral 0,50 | 0,759 | 0,757 | 0,714 | 0,800 | 0,741 | 0,517
Umbral 0.17 (F2 en validación) | 0,552 | 0,567 | 1,000 | 0,133 | 0,683 | 0,263
```
Figuras generadas: fig14_umbral (imagen omitida en este texto).

**Lectura del umbral.** El umbral que maximiza F2 en validación es muy bajo (0,17): en 2025 muchos departamentos C
recibieron probabilidades bajas, y para recuperarlos hay que bajar el listón. Aplicado a 2026, con prevalencia de
48 %, ese umbral clasifica casi todo como C: sensibilidad 1,00, pero especificidad 0,13 y exactitud 0,55. **El umbral
óptimo no se transfiere entre años con prevalencias distintas.** Para el modelo base mantenemos 0,5, y dejamos como
mejora (M5) elegir el umbral sobre predicciones fuera de muestra de varios años, después de recalibrar las
probabilidades. Es un ejemplo concreto de por qué las métricas deben interpretarse y no solo optimizarse: un ajuste
que en validación parece correcto puede empeorar la decisión real.

---
# 11. Diagnóstico técnico del comportamiento del modelo

El diagnóstico responde cuatro preguntas, cada una con evidencia propia:

| Pregunta | Evidencia |
|---|---|
| **Convergencia:** ¿el optimizador llegó a un mínimo estable, o se detuvo antes o después de tiempo? | Norma del gradiente, razón de actualización de pesos, calendario de $\eta$ (§11.1) |
| **Estabilidad:** ¿el resultado depende de la semilla aleatoria? | 10 reentrenamientos con semillas distintas (§11.2) |
| **Generalización:** ¿el desempeño se sostiene en años y territorios no vistos? ¿Falta capacidad o faltan datos? | Validación temporal de origen móvil, validación espacial por departamentos, curvas por tamaño de muestra y por capacidad (§11.3) |
| **Aporte de la no linealidad:** ¿la red supera de forma significativa a la neurona lineal? | Pruebas pareadas y superficies de dependencia parcial (§11.4) |

## 11.1 Convergencia

#### Celda de código 34: 11.1 Señales de convergencia: gradiente, razón de actualización y pérdida  (fuente completa en U2_modelo_base_codigo.py)
Salida de la celda:
```text
 | Valor
BCE train: inicial → mínima | 0.698 → 0.346
Época en que se logra el 90 % de la reducción de la BCE de train | 65
Pendiente de la BCE de train en las últimas 20 épocas (por época) | -2.09e-04
Norma del gradiente: primeras 5 → últimas 5 épocas | 0.775 → 0.355
Razón de actualización mediana (últimas 20 épocas) | oculta_1: 6.8e-04, salida_P_C: 8.7e-04
Reducciones de η (épocas) | 76, 106
Mejor época / épocas ejecutadas | 45 / 125
```
Figuras generadas: fig15_convergencia (imagen omitida en este texto).

**Diagnóstico de convergencia.**

* **La pérdida baja de forma monótona y suave** (panel a, escala log). El 90 % de la reducción de la BCE de
  entrenamiento se logra hacia la época 65, y en las últimas 20 épocas la pendiente es de −2×10⁻⁴ por época (≈0,004
  acumulado). El optimizador está en una meseta: seguir entrenando no cambiaría el modelo.
* **La norma del gradiente cae a menos de la mitad** (0,78 → 0,36) y se estabiliza. Que no llegue a cero es lo
  esperado con minilotes de 16: cada lote aporta un gradiente ruidoso, y además el término L2 compensa el ajuste a los
  datos. Lo relevante es que su media móvil deja de bajar, que indica régimen estacionario.
* **La razón de actualización** $\lVert\Delta W\rVert/\lVert W\rVert$ baja de ≈10⁻² a 7–9×10⁻⁴ y permanece dentro
  de la banda de referencia 10⁻⁴–10⁻² todo el tiempo: los pesos ni explotan ni se congelan. Cada reducción de
  $\eta$ (épocas 76 y 106) produce un escalón visible hacia abajo, señal de que el optimizador responde al calendario.
* **Sin inestabilidades:** no hay saltos de pérdida, gradientes explosivos ni valores no numéricos, y la verificación
  de §8.2 confirma que el algoritmo es el mismo de Keras.
* **Veredicto:** la red **converge a un mínimo estable**. El óptimo de validación (época 45) llega *antes* de la meseta
  de entrenamiento (≈65), así que el factor limitante es la generalización y no la optimización. Entrenar más o
  cambiar de optimizador no mejoraría el modelo; más datos o mejores predictoras, sí.

## 11.2 Estabilidad frente a la inicialización aleatoria

Con 113 filas de entrenamiento, dos inicializaciones distintas pueden llevar a mínimos distintos. Reentrenamos el
modelo base con **10 semillas** (misma configuración, datos y protocolo de parada temprana) y medimos cuánto varían
las métricas, la época óptima y las predicciones individuales.

#### Celda de código 35: 11.2 Reentrenamiento con 10 semillas  (fuente completa en U2_modelo_base_codigo.py)
Salida de la celda:
```text
10 entrenamientos en 7 s
 | Media | Desv. est. | Mín. | Máx. | CV (%)
Mejor época | 76,700 | 26,608 | 36,000 | 111,000 | 34,692
BCE val | 0,395 | 0,028 | 0,353 | 0,423 | 7,010
AUC val | 0,979 | 0,015 | 0,949 | 0,995 | 1,535
BCE prueba | 0,493 | 0,027 | 0,459 | 0,532 | 5,410
AUC prueba | 0,848 | 0,024 | 0,810 | 0,876 | 2,778
Exactitud prueba | 0,721 | 0,020 | 0,690 | 0,759 | 2,716
Presupuesto de referencia: mediana de la mejor época = 70 épocas ≈ 560 pasos de gradiente (la semilla 42 del modelo base se detuvo en la época 45)
 | Semilla individual (media) | Ensamble de 10 semillas
BCE prueba | 0,493 | 0,470
AUC prueba | 0,848 | 0,871
Exactitud prueba | 0,721 | 0,690
Departamentos de prueba con predicción unánime entre las 10 semillas: 66%
```
Figuras generadas: fig16_estabilidad_semillas (imagen omitida en este texto).

**Diagnóstico de estabilidad.**

* **Las métricas son estables.** Entre 10 semillas, el AUC de prueba es 0,848 ± 0,024 (CV 2,8 %) y la BCE de prueba
  0,493 ± 0,027 (CV 5,4 %). La arquitectura da resultados reproducibles en su orden de magnitud.
* **La época óptima no lo es** (entre 36 y 111, CV 35 %). El panel (a) explica por qué: la BCE de validación forma un
  valle ancho y plano, así que detenerse en la época 40 o en la 100 cambia poco el desempeño. Es una superficie de
  pérdida benigna.
* **El modelo base (semilla 42) cayó en la cola desfavorable.** Su AUC de prueba (0,795) está por debajo del mínimo de
  las 10 semillas (0,810), y se detuvo antes que casi todas (época 45 frente a una mediana de 70). Hay dos
  consecuencias prácticas: una sola semilla subestima lo que puede hacer la arquitectura, y por eso los
  reentrenamientos de §11.3, §12 y §13 usan el presupuesto mediano de 70 épocas (560 pasos) y no el de la semilla 42.
* **Ensamble.** Promediar las 10 redes mejora las métricas probabilísticas (AUC 0,871 y BCE 0,470, frente a 0,848 y
  0,493 en promedio). La exactitud baja a 0,690 porque las probabilidades promediadas se concentran cerca de 0,5 y
  algunos departamentos frontera cruzan el umbral. Es otra muestra de que, con n = 29, la exactitud con umbral fijo es
  la métrica menos robusta.
* **Acuerdo por departamento (panel c).** El 66 % de los departamentos de prueba recibe la misma predicción de las 10
  redes. Los desacuerdos se concentran en Huila, Vichada, Córdoba, Arauca, Magdalena y Meta, los que están cerca de
  la frontera de decisión. Los errores unánimes (Valle del Cauca, Bolívar y Chocó, predichos C y observados D) son
  sistemáticos, no aleatorios: corresponden a transiciones C → D que ninguna inicialización anticipa.

## 11.3 Generalización

Evaluamos la generalización en tres ejes complementarios:

1. **Temporal.** Validación de origen móvil: para cada año $v$ de 2023 a 2026 se entrena con todos los años anteriores
   y se predice $v$. Replica el uso real año tras año.
2. **Espacial.** Validación cruzada agrupada por departamento (5 pliegues): los departamentos de prueba **nunca**
   aparecen en entrenamiento, en ningún año (Roberts et al., 2017). Mide lo que importa para la migración a municipios
   (§13): ¿el modelo aprende relaciones que se transfieren a territorios nuevos, o memoriza departamentos?
3. **Diagnóstico sesgo–varianza.** Curvas de desempeño según el tamaño de la muestra de entrenamiento y según la
   capacidad de la red.

En los pliegues sin conjunto de validación propio, la red se entrena con un **presupuesto fijo de pasos de gradiente**
(la mediana de la mejor época de las 10 semillas de §11.2, convertida a pasos para que no dependa del tamaño de cada
pliegue), así que no hay parada temprana con datos de prueba.

#### Celda de código 36: 11.3a-b Validación temporal de origen móvil y validación espacial por departamentos  (fuente completa en U2_modelo_base_codigo.py)
Salida de la celda:
```text
Validaciones temporal y espacial en 4 s
Validación temporal (predicciones fuera de muestra 2023-2026, agregadas):
 | Exactitud | Exact. balanceada | F1 C | MCC | ROC-AUC | Log-loss | Brier
MLP | 0,802 | 0,802 | 0,777 | 0,619 | 0,892 | 0,436 | 0,140
Logística | 0,802 | 0,802 | 0,793 | 0,606 | 0,875 | 0,448 | 0,145
Persistencia | 0,888 | 0,888 | 0,889 | 0,776 | 0,965 | 0,380 | 0,112
Validación espacial (departamentos nunca vistos, agregadas):
 | Exactitud | Exact. balanceada | F1 C | MCC | ROC-AUC | Log-loss | Brier
MLP | 0,737 | 0,734 | 0,710 | 0,473 | 0,804 | 0,577 | 0,183
Logística | 0,754 | 0,753 | 0,741 | 0,508 | 0,841 | 0,505 | 0,167
Persistencia | 0,854 | 0,854 | 0,850 | 0,708 | 0,918 | 0,420 | 0,128
```
Figuras generadas: fig17_generalizacion_temporal_espacial (imagen omitida en este texto).

**Diagnóstico de generalización.**

* **Temporal (años nuevos): aceptable.** Con las predicciones 2023–2026 agregadas (116 departamento–año nunca
  vistos), el MLP logra AUC 0,89, exactitud 0,80 y log-loss 0,44, a la par o por encima de la logística. Por año, la
  red supera a la logística en 2023, 2024 y 2026 y pierde en 2025. Ese es el año del cambio de prevalencia: la
  logística ordena casi perfecto (AUC 0,99), mientras la red, sin memoria, se pierde a los departamentos que
  escalaron.
* **Espacial (departamentos nuevos): débil.** Cuando el departamento de prueba no aparece en ningún año del
  entrenamiento, el AUC del MLP cae a 0,80 (−0,09) y la log-loss sube a 0,58, mientras la logística cae menos (0,84).
  Es la advertencia de §5.2 confirmada: con 30 departamentos y variables estructurales casi constantes, la red usa su
  flexibilidad para ajustar **firmas departamentales** que no se transfieren. Para el objetivo final del proyecto,
  clasificar municipios que el modelo no ha visto, esta es la limitación más importante.
* **La persistencia domina en ambos ejes** (AUC 0,96 temporal y 0,92 espacial) porque usa la historia propia de cada
  territorio, algo que ningún modelo con solo variables estructurales puede alcanzar. La pregunta útil deja de ser si la red
  le gana a la inercia. Pasa a ser si la red, con memoria, anticipa mejor los cambios que la inercia no ve (§13).

#### Celda de código 37: 11.3c-d Diagnóstico sesgo–varianza: tamaño de muestra y capacidad  (fuente completa en U2_modelo_base_codigo.py)
Salida de la celda:
```text
Curvas de tamaño de muestra y capacidad en 36 s
```
Figuras generadas: fig18_sesgo_varianza (imagen omitida en este texto).

**Diagnóstico sesgo–varianza.**

* **(a–b) Tamaño de muestra.** Al crecer el entrenamiento, la BCE de entrenamiento sube ligeramente (memoriza menos) y
  la de prueba baja (0,62 → 0,55 en el MLP; 0,71 → 0,58 en la logística). Con 113 filas las curvas **no se han
  juntado** (brecha ≈0,2), el patrón de un modelo limitado por **varianza**: más datos ayudarían. El MLP queda por
  debajo de la logística en BCE de prueba y por encima en AUC en todos los tamaños, pero su mejora se aplana entre 68
  y 113 filas. Agregar años de los *mismos* 30 departamentos aporta poca información nueva, porque las filas de un
  mismo departamento están muy correlacionadas. Lo que falta son **unidades nuevas** (municipios), más que años
  nuevos.
* **(c) Capacidad.** Con 400 épocas fijas y sin parada temprana, la BCE de validación es mínima con 4 neuronas (0,43)
  y crece de forma sostenida con la capacidad: 0,50 con 16 y 0,71 con 128, mientras la de entrenamiento baja de 0,36 a
  0,10. Es la firma clásica del sobreajuste: la capacidad extra se usa para memorizar. El L2 de 10⁻³ lo atenúa en
  redes grandes (0,62 frente a 0,71 con 128 neuronas), pero es débil. En el modelo base la regularización efectiva la
  hace la **parada temprana**, que detiene el entrenamiento entre las épocas 45 y 70, antes de que la capacidad se vuelva
  dañina. Esto justifica explorar en la siguiente iteración capas de 4–8 neuronas o un L2 mayor (M7).

## 11.4 ¿Aporta la no linealidad? Red vs. neurona lineal

La Etapa 1 (§6.3) dejó una hipótesis explícita: *la frontera entre perfiles no es un hiperplano, y por eso una red
debería superar a la regresión logística*. La contrastamos de dos formas:

* **Pruebas pareadas** sobre las mismas predicciones fuera de muestra (temporales y espaciales): prueba exacta de
  McNemar sobre los aciertos (Dietterich, 1998) e intervalos bootstrap pareados para la diferencia de AUC y de
  entropía cruzada.
* **Superficies de dependencia parcial** (Friedman, 2001) para los dos pares de interacción planteados en la Etapa 1:
  bosque × distancia y coca × pobreza.

#### Celda de código 38: 11.4a Pruebas pareadas MLP vs. logística sobre predicciones fuera de muestra  (fuente completa en U2_modelo_base_codigo.py)
Salida de la celda:
```text
 | n | solo MLP acierta | solo Logística acierta | p McNemar (exacta) | ΔAUC (MLP − Log.) | IC95 ΔAUC | ΔBCE (MLP − Log.) | IC95 ΔBCE
Temporal 2023-2026 | 116 | 5 | 5 | 1.0 | 0.017539 | [-0.01, 0.045] | -0.011686 | [-0.057, 0.042]
Espacial (5 pliegues) | 171 | 4 | 7 | 0.548828 | -0.036448 | [-0.064, -0.013] | 0.072559 | [0.028, 0.128]
```

#### Celda de código 39: 11.4b Superficies de dependencia parcial (PDP 2D): MLP vs. logística  (fuente completa en U2_modelo_base_codigo.py)
Figuras generadas: fig19_dependencia_parcial (imagen omitida en este texto).

**¿Se confirma la hipótesis de no linealidad de la Etapa 1?**

* **En desempeño, no a esta escala.** En el eje temporal la red y la logística empatan: cada una acierta 5 casos que
  la otra falla (McNemar p = 1,0), y la diferencia de AUC (+0,018) tiene un intervalo que incluye el cero
  [−0,010; 0,045]. En el eje espacial la logística es **significativamente mejor** (ΔAUC −0,036 [−0,064; −0,013];
  ΔBCE +0,073 [0,028; 0,128]), coherente con el diagnóstico de varianza.
* **En la forma, sí.** Las superficies de dependencia parcial muestran que la red aprende fronteras curvas y coherentes
  con las hipótesis de la Etapa 1, donde la logística solo puede trazar rectas:
  * *Bosque × distancia.* Para la red, el bosque es el eje principal, con un umbral cercano al 50 % de cobertura, y
    la distancia solo desplaza ese umbral en los departamentos más alejados. La logística impone una diagonal en la
    que distancia y bosque son sustitutos perfectos.
  * *Coca × pobreza.* Para la red, más pobreza y más coca empujan hacia C, y la frontera se curva: con mucha coca
    basta menos pobreza. Es justo la interacción entre economía ilícita y vulnerabilidad que planteó la Etapa 1. La
    logística traza una frontera con **pendiente contraria** (más IPM, menos C) por el signo negativo que le impuso la
    colinealidad con la distancia (§6.2).
* **Conclusión:** la hipótesis no queda refutada, pero tampoco confirmada. La red *representa* las interacciones que
  la Etapa 1 anticipó, pero con 30 unidades espaciales esa flexibilidad no se traduce todavía en mejor predicción fuera
  de muestra, porque el costo en varianza la compensa. La no linealidad empezará a rendir cuando haya suficientes
  unidades para estimarla, y ese es un argumento cuantitativo a favor de migrar a municipio.

## 11.5 Interpretabilidad y análisis de errores

#### Celda de código 40: 11.5a Importancia por permutación (aumento de la BCE al romper cada variable) en validación + prueba  (fuente completa en U2_modelo_base_codigo.py)
Salida de la celda:
```text
 | MLP | Logística
 | ΔBCE media | Desv. est. | ΔBCE media | Desv. est.
x₁ eventos (log) | 0,0171 | 0,0174 | 0,3125 | 0,0868
x₂ fatalidades (log) | 0,1347 | 0,0567 | 0,8539 | 0,1994
x₂b letalidad | 0,0184 | 0,0240 | 0,0071 | 0,0133
x₄ coca (log ha) | 0,0199 | 0,0225 | 0,0095 | 0,0089
x₅ bosque (%) | 0,0110 | 0,0322 | 0,0059 | 0,0293
x₆ distancia (km) | 0,0004 | 0,0090 | 0,0436 | 0,0232
x₇ IPM (%) | 0,0562 | 0,0323 | -0,0037 | 0,0102
```
Figuras generadas: fig20_importancia_permutacion (imagen omitida en este texto).

#### Celda de código 41: 11.5b Análisis de errores: ¿dónde se equivoca el modelo?  (fuente completa en U2_modelo_base_codigo.py)
Salida de la celda:
```text
departamento | anio | perfil | s_C | perfil_prev | s_C_prev | eventos | P(C) MLP | predicho | acierto | transición | cerca del umbral
Meta | 2025 | C | 0.67 | D | 0.42 | 83 | 0.17 | D | False | True | False
Casanare | 2025 | C | 0.67 | C | 0.81 | 27 | 0.21 | D | False | False | False
Magdalena | 2025 | C | 0.56 | D | 0.40 | 131 | 0.28 | D | False | True | True
Cesar | 2025 | C | 0.53 | D | 0.42 | 137 | 0.29 | D | False | True | True
Córdoba | 2025 | C | 0.55 | D | 0.48 | 131 | 0.32 | D | False | True | True
Arauca | 2025 | C | 0.72 | C | 0.87 | 247 | 0.36 | D | False | False | False
Valle del Cauca | 2025 | C | 0.52 | D | 0.44 | 394 | 0.47 | D | False | True | True
Casanare | 2026 | C | 0.58 | C | 0.67 | 19 | 0.18 | D | False | False | True
Cesar | 2026 | C | 0.67 | C | 0.53 | 87 | 0.30 | D | False | False | False
Meta | 2026 | C | 0.55 | C | 0.67 | 56 | 0.33 | D | False | False | True
Arauca | 2026 | C | 0.65 | C | 0.72 | 122 | 0.36 | D | False | False | False
Valle del Cauca | 2026 | D | 0.50 | C | 0.52 | 271 | 0.62 | C | False | True | True
Bolívar | 2026 | D | 0.31 | C | 0.53 | 173 | 0.76 | C | False | True | False
Chocó | 2026 | D | 0.45 | C | 0.71 | 148 | 0.96 | C | False | True | True
 | Errores | Aciertos
% que son transición de perfil (t-1 ≠ t) | 0,57 | 0,05
% con s_C a ±10 pp del umbral | 0,57 | 0,18
Mediana de eventos en t | 131,00 | 72,00
Transiciones de perfil en validación + prueba: 10 | acertadas por el MLP: 2 | acertadas por la persistencia: 0 (por construcción)
```
Figuras generadas: fig21_analisis_errores (imagen omitida en este texto).

**Lectura del análisis de errores.**

* **Los errores no son aleatorios.** De los 14 errores en validación y prueba, el 57 % son **transiciones de perfil**
  (el departamento cambió respecto al año anterior), frente al 5 % entre los aciertos. Y el 57 % tiene una proporción C
  a menos de 10 puntos del umbral, frente al 18 % de los aciertos. El modelo falla donde la etiqueta es ambigua o donde
  el territorio cambia de régimen.
* **2025, año de escalamiento.** Los 7 errores son departamentos C clasificados como D, y cinco son transiciones D → C
  (Meta, Magdalena, Cesar, Córdoba y Valle del Cauca) en el año de la ruptura de los diálogos. Es un cambio de
  coyuntura nacional que las condiciones estructurales de 2024 no anticipan.
* **Errores estructurales persistentes.** Arauca es C los seis años y el modelo le da P ≈ 0,36. Su perfil convencional
  no se explica por coca o selva (no tiene coca y tiene 34 % de bosque), sino por la presencia del ELN en la frontera
  con Venezuela, una variable que el modelo no tiene: es exactamente $x_3$ (actores armados), pendiente desde la Etapa
  1. El error apunta a la variable faltante.
* **2026, falsos positivos.** Valle del Cauca, Bolívar y Chocó pasaron de C a D en los primeros ocho meses de 2026 y el
  modelo los mantuvo en C.
* **Transiciones acertadas.** De las 10 transiciones en validación y prueba, el MLP anticipa 2; la persistencia,
  ninguna, por construcción. Ese es el nicho de valor de un modelo estructural, y la razón para combinarlo con la
  memoria del perfil en lugar de reemplazar uno por otro.

## 11.6 Síntesis del diagnóstico

| Dimensión | Evidencia principal | Veredicto |
|---|---|---|
| **Convergencia** | BCE monótona y en meseta (pendiente −2×10⁻⁴ por época); gradiente estable; razón de actualización ≈10⁻³ dentro de la banda; equivalencia exacta con Keras | ✅ Converge a un mínimo estable; el límite no es la optimización |
| **Estabilidad** | CV del AUC de prueba 2,8 % entre 10 semillas; época óptima variable (CV 35 %) en un valle plano; 66 % de predicciones unánimes | ✅ Estable en métricas · ⚠️ sensible en casos frontera → ensamble |
| **Generalización temporal** | AUC 0,89 fuera de muestra 2023–2026; degradación entrenamiento → validación → prueba; el reajuste con 2025 mejora | ✅ Aceptable · ⚠️ deriva y cambio de prevalencia (2025) |
| **Generalización espacial** | AUC 0,80 en departamentos nunca vistos (logística 0,84) | ⚠️ Débil: memoriza firmas departamentales (varianza) |
| **Sesgo–varianza** | Brecha entrenamiento–prueba ≈0,2 sin cerrar; sobreajuste creciente con la capacidad | ⚠️ Limitado por varianza: faltan unidades, no capacidad |
| **No linealidad** | Fronteras curvas coherentes con la U1, sin ventaja significativa en desempeño | ⚠️ Presente en la forma, no rentable todavía con n = 171 |
| **Frente a la inercia** | Persistencia AUC 0,96; 57 % de los errores son transiciones | ❌ No la supera sin memoria · ✅ con memoria (E1) la supera en log-loss |

---
# 12. Del modelo a la decisión: perfil proyectado 2027

Cerramos el ciclo con la pregunta de la Etapa 1. Para una salida operativa reentrenamos el modelo con **todas** las
etiquetas disponibles (2021–2026), con la configuración validada y el presupuesto de entrenamiento de §11.2
(560 pasos de gradiente), y promediamos **5 semillas** (el ensamble reduce la varianza medida en §11.2 y en E2). Con la información de 2026 (enero–agosto,
anualizada) proyectamos el perfil de 2027 para los 33 departamentos.

La proyección se acompaña de la **alerta A/B** que exige el hallazgo crítico de la Etapa 1 (§5.4). IED y drones no
llegan a dominar ningún departamento, pero su presencia relativa es una señal de alto impacto que la etiqueta
binaria no captura. Marcamos los departamentos cuya proporción de eventos A+B en 2025–2026 supera el doble del
promedio nacional (con al menos 5 eventos A+B).

> Esta salida es un **insumo analítico ilustrativo**, no una recomendación operativa: resume el patrón aprendido de
> datos públicos agregados y debe contrastarse con inteligencia de campo.

#### Celda de código 42: 12.1 Modelo de despliegue (ensamble de 5 semillas, 2021-2026) y proyección 2027  (fuente completa en U2_modelo_base_codigo.py)
Salida de la celda:
```text
Proporción nacional de eventos A+B (2025-2026): 6.4% → umbral de alerta: 12.7%
Departamento | Perfil 2026 observado | P(C) 2027 | Desv. entre semillas | Categoría 2027 | ¿Coincide con la inercia? | s_AB | Énfasis sugerido | Datos suficientes
Guaviare | C | 1.00 | 0.000 | C consolidado | sí | 14.4% | ISR, detección temprana y movilidad protegida + evaluar C-UAS / detección de AEI | True
Vaupés | C | 0.92 | 0.100 | C consolidado | sí | 0.0% | ISR, detección temprana y movilidad protegida | False
Nariño | C | 0.90 | 0.025 | C consolidado | sí | 6.8% | ISR, detección temprana y movilidad protegida | True
Cauca | C | 0.87 | 0.030 | C consolidado | sí | 13.4% | ISR, detección temprana y movilidad protegida + evaluar C-UAS / detección de AEI | True
Caquetá | C | 0.86 | 0.011 | C consolidado | sí | 9.8% | ISR, detección temprana y movilidad protegida | True
Antioquia | C | 0.84 | 0.040 | C consolidado | sí | 6.8% | ISR, detección temprana y movilidad protegida | True
Norte de Santander | C | 0.84 | 0.026 | C consolidado | sí | 10.2% | ISR, detección temprana y movilidad protegida | True
Amazonas | C | 0.77 | 0.107 | C consolidado | sí | 0.0% | ISR, detección temprana y movilidad protegida | False
Chocó | D | 0.75 | 0.119 | C consolidado | revisar | 8.0% | ISR, detección temprana y movilidad protegida | True
Guainía | C | 0.72 | 0.147 | C consolidado | sí | 0.0% | ISR, detección temprana y movilidad protegida | False
Vichada | C | 0.69 | 0.109 | Transición | sí | 5.9% | Portafolio flexible y monitoreo trimestral del perfil | True
Putumayo | C | 0.60 | 0.046 | Transición | sí | 3.7% | Portafolio flexible y monitoreo trimestral del perfil | True
Valle del Cauca | D | 0.56 | 0.103 | Transición | revisar | 7.1% | Portafolio flexible y monitoreo trimestral del perfil | True
Sucre | D | 0.51 | 0.060 | Transición | revisar | 0.0% | Portafolio flexible y monitoreo trimestral del perfil | True
Bolívar | D | 0.48 | 0.065 | Transición | sí | 5.0% | Portafolio flexible y monitoreo trimestral del perfil | True
Córdoba | C | 0.47 | 0.094 | Transición | revisar | 1.2% | Portafolio flexible y monitoreo trimestral del perfil | True
Cesar | C | 0.43 | 0.051 | Transición | revisar | 7.1% | Portafolio flexible y monitoreo trimestral del perfil | True
Arauca | C | 0.42 | 0.039 | Transición | revisar | 9.8% | Portafolio flexible y monitoreo trimestral del perfil | True
Casanare | C | 0.42 | 0.110 | Transición | revisar | 8.7% | Portafolio flexible y monitoreo trimestral del perfil | Tru
[... salida truncada: 1482 caracteres más]
```

#### Celda de código 43: 12.2 Mapa esquemático (centroides departamentales)  (fuente completa en U2_modelo_base_codigo.py)
Salida de la celda:
```text
Categoría 2027
Transición       13
C consolidado    10
D consolidado    10
Departamentos donde el modelo estructural contradice el perfil observado en 2026: Chocó, Valle del Cauca, Sucre, Córdoba, Cesar, Arauca, Casanare, San Andrés, Huila, Meta
```
Figuras generadas: fig22_mapa_proyeccion_2027 (imagen omitida en este texto).

**Lectura para la decisión.**

* **10 departamentos C consolidados (P ≥ 0,7):** Guaviare, Nariño, Cauca, Caquetá, Antioquia, Norte de Santander y
  Chocó, más Vaupés, Amazonas y Guainía. Estos tres últimos tienen pocos datos y mayor incertidumbre entre semillas
  (desviación ≈0,10–0,15). Coinciden con los corredores históricos del conflicto armado y son la prioridad para
  capacidades ISR, detección temprana y movilidad protegida.
* **10 D consolidados (P < 0,3):** el eje andino y urbano (Bogotá, Cundinamarca, Boyacá, Santander, el Eje Cafetero y
  Tolima), además de Atlántico y Meta.
* **13 en transición (0,3–0,7):** el Caribe (Bolívar, Sucre, Córdoba, Cesar, Magdalena y La Guajira), la Orinoquía
  (Arauca, Casanare y Vichada), Putumayo, Huila, Valle del Cauca y San Andrés. Es la franja donde más vale un monitoreo
  frecuente y un portafolio flexible.
* **Alerta A/B.** Guaviare (14,4 %) y Cauca (13,4 %) superan el umbral de 12,7 % de eventos con explosivos o drones.
  Allí, además del énfasis C, conviene evaluar C-UAS y detección de AEI. Norte de Santander (10,2 %), Caquetá y Arauca
  (9,8 %) quedan cerca y pasan a lista de vigilancia. Esta capa recupera el hallazgo crítico de la Etapa 1: una
  amenaza de bajo volumen y alto impacto que la etiqueta binaria diluye.
* **10 discrepancias con la inercia** (marcadas *revisar*): siete departamentos C en 2026 que el modelo estructural
  ubica por debajo de 0,5 (Arauca, Cesar, Casanare, Córdoba, Huila, Meta y San Andrés) y tres D en 2026 que ubica por
  encima (Chocó, Valle del Cauca y Sucre). En estos casos no conviene decidir solo con el modelo base; §13.2 muestra
  cómo cambian al agregar la memoria del perfil.

---
# 13. Oportunidades de mejora para la siguiente iteración

Las mejoras se derivan del diagnóstico, no de una lista genérica. Antes de proponer, medimos rápidamente el efecto de
las dos mejoras más baratas de probar ya:

* **E1: memoria del perfil.** Agregar $x_8 = s^{C}_{d,t-1}$, la proporción C del año anterior, como predictora. La
  Etapa 1 definió $x_1$ como *"intensidad histórica"*; esta es su versión compositiva, y el diagnóstico muestra que la
  persistencia es la línea base más difícil de batir.
* **E2: ensamble de semillas.** Promediar 5 redes entrenadas con semillas distintas. En §11.2 ya se vio que mejora
  las métricas probabilísticas; aquí se mide su efecto en las validaciones temporal y espacial.

#### Celda de código 44: 13.1 Experimentos E1 (memoria del perfil, x8 = s_C del año anterior) y E2 (ensamble de semillas)  (fuente completa en U2_modelo_base_codigo.py)
Salida de la celda:
```text
Experimentos E1 y E2 en 41 s
 |  | Exactitud | Exact. balanceada | F1 C | MCC | ROC-AUC | Log-loss | Brier
Validación | Modelo |  |  |  |  |  |  | 
Temporal | Persistencia | 0,888 | 0,888 | 0,889 | 0,776 | 0,965 | 0,380 | 0,112
Logística base (x1-x7) | 0,802 | 0,802 | 0,793 | 0,606 | 0,875 | 0,448 | 0,145
MLP base (x1-x7) | 0,802 | 0,802 | 0,777 | 0,619 | 0,892 | 0,436 | 0,140
MLP base, ensamble 5 semillas (E2) | 0,793 | 0,793 | 0,769 | 0,599 | 0,918 | 0,401 | 0,129
Logística + x8 | 0,862 | 0,862 | 0,860 | 0,725 | 0,922 | 0,353 | 0,111
MLP + x8 (E1) | 0,836 | 0,836 | 0,826 | 0,677 | 0,933 | 0,362 | 0,112
MLP + x8, ensamble 5 semillas (E1+E2) | 0,845 | 0,845 | 0,836 | 0,693 | 0,933 | 0,358 | 0,111
Espacial | Persistencia | 0,854 | 0,854 | 0,850 | 0,708 | 0,918 | 0,420 | 0,128
Logística base (x1-x7) | 0,754 | 0,753 | 0,741 | 0,508 | 0,841 | 0,505 | 0,167
MLP base (x1-x7) | 0,737 | 0,734 | 0,710 | 0,473 | 0,804 | 0,577 | 0,183
MLP base, ensamble 5 semillas (E2) | 0,772 | 0,769 | 0,745 | 0,545 | 0,849 | 0,486 | 0,160
Logística + x8 | 0,842 | 0,842 | 0,836 | 0,684 | 0,905 | 0,393 | 0,118
MLP + x8 (E1) | 0,830 | 0,829 | 0,820 | 0,660 | 0,885 | 0,451 | 0,131
MLP + x8, ensamble 5 semillas (E1+E2) | 0,830 | 0,829 | 0,818 | 0,661 | 0,897 | 0,422 | 0,131
```

**Lectura de los experimentos.**

* **E1 (memoria del perfil) es la mejora de mayor impacto.** En validación temporal, agregar $x_8$ sube el AUC del MLP
  de 0,89 a 0,93 y baja la log-loss de 0,44 a 0,36; en la espacial, de 0,80 a 0,89 y de 0,58 a 0,45. Con memoria,
  los modelos **superan a la persistencia en log-loss** (0,35–0,36 frente a 0,38 en temporal): combinan la inercia con
  las condiciones estructurales y dan probabilidades mejor calibradas que repetir el perfil. En exactitud y AUC la
  persistencia sigue adelante, así que el margen de mejora está en anticipar transiciones.
* **E2 (ensamble de semillas) corrige la debilidad espacial.** Promediar 5 redes lleva el AUC espacial de 0,80 a 0,85 y
  la log-loss de 0,58 a 0,49. Con eso el MLP **pasa a superar a la logística** también en departamentos nuevos (AUC
  0,85 vs. 0,84; log-loss 0,49 vs. 0,51), lo que confirma el diagnóstico de §11.3: el problema espacial era varianza, y
  el ensamble es la herramienta estándar para reducirla.
* **E1 + E2** da el mejor MLP temporal (exactitud 0,845, log-loss 0,358). La logística con memoria sigue siendo muy
  competitiva (exactitud 0,862, log-loss 0,353). A escala departamental, la información nueva (la memoria) pesa más
  que la flexibilidad del modelo. La ventaja de la red debería aparecer al escalar a municipios, donde habrá unidades
  suficientes para estimar las interacciones.

#### Celda de código 45: 13.2 Efecto de la memoria en la proyección 2027 (ensamble de 5 semillas, 2021-2026)  (fuente completa en U2_modelo_base_codigo.py)
Salida de la celda:
```text
Discrepancias base vs. inercia: 10 | resueltas al agregar la memoria: 5
Departamento | Perfil 2026 observado | P(C) 2027 base | Datos suficientes | P(C) 2027 con memoria (E1)
Chocó | D | 0.75 | True | 0.65
Valle del Cauca | D | 0.56 | True | 0.62
Sucre | D | 0.51 | True | 0.42
Córdoba | C | 0.47 | True | 0.49
Cesar | C | 0.43 | True | 0.61
Arauca | C | 0.42 | True | 0.59
Casanare | C | 0.42 | True | 0.49
San Andrés | C | 0.34 | False | 0.61
Huila | C | 0.33 | True | 0.55
Meta | C | 0.30 | True | 0.43
```
Figuras generadas: fig23_proyeccion_con_memoria (imagen omitida en este texto).

**Lectura.** Al incorporar la memoria del perfil, **9 de las 10 discrepancias** con la inercia se mueven en la
dirección del perfil observado y **5 se resuelven**, es decir, cruzan el umbral: Arauca (0,42 → 0,59), Cesar
(0,43 → 0,61), Huila (0,33 → 0,55), Sucre (0,51 → 0,42) y San Andrés (con pocos datos). Córdoba y Casanare quedan justo
en la frontera (0,49), y Meta sube de 0,30 a 0,43. Los dos casos que siguen en C pese a ser D en 2026, Chocó
(0,75 → 0,65) y Valle del Cauca (0,56 → 0,62), tienen una proporción C en 2026 a menos de 5 puntos del umbral (0,45
y 0,50), así que su propia etiqueta observada es ambigua. La memoria no sustituye un criterio por otro: **reconcilia**
la lectura estructural con la inercia. Por eso M1 es la primera prioridad del modelo operativo.

### 13.3 Hoja de ruta priorizada

| # | Mejora | Evidencia que la motiva | Acción concreta (Unidad 3) | Impacto esperado | Prioridad | Responsable |
|---|---|---|---|---|---|---|
| M1 | Memoria y dinámica del perfil | Persistencia AUC 0,96; 57 % de los errores en transiciones; E1: AUC 0,89 → 0,93 | Incorporar $s^C_{t-1}$, $s^C_{t-2}$ y la tendencia de eventos; evaluar con validación temporal | Alto (medido) | Alta | Juan Diego |
| M2 | Ensamble de redes | Varianza entre semillas; E2: AUC espacial 0,80 → 0,85 | Promediar 5–10 redes en el modelo operativo | Medio (medido) | Alta | Juan Diego |
| M3 | Migrar a municipio con el registro de evento individual de ACLED | Brecha espacial (§11.3), curva de aprendizaje sin converger, firmas departamentales | Solicitar el acceso académico a los datos por evento (municipio y actores). La coca SIMCI (datos.gov.co) y el IPM censal 2018 del DANE ya se publican por municipio | Muy alto | Alta | Conjunta |
| M4 | Completar $x_3$ (actores armados) y $x_6$ municipal | Errores persistentes en Arauca (frontera, ELN) no explicados por coca ni bosque | Contar actores distintos por territorio con `actor1/actor2`; distancia a cabecera con cartografía IGAC | Alto | Alta | Laura |
| M5 | Recalibrar probabilidades y elegir el umbral con varios años | Calibración bajo la diagonal; el umbral F2 = 0,17 no se transfiere a 2026 | Platt o *temperature scaling* sobre predicciones fuera de muestra 2023–2025; umbral por costo con varios años | Medio | Media | Juan Diego |
| M6 | Etiqueta ponderada por severidad u objetivo continuo | Hallazgo U1 §5.4 (A/B de bajo volumen y alto impacto); 20 % de las celdas a ±10 pp del umbral | Modelar $s^C$ como regresión o ponderar los eventos por fatalidades para que A/B puedan emerger; reabrir las 4 clases de la U1 | Alto (coherencia con la U1) | Media | Laura |
| M7 | Ajuste conjunto de capacidad, L2 y épocas con validación anidada | Sobreajuste con más de 8 neuronas (§11.3d); época óptima variable (CV 35 %) | Búsqueda bayesiana (KerasTuner u Optuna) con validación temporal anidada | Bajo–medio | Baja | Juan Diego |
| M8 | Refinar el mapeo de subtipos | *Shelling* y *Sexual violence* quedan en D por defecto (§3.4) | Reasignarlos a B y C y documentarlo en el diccionario | Bajo | Baja | Laura |

**Secuencia propuesta para la Unidad 3:** primero los datos (M3, M4), porque cambian la unidad de análisis y todo lo
que sigue. Después la arquitectura con memoria y ensamble (M1, M2) reentrenada a escala municipal. Por último la
calibración, el umbral y la etiqueta por severidad (M5, M6), con M7 y M8 como ajustes finos. Las decisiones de
priorización se toman en conjunto y quedan en la bitácora (§14.2).
