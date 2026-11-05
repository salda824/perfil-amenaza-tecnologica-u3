import json
from lecturas_fijas import EQUIPO

L = {}

L["resumen"] = r"""
## Resumen ejecutivo

Este cuaderno cierra el ciclo del proyecto. Toma el modelo base de la U2, un MLP que no superaba la persistencia del
perfil, y lo optimiza con nueve intervenciones derivadas de su diagnóstico. Cada una se acepta o se rechaza con una
regla de retención fijada antes de ejecutarlas, y el modelo final queda integrado en un pipeline reproducible con pruebas
automáticas y un artefacto único de despliegue. Después de la revisión del informe agregamos dos controles que no
cambiaron el modelo: una persistencia calibrada como rival y una validación espacial que también bloquea el tiempo.

| | Resultado |
|---|---|
| Problema | Anticipar con un año de antelación si cada departamento tendrá perfil de amenaza C (convencional) o D (mixto), para orientar ISR, C-UAS y detección de AEI |
| Datos | Panel departamento–año 2021–2026 (171 filas, 30 departamentos, 82 C y 89 D), igual al de la U2. Dos variables nuevas retenidas: memoria del perfil ($m_1$) y composición del último semestre ($d_1$) |
| Modelo final | Ensamble de 10 MLP 9 → 16 ReLU → 1 (177 parámetros por red), L2 = 10⁻³, Adam, 560 pasos |
| Antes → después (origen móvil 2023–2025, n = 87) | Frente al modelo entregado en la U2 (semilla 42): log-loss 0,407 → 0,304. Frente a la misma arquitectura base con ensamble de 10 semillas: 0,376 → 0,304 (Δ −0,072; IC95 por departamento [−0,114; −0,033]) |
| Frente a la inercia | Persistencia cruda: log-loss 0,363 (el modelo gana por calibración; la persistencia tiene mejor AUC y exactitud). Persistencia calibrada: 0,328 (Δ −0,024; IC95 [−0,096; +0,052]). En departamentos nuevos y años futuros: 0,361 del modelo frente a 0,363 de la persistencia y 0,332 de la persistencia calibrada |
| Prueba 2026 (una sola vez) | Log-loss 0,380 (persistencia 0,429; persistencia calibrada 0,351; base U2 con ensamble 0,470); AUC 0,900; exactitud 0,83; sensibilidad C 0,93 |
| Descartadas con evidencia | Actores armados $x_3$ (UCDP), trayectoria $m_2,m_3$, tendencias $d_2,d_3$, vecindad, objetivo blando, salida multitarea A/B, GRU, CNN 1D, cambio de capacidad y calibración. El embedding de departamento cumplió la regla de retención, pero no se adoptó por la regla 8 (agregada al ver este caso); en 2026 fue mejor (log-loss 0,314) y se declara (§9.3) |
| Integración | Paquete `perfil_amenaza`, 10 pruebas automáticas, línea de comandos de 8 etapas, artefacto `.keras` de 152 KB, PSI con piso de ruido y tarjeta del modelo |
| Decisión 2027 | 11 departamentos C consolidados (9 con datos suficientes), 11 en transición y 11 D. Alerta A/B en Guaviare y Cauca. La proyección contradice el perfil de 2026 en 3 departamentos (en la U2, 10), lo que mide cercanía a la inercia y no acierto |

En una frase: con dos variables que resumen la composición reciente del conflicto, la red mejora con claridad frente al
modelo de la U2, pero frente a una inercia bien calibrada su ventaja no está demostrada a escala departamental. Su
debilidad sigue siendo anticipar los cambios de perfil: acierta 2 de 13 transiciones, frente a 3 del modelo de la U2 y 0
de la persistencia.
"""

L["etiqueta_parcial"] = r"""
**Lectura.** Con las primeras 33 semanas, la etiqueta coincide con la del año completo en el **94,9 %** de las 136
celdas de 2021–2025. Las 7 discrepancias tienen todas un $s^C$ anual a menos de 10 pp del umbral (Córdoba aparece
tres veces). La etiqueta parcial no cambia el perfil de los departamentos con un régimen claro; solo voltea casos que
ya eran ambiguos. Para la prueba de 2026 esperamos entonces **1 o 2 etiquetas** de 29 distintas de las que tendrá el
año cerrado, y todas en la franja ambigua. Por eso, en §9.3 una métrica probabilística (log-loss) es más confiable que
la exactitud.
"""

L["eda"] = r"""
**Lectura.**

* La composición reciente es la información más fuerte. $d_1$ (s_C del segundo semestre de $t-1$) tiene la mayor
  asociación monótona con $y$ (ρ = 0,75) y $m_1$ le sigue (0,72). Superan a cualquier variable estructural de la U1:
  la mejor era la letalidad, con ρ = 0,62.
* $d_1$ adelanta cambios. En el 74 % de las 19 transiciones de perfil de 2021–2025 (dejamos fuera la prueba 2026), el
  segundo semestre del año anterior ya se movía en la dirección del nuevo perfil (panel b: los círculos negros se separan
  de la diagonal). Es la justificación cuantitativa de I2: la vista anual promedia y diluye el giro reciente.
* Las tendencias no traen información propia. $m_3$, $d_2$ y $d_3$ tienen información mutua ≈ 0 con $y$. Son
  diferencias de variables que ya están en el modelo, y con 171 filas es poco probable que aporten. §6 lo comprueba.
* $m_1$, $m_2$ y $d_1$ están muy correlacionadas entre sí (panel c), como se esperaba. Por eso la cascada de §6 agrega
  una a la vez y exige que cada una mejore sobre la anterior.
"""

L["reproduccion"] = r"""
**Lectura.** La reproducción es casi exacta. La logística y la persistencia coinciden al dígito con la U2, y el MLP
difiere en 0,0015 de AUC (0,894 vs. 0,892), lo esperable al pasar de Windows a Linux con el mismo TensorFlow
2.21. Con el protocolo de la U3 (2023–2025, 5 semillas), el punto de partida es:

* **El MLP base (log-loss 0,383 ± 0,021) pierde contra la persistencia (0,363)**, y también en AUC (0,921 vs. 0,973).
  Es el problema que esta unidad debe resolver.
* El experimento E1 de la U2 (memoria cruda $x_8$) ya baja la log-loss a 0,347 y supera a la persistencia. Lo
  confirmamos con más semillas y un protocolo más estricto: la memoria es la palanca correcta.
* El modelo base acierta en promedio 2,4 de las 9 transiciones de 2023–2025. La persistencia acierta 0.
"""

L["i1"] = r"""
**Lectura.**

* **I1 se retiene.** Agregar $m_1$ baja la log-loss media de 0,383 a 0,347 (Δ = −0,036, más de tres errores
  estándar, EE = 0,009) y sube el AUC de 0,921 a 0,939. Con esto la red individual ya supera a la persistencia en
  log-loss (0,347 vs. 0,363).
* **La trayectoria ($m_2$, $m_3$) no aporta** (Δ = +0,002, dentro del ruido): lo que importa es el estado reciente,
  no la pendiente. Se descarta por la regla de la variante más simple.
* **Contracción vs. memoria cruda.** $m_1$ contraída y $x_8$ cruda dan la misma log-loss (0,347). Era lo esperado:
  casi todas las filas de $t-1$ tienen decenas de eventos y la contracción apenas las mueve. Su efecto se mide en §7,
  donde la cuidan las variables semestrales con menos eventos.
* **Hipótesis.** Se confirma que la memoria acerca la red a la persistencia y la supera en log-loss. No se confirma
  que mejore las transiciones: se mantienen en 2,4 de 9.
"""

L["i2"] = r"""
**Lectura.**

* **I2 se retiene.** El semestre reciente $d_1$ baja la log-loss de 0,347 a **0,310** (Δ = −0,037; EE = 0,007) y
  sube el AUC a 0,951. Es la mayor ganancia individual de la unidad, y el EDA anticipaba por qué: $d_1$ registra los
  giros de la segunda mitad del año que el promedio anual diluye.
* **Las tendencias $d_2$, $d_3$ empeoran** (Δ = +0,013): con 113–142 filas, dos variables de ruido cuestan más de lo
  que aportan. Se descartan.
* **La ganancia es general, no solo en 2025.** Por año, la red con $m_1 + d_1$ supera a la persistencia en log-loss en
  los tres pliegues (2023: 0,313 vs. 0,344; 2024: 0,224 vs. 0,346; 2025: 0,362 vs. 0,399). El modelo base solo lo
  lograba en 2024.
* **Hipótesis parcialmente refutada.** Esperábamos que I2 mejorara las transiciones, y no lo hizo (2,0 de 9 frente a
  2,4). $d_1$ mejora la confianza en los perfiles estables más que la detección de cambios. Las transiciones siguen
  siendo el límite del modelo (§10).
"""

L["i3"] = r"""
**Lectura.** **I3 se rechaza.** Con UCDP GED (843 filas departamento–año), $x_3$ empeora el modelo: la log-loss sube de
0,310 a **0,335** (Δ = +0,025, unas 4 veces su EE), el AUC baja de 0,951 a 0,937 y las transiciones acertadas pasan
de 2,0 a 1,4. Tres razones lo explican:

* **Redundancia.** El número de actores se asocia con el perfil C (ρ = 0,46), pero casi todo lo que aporta ya está en
  las fatalidades (ρ = 0,77 con $x_3$) y en la memoria. Agrega una dimensión más sin información nueva.
* **Ruido conceptual.** UCDP cuenta toda organización armada con muertes, incluidas bandas urbanas. Atlántico, que es
  D todos los años, pasa de 0 a 3 actores desde 2023 (Los Costeños, Los Pepes). Esas bandas no producen un perfil
  convencional en el sentido de la U1.
* **Cambio de cobertura.** UCDP registra 173 eventos en Colombia en 2022 y 422 en 2023. Parte del salto es de
  codificación, no de conflicto, y la variable deriva justo dentro de la ventana de validación.

El caso que motivaba $x_3$ ya estaba resuelto: Arauca, mal clasificada todos los años en la U2 (P ≈ 0,36), recibe
P(C) entre 0,74 y 0,84 sin $x_3$ (entre 0,82 y 0,91 con ella). La memoria del perfil capta de forma indirecta la
presencia sostenida del ELN. Los actores pueden importar; lo que no funcionó fue medirlos con un conteo anual
departamental. A escala municipal, el cambio en los actores presentes (entradas y
salidas de grupos) es la versión que vale la pena probar (R2, §13).
"""

L["i4"] = r"""
**Lectura.**

* I4 no se retiene. Su hipótesis era espacial, así que la regla se aplica a la validación espacial: ahí la vecindad
  empeora (log-loss 0,389 frente a 0,379; Δ = +0,009). En el eje temporal da una mejora mínima (Δ = −0,004, unas 2
  veces su EE), que por sí sola cumpliría la regla. Con 30 departamentos, la composición de los vecinos suma colinealidad
  con la memoria propia sin aportar información que se transfiera.
* El hallazgo espacial importante es otro. Con $m_1 + d_1$, la red individual pasa de log-loss 0,475 (modelo base) a
  0,379 en departamentos nunca vistos, y el AUC de 0,854 a 0,914. En la U2 las variables estructurales casi constantes
  funcionaban como identificadores; con información de composición la red ya no necesita memorizar firmas
  departamentales. §9.2 repite la comparación con un esquema que además respeta el tiempo.
* La logística con las mismas variables queda empatada en este eje (0,373).
"""

L["i5"] = r"""
**Lectura.** **I5 se rechaza.** Sin calibrar, el objetivo blando empeora mucho la log-loss (0,429 vs. 0,310). Era lo
previsto: la red aprende la proporción esperada, que se concentra cerca de 0,5, y no la probabilidad de cruzar el
umbral. Incluso después de calibrar ambas variantes con el mismo método, sigue peor (0,385 vs. 0,317) y pierde AUC
(0,929 vs. 0,944). La ponderación por $\sqrt{n}$ no ayuda. Concluimos que el ruido de etiqueta cerca del umbral no es
lo que limita al modelo: la etiqueta dura de la U1 se mantiene, y con ella la trazabilidad de la variable objetivo.
"""

L["i9"] = r"""
**Lectura.** **I9 se rechaza, y el resultado negativo es informativo.**

* Con λ = 0,5, la tarea auxiliar no perjudica a $\hat p$ (Δ = +0,001), pero tampoco ayuda. Con λ = 1 empeora
  (Δ = +0,010).
* La composición A/B predicha **no supera a la persistencia de la composición**: MAE 0,035 vs. 0,030, ρ = 0,47 vs.
  0,60, y AUC para detectar la alerta A/B de 0,68 vs. 0,70 (14 de las 87 filas tienen alerta).
* La proporción de explosivos y drones de un departamento es tan persistente como su perfil, y la red no encuentra en
  las variables de la U1 nada que anticipe mejor sus cambios. La alerta A/B operativa sigue siendo la regla
  descriptiva de la U2 (proporción observada en 2025–2026 de al menos el doble de la nacional). Modelarla exigiría
  datos por evento y municipio, donde A y B tienen variación suficiente.
"""

L["comparativos"] = r"""
**Lectura.**

* GRU (665 parámetros) y CNN 1D (393) empeoran (log-loss 0,319 y 0,324 frente a 0,310). Los cuatro trimestres de
  $t-1$ no contienen más información que su resumen semestral $d_1$, y la red secuencial paga en varianza los
  parámetros extra. Con 4 pasos de tiempo y 142 filas, una arquitectura recurrente no tiene de dónde aprender.
* El MLP profundo (16, 8) empata (Δ = −0,0004; EE = 0,014): la profundidad no aporta, igual que en la U2.
* La logística con las mismas variables empata (0,306 frente a 0,310; Δ = −0,003, EE 0,0043, que no cumple la regla).
  A escala departamental la información nueva pesa más que la flexibilidad del modelo, como en la U2 (§11.4). Mantenemos
  el MLP por lo que muestra §9.2: en departamentos nuevos el ensamble de redes obtiene menos log-loss que la logística,
  aunque con un intervalo que incluye el cero.
* El embedding de departamento es el único que cumple la regla 5 (log-loss 0,284; Δ = −0,026; EE = 0,010). No se adopta
  por la regla 8, que agregamos al encontrar este caso, y lo declaramos así:
  1. Acierta 0,8 de 9 transiciones frente a 2,0 de la referencia. Un vector fijo por territorio acerca el modelo a la
     persistencia, que ya está disponible sin modelo.
  2. En departamentos nuevos se reduce al MLP sin embedding: su log-loss espacial (0,369) es similar a la del MLP sin
     embedding (0,379) porque los departamentos nunca vistos reciben el vector inicial, cercano a cero. Su ganancia
     temporal viene de efectos fijos memorizados.
  3. Es incompatible con la migración a municipio (M3), donde la mayoría de unidades no tendría historia suficiente
     para estimar su vector.

  Para transparencia, en §9.3 reportamos también su desempeño en 2026.
"""

L["i6"] = r"""
**Lectura.**

* La superficie es plana. De 48 configuraciones, 31 quedan dentro de un error estándar de la mejor (32 neuronas,
  L2 = 10⁻², 1.120 pasos; log-loss 0,289 ± 0,028). La vigente de la U2 (16 ReLU, L2 = 10⁻³, 560 pasos) queda en el
  puesto 18 (0,303), dentro de esa banda. Los mapas (a) y (b) muestran el único patrón robusto: las redes pequeñas (4–8
  neuronas) necesitan más pasos, y con 280 pasos no alcanzan a converger. Con 560 o más, casi todo es equivalente.
  ReLU y tanh también empatan.
* Validación anidada de un solo pliegue externo. La configuración con mejor log-loss en 2023–2024 (0,234) obtiene 0,423
  en 2025, y la que elige la regla de 1 EE en 2023–2024 obtiene 0,411. La vigente obtiene 0,359, pero la comparación la
  favorece: su capacidad se eligió en la U2 con validación de origen móvil que incluía 2025, y su presupuesto de 560 pasos
  es la mediana de la parada temprana sobre 2025 (D10). Por eso no la leemos como prueba de que la vigente sea mejor,
  sino como señal de que, con 58 filas de validación, la rejilla no ofrece una alternativa que se sostenga al año
  siguiente (Cawley y Talbot, 2010).
* Tasa de aprendizaje. Con el presupuesto fijo, η = 3×10⁻⁴ deja la red sin entrenar (log-loss 0,452) y η = 3×10⁻³
  no mejora (0,319). η = 10⁻³ es el valor adecuado para 560 pasos.
* Decisión: mantenemos la configuración de la U2 por parsimonia. La capacidad no es el factor limitante y la rejilla no
  muestra una alternativa estable.
"""

L["i7"] = r"""
**Lectura.** **I7 se retiene por estabilidad, no por desempeño medio.** Una red individual tiene log-loss esperada de
0,308, pero según la semilla puede caer en cualquier punto entre 0,283 y 0,337 (banda del 5–95 %). Con 10 redes la
media apenas cambia (0,297), pero la banda se estrecha a **0,292–0,302**, unas 6 veces menos. Más allá de 10 la ganancia
es marginal. La dispersión de $P(C)$ entre redes por fila tiene mediana de 0,047 y máximo de 0,128, y el 85 % de las
filas recibe la misma clasificación de las 20 redes. El ensamble convierte un resultado que dependía de la semilla
(el problema de la semilla 42 en la U2, D10) en uno reproducible, y su desviación entre redes sirve de medida de
incertidumbre por departamento en §12.
"""

L["i8"] = r"""
**Lectura.** I8 se rechaza para la probabilidad y el umbral se mantiene en 0,5.

* La calibración solo mejora si se mira el futuro. Con encadenamiento hacia adelante (la única evaluación que respeta el
  tiempo), Platt baja el ECE de 0,142 a 0,097, pero sube la log-loss de 0,298 a 0,302 y el Brier de 0,093 a 0,099 en
  2024–2025. La temperatura también empeora (0,306).
* La corrección no es estable. Ajustado año por año, el parámetro de escala de Platt vale $a$ = 1,31 en 2023 y 3,38 en
  2024. En 2025 no converge ($a \approx 93$, $b \approx 152$) porque la red separa perfectamente las clases ese año
  (AUC = 1,0), aunque con umbral 0,5 clasifica mal seis escaladas: sin solapamiento entre clases, la mejor calibración es
  un escalón. El desajuste depende de la prevalencia de cada año (41 % a 62 % de C) y una corrección estimada con el
  pasado no sirve para el año siguiente.
* El umbral óptimo tampoco se transfiere. El que maximiza el MCC es 0,20 en 2023, 0,51 en 2024 y 0,15 en 2025.
  Usar 0,20 sobre el agregado subiría el MCC de 0,69 a 0,81, pero sería elegirlo con los mismos datos que lo evalúan:
  es el error que la U2 documentó con el umbral F2 (0,17), que no transfirió a 2026.
* Para la decisión no usamos un umbral único sino las bandas de §1.3: un departamento con $\hat p$ entre 0,3 y 0,7 es
  "transición" y recibe portafolio flexible. La incertidumbre de calibración queda así en la decisión.
"""

L["sintesis"] = r"""
**Lectura de la síntesis.** De las intervenciones evaluadas se retienen tres: $m_1$ (I1), $d_1$ (I2) y el ensamble (I7).
Juntas bajan la log-loss de 0,383 a 0,304 y suben el AUC de 0,921 a 0,956. El resto se descarta con una razón
cuantitativa registrada en la tabla: no cumplieron la regla de retención, salvo la vecindad, que mejoró solo en el eje
que no era el de su hipótesis, y el embedding, que la cumplió y se descartó por la regla 8. La escalera muestra que la
mejora vino de la representación de los datos: las dos variables retenidas resumen el estado reciente del conflicto, que
es lo que la U2 identificó como información faltante, mientras que capacidad, profundidad, redes secuenciales, salida
multitarea y calibración no cambiaron el resultado. El modelo final es tan simple como el de la U2 (una capa de 16
neuronas), con dos entradas más y unas seis veces menos dispersión entre semillas.
"""

L["controles"] = r"""
**Lectura de los controles.**

* La estandarización es indispensable. Sin ella la red no converge: la BCE de entrenamiento queda en 7,1 desde la
  época 10. Las variables en km (distancia hasta 800) y en % (bosque hasta 98) saturan las ReLU y la sigmoide de salida,
  y el gradiente se anula. La log-loss fuera de muestra sube de 0,310 a 6,09 (probabilidades saturadas en 0 o 1) y el
  AUC cae a 0,53. Es el impacto más directo del preprocesamiento sobre el entrenamiento (panel a).
* Los pesos de clase se justifican (sin ellos, Δ = +0,015, unas 8 veces el EE). En entrenamiento la prevalencia de C es
  44 % y en los años validados llega a 62 %; sin la ponderación la red se inclina hacia D.
* La contracción aporta poco, en la dirección correcta (sin ella, Δ = +0,002, unas 2 veces el EE): estabiliza las
  proporciones semestrales con pocos eventos.
* La fuga deliberada del escalador da una mejora artificial de 0,002. Mide cuánto inflaría la métrica ajustar la media
  y la desviación con el año de prueba. Es pequeña porque la estandarización es lineal y las distribuciones cambian poco
  entre años; las pruebas automáticas de §11 detectarían una fuga así.
* El mapeo alternativo de subtipos empeora un poco (+0,007): la regla de la U1 se sostiene.
* Con etiquetas barajadas (control negativo) el modelo no aprende (log-loss 0,815; AUC 0,32). El valor exacto depende de
  la permutación, pero queda lejos de la referencia: el desempeño medido proviene de señal real.
* `MIN_EVENTOS`. Evaluando en las mismas filas (≥ 20 eventos), entrenar con 5 (log-loss 0,300) o con 10 (0,307) es
  equivalente, y con 20 empeora (0,338) porque se pierden filas. El umbral de 10 de la U2 queda en la zona correcta.
* Un resultado que no anticipábamos: sin logaritmos, la log-loss mejora levemente (0,298 frente a 0,310; Δ = −0,012,
  unas 4 veces el EE). Con la memoria en el modelo, las variables de volumen pasan a segundo plano y la red tolera bien
  sus colas. Por la regla 7, no cambiamos el modelo en esta iteración. En el pliegue 2025, el más exigente, la curva (b)
  muestra lo contrario (BCE de validación 0,373 sin logaritmos frente a 0,340), y el logaritmo protege la proyección
  frente a valores extremos nuevos. Queda como tarea para la escala municipal.
"""

L["curvas"] = r"""
**Lectura de las curvas.**

* El presupuesto de 560 pasos (70 épocas) sigue cerca del óptimo de generalización. En el modelo final, la BCE de
  validación 2025 en el paso 560 es 0,377 (media de 5 semillas), frente a un mínimo de 0,360. Entrenar 3 veces más la
  sube a 0,441, que es sobreajuste. El presupuesto de la U2 (D10) sigue siendo válido con las variables nuevas. Este
  0,377 y el 0,359 de la rejilla de §6.8 son la misma configuración con distintas semillas (5 aquí, 3 allá).
* El modelo final aprende más y generaliza mejor a la vez: baja la BCE de entrenamiento (0,341 → 0,279) y la de
  validación (0,448 → 0,377), y la brecha entre ambas se reduce de 0,107 a 0,097. La capacidad es la misma, así que la
  mejora viene de la información de entrada, coherente con §6.11.
* Las bandas entre semillas son estrechas en ambas curvas: el entrenamiento es estable (convergencia verificada en la
  U2, §11.1).
"""

L["antes_despues"] = r"""
**Lectura de la comparación (2023–2025).**

* Contra el modelo de la U2 la mejora es clara. Frente al modelo entregado (semilla 42), la log-loss baja de 0,407 a
  0,304 (Δ = −0,103, IC95 por departamento [−0,185; −0,033]). Esa semilla estaba en la cola desfavorable, así que la
  comparación justa es con la misma arquitectura base promediada sobre las mismas 10 semillas: 0,376 → 0,304
  (Δ = −0,072; IC95 [−0,114; −0,033]). La mejora atribuible a $m_1$ y $d_1$ se mantiene con los dos criterios.
* Contra la persistencia cruda la ventaja es de calibración. Δ log-loss = −0,059 con IC95 [−0,142; +0,028] (el modelo
  gana en el 92 % de los remuestreos). La persistencia ordena mejor (AUC 0,973 frente a 0,956) y con umbral 0,5 clasifica
  mejor (exactitud 0,897 frente a 0,839; acierta 7 casos que el modelo falla y el modelo 2 que ella falla, McNemar
  p = 0,18), pero sus probabilidades son peores (ECE 0,173 frente a 0,124) porque usa $s^C$ como si fuera una
  probabilidad.
* Contra la persistencia calibrada (rival agregado tras la revisión) la ventaja no está demostrada: 0,328 frente
  a 0,304 (Δ = −0,024; IC95 [−0,096; +0,052]; el modelo gana en el 74 % de los remuestreos). Una logística de
  dos parámetros sobre la memoria del perfil recoge casi toda la información que usa el modelo.
* El MLP y la logística empatan con las mismas variables (Δ = −0,002). La mejora de la U3 se debe a la representación de
  los datos, y con 171 filas la red todavía no rinde más por ser no lineal.
* Las cifras de selección salen de una cascada de unas 20 comparaciones sobre las mismas 87 predicciones y por eso son
  algo optimistas (Cawley y Talbot, 2010); 2026 (§9.3) es la estimación que no pasó por la selección.
"""

L["espacial"] = r"""
**Lectura.** En departamentos nunca vistos el cambio frente a la U2 es grande. Con 5 grupos de departamentos, la red
individual pasa de log-loss 0,476 (base) a 0,373 (final), y el ensamble de 0,445 a 0,351, por debajo de la
persistencia (0,418; Δ = −0,067, IC95 [−0,135; +0,003]). Ese esquema, sin embargo, deja que la red aprenda de otros
departamentos en el mismo año que predice, ventaja que la persistencia no tiene. Con bloqueo espacial y temporal, que es
el escenario real de un territorio nuevo al año siguiente, el modelo obtiene 0,361, la persistencia 0,363
(Δ = −0,002; IC95 [−0,114; +0,130]), la persistencia calibrada 0,332 y la logística 0,416. Las variables de
composición transfieren mucho mejor que las estructurales de la U2, y el ensamble de redes obtiene menos log-loss que la
logística (Δ = −0,055; IC95 [−0,236; +0,067]), que es el argumento medible, aunque débil, para mantener el MLP. La ventaja
frente a la inercia en territorios nuevos no está demostrada: la persistencia calibrada es la mejor de las cinco.
"""

L["prueba_2026"] = r"""
**Lectura de la prueba 2026 (única evaluación).**

* El modelo final mejora frente al de la U2 en el año de prueba. Log-loss 0,380 frente a 0,524 del modelo entregado
  (0,547 en la fase 2 documentada en la U2) y 0,470 de la base con ensamble (Δ = −0,090; IC95 [−0,162; −0,024]).
  ECE 0,074, el mejor calibrado de los modelos de la tabla.
* Frente a la inercia no hay ventaja demostrada. Frente a la persistencia cruda, Δ = −0,049 con IC95 [−0,162; +0,092].
  La persistencia calibrada obtiene 0,351, mejor que el modelo (Δ = +0,029; IC95 [−0,055; +0,125]).
* La degradación desde la selección es moderada: log-loss 0,304 → 0,380 y AUC 0,956 → 0,900. 2026 es un año de
  desescalamiento, y el AUC de la persistencia también bajó (0,973 → 0,948).
* Lo que falló: las 4 transiciones de 2026, todas C → D (Bolívar, Chocó, Magdalena y Valle del Cauca). El modelo les dio
  entre 0,55 y 0,88 de probabilidad de C y no acertó ninguna; la persistencia tampoco, por construcción, y el modelo de la
  U2 acertó Magdalena (0,40). Tres de ellas tienen $s^C$ de 2026 entre 0,45 y 0,50, en la franja que §2.3 identificó como
  ambigua con 33 semanas, así que parte de ese error puede corregirse cuando cierre el año. La sensibilidad para C es alta
  (0,93: solo un C perdido, Cesar) y la especificidad de 0,73 resume esas desescaladas.
* El embedding descartado salió mejor en 2026, y lo reportamos tal cual. Obtuvo log-loss 0,314, AUC 0,933 y MCC 0,81, y
  acertó 1 de las 4 transiciones. Esto debilita en parte el argumento de la regla 8: en 2023–2025 acertaba menos
  transiciones, en 2026 una más. No cambiamos el modelo, porque elegir ahora con 2026 convertiría la prueba en un conjunto
  de selección y su cifra dejaría de ser una estimación válida. Mientras el uso se limite a los mismos 33 departamentos,
  los efectos fijos territoriales tienen valor predictivo que las variables actuales no capturan. Queda como
  recomendación (R7, §13): reevaluar el embedding cuando cierre 2026, con prueba en 2027, y a escala municipal
  reemplazarlo por las variables que expliquen esos efectos ($x_3$ municipal, distancia a cabecera).
"""

L["errores"] = r"""
**Lectura del análisis de errores.**

* Los errores se concentran donde el diagnóstico de la U2 lo predijo. De los 19 errores en 2023–2026, el 58 % son
  transiciones de perfil (frente al 2 % entre los aciertos) y el 58 % tiene $s^C$ a ±10 pp del umbral (frente al 19 %).
  El modelo falla donde la etiqueta es ambigua o donde el territorio cambia de régimen.
* Frente al modelo de la U2, 6 casos pasan de error a acierto y 2 de acierto a error; 17 fallan en ambos. Los fallos
  compartidos son casi todos transiciones.
* Arauca, el error estructural de la U2, quedó corregido: P(C) = 0,84, 0,77, 0,85 y 0,79 en 2023–2026, todos C. La
  persistencia del frente ELN entra por $m_1$ y $d_1$ (§6.3).
* Persisten cuatro patrones:
  1. Escaladas D → C de 2025 (Córdoba, Magdalena, Cesar, Meta, Bolívar y Valle): un cambio de coyuntura nacional, la
     ruptura de los diálogos, que la información de 2024 no contenía.
  2. Desescaladas C → D de 2026 (§9.3).
  3. Huila, que es C en 2023 y 2024 y recibe P(C) de 0,36 y 0,47.
  4. Casanare, con pocos eventos (13 a 31) y P(C) baja todos los años, aunque es C.

  Los dos primeros requieren información que no está en ACLED agregado ni en el conteo anual de actores de UCDP
  (I3 no ayudó): señales políticas como ceses al fuego o negociaciones. Los otros dos son ruido de pocos datos o de
  departamentos cerca del umbral.
* Transiciones. De 13 transiciones en 2023–2026, el modelo final acierta 2, el de la U2 3 (con ensamble,
  4), la persistencia calibrada 1 y la persistencia 0. La U3 mejoró la probabilidad en los territorios
  estables sin ganar capacidad de anticipar cambios.
* Regla de decisión. La decisión usa bandas (P ≥ 0,7, 0,3–0,7 y < 0,3) y no el umbral 0,5. En 2023–2025, 71 de 87
  filas caen en una banda consolidada y 66 de ellas aciertan (31 de 32 en C y 35 de 39 en D); en 2026,
  24 de 29, con 21 aciertos (10 de 13 en C y 11 de 11 en D). Con la persistencia cruda, 42 de las 87
  filas quedan en transición y la banda informa poco.
"""

L["interpretacion"] = r"""
**Lectura.**

* $d_1$ es, con diferencia, la variable más importante fuera de muestra (permutarla sube la log-loss en 0,115), más del
  doble que cualquier otra. Le siguen el bosque, $m_1$ y las fatalidades (≈ 0,05 cada una). La distancia y el volumen de
  eventos quedan casi en cero. Frente a la U2 cambió la jerarquía: las variables estructurales de la U1 siguen aportando
  (bosque, fatalidades, IPM y coca) y ahora modulan la inercia.
* La interacción que la red aprende es coherente con la U1. En (b), la frontera $P = 0{,}5$ entre memoria y letalidad es
  curva: con una letalidad de 0,6 fatalidades por evento basta una memoria de ≈ 0,45 para esperar C, y con letalidad nula
  hace falta ≈ 0,7. La violencia letal del año anterior adelanta el perfil convencional aunque la composición todavía no
  lo muestre, como anticipaba la U1 al decir que $x_2$ "distingue letalidad por arma". La logística (c) solo puede trazar
  una recta, y con pendiente contraria por la colinealidad entre memoria y letalidad (ρ = 0,84).
* Como $m_1$ y la letalidad están tan correlacionadas, la dependencia parcial promedia sobre combinaciones que casi no
  aparecen en los datos; la frontera solo debe leerse donde hay puntos. Y como la red y la logística empatan en
  desempeño (§9.1), esta interacción es plausible pero no está demostrada como necesaria. Con más unidades (municipios)
  será posible contrastarla.
"""

L["pipeline"] = r"""
**Lectura de la integración.**

* Las 10 pruebas pasan: contrato de datos y su caso negativo (un export sin una columna se rechaza), ausencia de fuga
  temporal (multiplicar por 5 los eventos de 2024 no cambia ninguna predictora de las filas de 2024, y sí las de 2025),
  escalador y capa `Normalization` del artefacto ajustados solo con el entrenamiento, persistencia calibrada y partición
  espacio-temporal sin acceso al año predicho, regla de retención, determinismo y reproducción exacta del dataset de la U2.
* La orden `pipeline todo` reproduce de extremo a extremo lo que el cuaderno evaluó, en 59 segundos. Las log-loss
  de selección (0,304) y de prueba (0,380) coinciden con §9 al cuarto decimal.
* El artefacto es único y autocontenido: un archivo `.keras` de 152 KB con la estandarización (capa `Normalization` con
  las medias y desviaciones de entrenamiento), las 10 redes de 177 parámetros, el promedio y la calibración. Recibe las 9
  variables en su escala original y devuelve P(C). La proyección de §12 se calcula con este archivo, no con objetos en
  memoria, y `pipeline predecir` funciona sin el cuaderno.
* Deriva (PSI). Comparamos la información de 2026 de los 30 departamentos del entrenamiento con las filas de
  entrenamiento. Con 30 filas el PSI tiene un piso de ruido alto: sin deriva, su percentil 95 está entre 0,21 y
  0,39 según la variable. Solo la letalidad lo supera (PSI 0,45 frente a un percentil 95 de 0,37): la distribución de fatalidades por evento de 2026 difiere de la del entrenamiento. $m_1$ (0,23) y $d_1$ (0,17) caen en la franja de vigilancia de los umbrales usuales (0,1–0,25), pero dentro del ruido de muestreo. La proyección 2027 debe leerse con esa cautela, sobre todo donde la letalidad pesa en la frontera (§10.2). La tarjeta del modelo fija el mantenimiento: reentrenar con cada año cerrado y
  revisar el PSI con su piso de ruido antes de usar la proyección.
"""

L["decision"] = r"""
**Lectura para la decisión.**

* 11 departamentos C consolidados (P ≥ 0,7): Guaviare, Nariño, Cauca, Caquetá, Putumayo, Antioquia, Norte de
  Santander, Vichada y Cesar, más Amazonas y Vaupés, que tienen menos de 10 eventos en 2026 y requieren revisión experta
  (en gris en el mapa). Prioridad para ISR, detección temprana y movilidad protegida. Guaviare (14,4 % de eventos A+B) y
  Cauca (13,4 %) cumplen la alerta A/B (proporción A+B de 2025–2026 de al menos el doble de la nacional, 6,4 %, y 5 o más
  eventos A+B): allí conviene sumar C-UAS y detección de AEI.
* 11 en transición (0,3–0,7): San Andrés y Guainía (pocos datos), Chocó, Valle del Cauca, Arauca, Córdoba, Casanare,
  Huila, Meta, Sucre y Magdalena. Portafolio flexible y monitoreo trimestral. Meta (0,50), Huila (0,51) y Casanare
  (0,52) están justo en el umbral.
* 11 D consolidados: el eje andino y urbano, más La Guajira, Bolívar y Atlántico. Orden público y capacidades mixtas.
* Frente a la U2, 25 de 33 departamentos conservan su categoría. Putumayo, Vichada y Cesar pasan de transición a C
  consolidado, porque la memoria pesa ahora más que la estructura. Chocó y Guainía bajan de C a transición, y Bolívar y
  La Guajira de transición a D. La proyección contradice el perfil observado en 2026 en 3 departamentos (Chocó, Valle del
  Cauca y Meta; en la U2 eran 10). Es una medida de cercanía a la inercia y no de acierto, y señala los casos que conviene
  revisar con inteligencia de campo. Chocó y Valle son desescaladas recientes con $s^C$ de 2026 en 0,45–0,50.
* Incertidumbre. La desviación entre las 10 redes es menor de 0,04 en los extremos (C y D consolidados) y crece en la
  franja de transición: Chocó 0,16, Córdoba 0,12 y los de pocos registros, San Andrés 0,29 y Guainía 0,24.
* Ventana de $d_1$. La información de 2026 cubre 33 semanas, así que $d_1$ resume las semanas 8 a 33 y no el segundo
  semestre. En 2020–2025, $d_1$ medido en esa ventana tiene correlación 0,88 con el semestral y difiere en 0,084 en
  promedio. La proyección debe repetirse al cerrar 2026, sobre todo para los departamentos en el umbral.
"""

L["conclusiones"] = r"""
1. Coherencia problema–datos–modelo–decisión. El modelo final responde la pregunta de la U1 a escala departamental
   (el export gratuito de ACLED no llega a municipio), con la misma etiqueta, el mismo diseño $t-1 \rightarrow t$ y las
   mismas variables estructurales. Agrega dos variables que llevan la "intensidad histórica" de la U1 a la composición
   reciente. Cada cambio frente a la U2 se justifica en un hallazgo del diagnóstico y se valida con el protocolo de §3.1.
2. Dónde estuvo la mejora. De diez frentes evaluados se retuvieron tres ($m_1$, $d_1$ y el ensamble). Las demás
   intervenciones no cumplieron la regla de retención, salvo la vecindad, que mejoró solo en el eje que no era el de su
   hipótesis, y el embedding, que la cumplió y se descartó por la regla 8. La rejilla de hiperparámetros no ofreció una
   alternativa que se sostuviera al año siguiente.
3. Frente al modelo de la U2 la mejora es clara: con ensamble en los dos lados, la log-loss baja de 0,376 a 0,304 en
   selección y de 0,470 a 0,380 en la prueba 2026.
4. Frente a la inercia, la ventaja es de calibración y no está demostrada a escala departamental. El modelo supera a la
   persistencia cruda en log-loss porque ella usa una proporción como probabilidad, pero ordena y clasifica peor que
   ella. Frente a una persistencia calibrada gana por 0,024 en selección, pierde por 0,029 en 2026 y en
   territorios nuevos queda en 0,361 frente a 0,332.
5. La limitación principal sigue siendo anticipar cambios de régimen. El modelo acierta 2 de 13 transiciones (el de la
   U2, 3). Las escaladas de 2025 y las desescaladas de 2026 responden a coyunturas que las variables de ACLED agregado no
   contienen.
6. Los efectos fijos territoriales importan. El embedding de departamento, descartado por la regla 8, fue mejor en la
   prueba 2026. Las variables actuales no explican todo lo que distingue a un departamento de otro (R2 y R7).
7. La red no demostró ser mejor que una logística en el eje temporal. En departamentos nuevos obtiene menos log-loss
   (0,361 frente a 0,416, con un IC95 de la diferencia que incluye el cero; §9.2), que es la razón, débil, para mantenerla. La logística y la persistencia calibrada quedan como rivales permanentes.
8. Para la pregunta de inversión, la evidencia sostiene para 2027 priorizar ISR, detección temprana y movilidad
   protegida en los 9 departamentos C consolidados con datos suficientes, sumar C-UAS y detección de AEI en Guaviare y
   Cauca, mantener portafolio flexible en los 11 de transición y revisar con inteligencia de campo Chocó, Valle del Cauca
   y Meta. Como una inercia calibrada da casi la misma probabilidad a esta escala, el modelo aporta hoy un procedimiento
   reproducible que integra composición reciente, variables estructurales e incertidumbre entre redes; su ventaja
   predictiva queda por probar a escala municipal (R1).

**Recomendaciones (en orden de impacto esperado):**

| # | Recomendación | Evidencia | Acción concreta |
|---|---|---|---|
| R1 | Migrar a municipio con el registro de ACLED por evento (M3) | Con 30 departamentos el modelo está limitado por varianza (U2) y en territorios nuevos empata con la inercia (§9.2) | Solicitar el acceso académico; reentrenar con el mismo pipeline (cambian `datos.py` y las constantes departamentales de `config.py`) |
| R2 | Redefinir $x_3$ a escala municipal | El conteo anual de actores por departamento empeoró el modelo (§6.3), pero los territorios sin historia necesitan una variable de actores | Con el registro por evento, medir actores por municipio y su cambio reciente (entradas y salidas de grupos) en lugar del conteo anual |
| R3 | Señales adelantadas de transición | 11 de 19 errores son transiciones; escaladas 2025 y desescaladas 2026 | Variables mensuales del último trimestre, ceses al fuego y negociaciones; evaluar con la misma regla |
| R4 | Reentrenar cada año con la etiqueta cerrada y vigilar el PSI con su piso de ruido | Prevalencia de C entre 41 % y 62 % en los años validados; en 2026 solo la letalidad supera el piso de ruido del PSI (§11.3) | Ejecutar `pipeline todo` al cierre de cada año; no usar umbrales fijos; decidir con las bandas |
| R5 | Re-evaluar los logaritmos y `MIN_EVENTOS` a escala municipal | Controles de §7 (Δ −0,012 sin logaritmos; 5 ≈ 10 eventos) | Repetir la batería de controles con el nuevo dataset |
| R6 | Mantener la persistencia calibrada y la logística como rivales permanentes | Empatan con el modelo dentro del ruido (§9) | Reportarlas en cada ejecución del pipeline y en la tarjeta del modelo |
| R7 | Reevaluar el embedding de departamento cuando cierre 2026 | Cumplió la regla 5 en 2023–2025 y fue mejor en la prueba 2026 (log-loss 0,314 frente a 0,380) | Incluirlo como candidato en la próxima selección (origen móvil 2024–2026, con prueba en 2027) y, a escala municipal, sustituirlo por $x_3$ y distancia municipal |
"""

L["equipo"] = EQUIPO

L["rubrica"] = r"""
| Criterio de la rúbrica (Excelente = 1,0) | Qué pide el nivel Excelente | Evidencia en este cuaderno |
|---|---|---|
| **Coherencia metodológica integral** (problema–datos–modelo–evaluación) | Integra problema, datos, modelo y conclusiones con trazabilidad clara y evidencia cuantitativa | Trazabilidad U1 → U2 → U3 de variables y decisiones (§1.1–§1.3); reproducción de la U2 al dígito (§2.4, §4); cada intervención nace de un diagnóstico U2 con hipótesis refutable (§5); la regla 8, agregada durante las intervenciones, se declara (§3.1); la salida vuelve a la pregunta de inversión (§12, §13.8); conclusiones con cifras (§13) |
| **Preparación y representación de datos** | Evalúa y justifica transformaciones, partición y controles experimentales, con impacto en el entrenamiento | Contrato y auditoría (§2.1); control de la etiqueta parcial de 2026, 94,9 % (§2.3); variables nuevas con justificación y contracción bayesiana (§2.4–§2.5); protocolo temporal sin fuga (§3); 9 controles con efecto medido y curvas de entrenamiento: sin estandarización no converge, fuga del escalador = +0,002 artificial, etiquetas barajadas AUC 0,32 (§7) |
| **Calidad técnica del modelo y optimización** | Diseña y optimiza la arquitectura con configuración documentada, curvas de desempeño e intervención estratégica basada en diagnóstico | 10 frentes de intervención con regla de retención previa (§3.1, §6); comparativos GRU, CNN, embedding y profundo (§6.7); rejilla de 48 configuraciones con validación anidada (§6.8); curva del ensamble (§6.9); configuración en `configuracion.json` (§8.1); curvas de aprendizaje antes vs. después (§8.3); escalera de mejora (§6.11) |
| **Evaluación del desempeño e interpretación** | Evalúa métricas pertinentes interpretando resultados, errores y comparación experimental con rigor técnico | Log-loss, Brier, ECE, AUC, MCC, exactitud y transiciones (§3.1); persistencia cruda y calibrada como rivales y ensamble del modelo base como antes justo (§9.1); IC bootstrap por departamento y McNemar (§9.1); validación espacial y espacio-temporal (§9.2); prueba 2026 única (§9.3); matrices de confusión y ROC (§9.4); análisis de errores y evaluación de las bandas de decisión (§10.1); importancia e interacción (§10.2); límites declarados (§9, §10.1, §13) |
| **Entregables finales y comunicación ejecutiva** | Integra informe y sustentación con estructura profesional, coherencia técnica y articulación del equipo | Resumen ejecutivo; pipeline con 10 pruebas, línea de comandos y artefacto único (§11); tarjeta del modelo; decisión territorial 2027 (§12); roles, bitácora D11–D20 y declaración de IA (§14); figuras en tamaño de impresión para el informe (Anexo A) |

**Contenidos del Producto 1 (informe de 12 páginas) → secciones de origen:** planteamiento (§1), dataset y
preprocesamiento (§2, §7), arquitectura (§6.7, §8), entrenamiento y evaluación (§3, §8.3, §9), optimización y
comparación (§5, §6, §9), análisis crítico (§9, §10), integración en el pipeline (§11), conclusiones y recomendaciones
(§13). **Producto 2 (video de 8–10 min):** síntesis del problema (§1), decisiones técnicas (§6.11, D11–D20), resultados
(§9, figuras R1–R2), conclusiones estratégicas (§12, §13).
"""

json.dump(L, open("lecturas.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(len(L), "lecturas")
