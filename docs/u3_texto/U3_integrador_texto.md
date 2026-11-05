> Texto extraído de **Proyecto_U3_Integrador.ipynb** (markdown y salidas de texto, sin imágenes). Código de cada celda en `U3_integrador_codigo.py`; la lógica completa vive en `src/perfil_amenaza/`.

# Perfil de amenaza tecnológica por departamento: del modelo base al modelo integrado
## Proyecto Integrador · Diseño, implementación y evaluación de un modelo neuronal aplicado (Unidad 3)

**Redes Neuronales · Universidad de La Sabana · Sector elegido: Seguridad y defensa**

| | |
|---|---|
| **Integrantes** | Laura Alejandra Díaz Sastoque (0000288870) · Juan Diego Saldarriaga Muñoz (0000287391) |
| **Actividad** | Actividad 5: Proyecto Integrador (evaluativa, grupal) |
| **Entregas anteriores** | Etapa 1: formulación, variables y dataset (U1) · Etapa 2: modelo base entrenado (U2) |
| **Fecha** | Noviembre de 2026 |
| **Herramientas** | Python · pandas · scikit-learn · TensorFlow / Keras 3 · paquete propio `perfil_amenaza` (este repositorio) |

## Resumen ejecutivo

Este cuaderno cierra el ciclo del proyecto. Toma el modelo base de la U2 (un MLP que **no** superaba la persistencia
del perfil) y lo optimiza con nueve intervenciones derivadas de su diagnóstico. Cada una se acepta o se rechaza con
una regla fijada de antemano, y el modelo final queda integrado en un pipeline reproducible con pruebas automáticas
y un artefacto único de despliegue.

| | Resultado |
|---|---|
| **Problema** | Anticipar con un año de antelación si cada departamento tendrá perfil de amenaza C (convencional) o D (mixto), para orientar ISR, C-UAS y detección de AEI |
| **Datos** | Panel departamento–año 2021–2026 (171 filas, 30 departamentos), reproducido al dígito de la U2. Dos variables nuevas retenidas: memoria del perfil ($m_1$) y composición del último semestre ($d_1$) |
| **Modelo final** | Ensamble de 10 MLP 9 → 16 ReLU → 1 (177 parámetros por red), L2 = 10⁻³, Adam, 560 pasos |
| **Antes → después** (origen móvil 2023–2025, n = 87) | Log-loss **0,407 → 0,304** (ΔLL −0,103; IC95 por departamento [−0,185; −0,033]); AUC 0,913 → 0,956; Brier 0,132 → 0,095 |
| **Frente al rival** (persistencia) | Mejor en log-loss (0,304 vs. 0,363) y Brier (0,095 vs. 0,106), con P(mejora) = 0,92; peor en AUC (0,956 vs. 0,973). En departamentos nunca vistos: log-loss 0,351 vs. 0,418 |
| **Prueba 2026** (una sola vez) | Log-loss **0,380** (persistencia 0,429; modelo U2 0,524); AUC 0,900; exactitud 0,83; sensibilidad C 0,93 |
| **Descartadas con evidencia** | Actores armados $x_3$ (UCDP), trayectoria $m_2,m_3$, tendencias $d_2,d_3$, vecindad, objetivo blando, salida multitarea A/B, GRU, CNN 1D, cambio de capacidad (la validación anidada mostró que la selección no transfiere) y calibración. El embedding de departamento cumplió la regla numérica, pero no se adoptó porque no transfiere a territorios nuevos; en 2026 fue mejor (log-loss 0,314), y se declara (§9.3) |
| **Integración** | Paquete `perfil_amenaza`, 5 pruebas automáticas, línea de comandos de 8 etapas, artefacto `.keras` de 152 KB, control de deriva (PSI) y tarjeta del modelo |
| **Decisión 2027** | 11 departamentos C consolidados, 11 en transición y 11 D. Alerta A/B en Guaviare y Cauca. Las discrepancias con el perfil observado bajan de 10 (U2) a 3 |

**En una frase:** con dos variables que resumen la composición reciente del conflicto, la red pasa de perder contra
la inercia a superarla en calidad probabilística, dentro y fuera de muestra. Su debilidad sigue siendo anticipar los
cambios de perfil: acierta 2 de 13 transiciones, frente a 0 de la persistencia.

## Cómo leer este notebook

La guía de la Unidad 3 pide un informe con ocho contenidos y la rúbrica evalúa cinco criterios. Cada fila indica dónde
está la evidencia. La matriz completa, con las cifras, está en §15.

| Contenido exigido (Producto 1) / criterio de la rúbrica | Sección |
|---|---|
| Planteamiento del problema y contexto · **Coherencia metodológica integral** | §1 (trazabilidad U1 → U2 → U3), §13 |
| Descripción del dataset y preprocesamiento · **Preparación y representación de datos** | §2 (datos y variables nuevas), §7 (controles experimentales) |
| Arquitectura del modelo y justificación · **Calidad técnica del modelo y optimización** | §6.6–§6.8, §8 |
| Proceso de entrenamiento y evaluación | §3 (protocolo), §8 (curvas), §9 |
| Optimización realizada y comparación de resultados | §5 (diagnóstico), §6 (intervenciones I1–I9), §9 (antes vs. después) |
| Análisis crítico del desempeño · **Evaluación del desempeño e interpretación** | §9, §10 |
| Integración final del modelo en un pipeline estructurado | §11 (paquete, línea de comandos, pruebas, artefacto único, deriva) |
| Conclusiones técnicas y recomendaciones futuras | §12 (decisión 2027), §13 |
| **Entregables finales y comunicación ejecutiva** (roles, bitácora, IA) | §14, §15, §16 |

> **Reproducibilidad.** El cuaderno se ejecuta de principio a fin desde la carpeta `Proyecto_Unidad_3/` (en local o en
> Colab, subiendo la carpeta). Todo el cálculo está en el paquete `src/perfil_amenaza/`, el mismo que usa la línea de
> comandos de §11. El cuaderno no duplica la lógica: la importa. Las cifras del texto corresponden a la ejecución
> guardada en este archivo (CPU, semillas fijas, determinismo de TensorFlow activado). En otro hardware pueden
> variar en el tercer decimal.

---
# 0. Entorno, semillas y paquete del proyecto

#### Celda de código 1: 0.1 Librerías, determinismo y paquete del proyecto
Salida de la celda:
```text
                                                              Valor
Python                                                      3.13.18
Sistema                 Linux-6.18.44-fc-v236-x86_64-with-glibc2.39
NumPy / pandas / SciPy                       2.5.3 / 3.0.5 / 1.18.1
scikit-learn                                                  1.9.1
TensorFlow / Keras                                  2.21.0 / 3.15.1
Dispositivos                                                    CPU
Semilla global                                                   42
Paquete del proyecto                           perfil_amenaza 3.0.0
```

---
# 1. Problema, contexto y trazabilidad U1 → U2 → U3

**Pregunta orientadora (Unidad 1):**

> *¿Cómo orientar la adquisición, transferencia de tecnología y fabricación de sistemas de seguridad y defensa de las
> Fuerzas de Colombia en función del perfil de amenaza tecnológica predominante de los municipios colombianos, según
> las variables de actividad armada histórica, geografía operacional y economías ilícitas, de modo que se optimice la
> inversión de recursos?*

**Formulación operativa (igual a la U2).** Para cada departamento $d$ y año $t$, la etiqueta es

$$
s^{C}_{d,t} = \frac{\#\{\text{eventos de subtipos C en } d,t\}}{\#\{\text{eventos en } d,t\}},\qquad
y_{d,t} = \mathbb{1}\left[s^{C}_{d,t} \ge 0.5\right],\qquad
\hat p_{d,t} = f_{\boldsymbol\theta}\left(\mathbf{x}_{d,t-1}\right) = P\left(y_{d,t} = 1 \mid \mathbf{x}_{d,t-1}\right).
$$

El modelo responde la pregunta que enfrenta quien planea la inversión: *con lo observado hasta el cierre del año,
¿qué perfil tendrá cada territorio el año siguiente?* El rezago $t-1 \rightarrow t$ evita la circularidad entre
predictoras y etiqueta (§1.1 de la U2).

## 1.1 Qué dejó cada etapa y qué cambia en esta

| Etapa | Resultado | Hallazgo que condiciona la siguiente |
|---|---|---|
| **U1** Formulación | Perfiles A/B/C/D con la regla del 50 %; 7 variables; ACLED agregado a departamento | Ningún departamento es A o B dominante → clasificación **C vs. D**; 33 filas no bastan → panel departamento–año |
| **U2** Modelo base | MLP 7 → 16 → 1 (145 parámetros); AUC temporal 0,89; equivalente a la logística | **La inercia domina**: la persistencia (AUC 0,965; log-loss 0,380) supera a la red. La memoria del perfil (E1) es la palanca más fuerte |
| **U3** Integración | Intervenciones I1–I9 derivadas del diagnóstico, controles de datos, pipeline reproducible | El rival a vencer es la **persistencia**, no el MLP base. El valor propio de un modelo está en **anticipar transiciones** |

## 1.2 Trazabilidad de variables (actualizada)

| ID | Variable de la U1 | Operacionalización | Fuente | U2 | U3 |
|---|---|---|---|---|---|
| $y$ | Perfil predominante | $\mathbb{1}[s^C_{d,t} \ge 0{,}5]$, mapeo de subtipos de la U1 | ACLED | ✅ | ✅ sin cambios + control de la etiqueta parcial de 2026 (§2.3) |
| $x_1$, $x_2$, $x_{2b}$ | Eventos, fatalidades, letalidad | $\log(1+\cdot)$ anualizado en $t-1$; fatalidades/evento | ACLED | ✅ | ✅ |
| $x_3$ | Grupos armados activos | $\log(1+$actores no estatales distintos en $t-1)$ | UCDP GED 26.1 (sustituye a ACLED por evento) | ⏳ | ✅ implementada y evaluada (**I3**, §6.3) |
| $x_4$, $x_5$, $x_6$, $x_7$ | Coca, bosque, distancia, IPM | Igual que la U2 | SIMCI-ODC, GFW, centroides, DANE | ✅ | ✅ |
| $m_1$–$m_3$ | *(derivada de $x_1$: "intensidad histórica")* | Composición C contraída en $t-1$, $t-2$ y su cambio | ACLED | E1 (cruda) | **I1** (§6.1) |
| $d_1$–$d_3$ | *(nueva)* dinámica intra-anual | $s^C$ del 2.º semestre de $t-1$, tendencias semestrales | ACLED semanal | – | **I2** (§6.2) |
| $v_1$, $v_2$ | *(nueva)* contexto espacial | Composición y volumen de los departamentos contiguos en $t-1$ | ACLED + contigüidad IGAC | – | **I4** (§6.4) |

## 1.3 De la predicción a la decisión (sin cambios frente a la U2)

| Perfil proyectado | Lectura operacional | Énfasis tecnológico |
|---|---|---|
| **C consolidado** ($\hat p \ge 0{,}7$) | Domina la violencia directa entre actores armados | ISR, detección temprana, movilidad protegida |
| **Transición** ($0{,}3 \le \hat p < 0{,}7$) | Perfil inestable | Portafolio flexible y monitoreo trimestral |
| **D consolidado** ($\hat p < 0{,}3$) | Protesta, disturbio y violencia mixta | Orden público y capacidades mixtas |
| **Alerta A/B** | Explosivos y drones: poco volumen, alto impacto | C-UAS y detección de AEI, cualquiera sea el perfil |

---
# 2. Datos y preprocesamiento

## 2.1 Contrato y auditoría del export de ACLED

El export es el mismo de la U2 (semana × departamento × subtipo, hasta el 15 de agosto de 2026). La función
`datos.cargar_acled` valida el **contrato de datos**: si falta una columna, el pipeline se detiene con un mensaje
explícito en lugar de producir un modelo silenciosamente distinto.

#### Celda de código 2: 2.1 Carga, contrato de datos y auditoría
Salida de la celda:
```text
                               Verificación               Resultado
                       Registros / columnas             17,573 / 12
                Registros fuera de Colombia                      32
                         Cobertura temporal 2019-12-28 → 2026-08-15
        Semanas distintas / saltos ≠ 7 días                 347 / 0
                          Valores faltantes                       0
  Duplicados por llave semana–depto–subtipo                       0
Rangos válidos (events ≥ 1, fatalities ≥ 0)                    True
                  Departamentos de Colombia                      33
Horizonte de estudio: 2020-01-04 → 2026-08-15 | 17,533 registros semanales | 33,275 eventos | 33 departamentos
Semanas de 2026 disponibles: 33 de 52
```

## 2.2 Panel departamento–año y variable objetivo

Se reutiliza sin cambios la construcción de la U2: mapeo de subtipos de la U1, regla del 50 % y umbral de
confiabilidad de 10 eventos (D3). Comprobamos que el panel reproduce los hallazgos de la U2 antes de agregar nada.

#### Celda de código 3: 2.2 Panel, etiqueta y persistencia (réplica de la U2)
Salida de la celda:
```text
Perfil t     C   D
Perfil t-1        
C           69  12
D           11  73
Celdas etiquetables 2021-2026: 171 | perfil A o B dominante: 0 | persistencia año a año: 86.1% | cambios de perfil: 23
         anio  2021  2022  2023  2024  2025  2026
Departamentos 28.00 27.00 29.00 29.00 29.00 29.00
Prevalencia C  0.32  0.56  0.48  0.41  0.62  0.48
```

## 2.3 Control nuevo: ¿es confiable la etiqueta parcial de 2026?

La prueba de 2026 solo tiene **33 semanas** (enero–agosto). Si la composición de la segunda mitad del año fuera
distinta, la etiqueta parcial no representaría el perfil del año, y la prueba mediría otra cosa. Lo verificamos
con datos que sí están completos: recalculamos la etiqueta de 2021–2025 usando **solo las primeras 33 semanas** de
cada año y la comparamos con la del año completo.

#### Celda de código 4: 2.3 Coincidencia entre la etiqueta a la semana 33 y la del año completo (2021-2025)
Salida de la celda:
```text
        anio   2021  2022   2023   2024   2025
      Celdas 28.000 25.00 27.000 28.000 28.000
Coincidencia  0.964  0.92  0.963  0.964  0.929
Coincidencia global: 94.9% de 136 celdas | discrepancias: 7, de las cuales 100% tienen s_C del año completo a ±10 pp del umbral
   departamento  anio  s_C  s_C_parcial perfil perfil_parcial  eventos
          Chocó  2021   48           60      D              C      169
        Córdoba  2022   49           52      D              C      114
      Magdalena  2022   49           55      D              C       87
      Antioquia  2023   53           45      C              D      596
        Córdoba  2024   48           56      D              C       92
        Córdoba  2025   55           48      C              D      131
Valle del Cauca  2025   52           49      C              D      394
[figura]
```

**Lectura.** Con las primeras 33 semanas, la etiqueta coincide con la del año completo en el **94,9 %** de las 136
celdas de 2021–2025. Las 7 discrepancias tienen todas un $s^C$ anual a menos de 10 pp del umbral (Córdoba aparece
tres veces). La etiqueta parcial no cambia el perfil de los departamentos con un régimen claro; solo voltea casos que
ya eran ambiguos. Para la prueba de 2026 esperamos entonces **1 o 2 etiquetas** de 29 distintas de las que tendrá el
año cerrado, y todas en la franja ambigua. Por eso, en §9.3 una métrica probabilística (log-loss) es más confiable que
la exactitud.

## 2.4 Dataset de modelado: reproducción de la U2 y variables nuevas

`variables.construir_dataset` arma una fila por departamento–año etiquetable, con todas las predictoras medidas en
$t-1$. Primero comprobamos que las 7 predictoras y la etiqueta de la U2 se reproducen **exactamente**, para que toda
diferencia posterior se deba a las intervenciones y no a la reconstrucción de los datos.

**Variables nuevas.** Cada grupo responde a un hallazgo del diagnóstico de la U2:

| Grupo | Variable | Definición (todas en $t-1$) | Diagnóstico U2 que la motiva |
|---|---|---|---|
| I1 memoria | $m_1$ | $\tilde s^C_{d,t-1} = \dfrac{C_{d,t-1} + \alpha\,p_0}{n_{d,t-1} + \alpha}$, con $\alpha = 4$, $p_0 = 0{,}5$ | La persistencia acierta el 86 %; 57 % de los errores son transiciones |
| | $m_2$, $m_3$ | $\tilde s^C_{d,t-2}$ y $m_1 - m_2$ | ¿Importa la trayectoria, no solo el último año? |
| I2 dinámica | $d_1$ | $\tilde s^C$ de las últimas 26 semanas de $t-1$ | La vista anual promedia cambios de régimen recientes (p. ej., 2025) |
| | $d_2$, $d_3$ | $\log(1+n_{H2}) - \log(1+n_{H1})$ y $\tilde s^C_{H2} - \tilde s^C_{H1}$ | Tendencia de volumen y de composición dentro del año |
| I4 vecindad | $v_1$, $v_2$ | $\tilde s^C$ agregada y $\log$ de eventos medios de los departamentos contiguos | Validación espacial débil (AUC 0,80) |

**¿Por qué contraer las proporciones?** Con $n$ eventos, $s^C$ tiene un error estándar de hasta $\sqrt{0{,}25/n}$.
La etiqueta exige $n \ge 10$ en $t$, pero las predictoras de $t-1$ o de un semestre pueden venir de pocos eventos.
La contracción bayesiana (un prior Beta(2, 2), equivalente a 4 eventos "neutros") acerca a 0,5 las proporciones
basadas en pocos datos y casi no toca las bien medidas: con 100 eventos, el ajuste máximo es de 2 pp. En §7 se
mide qué pasa sin ella.

#### Celda de código 5: 2.4 Dataset U3 y verificación contra el dataset de la U2
Salida de la celda:
```text
Reproducción U2: 171 filas, 30 departamentos, etiquetas idénticas, diferencia máxima en las 7 predictoras = 1.7e-13
UCDP (x3): disponible, 843 filas departamento-año
Tensor de secuencias trimestrales (comparativos GRU/CNN): (171, 4, 5)
                 Unnamed: 0  Media  Desv. est.   Mín.  Mediana  Máx.
              m₁ s_C en t-1  0.471       0.247  0.033    0.495 0.898
              m₂ s_C en t-2  0.467       0.256  0.035    0.480 0.898
  m₃ cambio s_C (t-1 − t-2)  0.004       0.140 -0.562    0.000 0.604
    d₁ s_C 2.º semestre t-1  0.479       0.237  0.040    0.500 0.900
    d₂ tendencia de eventos -0.142       0.531 -2.730   -0.045 1.386
        d₃ tendencia de s_C  0.012       0.123 -0.333    0.014 0.384
      v₁ s_C de vecinos t-1  0.512       0.179  0.129    0.517 0.855
v₂ eventos de vecinos (log)  4.947       0.722  2.120    5.083 6.005
```

#### Celda de código 6: 2.5 EDA de las variables nuevas: ¿qué información agregan sobre la inercia?
Salida de la celda:
```text
[figura]
Transiciones (cambio de perfil confiable): 23 | en 70% el 2.º semestre de t-1 ya se movía hacia el perfil de t
                 Unnamed: 0  ρ Spearman con y  Información mutua
              m₁ s_C en t-1             0.723              0.362
              m₂ s_C en t-2             0.707              0.357
  m₃ cambio s_C (t-1 − t-2)            -0.030              0.000
    d₁ s_C 2.º semestre t-1             0.753              0.392
    d₂ tendencia de eventos            -0.051              0.000
        d₃ tendencia de s_C             0.160              0.000
      v₁ s_C de vecinos t-1             0.255              0.109
v₂ eventos de vecinos (log)            -0.153              0.055
```

**Lectura.**

* **La composición reciente es la información más fuerte.** $d_1$ (s_C del segundo semestre de $t-1$) tiene la mayor
  asociación monótona con $y$ (ρ = 0,75) y $m_1$ le sigue (0,72). Superan a cualquier variable estructural de la U1:
  la mejor era la letalidad, con ρ = 0,62.
* **$d_1$ adelanta cambios.** En el **70 %** de las 23 transiciones de perfil, el segundo semestre del año anterior
  ya se movía en la dirección del nuevo perfil (panel b: los círculos negros se separan de la diagonal). Es la
  justificación cuantitativa de I2: la vista anual promedia y diluye el giro reciente.
* **Las tendencias no traen información propia.** $m_3$, $d_2$ y $d_3$ tienen información mutua ≈ 0 con $y$. Son
  diferencias de variables que ya están en el modelo, y con 171 filas es poco probable que aporten. §6 lo comprueba.
* **Colinealidad esperada.** $m_1$, $m_2$ y $d_1$ están muy correlacionadas entre sí (panel c). Por eso la cascada de
  §6 agrega una a la vez y exige que cada una mejore sobre la anterior.

---
# 3. Protocolo experimental de la Unidad 3

## 3.1 Reglas fijadas antes de ver resultados

1. **Selección solo con origen móvil 2023–2025.** Para cada año $v \in \{2023, 2024, 2025\}$ se entrena con
   $t < v$ y se predice $v$ (87 predicciones fuera de muestra). El escalador se ajusta en cada pliegue solo con su
   entrenamiento (prueba automática en §11.3).
2. **2026 es la prueba final y se usa una sola vez**, con el modelo ya elegido. Hay que declararlo: la U2 ya reportó
   métricas sobre 2026, así que no es un conjunto virgen. Ninguna decisión de esta unidad se tomó mirándolo.
3. **El rival es la persistencia** ($\hat p_{d,t} = s^C_{d,t-1}$; AUC 0,973 y log-loss 0,363 en 2023–2025), no el
   MLP base. Se reporta además a qué transiciones acierta cada modelo, porque la persistencia acierta 0 por
   construcción.
4. **Métrica de decisión: log-loss** (calidad probabilística completa), con AUC, Brier, MCC y ECE como métricas de
   apoyo. La exactitud con umbral fijo es la métrica menos estable con 29 departamentos por año (U2 §11.2).
5. **Regla de retención de una intervención.** Cada variante se entrena con las **mismas 5 semillas** que su
   referencia. Se conserva si la mejora media de la log-loss supera **un error estándar** de la diferencia pareada
   entre semillas, $\overline{\Delta LL} + \mathrm{EE}(\Delta LL) < 0$, y no baja el AUC medio más de 0,01. Si no se
   cumple, se prefiere la variante más simple (espíritu de la regla de 1 EE de la U2, D7).
6. **Presupuesto de entrenamiento fijo** de 560 pasos de gradiente (D10 de la U2), salvo cuando el presupuesto es
   la variable estudiada (§6.8). Así ninguna partición usa sus propios datos de prueba para detener el
   entrenamiento.
7. **Los controles de §7 verifican, no seleccionan.** Ponen a prueba decisiones de preprocesamiento heredadas de la
   U1 y la U2. Si un control favorece quitar una transformación, se reporta y queda como tarea para la siguiente
   iteración, pero no cambia el modelo dentro de esta. Si lo cambiara, se sumarían grados de libertad a la
   selección sobre las mismas 87 predicciones.
8. **Objetivo del proyecto por encima de la métrica.** Una variante que gana en log-loss solo porque se parece más a
   la persistencia (menos transiciones acertadas) o porque no puede aplicarse a territorios nuevos se documenta como
   alternativa, pero no se adopta. El valor diferencial de un modelo frente a la inercia está en los cambios de
   perfil y en la migración a municipio (M3 de la U2).

#### Celda de código 7: 3.2 Herramientas de evaluación: semillas pareadas, regla de retención y caché de resultados
Salida de la celda:
```text
Persistencia, origen móvil 2023-2025 (n = 87): AUC 0.973 | log-loss 0.363 | Brier 0.106 | transiciones acertadas 0 de 9
[figura]
```

---
# 4. Línea base de la U2 reproducida

Antes de intervenir comprobamos que el modelo base de la U2 se reproduce con el paquete nuevo. Primero con el mismo
protocolo de la U2 (origen móvil 2023–2026, semilla 42, 560 pasos), contra las cifras guardadas en
`referencias_u2/`. Después, con el protocolo de esta unidad (2023–2025, 5 semillas), que es la línea "antes" de §9.

#### Celda de código 8: 4.1 Reproducción con el protocolo de la U2 (2023-2026) y con el de la U3 (2023-2025)
Salida de la celda:
```text
Unnamed: 0_level_0 Documentado en la U2                                Reproducido (U3)                               
Unnamed: 0_level_1              ROC-AUC Log-loss Brier Exactitud   MCC          ROC-AUC Log-loss Brier Exactitud   MCC
       MLP base U2                0.892    0.436 0.140     0.802 0.619            0.894    0.437 0.140     0.802 0.619
    Logística base                0.875    0.448 0.145     0.802 0.606            0.875    0.448 0.145     0.802 0.606
      Persistencia                0.965    0.380 0.112     0.888 0.776            0.965    0.380 0.112     0.888 0.776
Diferencia máxima entre lo documentado y lo reproducido: 0.0015
              Unnamed: 0  ROC-AUC  Log-loss  Brier   MCC  Exact. balanceada   ECE Log-loss (DE)  Transiciones acertadas  Transiciones
             MLP base U2    0.921     0.383  0.123 0.632              0.808 0.086         0.021                     2.4             9
          Logística base    0.903     0.412  0.131 0.660              0.828 0.063         0.000                     3.0             9
U2 E1 (memoria cruda x8)    0.938     0.347  0.110 0.664              0.826 0.088         0.009                     2.4             9
            Persistencia    0.973     0.363  0.106 0.795              0.897 0.173             –                     0.0             9
```

**Lectura.** La reproducción es casi exacta. La logística y la persistencia coinciden al dígito con la U2, y el MLP
difiere en 0,0015 de AUC (0,894 vs. 0,892), lo esperable al pasar de Windows a Linux con el mismo TensorFlow
2.21. Con el protocolo de la U3 (2023–2025, 5 semillas), el punto de partida es:

* **El MLP base (log-loss 0,383 ± 0,021) pierde contra la persistencia (0,363)**, y también en AUC (0,921 vs. 0,973).
  Es el problema que esta unidad debe resolver.
* El experimento E1 de la U2 (memoria cruda $x_8$) ya baja la log-loss a 0,347 y supera a la persistencia. Lo
  confirmamos con más semillas y un protocolo más estricto: la memoria es la palanca correcta.
* El modelo base acierta en promedio 2,4 de las 9 transiciones de 2023–2025. La persistencia acierta 0.

---
# 5. Diagnóstico de la U2 → plan de intervención

Cada intervención responde a un hallazgo medido en la U2 y tiene una hipótesis que puede fallar. La columna "qué
la refutaría" se escribió antes de ejecutar los experimentos.

| # | Diagnóstico U2 (evidencia) | Intervención | Hipótesis | Qué la refutaría |
|---|---|---|---|---|
| I1 | La inercia domina: persistencia AUC 0,965; 57 % de los errores son transiciones | Memoria $m_1$–$m_3$ | Se acerca a la persistencia y la supera en log-loss | ΔLL dentro del ruido entre semillas |
| I2 | Solo vista anual; los errores de 2025 son escaladas recientes | Dinámica intra-anual $d_1$–$d_3$ | Mejora en transiciones | Sin ganancia en transiciones ni en LL |
| I3 | Arauca: C los seis años con P ≈ 0,36; falta $x_3$ | Actores armados (UCDP) | Corrige departamentos con actor dominante | Arauca sigue mal clasificada |
| I4 | Sin información espacial; validación espacial débil | Vecindad $v_1$, $v_2$ | Mejora la validación espacial | Sin mejora espacial |
| I5 | 20 % de las celdas a ±10 pp del umbral | Objetivo blando ($s^C$) ponderado por eventos | Menos ruido de etiqueta, mejor LL | LL peor incluso tras calibrar |
| I6 | 4 neuronas mejor que 16 a largo plazo; brecha sin cerrar | Capacidad, activación, L2 y presupuesto, con validación anidada | Baja la varianza | Lo elegido no transfiere al año siguiente |
| I7 | AUC 0,848 ± 0,024 entre semillas | Ensamble de semillas | Estabilidad | Sin reducción de dispersión |
| I8 | Subestima P(C); el umbral F2 no se transfiere | Calibración (Platt / temperatura) y umbral multianual | Mejor Brier y umbral estable | La calibración no mejora fuera de muestra |
| I9 | A/B nunca dominan: la etiqueta binaria las diluye | Salida multitarea de composición A/B/C/D | Alerta A/B modelada | No supera a la persistencia de la composición |
| – | Arquitectura MLP sin estructura temporal | Comparativos: GRU, CNN 1D, embedding de departamento | Una arquitectura secuencial aprovecha los trimestres | No ganan al MLP con las mismas variables |

Las intervenciones sobre **variables** (I1–I4) se aplican en cascada: cada una se evalúa sobre la mejor
especificación retenida hasta ese momento. Las de **modelo** (I5–I9) se evalúan después sobre el conjunto de
variables ya fijado, para no mezclar efectos.

---
# 6. Intervenciones incrementales

Cada experimento entrena la variante con las mismas 5 semillas que su referencia y la evalúa en los tres pliegues
de origen móvil. Se reportan la **media entre semillas** (lo que vale una red individual) y la decisión de la
regla 5 de §3.1. El ensamble de semillas se estudia aparte en I7, para no confundir su efecto con el de las
variables.

## 6.1 I1: memoria del perfil

#### Celda de código 9: 6.1 I1 memoria: m1 (s_C contraída de t-1) y la trayectoria m1, m2, m3
Salida de la celda:
```text
              Unnamed: 0  ROC-AUC  Log-loss  Brier   MCC  Exact. balanceada   ECE  Log-loss (DE)  Transiciones acertadas  Transiciones
             MLP base U2    0.921     0.383  0.123 0.632              0.808 0.086          0.021                     2.4             9
U2 E1 (memoria cruda x8)    0.938     0.347  0.110 0.664              0.826 0.088          0.009                     2.4             9
                 I1a +m1    0.939     0.347  0.110 0.664              0.826 0.084          0.010                     2.4             9
           I1b +m1 m2 m3    0.938     0.349  0.108 0.681              0.838 0.086          0.025                     1.6             9
         Intervención  Referencia  ΔLog-loss  EE  ΔAUC      Decisión
        I1 memoria m1 MLP base U2       -358  94   174    se retiene
I1 trayectoria m2, m3     I1a +m1         16 110    -1 no se retiene
Variables retenidas tras I1: ['m1_sC_t1']
```

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

## 6.2 I2: dinámica intra-anual

#### Celda de código 10: 6.2 I2 dinámica intra-anual sobre la especificación retenida
Salida de la celda:
```text
   Unnamed: 0  ROC-AUC  Log-loss  Brier   MCC  Exact. balanceada   ECE  Log-loss (DE)  Transiciones acertadas  Transiciones
      I1a +m1    0.939     0.347  0.110 0.664              0.826 0.084          0.010                     2.4             9
      I2a +d1    0.951     0.310  0.097 0.740              0.865 0.077          0.010                     2.0             9
I2b +d1 d2 d3    0.945     0.323  0.102 0.691              0.842 0.071          0.012                     2.2             9
           Intervención Referencia  ΔLog-loss  EE  ΔAUC      Decisión
I2 semestre reciente d1    I1a +m1       -374  68   127    se retiene
   I2 tendencias d2, d3    I2a +d1        130  87   -59 no se retiene
   particion  2023  2024  2025
 MLP base U2 0.367 0.316 0.434
     I1a +m1 0.351 0.272 0.386
     I2a +d1 0.313 0.224 0.362
Persistencia 0.344 0.346 0.399
Variables retenidas tras I2: ['m1_sC_t1', 'd1_sC_H2']
```

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

## 6.3 I3: actores armados activos ($x_3$, UCDP GED)

$x_3$ está pendiente desde la Etapa 1, porque el export gratuito de ACLED no trae actores. La fuente sustituta
verificada es **UCDP GED 26.1** (`ged261-csv.zip`, archivo `GEDEvent_v26_1.csv`, 39 MB). Cubre Colombia hasta 2025,
trae `adm_1` en el 99,8 % de los eventos desde 2020 y registra los dos bandos (`side_a`, `side_b`). La variable cuenta
los actores **no estatales** distintos por departamento–año: excluye gobiernos y "Civilians" y separa pares como
"ELN, FARC-EMC". Como UCDP solo registra violencia organizada letal, es un indicador **aproximado** de grupos
activos.

Como el archivo completo pesa 39 MB, el repositorio incluye `herramientas/preparar_ucdp.py`. El script lo reduce a
Colombia (`ucdp_colombia.csv`, unos cientos de KB) y a $x_3$ ya agregado (`ucdp_actores_departamento_anio.csv`, unos
pocos KB). La celda siguiente corre I3 si cualquiera de esos archivos está en `datos/externos/`. Si no está, lo declara
y sigue: el resto del cuaderno no depende de esta intervención.

#### Celda de código 11: 6.3 I3 actores armados (condicionada a la disponibilidad de UCDP GED)
Salida de la celda:
```text
Unnamed: 0  ROC-AUC  Log-loss  Brier   MCC  Exact. balanceada   ECE  Log-loss (DE)  Transiciones acertadas  Transiciones
   I2a +d1    0.951     0.310  0.097 0.740              0.865 0.077           0.01                     2.0             9
    I3 +x3    0.937     0.335  0.105 0.684              0.840 0.064           0.02                     1.4             9
         Intervención Referencia  ΔLog-loss   EE   ΔAUC      Decisión
I3 actores armados x3    I2a +d1      24955 5638 -14482 no se retiene
   departamento  anio perfil     p_ref      p_x3
2        Arauca  2023      C  0.823462  0.907339
31       Arauca  2024      C  0.742372  0.815017
60       Arauca  2025      C  0.837484  0.842019
```

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
presencia sostenida del ELN. La conclusión no es que los actores no importen, sino que **un conteo anual
departamental no es la forma útil de medirlos**. A escala municipal, el cambio en los actores presentes (entradas y
salidas de grupos) es la versión que vale la pena probar (R2, §13).

## 6.4 I4: vecindad espacial

La hipótesis de I4 es **espacial**: la composición de los vecinos debería ayudar sobre todo con departamentos que el
modelo nunca vio. Por eso la evaluamos con los dos protocolos de la U2, el temporal y el espacial (5 pliegues
agrupados por departamento, años hasta 2025). Se retiene si mejora la validación espacial según la regla 5 sin
empeorar la temporal.

#### Celda de código 12: 6.4 I4 vecindad: validación temporal y espacial
Salida de la celda:
```text
                     Unnamed: 0                Unnamed: 1  ROC-AUC  Log-loss  Brier   MCC  Exact. balanceada   ECE
             Temporal 2023-2025                   I2a +d1    0.951     0.310  0.097 0.740              0.865 0.077
             Temporal 2023-2025                 I4 +v1 v2    0.953     0.306  0.097 0.706              0.847 0.081
Espacial (deptos. nunca vistos)          esp: MLP base U2    0.854     0.475  0.154 0.547              0.768 0.054
Espacial (deptos. nunca vistos)       esp: logística base    0.852     0.505  0.158 0.534              0.766 0.050
Espacial (deptos. nunca vistos)           esp: referencia    0.914     0.379  0.118 0.662              0.829 0.045
Espacial (deptos. nunca vistos) esp: logística referencia    0.916     0.373  0.114 0.704              0.851 0.034
Espacial (deptos. nunca vistos)            esp: I4 +v1 v2    0.908     0.389  0.123 0.648              0.822 0.055
Espacial (deptos. nunca vistos)         esp: persistencia    0.914     0.418  0.127 0.704              0.852 0.117
          Intervención      Referencia  ΔLog-loss  EE  ΔAUC      Decisión
I4 vecindad (temporal)         I2a +d1        -40  19    20 no se retiene
I4 vecindad (espacial) esp: referencia         94  37   -54 no se retiene
Especificación de variables fijada (9): ['x1_eventos', 'x2_fatalidades', 'x2b_letalidad', 'x4_coca', 'x5_bosque', 'x6_distancia', 'x7_ipm', 'm1_sC_t1', 'd1_sC_H2']
```

**Lectura.**

* **I4 no se retiene.** En el eje temporal, la vecindad da una mejora mínima (Δ = −0,004, unas 2 veces su EE). En el
  eje donde la hipótesis la necesitaba, el **espacial**, empeora: log-loss 0,389 vs. 0,379 (Δ = +0,009). Con 30
  departamentos, la composición de los vecinos suma colinealidad con la memoria propia, pero no información
  transferible.
* **El hallazgo espacial importante es otro.** Con $m_1 + d_1$, la generalización a departamentos nunca vistos pasa de
  log-loss 0,475 (modelo base) a **0,379**, y el AUC de 0,854 a 0,914. La debilidad espacial de la U2 (§11.3) se debía
  sobre todo a la falta de memoria: las variables estructurales casi constantes funcionaban como identificadores.
  Con información de composición, la red ya no necesita memorizar firmas departamentales.
* La logística con las mismas variables queda empatada en el eje espacial (0,373). El aporte de la no linealidad
  sigue sin demostrarse a esta escala (§9).

## 6.5 I5: objetivo blando ponderado por eventos

En lugar de la etiqueta dura $y = \mathbb{1}[s^C \ge 0{,}5]$, la red aprende directamente $s^C_{d,t}$, con la
entropía cruzada aplicada a un objetivo continuo, opcionalmente ponderada por $\sqrt{n_{d,t}}$. La idea es que una
celda con $s^C = 0{,}51$ no le "enseñe" lo mismo que una con $s^C = 0{,}95$. La salida pasa a estimar la
**proporción esperada**, no la probabilidad de que la proporción supere 0,5. Por eso la comparación justa es después
de **calibrar** ambas variantes con el mismo método (temperatura con encadenamiento hacia adelante: el pliegue $v$
se calibra solo con los pliegues anteriores).

#### Celda de código 13: 6.5 I5 objetivo blando (con y sin ponderación por eventos), crudo y calibrado
Salida de la celda:
```text
                    Unnamed: 0  ROC-AUC  Log-loss  Brier   MCC  Exact. balanceada   ECE  Log-loss (DE)  Transiciones acertadas  Transiciones
                       I2a +d1    0.951     0.310  0.097 0.740              0.865 0.077          0.010                     2.0             9
                     I5 blando    0.947     0.429  0.132 0.731              0.861 0.172          0.003                     2.8             9
      I5 blando + peso eventos    0.945     0.431  0.135 0.709              0.847 0.159          0.007                     2.4             9
                 I2a +d1 · cal    0.944     0.317  0.099 0.740              0.865 0.069          0.020                     2.0             9
               I5 blando · cal    0.929     0.385  0.117 0.731              0.861 0.110          0.035                     2.8             9
I5 blando + peso eventos · cal    0.929     0.401  0.123 0.709              0.847 0.113          0.033                     2.4             9
                         Intervención    Referencia  ΔLog-loss  EE  ΔAUC      Decisión
    I5 objetivo blando (sin calibrar)       I2a +d1       1193  44   -44 no se retiene
I5 objetivo blando (ambos calibrados) I2a +d1 · cal        678  78  -152 no se retiene
Objetivo de entrenamiento elegido: duro
```

**Lectura.** **I5 se rechaza.** Sin calibrar, el objetivo blando empeora mucho la log-loss (0,429 vs. 0,310). Era lo
previsto: la red aprende la proporción esperada, que se concentra cerca de 0,5, y no la probabilidad de cruzar el
umbral. Incluso después de calibrar ambas variantes con el mismo método, sigue peor (0,385 vs. 0,317) y pierde AUC
(0,929 vs. 0,944). La ponderación por $\sqrt{n}$ no ayuda. Concluimos que el ruido de etiqueta cerca del umbral no es
lo que limita al modelo: la etiqueta dura de la U1 se mantiene, y con ella la trazabilidad de la variable objetivo.

## 6.6 I9: salida multitarea de composición A/B/C/D

La etiqueta binaria diluye el hallazgo crítico de la U1: explosivos (A) y drones (B) son de bajo volumen y alto
impacto. Agregamos a la red una **segunda cabeza softmax** que predice la composición completa
$(s^A, s^B, s^C, s^D)_{d,t}$, con un tronco oculto compartido:

$$
\mathcal{L} = \underbrace{\mathrm{BCE}(y, \hat p)}_{\text{perfil C}} \;+\; \lambda \cdot
\underbrace{\sum_k s^k \log \frac{s^k}{\hat s^k}}_{\mathrm{KL}(\mathbf s \,\|\, \hat{\mathbf s})} \;+\; \lambda_2 \lVert \mathbf W \rVert^2 .
$$

Se evalúa de dos formas: (1) si la tarea auxiliar ayuda o perjudica a $\hat p$ (regla 5) y (2) si la composición
predicha anticipa la **alerta A/B** mejor que repetir la composición del año anterior.

#### Celda de código 14: 6.6 I9 multitarea: efecto sobre P(C) y calidad de la composición predicha
Salida de la celda:
```text
         Unnamed: 0  ROC-AUC  Log-loss  Brier   MCC  Exact. balanceada   ECE  Log-loss (DE)  Transiciones acertadas  Transiciones
            I2a +d1    0.951     0.310  0.097 0.740              0.865 0.077          0.010                     2.0             9
I9 multitarea λ=0.5    0.951     0.311  0.097 0.740              0.865 0.085          0.009                     2.2             9
  I9 multitarea λ=1    0.947     0.319  0.099 0.731              0.861 0.082          0.015                     2.4             9
                                       Unnamed: 0  Modelo multitarea (λ=0,5)  Persistencia de la composición
                                 MAE de s_A + s_B                      0.035                           0.030
                 ρ de Spearman con s_A + s_B real                      0.465                           0.601
AUC para detectar alerta A/B (≥ 2× media del año)                      0.682                           0.700
                        Intervención Referencia  ΔLog-loss  EE  ΔAUC      Decisión
I9 multitarea λ=0.5 (efecto en P(C))    I2a +d1         13  27    -1 no se retiene
  I9 multitarea λ=1 (efecto en P(C))    I2a +d1         95  52   -42 no se retiene
Filas con alerta A/B real en 2023-2025: 14 de 87 | tipo de red elegido: mlp
```

**Lectura.** **I9 se rechaza, y el resultado negativo es informativo.**

* Con λ = 0,5, la tarea auxiliar no perjudica a $\hat p$ (Δ = +0,001), pero tampoco ayuda. Con λ = 1 empeora
  (Δ = +0,010).
* La composición A/B predicha **no supera a la persistencia de la composición**: MAE 0,035 vs. 0,030, ρ = 0,47 vs.
  0,60, y AUC para detectar la alerta A/B de 0,68 vs. 0,70 (14 de las 87 filas tienen alerta).
* La proporción de explosivos y drones de un departamento es tan persistente como su perfil, y la red no encuentra en
  las variables de la U1 nada que anticipe mejor sus cambios. La alerta A/B operativa sigue siendo la regla
  descriptiva de la U2 (proporción observada en 2025–2026 de al menos el doble de la nacional). Modelarla exigiría
  datos por evento y municipio, donde A y B tienen variación suficiente.

## 6.7 Comparativos de arquitectura: ¿una red secuencial o con identidad territorial?

Con las mismas variables tabulares, comparamos el MLP con:

* **GRU** y **CNN 1D** sobre la secuencia de los 4 trimestres de $t-1$ (eventos, fatalidades, $\tilde s^A$,
  $\tilde s^B$, $\tilde s^C$), concatenada a las variables tabulares. Responde si la forma de la trayectoria
  trimestral aporta más que su resumen semestral ($d_1$).
* **Embedding de departamento** (dimensión 2): un vector aprendido por territorio. Captura efectos fijos, pero no
  puede generalizar a departamentos nunca vistos, así que lo evaluamos también en la validación espacial.
* **MLP profundo** (16, 8) y **regresión logística** con las mismas variables (¿aporta la no linealidad?).

#### Celda de código 15: 6.7 Comparativos: GRU, CNN 1D, embedding, MLP profundo y logística con las mismas variables
Salida de la celda:
```text
                  Unnamed: 0  ROC-AUC  Log-loss  Brier   MCC  Exact. balanceada   ECE  Log-loss (DE)  Transiciones acertadas  Transiciones  Parámetros
                     I2a +d1    0.951     0.310  0.097 0.740              0.865 0.077          0.010                     2.0             9         177
              GRU trimestral    0.948     0.319  0.104 0.706              0.849 0.076          0.015                     1.8             9         665
           CNN 1D trimestral    0.945     0.324  0.103 0.724              0.858 0.078          0.023                     1.8             9         393
   Embedding de departamento    0.968     0.284  0.090 0.732              0.863 0.081          0.018                     0.8             9         275
        MLP profundo (16, 8)    0.954     0.309  0.098 0.731              0.863 0.083          0.025                     1.6             9         305
Logística (mismas variables)    0.949     0.306  0.097 0.737              0.863 0.068          0.000                     1.0             9          10
                Intervención Referencia  ΔLog-loss  EE  ΔAUC                                                               Decisión
              GRU trimestral    I2a +d1         98  45   -33                                                          no se retiene
           CNN 1D trimestral    I2a +d1        143 142   -64                                                          no se retiene
   Embedding de departamento    I2a +d1       -259 103   163 cumple la regla, no se adopta: acierta 0.8 de 9 transiciones (refer...
        MLP profundo (16, 8)    I2a +d1         -4 135    26                                                          no se retiene
Logística (mismas variables)    I2a +d1         27   0     5                                                          no se retiene
Embedding en validación espacial: log-loss 0.369 vs. MLP sin embedding 0.379
```

**Lectura.**

* **GRU (665 parámetros) y CNN 1D (393) empeoran** (log-loss 0,319 y 0,324 vs. 0,310). Los cuatro trimestres de
  $t-1$ no contienen más información que su resumen semestral $d_1$, y la red secuencial paga en varianza los
  parámetros extra. Con 4 pasos de tiempo y 142 filas, una arquitectura recurrente no tiene de dónde aprender.
* **El MLP profundo (16, 8) empata** (Δ = −0,0004; EE = 0,014): la profundidad no aporta, igual que en la U2.
* **La logística con las mismas variables empata** (0,306 vs. 0,310). A escala departamental, la información nueva pesa
  más que la flexibilidad del modelo, lo mismo que en la U2 (§11.4). El MLP se mantiene porque no es peor, porque es la
  arquitectura comprometida en la U1 y porque es la que debe escalar a municipio, donde habrá unidades para estimar
  interacciones.
* **El embedding de departamento es el único que cumple la regla 5** (log-loss 0,284; Δ = −0,026; EE = 0,010). **No se
  adopta por la regla 8**, que formalizamos al encontrar este caso, y lo declaramos así:
  1. Acierta **0,8 de 9 transiciones** frente a 2,0 de la referencia. Un vector fijo por territorio acerca el modelo a la
     persistencia, que ya es gratis, y le quita su nicho.
  2. No transfiere. Su resultado espacial (0,369) no empeora solo porque los departamentos nunca vistos reciben el
     vector inicial, cercano a cero: el modelo vuelve a ser el MLP sin embedding. Toda su ganancia temporal viene de
     efectos fijos memorizados.
  3. Es incompatible con la migración a municipio (M3), donde la mayoría de unidades no tendría historia suficiente
     para estimar su vector.

  Para transparencia, en §9.3 reportamos también su desempeño en 2026.

## 6.8 I6: capacidad, activación, regularización y presupuesto, con validación anidada

La U2 dejó dos señales contradictorias: 4 neuronas generalizaban mejor con 400 épocas fijas, pero la búsqueda con
parada temprana eligió 16. Exploramos una rejilla de **48 configuraciones**: 4 capacidades × 2 L2 × 2 activaciones
(ReLU, tanh) × 3 presupuestos (280, 560 y 1.120 pasos), con 3 semillas, sobre las variables ya fijadas. Además
hacemos una **validación anidada** que pone a prueba el propio proceso de selección: elegimos la configuración con
los pliegues 2023–2024 y medimos si lo elegido es mejor en 2025, el año que no participó en la elección.

#### Celda de código 16: 6.8 I6 rejilla de capacidad x L2 x activación x presupuesto (3 semillas, 3 pliegues)
Salida de la celda:
```text
48 configuraciones × 3 semillas × 3 pliegues = 432 entrenamientos en 264 s
Mejor media: ('(32,)', np.float64(0.01), 'relu', np.int64(1120)) LL 0.289 ± 0.028 | configuraciones dentro de 1 EE: 31 de 48
Configuración vigente ('(16,)', 0.001, 'relu', 560): LL 0.303 (puesto 18 de 48; dentro de 1 EE del mejor)
                           Unnamed: 0                                        Configuración LL 2023-2024 (selección)  LL 2025 (fuera de la selección)
                  Mínimo en 2023-2024 ('(32,)', np.float64(0.001), 'relu', np.int64(1120))                    0.234                            0.423
              Regla 1 EE en 2023-2024  ('(4,)', np.float64(0.001), 'tanh', np.int64(1120))                    0.259                            0.411
                         Vigente (U2)                        ('(16,)', 0.001, 'relu', 560)                    0.275                            0.359
Oráculo (mejor en 2025, inalcanzable)  ('(32,)', np.float64(0.001), 'relu', np.int64(280))                        –                            0.333
  Unnamed: 0  ROC-AUC  Log-loss  Log-loss (DE)  Brier
     I2a +d1    0.951     0.310          0.010  0.097
I6 lr=0.0003    0.886     0.452          0.070  0.147
 I6 lr=0.003    0.950     0.319          0.017  0.101
[figura]
```

**Lectura.** **I6 es la lección metodológica de la unidad.**

* **La superficie es plana.** De 48 configuraciones, **31 quedan dentro de un error estándar** de la mejor (32
  neuronas, L2 = 10⁻², 1.120 pasos; log-loss 0,289 ± 0,028). La vigente de la U2 (16 ReLU, L2 = 10⁻³, 560 pasos) queda
  en el puesto 18 (0,303), dentro de esa banda. Los mapas (a)–(b) muestran el único patrón robusto: las redes pequeñas
  (4–8 neuronas) necesitan más pasos, y con 280 pasos no alcanzan a converger. Con 560 o más, casi todo es equivalente.
  ReLU y tanh también empatan.
* **La validación anidada demuestra que elegir aquí sobreajusta la selección.** La configuración con mejor log-loss en
  2023–2024 (0,234) obtiene **0,423 en 2025**. La que elige la regla de 1 EE en 2023–2024 obtiene 0,411. La vigente,
  que nadie eligió mirando esos datos, obtiene **0,359**. Optimizar hiperparámetros con 58 filas de validación ajusta
  el ruido de esos dos años, y lo elegido empeora al año siguiente. Es el sesgo de selección descrito por Cawley y
  Talbot (2010).
* **Tasa de aprendizaje.** Con el presupuesto fijo, η = 3×10⁻⁴ deja la red sin entrenar (log-loss 0,452) y
  η = 3×10⁻³ no mejora (0,319). η = 10⁻³ está en el punto correcto para 560 pasos.
* **Decisión:** se mantiene la configuración de la U2. No es pereza, es la intervención que la evidencia respalda: la
  capacidad no es el factor limitante, y cambiarla sería reemplazar una configuración validada fuera de muestra por una
  elegida sobre ruido.

## 6.9 I7: ensamble de semillas

#### Celda de código 17: 6.9 I7 ensamble: 20 semillas de la especificación vigente y curva de tamaño del ensamble
Salida de la celda:
```text
Unnamed: 0_level_0           LL media              LL p5             LL p95          AUC media              LL DE
                 K Unnamed: 1_level_1 Unnamed: 2_level_1 Unnamed: 3_level_1 Unnamed: 4_level_1 Unnamed: 5_level_1
                 1             0.3077             0.2833             0.3366             0.9531             0.0142
                 2             0.3010             0.2859             0.3168             0.9556             0.0094
                 3             0.2991             0.2859             0.3107             0.9566             0.0076
                 5             0.2980             0.2905             0.3079             0.9566             0.0052
                 7             0.2978             0.2909             0.3043             0.9573             0.0042
                10             0.2970             0.2924             0.3018             0.9574             0.0028
                15             0.2969             0.2940             0.2993             0.9579             0.0017
                20             0.2965             0.2965             0.2965             0.9567             0.0000
Dispersión de P(C) entre semillas por fila: mediana 0.047, máx. 0.128 | filas con predicción unánime entre 20 redes: 85%
[figura]
```

**Lectura.** **I7 se retiene por estabilidad, no por desempeño medio.** Una red individual tiene log-loss esperada de
0,308, pero según la semilla puede caer en cualquier punto entre 0,283 y 0,337 (banda del 5–95 %). Con 10 redes la
media apenas cambia (0,297), pero la banda se estrecha a **0,292–0,302**, unas 6 veces menos. Más allá de 10 la ganancia
es marginal. La dispersión de $P(C)$ entre redes por fila tiene mediana de 0,047 y máximo de 0,128, y el 85 % de las
filas recibe la misma clasificación de las 20 redes. El ensamble convierte un resultado que dependía de la semilla
(el problema de la semilla 42 en la U2, D10) en uno reproducible, y su desviación entre redes sirve de medida de
incertidumbre por departamento en §12.

## 6.10 I8: calibración y umbral de decisión

La U2 vio probabilidades por debajo de la frecuencia real de C fuera de muestra y un umbral F2 (0,17) que no se
transfirió a 2026. Probamos dos calibradores sobre el ensamble ($K = 10$): **Platt** ($\sigma(a\,\mathrm{logit}\,p + b)$),
que corrige escala y sesgo, y **temperatura** ($\sigma(\mathrm{logit}\,p / T)$), que solo corrige la escala. Se
evalúan de dos formas:

* **Encadenamiento hacia adelante** (honesta en el tiempo): 2024 se calibra con 2023, y 2025 con 2023–2024.
* **Validación cruzada por año** (dejar un año fuera): usa más datos, pero calibra 2023 con información de 2025.

Si la calibración solo sirve cuando se mira el futuro, no se adopta.

#### Celda de código 18: 6.10 I8 calibración (Platt / temperatura) y estabilidad del umbral óptimo entre años
Salida de la celda:
```text
              Unnamed: 0_level_0       LL 2024-2025    Brier 2024-2025      ECE 2024-2025       LL 2023-2025      ECE 2023-2025
                     Calibración Unnamed: 1_level_1 Unnamed: 2_level_1 Unnamed: 3_level_1 Unnamed: 4_level_1 Unnamed: 5_level_1
                    Sin calibrar              0.298              0.093              0.142              0.304              0.124
                Platt · adelante              0.302              0.099              0.097              0.307              0.068
          Temperatura · adelante              0.306              0.100              0.100              0.310              0.071
      Platt · dejar un año fuera              0.330              0.113              0.048              0.348              0.061
Temperatura · dejar un año fuera              0.316              0.102              0.102              0.313              0.071
Parámetros de Platt ajustados en cada año por separado (muestran si el sesgo es estable):
   Unnamed: 0  2023  2024    2025
   a (escala) 1.312 3.379  93.448
    b (sesgo) 0.053 0.132 151.575
Prevalencia C 0.483 0.414   0.621
Umbral que maximiza el MCC en cada año: {'2023': 0.2, '2024': 0.51, '2025': 0.15, 'Agregado 2023-2025': 0.2}
MCC con umbral 0,5: 0.691 | con el óptimo agregado 0.20: 0.810
[figura]
Calibración adoptada: ninguna | umbral operativo: 0.5
```

**Lectura.** **I8 se rechaza para la probabilidad y el umbral se mantiene en 0,5.**

* **La calibración solo mejora si se mira el futuro.** Con encadenamiento hacia adelante (la única evaluación honesta
  en el tiempo), Platt baja el ECE de 0,142 a 0,097, pero **sube la log-loss** de 0,298 a 0,302 y el Brier de 0,093 a
  0,099. La temperatura también empeora (0,306).
* **La corrección no es estable.** Ajustado año por año, Platt da una escala $a$ = 1,31 en 2023 y 3,38 en 2024. En
  2025 diverge ($a \approx 93$, $b \approx 152$) porque la red ordena ese año sin errores: sin solapamiento entre clases, la mejor calibración es un
  escalón. El desajuste de calibración depende de la prevalencia de cada año (41 % a 62 % de C), no de un error
  sistemático de la red. Una corrección estimada con el pasado no sirve para el año siguiente.
* **El umbral óptimo tampoco se transfiere.** El que maximiza el MCC es 0,20 en 2023, 0,51 en 2024 y 0,15 en 2025.
  Usar 0,20 sobre el agregado subiría el MCC de 0,69 a 0,81, pero sería elegirlo con los mismos datos que lo evalúan:
  es el error que la U2 documentó con el umbral F2 (0,17), que no transfirió a 2026.
* **Consecuencia operativa.** Para la decisión no se usa un umbral único, sino las **bandas** de §1.3: un
  departamento con $\hat p$ entre 0,3 y 0,7 es "transición" y recibe portafolio flexible. Así la incertidumbre de
  calibración se gestiona en la decisión, no en el modelo.

## 6.11 Síntesis de las intervenciones

#### Celda de código 19: 6.11 Tabla de decisiones y escalera de mejora
Salida de la celda:
```text
                                Intervención                    Referencia  ΔLog-loss    EE   ΔAUC                                                               Decisión
                               I1 memoria m1                   MLP base U2       -358 00094 +00174                                                             se retiene
                       I1 trayectoria m2, m3                       I1a +m1         16 00110 -00001                                                          no se retiene
                     I2 semestre reciente d1                       I1a +m1       -374 00068 +00127                                                             se retiene
                        I2 tendencias d2, d3                       I2a +d1        130 00087 -00059                                                          no se retiene
                       I3 actores armados x3                       I2a +d1        250 00056 -00145                                                          no se retiene
                      I4 vecindad (temporal)                       I2a +d1        -40 00019 +00020                                                          no se retiene
                      I4 vecindad (espacial)               esp: referencia         94 00037 -00054                                                          no se retiene
           I5 objetivo blando (sin calibrar)                       I2a +d1       1193 00044 -00044                                                          no se retiene
       I5 objetivo blando (ambos calibrados)                 I2a +d1 · cal        678 00078 -00152                                                          no se retiene
        I9 multitarea λ=0.5 (efecto en P(C))                       I2a +d1         13 00027 -00001                                                          no se retiene
          I9 multitarea λ=1 (efecto en P(C))                       I2a +d1         95 00052 -00042                                                          no se retiene
                              GRU trimestral                       I2a +d1         98 00045 -00033                                                          no se retiene
                           CNN 1D trimestral                       I2a +d1        143 00142 -00064                                                          no se retiene
                   Embedding de departamento                       I2a +d1       -259 00103 +00163 cumple la regla, no se adopta: acierta 0.8 de 9 transiciones (refer...
                        MLP profundo (16, 8)                       I2a +d1         -4 00135 +00026                                                          no se retiene
                Logística (mismas variables)                       I2a +d1         27 00000 +00005                                                          no se retiene
I6 capacidad / activación / L2 / presupuesto ('(16,)', 0.001, 'relu', 560)        518     –      –                      no se retiene (la selección no transfiere a 2025)
                  I7 ensamble de 10 semillas                         1 red       -108     – +00043                                               se retiene (estabilidad)
                              I8 calibración                  sin calibrar         42     – +00000                          no se retiene (no mejora sin mirar el futuro)
Unnamed: 0_level_0           LL media              LL DE                AUC              Brier Transiciones acertadas
              Paso Unnamed: 1_level_1 Unnamed: 2_level_1 Unnamed: 3_level_1 Unnamed: 4_level_1     Unnamed: 5_level_1
    Modelo base U2              0.383              0.021              0.921              0.123                    2.4
   + I1 memoria m₁              0.347              0.010              0.939              0.110                    2.4
  + I2 semestre d₁              0.310              0.010              0.951              0.097                    2.0
+ I7 ensamble (10)              0.304              0.000              0.956              0.095                    2.0
[figura]
```

**Lectura de la síntesis.** De las intervenciones evaluadas, **tres se retienen**: $m_1$ (I1), $d_1$ (I2) y el
ensamble (I7). Juntas bajan la log-loss de 0,383 a 0,304 y suben el AUC de 0,921 a 0,956. El resto se descarta con
una razón cuantitativa registrada en la tabla. La escalera muestra el patrón general del proyecto: **la mejora vino de
la representación de los datos, no de la arquitectura ni de los hiperparámetros**. Las dos variables retenidas
resumen el estado reciente del conflicto, que es lo que la U2 identificó como información faltante. Las
intervenciones de modelo (I5, I6, I8, I9, GRU, CNN, profundidad) no superaron el ruido entre semillas. El modelo
final es tan simple como el de la U2 (una capa de 16 neuronas), con dos entradas más y diez veces más estable.

---
# 7. Controles experimentales de la preparación de datos

Cada decisión de preprocesamiento se pone a prueba **quitándola** (o introduciendo a propósito un error conocido)
y midiendo el efecto con el mismo protocolo, las mismas 5 semillas y la especificación final de variables. Un
control "pasa" cuando el resultado se mueve en la dirección que predice la teoría.

| Control | Qué se cambia | Qué esperamos si la decisión original es correcta |
|---|---|---|
| Sin estandarización | Variables en escala original (km, %, log) | Peor convergencia: el gradiente lo dominan las variables de gran escala (distancia, bosque) |
| Sin logaritmos | $x_1$, $x_2$, $x_4$ en conteos y hectáreas crudas | Peor: colas extremas (Cauca, Nariño) dominan el ajuste |
| Sin pesos de clase | BCE sin $\omega_c$ | Efecto pequeño (clases casi balanceadas), algo peor en sensibilidad |
| Sin contracción ($\alpha \approx 0$) | $m_1$, $d_1$ con proporciones crudas | Peor: semestres con pocos eventos producen proporciones extremas |
| Escalador con fuga (control negativo) | Media y desviación ajustadas también con el año de prueba | Mejora artificial pequeña: cuantifica cuánto inflaría una fuga típica |
| `MIN_EVENTOS` = 5 / 20 | Umbral de confiabilidad de la etiqueta en el entrenamiento | 10 equilibra tamaño y ruido de etiqueta (D3) |
| Mapeo alternativo (M8) | *Shelling* → B y *Sexual violence* → C | Sin cambio material (1 etiqueta de 171 cambia, U2 §3.4) |
| Etiquetas barajadas (cordura) | Se entrena con $y$ permutada | AUC ≈ 0,5: el modelo no "aprende" ruido |

#### Celda de código 20: 7.1 Ablaciones de preprocesamiento (5 semillas, origen móvil 2023-2025)
Salida de la celda:
```text
Controles en 98 s
Referencia: LL 0.310 | AUC 0.951
       Unnamed: 0_level_0          ΔLog-loss                 EE               ΔAUC           LL media          AUC media
             Intervención Unnamed: 1_level_1 Unnamed: 2_level_1 Unnamed: 3_level_1 Unnamed: 4_level_1 Unnamed: 5_level_1
ctrl: sin estandarización             5.7803             1.4454            -0.4180             6.0899             0.5332
     ctrl: sin logaritmos            -0.0117             0.0031             0.0049             0.2979             0.9561
 ctrl: sin pesos de clase             0.0154             0.0019            -0.0003             0.3250             0.9510
    ctrl: sin contracción             0.0021             0.0010            -0.0015             0.3117             0.9498
 ctrl: escalador con fuga            -0.0024             0.0012             0.0002             0.3072             0.9515
  ctrl: mapeo alternativo             0.0072             0.0007            -0.0045             0.3168             0.9467
ctrl: etiquetas barajadas             0.5057             0.0119            -0.6351             0.8153             0.3162
Umbral de confiabilidad (evaluación común: filas con >= 20 eventos):
Unnamed: 0_level_0                 LL                AUC filas de entrenamiento (último pliegue)
       MIN_EVENTOS Unnamed: 1_level_1 Unnamed: 2_level_1                      Unnamed: 3_level_1
                 5              0.300              0.951                                     120
                10              0.307              0.951                                     113
                20              0.338              0.940                                     104
```

#### Celda de código 21: 7.2 Impacto en el entrenamiento: curvas de pérdida con y sin cada transformación (pliegue 2025)
Salida de la celda:
```text
[figura]
                                Unnamed: 0  BCE train época 10  BCE train época 70  BCE val época 70  BCE val mínima
Referencia (estandarizado, log, contraído)               0.510               0.280             0.340           0.320
                       Sin estandarización               7.132               7.132            10.004          10.004
                            Sin logaritmos               0.507               0.266             0.373           0.353
                           Sin contracción               0.506               0.277             0.342           0.321
```

**Lectura de los controles.**

* **La estandarización es indispensable.** Sin ella la red no converge: la BCE de entrenamiento queda en 7,1 desde la
  época 10. Las variables en km (distancia hasta 800) y en % (bosque hasta 98) saturan las ReLU y la sigmoide de
  salida, y el gradiente se anula. La log-loss fuera de muestra sube de 0,310 a 6,09 (probabilidades saturadas en 0 o 1) y el AUC cae a 0,53. Es el
  impacto más directo del preprocesamiento sobre el entrenamiento (panel a).
* **Los pesos de clase se justifican** (sin ellos, Δ = +0,015, unas 8 veces el EE). La prevalencia de C varía entre
  años, y sin la ponderación la red se inclina hacia D.
* **La contracción aporta poco pero en la dirección correcta** (sin ella, Δ = +0,002, unas 2 veces el EE): estabiliza
  las proporciones semestrales con pocos eventos.
* **La fuga del escalador da una mejora artificial de 0,002.** Cuantifica cuánto inflaría la métrica ajustar la media
  y la desviación con el año de prueba. Es pequeña aquí porque la estandarización es lineal y las distribuciones
  cambian poco entre años, pero existe, y las pruebas automáticas de §11 la impiden.
* **El mapeo alternativo de subtipos empeora un poco** (+0,007): la regla de la U1 se sostiene.
* **Con etiquetas barajadas, el modelo no aprende** (log-loss 0,815; AUC 0,32). El valor exacto depende de la
  permutación, pero queda lejos de la referencia: el desempeño medido proviene de señal real, no de un artefacto del
  protocolo.
* **`MIN_EVENTOS`.** Evaluando en las mismas filas (≥ 20 eventos), entrenar con 5 (log-loss 0,300) o con 10 (0,307)
  es equivalente, y con 20 empeora (0,338) porque se pierden filas. El umbral de 10 de la U2 queda en la zona
  correcta.
* **Un resultado que no anticipábamos: sin logaritmos, la log-loss mejora levemente** (0,298 vs. 0,310; Δ = −0,012,
  unas 4 veces el EE). Al agregar la memoria, las variables de volumen pasan a segundo plano y la red tolera bien sus
  colas. Por la regla 7, **no cambiamos el modelo en esta iteración**. En el pliegue 2025, el más exigente, la curva
  (b) muestra lo contrario (BCE de validación 0,373 sin logaritmos vs. 0,340), y el logaritmo protege la proyección
  frente a valores extremos nuevos. Queda como tarea explícita para la escala municipal.

---
# 8. Modelo final integrado

## 8.1 Configuración documentada

#### Celda de código 22: 8.1 Especificación final, resumen de Keras y configuración exportada
Salida de la celda:
```text
[1mModel: "red_mlp"[0m

┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃[1m [0m[1mLayer (type)                   [0m[1m [0m┃[1m [0m[1mOutput Shape          [0m[1m [0m┃[1m [0m[1m      Param #[0m[1m [0m┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ x_t_1 ([38;5;33mInputLayer[0m)              │ ([38;5;45mNone[0m, [38;5;34m9[0m)              │             [38;5;34m0[0m │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ oculta_1 ([38;5;33mDense[0m)                │ ([38;5;45mNone[0m, [38;5;34m16[0m)             │           [38;5;34m160[0m │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ P_C ([38;5;33mDense[0m)                     │ ([38;5;45mNone[0m, [38;5;34m1[0m)              │            [38;5;34m17[0m │
└─────────────────────────────────┴────────────────────────┴───────────────┘

[1m Total params: [0m[38;5;34m177[0m (708.00 B)

[1m Trainable params: [0m[38;5;34m177[0m (708.00 B)

[1m Non-trainable params: [0m[38;5;34m0[0m (0.00 B)

Parámetros por red: 177 | redes en el ensamble: 10 | filas 2021-2025: 142 | parámetros por observación (una red): 1.25
                                                               Elección
Componente                                                             
Variables de entrada  9: x1_eventos, x2_fatalidades, x2b_letalidad, ...
Arquitectura                                   9 → 16 RELU → 1 sigmoide
Inicialización                         He normal (ReLU) / Glorot (tanh)
Regularización                                  L2 = 0.001; dropout = 0
Pérdida                                         BCE ponderada por clase
Optimizador                                    Adam, η = 0.001, lote 16
Presupuesto                              560 pasos de gradiente por red
Ensamble                 10 semillas (1-10), promedio de probabilidades
Calibración                                                     ninguna
Umbral operativo                                                    0.5
Escalado              Estandarización ajustada solo con el entrenami...
Configuración escrita en configuracion.json
```

#### Celda de código 23: 8.2 Esquema de la arquitectura final
Salida de la celda:
```text
[figura]
```

## 8.2 Curvas de desempeño del modelo final frente al base

Entrenamos las dos especificaciones con 2021–2024 y registramos la BCE de entrenamiento y de validación (2025) en
cada época, con 5 semillas y hasta 3 veces el presupuesto. Así se ve si los 560 pasos caen cerca del óptimo de
generalización también con las variables nuevas, y cómo cambia la brecha entre entrenamiento y validación.

#### Celda de código 24: 8.3 Curvas de aprendizaje (2021-2024 -> validación 2025), 5 semillas, 3x el presupuesto
Salida de la celda:
```text
[figura]
Unnamed: 0  BCE train (560 pasos)  BCE val (560 pasos)  Brecha val − train  BCE val mínima (media)  BCE val al final (3x)
   Base U2                  0.341                0.448               0.107                   0.439                  0.478
  Final U3                  0.279                0.377               0.097                   0.360                  0.441
```

**Lectura de las curvas.**

* **El presupuesto de 560 pasos sigue cerca del óptimo de generalización.** En el modelo final, la BCE de validación
  2025 en el paso 560 es 0,377, frente a un mínimo de 0,360. Entrenar 3 veces más la sube a 0,441, que es sobreajuste.
  La parada temprana implícita de la U2 (D10) sigue siendo válida con las variables nuevas.
* **El modelo final aprende más y generaliza mejor a la vez.** Baja la BCE de entrenamiento (0,341 → 0,279) y la de
  validación (0,448 → 0,377), y la brecha entre ambas se reduce de 0,107 a 0,097. No es un modelo más flexible que
  memoriza más: es un modelo mejor informado. Es la firma de una mejora en los datos y no en la capacidad, coherente
  con §6.11.
* Las bandas entre semillas son estrechas en ambas curvas: el entrenamiento es estable (convergencia verificada en la
  U2, §11.1).

---
# 9. Evaluación antes vs. después

## 9.1 Origen móvil 2023–2025 (selección)

"Antes" es el modelo base entregado en la U2 (MLP 7 → 16 → 1, semilla 42). "Después" es el modelo final de la U3
(ensamble de 10 redes). Las comparaciones pareadas usan **bootstrap por departamento**: se remuestrean territorios
completos con sus tres años, porque los años de un mismo departamento no son independientes.

#### Celda de código 25: 9.1 Antes vs. después en el origen móvil 2023-2025, con IC y pruebas pareadas
Salida de la celda:
```text
                     Unnamed: 0  ROC-AUC  Log-loss  Brier   ECE   MCC  Exact. balanceada  Sensibilidad C  Especificidad D  Transiciones acertadas  Transiciones
                   Persistencia    0.973     0.363  0.106 0.173 0.795              0.897           0.864            0.930                     0.0             9
Antes: MLP base U2 (semilla 42)    0.913     0.407  0.132 0.084 0.659              0.818           0.682            0.953                     2.0             9
   Logística (mismas variables)    0.949     0.306  0.097 0.068 0.737              0.863           0.773            0.953                     1.0             9
     Después: MLP U3 (ensamble)    0.956     0.304  0.095 0.124 0.691              0.840           0.750            0.930                     2.0             9
                                    IC95 ROC-AUC    IC95 Log-loss
Persistencia                     [0.942 – 0.994]  [0.317 – 0.412]
Antes: MLP base U2 (semilla 42)  [0.850 – 0.962]  [0.309 – 0.515]
Logística (mismas variables)     [0.902 – 0.980]  [0.220 – 0.407]
Después: MLP U3 (ensamble)       [0.914 – 0.986]  [0.228 – 0.391]
              Unnamed: 0  n  ΔLog-loss   IC95 ΔLog-loss  P(ΔLog-loss mejora)  ΔBrier      IC95 ΔBrier  ΔAUC        IC95 ΔAUC  solo A acierta  solo B acierta  p McNemar
Después vs. persistencia 87        -59 [-0.142; +0.028]                   92     -11 [-0.044; +0.023]   -17 [-0.054; +0.002]               2               7        180
  Después vs. antes (U2) 87       -103 [-0.185; -0.033]                  100     -36 [-0.068; -0.009]    43 [+0.005; +0.105]               3               1        625
   Después vs. logística 87         -2 [-0.060; +0.045]                   52      -2 [-0.023; +0.016]     7 [-0.023; +0.041]               1               3        625
```

**Lectura de la comparación (2023–2025).**

* **Contra el modelo de la U2, la mejora es clara y significativa.** Log-loss 0,407 → 0,304 (Δ = −0,103, IC95
  [−0,185; −0,033] remuestreando departamentos), Brier 0,132 → 0,095 (IC excluye el 0) y AUC 0,913 → 0,956 (IC
  [+0,005; +0,105]). La sensibilidad para C sube de 0,68 a 0,75 con una especificidad similar (0,95 → 0,93), porque
  la red base subestimaba C en el año de escalamiento.
* **Contra la persistencia, la mejora probabilística es probable pero no concluyente.** Δ log-loss = −0,059 con IC95
  [−0,142; +0,028]. En el 92 % de los remuestreos el modelo gana, pero con 30 departamentos el intervalo toca el 0. El
  AUC de la persistencia sigue siendo mayor (0,973 vs. 0,956): repetir el perfil ordena mejor, pero sus
  probabilidades son peores (ECE 0,173 vs. 0,124), porque asigna $s^C$ como si fuera una probabilidad. En exactitud
  la persistencia acierta 7 casos que el modelo falla y el modelo 2 que la persistencia falla (McNemar p = 0,18).
* **El MLP y la logística empatan** con las mismas variables (Δ = −0,002). Lo declaramos sin rodeos: la mejora de la U3
  se debe a la representación de los datos, y con 171 filas la red todavía no rinde más por ser no lineal.
* **La métrica pertinente es la log-loss.** El uso previsto es priorizar presupuesto con probabilidades y bandas
  (§1.3), no clasificar con un umbral. Por eso pesa más la calidad de la probabilidad que el AUC, donde la persistencia
  conserva la ventaja.

## 9.2 Generalización espacial: departamentos nunca vistos

#### Celda de código 26: 9.2 Validación espacial (5 pliegues por departamento, años <= 2025)
Salida de la celda:
```text
                             Unnamed: 0  ROC-AUC  Log-loss  Brier   MCC  Exact. balanceada   ECE
                           Persistencia    0.914     0.418  0.127 0.704              0.852 0.117
  Antes: MLP base U2 (media 5 semillas)    0.854     0.475  0.154 0.547              0.768 0.054
           Logística (mismas variables)    0.916     0.373  0.114 0.704              0.851 0.034
Después: MLP U3 (red individual, media)    0.915     0.373  0.116 0.666              0.832 0.046
             Después: MLP U3 (ensamble)    0.923     0.351  0.110 0.690              0.844 0.042
                                       n ΔLog-loss    IC95 ΔLog-loss      ΔAUC         IC95 ΔAUC
Después vs. persistencia (espacial)  142 -0.067343  [-0.135; +0.003]  0.008744  [-0.020; +0.037]
```

**Lectura.** **El cambio más grande frente a la U2 está en territorios nunca vistos.** El modelo base generalizaba mal
a departamentos nuevos (log-loss 0,475, AUC 0,854) y perdía con la persistencia. El modelo final, con ensamble, logra
**0,351 y 0,923**, y supera a la persistencia en log-loss (0,418; Δ = −0,067, IC95 [−0,135; +0,003], casi
significativo) y en AUC (0,914). Es la evidencia más relevante para la migración a municipio (M3): con información
de composición, la red aprende una relación entre el estado reciente y el perfil siguiente que **se transfiere** entre
territorios. La logística con las mismas variables queda en 0,373: el ensamble de redes es aquí la mejor opción.

## 9.3 Prueba final 2026 (una sola vez)

El modelo, las variables, la configuración, el ensamble, la calibración y el umbral ya están fijados (§8.1,
`configuracion.json`). Ahora, por única vez, se entrena con 2021–2025 y se predice 2026 (29 departamentos,
etiqueta de 33 semanas; §2.3 estimó un 95 % de coincidencia con la etiqueta anual).

#### Celda de código 27: 9.3 Prueba 2026: entrenamiento con 2021-2025, una sola evaluación
Salida de la celda:
```text
                               Unnamed: 0  ROC-AUC  Log-loss  Brier   ECE   MCC  Exactitud Sensibilidad C Especificidad D Transiciones acertadas Transiciones
                             Persistencia    0.948     0.429  0.131 0.207 0.755      0.862              1           0.733                  0.000            4
          Antes: MLP base U2 (semilla 42)    0.852     0.524  0.166 0.104 0.517      0.759          0.714           0.800                      1            4
             Logística (mismas variables)    0.929     0.383  0.129 0.114 0.531      0.759          0.857           0.667                  0.000            4
               Después: MLP U3 (ensamble)    0.900     0.380  0.121 0.074 0.672      0.828          0.929           0.733                  0.000            4
(alternativa no adoptada) MLP + embedding    0.933     0.314  0.100 0.123 0.812      0.897              1           0.800                      1            4
       (referencia) U2 fase 2 documentado    0.829     0.547  0.174     – 0.517      0.759              –               –                      –            –
                                               ROC-AUC               Log-loss
Persistencia                     0.948 [0.843 – 1.000]  0.429 [0.344 – 0.519]
Antes: MLP base U2 (semilla 42)  0.852 [0.686 – 0.971]  0.524 [0.316 – 0.806]
Logística (mismas variables)     0.929 [0.814 – 0.990]  0.383 [0.212 – 0.593]
Después: MLP U3 (ensamble)       0.900 [0.757 – 0.990]  0.380 [0.210 – 0.586]
              Unnamed: 0  n  ΔLog-loss   IC95 ΔLog-loss  ΔAUC        IC95 ΔAUC  solo A acierta  solo B acierta  p McNemar
Después vs. persistencia 29        -49 [-0.162; +0.092]   -48 [-0.171; +0.030]               0               1       1000
  Después vs. antes (U2) 29       -144 [-0.270; -0.033]    48 [-0.011; +0.124]               3               1        625
   Después vs. logística 29         -3 [-0.090; +0.100]   -29 [-0.115; +0.036]               2               0        500
Transiciones de 2026:
   departamento perfil_prev perfil    s_C   pers  antes  final
        Bolívar           C      D 310000 530000 760000 810000
          Chocó           C      D 450000 710000 960000 880000
      Magdalena           C      D 470000 560000 400000 550000
Valle del Cauca           C      D 500000 520000 650000 760000
```

#### Celda de código 28: 9.4 Matrices de confusión, ROC y desempeño por año (antes vs. después vs. persistencia)
Salida de la celda:
```text
[figura]
```

**Lectura de la prueba 2026 (única evaluación).**

* **El modelo final generaliza al año de prueba.** Log-loss **0,380** frente a 0,429 de la persistencia y 0,524 del
  modelo de la U2 (0,547 en la fase 2 documentada en la U2). Brier 0,121 vs. 0,131 y 0,166. ECE 0,074, el mejor
  calibrado de los cuatro. Frente al modelo de la U2, la mejora es significativa (Δ = −0,144, IC95 [−0,270; −0,033]).
  Frente a la persistencia no lo es: Δ = −0,049 con IC [−0,162; +0,092] en 29 departamentos.
* **La degradación desde la selección es moderada.** Log-loss 0,304 → 0,380 y AUC 0,956 → 0,900. Es coherente con la
  deriva medida (PSI de $m_1$ = 0,29, §11.3) y con un año de desescalamiento: el AUC de la persistencia también bajó
  (0,973 → 0,948).
* **Lo que falló: las 4 transiciones de 2026, todas C → D** (Bolívar, Chocó, Magdalena y Valle del Cauca). El modelo les
  dio entre 0,55 y 0,88 de probabilidad de C y no acertó ninguna. La persistencia tampoco, por construcción. Tres de
  ellas tienen $s^C$ de 2026 entre 0,45 y 0,50, en la franja que §2.3 identificó como ambigua con 33 semanas: parte de
  ese "error" puede corregirse cuando cierre el año. La sensibilidad para C es alta (0,93: solo un C perdido, Cesar)
  y la especificidad de 0,73 resume esas desescaladas.
* **El embedding descartado salió mejor en 2026, y lo reportamos tal cual.** Obtuvo log-loss 0,314, AUC 0,933 y
  MCC 0,81, y acertó 1 de las 4 transiciones. Esto debilita en parte el argumento de la regla 8: en 2023–2025 acertaba
  menos transiciones, en 2026 una más. **No cambiamos el modelo**, porque elegir ahora con 2026 convertiría la prueba en
  un conjunto de selección y su cifra dejaría de ser una estimación honesta. La lectura correcta es que, mientras el
  uso se limite a los mismos 33 departamentos, los **efectos fijos territoriales tienen valor predictivo** que las
  variables actuales no capturan. Queda como recomendación explícita (R7, §13): reevaluar el embedding cuando cierre
  2026, con el mismo protocolo, y a escala municipal reemplazarlo por las variables que expliquen esos efectos
  ($x_3$ municipal, distancia a cabecera).

---
# 10. Análisis crítico del desempeño

## 10.1 ¿Dónde se equivoca el modelo final y qué errores corrigió?

#### Celda de código 29: 10.1 Análisis de errores: 2023-2025 (origen móvil) + 2026 (prueba)
Salida de la celda:
```text
   departamento  anio perfil_prev perfil  s_C  eventos  P persist.  P antes  P después  transicion  cerca del umbral
       Casanare  2023           C      C   69       13          93       23         32       False             False
          Huila  2023           C      C   60       83          73       27         36       False             False
          Cesar  2023           C      C   52      124          51       31         38       False              True
        Bolívar  2023           D      D   35      409          39       66         53       False             False
Valle del Cauca  2023           C      D   44      285          61       86         84        True              True
       Casanare  2024           C      C   81       31          69        9         41       False             False
          Huila  2024           C      C   64       76          60       44         47       False             False
Valle del Cauca  2024           D      D   44      288          44       48         51       False              True
        Córdoba  2025           D      C   55      131          48       19         21        True              True
      Magdalena  2025           D      C   56      131          40       22         21        True              True
          Cesar  2025           D      C   53      137          42       26         25        True              True
           Meta  2025           D      C   67       83          42       12         25        True             False
        Bolívar  2025           D      C   53      332          36       45         31        True              True
Valle del Cauca  2025           D      C   52      394          44       37         48        True              True
          Cesar  2026           C      C   67       87          53       34         45       False             False
      Magdalena  2026           C      D   47      112          56       40         55        True              True
Valle del Cauca  2026           C      D   50      271          52       65         76        True              True
        Bolívar  2026           C      D   31      173          53       76         81        True             False
          Chocó  2026           C      D   45      148          71       96         88        True              True
           Unnamed: 0  Errores después  Aciertos después
       % transiciones             0.58              0.02
% a ±10 pp del umbral             0.58              0.19
   Mediana de eventos           131.00             92.00
                Filas            19.00             97.00
acierto después  después acierta  después falla
acierto antes                                  
antes acierta                 91              2
antes falla                    6             17
Transiciones 2023-2026: 13 | acertadas por la persistencia: 0 | antes: 3 | después: 2
                P(C) después                   Perfil observado               
anio                    2023  2024  2025  2026             2023 2024 2025 2026
departamento                                                                  
Arauca                  0.84  0.77  0.85  0.79                C    C    C    C
Bolívar                 0.53  0.35  0.31  0.81                D    D    C    D
Casanare                0.32  0.41  0.64  0.50                C    C    C    C
Cesar                   0.38  0.46  0.25  0.45                C    D    C    C
Chocó                   0.85  0.85  0.92  0.88                C    C    C    D
Meta                    0.78  0.41  0.25  0.76                C    D    C    C
Valle del Cauca         0.84  0.51  0.48  0.76                D    D    C    D
[figura]
```

**Lectura del análisis de errores.**

* **Los errores se concentran donde el diagnóstico de la U2 lo predijo.** De los 19 errores en 2023–2026, el 58 % son
  transiciones de perfil (frente al 2 % entre los aciertos) y el 58 % tiene $s^C$ a ±10 pp del umbral (frente al 19 %).
  El modelo falla donde la etiqueta es ambigua o donde el territorio cambia de régimen.
* **Errores corregidos y errores nuevos.** Frente al modelo de la U2, 6 casos pasan de error a acierto y 2 de acierto a
  error; 17 fallan en ambos. Los fallos compartidos son casi todos transiciones.
* **Arauca, el error estructural de la U2, quedó corregido.** P(C) = 0,84, 0,77, 0,85 y 0,79 en 2023–2026, todos C. La
  persistencia del frente ELN entra por $m_1$ y $d_1$ (§6.3).
* **Persisten tres patrones:**
  1. **Escaladas D → C de 2025** (Córdoba, Magdalena, Cesar, Meta, Bolívar y Valle): un cambio de coyuntura nacional,
     la ruptura de los diálogos, que la información de 2024 no contenía.
  2. **Desescaladas C → D de 2026** (§9.3).
  3. **Casanare**, con pocos eventos (13 a 31) y P(C) baja todos los años, aunque es C.

  Los dos primeros requieren información que no está en ACLED agregado ni en el conteo anual de actores de UCDP
  (I3 no ayudó): señales políticas como ceses al fuego o negociaciones. El tercero es ruido de pocos datos.
* **Transiciones.** De 13 transiciones en 2023–2026, el modelo final acierta 2, el de la U2 3 y la persistencia 0. La
  U3 mejoró la probabilidad en los territorios estables sin ganar capacidad de anticipar cambios. Lo declaramos como la
  limitación principal.

## 10.2 ¿Qué aprendió la red? Importancia por permutación e interacción memoria × estructura

#### Celda de código 30: 10.2 Importancia por permutación fuera de muestra (por pliegue) y dependencia parcial 2D
Salida de la celda:
```text
[figura]
             Unnamed: 0  ΔLog-loss media  Desv. est.
d₁ s_C 2.º semestre t-1           0.1145      0.0453
          x₅ bosque (%)           0.0485      0.0353
          m₁ s_C en t-1           0.0484      0.0287
   x₂ fatalidades (log)           0.0455      0.0351
             x₇ IPM (%)           0.0232      0.0196
          x₂b letalidad           0.0232      0.0156
       x₄ coca (log ha)           0.0209      0.0203
      x₆ distancia (km)           0.0074      0.0211
       x₁ eventos (log)           0.0037      0.0133
Interacción mostrada en (b)-(c): memoria m1 × x₂b letalidad
```

**Lectura.**

* **$d_1$ es, con diferencia, la variable más importante fuera de muestra** (permutarla sube la log-loss en 0,115), más
  del doble que cualquier otra. Le siguen el bosque, $m_1$ y las fatalidades (≈ 0,05 cada una). La distancia y el
  volumen de eventos quedan casi en cero. Respecto de la U2 cambió la jerarquía: las variables estructurales de la U1
  no desaparecen (bosque, fatalidades, IPM y coca siguen aportando), pero pasan a **modular** la inercia en lugar de
  sustituirla.
* **La interacción que la red aprende es coherente con la U1.** En (b), la frontera $P = 0{,}5$ entre memoria y
  letalidad es **curva**: con una letalidad de 0,6 fatalidades por evento basta una memoria de ≈ 0,45 para esperar
  C, y con letalidad nula hace falta ≈ 0,7. La violencia letal del año anterior "adelanta" el perfil convencional aunque la composición todavía no
  lo muestre. Es la intuición de la U1 de que $x_2$ "distingue letalidad por arma". La logística (c) solo puede trazar
  una recta, y con pendiente contraria por la colinealidad entre memoria y letalidad (ρ = 0,84).
* **Lectura honesta.** Como la red y la logística empatan en desempeño (§9.1), esta interacción es plausible y
  coherente, pero no está demostrada como necesaria. Con más unidades (municipios) será posible contrastarla.

---
# 11. Integración final: pipeline reproducible

El cuaderno **usa** el pipeline, pero no **es** el pipeline. Todo el proceso vive en el paquete `src/perfil_amenaza/`,
se ejecuta con una sola orden y produce un **artefacto único** de despliegue que no depende del cuaderno.

```
Proyecto_Unidad_3/
├── configuracion.json            ← escrita en §8.1: variables, red, ensamble, calibración, protocolo y decisiones
├── requirements.txt · README.md · pytest.ini
├── datos/acled-data-export.csv · datos/externos/ (coca, bosque y, si se agrega, UCDP)
├── src/perfil_amenaza/
│   ├── config.py      rutas, constantes, mapeo de subtipos, IPM, contigüidad departamental
│   ├── datos.py       contrato y auditoría, panel, bloques trimestrales, fuentes externas, UCDP, control 2026
│   ├── variables.py   predictoras en t-1 (BASE, MEMORIA, DINAMICA, VECINDAD, ACTORES) y secuencias
│   ├── modelo.py      redes Keras (MLP, multitarea, GRU, CNN, embedding), bucle GradientTape, calibración, artefacto
│   ├── evaluacion.py  métricas, ECE, bootstrap por departamento, McNemar, origen móvil, espacial, PSI
│   ├── pipeline.py    orquestación de 8 etapas y línea de comandos
│   └── graficos.py    estilo de figuras
├── tests/test_pipeline.py        contrato de datos, fuga temporal, escalador, determinismo, reproducción U2
├── Proyecto_U3_Integrador.ipynb  este cuaderno
└── salidas_u3/                   métricas, predicciones, figuras, modelos/, TARJETA_MODELO.md
```

#### Celda de código 31: 11.1 Diagrama del pipeline integrado
Salida de la celda:
```text
[figura]
```

#### Celda de código 32: 11.2 Pruebas automáticas y ejecución de extremo a extremo por línea de comandos
Salida de la celda:
```text
$ python -m pytest -q tests
5 passed in 7.02s
(9 s)
$ python -m perfil_amenaza.pipeline todo
[1-2] datos: 171 filas, 30 departamentos, 9 predictoras
[3-4] origen móvil 2023-2025 (n=87): AUC 0.956, log-loss 0.304 (persistencia 0.363; modelo base U2 0.407); calibración: ninguna
[5] prueba 2026 (n=29): AUC 0.900, log-loss 0.380 (persistencia 0.429)
[6] artefacto: perfil_amenaza_u3.keras (152 KB, 10 redes)
[7] proyección 2027: {'C consolidado': 11, 'Transición': 11, 'D consolidado': 11} | variables en deriva (PSI>0,25): 2
[8] tarjeta del modelo escrita. Tiempo total: 29 s
```

#### Celda de código 33: 11.3 El artefacto único: carga, estructura y consistencia con lo evaluado
Salida de la celda:
```text
perfil_amenaza_u3.keras: 152 KB | entrada: (None, 9) (variables crudas: x1_eventos, x2_fatalidades, x2b_letalidad, x4_coca, x5_bosque, x6_distancia, x7_ipm, m1_sC_t1, d1_sC_H2) | salidas: [(None, 1)] | calibración: ninguna
             Capa             Tipo  Parámetros entrenables
      x_crudo_t_1       InputLayer                       0
  estandarizacion    Normalization                       0
        miembro_1       Functional                     177
        miembro_2       Functional                     177
        miembro_3       Functional                     177
        miembro_4       Functional                     177
        miembro_5       Functional                     177
        miembro_6       Functional                     177
        miembro_7       Functional                     177
        miembro_8       Functional                     177
        miembro_9       Functional                     177
       miembro_10       Functional                     177
promedio_ensamble          Average                       0
      calibracion CalibracionPlatt                       0
                     Unnamed: 0  Cuaderno (§9)  Pipeline CLI  Diferencia
Log-loss origen móvil 2023-2025         0.3043        0.3043         0.0
           Log-loss prueba 2026         0.3800        0.3800         0.0
Deriva de las predictoras de la proyección (información 2026) frente al entrenamiento 2021-2026:
     Unnamed: 0_level_0                PSI             estado
               variable Unnamed: 1_level_1 Unnamed: 2_level_1
       x₁ eventos (log)              0.103            vigilar
   x₂ fatalidades (log)              0.023            estable
          x₂b letalidad              0.251             deriva
       x₄ coca (log ha)              0.005            estable
          x₅ bosque (%)              0.022            estable
      x₆ distancia (km)              0.027            estable
             x₇ IPM (%)              0.030            estable
          m₁ s_C en t-1              0.292             deriva
d₁ s_C 2.º semestre t-1              0.238            vigilar
# Tarjeta del modelo: perfil de amenaza tecnológica (U3)

**Uso previsto.** Priorizar, con un año de anticipación, el énfasis tecnológico (ISR, C-UAS, detección de AEI)
por departamento. Insumo analítico; no sustituye la inteligencia de campo.

**No usar para:** decisiones sobre personas o municipios individuales; departamentos con menos de 10 eventos
al año (Amazonía y Orinoquía con pocos registros) sin revisión experta.

**Datos.** ACLED semanal agregado a departamento (2020-2026), SIMCI-ODC, Global Forest Watch, DANE. 171 filas.
**Modelo.** mlp (16,) relu L2=0.001 do=0 lr=0.001 pasos=560 obj=duro; ensamble de 10 semillas; sin recalibración (la calibración Platt no mejoró la log-loss fuera de muestra; ver §6.10 del cuaderno).
**Variables (9).** x1_eventos, x2_fatalidades, x2b_letalidad, x4_coca, x5_bosque, x6_distancia, x7_ipm, m1_sC_t1, d1_sC_H2

| Métrica | Origen móvil 2023-2025 | Prueba 2026 |
|---|---|---|
| ROC-AUC | 0.956 | 0.900 |
| Log-loss | 0.304 | 0.380 |
| Brier | 0.095 | 0.121 |
| MCC | 0.691 | 0.672 |
| ECE | 0.124 | 0.074 |

| Rival (persistencia) | Origen móvil 2023-2025 | Prueba 2026 |
|---|---|---|
| Persistencia · ROC-AUC | 0.973 | 0.948 |
| Persistencia · Log-loss | 0.363 | 0.429 |
| Persistencia · Brier | 0.106 | 0.131 |

**Limitaciones.** 30 unidades espaciales; etiqueta 2026 con 33 semanas; x3 (actores) solo si hay UCDP; x6 es un
proxy departamental; la prevalencia de C cambia entre años (32 % a 62 %), por lo que la calibración debe
re-estimarse cada año.

**Deriva.** Variables con PSI > 0,25 en 2026 frente a 2021-2026: x2b_letalidad, m1_sC_t1.
**Mantenimiento.** Reentrenar cada año con la etiqueta cerrada, recalcular la calibración y revisar el PSI.
```

**Lectura de la integración.**

* **Las 5 pruebas pasan**: contrato de datos, ausencia de fuga temporal (multiplicar por 5 los eventos de 2024 no cambia
  ninguna predictora de las filas de 2024, y sí las de 2025), escalador ajustado solo con años anteriores,
  determinismo y reproducción exacta del dataset de la U2.
* **La orden `pipeline todo` reproduce de extremo a extremo** lo que el cuaderno evaluó, en unos 30 segundos. Las
  log-loss de selección (0,304) y de prueba (0,380) coinciden con §9 **al cuarto decimal**: el cuaderno y el pipeline
  son el mismo cálculo.
* **El artefacto es único y autocontenido.** Un archivo `.keras` de 152 KB con la estandarización (capa
  `Normalization` con las medias y desviaciones de entrenamiento), las 10 redes de 177 parámetros, el promedio y la
  calibración. Recibe las 9 variables en su escala original y devuelve P(C). La proyección de §12 se calcula con este
  archivo, no con objetos en memoria, y `pipeline predecir` funciona sin el cuaderno.
* **Deriva (PSI).** La información de 2026 que alimenta la proyección 2027 muestra deriva en $m_1$ (0,29) y en la
  letalidad (0,25), y conviene vigilar $d_1$ (0,24) y los eventos (0,10). 2026 es un año de recomposición (desescaladas
  C → D y 33 semanas), así que la proyección 2027 debe leerse con más cautela que la de un año estable. La tarjeta del
  modelo lo registra y fija el mantenimiento: reentrenar con cada año cerrado y revisar el PSI antes de usar la
  proyección.

---
# 12. Del modelo a la decisión: priorización territorial 2027

La proyección sale del **artefacto único** (no de objetos en memoria): ensamble de 10 redes reentrenado con todas
las etiquetas 2021–2026 y aplicado a la información de 2026 (33 semanas, anualizada). Se compara con la proyección
de la U2 para ver qué cambió al incorporar la memoria y la dinámica reciente.

> Es un **insumo analítico ilustrativo**, no una recomendación operativa: resume el patrón aprendido de datos públicos
> agregados y debe contrastarse con inteligencia de campo.

#### Celda de código 34: 12.1 Proyección 2027 con el artefacto, incertidumbre entre redes y cambio frente a la U2
Salida de la celda:
```text
      departamento Perfil 2026  s_C 2026  P(C) 2027  DE entre redes Categoría 2027  Categoría U2 ¿Coincide con el perfil 2026? s_AB observada 2025-26                                                       Énfasis sugerido  Datos suficientes
          Guaviare           C        76         99               7  C consolidado C consolidado                            sí                  14.4% ISR, detección temprana y movilidad protegida + evaluar C-UAS / det...               True
            Nariño           C        70         93              22  C consolidado C consolidado                            sí                   6.8%                          ISR, detección temprana y movilidad protegida               True
             Cauca           C        70         92              21  C consolidado C consolidado                            sí                  13.4% ISR, detección temprana y movilidad protegida + evaluar C-UAS / det...               True
           Caquetá           C        81         92              34  C consolidado C consolidado                            sí                   9.8%                          ISR, detección temprana y movilidad protegida               True
          Putumayo           C        81         89              80  C consolidado    Transición                            sí                   3.7%                          ISR, detección temprana y movilidad protegida               True
         Antioquia           C        65         88              23  C consolidado C consolidado                            sí                   6.8%                          ISR, detección temprana y movilidad protegida               True
          Amazonas           C        78         88              41  C consolidado C consolidado                            sí                   0.0%                          ISR, detección temprana y movilidad protegida              False
Norte de Santander           C        60         86              26  C consolidado C consolidado                            sí                  10.2%                          ISR, detección temprana y movilidad protegida               True
            Vaupés           C        50         84             131  C consolidado C consolidado                            sí                   0.0%                          ISR, detección temprana y movilidad protegida              False
           Vichada           C        73         79              87  C consolidado    Transición                            sí                   5.9%                          ISR, detección temprana y movilidad protegida               True
             Cesar           C        67         71              69  C consolidado    Transición                            sí                   7.1%                          ISR, detección temprana y movilidad protegida               True
        San Andrés           C       100         67             289     Transición    Transición                            sí                   0.0%                  Portafolio flexible y monitoreo trimestral del perfil              False
             Chocó           D        45         62             161     Transición C consolidado                       revisar                   8.0%                  Portafolio flexible y monitoreo trimestral del perfil               True
   Valle del Cauca           D        50         61              87     Transición    Transición                       revisar                   7.1%                  Portafolio flexible y monitoreo trimestral del perfil               True
            Arauca           C        65         61              81     Transición    Transición                            sí                   9.8%                  Portafolio flexible y monitoreo trimestral del perfil               True
           Córdoba           C        53         57             124     Transición    Transición                            sí                   1.2%                  Portafolio flexible y monitoreo trimestral del perfil               True
           Guainía           C        50         55             237     Transición C consolidado                            sí                   0.0%                  Portafolio flexible y monitoreo trimestral del perfil              False
          Casanare           C        58         52              99     Transición    Transición                            sí                   8.7%                  Portafolio flexible y monitoreo trimestral del perfil               True
             Huila           C        62         51              60     Transición    Transición                            sí                   7.5%                  Portafolio flexible y monitoreo trimestral del perfil               True
              Meta           C        55         50              90     Transición D consolidado                       revisar                   1.4%                  Portafolio flexible y monitoreo trimestral del perfil               True
             Sucre           D        48         43              42     Transición    Transición                            sí                   0.0%                  Portafolio flexible y monitoreo trimestral del perfil               True
         Magdalena           D        47         34              70     Transición    Transición                            sí                   0.8%                  Portafolio flexible y monitoreo trimestral del perfil               True
        La Guajira           D        45         28              90  D consolidado    Transición                            sí                   2.9%                                     Orden público y capacidades mixtas               True
           Bolívar           D        31         23              59  D consolidado    Transición                            sí                   5.0%                                     Orden público y capacidades mixtas               True
            Tolima           D        43         19              68  D consolidado D consolidado                            sí                   1.8%                                     Orden público y capacidades mixtas               True
         Atlántico           D        24         10              49  D consolidado D consolidado                            sí                   0.4%                                     Orden público y capacidades mixtas               True
         Risaralda           D        29          7              26  D consolidado D consolidado                            sí                   2.0%                                     Orden público y capacidades mixtas               True
            Caldas           D        32          7              28  D consolidado D consolidado                            sí                   0.9%                                     Orden público y capacidades mixtas               True
           Quindío           D        25          6              29  D consolidado D consolidado                            sí                   0.0%                                     Orden público y capacidades mixtas               True
         Santander           D        16          4              13  D consolidado D consolidado                            sí                   1.7%                                     Orden público y capacidades mixtas               True
      Cundinamarca           D        29          4              33  D consolidado D consolidado                            sí                   0.0%                                     Orden público y capacidades mixtas               True
            Boyacá           D        15          2              12  D consolidado D consolidado                            sí                   1.4%                                     Orden público y capacidades mixtas               True
       Bogotá D.C.           D         5          1               5  D consolidado D consolidado                            sí                   0.1%                                     Orden público y capacidades mixtas               True
Categorías 2027 (U3): {'C consolidado': 11, 'Transición': 11, 'D consolidado': 11} | U2: {'Transición': 13, 'C consolidado': 10, 'D consolidado': 10}
U3             C consolidado  D consolidado  Transición
U2                                                     
C consolidado              8              0           2
D consolidado              0              9           1
Transición                 3              2           8
Discrepancias con el perfil observado en 2026: 3 (Chocó, Valle del Cauca, Meta) | alerta A/B: Guaviare, Cauca
```

#### Celda de código 35: 12.2 Mapa esquemático de la proyección 2027 (centroides) y transición U2 -> U3
Salida de la celda:
```text
[figura]
```

**Lectura para la decisión.**

* **11 departamentos C consolidados** (P ≥ 0,7): Guaviare, Nariño, Cauca, Caquetá, Putumayo, Antioquia, Norte de
  Santander, Vichada y Cesar, más Amazonas y Vaupés, que tienen pocos datos (en gris en el mapa). Prioridad para ISR,
  detección temprana y movilidad protegida. **Guaviare (14,4 % de eventos A+B) y Cauca (13,4 %)** mantienen la alerta
  A/B: allí conviene sumar C-UAS y detección de AEI.
* **11 en transición** (0,3–0,7): San Andrés (pocos datos), Chocó, Valle del Cauca, Arauca, Córdoba, Guainía,
  Casanare, Huila, Meta, Sucre y Magdalena. Portafolio flexible y monitoreo trimestral. Meta (0,50), Huila (0,51) y
  Casanare (0,52) están justo en el umbral.
* **11 D consolidados**: el eje andino y urbano, más La Guajira, Bolívar y Atlántico. Orden público y capacidades mixtas.
* **Qué cambió frente a la U2.** 25 de 33 departamentos conservan su categoría. Putumayo, Vichada y Cesar pasan de
  transición a C consolidado, porque la memoria pesa ahora más que la estructura. Chocó y Guainía bajan de C a
  transición; Bolívar y La Guajira de transición a D. Las **discrepancias con el perfil observado en 2026 bajan de 10 a
  3** (Chocó, Valle del Cauca y Meta). Chocó y Valle son desescaladas recientes con $s^C$ de 2026 en 0,45–0,50, la
  franja ambigua. El modelo final integra la inercia y solo se aparta de ella donde la evidencia es débil.
* **Incertidumbre explícita.** La desviación entre las 10 redes es menor de 0,04 en los extremos (C y D
  consolidados) y crece en la franja de transición: Chocó 0,16, Córdoba 0,12 y los de pocos registros, San Andrés
  0,29 y Guainía 0,24. Donde el modelo duda, las redes también discrepan. La decisión debe leer juntos la
  probabilidad, la banda y esa dispersión.

---
# 13. Conclusiones técnicas y recomendaciones

1. **Coherencia problema–datos–modelo–decisión.** El modelo final responde la misma pregunta de la U1, con la misma
   etiqueta, el mismo diseño $t-1 \rightarrow t$ y las mismas variables estructurales. Agrega dos variables que
   operacionalizan la "intensidad histórica" de la U1 como composición reciente. Cada cambio frente a la U2 se
   justifica en un hallazgo del diagnóstico y se valida con un protocolo fijado antes de ver los resultados.
2. **Optimizar fue intervenir los datos, no la arquitectura.** De diez frentes evaluados se retuvieron tres ($m_1$,
   $d_1$ y el ensamble). La log-loss bajó de 0,407 a 0,304 en selección y de 0,524 a 0,380 en la prueba 2026. La
   capacidad, la activación, la profundidad, las arquitecturas secuenciales, la salida multitarea y la calibración no
   superaron el ruido. La validación anidada mostró que elegir hiperparámetros con 58 filas empeora el año siguiente
   (0,359 → 0,411–0,423).
3. **El modelo supera a la inercia en calidad probabilística, no en ordenamiento.** Gana a la persistencia en log-loss y
   Brier en selección, en territorios nuevos y en 2026, pero los intervalos tocan el cero y la persistencia conserva el
   mayor AUC. Es la conclusión honesta con 30 departamentos.
4. **La generalización espacial, debilidad central de la U2, quedó resuelta.** En departamentos nunca vistos la
   log-loss pasó de 0,475 a 0,351 (mejor que la persistencia, 0,418). Es el argumento cuantitativo más fuerte a favor de
   escalar a municipio con la misma arquitectura.
5. **La limitación principal sigue siendo anticipar cambios de régimen.** El modelo acierta 2 de 13 transiciones. Las
   escaladas de 2025 y las desescaladas de 2026 responden a coyunturas que las variables de ACLED agregado no contienen.
6. **Los efectos fijos territoriales importan.** El embedding de departamento, descartado por no transferir a
   territorios nuevos, fue mejor en la prueba 2026. Las variables actuales no explican todo lo que distingue a un
   departamento de otro. Es la pista para R2 y R7.
7. **La red no demostró ser mejor que una logística con las mismas variables.** La no linealidad aparece en la forma
   (memoria × letalidad), pero no en el desempeño. Se mantiene el MLP por coherencia con la U1 y porque es la
   arquitectura que debe escalar; la logística queda como control permanente.

**Recomendaciones (en orden de impacto esperado):**

| # | Recomendación | Evidencia | Acción concreta |
|---|---|---|---|
| R1 | Migrar a municipio con el registro de ACLED por evento (M3) | Generalización espacial resuelta (§9.2); varianza por pocas unidades (U2) | Solicitar el acceso académico; reentrenar con el mismo pipeline (cambia solo `datos.py`) |
| R2 | Redefinir $x_3$ a escala municipal | El conteo anual de actores por departamento empeoró el modelo (§6.3), pero los territorios sin historia necesitan una variable de actores | Con el registro por evento, medir actores por municipio y su cambio reciente (entradas y salidas de grupos), no el conteo anual |
| R3 | Señales adelantadas de transición | 11 de 19 errores son transiciones; escaladas 2025 y desescaladas 2026 | Variables mensuales del último trimestre, ceses al fuego y negociaciones; evaluar con la misma regla |
| R4 | Reentrenar cada año con la etiqueta cerrada y vigilar el PSI | PSI de $m_1$ = 0,29 en 2026; prevalencia de C entre 41 % y 62 % | Ejecutar `pipeline todo` al cierre de cada año; no usar umbrales fijos; decidir con las bandas |
| R5 | Re-evaluar los logaritmos y `MIN_EVENTOS` a escala municipal | Controles de §7 (Δ −0,012 sin logaritmos; 5 ≈ 10 eventos) | Repetir la batería de controles con el nuevo dataset |
| R6 | Mantener la persistencia y la logística como controles permanentes | Empates y ventajas estrechas (§9) | Reportarlas siempre junto al modelo en la tarjeta |
| R7 | Reevaluar el embedding de departamento con 2026 cerrado | Cumplió la regla 5 en 2023–2025 y fue mejor en la prueba 2026 (log-loss 0,314 vs. 0,380), pero no transfiere a territorios nuevos | Incluirlo como candidato en la próxima selección (origen móvil 2024–2026) y, a escala municipal, sustituirlo por $x_3$ y distancia municipal |

---
# 14. Trabajo en equipo, bitácora y uso de IA

## 14.1 Roles técnicos en la Unidad 3

La guía pide distribuir cuatro roles: modelado, evaluación, documentación y presentación. Mantenemos la división
de las Unidades 1 y 2 (Juan Diego en la implementación técnica, Laura en la interpretación estratégica y la
comunicación) y la llevamos a esos cuatro roles.

| Rol (guía U3) | Responsable | Entregables en este repositorio | Respaldo |
|---|---|---|---|
| **Modelado** | Juan Diego | `src/perfil_amenaza/` (datos, variables, modelo, pipeline); intervenciones I1–I9 (§6); rejilla y validación anidada (§6.8) | Laura |
| **Evaluación** | Conjunta: cálculo Juan Diego, interpretación Laura | Protocolo y regla de retención (§3), controles (§7), antes vs. después (§9), análisis de errores (§10) | – |
| **Documentación** | Laura | Lecturas del cuaderno, tarjeta del modelo, informe PDF (12 páginas), README | Juan Diego |
| **Presentación** | Laura (guion y narrativa) · Juan Diego (demostración del pipeline) | Video MP4 de 8–10 min y sustentación sincrónica | Conjunta |

**Articulación.** Las decisiones de diseño se toman en conjunto y quedan en la bitácora (§14.2). Ambos integrantes
deben poder reproducir el proceso con las tres órdenes de §11 y defender cada decisión con la evidencia citada.

## 14.2 Bitácora de decisiones técnicas (continúa la D1–D10 de la U2)

| # | Decisión | Alternativas consideradas | Justificación | Evidencia |
|---|---|---|---|---|
| D11 | Seleccionar solo con origen móvil 2023–2025; 2026 se evalúa una vez | Seguir usando 2026 en la selección | 2026 ya se miró en la U2; reusarlo para elegir sesgaría la prueba | §3.1 |
| D12 | La persistencia es el rival y la log-loss la métrica de decisión | Exactitud; comparar solo contra el MLP base | La persistencia es lo que haría un planificador sin modelo; la log-loss es una regla de puntuación propia (Gneiting y Raftery, 2007) | §3.1, §4 |
| D13 | Regla de retención: mejora > 1 EE pareado entre semillas | Aceptar cualquier mejora media | Con 87 predicciones, diferencias menores que el ruido entre semillas no se sostienen | §3.1, §6 |
| D14 | Proporciones con contracción bayesiana (α = 4) | Proporción cruda | Estabiliza celdas y semestres con pocos eventos; se verifica en §7 | §2.4, §7 |
| D15 | Variables finales según la cascada I1–I4 | Usar todas las candidatas | Solo entran las que superan la regla; las demás se documentan como descartadas | §6.1–§6.4 |
| D16 | Mantener capacidad, L2 y presupuesto de la U2 | Adoptar el mínimo de la rejilla | La validación anidada muestra que la selección no transfiere al año siguiente | §6.8 |
| D17 | Ensamble de 10 redes | 1 o 20 redes | La curva de K se estabiliza; reduce la dispersión entre semillas | §6.9 |
| D18 | Calibración solo si mejora sin mirar el futuro | Platt ajustado con 2023–2025 completo | El sesgo de calibración cambia de signo y tamaño entre años (prevalencia) | §6.10 |
| D19 | Pipeline como paquete con artefacto único `.keras` (estandarización + ensamble + calibración) | Cuaderno como único entregable | Reproducibilidad, pruebas automáticas y predicción sin el cuaderno | §11 |
| D20 | $x_3$ con UCDP GED (reducido a Colombia con `herramientas/preparar_ucdp.py`); no se retiene | Esperar el registro por evento de ACLED (licencia) | Fuente abierta verificada; se evaluó con la misma regla y empeoró la log-loss | §6.3 |

## 14.3 Declaración de uso de inteligencia artificial

Nos apoyamos en herramientas de inteligencia artificial (Claude, de Anthropic) para la **programación**: estructurar
el código del pipeline en módulos, depurarlo y organizar las celdas de este cuaderno. La propuesta del proyecto, la
formulación del problema, la búsqueda y selección de las fuentes de datos, el análisis de los datos, las decisiones
metodológicas, la interpretación de los resultados y las conclusiones son trabajo del equipo. Ambos integrantes
ejecutamos el cuaderno de principio a fin, verificamos las cifras citadas contra las salidas y estamos en capacidad de
explicar y defender cada decisión, así como de reproducir el proceso con las órdenes de §11.

---
# 15. Matriz rúbrica → evidencia

| Criterio de la rúbrica (Excelente = 1,0) | Qué pide el nivel Excelente | Evidencia en este cuaderno |
|---|---|---|
| **Coherencia metodológica integral** (problema–datos–modelo–evaluación) | Integra problema, datos, modelo y conclusiones con trazabilidad clara y evidencia cuantitativa | Trazabilidad U1 → U2 → U3 de variables y decisiones (§1.1–§1.3); reproducción de la U2 al dígito (§2.4, §4); cada intervención nace de un diagnóstico U2 con hipótesis refutable (§5); la salida vuelve a la pregunta de inversión (§12); conclusiones con cifras (§13) |
| **Preparación y representación de datos** | Evalúa y justifica transformaciones, partición y controles experimentales, con impacto en el entrenamiento | Contrato y auditoría (§2.1); control de la etiqueta parcial de 2026, 94,9 % (§2.3); variables nuevas con justificación y contracción bayesiana (§2.4–§2.5); protocolo temporal sin fuga (§3); 9 controles con efecto medido y curvas de entrenamiento: sin estandarización no converge, fuga del escalador = +0,002 artificial, etiquetas barajadas AUC 0,32 (§7) |
| **Calidad técnica del modelo y optimización** | Diseña y optimiza la arquitectura con configuración documentada, curvas de desempeño e intervención estratégica basada en diagnóstico | 10 frentes de intervención con regla de retención previa (§3.1, §6); comparativos GRU, CNN, embedding y profundo (§6.7); rejilla de 48 configuraciones con validación anidada (§6.8); curva del ensamble (§6.9); configuración en `configuracion.json` (§8.1); curvas de aprendizaje antes vs. después (§8.3); escalera de mejora (§6.11) |
| **Evaluación del desempeño e interpretación** | Evalúa métricas pertinentes interpretando resultados, errores y comparación experimental con rigor técnico | Log-loss, Brier, ECE, AUC, MCC y transiciones (§3.1); IC bootstrap por departamento y McNemar (§9.1); validación espacial (§9.2); prueba 2026 única (§9.3); matrices de confusión y ROC (§9.4); análisis de errores, importancia e interacción (§10); límites declarados (§9.1, §10.1, §13) |
| **Entregables finales y comunicación ejecutiva** | Integra informe y sustentación con estructura profesional, coherencia técnica y articulación del equipo | Resumen ejecutivo; pipeline con 5 pruebas, línea de comandos y artefacto único (§11); tarjeta del modelo; decisión territorial 2027 (§12); roles, bitácora D11–D20 y declaración de IA (§14); figuras en tamaño de impresión para el informe (Anexo A) |

**Contenidos del Producto 1 (informe de 12 páginas) → secciones de origen:** planteamiento (§1), dataset y
preprocesamiento (§2, §7), arquitectura (§6.7, §8), entrenamiento y evaluación (§3, §8.3, §9), optimización y
comparación (§5, §6, §9), análisis crítico (§9, §10), integración en el pipeline (§11), conclusiones y recomendaciones
(§13). **Producto 2 (video de 8–10 min):** síntesis del problema (§1), decisiones técnicas (§6.11, D11–D20), resultados
(§9, figuras R1–R2), conclusiones estratégicas (§12, §13).

---
# 16. Artefactos y figuras para el informe

#### Celda de código 36: 16.1 Artefactos producidos (cuaderno + pipeline)
Salida de la celda:
```text
   tiempo total del cuaderno (min)
0                             14.3
                               Archivo        KB
                     TARJETA_MODELO.md   1700000
        antes_despues_origen_movil.csv   1100000
         antes_despues_prueba_2026.csv   1500000
                   controles_datos.csv    900000
               dataset_modelado_u3.csv  69600000
         decisiones_intervenciones.csv   2200000
                   deriva_psi_2026.csv    400000
                   escalera_mejora.csv    400000
    figuras/fig01_etiqueta_parcial.png 133100000
figuras/fig02_eda_variables_nuevas.png 372000000
           figuras/fig03_protocolo.png 118800000
     figuras/fig04_rejilla_anidada.png 217400000
            figuras/fig05_ensamble.png 185600000
  figuras/fig06_calibracion_umbral.png 261700000
     figuras/fig07_escalera_mejora.png 134600000
     figuras/fig08_controles_datos.png 187700000
  figuras/fig09_arquitectura_final.png 487300000
  figuras/fig10_curvas_aprendizaje.png 202900000
       figuras/fig11_antes_despues.png 305600000
             figuras/fig12_errores.png 209800000
     figuras/fig13_importancia_pdp.png 256700000
            figuras/fig14_pipeline.png 237700000
     figuras/fig15_proyeccion_2027.png 446500000
              metricas_prueba_2026.csv    800000
                metricas_seleccion.csv    800000
        modelos/perfil_amenaza_u3.json   1100000
       modelos/perfil_amenaza_u3.keras 152000000
         predicciones_origen_movil.csv   9100000
          predicciones_prueba_2026.csv   2900000
                proyeccion_2027_u3.csv  14500000
                        rejilla_I6.csv  20600000
               validacion_espacial.csv   1200000
```

## Anexo A. Figuras del informe en tamaño de impresión

Las figuras del cuaderno están pensadas para pantalla. Esta celda redibuja las que usará el informe (máximo 12
páginas) con el ancho útil de una página carta (6,6 pulgadas), a 300 dpi, **a partir de los resultados ya
calculados**, sin reentrenar nada. Se guardan en `salidas_u3/figuras_informe/`.

#### Celda de código 37: Anexo A. Figuras R1-R5 para el informe (carta, 6,6" de ancho, 300 dpi)
Salida de la celda:
```text
Figuras del informe: ['R1_escalera.png', 'R2_antes_despues_anual.png', 'R3_controles.png', 'R4_anidada.png', 'R5_proyeccion.png']
Figuras del cuaderno (20): fig01_etiqueta_parcial.png, fig02_eda_variables_nuevas.png, fig03_protocolo.png, fig04_rejilla_anidada.png, fig05_ensamble.png, fig06_calibracion_umbral.png, fig07_escalera_mejora.png, fig08_controles_datos.png, fig09_arquitectura_final.png, fig10_curvas_aprendizaje.png, fig11_antes_despues.png, fig12_errores.png, fig13_importancia_pdp.png, fig14_pipeline.png, fig15_proyeccion_2027.png, R1_escalera.png, R2_antes_despues_anual.png, R3_controles.png, R4_anidada.png, R5_proyeccion.png
```

---
# Referencias

ACLED. (2024). *ACLED codebook*. Armed Conflict Location & Event Data Project. https://acleddata.com

Bergmeir, C., & Benítez, J. M. (2012). On the use of cross-validation for time series predictor evaluation. *Information Sciences, 191*, 192–213. https://doi.org/10.1016/j.ins.2011.12.028

Cawley, G. C., & Talbot, N. L. C. (2010). On over-fitting in model selection and subsequent selection bias in performance evaluation. *Journal of Machine Learning Research, 11*, 2079–2107.

Cho, K., van Merriënboer, B., Gulcehre, C., Bahdanau, D., Bougares, F., Schwenk, H., & Bengio, Y. (2014). Learning phrase representations using RNN encoder–decoder for statistical machine translation. En *Proceedings of EMNLP* (pp. 1724–1734).

Dietterich, T. G. (1998). Approximate statistical tests for comparing supervised classification learning algorithms. *Neural Computation, 10*(7), 1895–1923.

Friedman, J. H. (2001). Greedy function approximation: A gradient boosting machine. *The Annals of Statistics, 29*(5), 1189–1232.

Géron, A. (2017). *Hands-on machine learning with Scikit-Learn and TensorFlow*. O'Reilly Media.

Gneiting, T., & Raftery, A. E. (2007). Strictly proper scoring rules, prediction, and estimation. *Journal of the American Statistical Association, 102*(477), 359–378.

Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep learning*. MIT Press.

Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On calibration of modern neural networks. En *Proceedings of ICML* (pp. 1321–1330).

Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The elements of statistical learning* (2.ª ed.). Springer.

Hegre, H., Bell, C., Colaresi, M., Croicu, M., Hoyles, F., Jansen, R., … Vesco, P. (2021). ViEWS2020: Revising and evaluating the ViEWS political violence early-warning system. *Journal of Peace Research, 58*(3), 599–611.

Kingma, D. P., & Ba, J. (2015). Adam: A method for stochastic optimization. En *ICLR*. https://arxiv.org/abs/1412.6980

Lakshminarayanan, B., Pritzel, A., & Blundell, C. (2017). Simple and scalable predictive uncertainty estimation using deep ensembles. En *Advances in NeurIPS 30*.

Mitchell, M., Wu, S., Zaldivar, A., Barnes, P., Vasserman, L., Hutchinson, B., … Gebru, T. (2019). Model cards for model reporting. En *Proceedings of FAT\** (pp. 220–229).

Platt, J. (1999). Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods. En *Advances in Large Margin Classifiers* (pp. 61–74). MIT Press.

Raleigh, C., Linke, A., Hegre, H., & Karlsen, J. (2010). Introducing ACLED: An armed conflict location and event dataset. *Journal of Peace Research, 47*(5), 651–660.

Roberts, D. R., Bahn, V., Ciuti, S., Boyce, M. S., Elith, J., Guillera-Arroita, G., … Dormann, C. F. (2017). Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure. *Ecography, 40*(8), 913–929.

Ruder, S. (2017). An overview of multi-task learning in deep neural networks. *arXiv:1706.05098*.

Sundberg, R., & Melander, E. (2013). Introducing the UCDP Georeferenced Event Dataset. *Journal of Peace Research, 50*(4), 523–532.

Yurdakul, B. (2018). *Statistical properties of population stability index* (Tesis de doctorado). Western Michigan University.

*(Las referencias de datos de la U2 —SIMCI-UNODC/ODC, Global Forest Watch, DANE— siguen vigentes y se citan en el informe.)*
