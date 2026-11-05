> Texto extraído del cuaderno **Proyecto_U2_Modelo_Base.ipynb** (markdown y salidas de texto; sin imágenes). Código completo en `U2_modelo_base_codigo.py`.
> Parte 1 de 3.

# Clasificación del perfil de amenaza tecnológica por departamento
## Modelo base entrenado · Proyecto de Aplicación, Etapa 2 (Unidad 2)

**Redes Neuronales · Universidad de La Sabana · Sector elegido: Seguridad y defensa**

| | |
|---|---|
| **Integrantes** | Laura Alejandra Díaz Sastoque (0000288870) · Juan Diego Saldarriaga Muñoz (0000287391) |
| **Actividad** | Actividad 5: Avance del proyecto, modelo base entrenado |
| **Entrega anterior** | Etapa 1: formulación del problema, variables y dataset (Unidad 1) |
| **Fecha** | Septiembre de 2026 |
| **Herramientas** | Python · pandas · scikit-learn · TensorFlow / Keras 3 |

## Resumen ejecutivo

Este notebook implementa, entrena y diagnostica el **modelo base** del proyecto: un perceptrón multicapa en Keras que
estima, con la información disponible al cierre de un año, la probabilidad de que cada departamento tenga el año
siguiente perfil de amenaza **C (armas convencionales y emboscadas)** o **D (mixto)**, a partir de las variables
definidas en la Etapa 1.

| | Resultado |
|---|---|
| **Datos** | Panel departamento–año 2021–2026: 171 filas, 30 departamentos, 7 predictoras (6 de las 7 variables de la U1 y la letalidad derivada). Fuentes: ACLED, SIMCI/ODC (Datos Abiertos Colombia), Global Forest Watch y DANE |
| **Modelo** | MLP 7 → 16 ReLU → 1 sigmoide (145 parámetros), L2 = 10⁻³, Adam y parada temprana. Arquitectura elegida por validación cruzada temporal entre 16 configuraciones |
| **Desempeño fuera de muestra** | ROC-AUC 0,83 y exactitud 0,76 en la prueba 2026 (modelo reajustado). AUC 0,89 en validación temporal 2023–2026 (el azar da 0,50) |
| **Frente a la neurona lineal** | Equivalente en el tiempo (AUC 0,89 vs. 0,88; McNemar p = 1,0). Inferior en departamentos nuevos (0,80 vs. 0,84), salvo con ensamble de semillas (0,85) |
| **Frente a la persistencia** | Inferior (AUC 0,96): el perfil se repite el 86 % de las veces y las variables de la U1 son estructurales |
| **Diagnóstico** | Converge de forma estable y es reproducible (CV del AUC entre semillas 2,8 %). El límite es la generalización con 30 unidades espaciales, no la optimización |
| **Mejora medida** | Agregar la memoria del perfil (E1) sube el AUC temporal de 0,89 a 0,93 y baja la log-loss de 0,44 a 0,36, por debajo de la persistencia (0,38) |

**En una frase:** el modelo base es técnicamente viable y coherente con el problema, pero a escala departamental su
valor agregado sobre la inercia del conflicto es limitado. La evidencia señala dos palancas para la Unidad 3:
memoria del perfil y migración a municipio.

## Cómo leer este notebook

La guía de la actividad pide siete contenidos y la rúbrica evalúa cinco criterios. La tabla indica en qué sección
se responde cada uno, para que la revisión sea directa.

| Contenido exigido / criterio de la rúbrica | Sección |
|---|---|
| Recordatorio sintético del problema formulado · **Coherencia problema–modelo** | §1 (y tabla de trazabilidad U1 → U2), §12 |
| Descripción del dataset (fuente, tamaño, variables, calidad) | §2, §3, §4 |
| Arquitectura y configuración del modelo base · **Implementación y configuración técnica** | §6, §7, §8 |
| Curvas de aprendizaje · Métricas de entrenamiento y validación · **Presentación e interpretación de métricas y curvas** | §9, §10 |
| Diagnóstico técnico (convergencia, estabilidad, generalización) · **Diagnóstico técnico** | §11 |
| Oportunidades de mejora · **Mejoras y trabajo colaborativo** | §13, §14 |
| Conclusiones, artefactos para el informe y referencias | §15, §16, Anexo |

> **Reproducibilidad.** Todas las cifras citadas en el texto corresponden a la ejecución de referencia guardada
> en este archivo (semilla 42, CPU, versiones impresas en §0). El cuaderno corre de principio a fin en Google Colab
> o en local con `acled-data-export.csv` en la misma carpeta; las fuentes externas se descargan solas la primera vez
> y quedan en caché (`datos_externos/`). Si se ejecuta en otro hardware, las cifras pueden variar en decimales.

---
# 0. Configuración del entorno y reproducibilidad

Fijamos una semilla global y activamos el determinismo de operaciones de TensorFlow para que el entrenamiento
de la red sea repetible. También desactivamos las optimizaciones oneDNN, que reordenan sumas en punto flotante y
cambian los resultados en el último decimal de una corrida a otra.

#### Celda de código 1: 0.1 Librerías, semilla global y rutas de trabajo  (fuente completa en U2_modelo_base_codigo.py)
Salida de la celda:
```text
 | Valor
Python | 3.11.9
Sistema | Windows-10
NumPy / pandas | 2.1.2 / 2.2.3
scikit-learn | 1.9.0
TensorFlow / Keras | 2.21.0 / 3.15.1
Backend de Keras | tensorflow
Dispositivos | CPU
Semilla global | 42
```

#### Celda de código 2: 0.2 Estilo gráfico, paleta y funciones auxiliares  (fuente completa en U2_modelo_base_codigo.py)
Salida de la celda:
```text
Estilo y utilidades listos.
```

---
# 1. Recordatorio sintético del problema (Etapa 1)

**Pregunta orientadora formulada en la Unidad 1:**

> *¿Cómo orientar la adquisición, transferencia de tecnología y fabricación de sistemas de seguridad y defensa de las
> Fuerzas de Colombia en función del perfil de amenaza tecnológica predominante de los municipios colombianos, según
> las variables de actividad armada histórica, geografía operacional y economías ilícitas, de modo que se optimice la
> inversión de recursos?*

La Etapa 1 dejó definidos el fenómeno, la variable objetivo y las predictoras, y declaró tres hallazgos que
condicionan esta entrega:

1. **Resolución espacial.** El export gratuito de ACLED solo desagrega hasta **departamento** (33 unidades); el
   municipio requiere el registro de evento individual (con licencia). Por eso la unidad táctica sigue siendo el
   municipio, pero el modelo se valida por ahora a nivel departamental.
2. **Variable objetivo binaria.** Bajo la regla del 50 %, ningún departamento tuvo perfil A (IED/explosivos) o B
   (drones/remotos) dominante, así que el clasificador se ajustó a **C (armas convencionales y emboscadas) vs. D
   (amenaza mixta)**.
3. **Tamaño muestral.** Una fila por departamento (33 filas) no basta para entrenar una red. La Unidad 1 comprometió
   para esta etapa un **panel departamento–año** (33 × 6 ≈ 198 filas), que es lo que implementamos.

### 1.1 Formulación del modelo base

Para cada departamento $d$ y año $t$ la etiqueta se construye con la regla de la Etapa 1:

$$
s^{C}_{d,t} = \frac{\#\{\text{eventos de subtipos C en } d,t\}}{\#\{\text{eventos en } d,t\}},
\qquad
y_{d,t} = \mathbb{1}\left[s^{C}_{d,t} \ge 0.5\right] \quad (1 = \text{perfil C},\; 0 = \text{perfil D}).
$$

La red aprende la probabilidad del perfil del año siguiente a partir de la información disponible al cierre del año anterior:

$$
\hat{p}_{d,t} = f_{\boldsymbol{\theta}}\left(\mathbf{x}_{d,t-1}\right) = P\left(y_{d,t}=1 \mid \mathbf{x}_{d,t-1}\right).
$$

Usamos el rezago $t-1 \rightarrow t$ por dos razones:

* **Evita la circularidad.** La etiqueta de $t$ se calcula con los eventos de $t$. Si usáramos como predictores los
  eventos o las fatalidades del mismo año, parte de la respuesta entraría en la pregunta (las fatalidades son
  consecuencia de los mismos ataques que definen el perfil C) y las métricas quedarían infladas.
* **Replica el momento real de la decisión.** La asignación de tecnología para un año se decide con lo observado hasta el
  cierre del año anterior; por ejemplo, la inversión de 2026 citada en la Etapa 1 se planeó con datos de 2025. El
  modelo responde exactamente esa pregunta: *con lo que sé hoy, ¿qué perfil tendrá cada territorio el próximo año?*

### 1.2 Trazabilidad de variables: de la Etapa 1 al modelo base

| ID (U1) | Variable definida en la Etapa 1 | Operacionalización en el modelo base | Fuente | Estado |
|---|---|---|---|---|
| $y$ | Perfil de amenaza predominante | $\mathbb{1}[s^C_{d,t} \ge 0.5]$, regla y mapeo de subtipos **idénticos** a la U1 | ACLED | ✅ Implementada |
| $x_1$ | Frecuencia total de eventos | $\log(1+\text{eventos anualizados}_{t-1})$ | ACLED | ✅ Implementada |
| $x_2$ | Fatalidades acumuladas | $\log(1+\text{fatalidades anualizadas}_{t-1})$ | ACLED | ✅ Implementada |
| $x_{2b}$ | *(derivada de $x_2$)* letalidad por evento | $\text{fatalidades}_{t-1}/\text{eventos}_{t-1}$: la U1 justificó $x_2$ porque *"distingue letalidad por arma"* | ACLED | ✅ Derivada |
| $x_3$ | Nº de grupos armados activos | Requiere los campos `actor1/actor2` del registro de evento individual | ACLED (licencia) | ⏳ Pendiente (U3) |
| $x_4$ | Hectáreas de cultivos ilícitos | $\log(1+\text{ha de coca}_{t-1})$; se refina el promedio 2020-2024 de la U1 a un valor **anual** | SIMCI–UNODC vía MinJusticia/ODC (datos.gov.co) | ✅ Implementada |
| $x_5$ | Cobertura forestal (%) | % del área con dosel arbóreo ≥ 30 % en $t-1$ | Global Forest Watch / Hansen et al. (sustituye a IGAC) | ✅ Fuente sustituta |
| $x_6$ | Distancia a casco urbano (km) | Distancia geodésica del centroide departamental a Bogotá (lejanía del centro político) | ACLED (centroides) | ⚠️ Proxy departamental |
| $x_7$ | Índice de pobreza multidimensional | IPM 2018 por departamento (% de personas) | DANE, ECV 2018 | ✅ Implementada |

Todas las predictoras son **numéricas continuas**, como se definió en la Etapa 1. Las que cambian de fuente o de
forma se justifican en §4 y quedan registradas en la bitácora de decisiones (§14).

### 1.3 De la predicción a la decisión

| Perfil proyectado | Lectura operacional (Etapa 1, §2) | Énfasis tecnológico que orienta |
|---|---|---|
| **C: convencional / emboscada** | La violencia directa entre actores armados domina el territorio | ISR (inteligencia, vigilancia y reconocimiento), detección temprana, movilidad protegida |
| **D: mixto** | Ningún vector armado supera el 50 %; pesan la protesta, el disturbio y la violencia mixta | Portafolio flexible y capacidades de orden público, con vigilancia de señales A/B |
| **Señal A/B** (no dominante) | IED y drones: bajo volumen pero alto impacto (hallazgo crítico U1 §5.4) | Alerta complementaria para C-UAS y detección de AEI, aunque el perfil sea C o D |

---
# 2. Dataset ACLED: descripción y auditoría de calidad

**Fuente.** *Armed Conflict Location & Event Data* (ACLED; Raleigh et al., 2010), export agregado semanal de
Colombia, descargado con registro académico gratuito (licencia de uso no comercial). Cada fila es la combinación
**semana × departamento × subtipo de evento**.

| Columna | Descripción | Uso en el modelo |
|---|---|---|
| `week` | Fecha de inicio de la semana (sábado) | Construcción del año $t$ |
| `country`, `region` | País y región | Filtro (se excluyen aguas internacionales) |
| `admin1` | Departamento | Unidad de análisis |
| `event_type`, `sub_event_type` | Taxonomía ACLED (6 tipos, 21 subtipos presentes) | Variable objetivo $y$ |
| `events` | Número de eventos de ese subtipo en la semana | $x_1$ y $y$ |
| `fatalities` | Fatalidades reportadas | $x_2$, $x_{2b}$ |
| `population_exposure` | Población expuesta estimada (texto con separador de miles) | Solo auditoría |
| `disorder_type` | Violencia política / manifestaciones / desarrollos estratégicos | Solo auditoría |
| `centroid_latitude`, `centroid_longitude` | Centroide del departamento | $x_6$ y mapa (§12) |

#### Celda de código 3: 2.1 Carga del archivo original  (fuente completa en U2_modelo_base_codigo.py)
Salida de la celda:
```text
Archivo: acled-data-export.csv | 2.24 MB | 17,573 filas × 12 columnas
 | week | region | country | admin1 | event_type | sub_event_type | events | fatalities | population_exposure | disorder_type | centroid_latitude | centroid_longitude
0 | 2019-12-28 | South America | Colombia | Antioquia | Battles | Armed clash | 2 | 0 | 14,172 | Political violence | 7.157 | -75.445
1 | 2019-12-28 | South America | Colombia | Antioquia | Violence against civilians | Attack | 1 | 1 | 9,111 | Political violence | 7.157 | -75.445
2 | 2019-12-28 | South America | Colombia | Atlantico | Violence against civilians | Attack | 1 | 1 | 803,116 | Political violence | 10.668 | -74.994
3 | 2019-12-28 | South America | Colombia | Atlantico | Riots | Mob violence | 1 | 0 | 190,181 | Political violence | 10.668 | -74.994
```

#### Celda de código 4: 2.2 Auditoría de calidad: cada verificación con su implicación para el modelo  (fuente completa en U2_modelo_base_codigo.py)
Salida de la celda:
```text
Verificación | Resultado | Implicación para el modelo
Registros / columnas | 17,573 / 12 | Archivo ligero (≈2 MB), carga directa con pandas
Registros fuera de Colombia | 32 (North Atlantic Ocean) | Se excluyen: aguas internacionales, no son territorio de decisión
Cobertura temporal | 2019-12-28 → 2026-08-15 | Se descarta la semana 2019-12-28 (horizonte 2020+ de la U1)
Semanas distintas / saltos ≠ 7 días | 347 / 0 | Serie semanal continua, sin huecos
Valores faltantes (todas las columnas) | 0 | No se requiere imputación
Duplicados exactos | 0 | Sin doble conteo
Duplicados por llave semana–depto–subtipo | 0 | La llave es única: agregación segura
Rangos: events ≥ 1 y fatalities ≥ 0 | True | events máx = 85, fatalities máx = 48
Subtipos asignados a más de un tipo | 0 | Taxonomía consistente con el codebook
Centroides distintos por departamento (máx.) | 1 | Un centroide fijo por departamento
population_exposure no convertible a número | 0 | Texto con separador de miles: se limpia
Departamentos de Colombia (admin1) | 33 | 32 departamentos + Bogotá D.C.
```

#### Celda de código 5: 2.3 Filtrado al horizonte de estudio y conciliación con las cifras de la Etapa 1  (fuente completa en U2_modelo_base_codigo.py)
Salida de la celda:
```text
 | Etapa 1 (reportado) | Export actual, mismo corte | Export actual, completo
Registros semanales | 17491 | 17442 | 17533
Eventos | 33336 | 33144 | 33275
Fatalidades | 13732 | 13525 | 13591
Última semana | 2026-08-01 | 2026-08-01 | 2026-08-15
Diferencia de registros frente a la Etapa 1 con el mismo corte: -0.28 %
```

**Lectura.** El archivo es el mismo tipo de export de la Etapa 1 pero actualizado al 15 de agosto de 2026. ACLED
revisa retroactivamente su base (reclasifica y fusiona eventos a medida que verifica fuentes), así que con el mismo
corte los conteos difieren en menos de 0,5 %. La diferencia no altera la estructura de los datos ni las conclusiones de
la Etapa 1, pero la dejamos documentada porque las cifras de este informe no coinciden al dígito con las del
anterior. Más abajo (§3.2) comprobamos que el resultado central de la Etapa 1, la partición de departamentos entre
C y D, se reproduce con el export actualizado.

#### Celda de código 6: 2.4 Cobertura de los datos: dinámica nacional y densidad departamento–año  (fuente completa en U2_modelo_base_codigo.py)
Figuras generadas: fig01_cobertura_datos (imagen omitida en este texto).

**Lectura.** (a) La serie no tiene huecos, y el pico de 2021 corresponde al Paro Nacional, cuando la protesta
disparó el conteo sin cambiar la violencia armada de fondo; es la distorsión que la Etapa 1 ya había señalado.
(b) La densidad de datos es muy desigual: Cauca, Antioquia o Norte de Santander suelen superar los 400 eventos por
año, mientras que Vaupés, Guainía o San Andrés tienen un solo dígito. Con tan pocos eventos, un solo evento cambia la
proporción de subtipos en más de 10 puntos, y eso hace poco confiable la etiqueta de esas celdas. Lo tratamos
con un umbral mínimo de eventos (§3.3).

---
# 3. Construcción de la variable objetivo (panel departamento–año)

## 3.1 Mapeo de subtipos a perfiles (regla de la Etapa 1, sin cambios)

Reutilizamos **literalmente** el diccionario de la Etapa 1. Los subtipos que no aparecen en él toman el perfil D
(`fillna('D')` en el código original). Dejamos explícita esa asignación por defecto porque en la Etapa 1 quedaba
implícita en el código.

#### Celda de código 7: 3.1 Mapeo subtipo -> perfil (idéntico al notebook de la Etapa 1)  (fuente completa en U2_modelo_base_codigo.py)
Salida de la celda:
```text
 | Tipo | Subtipo | Eventos | Fatalidades | perfil | asignación | % eventos | fatalidades/evento
0 | Protests | Peaceful protest | 10.836 | 0 | D | explícita (U1) | 32,56 | 0,00
1 | Violence against civilians | Attack | 7.802 | 9.043 | C | explícita (U1) | 23,45 | 1,16
2 | Battles | Armed clash | 4.785 | 3.630 | C | explícita (U1) | 14,38 | 0,76
3 | Strategic developments | Looting/property destruction | 1.566 | 0 | C | explícita (U1) | 4,71 | 0,00
4 | Riots | Violent demonstration | 1.276 | 124 | D | por defecto → D (U1) | 3,83 | 0,10
5 | Riots | Mob violence | 1.257 | 183 | D | explícita (U1) | 3,78 | 0,15
6 | Violence against civilians | Abduction/forced disappearance | 1.223 | 0 | C | explícita (U1) | 3,68 | 0,00
7 | Strategic developments | Other | 1.145 | 3 | D | por defecto → D (U1) | 3,44 | 0,00
8 | Strategic developments | Disrupted weapons use | 1.072 | 18 | C | explícita (U1) | 3,22 | 0,02
9 | Explosions/Remote violence | Remote explosive/landmine/IED | 931 | 290 | A | explícita (U1) | 2,80 | 0,31
10 | Explosions/Remote violence | Grenade | 351 | 34 | A | explícita (U1) | 1,05 | 0,10
11 | Strategic developments | Change to group/activity | 296 | 0 | D | por defecto → D (U1) | 0,89 | 0,00
12 | Explosions/Remote violence | Air/drone strike | 264 | 191 | B | explícita (U1) | 0,79 | 0,72
13 | Protests | Protest with intervention | 173 | 0 | D | por defecto → D (U1) | 0,52 | 0,00
14 | Strategic developments | Arrests | 132 | 0 | D | explícita (U1) | 0,40 | 0,00
15 | Protests | Excessive force against protesters | 60 | 32 | D | explícita (U1) | 0,18 | 0,53
16 | Violence against civilians | Sexual violence | 48 | 11 | D | por defecto → D (U1) | 0,14 | 0,23
17 | Explosions/Remote violence | Shelling/artillery/missile attack | 45 | 25 | D | por defecto → D (U1) | 0,14 | 0,56
18 | Strategic developments | Agreement | 9 | 0 | D | por defecto → D (U1) | 0,03 | 0,00
19 | Explosions/Remote violence | Suicide bomb | 3 | 7 | A | explícita (U1) | 0,01 | 2,33
20 | Strategic developments | Headquarters or base established | 1 | 0 | D | por defecto → D (U1) | 0,00 | 0,00
perfil
A     3.9
B     0.8
C    49.4
D    45.9
```

**Lectura.** La etiqueta descansa sobre pocos subtipos. *Peaceful protest* (D), *Attack* y *Armed clash* (C) suman
el 70 % de los eventos, y los subtipos A y B juntos apenas llegan al 4,7 % (3,9 % y 0,8 %). La columna de letalidad
confirma la intuición de la Etapa 1: los dos subtipos C más frecuentes (*Attack*, 1,16 fatalidades por evento;
*Armed clash*, 0,76) son entre 5 y 10 veces más letales que los disturbios D, y la protesta pacífica tiene letalidad
nula. Eso respalda usar $x_2$ y $x_{2b}$, medidos en el año anterior, como predictores.

Dos subtipos con significado armado quedan en D por la regla por defecto: *Shelling/artillery/missile attack* y
*Sexual violence*. Los mantenemos así para no romper la trazabilidad con la Etapa 1, y en §3.4 medimos cuánto
cambiaría la etiqueta si se reasignaran.

#### Celda de código 8: 3.2 Panel departamento–año y réplica del resultado de la Etapa 1  (fuente completa en U2_modelo_base_codigo.py)
Salida de la celda:
```text
Panel: 33 departamentos × 7 años = 231 celdas (2020-2026; 2026 con 33 semanas)
Celdas con perfil A o B dominante: 0

Réplica Etapa 1 (perfil agregado 2020-2026): {'D': 17, 'C': 16} | Etapa 1 reportó {'D': 17, 'C': 16}
```

#### Celda de código 9: 3.3 Confiabilidad de la etiqueta y dinámica temporal de los perfiles  (fuente completa en U2_modelo_base_codigo.py)
Salida de la celda:
```text
Celdas 2021-2026 con eventos: 194 | con ≥ 10 eventos (etiqueta confiable): 171
Error estándar máximo de s_C con n = 10 eventos: ±0.158
Perfil t | C | D
Perfil t-1 |  | 
C | 69 | 12
D | 11 | 73
Persistencia año a año: 86.1% de los departamentos repite perfil
```
Figuras generadas: fig02_variable_objetivo (imagen omitida en este texto).

**Lectura.**

* **Umbral de confiabilidad.** Con $n$ eventos, la proporción $s^C$ tiene un error estándar de hasta
  $\sqrt{0.25/n}$. Con menos de 10 eventos ese error supera los 16 puntos y la etiqueta se decide casi al azar.
  Por eso excluimos esas celdas (rayadas en el panel a), que corresponden sobre todo a departamentos de la Amazonía
  y la Orinoquía con muy pocos registros. De 194 celdas 2021-2026 con eventos quedan **171 filas etiquetadas** (88 %),
  en línea con las ≈198 filas previstas en la Unidad 1.
* **Balance de clases.** Las clases quedan casi balanceadas en el agregado, pero **la prevalencia de C cambia con el
  tiempo** (panel b). Baja al 32 % en 2021 por el Paro Nacional, que infló el componente D, y alcanza su máximo
  (62 %) en 2025, año de la ruptura de los diálogos y de la crisis del Catatumbo. Este cambio de prevalencia entre años será clave para
  interpretar las métricas de validación (§10) y la calibración.
* **Persistencia.** El 86 % de los departamentos repite perfil de un año al siguiente (matriz de transición: 69 C→C,
  73 D→D y solo 23 cambios). Esa inercia hace que "repetir el perfil del año pasado" sea una línea base muy fuerte, y
  la incluimos explícitamente (§6).
* **Ambigüedad cerca del umbral.** El 20 % de las celdas cae a menos de 10 puntos del 50 % (panel c). En esas celdas
  la etiqueta puede cambiar por unos pocos eventos, así que esperamos que los errores del modelo se concentren ahí
  (lo verificamos en §11.5).

#### Celda de código 10: 3.4 Sensibilidad de la etiqueta a los subtipos asignados por defecto  (fuente completa en U2_modelo_base_codigo.py)
Salida de la celda:
```text
Etiquetas que cambian al reasignar Shelling→B y Sexual violence→C: 1 de 171 (0.6%)
perfil | departamento | anio | perfil_u1 | perfil_alt
135 | Magdalena | 2022 | D | C
```

**Lectura.** Reasignar los dos subtipos ambiguos cambia **una sola** etiqueta de 171 (Magdalena 2022, que tenía
$s^C = 0{,}494$, pegada al umbral). La regla de la Etapa 1 es robusta a esta decisión de mapeo, así que conservarla no
sesga el modelo base. Dejamos la reasignación como mejora de rigor conceptual para la siguiente iteración (§13).

---
# 4. Variables predictoras: integración de fuentes

Las variables $x_1$, $x_2$ y $x_{2b}$ salen de ACLED. Las estructurales ($x_4$–$x_7$) vienen de fuentes
oficiales o científicas y se cruzan por nombre de departamento normalizado. Cada fuente se descarga una sola vez y
queda en caché, así que el notebook es reproducible sin conexión.

| Variable | Fuente y justificación del cambio frente a la Etapa 1 |
|---|---|
| $x_4$ coca | *Detección de cultivos de coca (hectáreas)*, MinJusticia / Observatorio de Drogas de Colombia, publicado en **Datos Abiertos Colombia** (recurso obligatorio de la guía). Es el censo SIMCI–UNODC a nivel municipal (2001-2024), que agregamos por departamento y año. El conjunto abierto llega hasta el censo 2024, así que para 2025-2026 se arrastra el último valor disponible. |
| $x_5$ bosque | No encontramos una tabla departamental anual descargable de IGAC o IDEAM, así que usamos **Global Forest Watch** (Hansen et al., 2013; UMD), que da la cobertura arbórea del año 2010 y la pérdida anual por departamento. Con eso calculamos el % del área con dosel ≥ 30 % en cada año, que se acerca más al mecanismo de la hipótesis U1 (*la bóveda vegetal limita drones y favorece IED y emboscadas*). |
| $x_6$ distancia | La distancia a casco urbano es un dato municipal, y su promedio departamental requiere cartografía IGAC. Como *proxy* de presencia diferenciada del Estado (CNMH, 2022) usamos la distancia geodésica del centroide departamental a Bogotá. Se declara como aproximación. |
| $x_7$ IPM | **DANE, Pobreza multidimensional por departamentos 2018** (ECV 2018), la misma fuente declarada en la Etapa 1. Es estático (una medición). |

#### Celda de código 11: 4.1 Fuentes externas con descarga automática y caché local  (fuente completa en U2_modelo_base_codigo.py)
Salida de la celda:
```text
✓ Coca SIMCI/ODC (datos.gov.co): caché local (datos_externos\coca_departamento_anio.csv)
✓ Cobertura arbórea (Global Forest Watch): caché local (datos_externos\bosque_departamento_anio.csv)
 | Total nacional (ha, este dataset) | Cifra oficial publicada (ha, redondeada)
anio |  | 
2019 | 154476.0 | NaN
2020 | 142784.0 | 143000.0
2021 | 204257.0 | 204000.0
2022 | 230028.0 | 230000.0
2023 | 252572.0 | 253000.0
2024 | 261386.0 | NaN
```

#### Celda de código 12: 4.2 Variables estructurales estáticas: IPM (DANE) y distancia a Bogotá  (fuente completa en U2_modelo_base_codigo.py)
Salida de la celda:
```text
Cruce de llaves entre fuentes (departamentos sin dato): {'IPM': [], 'coca (sin registro = 0 ha)': ['atlantico', 'bogota', 'casanare', 'huila', 'quindio', 'risaralda', 'san andres', 'sucre', 'tolima'], 'bosque': []}
```

#### Celda de código 13: 4.3 Dataset de modelado: etiqueta en t, predictoras en t-1  (fuente completa en U2_modelo_base_codigo.py)
Salida de la celda:
```text
Filas | Departamentos | Años etiqueta | Perfil C | Perfil D | Predictoras | Faltantes
171 | 30 | 2021-2026 | 82 | 89 | 7 | 0
 | Media | Desv. est. | Mín. | Mediana | Máx.
x₁ eventos (log) | 4,61 | 1,17 | 0,69 | 4,73 | 6,61
x₂ fatalidades (log) | 3,46 | 1,40 | 0,00 | 3,47 | 6,02
x₂b letalidad | 0,41 | 0,31 | 0,00 | 0,37 | 1,57
x₄ coca (log ha) | 4,33 | 4,41 | 0,00 | 2,79 | 11,22
x₅ bosque (%) | 60,14 | 21,95 | 17,87 | 61,98 | 98,22
x₆ distancia (km) | 371,18 | 191,91 | 37,07 | 383,92 | 800,59
x₇ IPM (%) | 25,77 | 11,46 | 4,40 | 25,10 | 55,00
```

**Lectura.** El dataset final de modelado tiene **una fila por departamento–año etiquetable** (171 filas,
30 departamentos, etiquetas 2021-2026, sin faltantes), con 7 predictoras numéricas medidas en $t-1$ y la etiqueta en
$t$. Las escalas son muy heterogéneas: la distancia va de 37 a 801 km y el IPM de 4,4 a 55 puntos.
Por eso estandarizamos antes de entrenar, con media y desviación **calculadas solo con el conjunto de entrenamiento**
para no filtrar información del futuro (§6). Las transformaciones logarítmicas de $x_1$, $x_2$ y $x_4$ corrigen la
asimetría extrema de conteos y hectáreas, porque unos pocos departamentos, como Cauca o Nariño, concentran la mayor
parte de la masa.

---
# 5. Análisis exploratorio orientado al modelado

Aquí el EDA tiene un objetivo concreto: comprobar si las predictoras separan las clases y si hay indicios de relaciones
**no lineales o de interacción**, que es lo que justificaría la red frente a la regresión logística (argumento central
de la Etapa 1, §6).

#### Celda de código 14: 5.1 Distribución de cada predictora por perfil  (fuente completa en U2_modelo_base_codigo.py)
Figuras generadas: fig03_predictoras_por_clase (imagen omitida en este texto).

#### Celda de código 15: 5.2 Dependencia lineal/monótona vs. dependencia general, y correlación entre predictoras  (fuente completa en U2_modelo_base_codigo.py)
Salida de la celda:
```text
 | ρ Spearman con y | Información mutua
x₁ eventos (log) | 0.144 | 0.033
x₂ fatalidades (log) | 0.488 | 0.138
x₂b letalidad | 0.622 | 0.277
x₄ coca (log ha) | 0.526 | 0.223
x₅ bosque (%) | 0.381 | 0.326
x₆ distancia (km) | 0.241 | 0.384
x₇ IPM (%) | 0.256 | 0.431
```
Figuras generadas: fig04_dependencia_correlacion (imagen omitida en este texto).

#### Celda de código 16: 5.3 Hipótesis de interacción de la Etapa 1 (§6.2): bosque × distancia y coca × pobreza  (fuente completa en U2_modelo_base_codigo.py)
Figuras generadas: fig05_interacciones_hipotesis (imagen omitida en este texto).

**Lectura del EDA.**

* **Separación univariada.** Seis de las siete predictoras difieren entre perfiles con alta significancia
  (Mann-Whitney, $p < 0{,}01$), en la dirección que anticipaba la Etapa 1. Comparando medianas, los departamentos C
  tienen más del doble de letalidad por evento (0,52 vs. 0,20), coca donde los D casi no tienen (≈5.000 ha vs. 0),
  más bosque (75 % vs. 59 %), están más lejos de Bogotá (409 vs. 235 km) y son más pobres (IPM 30 vs. 17). La
  excepción es $x_1$ (volumen de eventos, $p = 0{,}06$), y coincide con el argumento de la Etapa 1: **el volumen no
  define el perfil**. Bogotá tiene muchos eventos y es D.
* **Indicios de no linealidad.** En el panel (a) de §5.2, $x_6$ (distancia) y $x_7$ (IPM) tienen la mayor
  información mutua (0,38 y 0,43) pero una asociación monótona débil (|ρ| ≈ 0,25). Una parte es no linealidad genuina:
  La Guajira, Sucre y Magdalena son de los departamentos más pobres y alejados y aun así son D todos los años. Otra
  parte es una advertencia: $x_6$ y $x_7$ son **estáticas**, es decir, toman un valor único por departamento, y
  combinadas con una etiqueta muy persistente funcionan casi como un identificador del departamento. Una red puede
  aprovechar eso para memorizar departamentos en lugar de aprender relaciones transferibles. Por eso la validación
  espacial de §11.3, con departamentos nunca vistos, es indispensable.
* **Colinealidad.** Hay pares fuertemente correlacionados: distancia–IPM (ρ = 0,79), eventos–fatalidades (0,77),
  fatalidades–coca (0,65), fatalidades–letalidad (0,64) y coca–bosque (0,56). Esto no impide entrenar, pero vuelve
  inestable la lectura individual de coeficientes lineales, y nos lleva a interpretar con importancia por permutación
  (§11.5) y no con pesos.
* **Interacciones.** En (a), C domina en la zona de "mucho bosque (> 65 %) y lejos (300–550 km)". Pero la frontera no
  es un hiperplano: Arauca es C todos los años con apenas 34 % de bosque y sin coca (conflicto de frontera, no de
  selva), mientras Bolívar, con 59 % de bosque, 475 km y ≈10.000 ha de coca, es D en cinco de seis años. En (b), la coca
  separa bien (casi todos los C tienen > 1.000 ha), pero Valle del Cauca y Bolívar tienen coca y son mayoritariamente D:
  la economía ilícita no basta sin el contexto territorial. La prueba formal de si esa forma aporta está en §11.4.

---
# 6. Protocolo experimental

## 6.1 Partición temporal (sin barajar)

Como el objetivo es **anticipar** el perfil del año siguiente, la partición respeta el tiempo: el modelo se
entrena con el pasado y se evalúa con años que nunca vio. Una partición aleatoria mezclaría filas de 2025 en el
entrenamiento y de 2022 en la prueba, y daría una estimación optimista que no se reproduce en el uso real.

| Conjunto | Años de la etiqueta $t$ | Predictoras $t-1$ | Uso |
|---|---|---|---|
| **Entrenamiento** | 2021–2024 | 2020–2023 | Ajuste de pesos y del escalador |
| **Validación** | 2025 | 2024 | Parada temprana, reducción de la tasa de aprendizaje y umbral de decisión |
| **Prueba** | 2026 (enero–agosto) | 2025 | Evaluación final, una sola vez |

El `StandardScaler` se ajusta **solo con entrenamiento** y se aplica igual a validación y prueba; así ninguna
estadística del futuro entra al modelo. La validación de hiperparámetros (§7.2) usa además **validación cruzada
temporal con origen móvil** (entrenar con años anteriores a $v$ y validar en $v$, para $v$ = 2023, 2024 y 2025),
la variante de validación cruzada que corresponde a datos ordenados en el tiempo (Bergmeir & Benítez, 2012).

#### Celda de código 17: 6.1 Partición temporal y estandarización ajustada solo con entrenamiento  (fuente completa en U2_modelo_base_codigo.py)
Salida de la celda:
```text
 | Años (t) | Filas | Perfil C | Prevalencia C | Departamentos
Entrenamiento | 2021-2024 | 113 | 50 | 0,44 | 29
Validación | 2025 | 29 | 18 | 0,62 | 29
Prueba | 2026 (ene-ago) | 29 | 14 | 0,48 | 29
```
Figuras generadas: fig06_particion_temporal (imagen omitida en este texto).

## 6.2 Líneas base

Una métrica solo se puede interpretar frente a una referencia. Comparamos la red contra tres líneas base de
complejidad creciente:

| Línea base | Regla | Qué mide |
|---|---|---|
| **Clase mayoritaria** | Predice siempre la clase más frecuente en entrenamiento (D) | El piso: cualquier modelo útil debe superarla |
| **Persistencia** | $\hat{p}_{d,t} = s^{C}_{d,t-1}$: el perfil del año pasado se repite | La inercia del conflicto. Es la línea base que ya usa, en la práctica, un planificador sin modelo |
| **Regresión logística** | $\hat{p} = \sigma(\mathbf{w}^\top\mathbf{x} + b)$ con las mismas 7 predictoras | El modelo lineal que la Etapa 1 (§6.3) prometió comparar contra la red: *una sola neurona* |

La regresión logística recibe el mismo trato que la red: mismas predictoras, mismo escalado, pesos de clase
balanceados, y su hiperparámetro de regularización $C$ se elige con la misma validación cruzada temporal.

#### Celda de código 18: 6.2 Líneas base: mayoritaria, persistencia y regresión logística (una neurona)  (fuente completa en U2_modelo_base_codigo.py)
Salida de la celda:
```text
C elegido para la logística (menor BCE media en validación temporal): 3.0
C
0.01     0.5714
0.03     0.5029
0.10     0.4506
0.30     0.4256
1.00     0.4146
3.00     0.4124
10.00    0.4141
 | Coeficiente (por 1 DE) | Razón de odds | Signo esperado (U1) | ¿Coincide?
x₁ eventos (log) | -1,779 | 0,169 | ± | —
x₂ fatalidades (log) | 2,554 | 12,862 | + | sí
x₂b letalidad | 0,339 | 1,404 | + | sí
x₄ coca (log ha) | 0,133 | 1,142 | + | sí
x₅ bosque (%) | 0,620 | 1,859 | + | sí
x₆ distancia (km) | 0,477 | 1,612 | + | sí
x₇ IPM (%) | -0,204 | 0,815 | + | no
```

**Lectura de la línea base lineal.** Los coeficientes estandarizados son la versión lineal de las hipótesis de la
Etapa 1: un coeficiente positivo aumenta la probabilidad de perfil C.

* Cinco de las seis hipótesis direccionales se cumplen (fatalidades, letalidad, coca, bosque y distancia empujan hacia
  C), así que la formulación del problema está bien orientada.
* $x_1$ y $x_2$ salen con signos opuestos y magnitudes grandes (−1,78 y +2,55). Juntos codifican una **razón**: a
  igualdad de fatalidades, más eventos significa más protesta y, por tanto, perfil D. La logística solo puede
  representar esa interacción multiplicativa como una resta de logaritmos, de ahí los coeficientes grandes.
* $x_7$ (IPM) sale **negativo**, contrario a la hipótesis U1. No quiere decir que la pobreza reduzca la amenaza
  convencional: es la colinealidad con la distancia (ρ = 0,79), que hace que la logística reparta el efecto entre las
  dos y le asigne signos arbitrarios. §11.4 muestra que la red recupera el efecto esperado.
* La $C$ elegida (3,0) implica una regularización débil: el modelo lineal no está sobreajustando, está limitado por
  su forma.

---
# 7. Arquitectura y configuración del modelo base

## 7.1 Decisiones de diseño

El modelo base es un **perceptrón multicapa (MLP) totalmente conectado** implementado con la API funcional de
Keras, tal como se anunció en la Etapa 1. Con datos tabulares de 7 variables numéricas y sin estructura espacial ni
secuencial explícita, un MLP es la arquitectura neuronal adecuada. Las redes convolucionales o recurrentes
suponen estructuras (rejillas, secuencias largas) que este panel no tiene.

| Componente | Elección | Justificación técnica |
|---|---|---|
| **Entrada** | 7 neuronas ($x_1, x_2, x_{2b}, x_4, x_5, x_6, x_7$) estandarizadas | Determinada por las variables de la Etapa 1. La estandarización equilibra las escalas para que ninguna variable domine el gradiente |
| **Capas ocultas** | Seleccionadas por validación cruzada temporal (§7.2) | Con $n \approx 113$ filas de entrenamiento la capacidad debe ser pequeña; la búsqueda compara 1 y 2 capas y 8–32 neuronas |
| **Activación oculta** | ReLU, $\phi(z) = \max(0, z)$ | Derivada 1 en la región activa: no satura ni desvanece el gradiente como la logística (derivada máx. 0,25), como vimos en la Actividad 3 |
| **Inicialización** | He normal, $\mathcal{N}(0, 2/n_{in})$ | Conserva la varianza de las activaciones a través de capas ReLU (He et al., 2015) |
| **Salida** | 1 neurona sigmoide | Produce $\hat{p} = P(\text{C})\in(0,1)$, una probabilidad y no solo una etiqueta, útil para priorizar territorios |
| **Pérdida** | Entropía cruzada binaria ponderada por clase + penalización L2 | Es la log-verosimilitud negativa de un modelo Bernoulli (Goodfellow et al., 2016); los pesos de clase compensan el leve desbalance (44 % C en entrenamiento) |
| **Regularización** | L2 sobre los pesos y *dropout* entre capas (seleccionados en §7.2) | Con ≈1–3 parámetros por observación el riesgo de sobreajuste es alto (Srivastava et al., 2014) |
| **Optimizador** | Adam, $\eta_0 = 10^{-3}$, lotes de 16 | Tasa adaptativa por parámetro, robusta con gradientes ruidosos de lotes pequeños (Kingma & Ba, 2015). El lote de 16 da ≈8 actualizaciones por época |
| **Control del entrenamiento** | Parada temprana (paciencia 80, se restauran los mejores pesos) + reducción de $\eta$ a la mitad tras 30 épocas sin mejora | Detiene el ajuste en el mínimo de la pérdida de validación (Prechelt, 1998) y afina la convergencia cerca del óptimo |

**Formulación.** Para una red con capas ocultas $h^{(1)}, \dots, h^{(L)}$:

$$
\mathbf{h}^{(\ell)} = \mathrm{ReLU}\left(\mathbf{W}^{(\ell)}\,\mathbf{h}^{(\ell-1)} + \mathbf{b}^{(\ell)}\right),\qquad
\hat{p} = \sigma\left(\mathbf{w}_{o}^{\top}\mathbf{h}^{(L)} + b_o\right),
$$

$$
\mathcal{L}(\boldsymbol\theta) = -\frac{1}{n}\sum_{i=1}^{n} \omega_{y_i}\left[y_i\log\hat{p}_i + (1-y_i)\log(1-\hat{p}_i)\right] + \lambda\sum_{\ell}\left\lVert\mathbf{W}^{(\ell)}\right\rVert_2^2,
\qquad \omega_c = \frac{n}{2\,n_c}.
$$

#### Celda de código 19: 7.1 Configuración y constructor del modelo (Keras, API funcional)  (fuente completa en U2_modelo_base_codigo.py)

## 7.2 Selección de capacidad y regularización por validación cruzada temporal

No fijamos la arquitectura a ojo. Buscamos en una rejilla pequeña y justificada el balance entre capacidad
(neuronas y capas) y regularización (L2 y dropout): cada configuración se entrena en los tres pliegues temporales
(validando en 2023, 2024 y 2025) con dos semillas, y se mide la **entropía cruzada de validación**. Usamos la
entropía cruzada y no la exactitud porque evalúa la probabilidad completa, no solo el lado del umbral en que cae.

Para elegir aplicamos la **regla de un error estándar** (Hastie et al., 2009): entre las configuraciones cuyo error
medio no supera al mejor en más de un error estándar, se escoge la de **menos parámetros**. Con muestras pequeñas,
diferencias menores que el ruido entre semillas y pliegues no justifican una red más grande.

El conjunto de prueba (2026) no participa en esta búsqueda.

#### Celda de código 20: 7.2 Bucle de entrenamiento (se define aquí porque la búsqueda lo usa; se explica en §8)  (fuente completa en U2_modelo_base_codigo.py)

#### Celda de código 21: 7.3 Búsqueda de capacidad y regularización (3 pliegues temporales × 2 semillas)  (fuente completa en U2_modelo_base_codigo.py)
Salida de la celda:
```text
96 entrenamientos en 120 s
 | capas | L2 | dropout | BCE_media | BCE_ee | AUC_media | epoca_mediana | parámetros
0 | (32, 16) | 0,0010 | 0,0000 | 0,2848 | 0,0343 | 0,9717 | 83,0000 | 801
1 | (32, 16) | 0,0100 | 0,0000 | 0,2965 | 0,0290 | 0,9710 | 164,5000 | 801
2 | (16, 8) | 0,0010 | 0,0000 | 0,3142 | 0,0296 | 0,9546 | 207,5000 | 273
3 | (16, 8) | 0,0100 | 0,0000 | 0,3143 | 0,0293 | 0,9578 | 276,0000 | 273
4 | (16,) | 0,0010 | 0,0000 | 0,3160 | 0,0368 | 0,9654 | 298,5000 | 145
5 | (32, 16) | 0,0010 | 0,2000 | 0,3182 | 0,0394 | 0,9565 | 131,0000 | 801
6 | (8,) | 0,0010 | 0,0000 | 0,3239 | 0,0254 | 0,9638 | 501,5000 | 73
7 | (32, 16) | 0,0100 | 0,2000 | 0,3260 | 0,0333 | 0,9606 | 142,5000 | 801
8 | (16,) | 0,0100 | 0,0000 | 0,3276 | 0,0335 | 0,9646 | 460,0000 | 145
9 | (16,) | 0,0010 | 0,2000 | 0,3293 | 0,0254 | 0,9546 | 308,0000 | 145
10 | (16, 8) | 0,0100 | 0,2000 | 0,3302 | 0,0230 | 0,9545 | 242,0000 | 273
11 | (8,) | 0,0100 | 0,0000 | 0,3372 | 0,0233 | 0,9653 | 391,5000 | 73
12 | (8,) | 0,0010 | 0,2000 | 0,3378 | 0,0328 | 0,9586 | 366,0000 | 73
13 | (16,) | 0,0100 | 0,2000 | 0,3381 | 0,0225 | 0,9479 | 447,0000 | 145
14 | (8,) | 0,0100 | 0,2000 | 0,3473 | 0,0344 | 0,9553 | 536,5000 | 73
15 | (16, 8) | 0,0010 | 0,2000 | 0,3519 | 0,0257 | 0,9393 | 265,0000 | 273

Mejor BCE media: (32, 16) L2=0.001 dropout=0.0 -> 0.2848 ± 0.0343
Configuraciones dentro de 1 EE del mejor: 6
ELEGIDA (regla 1-EE, menos parámetros): ConfigMLP(capas=(16,), l2=0.001, dropout=0.0, lr=0.001, batch=16, epocas_max=800, paciencia=80, paciencia_lr=30, factor_lr=0.5, lr_min=1e-05, min_delta=0.0001)
```

#### Celda de código 22: 7.4 Visualización de la búsqueda: capacidad vs. regularización  (fuente completa en U2_modelo_base_codigo.py)
Figuras generadas: fig07_busqueda_hiperparametros (imagen omitida en este texto).

**Lectura de la búsqueda.**

* La mejor BCE media la obtuvo la red más grande, (32, 16) con L2 = 10⁻³ y sin dropout (0,285 ± 0,034). Pero
  **seis configuraciones** quedan dentro de un error estándar de ella: las diferencias entre capacidades son del
  mismo orden que la variación entre pliegues y semillas.
* La regla de 1 EE elige la más simple de ese grupo: **una capa oculta de 16 neuronas ReLU, L2 = 10⁻³, sin dropout
  (145 parámetros)**, con BCE 0,316 ± 0,037, frente a los 801 parámetros de la mejor. Con 113 filas de entrenamiento
  eso son 1,3 parámetros por observación en lugar de 7.
* Una sola capa oculta basta en teoría. Por el teorema de aproximación universal (Hornik, 1991), el mismo argumento
  que la Etapa 1 citó a partir de Bishop & Bishop (2024), una capa oculta suficientemente ancha puede aproximar
  cualquier frontera continua. La profundidad adicional no mostró ventaja estadística con estos datos.
* **El dropout empeoró la BCE en las 8 comparaciones pareadas** (misma capacidad y L2). En redes tan pequeñas,
  apagar el 20 % de 8–32 neuronas quita capacidad útil en lugar de evitar coadaptaciones. La regularización efectiva
  viene del L2 y, sobre todo, de la parada temprana (§11.3d).
* Todas las configuraciones superan a la regresión logística en esta misma validación cruzada (BCE media 0,412 con
  su mejor $C$). Es una primera señal a favor de la red, que §11.4 somete a prueba formal.

#### Celda de código 23: 7.5 Modelo base final: resumen de Keras y conteo de parámetros  (fuente completa en U2_modelo_base_codigo.py)
Salida de la celda:
```text

Parámetros entrenables: 145 | Filas de entrenamiento: 113 | Parámetros por observación: 1.28
 | Valor
Hiperparámetro | 
capas | (16,)
l2 | 0.001
dropout | 0.0
lr | 0.001
batch | 16
epocas_max | 800
paciencia | 80
paciencia_lr | 30
factor_lr | 0.5
lr_min | 1e-05
min_delta | 0.0001
```

#### Celda de código 24: 7.6 Esquema de la arquitectura  (fuente completa en U2_modelo_base_codigo.py)
Figuras generadas: fig08_arquitectura (imagen omitida en este texto).
