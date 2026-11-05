# CONTEXTO DEL PROYECTO: perfil de amenaza tecnológica por departamento (Redes Neuronales, U. de La Sabana)

> **Para una IA o una persona que llega sin contexto:** este documento resume TODO el proyecto (Unidades 1, 2 y 3):
> qué problema resuelve, qué datos usa, qué se decidió en cada etapa y por qué, qué resultados hay, dónde está cada
> archivo y qué falta. Léelo completo antes de tocar código o documentos. El grafo de conocimiento está en
> `graphify-out/` (abre `graph.html` o lee `GRAPH_REPORT.md`). Última actualización: 5 de noviembre de 2026 (tras la
> revisión de `revision/REVISION_U3.md`: persistencia calibrada, validación espacio-temporal y antes justo).

---

## 0. Cómo orientarse en 2 minutos

| Si necesitas… | Ve a… |
|---|---|
| Entender el proyecto completo | Este archivo (secciones 1–6) |
| Ver el mapa de conceptos y relaciones | `graphify-out/graph.html` (interactivo) y `graphify-out/GRAPH_REPORT.md` |
| El entregable principal de la U3 (código + análisis) | `Proyecto_U3_Integrador.ipynb` (ejecutado) o su versión texto `docs/u3_texto/U3_integrador_texto.md` |
| El informe PDF de la U3 (Producto 1) | `informe/Informe_U3_Proyecto_Integrador.pdf` (y `.docx` editable) |
| El código reutilizable | `src/perfil_amenaza/` (paquete Python) |
| Ejecutar todo | `ejecutar_todo.bat` (Windows) o las órdenes de la sección 7 |
| Las guías y rúbricas oficiales | `docs/guias/` |
| Lo que se entregó en la U1 y la U2 | `docs/entregas/U1/` y `docs/entregas/U2/` (PDF entregados, cuadernos y su texto) |
| Las cifras finales en CSV | `salidas_u3/` |
| Qué falta por hacer | Sección 11 |
| La revisión estricta del informe y las correcciones aplicadas | `revision/REVISION_U3.md` |

**Idioma.** Todo el proyecto está en español (código, comentarios, documentos). Mantenerlo así. Los números usan coma
decimal en los documentos ("0,304") y punto en el código.

---

## 1. Identidad del proyecto

| | |
|---|---|
| **Curso** | Redes Neuronales, Universidad de La Sabana (maestría/posgrado en Ciencia de Datos) |
| **Equipo** | Laura Alejandra Díaz Sastoque (0000288870) y Juan Diego Saldarriaga Muñoz (0000287391) |
| **Sector elegido** | Seguridad y defensa |
| **Nombre del proyecto** | Clasificación del perfil de amenaza tecnológica de los territorios colombianos para orientar la adopción de tecnologías militares |
| **Repositorio** | https://github.com/salda824/perfil-amenaza-tecnologica-u3 (privado; propietario `salda824` = Juan Diego) |
| **Etapas** | U1 (formulación, entregada) → U2 (modelo base, entregada) → U3 (Proyecto Integrador, cuaderno + informe listos; faltan video y sustentación) |
| **Roles** | Juan Diego: modelado, pipeline, demostración técnica. Laura: interpretación estratégica, documentación, informe, guion del video. Evaluación y decisiones metodológicas: conjuntas |

---

## 2. El problema

**Contexto.** Colombia ejecuta en 2026 la mayor inversión de modernización militar de su historia (3,7 billones de
pesos, +47 % frente a 2025: Escudo Nacional Antidrones, blindados de INDUMIL, drones de la CIAC; Ministerio de Defensa,
2025) sin un modelo que oriente **qué tecnología desplegar en qué territorio**. Riesgos: sobreinvertir (C-UAS caros
donde la amenaza es IED) o subinvertir (sin detectores de AEI donde los explosivos son el vector).

**Pregunta orientadora (U1):** *¿Cómo orientar la adquisición, transferencia de tecnología y fabricación de sistemas de
seguridad y defensa de las Fuerzas de Colombia en función del perfil de amenaza tecnológica predominante de los
territorios, según su actividad armada histórica, geografía operacional y economías ilícitas, de modo que se optimice
la inversión?*

**Perfiles (U1).** Un territorio tiene perfil X si más del 50 % de sus eventos ACLED son de ese tipo:
**A** explosivos/IED · **B** drones y ataques remotos · **C** armas convencionales y emboscadas · **D** mixto (ningún
tipo supera el 50 %). **Hallazgo de la U1:** ningún departamento es A ni B dominante, así que la clasificación
operativa es **binaria, C vs. D**, y A/B se tratan como una **alerta** aparte (bajo volumen, alto impacto).

**Formulación operativa (U2 y U3).** Para cada departamento *d* y año *t*:

```
s_C(d,t) = eventos de subtipos C / eventos totales        y(d,t) = 1[ s_C(d,t) >= 0,5 ]   (1 = C, 0 = D)
p̂(d,t) = f_θ( x(d,t-1) ) = P( y(d,t) = 1 | información hasta el cierre de t-1 )
```

El rezago t−1 → t evita la circularidad (las fatalidades de t vienen de los mismos ataques que definen C) y replica la
decisión real: el presupuesto de un año se asigna con lo observado hasta el cierre del anterior.

**De la predicción a la decisión:**

| Categoría | Regla | Énfasis tecnológico |
|---|---|---|
| C consolidado | P(C) ≥ 0,7 | ISR (inteligencia, vigilancia, reconocimiento), detección temprana, movilidad protegida |
| Transición | 0,3 ≤ P(C) < 0,7 | Portafolio flexible y monitoreo trimestral |
| D consolidado | P(C) < 0,3 | Orden público y capacidades mixtas |
| Alerta A/B | proporción A+B en 2025–2026 ≥ 2 × la nacional y ≥ 5 eventos A+B | + C-UAS y detección de AEI |

---

## 3. Datos

| Fuente | Qué aporta | Archivo en el repo | Notas |
|---|---|---|---|
| **ACLED** (Raleigh et al., 2010) | Eventos semanales por departamento y subtipo, fatalidades, centroides | `datos/acled-data-export.csv` (17.573 filas, 2019-12-28 → 2026-08-15) | Export gratuito **agregado a departamento** (sin municipio ni actores). Licencia académica: no redistribuir en público → repo privado |
| **SIMCI–UNODC / ODC** (Datos Abiertos, recurso `acs4-3wgp`) | Hectáreas de coca (x4) | `datos/externos/coca_departamento_anio.csv` | Censo hasta 2024; 2025–2026 arrastran el último valor |
| **Global Forest Watch** (Hansen et al., 2013) | Cobertura arbórea % (x5) | `datos/externos/bosque_departamento_anio.csv` | Sustituye a IGAC (sin tabla departamental anual) |
| **DANE** (2019) | IPM 2018 (x7) | En `src/perfil_amenaza/config.py` (`IPM_2018`) | Estático |
| **Centroides ACLED** | Distancia a Bogotá (x6, proxy de presencia estatal) | Calculada | Proxy departamental de "distancia a casco urbano" |
| **UCDP GED 26.1** (Sundberg y Melander, 2013) | Actores armados no estatales (x3) | `datos/externos/ucdp_colombia.csv` (16.481 eventos de Colombia) y `ucdp_actores_departamento_anio.csv` | El original pesa 39 MB; se redujo con `herramientas/preparar_ucdp.py`. El dominio ucdp.uu.se está bloqueado en la nube de Claude: el usuario lo descargó y subió |

**Panel.** 33 departamentos × 7 años (2020–2026) = 231 celdas. Se etiquetan las celdas 2021–2026 con **≥ 10 eventos**
(error estándar de s_C ≤ 0,16) → **171 filas, 30 departamentos** (82 C, 89 D). 2026 solo tiene **33 semanas**.

**Mapeo de subtipos (U1, sin cambios):** A = IED/landmine, Grenade, Suicide bomb, Chemical weapon; B = Air/drone strike;
C = Armed clash, Attack, Shooting, Abduction, Looting/property destruction, Disrupted weapons use; D = protestas,
disturbios, arrestos y todo subtipo no listado (por defecto). Mapeo alternativo (control): Shelling → B, Sexual
violence → C (cambia 1 de 171 etiquetas).

**Rasgos que condicionan todo:** (1) **inercia**: el 86 % de los departamentos repite perfil de un año a otro; (2) la
prevalencia de C cambia entre años (32 % en 2021 por el Paro Nacional, 62 % en 2025 por la ruptura de diálogos y el
Catatumbo, 48 % en 2026); (3) el 20 % de las celdas tiene s_C a ±10 pp del umbral; (4) solo 30 unidades espaciales.

---

## 4. Historia por etapa

### U1 — Formulación (entregada; `docs/entregas/U1/`)
Problema, pregunta, 7 variables (x1 eventos, x2 fatalidades, x3 grupos armados, x4 coca, x5 bosque, x6 distancia,
x7 IPM), horizonte 2020–2025, ACLED como dataset, justificación de redes neuronales frente a modelos lineales
(hipótesis de no linealidad e interacciones bosque × distancia y coca × pobreza). Hallazgos: el export gratuito solo
llega a departamento; ningún perfil A/B dominante (→ C vs. D); 33 filas no bastan (→ panel departamento–año).

### U2 — Modelo base (entregada; `docs/entregas/U2/`)
- **Modelo:** MLP 7 → 16 ReLU → 1 sigmoide (145 parámetros), L2 = 10⁻³, Adam 10⁻³, lote 16, parada temprana; elegido
  entre 16 configuraciones con validación cruzada temporal y la regla de 1 error estándar. Bucle `tf.GradientTape`
  verificado contra `model.fit` (diferencia 2×10⁻⁶).
- **Resultados:** AUC 0,89 en validación temporal 2023–2026 (log-loss 0,436); prueba 2026 AUC 0,83 (fase 2). Empata
  con la logística. **Pierde con la persistencia** (AUC 0,965; log-loss 0,380). Espacial (departamentos nuevos): AUC 0,80.
- **Diagnóstico:** converge estable; limitado por varianza (30 unidades); memoriza firmas departamentales; 57 % de los
  errores son transiciones; Arauca mal clasificada (C los 6 años, P ≈ 0,36; falta x3).
- **Experimentos:** E1 memoria x8 = s_C(t−1) (AUC 0,89 → 0,93; log-loss 0,36 < persistencia); E2 ensamble de 5 semillas.
- **Hoja de ruta M1–M8:** memoria (M1), ensamble (M2), municipio (M3), x3 y x6 (M4), calibración y umbral (M5), etiqueta
  por severidad (M6), ajuste de capacidad con validación anidada (M7), mapeo de subtipos (M8).
- **Decisiones D1–D10:** panel departamento–año; t−1 → t; ≥ 10 eventos; mapeo U1; x3 pendiente; partición temporal +
  origen móvil; regla 1 EE; bucle explícito; reajuste con 2021–2025; presupuesto fijo de 560 pasos (mediana de 10 semillas).

### U3 — Proyecto Integrador (esta etapa; raíz del repo)
Objetivo de la guía (Actividad 5, `docs/guias/Guia_Actividades_Unidad3_completa.pdf`, págs. 11–13): consolidar el
ciclo completo, optimizar con mejoras fundamentadas e integrar el modelo en un pipeline. Productos: (1) informe PDF de
máximo 12 páginas con código en repositorio; (2) video MP4 de 8–10 minutos. Detalle de resultados en la sección 6.

---

## 5. Protocolo experimental de la U3 (fijado antes de ver resultados)

1. **Selección con origen móvil 2023–2025:** para v en {2023, 2024, 2025} se entrena con t < v y se predice v
   → 87 predicciones fuera de muestra. Escalador ajustado solo con el entrenamiento de cada pliegue.
2. **2026 = prueba final, una sola vez**, con todo congelado en `configuracion.json`. (Se declara que la U2 ya había mirado 2026.)
3. **Rival = persistencia** (p̂ = s_C(t−1); AUC 0,973, log-loss 0,363 en 2023–2025; acierta 0 transiciones por construcción).
   Tras la revisión se agregó la **persistencia calibrada** (logística de 2 parámetros sobre logit(m1), entrenada solo con
   años anteriores; `pipeline.persistencia_calibrada`). Se agregó después de elegir el modelo y no lo cambió.
4. **Métrica de decisión: log-loss**; apoyo: AUC, Brier, ECE, MCC, aciertos en transiciones.
5. **Regla de retención:** mismas 5 semillas que la referencia; se retiene si ΔLL medio + 1 EE(ΔLL pareado) < 0 y ΔAUC > −0,01
   (`evaluacion.regla_retencion`; el EE es entre semillas). En I4 (hipótesis espacial) se aplica a la validación espacial.
6. **Presupuesto fijo de 560 pasos** (D10), salvo cuando el presupuesto es lo estudiado.
7. **Los controles de datos verifican, no seleccionan** (no cambian el modelo dentro de la iteración).
8. **Objetivo del proyecto sobre la métrica:** una variante que gana solo por parecerse a la persistencia (menos
   transiciones) o por no transferir a territorios nuevos se documenta pero no se adopta. *(Formalizada al encontrar el
   caso del embedding; se declara así en el cuaderno.)*

---

## 6. Resultados de la U3

### 6.1 Intervenciones (Δ log-loss media entre 5 semillas pareadas, origen móvil 2023–2025; referencia en cascada)

| # | Diagnóstico U2 | Intervención | Resultado | Decisión |
|---|---|---|---|---|
| I1 | Inercia domina | Memoria m1 = s̃_C(t−1) contraída; m2, m3 (t−2 y cambio) | m1: −0,036 (EE 0,009), AUC 0,921 → 0,939; m2/m3: +0,002 | **Retiene m1** |
| I2 | Solo vista anual | d1 = s̃_C de las últimas 26 semanas de t−1; d2, d3 (tendencias) | d1: −0,037 (EE 0,007), LL 0,310, AUC 0,951; d2/d3: +0,013 | **Retiene d1** |
| I3 | Arauca, falta x3 | x3 = log(1 + actores no estatales, UCDP) | +0,025 (EE 0,006), AUC −0,014 | No (redundante con fatalidades ρ = 0,77; bandas urbanas; salto de cobertura UCDP 2023) |
| I4 | Espacial débil | Vecindad v1, v2 (contigüidad IGAC) | espacial +0,009 (criterio de la hipótesis); temporal −0,004 | No |
| I5 | Celdas cerca del umbral | Objetivo blando (s_C) ± peso √eventos | +0,119 crudo; +0,068 calibrado | No |
| I9 | A/B diluidas | Salida multitarea softmax A/B/C/D (KL) | P(C) +0,001; composición A/B peor que persistencia (MAE 0,035 vs. 0,030) | No |
| – | Sin estructura temporal | GRU, CNN 1D (4 trimestres), MLP (16,8), embedding de depto., logística | +0,010; +0,014; 0,000; **−0,026**; −0,0035 (EE 0,0043; antes se reportaba +0,003 por un error en `decidir`) | No (embedding: regla 8) |
| I6 | Capacidad inestable | Rejilla 48 config. (4–32 neuronas × L2 × ReLU/tanh × 280/560/1120 pasos) + anidada | 31/48 dentro de 1 EE; lo elegido en 2023–24 da 0,411–0,423 en 2025 vs. 0,359 de la vigente (comparación sesgada a favor de la vigente: se eligió en la U2 con 2025) | Mantiene config. U2 (parsimonia) |
| I7 | Varianza entre semillas | Ensamble de 10 redes | LL 0,308 → 0,297 (esperado); banda 5–95 % de 0,053 a 0,009 | **Retiene** |
| I8 | Subestima P(C) | Platt / temperatura (hacia adelante y dejando un año fuera); umbral por año | Platt LL 0,298 → 0,302; umbral óptimo MCC 0,20 / 0,51 / 0,15 | No; umbral 0,5 + bandas |

**Contracción bayesiana** usada en todas las proporciones nuevas: s̃ = (C + 2)/(n + 4) (prior Beta(2,2), α = 4).

### 6.2 Modelo final
Ensamble de **10 MLP 9 → 16 ReLU → 1 sigmoide** (177 parámetros por red), variables `x1_eventos, x2_fatalidades,
x2b_letalidad, x4_coca, x5_bosque, x6_distancia, x7_ipm, m1_sC_t1, d1_sC_H2`, L2 = 10⁻³, Adam η = 10⁻³, lote 16, 560
pasos, BCE ponderada por clase, semillas 1–10, sin calibración, umbral 0,5 (+ bandas 0,3/0,7). Configuración completa
en `configuracion.json`. Artefacto: `salidas_u3/modelos/perfil_amenaza_u3.keras` (152 KB; incluye estandarización,
10 redes, promedio y capa de calibración identidad).

### 6.3 Antes vs. después (Tabla 6 del informe; `salidas_u3/antes_despues_*.csv`)

| Conjunto | Modelo | AUC | Log-loss | Brier | ECE | MCC | Exactitud | Transiciones |
|---|---|---|---|---|---|---|---|---|
| Origen móvil 2023–2025 (n = 87) | Persistencia | 0,973 | 0,363 | 0,106 | 0,173 | 0,795 | 0,897 | 0/9 |
| | Persistencia calibrada (rival tras la revisión) | 0,966 | 0,328 | 0,099 | 0,116 | 0,778 | 0,885 | 1/9 |
| | Antes: base U2, semilla 42 (entregado) | 0,913 | 0,407 | 0,132 | 0,084 | 0,659 | 0,816 | 2/9 |
| | Antes: base U2, ensamble de 10 | 0,933 | 0,376 | 0,120 | 0,094 | 0,644 | 0,816 | 3/9 |
| | Logística (mismas 9 variables) | 0,949 | 0,306 | 0,097 | 0,068 | 0,737 | 0,862 | 1/9 |
| | **Después: MLP U3** | 0,956 | **0,304** | **0,095** | 0,124 | 0,691 | 0,839 | 2/9 |
| Prueba 2026 (n = 29) | Persistencia | 0,948 | 0,429 | 0,131 | 0,207 | 0,755 | 0,862 | 0/4 |
| | Persistencia calibrada | 0,948 | **0,351** | 0,105 | 0,150 | 0,755 | 0,862 | 0/4 |
| | Antes: base U2, semilla 42 | 0,852 | 0,524 | 0,166 | 0,104 | 0,517 | 0,759 | 1/4 |
| | Antes: base U2, ensamble de 10 | 0,862 | 0,470 | 0,157 | 0,109 | 0,517 | 0,759 | 1/4 |
| | Logística | 0,929 | 0,383 | 0,129 | 0,114 | 0,531 | 0,759 | 0/4 |
| | **Después: MLP U3** | 0,900 | 0,380 | 0,121 | **0,074** | 0,672 | 0,828 | 0/4 |
| | *(alternativa no adoptada) MLP + embedding* | 0,933 | 0,314 | 0,100 | 0,123 | 0,812 | 0,897 | 1/4 |

- Después vs. antes (semilla 42): ΔLL −0,103 [−0,185; −0,033] en selección; −0,144 [−0,270; −0,033] en 2026.
  Después vs. antes justo (base con ensamble de 10): −0,072 [−0,114; −0,033] en selección; −0,090 [−0,162; −0,024] en 2026.
- Después vs. persistencia cruda: ΔLL −0,059 [−0,142; +0,028] (P(mejora) = 0,92) en selección; −0,049 [−0,162; +0,092]
  en 2026. La persistencia ordena y clasifica mejor (exactitud 0,897 vs. 0,839; McNemar 7 contra 2, p = 0,18).
- Después vs. **persistencia calibrada**: −0,024 [−0,096; +0,052] (P(mejora) = 0,74) en selección; **+0,029**
  [−0,055; +0,125] en 2026. La ventaja frente a la inercia es de calibración y no está demostrada a escala departamental.
- **Espacial** (5 grupos de departamentos, años ≤ 2025 mezclados): ensamble base 0,445 → **0,351**; red individual
  0,476 → 0,373; persistencia 0,418; logística 0,373. **Espacio-temporal** (mismos grupos, solo años anteriores; n = 87):
  U3 0,361, persistencia 0,363 (Δ −0,002 [−0,114; +0,130]), persistencia calibrada **0,332**, logística 0,416, base
  ensamble 0,476. En territorios nuevos el modelo empata con la inercia.
- **Bandas de decisión:** 2023–2025, 71 de 87 filas en banda consolidada con 66 aciertos (31/32 en C, 35/39 en D); 2026,
  24 de 29 con 21 aciertos (10/13 en C, 11/11 en D). Con la persistencia cruda, 42 de 87 quedan en transición.
- **Curvas:** BCE de validación 2025 en 560 pasos: 0,448 (base) → 0,377 (final); mínimo 0,360; 3× pasos → 0,441.
- **Controles (§7 del cuaderno):** sin estandarización LL 6,09 (no converge); etiquetas barajadas LL 0,815 / AUC 0,32;
  sin pesos de clase +0,015; mapeo alternativo +0,007; sin contracción +0,002; fuga del escalador −0,002; **sin
  logaritmos −0,012** (no anticipado, no se cambia: regla 7); MIN_EVENTOS 5/10/20 → 0,300/0,307/0,338.
- **Etiqueta parcial:** a la semana 33 coincide con la anual en el 94,9 % de 136 celdas (2021–2025).
- **Errores:** 19 en 2023–2026; 58 % son transiciones, 58 % cerca del umbral. Arauca corregida (P 0,77–0,85). Persisten
  escaladas D→C 2025, desescaladas C→D 2026 (Bolívar, Chocó, Magdalena, Valle), Huila 2023–2024 y Casanare (pocos eventos).
  Transiciones 2023–2026 (13): U3 2, base U2 semilla 42 3, base U2 ensamble 4, persistencia calibrada 1, persistencia 0.
- **Importancia (permutación OOS):** d1 0,115 ≫ bosque 0,049 ≈ m1 0,048 ≈ fatalidades 0,046 > IPM, letalidad, coca > distancia, eventos.
- **Deriva (PSI de la información 2026 de los 30 departamentos del entrenamiento vs. filas de entrenamiento):** letalidad
  0,45 (supera el p95 del ruido de muestreo, 0,37); m1 0,23 y d1 0,17 (franja de vigilancia, dentro del ruido). Con n = 30
  el p95 del PSI sin deriva está entre 0,21 y 0,39. Con la información parcial de 2026, d1 resume las semanas 8–33
  (en 2020–2025 difiere del semestral en 0,084 en promedio, correlación 0,88).

### 6.4 Proyección 2027 (`salidas_u3/proyeccion_2027_u3.csv`)
11 C consolidados (Guaviare 0,99, Nariño, Cauca, Caquetá, Putumayo, Antioquia, Amazonas*, Norte de Santander, Vaupés*,
Vichada, Cesar), 11 en transición (San Andrés*, Chocó, Valle, Arauca, Córdoba, Guainía*, Casanare, Huila, Meta, Sucre,
Magdalena), 11 D. Alerta A/B (A+B 2025–2026 ≥ 2 × nacional = 12,7 % y ≥ 5 eventos): **Guaviare (14,4 %) y Cauca
(13,4 %)**. Contradicen el perfil 2026: 3 (Chocó, Valle, Meta; U2: 10); mide cercanía a la inercia, no acierto.
(*menos de 10 eventos en 2026: requieren revisión experta)

### 6.5 Conclusiones clave (para el video y la sustentación)
1. La mejora vino de representar mejor los datos (memoria + semestre reciente); la arquitectura es la de la U2.
2. Frente al modelo de la U2 la mejora es clara (ensamble contra ensamble: 0,376 → 0,304 en selección; 0,470 → 0,380 en 2026).
3. Frente a la inercia cruda la ventaja es de calibración. Frente a una inercia calibrada no está demostrada a escala
   departamental (gana 0,024 en selección, pierde 0,029 en 2026, pierde en territorios nuevos: 0,361 vs. 0,332).
4. En departamentos nuevos el modelo transfiere mucho mejor que el de la U2, pero empata con la persistencia cuando se
   respeta el tiempo (0,361 vs. 0,363). R1 (municipio) se justifica por la varianza de 30 unidades, no por una
   transferencia ya demostrada.
5. Límites: acierta 2 de 13 transiciones (la U2, 3); la logística empata en el eje temporal; el embedding descartado fue
   mejor en 2026. La regla 8 se agregó al ver el embedding y se declara así.
6. La rejilla de hiperparámetros (58 filas de validación) no ofrece una alternativa estable; la comparación anidada
   favorece a la vigente porque se eligió en la U2 con 2025.

---

## 7. Cómo ejecutar

```bash
pip install -r requirements.txt                                  # Python 3.11–3.13; versiones fijadas (TF 2.21.0, Keras 3.15.1)
python -m pytest -q tests                                        # 10 pruebas (contrato y caso negativo, fuga, escalador, artefacto,
                                                                 #   persistencia calibrada, espacio-temporal, regla de retención, determinismo, réplica U2)
PYTHONPATH=src python -m perfil_amenaza.pipeline todo            # 8 etapas, < 1 min; escribe salidas_u3/
PYTHONPATH=src python -m perfil_amenaza.pipeline predecir --anio 2026
jupyter nbconvert --to notebook --execute --inplace Proyecto_U3_Integrador.ipynb   # ~20 min en CPU
python herramientas/cuaderno/exportar_texto.py                                       # docs/u3_texto/ desde el cuaderno ejecutado
python informe/generar_figuras.py && python informe/construir_informe.py           # informe (requiere LibreOffice para el PDF)
```
Windows: `ejecutar_todo.bat "C:\ruta\a\ged261-csv.zip"` hace todo (entorno con Python 3.13/3.12/3.11, dependencias, UCDP, pruebas,
cuaderno en una copia `Proyecto_U3_Integrador_ejecutado.ipynb`, proyección).
Determinismo: `TF_ENABLE_ONEDNN_OPTS=0` + `enable_op_determinism()` + semillas fijas → dos ejecuciones dan cifras idénticas.

---

## 8. Estructura del repositorio

```
CONTEXTO_PROYECTO.md          ← este documento (empieza aquí)
CLAUDE.md / AGENTS.md         instrucciones breves para asistentes de IA
README.md                     uso rápido
configuracion.json            modelo final, protocolo y decisiones (la escribe §8.1 del cuaderno)
Proyecto_U3_Integrador.ipynb  cuaderno ejecutado de la U3 (103 celdas, 16 secciones)
ejecutar_todo.bat             ejecución completa en Windows
requirements.txt · pytest.ini · .gitignore
src/perfil_amenaza/
  config.py      rutas, constantes, mapeo de subtipos, IPM 2018, contigüidad departamental (VECINOS)
  datos.py       cargar_acled (contrato), auditar, construir_panel, bloques_trimestrales, fuentes_externas, cargar_ucdp, control_etiqueta_parcial
  variables.py   construir_dataset (predictoras en t−1: BASE, MEMORIA, DINAMICA, VECINDAD, ACTORES; secuencias), contraer, filas_proyeccion
  modelo.py      ConfigRed, construir_red (mlp/multitarea/gru/cnn/embedding), entrenar (bucle GradientTape), Escalador, Calibrador, empaquetar (artefacto único)
  evaluacion.py  metricas, ece, bootstrap_ic (con grupos), regla_retencion, comparar_pareado (bootstrap por departamento + McNemar),
                 particiones_origen_movil/espaciales/espacio_temporales, validar, psi, psi_ruido, tabla_psi
  pipeline.py    Especificacion, entrenar_especificacion, persistencia, persistencia_calibrada, ejecutar (8 etapas),
                 cargar_artefacto, predecir_con_artefacto, CLI
  graficos.py    estilo y tabla()
tests/test_pipeline.py        10 pruebas automáticas
herramientas/preparar_ucdp.py reduce UCDP GED (39 MB) a Colombia
herramientas/cuaderno/        scripts que generan el cuaderno (código en parte_a–d.py; textos en lecturas*.py) y exportar_texto.py
revision/                     REVISION_U3.md: revisión estricta del informe (fase 1) y correcciones aplicadas (fase 2)
informe/                      Informe_U3_Proyecto_Integrador.pdf/.docx, generar_figuras.py, construir_informe.py, figuras/
datos/                        ACLED + externos (coca, bosque, UCDP)
salidas_u3/                   métricas, predicciones, decisiones, rejilla, controles, proyección, figuras, modelos/, TARJETA_MODELO.md
referencias_u2/               CSV de resultados de la U2 (para comparar y para la prueba de réplica)
docs/guias/                   guías y rúbricas oficiales (U1, U2, U3 completa)
docs/entregas/U1, U2/         PDF entregados, cuadernos .ipynb y su texto (notebooks_texto/)
docs/planeacion_u3/           plan y andamiaje previos a construir la U3 (históricos)
docs/u3_texto/                cuaderno U3 en texto (markdown con salidas) y su código en un .py
graphify-out/                 grafo de conocimiento (graph.html, graph.json, GRAPH_REPORT.md)
```

---

## 9. Rúbrica de la U3 (Actividad 5, 5 criterios × 1,0 = 5,0) y dónde está la evidencia

| Criterio (nivel Excelente) | Evidencia |
|---|---|
| Coherencia metodológica integral (problema–datos–modelo–conclusiones con trazabilidad) | Cuaderno §1, §2.4, §4, §5, §12, §13; informe §1, §9 |
| Preparación y representación de datos (transformaciones, partición y controles con impacto en el entrenamiento) | Cuaderno §2, §3, §7; informe §2 (Tablas 2–3, Figs. 1–2) |
| Calidad técnica del modelo y optimización (arquitectura, configuración documentada, curvas, intervención por diagnóstico) | Cuaderno §5, §6, §8; informe §3–§5 |
| Evaluación e interpretación (métricas pertinentes, errores, comparación experimental rigurosa) | Cuaderno §9, §10; informe §5–§6 |
| Entregables y comunicación ejecutiva (informe + sustentación, articulación del equipo) | Informe completo; cuaderno §11, §14–§16; video (pendiente) |

Rúbrica de la Actividad 4 (individual, misma unidad): coherencia diagnóstico–mejoras, fundamentación de decisiones,
comparación antes/después, integración en pipeline, trabajo en equipo/reporte (6 páginas máx.).

---

## 10. Bitácora de decisiones (D1–D20)
D1–D10: U2 (sección 4). D11 selección solo con origen móvil 2023–2025 y 2026 una vez · D12 persistencia como rival y
log-loss como métrica · D13 regla de retención de 1 EE pareado · D14 contracción α = 4 · D15 variables por cascada
I1–I4 · D16 mantener la capacidad de la U2 (validación anidada) · D17 ensamble de 10 · D18 calibración solo si mejora sin
mirar el futuro · D19 paquete + artefacto único `.keras` · D20 x3 con UCDP, evaluada y no retenida.

---

Tras la revisión (fase 2 de `revision/REVISION_U3.md`) se agregaron, sin cambiar el modelo ni usar 2026 para elegir:
la persistencia calibrada como rival, el ensamble de 10 semillas del modelo base como "antes" justo, la validación
espacio-temporal, la evaluación de las bandas de decisión y el PSI con piso de ruido. También se corrigió `decidir`
(la logística se comparaba con la semilla 1 de la referencia).

## 11. Pendientes y próximos pasos

- [ ] **Video MP4 de 8–10 min** (Producto 2): síntesis del problema (§1), decisiones técnicas (escalera, validación
  anidada, regla de retención), resultados (Tabla 6.3, espacial, 2026), conclusiones estratégicas (proyección 2027,
  alerta A/B, límites). Guion: Laura; demostración del pipeline (`ejecutar_todo.bat` o `pipeline todo`): Juan Diego.
- [ ] Sustentación sincrónica: ambos deben poder explicar cada decisión (D11–D20) y reproducir el proceso.
- [ ] Agregar al profesor como colaborador del repositorio (Settings → Collaborators) o publicar sin el CSV de ACLED.
- [ ] Revisar y, si hace falta, reescribir con palabras propias las interpretaciones del cuaderno y del informe
  (la declaración de IA del informe y del cuaderno dice: apoyo en la programación del pipeline y del cuaderno y en la
  revisión de la redacción y la organización del informe; la propuesta, la búsqueda y selección de datos, el análisis,
  las decisiones metodológicas y las conclusiones son del equipo).
- [ ] Fechas de entrega y sustentación: no figuran en la guía; confirmar en la plataforma del curso.
- Recomendaciones técnicas (informe §9, Tabla 7): R1 municipio con ACLED por evento; R2 x3 municipal (entradas/salidas
  de grupos); R3 señales adelantadas de transición; R4 reentrenar cada año + PSI; R5 revisar logaritmos y MIN_EVENTOS;
  R6 persistencia calibrada y logística como rivales permanentes; R7 reevaluar el embedding con 2026 cerrado.

---

## 12. Advertencias para quien continúe (lecciones aprendidas)

- **No usar 2026 para seleccionar nada.** Es la prueba final. Cualquier cambio de modelo debe evaluarse con origen móvil
  2023–2025 (o, cuando cierre 2026, con 2024–2026, y la prueba pasaría a 2027).
- **Rival correcto:** la persistencia cruda usa una proporción como probabilidad; compárese siempre también con la
  persistencia calibrada (`pipeline.persistencia_calibrada`) y con la validación espacio-temporal, no solo la espacial.
- **Texto del cuaderno:** `docs/u3_texto/` se regenera con `python herramientas/cuaderno/exportar_texto.py`, que copia el
  texto de las tablas tal como se muestra (sin convertir números).
- **Formato de números en las tablas HTML del cuaderno:** `graficos.tabla` usa coma decimal; si se parsean con
  `pandas.read_html`, pasar `decimal=',', thousands='.'`. (Un parseo sin eso hizo leer 6,09 como 0,609 una vez.)
- Las **interpretaciones** del cuaderno (`herramientas/cuaderno/lecturas.py`) son texto fijo con cifras: si los
  resultados cambian, hay que actualizarlas a mano.
- El cuaderno es **determinista**; diferencias en el 3.er decimal en otro hardware son normales.
- La nube de Claude no puede descargar de ucdp.uu.se (política de red) ni crear repositorios: el usuario debe hacerlo.
- La regla 8 se formalizó al ver el caso del embedding: decirlo siempre así (transparencia).
- El export de ACLED no tiene municipio ni actores; el registro por evento requiere acceso académico (M3/R1).

---

## 13. Glosario
**ACLED**: Armed Conflict Location & Event Data. **UCDP GED**: Uppsala Conflict Data Program, Georeferenced Event
Dataset. **ISR**: inteligencia, vigilancia y reconocimiento. **C-UAS**: sistemas contra drones. **AEI/IED**:
artefactos explosivos improvisados. **Persistencia**: predecir el perfil del año anterior. **Origen móvil**: validación
temporal entrenando con el pasado y prediciendo el año siguiente. **Log-loss/BCE**: entropía cruzada binaria.
**ECE**: error de calibración esperado. **MCC**: coeficiente de Matthews. **PSI**: índice de estabilidad poblacional
(< 0,1 estable; 0,1–0,25 vigilar; > 0,25 deriva). **Transición**: cambio de perfil entre t−1 y t.
**s̃_C**: proporción de eventos C con contracción bayesiana. **EE**: error estándar.
