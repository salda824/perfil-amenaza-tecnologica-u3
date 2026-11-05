> **Documento histórico (4-nov-2026), escrito antes de construir la U3.** El estado final y los resultados están en `CONTEXTO_PROYECTO.md` (raíz del repositorio).

# Plan del cuaderno U3 (Proyecto Integrador): documento de trabajo

Estado al 4-nov-2026: diseño definido, cuaderno sin construir. Nada de este plan es un resultado: cada mejora se conserva solo si la evidencia la respalda.

## Reglas de evaluación
- Selección solo con origen móvil 2023-2025. El año 2026 ya se miró en la U2: se declara y se reporta una sola vez, al final.
- Rival a vencer: persistencia (AUC temporal 0,965; log-loss 0,380), no el MLP base.
- Métricas: AUC, log-loss, Brier, MCC y aciertos en transiciones (la persistencia acierta 0 por construcción). Pruebas pareadas e IC bootstrap.
- Control de la etiqueta 2026 parcial (33 semanas): ¿cuánto coincide la etiqueta a la semana 33 con la del año completo en 2021-2025?

## Secuencia: diagnóstico U2 -> intervención
| Diagnóstico U2 | Intervención | Hipótesis |
|---|---|---|
| La inercia domina | I1 memoria: x8 = s_C(t-1), rezago t-2 y cambio | Se acerca a la persistencia (E1: AUC ,933) |
| Solo vista anual | I2 dinámica intra-anual: s_C del último semestre y tendencia, desde el export semanal | Mejora en transiciones |
| Arauca C los 6 años con P=0,36; falta x3 | I3 x3 actores activos (UCDP GED, si cubre el periodo) | Corrige departamentos con actor dominante |
| Sin información espacial | I4 vecindad: s_C y eventos de departamentos vecinos en t-1 | Mejora la validación espacial |
| 20 % de celdas a ±10 pp del umbral | I5 objetivo blando: BCE sobre s_C, ponderado por eventos | Menos ruido de etiqueta, mejor log-loss |
| Capacidad: 4 neuronas mejor que 16 a largo plazo; brecha sin cerrar | I6 capacidad, L2, tasa y presupuesto con validación anidada | Baja la varianza |
| AUC 0,848±0,024 entre semillas | I7 ensamble de semillas | Estabilidad |
| Subestima P(C); umbral F2 no transfiere | I8 calibración por temperatura y umbral multianual | Mejor Brier y umbral estable |
| A/B nunca dominantes | I9 salida de composición A/B/C/D (softmax, KL ponderada) | Alerta A/B modelada; responde a la pregunta de tecnología |

Comparativos: GRU o CNN 1D sobre trimestres, y embedding de departamento, frente a MLP y logística. Se justifica su descarte si no ganan.

## Controles de datos (criterio 2 de la rúbrica)
Ablación con y sin: estandarización, logaritmos, pesos de clase, escalador ajustado por pliegue (fuga simulada), MIN_EVENTOS 5/10/20, mapeo alternativo de subtipos, etiquetas barajadas (prueba de cordura).

## Pipeline (S7, vacío principal)
`Proyecto Unidad 3/`: `configuracion.json`, `src/perfil_amenaza/{datos,variables,modelo,evaluacion,pipeline}.py`, `tests/` (fuga temporal, contrato de datos, determinismo), `requirements.txt`, `README.md`, `salidas_u3/`, tarjeta del modelo. Ejecución única por línea de comandos, predicción con un solo artefacto y control de deriva (PSI).

## Cuaderno `Proyecto_U3_Integrador.ipynb` (secciones S1-S8)
Resumen ejecutivo; trazabilidad U1->U2->U3; datos y controles; línea base U2 reproducida; diagnóstico; intervenciones incrementales con curvas; modelo final; evaluación antes vs. después; análisis crítico y errores; pipeline; priorización territorial 2027 con tecnología sugerida; matriz rúbrica->evidencia; roles y declaración de IA.

## Pendiente del usuario
Tipo de créditos, ACLED por evento, export nuevo, repositorio GitHub, fecha y roles.

## Verificado el 4-nov-2026: UCDP para x3 (I3)
- Fuente: `https://ucdp.uu.se/downloads/ged/ged261-csv.zip` (archivo `GEDEvent_v26_1.csv`, 39 MB). Descarga comprobada.
- Colombia: datos hasta 2025, suficiente para x3 en t-1 de las etiquetas 2021-2026. adm_1 presente en 99,8 % de los eventos desde 2020; 32 departamentos. Columnas útiles: `side_a`, `side_b`, `adm_1`, `year`.
- Eventos por año: 2020 147, 2021 172, 2022 173, 2023 422, 2024 479, 2025 575. Falta comprobar si el salto de 2023 es real o de codificación.
- Solo cubre violencia organizada letal: x3 es un indicador aproximado de actores activos. Excluir "Civilians" de `side_b` al contar actores y separar pares como "ELN, FARC-EMC".
- El candidato `GEDEvent_v26_0_8.csv` no sirve para x3: solo 206 eventos (24-abr a 31-ago-2026) y 151 con actor sin identificar (XXX100).

## Siguiente paso al retomar
1. Construir `src/perfil_amenaza/` reutilizando `Proyecto Unidad 2/notebooks_texto/U2_modelo_base_codigo.py`.
2. Reproducir las métricas del modelo base U2 antes de cambiar nada.
3. Ejecutar I1-I9 y los controles con origen móvil 2023-2025; después, escribir la interpretación con cifras reales.
