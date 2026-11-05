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