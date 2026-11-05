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
| Exactitud | 0.839 | 0.828 |

| Rivales | Origen móvil 2023-2025 | Prueba 2026 |
|---|---|---|
| Persistencia · ROC-AUC | 0.973 | 0.948 |
| Persistencia · Log-loss | 0.363 | 0.429 |
| Persistencia · Brier | 0.106 | 0.131 |
| Persistencia calibrada · ROC-AUC | 0.966 | 0.948 |
| Persistencia calibrada · Log-loss | 0.328 | 0.351 |
| Persistencia calibrada · Brier | 0.099 | 0.105 |

La persistencia calibrada (logística sobre logit(m1) entrenada con años anteriores) se agregó como rival tras la
revisión; no cambió el modelo adoptado. Frente a ella, la ventaja del modelo no está demostrada a escala departamental.

**Limitaciones.** 30 unidades espaciales; etiqueta 2026 con 33 semanas; x3 (actores, UCDP) evaluada y no retenida;
x6 es un proxy departamental; la prevalencia de C cambia entre años (32 % a 62 %); con la información parcial de
2026, d1 resume las semanas 8 a 33 y no el segundo semestre.

**Deriva.** PSI de la información 2026 frente al entrenamiento (mismos 30 departamentos): variables con PSI > 0,25: x2b_letalidad; por encima del percentil 95 del ruido de muestreo: x2b_letalidad.
**Mantenimiento.** Reentrenar cada año con la etiqueta cerrada, revisar el PSI con su piso de ruido y reevaluar la
calibración con encadenamiento hacia adelante (hoy no se calibra porque no mejora sin mirar el futuro).