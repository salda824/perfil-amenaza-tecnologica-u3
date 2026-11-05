EQUIPO = r"""
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
"""
