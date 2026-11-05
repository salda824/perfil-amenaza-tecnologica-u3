# Graph Report - Proyecto_Unidad_3  (2026-11-05)

## Corpus Check
- 37 files · ~123,691 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 1150 nodes · 2463 edges · 65 communities (63 shown, 2 thin omitted)
- Extraction: 94% EXTRACTED · 6% INFERRED · 0% AMBIGUOUS · INFERRED: 155 edges (avg confidence: 0.86)
- Token cost: 598,269 input · 44,000 output

## Community Hubs (Navigation)
- Módulo modelo.py (redes y calibración)
- Código completo de la U2
- Módulo pipeline.py (orquestación)
- Guía U3: actividades y rúbrica
- Módulo datos.py (carga y panel)
- Módulo evaluacion.py (métricas y particiones)
- Generador del informe (python-docx)
- Correcciones de la revisión en el PDF
- Estructura del proyecto (CONTEXTO)
- Informe U1: formulación y EDA
- Documentos y entregables del repo
- Informe U2: diagnóstico y hoja de ruta
- Modelo final, rivales y bitácora
- Datos ACLED y artefactos U2
- Conclusiones técnicas de la U2
- Pruebas automáticas (pytest)
- Revisión: declaración de IA y redacción
- Revisión: errores de código y regla 8
- Scripts del cuaderno U3
- Variables x1–x7 de la U1
- Intervenciones I1–I9 (cuaderno)
- Perfiles, ACLED y bandas de decisión
- Fuentes externas y variable x3
- Trayectoria U1 → U2 → U3
- Generación de figuras
- Artefactos y figuras de la U2
- Modelo base U2 y sus métricas
- Del diagnóstico U2 al modelo U3
- Entrenamiento del MLP en la U2
- Fundamentos del MLP base
- Etiqueta C vs. D y panel
- Módulo variables.py (predictoras)
- Guías U1–U2 y fuentes de coca
- Hipótesis de no linealidad (U1)
- Datos, panel y etiqueta (U3)
- Revisión: comparaciones injustas
- Bitácora D1–D10 y convergencia
- Arquitectura y entrenamiento U3 (informe)
- Revisión: PSI y decisión 2027
- Revisión: pregunta orientadora y escala
- Rúbrica U2 y pregunta orientadora
- Perfiles A/B/C/D y decisión tecnológica
- Proyección 2027 y artefacto
- Equipo y trabajo colaborativo
- Búsqueda de hiperparámetros U2
- Priorización territorial 2027
- Paquete, PSI y recomendaciones
- Protocolo, prueba 2026 y errores
- Variables nuevas e importancia
- Configuración y preparación de UCDP
- Pipeline integrado y validación
- Reglas de evaluación y controles
- Estilo gráfico (graficos.py)
- Revisión: cuaderno y trabajo en equipo
- Controles de preparación de datos
- Persistencia calibrada y memoria m1
- Revisión: rival débil (persistencia)
- Prueba 2026 y limitaciones
- Cuaderno U1: carga de ACLED
- Cuaderno U2 y hoja de ruta M1–M4
- Revisión: validación anidada sesgada
- Revisión: validación espacio-temporal
- Actores armados UCDP (I3)
- Multitarea y trazabilidad U1–U3
- Regla del 50 % y diseño t−1

## God Nodes (most connected - your core abstractions)
1. `Correcciones aplicadas (fase 2)` - 57 edges
2. `Perfil de amenaza tecnológica por departamento: del modelo base al modelo integrado (texto U3)` - 39 edges
3. `Proyecto de Aplicación U2 – Etapa 2: modelo base entrenado` - 38 edges
4. `Proyecto de Aplicación U1 – Etapa 1: clasificación por perfil de amenaza tecnológica` - 37 edges
5. `Revisión estricta del Proyecto Integrador U3` - 36 edges
6. `Informe U3 Proyecto Integrador: modelo neuronal integrado para anticipar el perfil de amenaza tecnológica` - 31 edges
7. `informe/construir_informe.py` - 25 edges
8. `Modelo base MLP 7 → 16 ReLU → 1 sigmoide (145 parámetros)` - 24 edges
9. `Modelo final U3: ensamble de 10 MLP 9 → 16 ReLU → 1` - 22 edges
10. `C3 · Rival débil: la inercia calibrada empata o gana` - 20 edges

## Surprising Connections (you probably didn't know these)
- `Persistencia calibrada (logística sobre logit(m1))` --semantically_similar_to--> `Calibración Platt / temperatura y umbral (I8)`  [INFERRED] [semantically similar]
  docs/u3_texto/U3_integrador_texto.md → informe/Informe_U3_Proyecto_Integrador.pdf
- `Embedding de departamento` --semantically_similar_to--> `Persistencia (referencia p̂ = sC de t−1)`  [INFERRED] [semantically similar]
  revision/REVISION_U3.md → informe/Informe_U3_Proyecto_Integrador.pdf
- `Rúbrica de la actividad (5 criterios)` --semantically_similar_to--> `Rúbrica U3 (Actividad 5, 5 criterios)`  [INFERRED] [semantically similar]
  docs/planeacion_u3/Unidad3_Proyecto_Integrador_Andamiaje.md → CONTEXTO_PROYECTO.md
- `Chocó` --conceptually_related_to--> `Transiciones de perfil (cambio entre t−1 y t)`  [EXTRACTED]
  informe/Informe_U3_Proyecto_Integrador.pdf → CONTEXTO_PROYECTO.md
- `Valle del Cauca` --conceptually_related_to--> `Transiciones de perfil (cambio entre t−1 y t)`  [EXTRACTED]
  informe/Informe_U3_Proyecto_Integrador.pdf → CONTEXTO_PROYECTO.md

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Comparación del modelo base contra las tres líneas base** — docs_entregas_u2_notebooks_texto_u2_modelo_base_parte01_clasificacion_del_perfil_de_amenaza_tecn_modelo_base_mlp, docs_entregas_u2_notebooks_texto_u2_modelo_base_parte01_clasificacion_del_perfil_de_amenaza_tecn_linea_base_clase_mayoritaria, docs_entregas_u2_notebooks_texto_u2_modelo_base_parte01_clasificacion_del_perfil_de_amenaza_tecn_persistencia_linea_base, docs_entregas_u2_notebooks_texto_u2_modelo_base_parte01_clasificacion_del_perfil_de_amenaza_tecn_regresion_logistica [EXTRACTED 1.00]
- **Diagnóstico técnico: convergencia, estabilidad, generalización y no linealidad** — docs_entregas_u2_notebooks_texto_u2_modelo_base_parte02_8_entrenamiento_del_modelo_base_diagnostico_convergencia, docs_entregas_u2_notebooks_texto_u2_modelo_base_parte02_8_entrenamiento_del_modelo_base_estabilidad_semillas, docs_entregas_u2_notebooks_texto_u2_modelo_base_parte02_8_entrenamiento_del_modelo_base_validacion_temporal_origen_movil_2023_2026, docs_entregas_u2_notebooks_texto_u2_modelo_base_parte02_8_entrenamiento_del_modelo_base_validacion_espacial_departamentos, docs_entregas_u2_notebooks_texto_u2_modelo_base_parte02_8_entrenamiento_del_modelo_base_diagnostico_sesgo_varianza, docs_entregas_u2_notebooks_texto_u2_modelo_base_parte02_8_entrenamiento_del_modelo_base_pruebas_pareadas_mcnemar [EXTRACTED 1.00]
- **Fuentes de datos públicas que alimentan las predictoras x1–x7** — docs_entregas_u1_proyecto_aplicacion_u1_lads_jdsm_dataset_acled_colombia, docs_entregas_u1_proyecto_aplicacion_u1_lads_jdsm_simci_unodc, docs_entregas_u1_proyecto_aplicacion_u1_lads_jdsm_igac_colombia_en_mapas, docs_entregas_u1_proyecto_aplicacion_u1_lads_jdsm_dane_censo_2018, docs_entregas_u1_proyecto_aplicacion_u1_lads_jdsm_variables_predictoras_x1_x7 [EXTRACTED 1.00]
- **Integración de fuentes en el panel departamento–año** — docs_entregas_u2_notebooks_texto_u2_modelo_base_parte01_clasificacion_del_perfil_de_amenaza_tecn_dataset_acled, docs_entregas_u2_notebooks_texto_u2_modelo_base_parte01_clasificacion_del_perfil_de_amenaza_tecn_fuente_simci_odc, docs_entregas_u2_notebooks_texto_u2_modelo_base_parte01_clasificacion_del_perfil_de_amenaza_tecn_fuente_global_forest_watch, docs_entregas_u2_notebooks_texto_u2_modelo_base_parte01_clasificacion_del_perfil_de_amenaza_tecn_fuente_dane_ipm, docs_entregas_u2_notebooks_texto_u2_modelo_base_parte01_clasificacion_del_perfil_de_amenaza_tecn_panel_departamento_anio [EXTRACTED 1.00]
- **Modelos comparados contra el MLP en la Tabla 4** — docs_entregas_u2_proyecto_aplicacion_u2_lads_jdsm_linea_base_clase_mayoritaria, docs_entregas_u2_proyecto_aplicacion_u2_lads_jdsm_linea_base_persistencia, docs_entregas_u2_proyecto_aplicacion_u2_lads_jdsm_regresion_logistica_base, docs_entregas_u2_proyecto_aplicacion_u2_lads_jdsm_mlp_7_16_1, docs_entregas_u2_proyecto_aplicacion_u2_lads_jdsm_metricas_entrenamiento_validacion [EXTRACTED 1.00]
- **Protocolo de entrenamiento y selección del modelo base** — docs_entregas_u2_proyecto_aplicacion_u2_lads_jdsm_particion_temporal, docs_entregas_u2_proyecto_aplicacion_u2_lads_jdsm_busqueda_hiperparametros_origen_movil, docs_entregas_u2_proyecto_aplicacion_u2_lads_jdsm_regla_1_error_estandar, docs_entregas_u2_proyecto_aplicacion_u2_lads_jdsm_parada_temprana, docs_entregas_u2_proyecto_aplicacion_u2_lads_jdsm_bce_ponderada_l2, docs_entregas_u2_proyecto_aplicacion_u2_lads_jdsm_mlp_7_16_1 [EXTRACTED 1.00]
- **Secuencia formativa de actividades 1–5 de la Unidad 3** — docs_guias_guia_actividades_unidad3_completa_actividad_1_integracion_modelo_flujo, docs_guias_guia_actividades_unidad3_completa_actividad_2_preparacion_representacion_datos, docs_guias_guia_actividades_unidad3_completa_actividad_3_analisis_interpretativo_desempeno, docs_guias_guia_actividades_unidad3_completa_actividad_4_optimizacion_estrategica, docs_guias_guia_actividades_unidad3_completa_actividad_5_proyecto_integrador [EXTRACTED 1.00]
- **Protocolo experimental de la U3 (fijado antes de ver resultados)** — contexto_proyecto_validacion_origen_movil, contexto_proyecto_prueba_final_2026, contexto_proyecto_persistencia, contexto_proyecto_log_loss, contexto_proyecto_regla_de_retencion, contexto_proyecto_presupuesto_560_pasos, contexto_proyecto_controles_de_datos, contexto_proyecto_regla_8_objetivo_sobre_metrica [EXTRACTED 1.00]
- **Fuentes de datos que alimentan el panel departamento–año** — contexto_proyecto_acled, contexto_proyecto_simci_unodc, contexto_proyecto_global_forest_watch, contexto_proyecto_dane_ipm_2018, contexto_proyecto_ucdp_ged, contexto_proyecto_distancia_a_bogota, contexto_proyecto_panel_departamento_anio [EXTRACTED 1.00]
- **De la predicción a la decisión tecnológica** — contexto_proyecto_bandas_de_decision, contexto_proyecto_alerta_a_b, contexto_proyecto_isr, contexto_proyecto_c_uas, contexto_proyecto_deteccion_aei, revision_revision_u3_proyeccion_2027 [EXTRACTED 1.00]
- **Intervenciones retenidas que forman el modelo final (m1, d1, ensamble)** — docs_u3_texto_u3_integrador_texto_i1_memoria, docs_u3_texto_u3_integrador_texto_i2_dinamica_intra_anual, docs_u3_texto_u3_integrador_texto_i7_ensamble_semillas, docs_u3_texto_u3_integrador_texto_modelo_final_u3, docs_u3_texto_u3_integrador_texto_escalera_de_mejora [EXTRACTED 1.00]
- **Salvaguardas contra el sesgo de selección** — docs_u3_texto_u3_integrador_texto_prueba_final_2026, docs_u3_texto_u3_integrador_texto_regla_7_controles_verifican, docs_u3_texto_u3_integrador_texto_validacion_anidada, docs_u3_texto_u3_integrador_texto_origen_movil_2023_2025, docs_u3_texto_u3_integrador_texto_ref_cawley_talbot_2010 [INFERRED 0.85]
- **Pipeline integrado: paquete, pruebas, artefacto, deriva, tarjeta y proyección** — docs_u3_texto_u3_integrador_texto_paquete_perfil_amenaza, docs_u3_texto_u3_integrador_texto_modulo_pipeline_py, docs_u3_texto_u3_integrador_texto_pruebas_automaticas, docs_u3_texto_u3_integrador_texto_artefacto_keras, docs_u3_texto_u3_integrador_texto_psi_piso_de_ruido, docs_u3_texto_u3_integrador_texto_tarjeta_del_modelo, docs_u3_texto_u3_integrador_texto_proyeccion_2027 [EXTRACTED 1.00]
- **Fugas de información hacia la selección del modelo** — revision_revision_u3_c1_regla_8_presentada_como_preregistrada, revision_revision_u3_c4_validacion_anidada_sesgada_a_favor_de_la_vigente, revision_revision_u3_c5_validacion_espacial_que_mezcla_anos, revision_revision_u3_m6_mayor_exploratorio_con_la_prueba_2026, revision_revision_u3_m22_menor_tabla_7_config_py_y_seleccion_contaminada [INFERRED 0.85]
- **Rivales y referencias comparados frente al MLP U3** — revision_revision_u3_mlp_u3_ensamble_de_10_semillas, revision_revision_u3_persistencia_cruda, revision_revision_u3_persistencia_calibrada_rival, revision_revision_u3_logistica_solo_con_d1, revision_revision_u3_logistica_con_las_mismas_9_variables, revision_revision_u3_mlp_base_u2_semilla_42, revision_revision_u3_base_u2_con_ensamble_de_10_semillas [EXTRACTED 1.00]
- **Esquemas de validación del proyecto** — revision_revision_u3_seleccion_por_origen_movil_2023_2025, revision_revision_u3_prueba_final_2026, revision_revision_u3_validacion_espacial_groupkfold_de_5_grupos, revision_revision_u3_validacion_espacio_temporal, revision_revision_u3_validacion_anidada_de_un_solo_pliegue_externo [EXTRACTED 1.00]
- **Protocolo de selección y prueba del modelo** — informe_informe_u3_proyecto_integrador_validacion_origen_movil, informe_informe_u3_proyecto_integrador_prueba_2026, informe_informe_u3_proyecto_integrador_persistencia, informe_informe_u3_proyecto_integrador_log_loss, informe_informe_u3_proyecto_integrador_regla_retencion, informe_informe_u3_proyecto_integrador_bootstrap_mcnemar, informe_informe_u3_proyecto_integrador_configuracion_json [EXTRACTED 1.00]
- **Pipeline reproducible y artefactos de despliegue** — informe_informe_u3_proyecto_integrador_paquete_perfil_amenaza, informe_informe_u3_proyecto_integrador_pipeline_8_etapas, informe_informe_u3_proyecto_integrador_artefacto_keras, informe_informe_u3_proyecto_integrador_pruebas_pytest, informe_informe_u3_proyecto_integrador_tarjeta_modelo, informe_informe_u3_proyecto_integrador_psi, informe_informe_u3_proyecto_integrador_proyeccion_2027_csv, informe_informe_u3_proyecto_integrador_ejecutar_todo_bat [EXTRACTED 1.00]
- **Fuentes integradas en el panel departamento–año** — contexto_proyecto_acled, informe_informe_u3_proyecto_integrador_simci, informe_informe_u3_proyecto_integrador_global_forest_watch, informe_informe_u3_proyecto_integrador_ipm_dane, docs_u3_texto_u3_integrador_texto_ucdp_ged_26_1, informe_informe_u3_proyecto_integrador_igac, informe_informe_u3_proyecto_integrador_panel_departamento_anio [EXTRACTED 1.00]

## Communities (65 total, 2 thin omitted)

### Community 0 - "Módulo modelo.py (redes y calibración)"
Cohesion: 0.07
Nodes (15): _bce_np(), CalibracionPlatt, Calibrador, nll(), calibrar_adelante(), ConfigRed, construir_red(), empaquetar() (+7 more)

### Community 1 - "Código completo de la U2"
Cohesion: 0.06
Nodes (16): bootstrap_ic(), calcular_features(), ConfigMLP, construir_panel(), descargar_bosque(), descargar_coca(), dibujar_arquitectura(), folds_temporales() (+8 more)

### Community 2 - "Módulo pipeline.py (orquestación)"
Cohesion: 0.11
Nodes (20): k6 · Tarjeta del modelo desactualizada, persistencia_calibrada(), salidas_u3/TARJETA_MODELO.md, ajustar_predecir(), f(), alerta_ab(), cargar_artefacto(), categoria() (+12 more)

### Community 3 - "Guía U3: actividades y rúbrica"
Cohesion: 0.08
Nodes (31): Actividad 1: Integración del modelo en un flujo de trabajo de ciencia de datos, Actividad 2: Diseño técnico del proceso de preparación y representación de datos, Actividad 3: Análisis interpretativo del desempeño del modelo neuronal, Actividad 4: Optimización estratégica e integración final del modelo neuronal, Actividad 5: Proyecto Integrador: diseño, implementación y evaluación de un modelo neuronal aplicado, Aggarwal (2023). Neural networks and deep learning: A textbook, Ajustes estratégicos (arquitectura, activación, tasa de aprendizaje, regularización, hiperparámetros), Codificación de variables categóricas (+23 more)

### Community 4 - "Módulo datos.py (carga y panel)"
Cohesion: 0.12
Nodes (16): auditar(), bloques_trimestrales(), cargar_acled(), cargar_o_descargar(), cargar_ucdp(), construir_panel(), control_etiqueta_parcial(), _descargar_bosque() (+8 more)

### Community 5 - "Módulo evaluacion.py (métricas y particiones)"
Cohesion: 0.10
Nodes (17): bootstrap_ic(), comparar_pareado(), k3 · bootstrap_ic sin grupos por departamento, particiones_espacio_temporales(), bce(), comparar_pareado(), ece(), metricas() (+9 more)

### Community 6 - "Generador del informe (python-docx)"
Cohesion: 0.10
Nodes (16): _bordes(), caja(), _f(), figura(), _fijar(), h1(), h2(), hipervinculo() (+8 more)

### Community 7 - "Correcciones de la revisión en el PDF"
Cohesion: 0.10
Nodes (30): calibrar_adelante(), Correcciones aplicadas (fase 2), Dependencia parcial memoria × letalidad, ejecutar_todo.bat, Exploratorio de transiciones (Figura 1a), Géron (2017), Guo et al. (2017), I1: memoria m₁ (+22 more)

### Community 8 - "Estructura del proyecto (CONTEXTO)"
Cohesion: 0.09
Nodes (25): Cuaderno Proyecto_U3_Integrador.ipynb, Declaración de uso de IA (Claude, Anthropic), Perfil de amenaza tecnológica por departamento: del modelo base al modelo integrado (texto U3), Matriz rúbrica → evidencia, Métricas de apoyo (AUC, Brier, MCC, ECE), evaluacion.py (métricas, bootstrap, McNemar, regla de retención, PSI), graficos.py (estilo de figuras), Origen móvil 2023–2025 (selección, n = 87) (+17 more)

### Community 9 - "Informe U1: formulación y EDA"
Cohesion: 0.13
Nodes (22): Acuerdo de Paz con las FARC-EP (2016), Análisis exploratorio de datos (Figuras 1-3), Conflicto armado colombiano (ELN, disidencias FARC-EP, Clan del Golfo), Hallazgo crítico: ausencia de perfiles A y B como dominantes, Proyecto de Aplicación U1 – Etapa 1: clasificación por perfil de amenaza tecnológica, Inversión de modernización militar 2026 (Escudo Nacional Antidrones), Paro Nacional 2021, Aggarwal (2023). Neural networks and deep learning (+14 more)

### Community 10 - "Documentos y entregables del repo"
Cohesion: 0.14
Nodes (23): Artefacto perfil_amenaza_u3.keras, CONTEXTO_PROYECTO.md (documento maestro del proyecto), Cuaderno Proyecto_U3_Integrador.ipynb, Determinismo (TF_ENABLE_ONEDNN_OPTS=0 + enable_op_determinism + semillas fijas), ejecutar_todo.bat, Grafo de conocimiento (graphify-out), Informe U3 (PDF, Producto 1), Log-loss (métrica de decisión) (+15 more)

### Community 11 - "Informe U2: diagnóstico y hoja de ruta"
Cohesion: 0.12
Nodes (22): Análisis de errores y síntesis (Tabla 5), ¿Aporta la no linealidad? (McNemar, bootstrap, dependencia parcial), Conclusiones de la Etapa 2, Tabla 3: configuración del modelo base (He normal, Adam, lotes de 16), Convergencia y estabilidad (10 semillas, AUC 0,848 ± 0,024), Diagnóstico técnico del comportamiento del modelo, E1: memoria del perfil x8 = s^C del año anterior, E2: ensamble de 5 semillas (+14 more)

### Community 12 - "Modelo final, rivales y bitácora"
Cohesion: 0.15
Nodes (17): Bitácora de decisiones D1–D20, Contracción bayesiana Beta(2,2) (s̃ = (C + 2)/(n + 4)), d1: s̃_C de las últimas 26 semanas de t−1, Hoja de ruta M1–M8, I1: memoria (m1, m2, m3), I2: dinámica semestral (d1, d2, d3), I6: rejilla de capacidad (48 config.) y validación anidada, I7: ensamble de 10 redes (+9 more)

### Community 13 - "Datos ACLED y artefactos U2"
Cohesion: 0.13
Nodes (22): acled-data-export.csv (export semanal ACLED, 17.573 filas), datos_externos/bosque_departamento_anio.csv, datos_externos/coca_departamento_anio.csv, dataset_modelado_departamento_anio.csv (171 filas), Manifiesto de artefactos del proyecto (Unidad 2), Guia_Proyecto_U2_Avance.pdf (guía y rúbrica de la actividad), Auditoría de calidad del export ACLED, Dataset ACLED (export agregado semanal de Colombia) (+14 more)

### Community 14 - "Conclusiones técnicas de la U2"
Cohesion: 0.13
Nodes (19): estabilidad_semillas.csv, experimento_E1_memoria.csv, proyeccion_2027.csv, validacion_temporal_espacial.csv, Persistencia del perfil (línea base de inercia, 86 % de repetición), Alerta A/B (proporción de eventos con explosivos o drones > 2× promedio nacional), Diagnóstico sesgo–varianza (tamaño de muestra y capacidad), Experimento E1: memoria del perfil (x8 = s_C del año anterior) (+11 more)

### Community 15 - "Pruebas automáticas (pytest)"
Cohesion: 0.12
Nodes (9): bootstrap_ic(), regla_retencion(), base(), test_artefacto_guarda_estandarizacion(), test_contrato_datos(), test_contrato_rechaza_columna_faltante(), test_fuga_temporal(), test_regla_retencion_variante_determinista() (+1 more)

### Community 16 - "Revisión: declaración de IA y redacción"
Cohesion: 0.20
Nodes (21): Balance de clases, C2 · Declaración de IA incompleta, C6 · Enlace al código no verificable, Cobertura de la guía, CONTEXTO_PROYECTO.md, Declaración de uso de IA, Guía de Actividades de la Unidad 3, informe/construir_informe.py (+13 more)

### Community 17 - "Revisión: errores de código y regla 8"
Cohesion: 0.15
Nodes (17): C1 · Regla 8 presentada como preregistrada, configuracion.json, decidir(), decisiones_intervenciones.csv, escalera_mejora.csv, evaluar(), Figura 8 (pipeline), herramientas/cuaderno/parte_a.py (+9 more)

### Community 18 - "Scripts del cuaderno U3"
Cohesion: 0.12
Nodes (5): calibrar_adelante(), evaluar(), particiones_de(), persistencia_oos(), registrar_calibrado()

### Community 19 - "Variables x1–x7 de la U1"
Cohesion: 0.18
Nodes (19): DANE (Censo 2018), Dataset ACLED Colombia 2020-2026, IGAC Colombia en Mapas, PNUD Colombia (2022). Índice de pobreza multidimensional, Vizarreta Balbuena (2025). El empleo de drones en la guerra rusa ucraniana, Variable objetivo y: perfil de amenaza (categórica), Variables predictoras x1–x7, x1 Frecuencia total de eventos violentos (+11 more)

### Community 20 - "Intervenciones I1–I9 (cuaderno)"
Cohesion: 0.19
Nodes (18): Grupos armados activos x3 (UCDP), GRU y CNN 1D sobre trimestres, Comparación antes y después frente a los rivales (Tabla 6), Arauca, Errores sistemáticos en transiciones, Escalera de mejora (base U2 → m1 → d1 → ensamble), Etapa U2: Modelo base (MLP 7→16→1), Etapa U3: Integración (+10 more)

### Community 21 - "Perfiles, ACLED y bandas de decisión"
Cohesion: 0.12
Nodes (15): ACLED (Armed Conflict Location & Event Data), I4: vecindad espacial (contigüidad IGAC), I5: objetivo blando (s_C) ponderado por √eventos, I9: salida multitarea softmax A/B/C/D, Mapeo de subtipos ACLED a perfiles (U1) y mapeo alternativo, Perfiles de amenaza A/B/C/D (explosivos, drones, armas convencionales, mixto), Raleigh et al. (2010), IGAC (Instituto Geográfico Agustín Codazzi) (+7 more)

### Community 22 - "Fuentes externas y variable x3"
Cohesion: 0.14
Nodes (17): Arauca (departamento mal clasificado en la U2), DANE: IPM 2018 (x7), Global Forest Watch (cobertura arbórea, x5), Hansen et al. (2013), I3: x3 actores armados (UCDP), SIMCI–UNODC / Observatorio de Drogas de Colombia (hectáreas de coca, x4), Sundberg y Melander (2013), UCDP GED 26.1 (actores armados no estatales) (+9 more)

### Community 23 - "Trayectoria U1 → U2 → U3"
Cohesion: 0.15
Nodes (15): Pregunta orientadora (U1), Sustentación sincrónica (pendiente), U1 — Formulación, U2 — Modelo base, U3 — Proyecto Integrador, Video MP4 de 8–10 min (Producto 2, pendiente), Unidad 3 — Proyecto Integrador (andamiaje histórico), Mapeo criterio → secciones (planeación del equipo) (+7 more)

### Community 24 - "Generación de figuras"
Cohesion: 0.13
Nodes (6): _coma(), curva_base(), guardar(), sel(), m15 (menor) · Figuras con punto decimal y rótulos superpuestos, m4 (menor) · Figura 2: título en inglés

### Community 25 - "Artefactos y figuras de la U2"
Cohesion: 0.14
Nodes (17): configuracion.json (hiperparámetros, partición, semilla, umbral), figuras/ (23 figuras fig01–fig23), figuras_informe/ (8 figuras R1–R8), historial_entrenamiento.csv, Proyecto de Aplicación_U2_LADS&JDSM.pdf (informe técnico entregado), intervalos_confianza_prueba.csv, metricas_modelos.csv, modelo_base_fase1.keras (+9 more)

### Community 26 - "Modelo base U2 y sus métricas"
Cohesion: 0.15
Nodes (16): Entropía cruzada binaria ponderada por clase + L2, Bucle de entrenamiento explícito con tf.GradientTape, Búsqueda de hiperparámetros con validación temporal de origen móvil (16 configuraciones), Curvas de aprendizaje (BCE, exactitud, ROC-AUC; mejor época 45), Estabilidad de la logística por remuestreo (500 réplicas), Línea base: clase mayoritaria, Tabla 4: métricas por modelo y conjunto (fase 1 y fase 2), MLP 7→16→1 (Keras, ReLU, sigmoide, 145 parámetros) (+8 more)

### Community 27 - "Del diagnóstico U2 al modelo U3"
Cohesion: 0.22
Nodes (14): Bitácora de decisiones técnicas D11–D20, Curvas de aprendizaje base vs. final (validación 2025), Escalado por temperatura, Síntesis de intervenciones y escalera de mejora, I5: objetivo blando ponderado por eventos, I6: rejilla de capacidad, activación, L2 y presupuesto, I7: ensamble de semillas (K = 10), I8: calibración y umbral de decisión (+6 more)

### Community 28 - "Entrenamiento del MLP en la U2"
Cohesion: 0.12
Nodes (10): agregado(), bce(), comparar_pareado(), construir_mlp(), entrenar_fijo(), entrenar_mlp(), evaluar_particion(), importancia_perm() (+2 more)

### Community 29 - "Fundamentos del MLP base"
Cohesion: 0.12
Nodes (17): Activación ReLU, Entropía cruzada binaria ponderada por clase + L2, Inicialización He normal, Línea base de clase mayoritaria, Modelo base MLP 7 → 16 ReLU → 1 sigmoide (145 parámetros), Optimizador Adam (η₀ = 10⁻³, lotes de 16), Parada temprana (paciencia 80) y reducción de la tasa de aprendizaje, Pregunta orientadora de la Etapa 1 (adquisición tecnológica según perfil de amenaza) (+9 more)

### Community 30 - "Etiqueta C vs. D y panel"
Cohesion: 0.14
Nodes (16): Perfil C (armas convencionales y emboscadas), Etapa U1: Formulación, Etiqueta parcial de 33 semanas (2026), Etiqueta y_d,t = 1[sC ≥ 0,5], Frontera curva entre memoria y letalidad, Global Forest Watch (cobertura arbórea), IPM del DANE (pobreza multidimensional 2018), Panel departamento–año (171 filas etiquetables) (+8 more)

### Community 31 - "Módulo variables.py (predictoras)"
Cohesion: 0.19
Nodes (6): construir_dataset(), contraer(), filas_proyeccion(), panel_extendido(), secuencias(), _variables_de_un_anio()

### Community 32 - "Guías U1–U2 y fuentes de coca"
Cohesion: 0.16
Nodes (16): Declaración de uso de IA (Claude, NotebookLM), UNODC (2024). Monitoreo de territorios afectados por cultivos ilícitos 2023, SIMCI-UNODC (cultivos ilícitos), x4 Hectáreas de cultivos ilícitos, Cultivos de coca – MinJusticia / Datos Abiertos Colombia, Anexo: notebook ejecutado en Google Colab, Entregable: propuesta PDF de máximo 5 páginas o 10 diapositivas, Guía de actividad – Actividad 4: Etapa 1 del Proyecto de Aplicación (+8 more)

### Community 33 - "Hipótesis de no linealidad (U1)"
Cohesion: 0.17
Nodes (15): Colinealidad entre predictoras (distancia–IPM ρ = 0,79), Análisis exploratorio orientado al modelado, DANE, Pobreza multidimensional por departamentos 2018 (ECV 2018), Hipótesis de interacción bosque × distancia y coca × pobreza (Etapa 1 §6.2), Hipótesis de no linealidad: la frontera entre perfiles no es un hiperplano, Regresión logística (una sola neurona lineal), x6: distancia del centroide departamental a Bogotá (proxy), x7: índice de pobreza multidimensional (IPM 2018) (+7 more)

### Community 34 - "Datos, panel y etiqueta (U3)"
Cohesion: 0.17
Nodes (11): Panel departamento–año 2021–2026 (171 filas, 30 departamentos), ACLED (export semanal departamento × subtipo), Fuentes estructurales (SIMCI-ODC, Global Forest Watch, DANE, centroides), config.py (rutas, constantes, mapeo de subtipos, contigüidad), datos.py (cargar_acled, panel, UCDP, control 2026), variables.py (construir_dataset, predictoras en t-1), ACLED (2024). ACLED codebook, Gneiting y Raftery (2007): reglas de puntuación estrictamente propias (+3 more)

### Community 35 - "Revisión: comparaciones injustas"
Cohesion: 0.25
Nodes (16): Base U2 con ensamble de 10 semillas, herramientas/cuaderno/parte_c.py, M2 (mayor) · Contradicción sobre transiciones, M3 (mayor) · 'Antes' injusto: semilla 42 frente a ensamble, M4 (mayor) · Espacial: individual frente a ensamble, m6 (menor) · Tabla 6: negrita y 'antes' reentrenado, metricas_prueba_2026.csv, metricas_seleccion.csv (+8 more)

### Community 36 - "Bitácora D1–D10 y convergencia"
Cohesion: 0.16
Nodes (7): x3: número de grupos armados activos (pendiente U3), Análisis de errores (transiciones de perfil y casos cerca del umbral), Diagnóstico de convergencia (norma del gradiente, razón de actualización), Verificación del bucle explícito contra model.fit (lote completo), Bitácora de decisiones técnicas D1–D10, Declaración de uso de inteligencia artificial (pendiente de completar), Cuaderno U2 parte 3: trabajo colaborativo, bitácora, conclusiones y referencias

### Community 37 - "Arquitectura y entrenamiento U3 (informe)"
Cohesion: 0.18
Nodes (14): BCE ponderada por clase + L2 (λ = 10⁻³), Bucle explícito tf.GradientTape, Calibración Platt / temperatura y umbral (I8), Ensamble de 10 perceptrones multicapa, Informe U3 Proyecto Integrador: modelo neuronal integrado para anticipar el perfil de amenaza tecnológica, Optimizador Adam (η = 10⁻³, lote 16), Pesos de clase, Géron (2017). Hands-on machine learning with Scikit-Learn and TensorFlow (+6 more)

### Community 38 - "Revisión: PSI y decisión 2027"
Cohesion: 0.20
Nodes (14): PSI (índice de estabilidad poblacional), Alerta A/B, Anexo A: código de verificación, Bandas de decisión 0,3/0,7, bandas_decision.csv, Informe U2, M10 (mayor) · Sección de decisión 2027 incompleta, M11 (mayor) · PSI sobreinterpretado con n = 33 (+6 more)

### Community 39 - "Revisión: pregunta orientadora y escala"
Cohesion: 0.20
Nodes (15): ACLED, Brecha de escala municipio–departamento, Informe U1, M1 (mayor) · Pregunta orientadora alterada, m22 (menor) · Tabla 7: config.py y selección contaminada, m2 (menor) · Umbral s_C ≥ 0,5, M9 (mayor) · Siglas y notación sin definir, Mapeo de subtipos A/B/C/D (+7 more)

### Community 40 - "Rúbrica U2 y pregunta orientadora"
Cohesion: 0.14
Nodes (13): Checklist de calidad del dataset, Perfiles de amenaza tecnológica A/B/C/D, Pregunta orientadora del proyecto, ACLED (2024). ACLED codebook, Alerta A/B (IED/drones) para Guaviare y Cauca, Proyección 2027 por departamento (10 C, 13 en transición, 10 D), Criterio: definición de variables (1 pt), Criterio: formulación del problema (1 pt) (+5 more)

### Community 41 - "Perfiles A/B/C/D y decisión tecnológica"
Cohesion: 0.20
Nodes (12): Cambio de prevalencia del perfil C entre años (Paro Nacional 2021, escalamiento 2025), De la predicción a la decisión: énfasis tecnológico por perfil (ISR, C-UAS, AEI), Mapeo de subtipos ACLED a perfiles (diccionario de la Etapa 1, por defecto → D), Perfil D (amenaza mixta), Perfiles A/B (IED/explosivos y drones), bajo volumen y alto impacto, Sensibilidad de la etiqueta a subtipos asignados por defecto (Shelling, Sexual violence), Variable objetivo y: perfil C vs. D (regla del 50 %), Calibración de probabilidades (+4 more)

### Community 42 - "Proyección 2027 y artefacto"
Cohesion: 0.24
Nodes (11): Artefacto único perfil_amenaza_u3.keras (152 KB), Énfasis tecnológico (ISR, C-UAS, detección de AEI), Figuras del informe R1–R5 (Anexo A), pipeline.py (orquestación de 8 etapas y línea de comandos), Proyección y priorización territorial 2027, salidas_u3/ (métricas, predicciones, figuras, modelos), Tarjeta del modelo (TARJETA_MODELO.md), Alerta A/B (explosivos o drones) (+3 more)

### Community 43 - "Equipo y trabajo colaborativo"
Cohesion: 0.23
Nodes (13): Curso Redes Neuronales (Universidad de La Sabana), Ministerio de Defensa (2025), Modernización militar de Colombia 2026 (3,7 billones; Escudo Nacional Antidrones, INDUMIL, CIAC), Clasificación del perfil de amenaza tecnológica de los territorios colombianos, Juan Diego Saldarriaga Muñoz, Laura Alejandra Díaz Sastoque, Trabajo colaborativo y organización del equipo (Etapa 1), Trabajo colaborativo y organización del equipo (Etapa 2) (+5 more)

### Community 44 - "Búsqueda de hiperparámetros U2"
Cohesion: 0.17
Nodes (8): busqueda_hiperparametros.csv, Búsqueda de capacidad y regularización (16 configuraciones, 3 pliegues × 2 semillas), Dropout, Regularización L2 (λ = 10⁻³), Validación cruzada temporal con origen móvil (2023, 2024, 2025), Bergmeir & Benítez (2012). Cross-validation for time series predictor evaluation, Hastie, Tibshirani & Friedman (2009). The elements of statistical learning, Srivastava et al. (2014). Dropout

### Community 45 - "Priorización territorial 2027"
Cohesion: 0.20
Nodes (8): C-UAS (sistemas contra drones), Detección de AEI/IED (artefactos explosivos improvisados), ISR (inteligencia, vigilancia y reconocimiento), Chocó, Meta, Valle del Cauca, Matriz de confusión de la prueba 2026, Priorización territorial 2027

### Community 46 - "Paquete, PSI y recomendaciones"
Cohesion: 0.23
Nodes (8): Arquitecturas alternativas (GRU, CNN 1D, MLP 16-8), Embedding de departamento (alternativa no adoptada), Paquete src/perfil_amenaza, PSI con piso de ruido (deriva), Recomendaciones técnicas R1–R7, Revisión estricta del informe (REVISION_U3.md), x2b letalidad, Pipeline planeado (S7, vacío principal)

### Community 47 - "Protocolo, prueba 2026 y errores"
Cohesion: 0.29
Nodes (6): Análisis de errores del modelo final, configuracion.json, Escaladas D → C de 2025 y desescaladas C → D de 2026, Prueba final 2026 (una sola vez), Transiciones de perfil (cambios de régimen), Embedding de departamento

### Community 48 - "Variables nuevas e importancia"
Cohesion: 0.27
Nodes (11): Importancia por permutación fuera de muestra, Calibración de Platt, CNN 1D trimestral, EDA de variables nuevas (Spearman e información mutua), GRU trimestral, I2: dinámica intra-anual, modelo.py (redes Keras, GradientTape, calibración, artefacto), Cho et al. (2014): codificador–decodificador RNN (GRU) (+3 more)

### Community 50 - "Pipeline integrado y validación"
Cohesion: 0.22
Nodes (10): Artefacto perfil_amenaza_u3.keras, configuracion.json (decisiones congeladas), ejecutar_todo.bat, Pipeline integrado de 8 etapas, proyeccion_2027_u3.csv, Prueba 2026 (evaluada una sola vez), Bergmeir y Benítez (2012). On the use of cross-validation for time series predictor evaluation, Mitchell et al. (2019). Model cards for model reporting (+2 more)

### Community 51 - "Reglas de evaluación y controles"
Cohesion: 0.24
Nodes (10): Bootstrap por departamento y prueba de McNemar, Controles experimentales de preparación de datos, Escalador con fuga (fuga deliberada), Etiquetas barajadas (control negativo), Log-loss (regla de puntuación propia), MIN_EVENTOS (umbral de eventos por celda = 10), Protocolo de evaluación (reglas 1–8), 10 pruebas automáticas (pytest) (+2 more)

### Community 53 - "Revisión: cuaderno y trabajo en equipo"
Cohesion: 0.22
Nodes (11): Bitácora D11–D20, Cuaderno integrador U3, herramientas/cuaderno/exportar_texto.py, herramientas/cuaderno/lecturas_fijas.py, herramientas/cuaderno/parte_d.py, M13 (mayor) · Trabajo en equipo insuficiente, m14 (menor) · Errores omitidos (Huila, Valle, Cesar), m24 (menor) · Paginación de §10 y Tabla 8 (+3 more)

### Community 54 - "Controles de preparación de datos"
Cohesion: 0.27
Nodes (6): Escalador con fuga (control negativo), Estandarización de variables, Etiquetas barajadas (control de cordura), Pesos de clase en la BCE, tests/test_pipeline.py (10 pruebas automáticas), Transformación logarítmica log(1+·)

### Community 55 - "Persistencia calibrada y memoria m1"
Cohesion: 0.27
Nodes (9): Experimento E1 de la U2 (memoria cruda x8), I1: memoria del perfil, Interacción memoria m1 × letalidad x2b (dependencia parcial 2D), Memoria del perfil m1 (s^C contraída en t-1), Persistencia calibrada (logística sobre logit(m1)), Friedman (2001): Greedy function approximation, Trayectoria m2, m3 (s^C en t-2 y cambio), Recomendaciones priorizadas R1–R7 (+1 more)

### Community 56 - "Revisión: rival débil (persistencia)"
Cohesion: 0.29
Nodes (10): C3 · Rival débil: la inercia calibrada empata o gana, k4 · Pruebas incompletas y docstring erróneo, k7 · README y CONTEXTO desactualizados, Logística solo con d₁, Persistencia calibrada (rival), Persistencia cruda, persistencia_oos(), README.md (+2 more)

### Community 57 - "Prueba 2026 y limitaciones"
Cohesion: 0.29
Nodes (5): CLAUDE.md (instrucciones para asistentes de IA), configuracion.json (modelo final, protocolo y decisiones), x6 distancia a Bogotá (proxy de presencia estatal), Limitaciones del modelo, Métricas del modelo y de los rivales (origen móvil 2023–2025 y prueba 2026)

### Community 58 - "Cuaderno U1: carga de ACLED"
Cohesion: 0.36
Nodes (7): Carga del dataset ACLED agregado (68.045 filas, 12 columnas), Columnas del dataset (week, admin1, event_type, sub_event_type, events, fatalities...), Conteo de subtipos de evento en Colombia (Peaceful protest 10.813, Attack 7.946, Armed clash 4.799), Cuaderno Proyecto_U1_Exploracion (texto extraído), Filtrado Colombia 2020-2026 (17.491 registros, 33 departamentos), Perfiles de amenaza por departamento (17 D, 16 C), M3: migrar a municipio

### Community 59 - "Cuaderno U2 y hoja de ruta M1–M4"
Cohesion: 0.32
Nodes (8): Proyecto_U2_Modelo_Base.ipynb (cuaderno ejecutado de la Unidad 2), Cuaderno U2 parte 1: clasificación del perfil de amenaza tecnológica por departamento, Juan Diego Saldarriaga Muñoz, Laura Alejandra Díaz Sastoque, U2_modelo_base_codigo.py (código completo del cuaderno), M1: memoria y dinámica del perfil, M4: completar x3 (actores armados) y x6 municipal, Distribución de responsabilidades en la Etapa 2

### Community 60 - "Revisión: validación anidada sesgada"
Cohesion: 0.39
Nodes (8): C4 · Validación anidada sesgada a favor de la vigente, Cawley y Talbot (2010), Configuración vigente de la U2, I6: rejilla de capacidad, m25 (menor) · Validación anidada mal nombrada, m7 (menor) · 0,359 frente a 0,377, rejilla_I6.csv, Validación anidada de un solo pliegue externo

### Community 61 - "Revisión: validación espacio-temporal"
Cohesion: 0.43
Nodes (8): C5 · Validación espacial que mezcla años, herramientas/cuaderno/lecturas.py, Logística con las mismas 9 variables, M12 (mayor) · Justificación débil del MLP, particiones_espaciales(), Script (d): validación espacial con bloqueo temporal, Validación espacio-temporal, validacion_espacio_temporal.csv

### Community 62 - "Actores armados UCDP (I3)"
Cohesion: 0.38
Nodes (6): Actores armados activos x3, Arauca (error estructural de la U2, frente ELN), I3: actores armados (UCDP), Sundberg y Melander (2013): UCDP Georeferenced Event Dataset, UCDP GED 26.1, Sundberg y Melander (2013). Introducing the UCDP Georeferenced Event Dataset

### Community 63 - "Multitarea y trazabilidad U1–U3"
Cohesion: 0.29
Nodes (7): Etapa 1 (U1): formulación, perfiles A/B/C/D y variables, I9: salida multitarea de composición A/B/C/D, Mapeo alternativo de subtipos (M8), Perfil de amenaza tecnológica C (convencional) vs. D (mixto), Pregunta orientadora de la U1 (inversión en seguridad y defensa), Ruder (2017): aprendizaje multitarea en redes profundas, Trazabilidad U1 → U2 → U3

### Community 64 - "Regla del 50 % y diseño t−1"
Cohesion: 0.33
Nodes (3): Raleigh et al. (2010). Introducing ACLED, Proporción de eventos C (s^C) y etiqueta y = 1[s^C ≥ 0,5], Criterio: coherencia entre problema y modelo implementado

## Ambiguous Edges - Review These
- `Proyecto de Aplicación U1 – Etapa 1: clasificación por perfil de amenaza tecnológica` → `Entregable: propuesta PDF de máximo 5 páginas o 10 diapositivas`  [AMBIGUOUS]
  docs/entregas/U1/Proyecto_Aplicacion_U1_LADS_JDSM.pdf · relation: references

## Knowledge Gaps
- **91 isolated node(s):** `Bucle de entrenamiento explícito con tf.GradientTape`, `Recomendaciones (interpretar métricas, evitar conclusiones descriptivas)`, `Acuerdo de Paz con las FARC-EP (2016)`, `Sector seguridad y defensa`, `DANE (Censo 2018)` (+86 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 235 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **2 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Perfil de amenaza tecnológica por departamento: del modelo base al modelo integrado (texto U3)` connect `Estructura del proyecto (CONTEXTO)` to `Datos, panel y etiqueta (U3)`, `Informe U1: formulación y EDA`, `Proyección 2027 y artefacto`, `Equipo y trabajo colaborativo`, `Protocolo, prueba 2026 y errores`, `Variables nuevas e importancia`, `Persistencia calibrada y memoria m1`, `Del diagnóstico U2 al modelo U3`, `Actores armados UCDP (I3)`, `Multitarea y trazabilidad U1–U3`?**
  _High betweenness centrality (0.163) - this node is a cross-community bridge._
- **Are the 3 inferred relationships involving `Proyecto de Aplicación U2 – Etapa 2: modelo base entrenado` (e.g. with `Contenido requerido del informe (recordatorio, dataset, arquitectura, métricas, curvas, diagnóstico, mejoras)` and `Entregable: informe técnico PDF de máximo 8 páginas o 12 diapositivas`) actually correct?**
  _`Proyecto de Aplicación U2 – Etapa 2: modelo base entrenado` has 3 INFERRED edges - model-reasoned connections that need verification._
- **What connects `Bucle de entrenamiento explícito con tf.GradientTape`, `Recomendaciones (interpretar métricas, evitar conclusiones descriptivas)`, `Acuerdo de Paz con las FARC-EP (2016)` to the rest of the system?**
  _91 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Módulo modelo.py (redes y calibración)` be split into smaller, more focused modules?**
  _Cohesion score 0.06763285024154589 - nodes in this community are weakly interconnected._
- **Why does `Panel departamento–año 2021–2026 (171 filas, 30 departamentos)` connect `Datos, panel y etiqueta (U3)` to `Perfiles A/B/C/D y decisión tecnológica`, `Bitácora D1–D10 y convergencia`, `Datos ACLED y artefactos U2`, `Conclusiones técnicas de la U2`?**
  _High betweenness centrality (0.138) - this node is a cross-community bridge._
- **Should `Código completo de la U2` be split into smaller, more focused modules?**
  _Cohesion score 0.05689900426742532 - nodes in this community are weakly interconnected._

### Low-confidence Hints
_AMBIGUOUS edges — the extractor was unsure. Verify before acting on these._

- **What is the exact relationship between `Proyecto de Aplicación U1 – Etapa 1: clasificación por perfil de amenaza tecnológica` and `Entregable: propuesta PDF de máximo 5 páginas o 10 diapositivas`?**
  _Edge tagged AMBIGUOUS (relation: references) - confidence is low._