# Graph Report - Proyecto_Unidad_3  (2026-11-05)

## Corpus Check
- 36 files · ~100,517 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 4 file(s) not represented in the graph (top: (none) 2, .bat 1, .ini 1)

## Summary
- 911 nodes · 1672 edges · 65 communities (55 shown, 10 thin omitted)
- Extraction: 89% EXTRACTED · 11% INFERRED · 0% AMBIGUOUS · INFERRED: 176 edges (avg confidence: 0.87)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- Datos y variables (código U3)
- Informe U2 y su rúbrica
- Modelos Keras y calibración
- Figuras y código del cuaderno
- Pipeline y especificaciones
- Generador del informe U3
- Fuentes y artefactos de datos U2
- Métricas y validación (código)
- Modelo final y tarjeta
- Documento maestro y etapas
- Problema y justificación (U1)
- Variables x1–x7 (U1)
- Informe U3 y productos
- Repositorio y pipeline U3
- Guía y rúbrica U3
- Protocolo y rival persistencia
- Exploración ACLED (U1)
- Panel, memoria y proyección U2
- Configuración y estilo gráfico
- Formulación y decisión
- Fases y artefactos U2
- Arquitectura del MLP base
- Pruebas y herramienta UCDP
- Intervenciones y modelo final
- README y paquete
- Hipótesis y EDA (U2)
- Límites, embedding y municipio
- Dinámica, vecindad y GRU
- Bucle de entrenamiento U3
- Diagnóstico de generalización U2
- Bitácora D1–D10 y partición
- Actividad 3 y análisis crítico
- Escalera de intervenciones
- Hoja de ruta M1–M8 y equipo
- Funciones del cuaderno U2
- Pipeline integrado y Actividad 1
- Antes vs. después
- Búsqueda y convergencia U2
- Memoria y letalidad
- Despliegue y bitácora D11–D20
- Guía U1 y fuentes de coca
- Diagnóstico y mejoras U2
- Calibración, bandas y alerta A/B
- Reproducibilidad y versiones
- Controles de datos y pruebas
- Actores armados UCDP (I3)
- Calidad del dataset (Act. 4)
- Entrenamiento MLP U2 (código)
- Trabajo colaborativo
- Métricas de decisión
- Validación U2 (código)
- No linealidad: red vs. logística
- Actividad 2: preprocesamiento
- Log-loss e importancia
- Descarga de fuentes externas
- Bootstrap de intervalos
- Cálculo de predictoras
- Construcción del panel
- Pliegues temporales
- Guardado de figuras
- Distancia geodésica
- Panel extendido
- Dependencia parcial 2D
- Formato de tablas
- Score de persistencia

## God Nodes (most connected - your core abstractions)
1. `Proyecto de Aplicación U2 – Etapa 2: modelo base entrenado` - 38 edges
2. `Informe U3: Modelo neuronal integrado para anticipar el perfil de amenaza tecnológica de los territorios colombianos` - 38 edges
3. `Proyecto de Aplicación U1 – Etapa 1: clasificación por perfil de amenaza tecnológica` - 37 edges
4. `Proyecto Integrador U3: perfil de amenaza tecnológica por departamento` - 32 edges
5. `Modelo base MLP 7 → 16 ReLU → 1 sigmoide (145 parámetros)` - 24 edges
6. `Modelo final: ensamble de 10 MLP 9 → 16 → 1` - 21 edges
7. `Validación de origen móvil 2023–2025` - 15 edges
8. `Tarjeta del modelo: perfil de amenaza tecnológica (U3)` - 15 edges
9. `Regla de retención: mejora > 1 EE pareado entre 5 semillas` - 15 edges
10. `Modelo final: ensamble de 10 MLP 9 → 16 ReLU → 1 (177 parámetros por red)` - 15 edges

## Surprising Connections (you probably didn't know these)
- `Producto 1: informe técnico final (PDF ≤ 12 págs.)` --semantically_similar_to--> `Informe U3 Proyecto Integrador (PDF, Producto 1)`  [INFERRED] [semantically similar]
  docs/planeacion_u3/Unidad3_Proyecto_Integrador_Andamiaje.md → CONTEXTO_PROYECTO.md
- `Producto 2: video de presentación ejecutiva` --semantically_similar_to--> `Video MP4 8–10 min (Producto 2)`  [INFERRED] [semantically similar]
  docs/planeacion_u3/Unidad3_Proyecto_Integrador_Andamiaje.md → CONTEXTO_PROYECTO.md
- `Rúbrica de la actividad (5 criterios)` --semantically_similar_to--> `Rúbrica U3 (Actividad 5, 5 criterios)`  [INFERRED] [semantically similar]
  docs/planeacion_u3/Unidad3_Proyecto_Integrador_Andamiaje.md → CONTEXTO_PROYECTO.md
- `Mantenimiento (reentrenar anual, recalibrar, revisar PSI)` --semantically_similar_to--> `Recomendaciones técnicas R1–R7`  [INFERRED] [semantically similar]
  salidas_u3/TARJETA_MODELO.md → CONTEXTO_PROYECTO.md
- `Estandarización con media y desviación solo del entrenamiento de cada pliegue` --semantically_similar_to--> `Escalamiento y normalización de datos`  [INFERRED] [semantically similar]
  informe/Informe_U3_Proyecto_Integrador.pdf → docs/guias/Guia_Actividades_Unidad3_completa.pdf

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Escalera de intervenciones U3 evaluadas con la regla de retención** — contexto_proyecto_intervencion_i1_memoria, contexto_proyecto_intervencion_i2_dinamica, contexto_proyecto_intervencion_i3_actores, contexto_proyecto_intervencion_i4_vecindad, contexto_proyecto_intervencion_i5_objetivo_blando, contexto_proyecto_intervencion_i6_capacidad, contexto_proyecto_intervencion_i7_ensamble, contexto_proyecto_intervencion_i8_calibracion, contexto_proyecto_intervencion_i9_multitarea, contexto_proyecto_regla_retencion_1ee, contexto_proyecto_origen_movil_2023_2025 [EXTRACTED 1.00]
- **Las 9 variables de entrada del modelo final** — contexto_proyecto_x1_eventos, contexto_proyecto_x2_fatalidades, contexto_proyecto_x2b_letalidad, contexto_proyecto_x4_coca, contexto_proyecto_x5_bosque, contexto_proyecto_x6_distancia_bogota, contexto_proyecto_x7_ipm, contexto_proyecto_m1_sc_t1, contexto_proyecto_d1_sc_h2, contexto_proyecto_modelo_final_ensamble_10_mlp [EXTRACTED 1.00]
- **De la predicción a la decisión tecnológica** — contexto_proyecto_categoria_c_consolidado, contexto_proyecto_categoria_transicion, contexto_proyecto_categoria_d_consolidado, contexto_proyecto_alerta_ab, contexto_proyecto_formulacion_operativa [EXTRACTED 1.00]
- **Intervenciones retenidas que forman el modelo final (m1, d1, ensamble)** — docs_u3_texto_u3_integrador_texto_i1_memoria, docs_u3_texto_u3_integrador_texto_i2_dinamica, docs_u3_texto_u3_integrador_texto_i7_ensamble, docs_u3_texto_u3_integrador_texto_modelo_final [EXTRACTED 1.00]
- **Integración final: paquete, pipeline, pruebas, artefacto, PSI y tarjeta** — docs_u3_texto_u3_integrador_texto_paquete_perfil_amenaza, docs_u3_texto_u3_integrador_texto_pipeline_cli, docs_u3_texto_u3_integrador_texto_pruebas_automaticas, docs_u3_texto_u3_integrador_texto_artefacto_keras, docs_u3_texto_u3_integrador_texto_psi_deriva, docs_u3_texto_u3_integrador_texto_tarjeta_modelo [EXTRACTED 1.00]
- **Protocolo experimental de la U3 (reglas fijadas antes de ver resultados)** — docs_u3_texto_u3_integrador_texto_origen_movil_2023_2025, docs_u3_texto_u3_integrador_texto_prueba_final_2026, docs_u3_texto_u3_integrador_texto_persistencia, docs_u3_texto_u3_integrador_texto_log_loss_metrica, docs_u3_texto_u3_integrador_texto_regla_retencion, docs_u3_texto_u3_integrador_texto_regla_7_controles, docs_u3_texto_u3_integrador_texto_regla_8_objetivo, docs_u3_texto_u3_integrador_texto_presupuesto_560_pasos [EXTRACTED 1.00]
- **Comparación del modelo base contra las tres líneas base** — docs_entregas_u2_notebooks_texto_u2_modelo_base_parte01_clasificacion_del_perfil_de_amenaza_tecn_modelo_base_mlp, docs_entregas_u2_notebooks_texto_u2_modelo_base_parte01_clasificacion_del_perfil_de_amenaza_tecn_linea_base_clase_mayoritaria, docs_entregas_u2_notebooks_texto_u2_modelo_base_parte01_clasificacion_del_perfil_de_amenaza_tecn_persistencia_linea_base, docs_entregas_u2_notebooks_texto_u2_modelo_base_parte01_clasificacion_del_perfil_de_amenaza_tecn_regresion_logistica [EXTRACTED 1.00]
- **Integración de fuentes en el panel departamento–año** — docs_entregas_u2_notebooks_texto_u2_modelo_base_parte01_clasificacion_del_perfil_de_amenaza_tecn_dataset_acled, docs_entregas_u2_notebooks_texto_u2_modelo_base_parte01_clasificacion_del_perfil_de_amenaza_tecn_fuente_simci_odc, docs_entregas_u2_notebooks_texto_u2_modelo_base_parte01_clasificacion_del_perfil_de_amenaza_tecn_fuente_global_forest_watch, docs_entregas_u2_notebooks_texto_u2_modelo_base_parte01_clasificacion_del_perfil_de_amenaza_tecn_fuente_dane_ipm, docs_entregas_u2_notebooks_texto_u2_modelo_base_parte01_clasificacion_del_perfil_de_amenaza_tecn_panel_departamento_anio [EXTRACTED 1.00]
- **Diagnóstico técnico: convergencia, estabilidad, generalización y no linealidad** — docs_entregas_u2_notebooks_texto_u2_modelo_base_parte02_8_entrenamiento_del_modelo_base_diagnostico_convergencia, docs_entregas_u2_notebooks_texto_u2_modelo_base_parte02_8_entrenamiento_del_modelo_base_estabilidad_semillas, docs_entregas_u2_notebooks_texto_u2_modelo_base_parte02_8_entrenamiento_del_modelo_base_validacion_temporal_origen_movil_2023_2026, docs_entregas_u2_notebooks_texto_u2_modelo_base_parte02_8_entrenamiento_del_modelo_base_validacion_espacial_departamentos, docs_entregas_u2_notebooks_texto_u2_modelo_base_parte02_8_entrenamiento_del_modelo_base_diagnostico_sesgo_varianza, docs_entregas_u2_notebooks_texto_u2_modelo_base_parte02_8_entrenamiento_del_modelo_base_pruebas_pareadas_mcnemar [EXTRACTED 1.00]
- **Fuentes de datos públicas que alimentan las predictoras x1–x7** — docs_entregas_u1_proyecto_aplicacion_u1_lads_jdsm_dataset_acled_colombia, docs_entregas_u1_proyecto_aplicacion_u1_lads_jdsm_simci_unodc, docs_entregas_u1_proyecto_aplicacion_u1_lads_jdsm_igac_colombia_en_mapas, docs_entregas_u1_proyecto_aplicacion_u1_lads_jdsm_dane_censo_2018, docs_entregas_u1_proyecto_aplicacion_u1_lads_jdsm_variables_predictoras_x1_x7 [EXTRACTED 1.00]
- **Protocolo de entrenamiento y selección del modelo base** — docs_entregas_u2_proyecto_aplicacion_u2_lads_jdsm_particion_temporal, docs_entregas_u2_proyecto_aplicacion_u2_lads_jdsm_busqueda_hiperparametros_origen_movil, docs_entregas_u2_proyecto_aplicacion_u2_lads_jdsm_regla_1_error_estandar, docs_entregas_u2_proyecto_aplicacion_u2_lads_jdsm_parada_temprana, docs_entregas_u2_proyecto_aplicacion_u2_lads_jdsm_bce_ponderada_l2, docs_entregas_u2_proyecto_aplicacion_u2_lads_jdsm_mlp_7_16_1 [EXTRACTED 1.00]
- **Modelos comparados contra el MLP en la Tabla 4** — docs_entregas_u2_proyecto_aplicacion_u2_lads_jdsm_linea_base_clase_mayoritaria, docs_entregas_u2_proyecto_aplicacion_u2_lads_jdsm_linea_base_persistencia, docs_entregas_u2_proyecto_aplicacion_u2_lads_jdsm_regresion_logistica_base, docs_entregas_u2_proyecto_aplicacion_u2_lads_jdsm_mlp_7_16_1, docs_entregas_u2_proyecto_aplicacion_u2_lads_jdsm_metricas_entrenamiento_validacion [EXTRACTED 1.00]
- **Fuentes de datos integradas en el panel departamento–año** — informe_informe_u3_proyecto_integrador_panel_departamento_anio, informe_informe_u3_proyecto_integrador_acled, informe_informe_u3_proyecto_integrador_simci_coca, informe_informe_u3_proyecto_integrador_global_forest_watch, informe_informe_u3_proyecto_integrador_ipm_dane, informe_informe_u3_proyecto_integrador_ucdp_ged [EXTRACTED 1.00]
- **Componentes del pipeline integrado reproducible U3** — informe_informe_u3_proyecto_integrador_pipeline_integrado_8_etapas, informe_informe_u3_proyecto_integrador_paquete_perfil_amenaza, informe_informe_u3_proyecto_integrador_configuracion_json, informe_informe_u3_proyecto_integrador_artefacto_perfil_amenaza_u3_keras, informe_informe_u3_proyecto_integrador_pruebas_pytest, informe_informe_u3_proyecto_integrador_tarjeta_del_modelo, informe_informe_u3_proyecto_integrador_ejecutar_todo_bat [EXTRACTED 1.00]
- **Secuencia formativa de actividades 1–5 de la Unidad 3** — docs_guias_guia_actividades_unidad3_completa_actividad_1_integracion_modelo_flujo, docs_guias_guia_actividades_unidad3_completa_actividad_2_preparacion_representacion_datos, docs_guias_guia_actividades_unidad3_completa_actividad_3_analisis_interpretativo_desempeno, docs_guias_guia_actividades_unidad3_completa_actividad_4_optimizacion_estrategica, docs_guias_guia_actividades_unidad3_completa_actividad_5_proyecto_integrador [EXTRACTED 1.00]

## Communities (65 total, 10 thin omitted)

### Community 0 - "Datos y variables (código U3)"
Cohesion: 0.08
Nodes (22): auditar(), bloques_trimestrales(), cargar_acled(), cargar_o_descargar(), cargar_ucdp(), construir_panel(), control_etiqueta_parcial(), _descargar_bosque() (+14 more)

### Community 1 - "Informe U2 y su rúbrica"
Cohesion: 0.11
Nodes (29): Entropía cruzada binaria ponderada por clase + L2, Bucle de entrenamiento explícito con tf.GradientTape, Búsqueda de hiperparámetros con validación temporal de origen móvil (16 configuraciones), Conclusiones de la Etapa 2, Tabla 3: configuración del modelo base (He normal, Adam, lotes de 16), Curvas de aprendizaje (BCE, exactitud, ROC-AUC; mejor época 45), E1: memoria del perfil x8 = s^C del año anterior, Hallazgo central: inercia del perfil (86 % repite) (+21 more)

### Community 2 - "Modelos Keras y calibración"
Cohesion: 0.09
Nodes (9): CalibracionPlatt, Calibrador, nll(), ConfigRed, construir_red(), empaquetar(), Escalador, _logit() (+1 more)

### Community 3 - "Figuras y código del cuaderno"
Cohesion: 0.08
Nodes (5): calibrar_adelante(), registrar_calibrado(), curva_base(), guardar(), sel()

### Community 4 - "Pipeline y especificaciones"
Cohesion: 0.13
Nodes (15): ajustar_predecir(), f(), alerta_ab(), cargar_artefacto(), categoria(), ejecutar(), fn(), _entradas() (+7 more)

### Community 5 - "Generador del informe U3"
Cohesion: 0.12
Nodes (13): _bordes(), caja(), figura(), _fijar(), h1(), h2(), lado_a_lado(), leyenda() (+5 more)

### Community 6 - "Fuentes y artefactos de datos U2"
Cohesion: 0.12
Nodes (23): acled-data-export.csv (export semanal ACLED, 17.573 filas), datos_externos/bosque_departamento_anio.csv, datos_externos/coca_departamento_anio.csv, dataset_modelado_departamento_anio.csv (171 filas), Manifiesto de artefactos del proyecto (Unidad 2), Guia_Proyecto_U2_Avance.pdf (guía y rúbrica de la actividad), Auditoría de calidad del export ACLED, Dataset ACLED (export agregado semanal de Colombia) (+15 more)

### Community 7 - "Métricas y validación (código)"
Cohesion: 0.12
Nodes (11): bce(), comparar_pareado(), ece(), metricas(), particiones_espaciales(), particiones_origen_movil(), por_particion(), psi() (+3 more)

### Community 8 - "Modelo final y tarjeta"
Cohesion: 0.14
Nodes (23): ACLED (Armed Conflict Location & Event Data), Contracción bayesiana s̃ = (C+2)/(n+4), d1_sC_H2 último semestre de t−1, DANE IPM 2018, Global Forest Watch (cobertura arbórea), Importancia por permutación OOS, m1_sC_t1 memoria contraída, Modelo final: ensamble de 10 MLP 9 → 16 → 1 (+15 more)

### Community 9 - "Documento maestro y etapas"
Cohesion: 0.16
Nodes (20): CONTEXTO_PROYECTO (documento maestro), Experimento E2: ensamble de 5 semillas, Hoja de ruta M1–M8, Informe U3 Proyecto Integrador (PDF, Producto 1), Pendientes y próximos pasos, Repositorio perfil-amenaza-tecnologica-u3 (privado), Rúbrica U3 (Actividad 5, 5 criterios), Sustentación sincrónica (+12 more)

### Community 10 - "Problema y justificación (U1)"
Cohesion: 0.14
Nodes (19): Acuerdo de Paz con las FARC-EP (2016), Conflicto armado colombiano (ELN, disidencias FARC-EP, Clan del Golfo), Proyecto de Aplicación U1 – Etapa 1: clasificación por perfil de amenaza tecnológica, Inversión de modernización militar 2026 (Escudo Nacional Antidrones), Aggarwal (2023). Neural networks and deep learning, Bishop & Bishop (2024). Deep learning: Foundations and concepts, Canal 26 (2026). La guerra en Ucrania enciende una alarma en Colombia, CNMH (2022). Presencia diferenciada del Estado colombiano (+11 more)

### Community 11 - "Variables x1–x7 (U1)"
Cohesion: 0.17
Nodes (19): DANE (Censo 2018), Dataset ACLED Colombia 2020-2026, IGAC Colombia en Mapas, PNUD Colombia (2022). Índice de pobreza multidimensional, Raleigh et al. (2010). Introducing ACLED, Variable objetivo y: perfil de amenaza (categórica), Variables predictoras x1–x7, x1 Frecuencia total de eventos violentos (+11 more)

### Community 12 - "Informe U3 y productos"
Cohesion: 0.14
Nodes (21): Producto 1: Informe técnico final (PDF, máx. 12 páginas), Producto 1: Informe técnico final (PDF, máx. 12 páginas), ACLED (export semanal Colombia), DANE (2019). Pobreza multidimensional por departamentos, Cobertura arbórea Global Forest Watch, Guo et al. (2017). On calibration of modern neural networks, Hansen et al. (2013). High-resolution global maps of forest cover change, Informe U3: Modelo neuronal integrado para anticipar el perfil de amenaza tecnológica de los territorios colombianos (+13 more)

### Community 13 - "Repositorio y pipeline U3"
Cohesion: 0.13
Nodes (21): Clasificación de perfil de amenaza C (convencional) vs. D (mixto), Declaración de uso de IA (Claude, de Anthropic), Dietterich (1998): pruebas estadísticas para comparar clasificadores, Proyecto Integrador U3: perfil de amenaza tecnológica por departamento, Etapa 1 (U1): formulación, variables y dataset, Etapa 2 (U2): modelo base entrenado, Géron (2017): Hands-on machine learning, Goodfellow, Bengio y Courville (2016): Deep learning (+13 more)

### Community 14 - "Guía y rúbrica U3"
Cohesion: 0.15
Nodes (16): Actividad 4: Optimización estratégica e integración final del modelo neuronal, Actividad 5: Proyecto Integrador: diseño, implementación y evaluación de un modelo neuronal aplicado, Ajustes estratégicos (arquitectura, activación, tasa de aprendizaje, regularización, hiperparámetros), Guía de actividades Unidad 3 (Redes Neuronales), Producto 2: Video de presentación ejecutiva (8–10 min, MP4), Reporte técnico de optimización e integración (máx. 6 páginas), Resultados de aprendizaje (RA) de la Unidad 3, Roles técnicos del equipo (modelado, evaluación, documentación, presentación) (+8 more)

### Community 15 - "Protocolo y rival persistencia"
Cohesion: 0.14
Nodes (9): Control de etiqueta parcial (semana 33), Controles de datos (ablaciones), MLP + embedding de departamento (alternativa no adoptada), Inercia del perfil (86 % repite), MLP base U2 7 → 16 ReLU → 1 sigmoide, Persistencia (rival), Plan del cuaderno U3 (documento histórico), Secciones S1–S8 del cuaderno planeado (+1 more)

### Community 16 - "Exploración ACLED (U1)"
Cohesion: 0.15
Nodes (16): Carga del dataset ACLED agregado (68.045 filas, 12 columnas), Columnas del dataset (week, admin1, event_type, sub_event_type, events, fatalities...), Conteo de subtipos de evento en Colombia (Peaceful protest 10.813, Attack 7.946, Armed clash 4.799), Cuaderno Proyecto_U1_Exploracion (texto extraído), Filtrado Colombia 2020-2026 (17.491 registros, 33 departamentos), Perfiles de amenaza por departamento (17 D, 16 C), Análisis exploratorio de datos (Figuras 1-3), Hallazgo crítico: ausencia de perfiles A y B como dominantes (+8 more)

### Community 17 - "Panel, memoria y proyección U2"
Cohesion: 0.18
Nodes (14): proyeccion_2027.csv, De la predicción a la decisión: énfasis tecnológico por perfil (ISR, C-UAS, AEI), Panel departamento–año 2021–2026 (171 filas, 30 departamentos), Perfil C (armas convencionales y emboscadas), Perfil D (amenaza mixta), Perfiles A/B (IED/explosivos y drones), bajo volumen y alto impacto, Persistencia del perfil (línea base de inercia, 86 % de repetición), Variable objetivo y: perfil C vs. D (regla del 50 %) (+6 more)

### Community 18 - "Configuración y estilo gráfico"
Cohesion: 0.12
Nodes (3): cargar_config(), _fmt(), tabla()

### Community 19 - "Formulación y decisión"
Cohesion: 0.15
Nodes (13): Alerta A/B (+ C-UAS y detección de AEI), C consolidado (P(C) ≥ 0,7): ISR y movilidad protegida, D consolidado (P(C) < 0,3): orden público, Transición (0,3 ≤ P(C) < 0,7): portafolio flexible, Mapeo de subtipos ACLED a perfiles A/B/C/D, Panel departamento–año (171 filas, 30 departamentos), Perfil A: explosivos/IED, Perfil B: drones y ataques remotos (+5 more)

### Community 20 - "Fases y artefactos U2"
Cohesion: 0.15
Nodes (17): configuracion.json (hiperparámetros, partición, semilla, umbral), figuras/ (23 figuras fig01–fig23), figuras_informe/ (8 figuras R1–R8), historial_entrenamiento.csv, Proyecto de Aplicación_U2_LADS&JDSM.pdf (informe técnico entregado), intervalos_confianza_prueba.csv, metricas_modelos.csv, modelo_base_fase1.keras (+9 more)

### Community 21 - "Arquitectura del MLP base"
Cohesion: 0.12
Nodes (17): Activación ReLU, Entropía cruzada binaria ponderada por clase + L2, Inicialización He normal, Línea base de clase mayoritaria, Modelo base MLP 7 → 16 ReLU → 1 sigmoide (145 parámetros), Optimizador Adam (η₀ = 10⁻³, lotes de 16), Parada temprana (paciencia 80) y reducción de la tasa de aprendizaje, Pregunta orientadora de la Etapa 1 (adquisición tecnológica según perfil de amenaza) (+9 more)

### Community 22 - "Pruebas y herramienta UCDP"
Cohesion: 0.14
Nodes (4): base(), test_contrato_datos(), test_fuga_temporal(), test_reproduce_u2()

### Community 23 - "Intervenciones y modelo final"
Cohesion: 0.15
Nodes (14): Calibración de Platt y umbral (I8), Cawley y Talbot (2010). On over-fitting in model selection, Ensamble de 10 perceptrones multicapa (9 → 16 ReLU → 1 sigmoide), Goodfellow, Bengio y Courville (2016). Deep learning, Alternativas secuenciales GRU y CNN 1D, Intervenciones I1–I9 (diagnóstico U2 → decisión), Kingma y Ba (2015). Adam: A method for stochastic optimization, Lakshminarayanan et al. (2017). Deep ensembles (+6 more)

### Community 24 - "README y paquete"
Cohesion: 0.19
Nodes (14): CLAUDE.md (instrucciones para asistentes de IA), configuracion.json, Proyecto_U3_Integrador.ipynb, Curso Redes Neuronales, Universidad de La Sabana, ejecutar_todo.bat, Juan Diego Saldarriaga Muñoz, Laura Alejandra Díaz Sastoque, Modernización militar de Colombia 2026 (3,7 billones) (+6 more)

### Community 25 - "Hipótesis y EDA (U2)"
Cohesion: 0.17
Nodes (15): Colinealidad entre predictoras (distancia–IPM ρ = 0,79), Análisis exploratorio orientado al modelado, DANE, Pobreza multidimensional por departamentos 2018 (ECV 2018), Hipótesis de interacción bosque × distancia y coca × pobreza (Etapa 1 §6.2), Hipótesis de no linealidad: la frontera entre perfiles no es un hiperplano, Regresión logística (una sola neurona lineal), x6: distancia del centroide departamental a Bogotá (proxy), x7: índice de pobreza multidimensional (IPM 2018) (+7 more)

### Community 26 - "Límites, embedding y municipio"
Cohesion: 0.16
Nodes (15): Análisis de errores 2023–2026, Caso Arauca (error estructural de la U2 corregido), Embedding de departamento (alternativa no adoptada), Hegre et al. (2021): ViEWS2020 alerta temprana, I3: actores armados x3 (rechazada), Migración a escala municipal (M3, R1), Script herramientas/preparar_ucdp.py, Recomendaciones R1–R7 (+7 more)

### Community 27 - "Dinámica, vecindad y GRU"
Cohesion: 0.13
Nodes (14): Cawley y Talbot (2010): sobreajuste en la selección de modelos, Cho et al. (2014): codificador–decodificador RNN (GRU), CNN 1D trimestral (comparativo), d1: composición del 2.º semestre de t-1, d2, d3: tendencias semestrales de volumen y composición, GRU trimestral (comparativo), I2: dinámica intra-anual (se retiene d1), I4: vecindad espacial (rechazada) (+6 more)

### Community 28 - "Bucle de entrenamiento U3"
Cohesion: 0.18
Nodes (6): bootstrap_ic(), _bce_np(), entrenar(), n_parametros(), predecir_ensamble(), RedEntrenada

### Community 29 - "Diagnóstico de generalización U2"
Cohesion: 0.17
Nodes (12): estabilidad_semillas.csv, experimento_E1_memoria.csv, validacion_temporal_espacial.csv, Diagnóstico sesgo–varianza (tamaño de muestra y capacidad), Experimento E2: ensamble de 5 semillas, Estabilidad frente a la inicialización (10 semillas), M2: ensamble de redes, M7: ajuste conjunto de capacidad, L2 y épocas con validación anidada (+4 more)

### Community 30 - "Bitácora D1–D10 y partición"
Cohesion: 0.14
Nodes (8): Mapeo de subtipos ACLED a perfiles (diccionario de la Etapa 1, por defecto → D), Sensibilidad de la etiqueta a subtipos asignados por defecto (Shelling, Sexual violence), Validación cruzada temporal con origen móvil (2023, 2024, 2025), M8: refinar el mapeo de subtipos, Bitácora de decisiones técnicas D1–D10, Declaración de uso de inteligencia artificial (pendiente de completar), Cuaderno U2 parte 3: trabajo colaborativo, bitácora, conclusiones y referencias, Bergmeir & Benítez (2012). Cross-validation for time series predictor evaluation

### Community 31 - "Actividad 3 y análisis crítico"
Cohesion: 0.15
Nodes (15): Actividad 3: Análisis interpretativo del desempeño del modelo neuronal, Desbalance de clases, sesgos y sobreajuste, Representación vectorial con embeddings básicos, Matriz de confusión como herramienta diagnóstica, Métricas de evaluación (accuracy, precision, recall, F1-score), Análisis crítico del desempeño, Embedding de departamento, Errores sistemáticos en transiciones de perfil (+7 more)

### Community 32 - "Escalera de intervenciones"
Cohesion: 0.33
Nodes (11): Comparativos GRU, CNN 1D, MLP (16,8), embedding, logística, Experimento E1: memoria x8 = s_C(t−1), I1 Memoria (m1 retenida), I2 Dinámica intra-anual (d1 retenida), I4 Vecindad (no retenida), I5 Objetivo blando (no retenida), I6 Capacidad con validación anidada (mantiene config. U2), I7 Ensamble de 10 redes (retenida) (+3 more)

### Community 33 - "Hoja de ruta M1–M8 y equipo"
Cohesion: 0.19
Nodes (13): Proyecto_U2_Modelo_Base.ipynb (cuaderno ejecutado de la Unidad 2), Cambio de prevalencia del perfil C entre años (Paro Nacional 2021, escalamiento 2025), Cuaderno U2 parte 1: clasificación del perfil de amenaza tecnológica por departamento, Juan Diego Saldarriaga Muñoz, Laura Alejandra Díaz Sastoque, U2_modelo_base_codigo.py (código completo del cuaderno), Calibración de probabilidades, Hoja de ruta priorizada de mejoras M1–M8 (Unidad 3) (+5 more)

### Community 34 - "Funciones del cuaderno U2"
Cohesion: 0.15
Nodes (3): ConfigMLP, dibujar_arquitectura(), n_parametros()

### Community 35 - "Pipeline integrado y Actividad 1"
Cohesion: 0.14
Nodes (13): Actividad 1: Integración del modelo en un flujo de trabajo de ciencia de datos, Aggarwal (2023). Neural networks and deep learning: A textbook, freeCodeCamp.org (2020). Keras with TensorFlow course [Video], Géron (2017). Hands-on machine learning with Scikit-Learn and TensorFlow, Grus (2024). Ciencia de datos desde cero, Alerta A/B (Guaviare y Cauca), Artefacto perfil_amenaza_u3.keras, ejecutar_todo.bat (+5 more)

### Community 36 - "Antes vs. después"
Cohesion: 0.17
Nodes (11): Bergmeir y Benítez (2012). Cross-validation for time series predictor evaluation, Comparación antes (MLP U2) vs. después (MLP U3 ensamble), configuracion.json (decisiones congeladas), Control de la etiqueta parcial (33 semanas, 94,9 % coincidencia), Dietterich (1998). Approximate statistical tests for comparing classifiers, Gneiting y Raftery (2007). Strictly proper scoring rules, Log-loss como métrica de decisión (regla de puntuación propia), Perfil de amenaza C (convencional) vs. D (amenaza mixta) (+3 more)

### Community 37 - "Búsqueda y convergencia U2"
Cohesion: 0.21
Nodes (8): busqueda_hiperparametros.csv, Búsqueda de capacidad y regularización (16 configuraciones, 3 pliegues × 2 semillas), Dropout, Regularización L2 (λ = 10⁻³), Diagnóstico de convergencia (norma del gradiente, razón de actualización), Verificación del bucle explícito contra model.fit (lote completo), Hastie, Tibshirani & Friedman (2009). The elements of statistical learning, Srivastava et al. (2014). Dropout

### Community 38 - "Memoria y letalidad"
Cohesion: 0.20
Nodes (11): ACLED (export semanal departamento × subtipo), Etiqueta y = 1[s^C ≥ 0,5] (regla del 50 %), Friedman (2001): dependencia parcial / gradient boosting, Fuentes externas SIMCI-ODC, Global Forest Watch y DANE, I1: memoria del perfil (se retiene m1), Interacción memoria m1 × letalidad x2b (dependencia parcial 2D), m1: memoria del perfil (s^C contraída en t-1), m2, m3: trayectoria de la memoria (t-2 y cambio) (+3 more)

### Community 39 - "Despliegue y bitácora D11–D20"
Cohesion: 0.20
Nodes (7): Bitácora de decisiones técnicas D11–D20, configuracion.json, Estandarización ajustada solo con entrenamiento, I7: ensamble de semillas (se retiene, 10 redes), Kingma y Ba (2015): Adam, Lakshminarayanan et al. (2017): ensambles profundos, Modelo final: ensamble de 10 MLP 9 → 16 ReLU → 1 (177 parámetros por red)

### Community 40 - "Guía U1 y fuentes de coca"
Cohesion: 0.20
Nodes (11): Declaración de uso de IA (Claude, NotebookLM), UNODC (2024). Monitoreo de territorios afectados por cultivos ilícitos 2023, SIMCI-UNODC (cultivos ilícitos), x4 Hectáreas de cultivos ilícitos, Cultivos de coca – MinJusticia / Datos Abiertos Colombia, Entregable: propuesta PDF de máximo 5 páginas o 10 diapositivas, Guía de actividad – Actividad 4: Etapa 1 del Proyecto de Aplicación, Recomendaciones para la entrega (problema medible, roles, IA solo de apoyo) (+3 more)

### Community 41 - "Diagnóstico y mejoras U2"
Cohesion: 0.20
Nodes (9): Análisis de errores y síntesis (Tabla 5), Convergencia y estabilidad (10 semillas, AUC 0,848 ± 0,024), Diagnóstico técnico del comportamiento del modelo, E2: ensamble de 5 semillas, Generalización temporal y espacial, Hoja de ruta de mejoras M1–M7 (Tabla 7), M3: migrar a municipio, Roberts et al. (2017). Cross-validation strategies for structured data (+1 more)

### Community 42 - "Calibración, bandas y alerta A/B"
Cohesion: 0.22
Nodes (9): Alerta A/B (explosivos y drones), Calibración de Platt, Énfasis tecnológico (ISR, C-UAS, detección de AEI), Guo et al. (2017): calibración de redes neuronales, I8: calibración Platt / temperatura y umbral (rechazada), I9: salida multitarea de composición A/B/C/D (rechazada), Platt (1999): salidas probabilísticas, Priorización territorial / proyección 2027 (+1 more)

### Community 43 - "Reproducibilidad y versiones"
Cohesion: 0.25
Nodes (4): Artefacto perfil_amenaza_u3.keras, Bucle de entrenamiento tf.GradientTape, keras==3.15.1, tensorflow==2.21.0

### Community 44 - "Controles de datos y pruebas"
Cohesion: 0.25
Nodes (7): Controles experimentales de preprocesamiento (§7), Escalador con fuga (control negativo), Control de etiquetas barajadas (cordura), Umbral de confiabilidad MIN_EVENTOS = 10 (D3), Pesos de clase en la BCE, 5 pruebas automáticas (tests/test_pipeline.py), Transformación logarítmica de x1, x2, x4

### Community 45 - "Actores armados UCDP (I3)"
Cohesion: 0.43
Nodes (8): Análisis de errores (transiciones, Arauca), Arauca (caso mal clasificado en U2), I3 x3 actores UCDP (no retenida), UCDP GED 26.1 (actores armados no estatales), x3 grupos/actores armados no estatales, Salto de eventos UCDP en 2023, Verificación UCDP para x3 (4-nov-2026), Variable x3 opcional (UCDP)

### Community 46 - "Calidad del dataset (Act. 4)"
Cohesion: 0.25
Nodes (8): Checklist de calidad del dataset, Tabla 2: evaluación de calidad del dataset de modelado, Export semanal agregado de ACLED (17.573 registros, actualizado 15-08-2026), Panel departamento–año (171 filas, 30 departamentos), Criterio: definición de variables (1 pt), Criterio: justificación del uso de redes neuronales (1 pt), Criterio: selección y calidad del dataset (1,5 pts), Rúbrica de evaluación Actividad 4 (total 5,0)

### Community 47 - "Entrenamiento MLP U2 (código)"
Cohesion: 0.25
Nodes (3): construir_mlp(), entrenar_fijo(), entrenar_mlp()

### Community 48 - "Trabajo colaborativo"
Cohesion: 0.47
Nodes (6): Juan Diego Saldarriaga Muñoz, Laura Alejandra Díaz Sastoque, Trabajo colaborativo y organización del equipo (Etapa 1), Trabajo colaborativo y organización del equipo (Etapa 2), Criterio: trabajo colaborativo y organización (0,5 pts), Criterio: oportunidades de mejora y trabajo colaborativo

### Community 49 - "Métricas de decisión"
Cohesion: 0.33
Nodes (5): Bergmeir y Benítez (2012): validación cruzada en series temporales, Bootstrap por departamento (IC pareados), Gneiting y Raftery (2007): reglas de puntuación propias, Validación de origen móvil 2023–2025 (selección, n = 87), Persistencia del perfil (rival a vencer)

### Community 50 - "Validación U2 (código)"
Cohesion: 0.40
Nodes (4): agregado(), evaluar_particion(), metricas(), validacion()

### Community 51 - "No linealidad: red vs. logística"
Cohesion: 0.40
Nodes (5): ¿Aporta la no linealidad? (McNemar, bootstrap, dependencia parcial), Estabilidad de la logística por remuestreo (500 réplicas), Dietterich (1998). Approximate statistical tests for comparing classifiers, Friedman (2001). Greedy function approximation, Regresión logística (línea base con las siete predictoras)

### Community 52 - "Actividad 2: preprocesamiento"
Cohesion: 0.50
Nodes (3): Actividad 2: Diseño técnico del proceso de preparación y representación de datos, Codificación de variables categóricas, Escalamiento y normalización de datos

### Community 53 - "Log-loss e importancia"
Cohesion: 0.50
Nodes (3): bce(), comparar_pareado(), importancia_perm()

### Community 54 - "Descarga de fuentes externas"
Cohesion: 0.50
Nodes (3): descargar_bosque(), descargar_coca(), normalizar()

## Ambiguous Edges - Review These
- `Proyecto de Aplicación U1 – Etapa 1: clasificación por perfil de amenaza tecnológica` → `Entregable: propuesta PDF de máximo 5 páginas o 10 diapositivas`  [AMBIGUOUS]
  docs/entregas/U1/Proyecto_Aplicacion_U1_LADS_JDSM.pdf · relation: references

## Knowledge Gaps
- **77 isolated node(s):** `Curso Redes Neuronales, Universidad de La Sabana`, `Modernización militar de Colombia 2026 (3,7 billones)`, `D consolidado (P(C) < 0,3): orden público`, `Regresión logística (mismas 9 variables)`, `Análisis de errores (transiciones, Arauca)` (+72 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 209 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **10 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `UCDP GED 26.1 (actores armados no estatales)` connect `Actores armados UCDP (I3)` to `Modelo final y tarjeta`, `Pruebas y herramienta UCDP`?**
  _High betweenness centrality (0.033) - this node is a cross-community bridge._
- **Are the 3 inferred relationships involving `Proyecto de Aplicación U2 – Etapa 2: modelo base entrenado` (e.g. with `Contenido requerido del informe (recordatorio, dataset, arquitectura, métricas, curvas, diagnóstico, mejoras)` and `Entregable: informe técnico PDF de máximo 8 páginas o 12 diapositivas`) actually correct?**
  _`Proyecto de Aplicación U2 – Etapa 2: modelo base entrenado` has 3 INFERRED edges - model-reasoned connections that need verification._
- **What connects `Curso Redes Neuronales, Universidad de La Sabana`, `Modernización militar de Colombia 2026 (3,7 billones)`, `D consolidado (P(C) < 0,3): orden público` to the rest of the system?**
  _77 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Datos y variables (código U3)` be split into smaller, more focused modules?**
  _Cohesion score 0.08078231292517007 - nodes in this community are weakly interconnected._
- **Why does `Variable x3 opcional (UCDP)` connect `Actores armados UCDP (I3)` to `README y paquete`, `Pruebas y herramienta UCDP`?**
  _High betweenness centrality (0.032) - this node is a cross-community bridge._
- **Are the 3 inferred relationships involving `Informe U3: Modelo neuronal integrado para anticipar el perfil de amenaza tecnológica de los territorios colombianos` (e.g. with `Actividad 5: Proyecto Integrador: diseño, implementación y evaluación de un modelo neuronal aplicado` and `Producto 1: Informe técnico final (PDF, máx. 12 páginas)`) actually correct?**
  _`Informe U3: Modelo neuronal integrado para anticipar el perfil de amenaza tecnológica de los territorios colombianos` has 3 INFERRED edges - model-reasoned connections that need verification._

### Low-confidence Hints
_AMBIGUOUS edges — the extractor was unsure. Verify before acting on these._

- **What is the exact relationship between `Proyecto de Aplicación U1 – Etapa 1: clasificación por perfil de amenaza tecnológica` and `Entregable: propuesta PDF de máximo 5 páginas o 10 diapositivas`?**
  _Edge tagged AMBIGUOUS (relation: references) - confidence is low._