# Manifiesto de artefactos del proyecto (Unidad 2)

Lista de archivos de datos y salidas que no son texto ni código, con su origen. Todas las rutas son relativas a `Proyecto Unidad 2/`.

## Datos de entrada

| Archivo | Qué es | Fuente |
|---|---|---|
| `acled-data-export.csv` | Export semanal de ACLED para Colombia (semana × departamento × subtipo de evento; 17.573 filas, 12 columnas; hasta el 15 de agosto de 2026) | ACLED (Raleigh et al., 2010), descarga gratuita agregada a departamento |
| `datos_externos/coca_departamento_anio.csv` | Hectáreas de coca por departamento y año, agregadas desde el censo municipal | Observatorio de Drogas de Colombia (MinJusticia), Datos Abiertos Colombia, recurso `acs4-3wgp`; serie SIMCI–UNODC |
| `datos_externos/bosque_departamento_anio.csv` | Porcentaje de cobertura arbórea (dosel ≥ 30 %) por departamento y año | Global Forest Watch (Hansen et al., 2013) |
| IPM 2018 por departamento | Diccionario escrito en el cuaderno (sección 4.2), no es un archivo | DANE, Pobreza multidimensional por departamentos 2018 |

## Salidas del cuaderno (`salidas_u2/`)

| Archivo | Contenido | Sección del cuaderno que lo genera |
|---|---|---|
| `dataset_modelado_departamento_anio.csv` | Tabla de modelado: 171 filas departamento–año con las predictoras en t−1 y la etiqueta en t | 4.3 |
| `busqueda_hiperparametros.csv` | Resultados de las 16 configuraciones evaluadas con validación cruzada temporal | 7.3 |
| `historial_entrenamiento.csv` | Pérdida, exactitud, AUC, gradiente y tasa de aprendizaje por época del modelo base | 8.3 y 9 |
| `metricas_modelos.csv` | Métricas por modelo (mayoritaria, persistencia, logística, MLP fase 1 y 2) y conjunto | 10.1 |
| `intervalos_confianza_prueba.csv` | Intervalos de confianza bootstrap al 95 % en la prueba 2026 | 10.2 |
| `estabilidad_semillas.csv` | Resultados del reentrenamiento con 10 semillas | 11.2 |
| `validacion_temporal_espacial.csv` | Validación temporal de origen móvil y validación espacial por departamentos | 11.3 |
| `experimento_E1_memoria.csv` | Comparación de persistencia, logística y MLP con y sin memoria del perfil y con ensamble | 13.1 |
| `proyeccion_2027.csv` | Probabilidad de perfil C proyectada para 2027 por departamento, categoría y alerta A/B | 12.1 y 13.2 |
| `configuracion.json` | Hiperparámetros elegidos, partición temporal, semilla y umbral | 16.1 |
| `modelo_base_fase1.keras` | Modelo base entrenado con 2021–2024 y validado con 2025 | 16.1 |
| `modelo_reajustado_fase2.keras` | Modelo reentrenado con 2021–2025 | 16.1 |
| `figuras/` | 23 figuras del cuaderno (fig01 a fig23) en PNG | a lo largo del cuaderno |
| `figuras_informe/` | 8 figuras a tamaño de impresión (R1 a R8) usadas en el informe | Anexo B |

## Documentos

| Archivo | Qué es |
|---|---|
| `Proyecto_U2_Modelo_Base.ipynb` | Cuaderno ejecutado de la Unidad 2 (texto y código convertidos en `notebooks_texto/`) |
| `Proyecto de Aplicación_U2_LADS&JDSM.pdf` | Informe técnico de la Unidad 2 entregado (8 páginas) |
| `Guia_Proyecto_U2_Avance.pdf` | Guía y rúbrica de la actividad de avance del proyecto (modelo base entrenado) |
