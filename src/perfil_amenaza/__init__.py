"""
perfil_amenaza: pipeline reproducible del Proyecto Integrador (Unidad 3).

Clasificación del perfil de amenaza tecnológica (C convencional vs. D mixto) por departamento y año,
con predictoras medidas al cierre del año anterior.

Módulos
-------
config      Rutas, constantes y lectura de configuracion.json.
datos       Carga y auditoría de ACLED, panel departamento-año, etiqueta y fuentes externas.
variables   Ingeniería de variables (base U2, memoria, dinámica intra-anual, vecindad, actores) y secuencias.
modelo      Redes en Keras (MLP, MLP multitarea, GRU, CNN 1D, embedding), bucle de entrenamiento y calibración.
evaluacion  Métricas, intervalos bootstrap, pruebas pareadas, validación de origen móvil y espacial, PSI.
pipeline    Orquestación de extremo a extremo y línea de comandos (entrenar, evaluar, proyectar, predecir).
"""
__version__ = "3.0.0"
