# Instrucciones para asistentes de IA (Claude Code y otros)

1. **Lee primero `CONTEXTO_PROYECTO.md`**: problema, datos, historia U1→U2→U3, resultados, estructura y pendientes.
2. Para preguntas sobre relaciones entre conceptos, archivos o decisiones, consulta el grafo: `graphify-out/GRAPH_REPORT.md`
   (o `graphify query "<pregunta>"` si graphify está instalado). Si cambias documentos o código de forma importante,
   actualízalo con `/graphify . --update`.
3. Reglas del proyecto:
   - Todo en **español** (código, comentarios, documentos); coma decimal en los textos.
   - **No uses 2026 para seleccionar modelos**: es la prueba final. Selección solo con origen móvil 2023–2025.
   - La lógica vive en `src/perfil_amenaza/`; el cuaderno la importa. No dupliques código en el cuaderno.
   - Cualquier cifra que cites debe salir de una ejecución (cuaderno, pipeline o `salidas_u3/`), nunca de memoria.
   - Corre `python -m pytest -q tests` después de tocar `src/`.
   - El repositorio es privado porque incluye el export de ACLED (licencia académica): no lo publiques.
4. Equipo: Laura Alejandra Díaz Sastoque y Juan Diego Saldarriaga Muñoz (Redes Neuronales, Universidad de La Sabana).
