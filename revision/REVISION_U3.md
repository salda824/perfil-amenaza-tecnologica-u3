# Revisión estricta del Proyecto Integrador U3 (fase 1: solo revisión)

**Objeto revisado:** `informe/Informe_U3_Proyecto_Integrador.pdf` (8 páginas; fuente `informe/construir_informe.py`,
figuras `informe/generar_figuras.py`), contrastado con la guía oficial (`docs/guias/Guia_Actividades_Unidad3_completa.pdf`,
Actividades 1 a 5 y sus rúbricas), el cuaderno ejecutado (`docs/u3_texto/U3_integrador_texto.md`), los CSV de
`salidas_u3/`, `configuracion.json`, el código de `src/perfil_amenaza/`, las pruebas y los informes U1 y U2.

**Qué se ejecutó (nada escribe en el proyecto, salvo este archivo):**

* `python3 -m pytest -q tests`: **5 passed** en 9,6 s. `git status` quedó limpio.
* Scripts de verificación propios en la carpeta temporal de la sesión (el código esencial está en el Anexo A para que
  el equipo lo reproduzca): (a) persistencia calibrada y logística con una sola variable; (b) ensamble de 10 semillas
  del modelo base U2; (c) IC por departamento de las intervenciones retenidas; (d) validación espacial con bloqueo
  también temporal; (e) ruido de muestreo del PSI con n = 33; (f) ventana de d₁ con el año parcial. Todos usan el
  paquete `perfil_amenaza` y los datos del repositorio. **Regla del proyecto:** antes de citar en el informe cualquier
  cifra nueva de esta revisión, hay que generarla dentro del cuaderno o del pipeline.

**Convenciones.** Severidad: **CRÍTICO** (error de fondo o que baja la nota), **MAYOR**, **MENOR**. Criterios de la
rúbrica de la Actividad 5: **R1** coherencia metodológica integral, **R2** preparación y representación de datos,
**R3** calidad técnica del modelo y optimización, **R4** evaluación e interpretación, **R5** entregables y comunicación
ejecutiva. "L###" es la línea de `informe/construir_informe.py` donde está el texto citado (versión del commit
`823d68c`).

---

## 0. Resumen

| Severidad | Número |
|---|---|
| CRÍTICO | 6 |
| MAYOR | 14 |
| MENOR | 33 (25 del informe y 8 de código y repositorio) |

**Los diez más importantes**

1. **C3.** El rival "persistencia" se usa sin calibrar. Una persistencia calibrada (logística de 2 parámetros sobre
   s̃_C(t−1), entrenada solo con años anteriores) da log-loss 0,328 en selección y **0,351 en 2026**, mejor que el modelo
   final (0,380). Una logística con solo d₁ da 0,305 y 0,323. La conclusión central ("supera a la inercia en calidad
   probabilística") no resiste ese rival.
2. **C5.** La validación espacial mezcla años. Si además se respeta el tiempo (otros departamentos, solo años
   anteriores), el modelo queda en **0,361 frente a 0,363** de la persistencia. "La debilidad espacial quedó resuelta"
   y la recomendación R1 se apoyan en un esquema que favorece al modelo.
3. **C1.** El informe presenta la regla 8 como "fijada antes de ver resultados". El cuaderno dice que se formalizó **al
   encontrar el caso del embedding**, que fue el mejor modelo en 2026.
4. **C2.** La declaración de IA (solo "programación") contradice `CONTEXTO_PROYECTO.md` §11, que está en el repositorio:
   "los textos interpretativos fueron redactados con apoyo de IA".
5. **C4.** La "lección de I6" compara contra una configuración (16 neuronas, 560 pasos) que la U2 eligió **con
   validación que incluía 2025**, el mismo año que se usa como pliegue "fuera de la selección".
6. **C6.** El repositorio es privado y el enlace no es clicable. Si el profesor no tiene acceso, falta un entregable
   explícito de la guía.
7. **M3 y M4.** Las comparaciones antes/después no son justas: una red con la semilla 42 (que la U2 documentó como cola
   desfavorable) frente a un ensamble de 10, y en el eje espacial, la media de redes individuales (0,475) frente al
   ensamble (0,351).
8. **M2.** El texto dice que las 4 transiciones de 2026 "ningún modelo [las] acierta", pero la propia Tabla 6 da
   "1 de 4" al modelo de la U2. Además omite que la U2 anticipaba 3 de 13 transiciones y la U3 anticipa 2.
9. **M7.** La matriz de confusión (Figura 6c) no se interpreta en ninguna parte, y faltan las métricas de
   clasificación. Tampoco se reporta que, con umbral 0,5, la persistencia clasifica mejor (exactitud 0,897 frente a
   0,839; McNemar 7 contra 2).
10. **M14, M8 y M1.** Hay un error de código en `decidir`: la logística se compara con la semilla 1 y no con la media
    (Δ +0,0027 en vez de −0,0035; EE 0,0043). Falta el lazo error → gradiente → actualización que pide la Actividad 1. La pregunta
    orientadora citada no coincide con la de la Etapa 1 ("territorios" en lugar de "municipios").

**Lo que está bien (verificado):**

* Todas las cifras de la Tabla 6 coinciden con `metricas_seleccion.csv`, `antes_despues_*.csv` y
  `metricas_prueba_2026.csv`, redondeadas a 3 decimales.
* Cuadran con los CSV y el cuaderno las Tablas 3 y 5, los parámetros (177, 145, 393–665), las curvas (0,341/0,279;
  0,448/0,377; 0,360; 0,441), las cifras de errores (19; 58 %; 2 %; 11 de 19), Arauca (0,77–0,85), la proyección (11/11/11;
  14,4 % y 13,4 %; 25 de 33; 10 → 3) y el PSI (0,29/0,25).
* Las predictoras no tienen fuga temporal (revisé `variables.py` y `datos.py`, y la prueba de fuga es correcta).
* Las intervenciones retenidas son robustas al remuestrear departamentos: m₁ da ΔLL −0,036, IC95 [−0,056; −0,019], y d₁
  da −0,037, IC95 [−0,065; −0,011] (script c).
* La prueba 2026 no se usó para elegir el modelo y el texto lo declara. Están bien la reproducción de la U2 al dígito,
  el artefacto `.keras` (152 KB) y la coincidencia pipeline–cuaderno al cuarto decimal.
* Extensión: 8 de 12 páginas, lo que deja unas 4 páginas libres para corregir los faltantes. Tablas 1–8 y Figuras 1–9
  numeradas en orden.

---

## 1. Hallazgos CRÍTICOS

### C1 · CRÍTICO · R1, R5 (integridad) · La regla 8 se presenta como preregistrada

* **Dónde:** pág. 3, §4.1. El título L308 dice "4.1 Protocolo fijado antes de ver resultados". L316–318: "**Reglas 7 y
  8.** Los controles de datos verifican, no seleccionan. Una variante que gana solo porque se parece más a la
  persistencia, o porque no puede aplicarse a territorios nuevos, se documenta pero no se adopta." En la pág. 5, L397–398:
  "no se adoptó por la regla 8".
* **Evidencia:** el cuaderno §6.7 (`herramientas/cuaderno/lecturas.py:166`) dice: "No se adopta por la regla 8, **que
  formalizamos al encontrar este caso**, y lo declaramos así". `CONTEXTO_PROYECTO.md` §5.8 y §12 piden: "La regla 8 se
  formalizó al ver el caso del embedding: decirlo siempre así". La única variante descartada por esa regla es el
  embedding, que fue el mejor modelo en 2026 (log-loss 0,314 frente a 0,380). Si el profesor abre el cuaderno, verá
  que el informe presenta una regla posterior como previa. El cuaderno también se contradice: §3.1 ("Reglas fijadas
  antes de ver resultados", `parte_a.py:320–346`) incluye la regla 8.
* **Corrección (L308 y L316–318):** cambiar el título a "4.1 Protocolo de evaluación" y reemplazar el bloque por:
  > **Controles y alcance.** Los controles de la §2.2 solo verifican: si uno sugiere quitar una transformación, se
  > reporta y queda para la siguiente iteración (regla 7). Durante las intervenciones agregamos una regla más
  > (regla 8), y lo declaramos: la escribimos al encontrar el caso del embedding de departamento (§6). Según ella, una
  > variante que mejora la log-loss solo porque se parece más a la persistencia (acierta menos transiciones) o porque no
  > puede aplicarse a territorios nuevos se documenta, pero no se adopta.
* En L397–398: "no se adoptó por la regla 8, que agregamos al ver este resultado". En el cuaderno, sacar la regla 8 de
  "Reglas fijadas antes de ver resultados" o marcarla como agregada después (`parte_a.py:343`).

### C2 · CRÍTICO · R5 (integridad académica) · La declaración de IA es incompleta frente a lo que dice el repositorio

* **Dónde:** pág. 8, L482–484: "declaramos el uso de inteligencia artificial (Claude, de Anthropic) como apoyo en la
  programación del pipeline y del cuaderno. La propuesta del proyecto, la búsqueda de datos, el análisis, las
  decisiones metodológicas y las conclusiones son del equipo".
* **Evidencia:** `CONTEXTO_PROYECTO.md:300`, en el mismo repositorio que el informe enlaza: "la declaración de IA dice
  que la IA apoyó en programación; los textos interpretativos fueron redactados con apoyo de IA". Todos los commits
  llevan `Co-Authored-By: Claude`. La declaración de la U2 era más amplia: incluía "proponer ajustes de redacción". La
  guía (pág. 12) dice que "el uso no responsable o **no declarado** de IA constituye falta a la integridad académica",
  y permite el uso de IA para "revisión de redacción, organización del informe o contraste conceptual".
* **Corrección:** la declaración tiene que describir lo que pasó. Si los textos se reescriben ahora con palabras
  propias, una redacción posible es:
  > **Uso de IA.** Usamos Claude (Anthropic) para programar y depurar el pipeline y el cuaderno, revisar la coherencia
  > entre cuaderno e informe y proponer redacción, que revisamos y reescribimos. La formulación del problema, la elección
  > de fuentes, el protocolo, las decisiones metodológicas y las conclusiones son del equipo. Ambos ejecutamos el
  > cuaderno, verificamos cada cifra contra las salidas y podemos explicar y reproducir cada decisión.
* Hay que alinear también `lecturas_fijas.py:35` (cuaderno §14.3) y, cuando corresponda, la línea 300 de
  `CONTEXTO_PROYECTO.md`. No se trata de ocultarla, sino de que todos los documentos digan lo mismo.

### C3 · CRÍTICO · R4, R1 · El rival es débil: una inercia calibrada empata o gana, y la conclusión central no se sostiene

* **Dónde:** pág. 5, L377–381: "la conclusión honesta es que el modelo supera a la inercia en calidad probabilística y
  no en ordenamiento". Conclusión 3, L451–453: "Frente a la persistencia, el modelo es mejor en calidad probabilística
  (selección, territorios nuevos y 2026)". Tabla 1, L233: "El rival es la persistencia".
* **Evidencia:** el propio texto atribuye la ventaja a la calibración: la persistencia "usa s^C como si fuera una
  probabilidad" (ECE 0,173). El rival probabilístico correcto es la inercia **calibrada**. Calculé (script a) una
  logística de y sobre logit(s̃_C(t−1)) = logit(m₁), entrenada hacia adelante (para cada v, solo con t < v; no mira el
  futuro):

  | Rival (2 parámetros, hacia adelante) | LL 2023–2025 | AUC | LL 2026 | AUC 2026 | Δ(U3 − rival) selección, IC95 por depto. | Δ 2026, IC95 |
  |---|---|---|---|---|---|---|
  | Persistencia cruda (la del informe) | 0,363 | 0,973 | 0,429 | 0,948 | −0,059 [−0,142; +0,028] | −0,049 [−0,162; +0,092] |
  | **Persistencia calibrada** (logit m₁) | **0,328** | 0,966 | **0,351** | 0,948 | −0,024 [−0,096; +0,052], P(mejora) 0,74 | **+0,029** [−0,055; +0,125], P(mejora) 0,27 |
  | Logística solo con d₁ (post hoc) | 0,305 | 0,965 | 0,323 | 0,962 | −0,001 [−0,074; +0,075] | +0,057 [−0,046; +0,197] |
  | MLP U3 (ensamble, 9 variables) | 0,304 | 0,956 | 0,380 | 0,900 | – | – |

  Con la persistencia calibrada, el modelo gana por poco en selección y pierde en 2026. Una logística de un parámetro
  sobre d₁ empata en selección (y además tiene MCC 0,799 y exactitud 0,897, frente a 0,691 y 0,839 del modelo). La
  variante con d₁ se eligió después de saber que d₁ es la variable más importante; la persistencia calibrada, en
  cambio, es el rival natural que se pudo fijar de antemano.
* **Corrección:**
  1. Agregar la "persistencia calibrada" como rival en el cuaderno (§3.2, `persistencia_oos`) y en el pipeline
     (`pipeline.py:106`), con la selección 2023–2025. Reportarla en la Tabla 6, también en 2026, con la nota "rival
     agregado después de la revisión; no cambia el modelo adoptado".
  2. Reescribir la viñeta de L377–381:
     > **Frente a la inercia, la ventaja es de calibración.** Frente a la persistencia cruda, el modelo baja la log-loss
     > en 0,059 (IC95 [−0,142; +0,028]), pero gran parte de esa diferencia se debe a que la persistencia usa s^C como si
     > fuera una probabilidad (ECE 0,173). Calibrada con una logística entrenada solo con años anteriores, la inercia
     > obtiene 0,328 en 2023–2025 y 0,351 en 2026. El modelo la supera en selección (−0,024; IC95 [−0,096; +0,052]) y
     > queda por debajo en 2026 (+0,029). La información útil está en la composición reciente (m₁ y d₁): a escala
     > departamental ni la red ni las variables estructurales agregan una ventaja demostrable sobre una inercia bien
     > calibrada.
  3. Ajustar en la misma línea la conclusión 3 (L451–453), el resumen del cuaderno y `CONTEXTO_PROYECTO.md` §6.5.
  4. Mantener la Tabla 1 (L233), pero aclarar "persistencia calibrada".

### C4 · CRÍTICO · R3 · La validación anidada favorece a la configuración "vigente", elegida con 2025

* **Dónde:** pág. 4, L347–351: "La configuración con mejor log-loss en 2023–2024 obtiene 0,423 en 2025, la elegida por
  la regla de 1 EE obtiene 0,411 y la vigente de la U2 obtiene **0,359** (Figura 5b). Con 58 filas de validación,
  optimizar hiperparámetros ajusta ruido". Tabla 4, L299: "la validación anidada mostró que elegir otra no transfiere".
  En el cuaderno (`lecturas.py:188`): "La vigente, **que nadie eligió mirando esos datos**".
* **Evidencia:** el informe U2 (§3.3) eligió 16 neuronas y L2 = 10⁻³ "con validación cruzada temporal de origen móvil,
  validando en 2023, 2024 **y 2025**". El presupuesto de 560 pasos (D10) es la mediana de la mejor época de 10 semillas
  con **parada temprana sobre la validación 2025** (U2 §3.1 y §6.1). La vigente se ajustó con el mismo año que aquí se
  usa como "fuera de la selección". Hay atenuantes: fue con 7 variables y no con 9. Aun así, la comparación de la
  Figura 5b está sesgada a favor de la vigente. Además, la "anidada" tiene un solo pliegue externo y 3 semillas.
* **Corrección (L347–351):**
  > **I6.** La rejilla de capacidad (4–32 neuronas), L2, activación y presupuesto (280–1.120 pasos) es plana: 31 de 48
  > configuraciones quedan dentro de un error estándar de la mejor, y la vigente (puesto 18) está en esa banda. Para
  > ver si elegir en la rejilla ayudaría, seleccionamos con 2023–2024 y medimos en 2025. La mejor de 2023–2024 obtiene
  > 0,423 y la elegida por la regla de 1 EE obtiene 0,411; la vigente obtiene 0,359. Esta comparación favorece a la
  > vigente, porque su capacidad y su presupuesto se eligieron en la U2 con validación que incluía 2025. Por eso no la
  > leemos como prueba de que la vigente sea mejor, sino como señal de que, con 58 filas, la rejilla no ofrece una
  > alternativa que se sostenga al año siguiente (Cawley y Talbot, 2010). Mantenemos la configuración de la U2 por
  > parsimonia.
* En Tabla 4 (L299): "La rejilla de 48 configuraciones fue plana; ninguna alternativa elegida con 2023–2024 mejoró en
  2025 (§5)". En el cuaderno, quitar "que nadie eligió mirando esos datos" (`lecturas.py:188`).

### C5 · CRÍTICO · R4 · La validación espacial mezcla años y la ventaja sobre la persistencia desaparece al respetar el tiempo

* **Dónde:** pág. 5, L382–386: "**La debilidad espacial de la U2 quedó resuelta.** [...] pasa de 0,475 (modelo base) a
  **0,351**, mejor que la persistencia (0,418 [...])". Conclusión 3 (L452: "territorios nuevos"), Tabla 7 R1 (L460:
  "Generalización espacial resuelta (§6)").
* **Evidencia:** `evaluacion.particiones_espaciales` (L100–107) agrupa por departamento sobre **todos** los años ≤ 2025.
  Para predecir el año t de un departamento que se dejó fuera, la red se entrena con las etiquetas de **ese mismo año t**
  (y de años posteriores) de los demás departamentos. Así aprende, por ejemplo, la prevalencia de 62 % de C en 2025.
  La persistencia no se entrena y no recibe esa ventaja, así que la comparación es asimétrica. Con bloqueo espacial y
  temporal a la vez (mismos 5 grupos, pero entrenando solo con t < v de los otros departamentos; script d; n = 87):

  | Modelo | Log-loss | AUC |
  |---|---|---|
  | Persistencia | 0,363 | 0,973 |
  | MLP U3 (ensamble 10) | **0,361** | 0,932 |
  | Logística (mismas 9 variables) | 0,416 | 0,909 |

  Δ(U3 − persistencia) = −0,002, IC95 por departamento [−0,114; +0,130]. En el escenario real (un territorio nuevo, el
  año siguiente), el modelo empata con la inercia. Hay un dato a favor del MLP que el informe no usa: en este esquema le
  gana a la logística por 0,055 (ver M12).
* **Corrección (L382–386):**
  > **En territorios nuevos, el modelo mejora mucho frente a la U2, pero no supera a la inercia.** Con 5 grupos de
  > departamentos (años ≤ 2025), la log-loss del ensamble baja de 0,445 (ensamble del modelo base) a 0,351. Ese esquema,
  > sin embargo, deja que la red aprenda de otros departamentos del mismo año. Si además se entrena solo con años
  > anteriores, el modelo obtiene 0,361 frente a 0,363 de la persistencia (IC95 de la diferencia [−0,114; +0,130]) y
  > 0,416 de la logística. Las variables de composición transfieren mejor que las estructurales, pero la ventaja frente
  > a la inercia en territorios nuevos no está demostrada.
* Ajustar R1 (L460), la evidencia "Generalización espacial resuelta (§6)" (por ejemplo: "Varianza por 30 unidades;
  transferencia espacial mejorada frente a la U2") y la conclusión 3. Incorporar el esquema espacio-temporal al cuaderno
  §9.2 y a `evaluacion.py`.

### C6 · CRÍTICO (si no se resuelve antes de entregar) · R5 · El enlace al código no es verificable

* **Dónde:** pág. 1, L209: "Código: https://github.com/salda824/perfil-amenaza-tecnologica-u3".
* **Evidencia:** según `CONTEXTO_PROYECTO.md` §1, el repositorio es privado; en §11 está pendiente "Agregar al profesor
  como colaborador". La guía (pág. 11) exige "El código deberá incluirse como enlace a repositorio (GitHub o similar)".
  En el PDF el enlace no es clicable (`pdfinfo -url` no devuelve ninguna URL).
* **Corrección:** dar acceso al profesor (Settings → Collaborators) o publicar una copia sin el CSV de ACLED, con
  instrucciones para descargarlo. Agregar debajo del enlace: "Repositorio privado por la licencia académica de ACLED;
  acceso concedido al profesor del curso." Hacer clicable la URL, por ejemplo con un hipervínculo en python-docx
  (`w:hyperlink`) en L209.

---

## 2. Hallazgos MAYORES

### M1 · MAYOR · R1 · La pregunta orientadora citada no es la de la Etapa 1, y la brecha de escala no se explica

* **Dónde:** pág. 1, L216–220: "En la Etapa 1 formulamos la pregunta orientadora: ¿Cómo orientar [...] del perfil de
  amenaza tecnológica predominante de **los territorios, según su actividad armada histórica** [...] de modo que se
  optimice **la inversión**?"
* **Evidencia:** los PDF U1 (pág. 1) y U2 (pág. 1) dicen: "[...] predominante de **los municipios colombianos según las
  variables de** actividad armada histórica, geografía operacional y economías ilícitas, de modo que se optimice la
  inversión **de recursos en búsqueda del bienestar de los colombianos y el desarrollo del país**?". Se presenta como
  cita de la Etapa 1 un texto modificado, y la modificación esconde la diferencia de escala entre la pregunta
  (municipios) y el modelo (departamentos). La U2 explicaba esa decisión; la U3 no. Las conclusiones (§9) tampoco
  responden de forma explícita la pregunta.
* **Corrección:** citar la pregunta textual de la U1 y agregar después del recuadro:
  > La pregunta pide municipios, pero el export gratuito de ACLED solo desagrega a departamento (hallazgo de la
  > Etapa 1). Por eso la respondemos a esa escala y dejamos la migración a municipio como la primera recomendación (R1).
* Agregar en §9 una conclusión de decisión que responda la pregunta: qué énfasis tecnológico en cuántos departamentos
  (11 C con ISR, detección temprana y movilidad protegida; C-UAS y detección de AEI en Guaviare y Cauca; portafolio
  flexible en 11), con qué incertidumbre y con qué límite (la inercia calibrada da casi lo mismo, C3).

### M2 · MAYOR · R4 · Contradicción con la Tabla 6 y omisión de que la U2 anticipaba más transiciones

* **Dónde:** pág. 5, L390–392: "las desescaladas C → D de 2026 (Bolívar, Chocó, Magdalena y Valle, las 4 transiciones de
  la prueba, **que ningún modelo acierta**) [...] El modelo acierta 2 de 13 transiciones y la persistencia 0". Límites,
  L454: "Anticipa 2 de 13 transiciones".
* **Evidencia:** la Tabla 6 (L364) da "Antes: MLP base U2 ... **1 de 4**" en 2026. En
  `predicciones_prueba_2026.csv`, Magdalena tiene P = 0,398 en el modelo U2, que acierta D. El embedding también acierta
  1 de 4 (`antes_despues_prueba_2026.csv`). El cuaderno §10.1 dice: "Transiciones 2023-2026: 13 | persistencia: 0 |
  **antes: 3** | después: 2". La Tabla 1 (L233) define que "el valor propio de un modelo está en anticipar cambios de
  perfil", y en esa métrica la U3 retrocede frente a la U2, cosa que el informe no dice.
* **Corrección (L390–392):** "...las 4 transiciones de la prueba, que ni el modelo final ni la persistencia aciertan (el
  modelo de la U2 acertó Magdalena, con P = 0,40)... El modelo final acierta 2 de 13 transiciones en 2023–2026; el de la
  U2, 3, y la persistencia, 0." En L454: "Anticipa 2 de 13 transiciones (la U2 anticipaba 3): la mejora probabilística
  vino de los territorios estables, no de los cambios de perfil."

### M3 · MAYOR · R4, A4 "comparación antes vs. después" · El "antes" es una semilla desfavorable frente a un ensamble de 10

* **Dónde:** Tabla 6 (L357–367), párrafo "Antes vs. después" (L368–371: "Δ −0,103 [...] Δ −0,144 [...] La mejora es
  significativa"). La Figura 5a muestra el modelo base en **0,383** y la Tabla 6 en **0,407**, sin explicar la
  diferencia.
* **Evidencia:** la U2 (§6.1) documentó que "el modelo base (semilla 42) quedó en la cola desfavorable", y la escalera
  (`escalera_mejora.csv`) da 0,383 como media de 5 semillas. "Después" es un ensamble de 10, así que la diferencia mezcla
  variables nuevas, ensamble y mala suerte de semilla. El ensamble de 10 semillas de la misma arquitectura base (script
  b, semillas 1–10, como el modelo final) da:

  | | Selección LL (AUC) | 2026 LL (AUC) |
  |---|---|---|
  | Base U2, semilla 42 (la "antes" actual) | 0,407 (0,913) | 0,524 (0,852) |
  | **Base U2, ensamble de 10 semillas** | **0,376 (0,933)** | **0,470 (0,862)** |
  | Final U3, ensamble de 10 | 0,304 (0,956) | 0,380 (0,900) |
  | Δ U3 − base (ensamble), IC95 por depto. | −0,072 [−0,114; −0,033] | −0,090 [−0,162; −0,024] |

* **Corrección:** agregar la fila "Antes: base U2 con ensamble de 10 semillas" en la Tabla 6 (después de generarla en el
  cuaderno §9) y reescribir L368–371:
  > **Antes vs. después.** Frente al modelo entregado en la U2 (una red, semilla 42), la log-loss baja de 0,407 a 0,304
  > en selección y de 0,524 a 0,380 en 2026. Para separar el efecto de las variables del efecto del ensamble, comparamos
  > también con el modelo base promediado sobre las mismas 10 semillas: la log-loss baja de 0,376 a 0,304 (Δ −0,072;
  > IC95 por departamento [−0,114; −0,033]) y en 2026 de 0,470 a 0,380 (Δ −0,090; [−0,162; −0,024]). La mejora atribuible
  > a m₁ y d₁ se mantiene con los dos criterios.
* En la leyenda de la Figura 5 (L355): "(a) Escalera: media de 5 semillas de una red individual; el modelo entregado en
  la U2 (semilla 42) tenía 0,407".

### M4 · MAYOR · R4, R1 · Validación espacial: comparación individual contra ensamble, y cambio de protocolo sin explicar

* **Dónde:** L383–384: "pasa de 0,475 (modelo base) a **0,351**".
* **Evidencia:** en `validacion_espacial.csv`, 0,475 es "MLP base U2 (**media 5 semillas**)", es decir, redes
  individuales, mientras que 0,351 es "MLP U3 (**ensamble**)". La red individual de la U3 da **0,373**, igual que la
  logística (0,373). El ensamble del modelo base da 0,445 (script b). Además, la U2 reportó la validación espacial
  **dejando un departamento fuera** (log-loss espacial del MLP base 0,605; AUC 0,78; Tabla 6 del informe U2) y explicó que
  descartaba GroupKFold de 5 grupos porque los desempates cambiaban entre versiones de scikit-learn. La U3 volvió a
  GroupKFold(5) (`evaluacion.py:104`) sin decirlo, así que el 0,475 no se puede rastrear hasta la U2. Con
  scikit-learn ≥ 1.6 el desempate ya es estable (`kind="stable"`), pero sigue siendo una única partición.
* **Corrección:** comparar ensamble con ensamble (0,445 → 0,351) o red individual con red individual (0,475 → 0,373), y
  decir en una frase por qué se usan 5 grupos y no dejar un departamento fuera; o volver a dejar un departamento fuera
  para comparar con la U2. Ver también C5.

### M5 · MAYOR · R3, A4 "fundamentación" · I4 cumple la regla escrita pero figura como "No retiene"

* **Dónde:** Tabla 5, L339: "I4 [...] Temporal −0,004 (0,002); espacial +0,009 | No retiene". La regla de L314–316 dice
  que se retiene "si la mejora media de la log-loss supera un error estándar de la diferencia pareada". Conclusión 2,
  L448–449: "la vecindad [...] no superaron el ruido entre semillas".
* **Evidencia:** en el eje temporal, 0,0040 > 0,0019 (EE) y ΔAUC +0,002, así que la regla escrita se cumple. El cuaderno
  usó para I4 un criterio distinto, que el informe no menciona: "Se retiene si mejora la validación espacial según la
  regla 5 sin empeorar la temporal" (cuaderno §6.4; `parte_b.py:129`). La frase de la conclusión 2 también es falsa para
  la vecindad temporal y para el embedding (−0,026; EE 0,010), que sí superaron el ruido entre semillas.
* **Corrección:** en la Tabla 5, decisión "No retiene (su hipótesis era espacial y en ese eje empeora)". En §4.1, agregar
  después de la regla de retención: "Para I4, cuya hipótesis es espacial, la regla se aplicó a la validación espacial y
  se exigió no empeorar la temporal." En L447–450: "...no cumplieron la regla de retención (la vecindad solo mejoró en el
  eje que no era el de su hipótesis; el embedding la cumplió y se descartó por la regla 8)".

### M6 · MAYOR · R2, R4 · El análisis exploratorio que motiva I2 usa la prueba 2026

* **Dónde:** Figura 1a y su leyenda, L265: "En el 70 % de las **23 transiciones**, el segundo semestre del año anterior
  (d₁) ya se movía hacia el nuevo perfil".
* **Evidencia:** las 23 transiciones son 5 (2021), 5 (2022), 1 (2023), 2 (2024), 6 (2025) y **4 (2026)**; en el
  dataset, `transicion` está por año. El cuaderno §2.5 dice que ese 70 % "es la justificación cuantitativa de I2", de
  modo que la prueba entra en la motivación de una variable. Eso choca con "2026 se evalúa una sola vez" (L310). Sin
  2026, la cifra mejora: **74 % de 19 transiciones (2021–2025)**. Lo reproduje con la misma definición (d₁ frente a m₁ en
  la dirección del perfil nuevo), que con las 23 da 69,6 %.
* **Corrección:** en `generar_figuras.py:61–65` y en la celda del cuaderno §2.5, filtrar `datos.anio <= 2025`.
  Leyenda: "(a) En el 74 % de las 19 transiciones de 2021–2025 (sin la prueba), el segundo semestre del año anterior ya
  se movía hacia el nuevo perfil."

### M7 · MAYOR · R4 (Actividad 3) · La matriz de confusión no se interpreta y faltan las métricas de clasificación y el balance

* **Dónde:** la Figura 6c (pág. 5) no se comenta en ninguna parte del texto. La Tabla 6 no incluye exactitud,
  sensibilidad, especificidad, precisión ni F1. Tampoco aparece el balance de clases (82 C / 89 D) ni el resultado de
  McNemar, aunque §4.1 dice que se usa.
* **Evidencia:** la guía (Actividad 3) pide "Interpretación técnica de la matriz de confusión [...] como herramienta
  diagnóstica, no como elemento decorativo" e "Identificación de posibles desbalances". Los CSV dan, en 2026:
  sensibilidad C 0,929, especificidad D 0,733, exactitud 0,828 y precisión C 0,765. Con umbral 0,5 la persistencia
  clasifica **mejor** en los dos conjuntos: exactitud 0,897 frente a 0,839 y MCC 0,795 frente a 0,691 en selección;
  0,862 frente a 0,828 y 0,755 frente a 0,672 en 2026. McNemar en selección: la persistencia acierta 7 casos que el modelo
  falla y el modelo acierta 2 que la persistencia falla, p = 0,18 (cuaderno §9.1). El informe no lo menciona.
* **Corrección:** agregar un párrafo en §6, después de la Figura 6:
  > **Matriz de confusión (Figura 6c).** En 2026 el modelo clasifica bien 13 de 14 departamentos C y 11 de 15 D
  > (sensibilidad 0,93, especificidad 0,73, exactitud 0,83). Los 4 falsos positivos son exactamente las 4 desescaladas
  > C → D (Bolívar, Chocó, Magdalena y Valle del Cauca): el modelo sigue la inercia y sobrestima C cuando el conflicto
  > baja, con el riesgo de asignar ISR y movilidad protegida donde el perfil ya cambió. El único falso negativo es Cesar
  > (P = 0,45), que cae en la banda de transición y recibiría portafolio flexible. Con umbral 0,5 la persistencia
  > clasifica mejor (exactitud 0,90 frente a 0,84 en selección; 7 casos que solo ella acierta frente a 2, McNemar
  > p = 0,18). El modelo gana en la calidad de la probabilidad, que es lo que usa la decisión por bandas.
* En §2.1, después de "171 filas": "(82 C y 89 D; la prevalencia de C va de 32 % en 2021 a 62 % en 2025, y es 48 % en
  2026)" (cuaderno §2.2). Agregar a la Tabla 6 las columnas "Exactitud" y "Sens. C / Espec. D".

### M8 · MAYOR · R3 (Actividad 1) · Falta el lazo error → gradiente → actualización de pesos, y la función de pérdida

* **Dónde:** la Figura 8 (pipeline) y la Figura 3 (arquitectura) no muestran cómo se calcula el error ni cómo se
  actualizan los pesos. §3 y §4.2 no traen la ecuación de la pérdida (la U2 sí la traía, en §3.2).
* **Evidencia:** la guía, en la Actividad 1, dice: "Evitar representaciones fragmentadas que no muestren la relación entre
  el cálculo del error y la actualización de los pesos". La lista de chequeo pide "Se explica el proceso de entrenamiento
  (función de pérdida y optimizador)".
* **Corrección:** (1) en §4.2, después de la primera frase:
  > En cada paso, la red calcula p̂ para un lote de 16 filas (propagación hacia adelante), evalúa
  > L(θ) = −(1/n) Σᵢ ω(yᵢ)[yᵢ log p̂ᵢ + (1 − yᵢ) log(1 − p̂ᵢ)] + λ‖W‖², con ω(c) = n/(2n_c) y λ = 10⁻³, obtiene ∂L/∂θ con
  > `tf.GradientTape` (retropropagación) y Adam actualiza pesos y sesgos (η = 10⁻³). Con 113 filas de entrenamiento, 560
  > pasos equivalen a 70 épocas.

  (2) En `generar_figuras.py:232–256`, agregar bajo la etapa 3 o la 6 un mini-diagrama "x(t−1) → red → p̂ → BCE ponderada
  + L2 → ∇θ (GradientTape) → Adam → θ" con una flecha de retorno.

### M9 · MAYOR · R5, R1 · Siglas, variables, reglas y mapeo sin definir para quien no vio el cuaderno

Lista verificada en el PDF (primera aparición):

| Término | Dónde | Qué falta |
|---|---|---|
| ACLED, UCDP GED, SIMCI, GFW, IGAC, DANE, IPM | §2.1, Tabla 2 | Nombre completo una vez |
| ISR, C-UAS, AEI | §8 (L431–434) | Inteligencia, vigilancia y reconocimiento; sistemas contra drones; artefactos explosivos improvisados |
| AUC, Brier, ECE, MCC, log-loss = BCE | §4.1, §4.2 | Una frase cada una. ECE: 5 grupos de igual frecuencia (`evaluacion.py:20`). Log-loss y BCE son la misma cantidad |
| LL, EE, DE | Tablas 3 y 5, Fig. 5a | Expandir o no abreviar |
| "Reglas 7 y 8" | §4.1, Tabla 3 ("regla 7, §4.1") | Las reglas 1–6 no están numeradas en el PDF |
| D10, D11–D20, E1, M3, I1–I9 | Tabla 4, §10, Tabla 1, Tabla 7, Tabla 2 (antes de la Tabla 5) | Bitácora y códigos de la U2 que el lector no tiene. Quitarlos o explicarlos en la primera mención |
| "transición" | §4.1 ("aciertos en transiciones") | Cambio de perfil entre t−1 y t, con t−1 confiable |
| `MIN_EVENTOS` | Tabla 3, Tabla 7 | Es el umbral de 10 eventos de la §2.1 |
| Mapeo de subtipos A/B/C/D | En ninguna parte (pero hay un control "Mapeo alternativo") | A = IED/landmine, Grenade, Suicide bomb, Chemical weapon; B = Air/drone strike; C = Armed clash, Attack, Shooting, Abduction, Looting, Disrupted weapons use; D = protestas, disturbios, arrestos y el resto (`config.py:28–35`) |
| "D (amenaza mixta)" | L223 | Decir que D incluye la protesta y los disturbios (U2 §2: "D como territorio donde la violencia armada directa no alcanza la mitad de los eventos") |
| x₄–x₇ "Igual que la U2" | Tabla 2 (L250) | log(1 + ha de coca); % de dosel ≥ 30 %; distancia del centroide a Bogotá (km, proxy de presencia estatal); IPM 2018 (%) |
| Persistencia AUC 0,965/LL 0,380 frente a 0,973/0,363 | Tabla 1, Tabla 5 I1 frente a §4.1 | Aclarar que 0,965 es la validación 2023–2026 de la U2 |

* **Corrección:** un recuadro "Notación y siglas" de unas 10 líneas al final de §1 o en §2.1, más las definiciones en
  la primera mención. Hay espacio: el informe usa 8 de 12 páginas.

### M10 · MAYOR · R1, R4 · Sección de decisión 2027: reglas sin definir, una métrica mal planteada y supuestos no declarados

* **Dónde:** §8, L427–438.
* **Evidencia y correcciones:**
  1. **La alerta A/B no está definida.** "Guaviare (14,4 %) y Cauca (13,4 %) mantienen la alerta A/B" (L432–433). La regla
     es: proporción de eventos A+B en 2025–2026 ≥ 2 × la nacional (6,4 %, umbral 12,7 %) y al menos 5 eventos A+B
     (`pipeline.py:112–118`). Agregarla tal cual.
  2. **Las bandas no se explican.** Solo aparece "P ≥ 0,7". Agregar: transición 0,3 ≤ P < 0,7 y D consolidado P < 0,3.
  3. **"Las discrepancias con el perfil observado en 2026 bajan de 10 a 3" se presenta como mejora (L435–436).** Una
     proyección 2027 que se parece más al perfil 2026 solo está más cerca de la persistencia, que es lo que la regla 8
     penaliza. La U2 trataba esas discrepancias como "casos de revisión". Corrección: "La proyección contradice el perfil
     observado en 2026 en 3 departamentos (Chocó, Valle del Cauca y Meta; en la U2 eran 10). Es una medida de cercanía a
     la inercia, no de acierto; esos tres casos son los que conviene revisar con inteligencia de campo."
  4. **Departamentos con pocos datos dentro de los conteos.** Amazonas y Vaupés cuentan en los 11 C consolidados y San
     Andrés y Guainía en la transición, aunque la tarjeta del modelo dice "No usar para [...] departamentos con menos de
     10 eventos al año [...] sin revisión experta". Guainía, San Andrés y Vaupés nunca estuvieron en el entrenamiento
     (U2 §2). Corrección: "9 C consolidados con datos suficientes, más Amazonas y Vaupés, que requieren revisión experta".
  5. **d₁ se mide en otra ventana en la proyección.** Para 2027, las "últimas 26 semanas" de 2026 son las semanas 8 a 33
     (febrero–agosto), no el segundo semestre (`variables.py:71` y `datos.bloques_trimestrales`). En 2020–2025, d₁
     calculado así frente al d₁ del segundo semestre tiene correlación 0,88 y error absoluto medio de **0,084** (script
     f). d₁ es la variable más importante (permutación 0,115), y Meta (0,50), Huila (0,51) y Casanare (0,52) están en el
     umbral. Agregar: "La información de 2026 cubre 33 semanas: d₁ resume febrero–agosto y no el segundo semestre. La
     proyección debe repetirse al cerrar 2026."
  6. **No se evalúa la regla de decisión.** Datos para agregar (calculados con `predicciones_*.csv`): en 2023–2025, 71 de
     87 filas caen en una banda consolidada y 66 de ellas aciertan (31/32 en C, 35/39 en D); en 2026, 24 de 29, con 21
     aciertos (10/13 en C, 11/11 en D). Con la persistencia, 42 de 87 filas quedan en transición. Es la evidencia más
     directa de utilidad para la decisión, y no está en el informe.

### M11 · MAYOR · R4 · El PSI se sobreinterpreta con n = 33

* **Dónde:** §7, L420–422: "El índice de estabilidad poblacional (PSI) [...] marca deriva en m₁ (0,29) y en la letalidad
  (0,25)". Tabla 7, R4. El cuaderno §9.3 atribuye a esa "deriva medida" la caída 0,304 → 0,380.
* **Evidencia:** el PSI (`evaluacion.psi`, 5 grupos) se calcula con 33 filas frente a 171. Simulé el PSI sin deriva,
  sacando 33 filas de la propia referencia (2.000 repeticiones, script e): mediana ≈ 0,11 y percentil 95 ≈ 0,31–0,35. Los
  valores observados (0,29; 0,25; 0,24) tienen p ≈ 0,06; 0,10; 0,13. Los umbrales 0,1/0,25 son para muestras grandes. Las
  poblaciones tampoco coinciden: la proyección incluye los 33 departamentos, y la referencia, 30. Con solo los
  confiables, m₁ baja a 0,21 y la letalidad sube a 0,47.
* **Corrección (L420–422):** "Con 33 departamentos, el PSI tiene un ruido de muestreo alto: sin deriva, su mediana es
  ≈ 0,11 y su percentil 95 ≈ 0,31. Los valores de m₁ (0,29) y letalidad (0,25) son altos, pero no prueban deriva; los
  tomamos como una alerta para revisar la proyección cuando cierre 2026." Ajustar R4 y la explicación de la caída en 2026
  (cuaderno §9.3).

### M12 · MAYOR · R3, R1 · La justificación para mantener el MLP es débil, y el mejor argumento no se usa

* **Dónde:** L393–396: "Se mantiene el MLP por coherencia con la U1 y porque es la arquitectura que debe escalar."
  Conclusión 4, L454–455.
* **Evidencia:** "coherencia con la U1" no es un criterio técnico, y "debe escalar" es una afirmación sin respaldo. La
  guía pide "Priorizar estabilidad y generalización" y no admite "ajustes arbitrarios". Con las mismas variables la
  logística empata en el eje temporal, pero en el esquema espacio-temporal de C5 el ensamble MLP obtiene 0,361 y la
  logística 0,416, y en el espacial del proyecto 0,351 frente a 0,373. Ese es el argumento cuantitativo, y no aparece.
* **Corrección (L396):**
  > Mantenemos el MLP por dos razones medibles: en departamentos no vistos el ensamble supera a la logística (0,351
  > frente a 0,373 con 5 grupos; 0,361 frente a 0,416 si además se respeta el tiempo), y su frontera curva memoria ×
  > letalidad coincide con la hipótesis de no linealidad de la Etapa 1. La logística queda como control permanente.

### M13 · MAYOR · R5 ("articulación del equipo"), A4 · La sección de trabajo en equipo es insuficiente

* **Dónde:** §10, L470–471 (dos líneas) y Tabla 8.
* **Evidencia:** el nivel Excelente pide "articulación efectiva del equipo" y "roles definidos". El texto remite a "la
  bitácora del cuaderno (D11–D20 [...])", que el lector del PDF no tiene. No dice cómo se coordinaron (revisión cruzada,
  quién verificó qué, cómo se resolvieron desacuerdos). En el repositorio, los 5 commits son de un solo integrante y del
  mismo día (`git log`). Los nombres no son homogéneos: "Laura Díaz Sastoque", "Juan Diego Saldarriaga" y "Laura".
* **Corrección:** 4 a 6 líneas que digan cómo trabajaron: quién propuso cada intervención, quién revisó las cifras del
  otro, qué decisiones conjuntas se tomaron (D11–D20 resumidas en una línea cada una, o las 4 más importantes: protocolo
  con 2026 una sola vez, persistencia como rival, regla de retención y mantener la capacidad de la U2) y qué cambió por
  discusión del equipo. Usar nombres completos y homogéneos. Opcional: que Laura haga los commits de documentación.

### M14 · MAYOR · R3, R4 (código) · Error en `decidir`: la logística se compara con la semilla 1 y no con la media

* **Dónde:** `herramientas/cuaderno/parte_a.py:385–395`.
  ```python
  if len(a) != len(b):
      b = pd.concat([b] * len(a)).iloc[:len(a)].set_index(a.index)
  ```
  Si la candidata tiene 1 "semilla" (la logística) y la referencia 5, `b` queda reducida a la **primera** semilla.
* **Evidencia:** `decisiones_intervenciones.csv` da para la logística ΔLL = **+0,0027**, EE = 0. La semilla 1 de la
  referencia tiene LL 0,3034 (media de los tres pliegues de `rejilla_I6.csv`: 0,3434; 0,2273; 0,3396), y
  0,3061 − 0,3034 = +0,0027. Contra la media de las 5 semillas (0,3096), el Δ correcto es **−0,0035**, con EE ≈ 0,0043
  (DE 0,0096/√5; `escalera_mejora.csv`). La decisión no cambia por poco: −0,0035 + 0,0043 = +0,0008 > 0. La cifra errónea está en el cuaderno §6.7 y §6.11,
  en `configuracion.json["decisiones"]` y en `CONTEXTO_PROYECTO.md` §6.1 ("logística +0,003").
* **Corrección:**
  ```python
  if len(a) == 1 and len(b) > 1:
      a = pd.concat([a] * len(b)).set_index(b.index)
  elif len(b) == 1 and len(a) > 1:
      b = pd.concat([b] * len(a)).set_index(a.index)
  ```
  Después, regenerar el cuaderno y corregir CONTEXTO §6.1.

---

## 3. Hallazgos MENORES del informe

| # | Dónde (pág., L###) | Problema (evidencia) | Corrección |
|---|---|---|---|
| m1 | pág. 4, L353–354; §8 L431 | "las bandas 0,3/0,7 **de la Etapa 1**": las bandas aparecen por primera vez en el informe U2 (§6.5); el PDF U1 no las tiene | "de la Etapa 2" |
| m2 | L223; Tabla 1 | Fórmula y código (`datos.py:105`) usan s_C **≥** 0,5; la U1 dice "**supera** el 50 %" | Decir "al menos la mitad" o justificar el ≥ |
| m3 | Tabla 3, L271–279 | La columna "Δ log-loss (EE)" muestra en la fila MIN_EVENTOS log-loss absolutas. "Prevalencias de C entre 41 % y 62 %" son de los años validados; en entrenamiento la prevalencia es ≈ 44 % (2021–2024: 50/113). "Escalador con fuga (control negativo)": un control negativo es el que no debería tener efecto (aquí lo es "etiquetas barajadas"). "Las pruebas automáticas la impiden": las pruebas la detectan | Fila: "LL 0,300 / 0,307 / 0,338 (absolutas)". "La prevalencia de C en entrenamiento es 44 % y en los años validados llega a 62 %". "Escalador con fuga (fuga deliberada)". "las pruebas automáticas la detectan" |
| m4 | Fig. 2, L281; `generar_figuras.py:166` | Título "(— train · ··· val. 2025)" en inglés. La leyenda dice "con y sin **cada** transformación", pero solo muestra estandarización y logaritmos, con una sola semilla | "(a) Entrenamiento (—) y validación 2025 (···), semilla 1"; "con y sin estandarización y logaritmos" |
| m5 | Tabla 5, L334–346 | Orden I1–I5, **I9**, –, I6, I7, I8. I2 da AUC absoluto (0,951) y las demás, ΔAUC. I7 e I8 sin EE. "LL" sin definir. I8 "0,298 → 0,302" es sobre **2024–2025** (calibración hacia adelante), no 2023–2025 | Ordenar por número o explicar que I9 se agregó después. "ΔAUC +0,012". "(2024–2025)" |
| m6 | Tabla 6, L362 | **0,956** en negrita, como si fuera el mejor AUC, cuando la persistencia tiene 0,973. "Antes" se reentrenó con 560 pasos fijos: no es el modelo entregado (la U2 fase 2 reportó 0,547 en 2026; cuaderno §9.3) | Quitar la negrita del AUC. Aclarar "base U2 reentrenada con el protocolo U3" |
| m7 | §4.2 (L323–324) frente a §5 (L350) | Para la misma configuración, entrenamiento 2021–2024 y validación 2025: 0,377 (Fig. 4, 5 semillas) y 0,359 (Fig. 5b, 3 semillas; `rejilla_I6.csv`). Además, la Fig. 4 usa "Época" y el texto, "pasos" | "(media de 3 semillas en la rejilla; 0,377 con 5 en la Figura 4)". "560 pasos (70 épocas)" |
| m8 | L351–352 | "Platt da una escala de 1,31 [...] y diverge en 2025, cuando la red ordena sin errores" no se entiende ("escala" es el parámetro a). Además, parece contradecir los 6 errores de 2025: el AUC de 2025 es 1,0, pero con umbral 0,5 falla 6 escaladas | "El parámetro de escala de Platt, a, vale 1,31 en 2023 y 3,38 en 2024, y en 2025 no converge porque la red separa perfectamente las clases (AUC = 1,0), aunque con umbral 0,5 clasifica mal seis escaladas" |
| m9 | L370–371 | "el modelo queda mejor calibrado que las alternativas (ECE 0,074)" solo vale en 2026. En selección su ECE (0,124) es peor que el de la logística (0,068) y el de la base (0,084). Con 29 filas y 5 grupos, el ECE es muy ruidoso | Mencionar los dos conjuntos y el ruido |
| m10 | L398 | "no transfiere a territorios nuevos": en el espacial, el embedding da 0,369 frente a 0,379 sin embedding (cuaderno §6.7). Lo que pasa es que en departamentos nuevos se reduce al MLP sin embedding | "en departamentos nuevos se reduce al MLP sin embedding (0,369 frente a 0,379), así que su ganancia viene de efectos fijos memorizados" |
| m11 | L394–395, Fig. 7b | La dependencia parcial se calcula con m₁ y letalidad correlacionadas (ρ = 0,84; cuaderno §10.2), así que extrapola a combinaciones sin datos. La frontera es apenas curva | Decirlo y limitar la lectura a la zona con datos, o usar ALE. "levemente curva" |
| m12 | L371, L314–316 | "La mejora es significativa [...] en los dos conjuntos": en selección, el resultado viene de una cascada de unas 20 comparaciones sobre las mismas 87 predicciones (el sesgo de selección de Cawley y Talbot, que el informe cita). El EE de la regla es **entre semillas**: mide el ruido del entrenamiento, no el del muestreo | "En 2026, la única estimación sin selección, la mejora se mantiene (IC95 [−0,270; −0,033])". Agregar los IC por departamento de I1 [−0,056; −0,019] e I2 [−0,065; −0,011] (script c) |
| m13 | L288 | "177 parámetros por red (1,25 por observación)": falta N (142 filas 2021–2025). Hay más parámetros que datos y no se comenta | "(1,25 por fila de entrenamiento, 142 filas; la L2 y el presupuesto fijo controlan el sobreajuste)" |
| m14 | L387–392 | Errores omitidos: Huila (C en 2023 y 2024, P 0,36 y 0,47), Valle 2023 (transición C → D), Cesar 2026 (único falso negativo, P 0,45) | Mencionar a Huila como error recurrente y a Cesar en la matriz de confusión (M7) |
| m15 | Figuras 1–9 | Ejes con punto decimal (0.2, 0.4...) en un texto con coma decimal. Rótulos superpuestos en las Figs. 7a y 9. "Valle del 2025" por el recorte `[:10]` (`generar_figuras.py:219`). En la Fig. 9, "tamaño proporcional a los eventos": el código usa la raíz (`:260`). Letras de 4,3–5,3 pt en las Figs. 3, 8 y 9 | `ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}".replace(".", ",")))`; abreviar "Valle"; quitar rótulos del centro de la Fig. 9 o usar una tabla. "tamaño según la raíz de los eventos" |
| m16 | Texto | Las Tablas 4, 6, 7 y 8 y las Figuras 1a, 6 y 9 no se citan en el cuerpo, solo en su leyenda | Agregar "(Tabla 4)", "(Tabla 6)", "(Figura 6)", "(Figura 9)", etc. |
| m17 | Referencias, L485–504 | Sin cita en el texto: Géron (2017), Guo et al. (2017), Kingma y Ba (2015) y Platt (1999): se menciona "Platt", pero no "(Platt, 1999)". APA 7 pide todos los autores hasta 20, y Hansen (15), Mitchell (9) y Roberts (14) están truncados con "…" (estilo APA 6). La cita "(Observatorio de Drogas de Colombia, 2026)" no coincide con el autor de la lista ("Ministerio de Justicia y del Derecho – Observatorio..."). Faltan DOI o URL en Cawley, Guo, Lakshminarayanan (y sus págs. 6402–6413), DANE y MinDefensa. Faltan las citas de los datos: ACLED (export, fecha de acceso), UCDP GED 26.1 y GFW/WRI (la U2 citaba World Resources Institute, 2025) | Citar Kingma y Ba en "Adam" (Tabla 4), Platt (1999) y Guo et al. (2017) en I8, y Géron en §4.2, o quitarlos. Completar los autores. Unificar el autor del Observatorio. Agregar las referencias de los datos |
| m18 | Encabezado, L207–209; metadatos | "Sector **E**legido" (la U2 escribía "elegido"); "&" en un texto en español; no figuran el curso, la universidad, el profesor ni la fecha; los metadatos del PDF dicen "Author: python-docx" y no tienen título | "Sector elegido"; "y"; agregar "Redes Neuronales · Universidad de La Sabana · noviembre de 2026"; `doc.core_properties.title/author` |
| m19 | Todo el texto | Marcas de redacción de IA (ver la sección 5) | Reescribir |
| m20 | L221–224 ("con lo observado hasta el cierre de t−1") | La coca del censo SIMCI del año t−1 y la pérdida de bosque de GFW se publican durante el año t, así que al cierre de t−1 todavía no están disponibles | Declararlo como límite: el efecto es pequeño porque son variables lentas |
| m21 | §2.1, L242–244 | No dice cuáles son los 3 departamentos excluidos (Guainía, San Andrés y Vaupés: ninguna celda con 10 eventos o más) ni que igual se proyectan en §8 | Nombrarlos |
| m22 | Tabla 7, R1 y R7 (L460, L465) | "el pipeline solo cambia `datos.py`": `config.py` tiene constantes por departamento (IPM_2018, VECINOS, NOMBRE). R7, "Candidato en la selección 2024-2026": 2026 ya se miró y se sabe que el embedding fue mejor ahí, así que esa selección estaría contaminada | "cambian `datos.py` y las constantes de `config.py`". "con selección 2024–2026 y prueba en 2027" |
| m23 | Notación | Se mezclan "U1/U2/U3" y "Etapa 1/3"; "2023-2025" y "2023–2025"; "5-95 %"; subíndices Unicode (m₁) y de Word (m_{2}) en la misma tabla; "x₂b" | Elegir una forma: "Etapa 1 (U1)" la primera vez y raya – en los rangos |
| m24 | Págs. 7–8 | El título de §10 queda en la pág. 7 y la Tabla 8 en la pág. 8 | `keep_with_next` o un salto de página antes de §10 |
| m25 | L349 | "validación anidada" con un solo pliegue externo (2025) y 3 semillas: es una selección en 2023–2024 con evaluación fuera de muestra en 2025 | "validación anidada de un solo pliegue externo" |

---

## 4. Código y repositorio (MENORES; el error de `decidir` está en M14)

| # | Archivo | Problema | Corrección |
|---|---|---|---|
| k1 | `src/perfil_amenaza/pipeline.py:157–161` | Si `calibracion` ≠ "ninguna", el calibrador se ajusta con **todas** las predicciones 2023–2025 y se evalúa sobre ellas mismas (dentro de muestra). Hoy no tiene efecto, porque la calibración es "ninguna", pero la Fig. 8 promete "solo si mejora sin mirar el futuro" y el pipeline no lo implementa (el encadenamiento hacia adelante solo está en el cuaderno) | Implementar el ajuste hacia adelante en `pipeline.py` o quitar la etapa 4 del diagrama |
| k2 | `pipeline.py:149–168`; Fig. 8 | La etapa "3 · Selección" solo **evalúa** la especificación congelada. La regla de retención y la cascada (`evaluar` y `decidir`) están en `herramientas/cuaderno/parte_a.py`, no en `src/`. Eso contradice "la lógica vive en `src/`" (CLAUDE.md) y la L409 ("El cuaderno importa ese paquete en lugar de duplicar la lógica") | Mover `evaluar` y `decidir` a `evaluacion.py`, o renombrar la etapa "3 · Evaluación" |
| k3 | `evaluacion.py:50–56` | `bootstrap_ic` estratifica por clase pero **no por departamento**. Los IC por modelo del cuaderno §9.1 tratan como independientes los años de un mismo departamento, a diferencia de `comparar_pareado` | Agregar `grupos` igual que en `comparar_pareado` |
| k4 | `tests/test_pipeline.py` | (a) Docstring de la línea 44: "Duplicar los eventos C", pero el código multiplica por 5. (b) `test_escalador_sin_fuga` solo prueba la logística y no la capa `Normalization` del artefacto. (c) `test_contrato_datos`, línea 34, es tautológica: `cargar_acled` ya lanza el error, y falta un caso negativo (CSV sin una columna → `ValueError`). (d) El determinismo solo se prueba dentro de un mismo proceso | Corregir el docstring. Agregar una prueba de que las medias de `Normalization` sean las del entrenamiento y un caso negativo del contrato |
| k5 | `ejecutar_todo.bat` | `--inplace` sobrescribe el cuaderno entregado y `configuracion.json`: en Windows pueden cambiar decimales y dejar de coincidir con el PDF. `py -3` no fija la versión (CONTEXTO pide 3.11–3.13; con 3.14 puede no haber ruedas de TF 2.21) | `--output Proyecto_U3_Integrador_ejecutado.ipynb`; `py -3.13 -m venv .venv` con `py -3.12` como respaldo |
| k6 | `salidas_u3/TARJETA_MODELO.md` (`pipeline.py:256–260`) | "x3 (actores) solo si hay UCDP" quedó desactualizado (x3 se evaluó y se descartó). "recalcular la calibración" contradice I8 (no se calibra porque no transfiere) | "x3 evaluada con UCDP y no retenida". "Reentrenar cada año; reevaluar la calibración con encadenamiento hacia adelante" |
| k7 | `README.md`, `CONTEXTO_PROYECTO.md` | El README dice "~20 min" y CONTEXTO §7 "~15 min" (el cuaderno midió 14,3). CONTEXTO §11 dice que R6 está en la Tabla 7, pero no está. CONTEXTO §6.3 y §6.5 repiten "generalización espacial resuelta" (C5) y §6.1 "logística +0,003" (M14) | Actualizar después de aplicar C3, C5 y M14 |
| k8 | `requirements.txt` | numpy, pandas y scikit-learn sin versión fija (`>=`). Con otras versiones pueden cambiar los decimales, aunque el desempate de `GroupKFold` es estable desde scikit-learn 1.6 | Fijar las versiones de la ejecución de referencia (numpy 2.5.3, pandas 3.0.5, scikit-learn 1.9.1, SciPy 1.18.1, según el cuaderno §0) |

---

## 5. Redacción que suena a texto generado (m19, detalle)

Hay que reescribir con palabras propias, sobre todo porque la declaración de IA (C2) lo exige:

| Texto actual (L###) | Problema | Propuesta |
|---|---|---|
| "No es una red más flexible que memoriza más, es una red mejor informada." (L326) | Contraste "no es X, es Y" usado como cierre | "La brecha baja porque la red recibe mejor información, no porque tenga más capacidad." O eliminar la frase |
| "La arquitectura no se mantuvo por inercia." (L291) | Frase de entrada escenificada | "También comparamos otras arquitecturas:" |
| "**Un resultado incómodo que reportamos tal cual.**" (L397) | Dramatización | "**El embedding de departamento.**" |
| "la conclusión honesta es que..." (L380); "Lo declaramos:" (L311) | Autoelogio de la honestidad | Quitar "honesta" y "Lo declaramos:" |
| "Optimizar fue intervenir los datos, no la arquitectura." (L447); "Los controles de datos verifican, no seleccionan." (L316); "El rival es la persistencia, no el MLP base" (L233); "no con un umbral ajustado" (L354); "no un reemplazo del análisis de inteligencia" (L438); "no el conteo anual" (Tabla 7); "no con un umbral fijo" (Tabla 7) | Siete contrastes del tipo "X, no Y" | Dejar uno o dos y afirmar en positivo en los demás |
| "Pero no aparece en el desempeño." (L395); "anticipar cambios de régimen sigue siendo la limitación principal." (L392) | Cierres de una línea | Integrarlos en la oración anterior |
| "al dígito" (L244) | Coloquial | "exactamente (diferencia máxima 1,7 × 10⁻¹³)" |
| Negritas en serie: **Fuentes.** **Unidad.** **Representación.** **Selección.** **Prueba.** **Rival.** **Métrica de decisión.** **Regla de retención.** **Reglas 7 y 8.** **La lección de I6.** **I8.**, más todas las viñetas de §6 y §7 | Negrita decorativa en exceso | Pasar a subtítulos h2 o a cursiva y dejar la negrita para 2 o 3 cifras clave |
| "es el argumento cuantitativo para escalar a municipio" (L386) | Sobreafirmación (ver C5) | "sugiere que vale la pena probar la escala municipal" |
| "La Tabla 1 resume qué dejó cada etapa y por qué la tercera se planteó como lo hicimos." (L225) | Mezcla impersonal y primera persona | "...y por qué planteamos así la tercera." |

---

## 6. Verificación de cifras del PDF

Se verificaron todas las cifras de las páginas 1–8 contra los CSV, el cuaderno y `configuracion.json`. **Coinciden
todas, salvo:**

| Cifra en el PDF | Lo que dicen los datos | Hallazgo |
|---|---|---|
| "ningún modelo acierta" las 4 transiciones de 2026 | U2: 1/4 (Magdalena); embedding: 1/4 | M2 |
| "Antes" 0,383 (Fig. 5a) frente a 0,407 (Tabla 6) | Son dos "antes" distintos (media de 5 semillas frente a la semilla 42) | M3 |
| Espacial 0,475 → 0,351 | Individual frente a ensamble (individual U3 0,373; ensamble base 0,445) | M4 |
| 70 % de 23 transiciones | Incluye 2026; sin la prueba, 74 % de 19 | M6 |
| 0,359 frente a 0,377 (vigente en 2025) | Distintas semillas (3 frente a 5) | m7 |
| I8 "LL 0,298 → 0,302" | Es sobre 2024–2025 | m5 |
| Bandas "de la Etapa 1" | Son de la Etapa 2 | m1 |
| "prevalencias de C entre 41 % y 62 %" | Años validados; en entrenamiento ≈ 44 % | m3 |
| Logística "(0,306)" en §3, frente a ΔLL +0,0027 en `decisiones_intervenciones.csv` | El Δ del CSV está mal (M14); 0,306 es correcto | M14 |

Cifras confirmadas (muestra): 17.573; 171/30; 1,7×10⁻¹³; 94,9 % de 136; 7 discrepancias; +5,780 (1,445); 7,1; +0,506;
AUC 0,32; +0,015; +0,007; +0,002; −0,002; −0,012; 0,300/0,307/0,338; 177; 1,25; 393–665; 0,319/0,324/0,309/0,284/0,306;
6 veces (0,053/0,009); 87; 0,973/0,363; 2×10⁻⁶; 0,377/0,360/0,441; 0,341 → 0,279; 0,448 → 0,377; 0,107 → 0,097; toda la
Tabla 5; 31 de 48; 0,423/0,411/0,359; 58 filas; 1,31/3,38; 0,20/0,51/0,15; toda la Tabla 6; −0,103 [−0,185; −0,033];
Brier IC [−0,068; −0,009]; AUC IC [+0,005; +0,105]; −0,144 [−0,270; −0,033]; −0,059 [−0,142; +0,028]; 92 %; −0,049
[−0,162; +0,092]; −0,067 [−0,135; +0,003]; 19 errores; 58 %/2 %/58 %; 0,77–0,85; 2 de 13; −0,002/−0,003; 0,8 de 9; 0,314
y 0,933; 152 KB; 0,304/0,380; PSI 0,29/0,25; 11/11/11; 14,4 %/13,4 %; 25 de 33; 10 → 3; 0,45–0,50; Chocó 0,16 y Córdoba
0,12; matriz de confusión 11/4/1/13.

---

## 7. Cobertura de la guía (Producto 1 y Actividades 1–4)

| Requisito | Dónde está | Estado |
|---|---|---|
| Planteamiento y contexto | §1 | Hay, pero la pregunta está alterada y falta la brecha de escala (M1) |
| Dataset y preprocesamiento | §2 | Bien en controles. Falta el mapeo de subtipos, el balance de clases y x₄–x₇ (M7, M9) |
| Arquitectura y justificación | §3, Tabla 4 | Hay. La justificación del MLP es débil (M12) y falta la pérdida (M8) |
| Entrenamiento y evaluación | §4 | Falta el lazo error → actualización (M8). Protocolo claro, salvo la regla 8 (C1) |
| Optimización y comparación | §5, Tabla 6 | Hay. Comparaciones injustas (M3, M4), I4 (M5), I6 sesgada (C4) |
| Análisis crítico | §6 | Hay. Rival débil (C3), espacial (C5), matriz de confusión sin leer (M7), contradicción (M2) |
| Pipeline integrado | §7, Fig. 8 | Hay. Falta el lazo de entrenamiento en el diagrama (M8); etapas 3 y 4 sobredescritas (k1, k2) |
| Conclusiones y recomendaciones | §9, Tabla 7 | Hay. No responden la pregunta (M1); R1 sobreafirma (C5); falta R6 (m17) |
| Código en repositorio | Encabezado | Enlace privado y no clicable (C6) |
| Act. 1: datos → preprocesamiento → modelo → evaluación + error → pesos | Fig. 8 | Falta error → pesos (M8) |
| Act. 2: escalamiento y normalización, codificación categórica, embeddings, impacto en la estabilidad | §2.2 | El impacto está bien demostrado (sin estandarización no converge). Falta por qué z-score y no min-max, y cómo se codifica el departamento en el embedding (ID entero → vector de 2 dimensiones) |
| Act. 3: métricas, matriz de confusión, desbalance, comparación | §5–§6 | Faltan la matriz de confusión, el balance y las métricas de clasificación (M7) |
| Act. 4: resumen base, diagnóstico, mejoras, justificación, antes/después, conclusión, diagrama | §1, §5, §7 | Bien en diagnóstico → mejora. Justificación incompleta (M5, C4) y antes/después injusto (M3) |

---

## 8. Rúbrica (Actividad 5) → nivel que obtendría hoy → qué falta para Excelente

| Criterio | Nivel hoy (estimado) | Qué falta para Excelente (1,0) |
|---|---|---|
| **R1. Coherencia metodológica integral** | Bueno (0,8) | Citar la pregunta textual y explicar la escala (M1). Declarar que la regla 8 es posterior (C1). Conclusiones que respondan la pregunta de inversión y se apoyen en el rival correcto (C3). Quitar "por coherencia con la U1" como argumento (M12) |
| **R2. Preparación y representación de datos** | Bueno alto (0,8–0,9) | Mapeo de subtipos y operacionalización de x₄–x₇ (M9). Balance de clases (M7). Sacar 2026 del análisis exploratorio (M6). Explicar z-score frente a min-max y la codificación del embedding (Act. 2). Declarar la ventana de d₁ en 2026 (M10.5) y el rezago de publicación de coca y bosque (m20) |
| **R3. Calidad técnica y optimización** | Bueno (0,8) | Pérdida y lazo de entrenamiento (M8). Reformular I6 reconociendo el sesgo a favor de la vigente (C4). Criterio de I4 explícito (M5). Justificar el MLP con evidencia (M12). Corregir `decidir` (M14) |
| **R4. Evaluación e interpretación** | Bueno (0,8). Puede bajar a 0,7 si el profesor detecta C3 o C5 | Persistencia calibrada como rival (C3). Validación espacio-temporal (C5). Antes/después justo (M3, M4). Interpretar la matriz de confusión y las métricas de clasificación, McNemar incluido (M7). Corregir "ningún modelo acierta" (M2). PSI con su ruido (M11). Evaluar la regla de bandas (M10.6) |
| **R5. Entregables y comunicación ejecutiva** | Bueno (0,8). Riesgo de integridad si no se corrige C2 | Declaración de IA fiel (C2). Acceso al repositorio y enlace clicable (C6). Siglas y notación (M9). Sección de equipo con articulación real (M13). Referencias APA 7 completas y citadas (m17). Tablas y figuras citadas en el texto (m16). Redacción propia (§5). Video y sustentación (pendientes) |

**Nota estimada hoy:** ≈ 4,0 / 5,0. **Con las correcciones CRÍTICAS y MAYORES aplicadas:** 4,7–5,0. Las páginas 9–12
libres alcanzan para agregar la matriz de confusión, el lazo de entrenamiento, el recuadro de notación, la fila de la
persistencia calibrada y la evaluación de las bandas sin pasar el límite.

---

## Anexo A. Código de verificación (para reproducir dentro del cuaderno o del pipeline)

Todos los scripts se ejecutan desde cualquier carpeta, con `sys.path.insert(0, "<raíz>/src")`, y solo leen del
proyecto.

**(a) Persistencia calibrada y logística con solo d₁ (hacia adelante, sin mirar el futuro)**
```python
d = pd.read_csv("salidas_u3/dataset_modelado_u3.csv")
logit = lambda p: np.log(np.clip(p, 1e-3, 1 - 1e-3) / (1 - np.clip(p, 1e-3, 1 - 1e-3)))
for v in (2023, 2024, 2025, 2026):                      # 2026: entrenado con 2021-2025
    tr, te = d.anio < v, d.anio == v
    m = LogisticRegression(C=1e6).fit(logit(d.loc[tr, ["m1_sC_t1"]].values), d.loc[tr, "y"])   # o "d1_sC_H2"
    p_te = m.predict_proba(logit(d.loc[te, ["m1_sC_t1"]].values))[:, 1]
# comparar con E.comparar_pareado(y, p_U3, p_rival, grupos=depto)
# -> selección 0,328 (U3 0,304; Δ −0,024 [−0,096; +0,052]); 2026 0,351 (U3 0,380; Δ +0,029 [−0,055; +0,125])
```

**(b) Ensamble de 10 semillas del modelo base U2 (el "antes" justo)**
```python
B10 = replace(BASE, nombre="base10", semillas=tuple(range(1, 11)))
# origen móvil: E.validar(datos, ..., E.particiones_origen_movil(datos)) -> LL 0,376, AUC 0,933
# prueba: entrenar con anio <= 2025, predecir 2026                    -> LL 0,470, AUC 0,862
# espacial (5 grupos, años <= 2025)                                    -> LL 0,445
```

**(c) IC por departamento de I1 e I2** (5 semillas por especificación, media de la log-loss por semilla en cada
remuestreo de departamentos, 1.000 réplicas): m₁ −0,036 [−0,056; −0,019]; d₁ −0,037 [−0,065; −0,011].

**(d) Validación espacial con bloqueo temporal**
```python
for nombre, tr_esp, te_esp in E.particiones_espaciales(datos):
    for v in (2023, 2024, 2025):
        tr = tr_esp & (datos.anio < v).values              # otros departamentos, solo el pasado
        te = te_esp & (datos.anio == v).values
# -> U3 0,361 · persistencia 0,363 · logística 0,416 (n = 87); Δ U3 − pers. −0,002 [−0,114; +0,130]
```

**(e) Ruido del PSI con n = 33:** `psi(ref, rng.choice(ref, 33))`, 2.000 réplicas → mediana ≈ 0,11; p95 0,31–0,35.

**(f) Ventana de d₁ con el año parcial:** d₁ con `acled[acled.semana_anio <= 33]` frente al d₁ con el año completo,
2020–2025 → correlación 0,88; error absoluto medio 0,084.
