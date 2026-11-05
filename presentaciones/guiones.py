"""
Texto de los guiones (lo que dice cada persona en cada diapositiva). Lo usa construir_guiones.py.

Cada diapositiva: (título exacto de data-titulo en el HTML, orador, segundos meta, qué hacer en pantalla, intervenciones)
donde intervenciones es una lista de (orador, texto). Oradores: "laura", "juan".
Las cifras coinciden con las que muestran las diapositivas (salen de salidas_u3/ y referencias_u2/).
"""

VIDEO = [
    ("Portada", "laura", 20, "Nada que hacer: el mapa de fondo se anima solo.", [
        ("laura", "Hola, somos Laura Díaz y Juan Diego Saldarriaga, del curso de Redes Neuronales de la Universidad de La Sabana. "
                  "En este video presentamos nuestro proyecto integrador: un modelo neuronal que anticipa el perfil de amenaza de cada "
                  "departamento de Colombia para orientar en qué tecnología de defensa conviene invertir."),
    ]),
    ("El problema", "laura", 45, "Nada que hacer. Si quieren, señalen con el cursor la cifra de 3,7 billones.", [
        ("laura", "Colombia ejecuta en 2026 la mayor modernización militar de su historia: 3,7 billones de pesos, un 47 % más que en "
                  "2025. Incluye un escudo antidrones, blindados y drones de vigilancia. Lo que no existe es un modelo que diga qué "
                  "tecnología necesita cada territorio. Si se compran antidrones donde la amenaza son las emboscadas, se sobreinvierte; "
                  "si faltan detectores de explosivos donde ese es el riesgo, se subinvierte. Desde la primera unidad definimos cuatro "
                  "perfiles: A, explosivos; B, drones; C, armas convencionales y emboscadas, y D, mixto. Un departamento tiene un perfil "
                  "cuando más de la mitad de sus eventos son de ese tipo."),
    ]),
    ("Datos y formulación", "laura", 50, "Opcional: pasar el cursor por la barra de 2025 para mostrar los conteos.", [
        ("laura", "Para medirlo usamos ACLED, que registra cada semana los eventos de violencia política por departamento: entre 2020 "
                  "y 2026 son 33.275 eventos. La gráfica muestra lo que cambió el planteamiento. Año por año, los explosivos y los "
                  "drones juntos no superan el 7 % de los eventos, y ningún departamento es A ni B. Por eso el problema es binario, C "
                  "contra D, y A y B quedan como una alerta aparte. Predecimos el perfil del año siguiente solo con lo que se sabe al "
                  "cierre del año anterior, que es como se asigna un presupuesto. Con eso armamos un panel de 171 filas de 30 "
                  "departamentos."),
    ]),
    ("El rival: la inercia", "laura", 40, "Pasar el cursor por una celda con borde, por ejemplo Chocó 2026.", [
        ("laura", "Cada fila de este mapa de calor es un departamento y cada columna un año; el color indica cuánto domina lo "
                  "convencional. Casi todo se mantiene: el 86,5 % de las filas repite el perfil del año anterior, y los cambios, "
                  "marcados con borde, son pocos. Por eso el rival más difícil es simplemente copiar el año anterior, lo que llamamos "
                  "persistencia. En la Unidad 2 nuestro modelo base tuvo una log-loss de 0,407, peor que el 0,363 de la persistencia. "
                  "Ese diagnóstico es el punto de partida de esta unidad."),
    ]),
    ("Protocolo experimental", "juan", 45, "Presionar → tres veces mientras habla: aparecen la métrica, la regla y los rivales.", [
        ("juan", "Gracias, Laura. Con tan pocos datos, el mayor riesgo era engañarnos eligiendo el modelo con los mismos datos con que "
                 "lo evaluamos, así que fijamos las reglas antes de ver resultados. Para elegir usamos validación con origen móvil: "
                 "para predecir 2023 entrenamos solo con los años anteriores, y lo mismo para 2024 y 2025. Eso da 87 predicciones "
                 "fuera de muestra. El año 2026 quedó guardado como prueba final y lo miramos una sola vez. La métrica de decisión es "
                 "la log-loss, porque castiga la confianza equivocada, y una mejora solo se retiene si supera un error estándar entre "
                 "cinco semillas pareadas."),
    ]),
    ("Intervenciones I1–I9", "juan", 60, "Pasar el cursor por las filas de I1, I2 y del embedding para mostrar el detalle.", [
        ("juan", "A partir del diagnóstico probamos nueve intervenciones, cada una con una hipótesis. Cada punto del gráfico es el "
                 "cambio en log-loss frente a su referencia, con un intervalo de confianza que remuestrea departamentos; a la "
                 "izquierda del cero, el modelo mejora. Solo dos pasan la regla, y las dos son variables: la memoria del perfil, que "
                 "es la proporción convencional del año anterior suavizada con una contracción bayesiana, y el último semestre, que "
                 "capta los giros que el promedio anual esconde. Bajan la log-loss en 0,036 y 0,037. Los actores armados de UCDP, la "
                 "vecindad, el objetivo blando, la salida multitarea y las redes recurrentes y convolucionales no mejoraron. El "
                 "embedding de departamento sí mejoraba la métrica, pero acertaba menos cambios de perfil y no sirve para un "
                 "territorio nuevo, así que lo documentamos sin adoptarlo."),
    ]),
    ("Modelo final", "juan", 50, "Clic en «propagación» al describir el paso hacia adelante y en «retropropagación» al decir «gradiente».", [
        ("juan", "El modelo final conserva la arquitectura de la Unidad 2: una capa oculta de 16 neuronas ReLU y una salida sigmoide. "
                 "Ahora recibe nueve variables, con las dos nuevas en verde, y promediamos diez redes entrenadas con semillas "
                 "distintas. Cada red se entrena con un bucle explícito: propagación hacia adelante, entropía cruzada ponderada más "
                 "regularización L2, gradiente con GradientTape y actualización con Adam. A la derecha está la escalera de mejora: la "
                 "red base parte en 0,383, la memoria la baja a 0,347, el semestre reciente a 0,310 y el ensamble a 0,304. La línea "
                 "amarilla es la persistencia, y desde el segundo escalón ya la superamos."),
    ]),
    ("Pipeline integrado", "juan", 45, "La terminal se escribe sola al entrar. Si se cortó, clic en «reproducir».", [
        ("juan", "Todo quedó integrado en un paquete de Python con un pipeline de ocho etapas, desde los datos hasta la tarjeta del "
                 "modelo, y se ejecuta con una sola orden. En la terminal se ve la salida real: primero pasan las diez pruebas "
                 "automáticas, que verifican, por ejemplo, que no se filtre información del futuro. Luego el pipeline reproduce las "
                 "cifras del cuaderno y guarda un único artefacto de Keras de 152 kilobytes, con el escalador y las diez redes "
                 "adentro. Con ese artefacto se proyecta 2027 sin volver a entrenar."),
    ]),
    ("Resultados antes y después", "juan", 55, "Clic en «Prueba 2026» al mencionar 2026; opcional, «Departamentos nuevos y años futuros».", [
        ("juan", "En selección, el modelo final tiene la mejor log-loss: 0,304, frente a 0,407 del modelo que entregamos en la Unidad "
                 "2. Para que la comparación fuera justa también promediamos diez redes de la Unidad 2, y ensamble contra ensamble la "
                 "mejora es de 0,072. Esa mejora se sostiene en la prueba de 2026, que no usamos para nada antes: la log-loss baja "
                 "0,090 y el intervalo por departamento no toca el cero. En 2026 el embedding que descartamos queda primero, y lo "
                 "mostramos tal como salió. En la práctica, cuando el modelo pone a un departamento en una banda consolidada acierta "
                 "66 de 71 veces. Laura les cuenta cómo leer esto frente a la inercia."),
    ]),
    ("Lectura crítica", "laura", 45, "Presionar → tres veces para mostrar las tarjetas, una por idea.", [
        ("laura", "Frente a la persistencia cruda ganamos, y hay que decir por qué: esa persistencia usa una proporción como si fuera "
                  "una probabilidad y queda demasiado confiada. Si se calibra la misma inercia, el rival es más duro, y en 2026 logra "
                  "0,351 contra nuestro 0,380. En departamentos que el modelo nunca vio, empatamos con la inercia. El gran límite son "
                  "los cambios de perfil: el modelo acertó 2 de 13 transiciones entre 2023 y 2026. Nuestra conclusión es que la "
                  "ventaja del modelo es de calibración y que, con 30 departamentos, todavía no se demuestra frente a una inercia bien "
                  "calibrada."),
    ]),
    ("Proyección 2027", "laura", 55, "Clic en Guaviare y luego en Cauca para mostrar su detalle a la derecha.", [
        ("laura", "¿Qué significa esto para decidir? Este es el mapa de prioridades de 2027, generado con el artefacto. Once "
                  "departamentos quedan como C consolidado, con probabilidad de 0,7 o más, como Guaviare, Nariño y Cauca; ahí el "
                  "énfasis es inteligencia, vigilancia y reconocimiento, detección temprana y movilidad protegida. Once están en "
                  "transición, entre 0,3 y 0,7, como Chocó, Valle y Arauca, y para ellos recomendamos un portafolio flexible con "
                  "monitoreo trimestral. Los otros once son D consolidado. Guaviare y Cauca activan además la alerta A/B: sus eventos "
                  "con explosivos y drones superan el doble de la proporción nacional, así que se suman antidrones y detección de "
                  "explosivos."),
    ]),
    ("Conclusiones", "ambos", 45, "Presionar → para cada tarjeta: Laura las dos primeras, Juan Diego las dos últimas.", [
        ("laura", "Cerramos con cuatro ideas. La mejora vino de representar mejor los datos, con la memoria y el último semestre, y "
                  "no de una red más grande. Y las probabilidades calibradas permiten decidir con bandas en lugar de un solo umbral."),
        ("juan", "La alerta A/B focaliza los antidrones y la detección de explosivos en Guaviare y Cauca. El siguiente paso es la "
                 "escala municipal: con 30 departamentos el modelo no logra superar a una inercia calibrada, y con municipios "
                 "tendríamos cientos de unidades. El código y el informe están en el repositorio. Muchas gracias."),
    ]),
]

CLASE = [
    ("Portada", "ambos", 30, "Nada que hacer: el mapa de fondo se anima solo.", [
        ("laura", "Buenos días. Somos Laura Díaz y Juan Diego Saldarriaga, y vamos a presentar el proyecto completo del curso: cómo "
                  "pasamos de una pregunta sobre inversión en defensa a un modelo neuronal integrado que anticipa el perfil de "
                  "amenaza de cada departamento."),
        ("juan", "Vamos a recorrer las tres unidades y a mostrar el modelo funcionando. La presentación es interactiva, así que en "
                 "varias diapositivas vamos a explorar los datos en vivo."),
    ]),
    ("Recorrido U1 → U2 → U3", "laura", 40, "Opcional: señalar cada nodo. Si alguien pregunta por una unidad, clic en el nodo lleva a su sección.", [
        ("laura", "El proyecto tuvo tres etapas. En la Unidad 1 formulamos el problema, definimos los perfiles de amenaza y elegimos "
                  "los datos. En la Unidad 2 construimos un modelo base e hicimos un diagnóstico honesto de por qué no era "
                  "suficiente. En la Unidad 3 convertimos ese diagnóstico en intervenciones con hipótesis, elegimos el modelo final con "
                  "un protocolo estricto y lo integramos en un pipeline que proyecta 2027. Todo el tiempo nos guió la misma pregunta: "
                  "qué tecnología conviene desplegar en cada territorio."),
    ]),
    ("Contexto: modernización 2026", "laura", 50, "Nada que hacer.", [
        ("laura", "Empecemos por el contexto. En 2026 Colombia ejecuta la mayor modernización militar de su historia, con 3,7 "
                  "billones de pesos, un 47 % más que el año anterior, según el Ministerio de Defensa. Esa inversión incluye cosas "
                  "muy distintas. El Escudo Nacional Antidrones responde a ataques con drones, que en nuestro esquema es el perfil B. "
                  "Los blindados de INDUMIL y los drones de vigilancia de la CIAC sirven para combates y emboscadas, el perfil C. Y la "
                  "detección de artefactos explosivos es lo que hace falta donde domina el perfil A. Cada tecnología tiene sentido en "
                  "un territorio distinto, y de ahí sale nuestro problema."),
    ]),
    ("Pregunta orientadora", "laura", 50, "Nada que hacer; leer la pregunta con calma.", [
        ("laura", "Por eso la pregunta orientadora, desde la Unidad 1, es esta: cómo orientar la adquisición, la transferencia de "
                  "tecnología y la fabricación de sistemas de defensa según el perfil de amenaza predominante de cada territorio, de "
                  "modo que se optimice la inversión. Hay dos riesgos. Uno es sobreinvertir, por ejemplo comprar antidrones costosos "
                  "donde la amenaza real son los combates. El otro es subinvertir y quedarse sin detección de explosivos donde ese es "
                  "el riesgo principal. Nuestra respuesta es anticipar el perfil de cada departamento para el año siguiente y "
                  "traducir esa predicción en un énfasis tecnológico."),
    ]),
    ("Perfiles A/B/C/D", "laura", 60, "Clic en cada tarjeta para girarla cuando mencione su porcentaje (A y B al final).", [
        ("laura", "Definimos cuatro perfiles a partir de los tipos de evento. El perfil A agrupa explosivos: artefactos "
                  "improvisados, minas y granadas. El B son ataques con drones. El C es la violencia convencional: combates, ataques "
                  "y secuestros. Y el D es mixto, cuando ningún tipo supera la mitad, normalmente porque predominan protestas y "
                  "disturbios. La regla es simple: un territorio tiene el perfil X si más del 50 % de sus eventos son de ese tipo. Al "
                  "girar las tarjetas vemos cuánto pesa cada perfil en el país entre 2020 y 2026: los explosivos son cerca del 4 % y "
                  "los drones menos del 1 %. Ese dato cambió la formulación, como explica Juan Diego."),
    ]),
    ("Datos: ACLED", "juan", 55, "Las cifras se animan solas. Opcional: pasar el cursor por las barras.", [
        ("juan", "Gracias, Laura. La fuente principal es ACLED, una base académica que registra eventos de violencia política. "
                 "Trabajamos con su export semanal por departamento: 17.573 filas que suman 33.275 eventos entre 2020 y 2026, en 33 "
                 "departamentos. A la derecha están los diez subtipos más frecuentes, con el color del perfil al que los asignamos: "
                 "las protestas pacíficas encabezan la lista, seguidas de ataques y combates. Una restricción marcó todo el proyecto: "
                 "el export gratuito llega solo a departamento, sin municipio ni actores. Lo complementamos con coca, cobertura de "
                 "bosque, pobreza multidimensional del DANE y, en la tercera unidad, actores armados de UCDP."),
    ]),
    ("Hallazgos de la U1", "juan", 60, "Opcional: cursor sobre la franja roja y azul de 2025 y 2026 (A y B).", [
        ("juan", "El primer hallazgo está en esta gráfica: año por año, los explosivos y los drones juntos no superan el 7 % de los "
                 "eventos. Ningún departamento tiene perfil A o B dominante, así que en la práctica el problema es binario, C contra "
                 "D, y A y B se vigilan como una alerta de poco volumen y alto impacto. En la Unidad 1 propusimos siete variables: "
                 "eventos, fatalidades, número de grupos armados, coca, bosque, distancia a Bogotá como aproximación a la presencia "
                 "del Estado y pobreza multidimensional. Justificamos usar una red neuronal con una hipótesis de no linealidad: el "
                 "bosque no pesa igual cerca o lejos de Bogotá, ni la coca en un territorio pobre que en uno rico, y un modelo lineal "
                 "no captura esas interacciones."),
    ]),
    ("Formulación operativa", "juan", 60, "Señalar la fórmula y después la figura del diseño temporal.", [
        ("juan", "En la Unidad 2 volvimos esto un problema de aprendizaje supervisado. Para cada departamento y año calculamos la "
                 "proporción de eventos convencionales. La etiqueta es uno si esa proporción es al menos 0,5, perfil C, y cero si es "
                 "D. El modelo estima la probabilidad de C usando solo las variables del año anterior, por dos razones. Evita la "
                 "circularidad, porque las fatalidades de un año vienen de los mismos ataques que definen el perfil, y replica la "
                 "decisión real, porque el presupuesto se asigna con lo observado al cierre del año anterior. Como 33 filas no "
                 "bastaban, armamos un panel departamento-año; solo etiquetamos celdas con al menos 10 eventos, y quedan 171 filas "
                 "de 30 departamentos."),
    ]),
    ("La inercia del perfil", "laura", 60, "Pasar el cursor por Arauca y por una celda con borde (Huila o Chocó).", [
        ("laura", "Este mapa de calor resume el rasgo que condiciona todo el proyecto. Cada fila es un departamento y cada columna "
                  "un año; el ámbar indica dominio de lo convencional y el gris, perfil mixto. Las celdas con borde son cambios de "
                  "perfil, y son pocas: el 86,5 % de las filas repite el perfil del año anterior. Aun así, la proporción de "
                  "departamentos C cambia bastante entre años: fue de 32 % en 2021, el año del Paro Nacional, y llegó a 62 % en 2025, "
                  "con la ruptura de los diálogos y la crisis del Catatumbo. Al pasar el cursor vemos el detalle de cada celda, "
                  "incluida la probabilidad que le dio el modelo final. Esta inercia hace que el rival natural de cualquier modelo sea "
                  "copiar el año anterior."),
    ]),
    ("Modelo base U2", "juan", 70, "Clic en «propagación» y luego en «retropropagación» al explicar el bucle.", [
        ("juan", "El modelo base fue un perceptrón multicapa con siete variables de entrada, una capa oculta de 16 neuronas ReLU y "
                 "una salida sigmoide: 145 parámetros. Lo elegimos entre 16 configuraciones con validación cruzada temporal y la "
                 "regla de un error estándar, que prefiere la opción más simple cuando las diferencias son pequeñas. Entrenamos con "
                 "Adam, tasa de aprendizaje de 10 a la menos 3, regularización L2 y entropía cruzada ponderada por clase. Escribimos "
                 "el bucle de entrenamiento con GradientTape para ver cada paso. En la animación se ve el ciclo: las entradas se "
                 "propagan, se calcula la pérdida, el gradiente viaja hacia atrás y Adam actualiza los pesos. Verificamos que el "
                 "bucle da lo mismo que model.fit, con una diferencia de dos millonésimas. Abajo están las curvas de aprendizaje de "
                 "la Unidad 2."),
    ]),
    ("Resultados y diagnóstico U2", "laura", 60, "Señalar la barra violeta (persistencia) y las verdes (con memoria).", [
        ("laura", "Aquí viene el diagnóstico. En validación temporal, el modelo base tuvo una log-loss de 0,436, y copiar el año "
                  "anterior logró 0,380: nuestro modelo perdía contra la inercia. Encontramos tres problemas y una pista. Había mucha "
                  "varianza, porque con 30 departamentos la red memorizaba la firma de cada territorio. El 57 % de los errores eran "
                  "cambios de perfil. Y Arauca fue C los seis años mientras el modelo le daba una probabilidad de 0,36, porque "
                  "faltaba información de actores armados. La pista: al agregar la memoria del perfil, en el experimento E1, el "
                  "modelo ya superaba a la persistencia, con 0,362. Ese fue el punto de partida de la tercera unidad."),
    ]),
    ("Del diagnóstico a la hoja de ruta", "laura", 45, "Recorrer la tabla de arriba abajo con el cursor.", [
        ("laura", "Cada debilidad se volvió una mejora en la hoja de ruta de la Unidad 2, y cada mejora, una intervención con "
                  "hipótesis en la Unidad 3. La memoria se convirtió en I1 e I2 y se retuvo; el ensamble, I7, también. Los actores "
                  "armados, la calibración y la etiqueta por severidad se probaron y no mejoraron. Para la capacidad de la red hicimos "
                  "una validación anidada y mantuvimos la configuración de la Unidad 2. El mapeo alternativo de subtipos quedó como "
                  "control de sensibilidad. La migración a municipio no fue posible con el export gratuito, así que queda como "
                  "recomendación."),
    ]),
    ("Protocolo experimental", "juan", 70, "Señalar la cuadrícula al explicar el origen móvil; luego recorrer las 8 reglas.", [
        ("juan", "Para la Unidad 3 fijamos el protocolo antes de mirar resultados, porque con tan pocos datos es fácil engañarse. "
                 "La selección se hace con origen móvil: para predecir 2023 se entrena con los años anteriores, y así con 2024 y "
                 "2025. Son 87 predicciones fuera de muestra, y el escalador se ajusta solo con el entrenamiento de cada pliegue. "
                 "2026 quedó como prueba final y la miramos una vez. El rival es la persistencia; después de una revisión agregamos "
                 "también su versión calibrada. La métrica de decisión es la log-loss, y una intervención se retiene solo si mejora "
                 "más de un error estándar entre cinco semillas pareadas sin bajar el AUC. El presupuesto es fijo, de 560 pasos, y "
                 "los controles verifican pero no eligen. La regla 8 dice que el objetivo del proyecto pesa más que la métrica; la "
                 "escribimos cuando apareció el caso del embedding, y así lo declaramos."),
    ]),
    ("Intervenciones: resultado global", "juan", 70, "Cursor sobre I1, I2 y el embedding (violeta) para mostrar el detalle.", [
        ("juan", "Con ese protocolo probamos nueve intervenciones y varias arquitecturas. Cada punto es el cambio en log-loss frente "
                 "a la referencia, promediando cinco semillas, y la línea es el intervalo de confianza del 95 % remuestreando "
                 "departamentos. Lo que queda a la izquierda del cero mejora. Solo dos intervenciones lo logran con claridad: la "
                 "memoria del perfil, que baja la log-loss en 0,036, y el semestre reciente, que la baja en 0,037. Sus intervalos no "
                 "tocan el cero aunque cambiemos la muestra de departamentos. El objetivo blando y los actores armados empeoran. El "
                 "embedding, en violeta, mejora la métrica, pero lo descartamos por la regla 8. El ensamble de diez redes también se "
                 "retuvo; no aparece aquí porque su efecto se mide como estabilidad entre semillas."),
    ]),
    ("Intervenciones: detalle", "laura", 55, "Clic en las pestañas I3, I4 e I8 mientras habla; si sobra tiempo, I7.", [
        ("laura", "Aquí están todas las intervenciones con su diagnóstico, su hipótesis y su resultado. Les muestro tres que no "
                  "funcionaron, porque también enseñan. Los actores armados de UCDP debían resolver el caso de Arauca, pero resultaron "
                  "redundantes con las fatalidades, y además la cobertura de UCDP cambia en 2023. La vecindad entre departamentos "
                  "buscaba mejorar la generalización espacial, y en la validación espacial, que era su hipótesis, no ayudó. La "
                  "calibración tampoco: el umbral óptimo cambiaba cada año, de 0,20 a 0,51 y a 0,15, así que mantuvimos 0,5 y "
                  "agregamos bandas de decisión."),
    ]),
    ("Controles de datos", "juan", 50, "Señalar en la figura las dos barras rojas largas (sin estandarizar y etiquetas barajadas).", [
        ("juan", "También pusimos a prueba la preparación de los datos con siete controles. Sin estandarizar las variables, la red "
                 "no converge y la log-loss sube a 6,09. Con las etiquetas barajadas, el AUC cae a 0,32, lo que confirma que el "
                 "modelo aprende una señal real. Si dejamos que el escalador vea los datos de prueba, el resultado casi no cambia, "
                 "así que la fuga no explica nuestro desempeño. Hubo una sorpresa: quitar los logaritmos mejoraba un poco, 0,012. "
                 "No lo cambiamos por la regla 7, porque los controles verifican y no eligen; queda como recomendación."),
    ]),
    ("Modelo final", "juan", 65, "Clic en «propagación»; luego señalar los escalones de la derecha uno por uno.", [
        ("juan", "El modelo final mantiene la arquitectura de la Unidad 2 con nueve variables de entrada: las siete del modelo base "
                 "más la memoria y el semestre reciente, en verde. Cada red tiene 177 parámetros y promediamos diez. Mantuvimos la "
                 "capacidad de la Unidad 2 porque la rejilla de I6 no ofreció una alternativa estable: lo que se elegía con 2023 y "
                 "2024 empeoraba en 2025. No usamos calibración y el umbral es 0,5, con bandas de decisión. La escalera resume la "
                 "mejora: la red base parte en 0,383; "
                 "con la memoria baja a 0,347; con el semestre reciente, a 0,310, y con el ensamble llega a 0,304. Las líneas "
                 "punteadas son los rivales: desde el segundo escalón superamos a la persistencia, y al final también quedamos por "
                 "debajo de la persistencia calibrada, que marca 0,328."),
    ]),
    ("Curvas y estabilidad", "juan", 50, "Señalar primero la figura de la izquierda y luego la de la derecha.", [
        ("juan", "A la izquierda están las curvas de entrenamiento con el mismo presupuesto de 560 pasos. La pérdida de validación "
                 "en 2025 baja de 0,448 en el modelo base a 0,377 en el modelo final. A la derecha está el efecto del ensamble: con "
                 "una sola red la log-loss depende bastante de la semilla, y a medida que promediamos redes esa dispersión cae. La "
                 "banda entre los percentiles 5 y 95 pasa de 0,053 a 0,009. Por eso retuvimos el ensamble: mejora un poco la "
                 "log-loss y la vuelve mucho más estable."),
    ]),
    ("Pipeline integrado", "juan", 65, "La terminal se escribe sola; señalar las etapas de arriba, que se iluminan en orden.", [
        ("juan", "Todo el proceso vive en un paquete de Python, perfil_amenaza, con un pipeline de ocho etapas: datos, variables, "
                 "evaluación, calibración, prueba, despliegue, proyección y tarjeta del modelo. El cuaderno usa el paquete, pero el "
                 "pipeline funciona solo y se ejecuta con una orden. En la terminal vemos la salida real. Primero corren las diez "
                 "pruebas automáticas, que revisan el contrato de los datos, la fuga temporal, el escalador, el artefacto, el "
                 "determinismo y la réplica de la Unidad 2. Después el pipeline reproduce las cifras del cuaderno, evalúa 2026 y "
                 "guarda un artefacto único de Keras de 152 kilobytes, con la estandarización y las diez redes. Con él se proyecta "
                 "2027 sin reentrenar. Todas las decisiones quedan congeladas en configuracion.json, y la tarjeta del modelo "
                 "documenta su uso previsto, sus límites y cómo vigilar la deriva."),
    ]),
    ("Resultados antes y después", "juan", 70, "Clic en «Prueba 2026», luego en «Departamentos nuevos y años futuros» y al final en la métrica «Exactitud».", [
        ("juan", "Estos son los resultados frente a todos los rivales. En selección, el modelo final tiene la mejor log-loss, 0,304. "
                 "Frente al modelo de la Unidad 2 la mejora es clara: comparando ensamble contra ensamble, en la prueba de 2026 la "
                 "log-loss baja 0,090 y el intervalo por departamento no toca el cero. En esa misma prueba el embedding que descartamos "
                 "por la regla 8 tiene la mejor cifra, y lo dejamos a la vista. Frente a la persistencia cruda también "
                 "mejoramos, con una probabilidad de mejora de 0,92 en selección. Contra la persistencia calibrada el resultado "
                 "depende del año: ganamos en selección y en 2026 perdemos por 0,029, con un intervalo que incluye el cero. En "
                 "departamentos nuevos y años futuros el modelo empata con la persistencia. Y en exactitud, con el umbral de 0,5, la "
                 "persistencia clasifica mejor."),
    ]),
    ("Lectura crítica y errores", "laura", 65, "Señalar el trío de barras de arriba y luego la figura de errores.", [
        ("laura", "¿Cómo leemos esto? Arriba están los tres escenarios lado a lado. El modelo gana a la persistencia cruda porque esa "
                  "persistencia usa una proporción como probabilidad y queda demasiado confiada: la ventaja es de calibración. Contra "
                  "una inercia calibrada, a escala departamental, la ventaja no está demostrada. Abajo están los errores: la mayoría "
                  "son departamentos cerca del umbral o que cambiaron de perfil. El modelo acertó 2 de 13 transiciones entre 2023 y "
                  "2026; el de la Unidad 2 acertaba 3 y la persistencia, ninguna. Hubo una mejora concreta: Arauca, que la Unidad 2 "
                  "clasificaba mal, recibe ahora 0,79 en 2026. Los errores que persisten son las escaladas de 2025 y las "
                  "desescaladas de 2026, como Bolívar, Chocó y Magdalena."),
    ]),
    ("Qué aprendió la red", "laura", 50, "Señalar la barra verde de d₁ y después la tarjeta de la interacción.", [
        ("laura", "¿Qué aprendió la red? Medimos la importancia de cada variable permutándola fuera de muestra. El semestre "
                  "reciente es por mucho la más importante: al desordenarlo, la log-loss sube 0,115, más del doble que cualquier "
                  "otra. Le siguen el bosque, la memoria y las fatalidades, así que las variables estructurales de la Unidad 1 siguen "
                  "contando y modulan la inercia. La red también aprendió una interacción con sentido: con una letalidad alta basta "
                  "una memoria cercana a 0,45 para esperar perfil C, y sin letalidad hace falta cerca de 0,7. Lo tomamos con cautela, "
                  "porque memoria y letalidad están muy correlacionadas y la logística empata."),
    ]),
    ("Proyección 2027", "laura", 65, "Clic en Cauca (ya está seleccionado) y en Guaviare; al final, botón «Histórico» y un par de años.", [
        ("laura", "Con el artefacto proyectamos 2027. Once departamentos son C consolidado, con probabilidad de 0,7 o más: Guaviare, "
                  "Nariño, Cauca, Caquetá, Putumayo y Antioquia, entre otros. Ahí el énfasis es inteligencia, vigilancia y "
                  "reconocimiento, detección temprana y movilidad protegida. Once están en transición, entre 0,3 y 0,7, como Chocó, "
                  "Valle y Arauca, con un portafolio flexible y monitoreo trimestral. Los otros once son D consolidado, sobre todo en "
                  "la región andina y el Caribe. Hay dos alertas A/B, Guaviare y Cauca, donde los explosivos y drones superan el "
                  "doble de la proporción nacional. Al hacer clic en un departamento vemos su detalle, y con el botón Histórico "
                  "recorremos los años y vemos dónde ocurrieron los cambios."),
    ]),
    ("De la probabilidad a la decisión", "laura", 50, "Mover el deslizador a 0,8, activar la alerta A/B y luego bajar a 0,2.", [
        ("laura", "Así se convierte una probabilidad en una decisión. Con el deslizador movemos la probabilidad de un departamento y "
                  "vemos su banda: por debajo de 0,3 es D consolidado, entre 0,3 y 0,7 es transición y desde 0,7 es C consolidado. "
                  "Si se activa la alerta A/B, se suman antidrones y detección de explosivos, y abajo aparecen los departamentos de "
                  "2027 que caen en cada banda. Las bandas funcionan: entre 2023 y 2025, cuando el modelo puso un departamento en "
                  "banda consolidada acertó 66 de 71 veces. A la derecha está el monitoreo de deriva: en 2026 solo la letalidad supera "
                  "el ruido esperado, y por eso conviene reentrenar cada año."),
    ]),
    ("Conclusiones", "ambos", 55, "Presionar → para cada tarjeta: Laura las tres primeras, Juan Diego las tres últimas.", [
        ("laura", "Cerramos con seis ideas. La mejora vino de representar mejor el dato, con la memoria y el último semestre; la "
                  "arquitectura es la de la Unidad 2. Frente a la Unidad 2 la mejora es clara, de 0,470 a 0,380 en 2026 comparando "
                  "ensambles. Y frente a una inercia calibrada la ventaja no está demostrada a escala departamental."),
        ("juan", "En territorios nuevos el modelo transfiere mucho mejor que el de la Unidad 2, aunque empata con la persistencia si "
                 "se respeta el tiempo. El límite son los cambios de perfil: acertamos 2 de 13. Y fuimos transparentes con el "
                 "protocolo: 2026 nunca se usó para elegir y la regla 8 la declaramos como agregada durante el proceso."),
    ]),
    ("Recomendaciones R1–R7", "juan", 45, "Nada que hacer; señalar R1 al empezar.", [
        ("juan", "Para la siguiente iteración proponemos siete recomendaciones. La principal es bajar a escala municipal con el "
                 "registro de ACLED por evento, que requiere acceso académico: pasaríamos de 30 a cientos de unidades, que es lo que "
                 "hoy limita al modelo. Con eso, medir los actores armados por municipio y buscar señales que anticipen las "
                 "transiciones. Además, reentrenar cada año vigilando la deriva, revisar los logaritmos y el mínimo de eventos, "
                 "mantener la persistencia calibrada y la logística como rivales permanentes y reevaluar el embedding con 2027 como "
                 "prueba. Estas salidas son un insumo analítico, no una recomendación operativa."),
    ]),
    ("Equipo, IA y preguntas", "laura", 35, "Nada que hacer; dejar esta diapositiva durante las preguntas.", [
        ("laura", "Sobre el trabajo en equipo: yo me encargué de la interpretación estratégica, la documentación, el informe y el "
                  "guion; Juan Diego, del modelado, el pipeline y la demostración técnica, y las decisiones metodológicas las tomamos "
                  "juntos. Usamos inteligencia artificial como apoyo en la programación y en la revisión de la redacción; la "
                  "propuesta, los datos, el análisis y las conclusiones son nuestros. Muchas gracias, quedamos atentos a sus "
                  "preguntas."),
    ]),
]

PREGUNTAS = [
    ("¿Por qué una red neuronal si la logística empata?", "juan",
     "En el eje temporal empatan (0,304 frente a 0,306). En departamentos nuevos y años futuros la red es mejor (0,361 frente a "
     "0,416), aunque el intervalo incluye el cero. Además, la red aprende una interacción memoria × letalidad coherente con la "
     "hipótesis de la U1. Lo decimos así: la red es defendible, pero su ventaja sobre la logística no está demostrada."),
    ("¿Por qué log-loss y no exactitud?", "juan",
     "Porque decidimos con probabilidades y bandas, no con un solo corte. La log-loss castiga la confianza equivocada. La "
     "exactitud con 29 departamentos por año cambia mucho con un solo acierto o error."),
    ("¿Cómo garantizan que 2026 no se usó para elegir?", "juan",
     "Todo se eligió con origen móvil 2023–2025; la configuración quedó congelada en configuracion.json antes de evaluar 2026, y "
     "la prueba se corrió una vez. Declaramos además que la U2 sí había mirado 2026, para que se tenga en cuenta."),
    ("¿Por qué descartar el embedding si fue mejor en 2026?", "laura",
     "Porque mejoraba la métrica pareciéndose más a la inercia: acertaba 0,8 de 9 transiciones frente a 2,0 de la referencia, y "
     "en departamentos nuevos no aporta. La regla 8 se escribió al ver este caso y así está declarado. Lo reevaluamos en R7."),
    ("¿Qué es la contracción bayesiana?", "juan",
     "Calcular la proporción como (C + 2) / (n + 4), que equivale a un prior Beta(2, 2). Con pocos eventos evita proporciones "
     "extremas como 0 o 1; con muchos eventos casi no cambia nada."),
    ("¿Por qué no se usó x₃ si Arauca lo pedía?", "laura",
     "La intervención I3 con UCDP empeoró la log-loss en 0,025. Los actores estaban muy correlacionados con las fatalidades "
     "(ρ = 0,77) y la cobertura de UCDP cambia en 2023. Arauca se corrigió con la memoria del perfil: ahora recibe 0,79."),
    ("¿Qué pasa con los departamentos con pocos eventos?", "laura",
     "Las celdas con menos de 10 eventos no se etiquetan. En la proyección 2027, Amazonas, Vaupés, San Andrés y Guainía tienen "
     "menos de 10 eventos en 2026 y están marcados para revisión experta."),
    ("¿Por qué ponderar la entropía cruzada por clase?", "juan",
     "Hay 82 filas C y 89 D, pero la prevalencia cambia entre años (32 % a 62 %). El control sin pesos empeora la log-loss en "
     "0,015."),
    ("¿Qué es la persistencia calibrada?", "juan",
     "Una logística de dos parámetros sobre el logit de la memoria m₁, entrenada solo con años anteriores. Toma la misma inercia y "
     "la convierte en una probabilidad bien calibrada. Es el rival más exigente que encontramos."),
    ("¿Cómo se reproduce todo el proyecto?", "juan",
     "Con ejecutar_todo.bat en Windows o con python -m perfil_amenaza.pipeline todo. Las 10 pruebas pasan, el entrenamiento es "
     "determinista y el cuaderno completo corre en unos 20 minutos en CPU."),
    ("¿Sirve esto para decidir operaciones?", "laura",
     "No directamente. Es un insumo analítico para priorizar la inversión tecnológica por territorio; las decisiones operativas "
     "requieren inteligencia y criterio experto que el modelo no tiene."),
]
