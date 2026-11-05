from nbhelp import md, code, lectura

md(r"""
---
# 6. Intervenciones incrementales

Cada experimento entrena la variante con las mismas 5 semillas que su referencia y la evalúa en los tres pliegues
de origen móvil. Se reportan la **media entre semillas** (lo que vale una red individual) y la decisión de la
regla 5 de §3.1. El ensamble de semillas se estudia aparte en I7, para no confundir su efecto con el de las
variables.
""")

md(r"""
## 6.1 I1: memoria del perfil
""")

code(r"""
# =============================================================================
# 6.1 I1 memoria: m1 (s_C contraída de t-1) y la trayectoria m1, m2, m3
# =============================================================================
DECISIONES, ESCALERA = [], [("Modelo base U2", "MLP base U2")]
evaluar(P.Especificacion("I1a +m1", V.BASE + ["m1_sC_t1"], CFG_U2))
evaluar(P.Especificacion("I1b +m1 m2 m3", V.BASE + V.MEMORIA, CFG_U2))
VARS, REF = list(V.BASE), "MLP base U2"
d = decidir("I1a +m1", REF, "I1 memoria m1"); DECISIONES.append(d)
if d["Decisión"] == "se retiene":
    VARS, REF = VARS + ["m1_sC_t1"], "I1a +m1"; ESCALERA.append(("+ I1 memoria m₁", REF))
d = decidir("I1b +m1 m2 m3", REF, "I1 trayectoria m2, m3"); DECISIONES.append(d)
if d["Decisión"] == "se retiene":
    VARS, REF = list(V.BASE) + V.MEMORIA, "I1b +m1 m2 m3"; ESCALERA.append(("+ I1 trayectoria", REF))
tabla(resumen(["MLP base U2", "U2 E1 (memoria cruda x8)", "I1a +m1", "I1b +m1 m2 m3"]), dec=3)
display(pd.DataFrame(DECISIONES).style.format({"ΔLog-loss": "{:+.4f}", "EE": "{:.4f}", "ΔAUC": "{:+.4f}"}).hide(axis="index"))
print("Variables retenidas tras I1:", [v for v in VARS if v not in V.BASE])
""")

lectura("i1")

md(r"""
## 6.2 I2: dinámica intra-anual
""")

code(r"""
# =============================================================================
# 6.2 I2 dinámica intra-anual sobre la especificación retenida
# =============================================================================
evaluar(P.Especificacion("I2a +d1", VARS + ["d1_sC_H2"], CFG_U2))
evaluar(P.Especificacion("I2b +d1 d2 d3", VARS + V.DINAMICA, CFG_U2))
ref_i2 = REF
d = decidir("I2a +d1", REF, "I2 semestre reciente d1"); DECISIONES.append(d)
if d["Decisión"] == "se retiene":
    VARS, REF = VARS + ["d1_sC_H2"], "I2a +d1"; ESCALERA.append(("+ I2 semestre d₁", REF))
d = decidir("I2b +d1 d2 d3", REF, "I2 tendencias d2, d3"); DECISIONES.append(d)
if d["Decisión"] == "se retiene":
    VARS, REF = [v for v in VARS if v != "d1_sC_H2"] + V.DINAMICA, "I2b +d1 d2 d3"; ESCALERA.append(("+ I2 tendencias", REF))
tabla(resumen([ref_i2, "I2a +d1", "I2b +d1 d2 d3"]), dec=3)
display(pd.DataFrame(DECISIONES[-2:]).style.format({"ΔLog-loss": "{:+.4f}", "EE": "{:.4f}", "ΔAUC": "{:+.4f}"}).hide(axis="index"))

# ¿Dónde aporta? Log-loss por año y aciertos en transiciones
por_anio = pd.DataFrame({k: RES[k]["oos"].assign(p=RES[k]["P"].mean(0)).groupby("particion").apply(lambda g: E.bce(g.y, g.p))
                         for k in ["MLP base U2", "I1a +m1", "I2a +d1"]})
por_anio["Persistencia"] = PERS.groupby("particion").apply(lambda g: E.bce(g.y, g.p))
tabla(por_anio.T.rename(columns=str), dec=3)
print("Variables retenidas tras I2:", [v for v in VARS if v not in V.BASE])
""")

lectura("i2")

md(r"""
## 6.3 I3: actores armados activos ($x_3$, UCDP GED)

$x_3$ está pendiente desde la Etapa 1, porque el export gratuito de ACLED no trae actores. La fuente sustituta
verificada es **UCDP GED 26.1** (`ged261-csv.zip`, archivo `GEDEvent_v26_1.csv`, 39 MB). Cubre Colombia hasta 2025,
trae `adm_1` en el 99,8 % de los eventos desde 2020 y registra los dos bandos (`side_a`, `side_b`). La variable cuenta
los actores **no estatales** distintos por departamento–año: excluye gobiernos y "Civilians" y separa pares como
"ELN, FARC-EMC". Como UCDP solo registra violencia organizada letal, es un indicador **aproximado** de grupos
activos.

Como el archivo completo pesa 39 MB, el repositorio incluye `herramientas/preparar_ucdp.py`. El script lo reduce a
Colombia (`ucdp_colombia.csv`, unos cientos de KB) y a $x_3$ ya agregado (`ucdp_actores_departamento_anio.csv`, unos
pocos KB). La celda siguiente corre I3 si cualquiera de esos archivos está en `datos/externos/`. Si no está, lo declara
y sigue: el resto del cuaderno no depende de esta intervención.
""")

code(r"""
# =============================================================================
# 6.3 I3 actores armados (condicionada a la disponibilidad de UCDP GED)
# =============================================================================
if ACTORES is not None and "x3_actores" in datos:
    evaluar(P.Especificacion("I3 +x3", VARS + ["x3_actores"], CFG_U2))
    d = decidir("I3 +x3", REF, "I3 actores armados x3"); DECISIONES.append(d)
    tabla(resumen([REF, "I3 +x3"]), dec=3)
    display(pd.DataFrame([d]).style.hide(axis="index"))
    ar = RES["I3 +x3"]["oos"].assign(p_x3=RES["I3 +x3"]["P"].mean(0), p_ref=RES[REF]["P"].mean(0)).query("depto == 'arauca'")
    display(ar[["departamento", "anio", "perfil", "p_ref", "p_x3"]])
    if d["Decisión"] == "se retiene":
        VARS, REF = VARS + ["x3_actores"], "I3 +x3"; ESCALERA.append(("+ I3 actores x₃", REF))
    I3_ESTADO = "evaluada"
else:
    I3_ESTADO = "pendiente de fuente"
    DECISIONES.append({"Intervención": "I3 actores armados x3", "Referencia": REF, "ΔLog-loss": np.nan, "EE": np.nan,
                       "ΔAUC": np.nan, "Decisión": "pendiente: UCDP GED no disponible en esta ejecución"})
    display(Markdown("**I3 no se ejecutó:** `GEDEvent_v26_1.csv` no está en `datos/externos/`. Descárguelo de "
                     "https://ucdp.uu.se/downloads/ged/ged261-csv.zip, cópielo en esa carpeta y vuelva a ejecutar el cuaderno."))
""")

lectura("i3")

md(r"""
## 6.4 I4: vecindad espacial

La hipótesis de I4 es **espacial**: la composición de los vecinos debería ayudar sobre todo con departamentos que el
modelo nunca vio. Por eso la evaluamos con los dos protocolos de la U2, el temporal y el espacial (5 pliegues
agrupados por departamento, años hasta 2025). Se retiene si mejora la validación espacial según la regla 5 sin
empeorar la temporal.
""")

code(r"""
# =============================================================================
# 6.4 I4 vecindad: validación temporal y espacial
# =============================================================================
spec_ref, ref_i4 = RES[REF]["spec"], REF
evaluar(P.Especificacion("I4 +v1 v2", VARS + V.VECINDAD, CFG_U2))
d_t = decidir("I4 +v1 v2", REF, "I4 vecindad (temporal)")
for k, s in [("esp: MLP base U2", BASE_U2), ("esp: referencia", spec_ref), ("esp: I4 +v1 v2", RES["I4 +v1 v2"]["spec"]),
             ("esp: logística base", LOG_BASE), ("esp: logística referencia", P.Especificacion("lr", VARS, None))]:
    evaluar(replace(s, nombre=k), particiones="espacial", clave=k)
PERS_E = persistencia_oos("espacial")
d_e = decidir("esp: I4 +v1 v2", "esp: referencia", "I4 vecindad (espacial)")
retener_i4 = d_e["Decisión"] == "se retiene" and d_t["ΔLog-loss"] < d_t["EE"]
d_t["Decisión"] = d_e["Decisión"] = "se retiene" if retener_i4 else "no se retiene"
DECISIONES += [d_t, d_e]
if retener_i4:
    VARS, REF = VARS + V.VECINDAD, "I4 +v1 v2"; ESCALERA.append(("+ I4 vecindad", REF))
esp = resumen(["esp: MLP base U2", "esp: logística base", "esp: referencia", "esp: logística referencia", "esp: I4 +v1 v2"])
M_PERS_E = E.metricas(PERS_E.y, PERS_E.p)
esp.loc["esp: persistencia", E.COLS_PRINCIPALES] = [M_PERS_E[c] for c in E.COLS_PRINCIPALES]
tabla(pd.concat({"Temporal 2023-2025": resumen([ref_i4, "I4 +v1 v2"])[E.COLS_PRINCIPALES],
                 "Espacial (deptos. nunca vistos)": esp[E.COLS_PRINCIPALES]}), dec=3)
display(pd.DataFrame([d_t, d_e]).style.format({"ΔLog-loss": "{:+.4f}", "EE": "{:.4f}", "ΔAUC": "{:+.4f}"}).hide(axis="index"))
VARS_FINAL, REF_VARS = list(VARS), REF
print(f"Especificación de variables fijada ({len(VARS_FINAL)}):", VARS_FINAL)
""")

lectura("i4")

md(r"""
## 6.5 I5: objetivo blando ponderado por eventos

En lugar de la etiqueta dura $y = \mathbb{1}[s^C \ge 0{,}5]$, la red aprende directamente $s^C_{d,t}$, con la
entropía cruzada aplicada a un objetivo continuo, opcionalmente ponderada por $\sqrt{n_{d,t}}$. La idea es que una
celda con $s^C = 0{,}51$ no le "enseñe" lo mismo que una con $s^C = 0{,}95$. La salida pasa a estimar la
**proporción esperada**, no la probabilidad de que la proporción supere 0,5. Por eso la comparación justa es después
de **calibrar** ambas variantes con el mismo método (temperatura con encadenamiento hacia adelante: el pliegue $v$
se calibra solo con los pliegues anteriores).
""")

code(r"""
# =============================================================================
# 6.5 I5 objetivo blando (con y sin ponderación por eventos), crudo y calibrado
# =============================================================================
def calibrar_adelante(oos, p, metodo="temperatura"):
    # Calibración con encadenamiento hacia adelante (modelo.calibrar_adelante): el pliegue v se calibra con los pliegues < v.
    return M.calibrar_adelante(oos["particion"].values, p, oos["y"].values, metodo)

def registrar_calibrado(clave, metodo="temperatura"):
    r = RES[clave]; oos = r["oos"]
    Pc = np.array([calibrar_adelante(oos, p, metodo) for p in r["P"]])
    filas = [{"semilla": s, **E.metricas(oos.y, pc, transicion=oos.transicion)} for s, pc in zip(r["por_semilla"].index, Pc)]
    RES[clave + " · cal"] = {"oos": oos.assign(p=Pc.mean(0)), "por_semilla": pd.DataFrame(filas).set_index("semilla"), "P": Pc, "spec": r["spec"]}

CFG_REF = RES[REF_VARS]["spec"].red
evaluar(P.Especificacion("I5 blando", VARS_FINAL, replace(CFG_REF, objetivo="blando")))
evaluar(P.Especificacion("I5 blando + peso eventos", VARS_FINAL, replace(CFG_REF, objetivo="blando", peso_eventos="raiz")))
for k in (REF_VARS, "I5 blando", "I5 blando + peso eventos"):
    registrar_calibrado(k)
d_crudo = decidir("I5 blando", REF_VARS, "I5 objetivo blando (sin calibrar)")
d_cal = decidir("I5 blando · cal", REF_VARS + " · cal", "I5 objetivo blando (ambos calibrados)")
DECISIONES += [d_crudo, d_cal]
OBJETIVO = "blando" if d_cal["Decisión"] == "se retiene" else "duro"
tabla(resumen([REF_VARS, "I5 blando", "I5 blando + peso eventos", REF_VARS + " · cal", "I5 blando · cal", "I5 blando + peso eventos · cal"]), dec=3)
display(pd.DataFrame([d_crudo, d_cal]).style.format({"ΔLog-loss": "{:+.4f}", "EE": "{:.4f}", "ΔAUC": "{:+.4f}"}).hide(axis="index"))
print("Objetivo de entrenamiento elegido:", OBJETIVO)
""")

lectura("i5")

md(r"""
## 6.6 I9: salida multitarea de composición A/B/C/D

La etiqueta binaria diluye el hallazgo crítico de la U1: explosivos (A) y drones (B) son de bajo volumen y alto
impacto. Agregamos a la red una **segunda cabeza softmax** que predice la composición completa
$(s^A, s^B, s^C, s^D)_{d,t}$, con un tronco oculto compartido:

$$
\mathcal{L} = \underbrace{\mathrm{BCE}(y, \hat p)}_{\text{perfil C}} \;+\; \lambda \cdot
\underbrace{\sum_k s^k \log \frac{s^k}{\hat s^k}}_{\mathrm{KL}(\mathbf s \,\|\, \hat{\mathbf s})} \;+\; \lambda_2 \lVert \mathbf W \rVert^2 .
$$

Se evalúa de dos formas: (1) si la tarea auxiliar ayuda o perjudica a $\hat p$ (regla 5) y (2) si la composición
predicha anticipa la **alerta A/B** mejor que repetir la composición del año anterior.
""")

code(r"""
# =============================================================================
# 6.6 I9 multitarea: efecto sobre P(C) y calidad de la composición predicha
# =============================================================================
for lam in (0.5, 1.0):
    evaluar(P.Especificacion(f"I9 multitarea λ={lam:g}", VARS_FINAL, replace(CFG_REF, tipo="multitarea", lambda_comp=lam, objetivo=OBJETIVO)))
d9 = [decidir(f"I9 multitarea λ={lam:g}", REF_VARS, f"I9 multitarea λ={lam:g} (efecto en P(C))") for lam in (0.5, 1.0)]

# Composición fuera de muestra (ensamble de 5 semillas por pliegue)
spec9 = replace(RES["I9 multitarea λ=0.5"]["spec"], semillas=SEM5)
comp_oos = []
for v, tr, te in E.particiones_origen_movil(datos):
    m_, e_ = P.entrenar_especificacion(datos, SEQ, spec9, tr)
    comp_oos.append(P.predecir_especificacion(m_, e_, datos, SEQ, spec9, te, composicion=True))
comp_oos = np.concatenate(comp_oos)
te_ = datos.anio.isin(C.ANIOS_SELECCION).values
real = datos.loc[te_, V.COMPOSICION].values
prev_ab = datos.loc[te_, ["depto", "anio"]].merge(pext[["depto", "anio", "s_A", "s_B"]].assign(anio=lambda d: d.anio + 1), on=["depto", "anio"], how="left")
sAB_real, sAB_pers, sAB_mod = real[:, 0] + real[:, 1], (prev_ab.s_A + prev_ab.s_B).fillna(0).values, comp_oos[:, 0] + comp_oos[:, 1]
umbral_ab = pd.Series(sAB_real).groupby(datos.loc[te_, "anio"].values).transform(lambda s: 2 * s.mean()).values
alto_ab = sAB_real >= umbral_ab
calidad = pd.DataFrame({
    "Modelo multitarea (λ=0,5)": [np.mean(np.abs(sAB_mod - sAB_real)), stats.spearmanr(sAB_mod, sAB_real).correlation, roc_auc_score(alto_ab, sAB_mod)],
    "Persistencia de la composición": [np.mean(np.abs(sAB_pers - sAB_real)), stats.spearmanr(sAB_pers, sAB_real).correlation, roc_auc_score(alto_ab, sAB_pers)]},
    index=["MAE de s_A + s_B", "ρ de Spearman con s_A + s_B real", "AUC para detectar alerta A/B (≥ 2× media del año)"])
I9_COMPOSICION_GANA = calidad.iloc[2, 0] > calidad.iloc[2, 1] and calidad.iloc[0, 0] < calidad.iloc[0, 1]
for x in d9:
    if x["Decisión"] == "se retiene" and not I9_COMPOSICION_GANA:
        x["Decisión"] = "no se retiene (no aporta en P(C) ni en A/B)"
    elif x["Decisión"] != "se retiene" and I9_COMPOSICION_GANA and x["ΔLog-loss"] < x["EE"]:
        x["Decisión"] = "se retiene (no empeora P(C) y anticipa A/B)"
DECISIONES += d9
TIPO = "multitarea" if any(x["Decisión"].startswith("se retiene") for x in d9) else "mlp"
CFG_ACTUAL = RES["I9 multitarea λ=0.5"]["spec"].red if TIPO == "multitarea" else replace(CFG_REF, objetivo=OBJETIVO)
tabla(resumen([REF_VARS, "I9 multitarea λ=0.5", "I9 multitarea λ=1"]), dec=3)
tabla(calidad, dec=3)
display(pd.DataFrame(d9).style.format({"ΔLog-loss": "{:+.4f}", "EE": "{:.4f}", "ΔAUC": "{:+.4f}"}).hide(axis="index"))
print(f"Filas con alerta A/B real en 2023-2025: {alto_ab.sum()} de {len(alto_ab)} | tipo de red elegido: {TIPO}")
""")

lectura("i9")

md(r"""
## 6.7 Comparativos de arquitectura: ¿una red secuencial o con identidad territorial?

Con las mismas variables tabulares, comparamos el MLP con:

* **GRU** y **CNN 1D** sobre la secuencia de los 4 trimestres de $t-1$ (eventos, fatalidades, $\tilde s^A$,
  $\tilde s^B$, $\tilde s^C$), concatenada a las variables tabulares. Responde si la forma de la trayectoria
  trimestral aporta más que su resumen semestral ($d_1$).
* **Embedding de departamento** (dimensión 2): un vector aprendido por territorio. Captura efectos fijos, pero no
  puede generalizar a departamentos nunca vistos, así que lo evaluamos también en la validación espacial.
* **MLP profundo** (16, 8) y **regresión logística** con las mismas variables (¿aporta la no linealidad?).
""")

code(r"""
# =============================================================================
# 6.7 Comparativos: GRU, CNN 1D, embedding, MLP profundo y logística con las mismas variables
# =============================================================================
comparativos = {
    "GRU trimestral": replace(CFG_REF, tipo="gru", unidades_seq=8),
    "CNN 1D trimestral": replace(CFG_REF, tipo="cnn", unidades_seq=8),
    "Embedding de departamento": replace(CFG_REF, tipo="embedding", emb_dim=2),
    "MLP profundo (16, 8)": replace(CFG_REF, capas=(16, 8)),
}
for k, cfg_k in comparativos.items():
    evaluar(P.Especificacion(k, VARS_FINAL, cfg_k))
LOG_F = P.Especificacion("Logística (mismas variables)", VARS_FINAL, None)
evaluar(LOG_F)
d7 = [decidir(k, REF_VARS, k) for k in [*comparativos, "Logística (mismas variables)"]]
emb = next(x for x in d7 if x["Intervención"] == "Embedding de departamento")
TR_EMB = RES["Embedding de departamento"]["por_semilla"]["Transiciones acertadas"].mean()
TR_REF = RES[REF_VARS]["por_semilla"]["Transiciones acertadas"].mean()
if emb["Decisión"] == "se retiene" and TR_EMB < TR_REF:
    emb["Decisión"] = (f"cumple la regla 5, no se adopta por la regla 8: acierta {TR_EMB:.1f} de {int(M_PERS['Transiciones'])} "
                       f"transiciones (referencia {TR_REF:.1f}) y en departamentos nuevos se reduce al MLP sin embedding")
DECISIONES += d7
evaluar(P.Especificacion("esp: embedding", VARS_FINAL, comparativos["Embedding de departamento"]), particiones="espacial", clave="esp: embedding")
tab7 = resumen([REF_VARS, *comparativos, "Logística (mismas variables)"])
tab7["Parámetros"] = [RES[k]["spec"].red and M.n_parametros(M.construir_red(len(VARS_FINAL), RES[k]["spec"].red, SEQ.shape[1:] if RES[k]["spec"].red.tipo in ("gru", "cnn") else None)) or len(VARS_FINAL) + 1 for k in tab7.index]
tabla(tab7, dec=3)
display(pd.DataFrame(d7).style.format({"ΔLog-loss": "{:+.4f}", "EE": "{:.4f}", "ΔAUC": "{:+.4f}"}).hide(axis="index"))
print(f"Embedding en validación espacial: log-loss {RES['esp: embedding']['por_semilla']['Log-loss'].mean():.3f} "
      f"vs. MLP sin embedding {RES['esp: referencia']['por_semilla']['Log-loss'].mean():.3f}")
""")

lectura("comparativos")

md(r"""
## 6.8 I6: capacidad, activación, regularización y presupuesto, con validación anidada

La U2 dejó dos señales contradictorias: 4 neuronas generalizaban mejor con 400 épocas fijas, pero la búsqueda con
parada temprana eligió 16. Exploramos una rejilla de **48 configuraciones**: 4 capacidades × 2 L2 × 2 activaciones
(ReLU, tanh) × 3 presupuestos (280, 560 y 1.120 pasos), con 3 semillas, sobre las variables ya fijadas. Además
hacemos una **validación anidada** que pone a prueba el propio proceso de selección: elegimos la configuración con
los pliegues 2023–2024 y medimos si lo elegido es mejor en 2025, el año que no participó en la elección.
""")

code(r"""
# =============================================================================
# 6.8 I6 rejilla de capacidad x L2 x activación x presupuesto (3 semillas, 3 pliegues)
# =============================================================================
t0 = time.time(); filas = []
REJILLA = [(c, l2, act, pasos) for c in [(4,), (8,), (16,), (32,)] for l2 in [1e-3, 1e-2] for act in ["relu", "tanh"] for pasos in [280, 560, 1120]]
for capas, l2, act, pasos in REJILLA:
    cfg_g = replace(CFG_ACTUAL, capas=capas, l2=l2, activacion=act, pasos=pasos)
    for s in (1, 2, 3):
        sp = P.Especificacion("g", VARS_FINAL, cfg_g, semillas=(s,))
        oos = E.validar(datos, P.ajustar_predecir(datos, SEQ, sp), E.particiones_origen_movil(datos))
        for v, g in oos.groupby("particion"):
            filas.append({"capas": str(capas), "L2": l2, "activación": act, "pasos": pasos, "semilla": s, "año": int(v), "LL": E.bce(g.y, g.g)})
REJ = pd.DataFrame(filas)
print(f"{len(REJILLA)} configuraciones × 3 semillas × 3 pliegues = {len(REJILLA) * 9} entrenamientos en {time.time() - t0:.0f} s")
K_ = ["capas", "L2", "activación", "pasos"]
N_PAR = {c: M.n_parametros(M.construir_red(len(VARS_FINAL), replace(CFG_ACTUAL, capas=eval(c)))) for c in REJ.capas.unique()}

def seleccionar(df):
    a = df.groupby(K_)["LL"].agg(media="mean", ee=lambda s: s.std(ddof=1) / np.sqrt(len(s))).reset_index()
    a["parámetros"] = a.capas.map(N_PAR)
    mejor = a.loc[a.media.idxmin()]
    cand = a[a.media <= mejor.media + mejor.ee].sort_values(["parámetros", "pasos", "media"])
    return a.sort_values("media"), mejor, cand.iloc[0], len(cand)

TAB_REJ, MEJOR, UNO_EE, N_CAND = seleccionar(REJ)
DEF = (str(CFG_ACTUAL.capas), CFG_ACTUAL.l2, CFG_ACTUAL.activacion, CFG_ACTUAL.pasos)
ll_def = TAB_REJ.set_index(K_).loc[DEF]
print(f"Mejor media: {tuple(MEJOR[K_])} LL {MEJOR.media:.3f} ± {MEJOR.ee:.3f} | configuraciones dentro de 1 EE: {N_CAND} de {len(TAB_REJ)}")
print(f"Configuración vigente {DEF}: LL {ll_def.media:.3f} (puesto {int((TAB_REJ.media < ll_def.media).sum()) + 1} de {len(TAB_REJ)}; "
      f"{'dentro' if ll_def.media <= MEJOR.media + MEJOR.ee else 'fuera'} de 1 EE del mejor)")

# Validación anidada: seleccionar con 2023-2024 y medir en 2025
_, m_in, u_in, _ = seleccionar(REJ[REJ["año"] < 2025])
ll25 = REJ[REJ["año"] == 2025].groupby(K_)["LL"].agg(["mean", "std"])
ANIDADA = pd.DataFrame({
    "Configuración": [str(tuple(m_in[K_])), str(tuple(u_in[K_])), str(DEF), str(tuple(ll25["mean"].idxmin()))],
    "LL 2023-2024 (selección)": [m_in.media, u_in.media, REJ[REJ["año"] < 2025].groupby(K_)["LL"].mean().loc[DEF], np.nan],
    "LL 2025 (fuera de la selección)": [ll25.loc[tuple(m_in[K_]), "mean"], ll25.loc[tuple(u_in[K_]), "mean"], ll25.loc[DEF, "mean"], ll25["mean"].min()],
}, index=["Mínimo en 2023-2024", "Regla 1 EE en 2023-2024", "Vigente (U2)", "Oráculo (mejor en 2025, inalcanzable)"])
tabla(ANIDADA, dec=3)
SELECCION_TRANSFIERE = min(ANIDADA.iloc[0, 2], ANIDADA.iloc[1, 2]) < ANIDADA.iloc[2, 2]
CFG_I6 = CFG_ACTUAL if not SELECCION_TRANSFIERE else replace(CFG_ACTUAL, capas=eval(UNO_EE.capas), l2=float(UNO_EE.L2), activacion=UNO_EE["activación"], pasos=int(UNO_EE.pasos))
DECISIONES.append({"Intervención": "I6 capacidad / activación / L2 / presupuesto", "Referencia": str(DEF), "ΔLog-loss": ANIDADA.iloc[1, 2] - ANIDADA.iloc[2, 2],
                   "EE": np.nan, "ΔAUC": np.nan, "Decisión": "se retiene" if SELECCION_TRANSFIERE else "no se retiene (la selección no transfiere a 2025)"})

# Tasa de aprendizaje (con el presupuesto vigente)
for lr in (3e-4, 3e-3):
    evaluar(P.Especificacion(f"I6 lr={lr:g}", VARS_FINAL, replace(CFG_I6, lr=lr)))
tabla(resumen([REF_VARS, "I6 lr=0.0003", "I6 lr=0.003"])[["ROC-AUC", "Log-loss", "Log-loss (DE)", "Brier"]], dec=3)

fig, axes = plt.subplots(1, 3, figsize=(16, 4.4), gridspec_kw={"width_ratios": [1, 1, 1.15]})
for ax, act in zip(axes[:2], ["relu", "tanh"]):
    m_ = REJ[REJ["activación"] == act].groupby(["capas", "pasos"])["LL"].mean().unstack().loc[["(4,)", "(8,)", "(16,)", "(32,)"]]
    im = ax.imshow(m_.values, cmap=G.CMAP_P.reversed(), vmin=TAB_REJ.media.min(), vmax=.42, aspect="auto")
    for i in range(m_.shape[0]):
        for j in range(m_.shape[1]):
            ax.text(j, i, f"{m_.values[i, j]:.3f}", ha="center", va="center", fontsize=9, color="white" if m_.values[i, j] < .31 else TINTA)
    ax.set_xticks(range(3), [f"{p} pasos" for p in m_.columns]); ax.set_yticks(range(4), [f"{c} · {N_PAR[c]} par." for c in m_.index])
    ax.grid(False); ax.set_title(f"({'ab'[act == 'tanh']}) Log-loss media · {act.upper()} (promedio de L2)")
ax = axes[2]
x = np.arange(len(ANIDADA) - 1)
ax.bar(x - .18, ANIDADA.iloc[:3, 1], width=.34, color=COL["D"], label="2023-2024 (donde se elige)")
ax.bar(x + .18, ANIDADA.iloc[:3, 2], width=.34, color=COL["val"], label="2025 (fuera de la elección)")
ax.axhline(ANIDADA.iloc[3, 2], color=TINTA, lw=1, ls=":"); ax.text(2.45, ANIDADA.iloc[3, 2], " oráculo 2025", fontsize=8, va="bottom", ha="right")
ax.set_xticks(x, ["Mínimo\n2023-24", "Regla 1 EE\n2023-24", "Vigente\n(U2)"]); ax.set_ylabel("Log-loss")
ax.set_title("(c) Validación anidada del proceso de selección"); ax.legend(fontsize=8, loc="upper left")
fig.tight_layout(); guardar(fig, "fig04_rejilla_anidada"); plt.show()
""")

lectura("i6")

md(r"""
## 6.9 I7: ensamble de semillas
""")

code(r"""
# =============================================================================
# 6.9 I7 ensamble: 20 semillas de la especificación vigente y curva de tamaño del ensamble
# =============================================================================
SPEC_I7 = P.Especificacion("I7 base de semillas", VARS_FINAL, CFG_I6)
r20 = evaluar(SPEC_I7, semillas=tuple(range(1, 21)), clave="I7 20 semillas")
P20, y_oos = r20["P"], r20["oos"]["y"].values
rng_k = np.random.default_rng(C.SEED); curva = []
for k in (1, 2, 3, 5, 7, 10, 15, 20):
    ll = [E.bce(y_oos, P20[rng_k.choice(20, k, replace=False)].mean(0)) for _ in range(200 if k < 20 else 1)]
    au = [roc_auc_score(y_oos, P20[rng_k.choice(20, k, replace=False)].mean(0)) for _ in range(200 if k < 20 else 1)]
    curva.append({"K": k, "LL media": np.mean(ll), "LL p5": np.percentile(ll, 5), "LL p95": np.percentile(ll, 95), "AUC media": np.mean(au), "LL DE": np.std(ll)})
CURVA_K = pd.DataFrame(curva).set_index("K")
K_ENS = 10
desv_ind = P20.std(0)
unan = np.mean(((P20 >= .5).mean(0) == 0) | ((P20 >= .5).mean(0) == 1))
tabla(CURVA_K, dec=4)
print(f"Dispersión de P(C) entre semillas por fila: mediana {np.median(desv_ind):.3f}, máx. {desv_ind.max():.3f} | "
      f"filas con predicción unánime entre 20 redes: {unan:.0%}")
RANGO_K = (CURVA_K.loc[1, "LL p95"] - CURVA_K.loc[1, "LL p5"], CURVA_K.loc[K_ENS, "LL p95"] - CURVA_K.loc[K_ENS, "LL p5"])
DECISIONES.append({"Intervención": f"I7 ensamble de {K_ENS} semillas", "Referencia": "1 red", "ΔLog-loss": CURVA_K.loc[K_ENS, "LL media"] - CURVA_K.loc[1, "LL media"],
                   "EE": np.nan, "ΔAUC": CURVA_K.loc[K_ENS, "AUC media"] - CURVA_K.loc[1, "AUC media"], "Decisión": "se retiene (estabilidad)"})

fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
ax = axes[0]
ax.fill_between(CURVA_K.index, CURVA_K["LL p5"], CURVA_K["LL p95"], color=COL["u3"], alpha=.18, lw=0, label="banda 5-95 % entre sorteos")
ax.plot(CURVA_K.index, CURVA_K["LL media"], color=COL["u3"], marker="o", label="log-loss media")
ax.axhline(M_PERS["Log-loss"], color=COL["pers"], ls="--", lw=1.2); ax.text(20, M_PERS["Log-loss"], "persistencia ", ha="right", va="bottom", fontsize=8)
ax.axvline(K_ENS, color=TINTA, lw=1, ls=":"); ax.set_xlabel("Redes en el ensamble (K)"); ax.set_ylabel("Log-loss origen móvil 2023-2025")
ax.set_title("(a) El ensamble reduce sobre todo la dispersión"); ax.legend(fontsize=8)
ax = axes[1]
orden = np.argsort(r20["oos"]["p"].values)
ax.errorbar(np.arange(len(orden)), P20.mean(0)[orden], yerr=desv_ind[orden], fmt="none", ecolor=EJE, lw=1)
ax.scatter(np.arange(len(orden)), P20.mean(0)[orden], s=14, c=np.where(y_oos[orden] == 1, COL["C"], COL["D"]), zorder=3)
ax.axhline(.5, color=TINTA, lw=.8, ls="--"); ax.set_xlabel("Departamento-año (ordenado por P(C) del ensamble)"); ax.set_ylabel("P(C) ± DE entre 20 redes")
ax.set_title("(b) Incertidumbre por fila (color = perfil real)")
fig.tight_layout(); guardar(fig, "fig05_ensamble"); plt.show()
""")

lectura("i7")

md(r"""
## 6.10 I8: calibración y umbral de decisión

La U2 vio probabilidades por debajo de la frecuencia real de C fuera de muestra y un umbral F2 (0,17) que no se
transfirió a 2026. Probamos dos calibradores sobre el ensamble ($K = 10$): **Platt** ($\sigma(a\,\mathrm{logit}\,p + b)$),
que corrige escala y sesgo, y **temperatura** ($\sigma(\mathrm{logit}\,p / T)$), que solo corrige la escala. Se
evalúan de dos formas:

* **Encadenamiento hacia adelante** (honesta en el tiempo): 2024 se calibra con 2023, y 2025 con 2023–2024.
* **Validación cruzada por año** (dejar un año fuera): usa más datos, pero calibra 2023 con información de 2025.

Si la calibración solo sirve cuando se mira el futuro, no se adopta.
""")

code(r"""
# =============================================================================
# 6.10 I8 calibración (Platt / temperatura) y estabilidad del umbral óptimo entre años
# =============================================================================
oos7 = r20["oos"]; p_ens = P20[:K_ENS].mean(0); yv = oos7["y"].values
def calibrar_lofo(p, metodo):
    q = p.copy()
    for v in oos7["particion"].unique():
        m = (oos7["particion"] == v).values
        q[m] = M.Calibrador(metodo).ajustar(p[~m], yv[~m]).aplicar(p[m])
    return q
mask_fwd = oos7["particion"].isin(["2024", "2025"]).values           # el encadenamiento no calibra el primer pliegue
filas = []
for nombre, q in [("Sin calibrar", p_ens), ("Platt · adelante", calibrar_adelante(oos7, p_ens, "platt")),
                  ("Temperatura · adelante", calibrar_adelante(oos7, p_ens, "temperatura")),
                  ("Platt · dejar un año fuera", calibrar_lofo(p_ens, "platt")), ("Temperatura · dejar un año fuera", calibrar_lofo(p_ens, "temperatura"))]:
    filas.append({"Calibración": nombre, "LL 2024-2025": E.bce(yv[mask_fwd], q[mask_fwd]), "Brier 2024-2025": np.mean((q[mask_fwd] - yv[mask_fwd]) ** 2),
                  "ECE 2024-2025": E.ece(yv[mask_fwd], q[mask_fwd]), "LL 2023-2025": E.bce(yv, q), "ECE 2023-2025": E.ece(yv, q)})
CAL = pd.DataFrame(filas).set_index("Calibración")
tabla(CAL, dec=3)
par_anio = pd.DataFrame({v: (lambda c: {"a (escala)": c.a, "b (sesgo)": c.b, "Prevalencia C": yv[(oos7.particion == v).values].mean()})(
    M.Calibrador("platt").ajustar(p_ens[(oos7.particion == v).values], yv[(oos7.particion == v).values])) for v in sorted(oos7.particion.unique())})
print("Parámetros de Platt ajustados en cada año por separado (muestran si el sesgo es estable):")
tabla(par_anio, dec=3)
mejor_cal = CAL.iloc[1:3]["LL 2024-2025"].idxmin()
CALIBRACION = ("platt" if "Platt" in mejor_cal else "temperatura") if CAL.loc[mejor_cal, "LL 2024-2025"] < CAL.loc["Sin calibrar", "LL 2024-2025"] - 0.005 else "ninguna"
DECISIONES.append({"Intervención": "I8 calibración", "Referencia": "sin calibrar", "ΔLog-loss": CAL.loc[mejor_cal, "LL 2024-2025"] - CAL.loc["Sin calibrar", "LL 2024-2025"],
                   "EE": np.nan, "ΔAUC": 0.0, "Decisión": "se retiene" if CALIBRACION != "ninguna" else "no se retiene (no mejora sin mirar el futuro)"})

# Umbral: ¿el umbral óptimo es estable entre años?
from sklearn.metrics import matthews_corrcoef
umbrales = np.round(np.linspace(.1, .9, 81), 2)
U = pd.DataFrame({v: [matthews_corrcoef(yv[(oos7.particion == v).values], p_ens[(oos7.particion == v).values] >= u) for u in umbrales]
                  for v in sorted(oos7.particion.unique())}, index=umbrales)
U["Agregado 2023-2025"] = [matthews_corrcoef(yv, p_ens >= u) for u in umbrales]
U_OPT = U.idxmax()
print("Umbral que maximiza el MCC en cada año:", U_OPT.round(2).to_dict())
U_AGR = float(U_OPT["Agregado 2023-2025"]); UMBRAL = 0.5
print(f"MCC con umbral 0,5: {U.loc[0.5, 'Agregado 2023-2025']:.3f} | con el óptimo agregado {U_AGR:.2f}: {U['Agregado 2023-2025'].max():.3f}")

fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
ax = axes[0]; ax.plot([0, 1], [0, 1], color=EJE, lw=1)
for nombre, q, color in [("Sin calibrar", p_ens, COL["u3"]), ("Platt · adelante", calibrar_adelante(oos7, p_ens, "platt"), COL["logit"]), ("Persistencia", PERS["p"].values, COL["pers"])]:
    yy_ = yv if nombre != "Persistencia" else PERS["y"].values
    fr, mp = calibration_curve(yy_, q, n_bins=5, strategy="quantile")
    ax.plot(mp, fr, marker="o", color=color, mec=FONDO, mew=1.5, label=f"{nombre} (ECE {E.ece(yy_, q):.3f})")
ax.set_xlabel("P(C) media predicha"); ax.set_ylabel("Fracción observada de C"); ax.set_title("(a) Confiabilidad · origen móvil 2023-2025"); ax.legend(fontsize=8)
ax = axes[1]
for v, color in zip(sorted(oos7.particion.unique()), [COL["mlp"], COL["logit"], COL["test"]]):
    ax.plot(U.index, U[v], color=color, lw=1.4, label=f"{v} (óptimo {U_OPT[v]:.2f})")
ax.plot(U.index, U["Agregado 2023-2025"], color=TINTA, lw=2, label="agregado"); ax.axvline(.5, color=EJE, lw=1, ls="--")
ax.set_xlabel("Umbral sobre P(C)"); ax.set_ylabel("MCC"); ax.set_title("(b) El umbral óptimo cambia con el año"); ax.legend(fontsize=8)
fig.tight_layout(); guardar(fig, "fig06_calibracion_umbral"); plt.show()
print("Calibración adoptada:", CALIBRACION, "| umbral operativo:", UMBRAL)
""")

lectura("i8")

md(r"""
## 6.11 Síntesis de las intervenciones
""")

code(r"""
# =============================================================================
# 6.11 Tabla de decisiones y escalera de mejora
# =============================================================================
DEC = pd.DataFrame(DECISIONES)
display(DEC.style.format({"ΔLog-loss": "{:+.4f}", "EE": "{:.4f}", "ΔAUC": "{:+.4f}"}, na_rep="–").hide(axis="index")
        .map(lambda v: f"color: {COL['u3']}; font-weight: 600" if isinstance(v, str) and v.startswith("se retiene") else "", subset=["Decisión"]))
ESCALERA_T = [(e, k) for e, k in ESCALERA]
filas = []
for etiqueta, k in ESCALERA_T:
    ps = RES[k]["por_semilla"]
    filas.append({"Paso": etiqueta, "LL media": ps["Log-loss"].mean(), "LL DE": ps["Log-loss"].std(ddof=1), "AUC": ps["ROC-AUC"].mean(), "Brier": ps["Brier"].mean(),
                  "Transiciones acertadas": ps["Transiciones acertadas"].mean()})
m10 = E.metricas(yv, p_ens, transicion=oos7["transicion"])
filas.append({"Paso": f"+ I7 ensamble ({K_ENS})", "LL media": m10["Log-loss"], "LL DE": 0, "AUC": m10["ROC-AUC"], "Brier": m10["Brier"], "Transiciones acertadas": m10["Transiciones acertadas"]})
ESC = pd.DataFrame(filas).drop_duplicates("Paso").set_index("Paso")
tabla(ESC, dec=3)

fig, axes = plt.subplots(1, 2, figsize=(14, 4.4))
for ax, col, et in [(axes[0], "LL media", "Log-loss (↓ mejor)"), (axes[1], "AUC", "ROC-AUC (↑ mejor)")]:
    x = np.arange(len(ESC))
    ax.bar(x, ESC[col], color=[COL["mlp"]] + [COL["u3"]] * (len(ESC) - 1), yerr=ESC["LL DE"] if col == "LL media" else None, ecolor=TINTA2, capsize=3, width=.62)
    for xi, v in zip(x, ESC[col]):
        ax.text(xi, v + (0.006 if col == "LL media" else 0.003), f"{v:.3f}", ha="center", fontsize=8.5)
    ref_p = M_PERS["Log-loss" if col == "LL media" else "ROC-AUC"]
    ax.axhline(ref_p, color=COL["pers"], ls="--", lw=1.2); ax.text(len(ESC) - .5, ref_p, "persistencia", ha="right", va="bottom", fontsize=8)
    ax.set_xticks(x, ESC.index, fontsize=8.5, rotation=15); ax.set_ylabel(et)
    ax.set_ylim((0.25, max(ESC[col].max(), ref_p) * 1.12) if col == "LL media" else (0.88, 1.0))
axes[0].set_title("(a) Escalera de mejora · log-loss (media ± DE entre semillas)"); axes[1].set_title("(b) ROC-AUC")
fig.tight_layout(); guardar(fig, "fig07_escalera_mejora"); plt.show()
""")

lectura("sintesis")
