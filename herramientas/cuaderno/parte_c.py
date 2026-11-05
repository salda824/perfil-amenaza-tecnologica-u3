from nbhelp import md, code, lectura

md(r"""
---
# 7. Controles experimentales de la preparación de datos

Cada decisión de preprocesamiento se pone a prueba **quitándola** (o introduciendo a propósito un error conocido)
y midiendo el efecto con el mismo protocolo, las mismas 5 semillas y la especificación final de variables. Un
control "pasa" cuando el resultado se mueve en la dirección que predice la teoría.

| Control | Qué se cambia | Qué esperamos si la decisión original es correcta |
|---|---|---|
| Sin estandarización | Variables en escala original (km, %, log) | Peor convergencia: el gradiente lo dominan las variables de gran escala (distancia, bosque) |
| Sin logaritmos | $x_1$, $x_2$, $x_4$ en conteos y hectáreas crudas | Peor: colas extremas (Cauca, Nariño) dominan el ajuste |
| Sin pesos de clase | BCE sin $\omega_c$ | Efecto pequeño (clases casi balanceadas), algo peor en sensibilidad |
| Sin contracción ($\alpha \approx 0$) | $m_1$, $d_1$ con proporciones crudas | Peor: semestres con pocos eventos producen proporciones extremas |
| Escalador con fuga (control negativo) | Media y desviación ajustadas también con el año de prueba | Mejora artificial pequeña: cuantifica cuánto inflaría una fuga típica |
| `MIN_EVENTOS` = 5 / 20 | Umbral de confiabilidad de la etiqueta en el entrenamiento | 10 equilibra tamaño y ruido de etiqueta (D3) |
| Mapeo alternativo (M8) | *Shelling* → B y *Sexual violence* → C | Sin cambio material (1 etiqueta de 171 cambia, U2 §3.4) |
| Etiquetas barajadas (cordura) | Se entrena con $y$ permutada | AUC ≈ 0,5: el modelo no "aprende" ruido |
""")

code(r"""
# =============================================================================
# 7.1 Ablaciones de preprocesamiento (5 semillas, origen móvil 2023-2025)
# =============================================================================
t0 = time.time()
REF_CTRL = P.Especificacion("ctrl: referencia", VARS_FINAL, CFG_I6)
evaluar(REF_CTRL)
evaluar(replace(REF_CTRL, nombre="ctrl: sin estandarización", escalar=False))
evaluar(replace(REF_CTRL, nombre="ctrl: escalador con fuga", fuga_escalador=True))
evaluar(replace(REF_CTRL, nombre="ctrl: sin pesos de clase", red=replace(CFG_I6, pesos_clase=False)))
d_nolog = datos.copy()
for x_ in ["x1_eventos", "x2_fatalidades", "x4_coca"]:
    d_nolog[x_] = np.expm1(datos[x_])
evaluar(replace(REF_CTRL, nombre="ctrl: sin logaritmos"), base=d_nolog)
V.ALFA = 1e-9
d_cruda, S_cruda, _ = V.construir_dataset(acled, actores=ACTORES)
V.ALFA = 4.0
evaluar(replace(REF_CTRL, nombre="ctrl: sin contracción"), base=d_cruda, seq=S_cruda)
d_alt, S_alt, _ = V.construir_dataset(acled, mapa=C.MAPA_ALTERNATIVO, actores=ACTORES)
evaluar(replace(REF_CTRL, nombre="ctrl: mapeo alternativo"), base=d_alt, seq=S_alt)

# Etiquetas barajadas: se entrena con y permutada y se evalúa contra la etiqueta real
rng_p = np.random.default_rng(0); idx_p = rng_p.permutation(len(datos))
d_perm = datos.copy(); d_perm["y"] = datos["y"].values[idx_p]; d_perm["s_C"] = datos["s_C"].values[idx_p]
filas = []
for s in SEM5:
    sp = replace(REF_CTRL, semillas=(s,)); ps, ys, ts = [], [], []
    for v, tr, te in E.particiones_origen_movil(datos):
        m_, e_ = P.entrenar_especificacion(d_perm, SEQ, sp, tr)
        ps.append(P.predecir_especificacion(m_, e_, datos, SEQ, sp, te)); ys.append(datos.loc[te, "y"].values); ts.append(datos.loc[te, "transicion"].values)
    filas.append({"semilla": s, **E.metricas(np.concatenate(ys), np.concatenate(ps), transicion=np.concatenate(ts))})
RES["ctrl: etiquetas barajadas"] = {"por_semilla": pd.DataFrame(filas).set_index("semilla"), "spec": REF_CTRL}

# Umbral de confiabilidad: se entrena con MIN_EVENTOS = 5, 10, 20 y se evalúa en las mismas filas (>= 20 eventos)
d5, S5, _ = V.construir_dataset(acled, min_eventos=5, actores=ACTORES)
MINEV = []
for m_ev in (5, 10, 20):
    for s in SEM5:
        sp = replace(REF_CTRL, semillas=(s,)); ps, ys = [], []
        for v in C.ANIOS_SELECCION:
            tr = ((d5.anio < v) & (d5.eventos >= m_ev)).values; te = ((d5.anio == v) & (d5.eventos >= 20)).values
            m_, e_ = P.entrenar_especificacion(d5, S5, sp, tr)
            ps.append(P.predecir_especificacion(m_, e_, d5, S5, sp, te)); ys.append(d5.loc[te, "y"].values)
        MINEV.append({"MIN_EVENTOS": m_ev, "semilla": s, "filas de entrenamiento (último pliegue)": int(tr.sum()),
                      "LL": E.bce(np.concatenate(ys), np.concatenate(ps)), "AUC": roc_auc_score(np.concatenate(ys), np.concatenate(ps))})
MINEV = pd.DataFrame(MINEV)
print(f"Controles en {time.time() - t0:.0f} s")

ctrl_k = ["ctrl: sin estandarización", "ctrl: sin logaritmos", "ctrl: sin pesos de clase", "ctrl: sin contracción",
          "ctrl: escalador con fuga", "ctrl: mapeo alternativo", "ctrl: etiquetas barajadas"]
CTRL = pd.DataFrame([decidir(k, "ctrl: referencia") for k in ctrl_k]).set_index("Intervención")[["ΔLog-loss", "EE", "ΔAUC"]]
CTRL["LL media"] = [RES[k]["por_semilla"]["Log-loss"].mean() for k in ctrl_k]
CTRL["AUC media"] = [RES[k]["por_semilla"]["ROC-AUC"].mean() for k in ctrl_k]
ll_ref = RES["ctrl: referencia"]["por_semilla"]["Log-loss"].mean()
mv = MINEV.groupby("MIN_EVENTOS")[["LL", "AUC", "filas de entrenamiento (último pliegue)"]].mean()
print(f"Referencia: LL {ll_ref:.3f} | AUC {RES['ctrl: referencia']['por_semilla']['ROC-AUC'].mean():.3f}")
tabla(CTRL, dec=4)
print("Umbral de confiabilidad (evaluación común: filas con >= 20 eventos):")
tabla(mv, dec=3)
""")

code(r"""
# =============================================================================
# 7.2 Impacto en el entrenamiento: curvas de pérdida con y sin cada transformación (pliegue 2025)
# =============================================================================
def curva_entrenamiento(base, spec, epocas=160, semilla=1):
    tr, va = (base.anio < 2025).values, (base.anio == 2025).values
    esc = M.Escalador().ajustar(base.loc[tr, spec.variables].values)
    if not spec.escalar:
        esc.mu, esc.sd = np.zeros_like(esc.mu), np.ones_like(esc.sd)
    Xa, Xv = esc.transformar(base.loc[tr, spec.variables].values), esc.transformar(base.loc[va, spec.variables].values)
    y_obj = base.loc[tr, "s_C"].values if spec.red.objetivo == "blando" else base.loc[tr, "y"].values
    r = M.entrenar(Xa, y_obj, spec.red, semilla=semilla, y_duro=base.loc[tr, "y"].values, eventos=base.loc[tr, "eventos"].values,
                   comp=base.loc[tr, V.COMPOSICION].values, val=(Xv, base.loc[va, "y"].values), diagnostico=True, epocas=epocas)
    return r.historial
curvas_c = {"Referencia (estandarizado, log, contraído)": curva_entrenamiento(datos, REF_CTRL),
            "Sin estandarización": curva_entrenamiento(datos, replace(REF_CTRL, escalar=False)),
            "Sin logaritmos": curva_entrenamiento(d_nolog, REF_CTRL),
            "Sin contracción": curva_entrenamiento(d_cruda, REF_CTRL)}
EP_560 = math.ceil(560 / math.ceil((datos.anio < 2025).sum() / CFG_I6.batch))
fig, axes = plt.subplots(1, 3, figsize=(16, 4.3), gridspec_kw={"width_ratios": [1, 1, 1.1]})
colores_c = [COL["u3"], COL["A"], COL["logit"], COL["B"]]
for (nombre, H), color in zip(curvas_c.items(), colores_c):
    axes[0].plot(H["epoca"], H["bce_tr"], color=color, lw=1.6, label=nombre)
    axes[1].plot(H["epoca"], H["bce_va"], color=color, lw=1.6, label=nombre)
for ax, t in zip(axes[:2], ["(a) BCE de entrenamiento (convergencia)", "(b) BCE de validación 2025"]):
    ax.axvline(EP_560, color=TINTA, lw=1, ls=":"); ax.text(EP_560, ax.get_ylim()[1] * .98, " 560 pasos", fontsize=8, va="top")
    ax.set_xlabel("Época"); ax.set_title(t); ax.set_ylabel("BCE")
axes[0].legend(fontsize=7.5)
ax = axes[2]
orden = CTRL["ΔLog-loss"].sort_values().index
ax.barh(range(len(orden)), CTRL.loc[orden, "ΔLog-loss"], xerr=CTRL.loc[orden, "EE"], color=[COL["A"] if v > 0 else COL["u3"] for v in CTRL.loc[orden, "ΔLog-loss"]],
        error_kw=dict(ecolor=TINTA2, lw=1), height=.6)
ax.axvline(0, color=TINTA, lw=1); ax.set_yticks(range(len(orden)), [o.replace("ctrl: ", "") for o in orden], fontsize=8.5)
ax.set_xlabel("Δ log-loss frente a la referencia (+ = peor)"); ax.set_title("(c) Efecto de cada control (media ± EE, 5 semillas)")
fig.tight_layout(); guardar(fig, "fig08_controles_datos"); plt.show()
conv = pd.DataFrame({k: {"BCE train época 10": H["bce_tr"][9], f"BCE train época {EP_560}": H["bce_tr"][EP_560 - 1],
                         f"BCE val época {EP_560}": H["bce_va"][EP_560 - 1], "BCE val mínima": min(H["bce_va"])} for k, H in curvas_c.items()}).T
tabla(conv, dec=3)
""")

lectura("controles")

md(r"""
---
# 8. Modelo final integrado

## 8.1 Configuración documentada
""")

code(r"""
# =============================================================================
# 8.1 Especificación final, resumen de Keras y configuración exportada
# =============================================================================
FINAL = P.Especificacion("MLP U3", VARS_FINAL, CFG_I6, semillas=tuple(range(1, K_ENS + 1)))
miembro = M.construir_red(len(VARS_FINAL), FINAL.red)
miembro.summary()
N_PAR_FINAL = M.n_parametros(miembro)
n_tr_2025 = int((datos.anio <= 2025).sum())
print(f"\nParámetros por red: {N_PAR_FINAL} | redes en el ensamble: {K_ENS} | filas 2021-2025: {n_tr_2025} | "
      f"parámetros por observación (una red): {N_PAR_FINAL / n_tr_2025:.2f}")
config_final = pd.DataFrame([
    ("Variables de entrada", f"{len(VARS_FINAL)}: " + ", ".join(VARS_FINAL)),
    ("Arquitectura", f"{len(VARS_FINAL)} → {' → '.join(map(str, FINAL.red.capas))} {FINAL.red.activacion.upper()} → 1 sigmoide" + (" + softmax(4)" if FINAL.red.tipo == "multitarea" else "")),
    ("Inicialización", "He normal (ReLU) / Glorot (tanh)"), ("Regularización", f"L2 = {FINAL.red.l2:g}; dropout = {FINAL.red.dropout:g}"),
    ("Pérdida", "BCE ponderada por clase" + (" con objetivo blando" if FINAL.red.objetivo == "blando" else "") + (f" + {FINAL.red.lambda_comp:g}·KL" if FINAL.red.tipo == "multitarea" else "")),
    ("Optimizador", f"Adam, η = {FINAL.red.lr:g}, lote {FINAL.red.batch}"), ("Presupuesto", f"{FINAL.red.pasos} pasos de gradiente por red"),
    ("Ensamble", f"{K_ENS} semillas (1-{K_ENS}), promedio de probabilidades"), ("Calibración", CALIBRACION), ("Umbral operativo", f"{UMBRAL}"),
    ("Escalado", "Estandarización ajustada solo con el entrenamiento de cada pliegue"),
], columns=["Componente", "Elección"]).set_index("Componente")
display(config_final)

CONFIG = {
    "descripcion": "Configuración del modelo final de la U3 (generada por la §8.1 del cuaderno).",
    "especificacion_final": FINAL.a_dict(),
    "especificacion_base_u2": replace(BASE_U2, semillas=(C.SEED,)).a_dict(),
    "calibracion": CALIBRACION, "umbral": UMBRAL,
    "protocolo": {"seleccion_origen_movil": list(C.ANIOS_SELECCION), "prueba": C.ANIO_PRUEBA, "min_eventos": C.MIN_EVENTOS,
                  "alfa_contraccion": V.ALFA, "semillas_intervenciones": list(SEM5)},
    "decisiones": [{k: (None if (isinstance(v, float) and np.isnan(v)) else v) for k, v in d.items()} for d in DECISIONES],
}
C.RUTA_CONFIG.write_text(json.dumps(CONFIG, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
print(f"Configuración escrita en {C.RUTA_CONFIG.name}")
""")

code(r"""
# =============================================================================
# 8.2 Esquema de la arquitectura final
# =============================================================================
def dibujar_arquitectura(cfg, nombres):
    tam = [len(nombres), *cfg.capas, 1]
    fig, ax = plt.subplots(figsize=(13, 5.8)); ax.axis("off"); ax.grid(False)
    xs = np.arange(len(tam)) * 3.0
    pos = [[(x, y) for y in np.linspace(-(n - 1) / 2, (n - 1) / 2, n) * min(.62, 7 / max(n, 1))] for x, n in zip(xs, tam)]
    for a, b in zip(pos[:-1], pos[1:]):
        for (x1, y1) in a:
            for (x2, y2) in b:
                ax.plot([x1, x2], [y1, y2], color=GRID, lw=.5, zorder=1)
    nuevas_ = set(V.MEMORIA + V.DINAMICA + V.VECINDAD + V.ACTORES)
    for k, capa in enumerate(pos):
        for j, (x, y) in enumerate(capa):
            color = (COL["u3"] if nombres[j] in nuevas_ else COL["D"]) if k == 0 else (COL["C"] if k == len(pos) - 1 else COL["mlp"])
            ax.scatter(x, y, s=140, color=color, edgecolor=FONDO, lw=2, zorder=3)
    for (x, y), n in zip(pos[0], nombres):
        ax.text(x - .35, y, V.ETIQ.get(n, n), ha="right", va="center", fontsize=8.5, color=TINTA2)
    ax.text(pos[-1][0][0] + .35, 0, f"P(perfil C | x$_{{t-1}}$)\nensamble de {K_ENS} redes", ha="left", va="center", fontsize=9.5)
    top = max(max(abs(y) for _, y in c) for c in pos)
    desc = [f"Entrada\n{len(nombres)} variables\nestandarizadas"] + [f"Oculta {i}\n{u} {cfg.activacion.upper()}\nL2={cfg.l2:g}" for i, u in enumerate(cfg.capas, 1)] + ["Salida\n1 sigmoide"]
    for x, d_ in zip(xs, desc):
        ax.text(x, -top - .55, d_, ha="center", va="top", fontsize=8.5, color=TINTA2)
    ax.set_title(f"Arquitectura final U3 · {N_PAR_FINAL} parámetros por red (verde = variables nuevas de la U3)")
    ax.set_xlim(xs[0] - 3.2, xs[-1] + 3.2); ax.set_ylim(-top - 2.1, top + .6)
    return fig
fig = dibujar_arquitectura(FINAL.red, VARS_FINAL); guardar(fig, "fig09_arquitectura_final"); plt.show()
""")

md(r"""
## 8.2 Curvas de desempeño del modelo final frente al base

Entrenamos las dos especificaciones con 2021–2024 y registramos la BCE de entrenamiento y de validación (2025) en
cada época, con 5 semillas y hasta 3 veces el presupuesto. Así se ve si los 560 pasos caen cerca del óptimo de
generalización también con las variables nuevas, y cómo cambia la brecha entre entrenamiento y validación.
""")

code(r"""
# =============================================================================
# 8.3 Curvas de aprendizaje (2021-2024 -> validación 2025), 5 semillas, 3x el presupuesto
# =============================================================================
CURVAS = {}
for nombre, spec_c in [("Base U2", BASE_U2), ("Final U3", FINAL)]:
    CURVAS[nombre] = [curva_entrenamiento(datos, replace(spec_c, semillas=(s,)), epocas=3 * EP_560, semilla=s) for s in SEM5]
fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
for ax, (nombre, Hs), color in zip(axes, CURVAS.items(), [COL["mlp"], COL["u3"]]):
    tr_ = np.array([h["bce_tr"] for h in Hs]); va_ = np.array([h["bce_va"] for h in Hs]); ep_ = Hs[0]["epoca"]
    ax.fill_between(ep_, tr_.min(0), tr_.max(0), color=color, alpha=.12, lw=0); ax.plot(ep_, tr_.mean(0), color=color, lw=1.6, ls=":", label="entrenamiento 2021-2024")
    ax.fill_between(ep_, va_.min(0), va_.max(0), color=COL["val"], alpha=.12, lw=0); ax.plot(ep_, va_.mean(0), color=COL["val"], lw=1.8, label="validación 2025")
    k_min = int(np.argmin(va_.mean(0))); ax.scatter(ep_[k_min], va_.mean(0)[k_min], color=TINTA, zorder=4, s=25)
    ax.axvline(EP_560, color=TINTA, lw=1, ls="--"); ax.text(EP_560, .62, " 560 pasos", fontsize=8)
    ax.set_ylim(.05, .7); ax.set_xlabel("Época"); ax.set_ylabel("BCE"); ax.legend(fontsize=8.5)
    ax.set_title(f"({'ab'[nombre == 'Final U3']}) {nombre}: BCE val. en 560 pasos = {va_.mean(0)[EP_560 - 1]:.3f} · mínimo {va_.mean(0)[k_min]:.3f} (época {ep_[k_min]})")
fig.tight_layout(); guardar(fig, "fig10_curvas_aprendizaje"); plt.show()
resumen_curvas = pd.DataFrame({n: {"BCE train (560 pasos)": np.mean([h["bce_tr"][EP_560 - 1] for h in Hs]), "BCE val (560 pasos)": np.mean([h["bce_va"][EP_560 - 1] for h in Hs]),
                                   "Brecha val − train": np.mean([h["bce_va"][EP_560 - 1] - h["bce_tr"][EP_560 - 1] for h in Hs]),
                                   "BCE val mínima (media)": float(np.min(np.array([h["bce_va"] for h in Hs]).mean(0))),
                                   "BCE val al final (3x)": np.mean([h["bce_va"][-1] for h in Hs])} for n, Hs in CURVAS.items()}).T
tabla(resumen_curvas, dec=3)
""")

lectura("curvas")

md(r"""
---
# 9. Evaluación antes vs. después

## 9.1 Origen móvil 2023–2025 (selección)

Comparamos dos versiones del "antes". La primera es el modelo base entregado en la U2 (MLP 7 → 16 → 1, semilla 42),
que la U2 documentó en la cola desfavorable entre semillas. La segunda es la misma arquitectura base promediada sobre
las mismas 10 semillas del modelo final, para que la diferencia no mezcle el efecto de las variables con el del
ensamble. "Después" es el modelo final de la U3 (ensamble de 10 redes). Como rival agregamos, además de la persistencia,
su versión calibrada (§3.1, regla 3). Las comparaciones pareadas y los intervalos usan bootstrap por departamento: se
remuestrean territorios completos con sus tres años, porque los años de un mismo departamento no son independientes.
""")

code(r"""
# =============================================================================
# 9.1 Antes vs. después en el origen móvil 2023-2025, con IC y pruebas pareadas
# =============================================================================
evaluar(replace(BASE_U2, nombre="MLP base U2 (semilla 42)"), semillas=(C.SEED,))
evaluar(replace(BASE_U2, nombre=f"MLP base U2 (ensamble {K_ENS})"), semillas=tuple(range(1, K_ENS + 1)))
r_fin = RES["I7 20 semillas"]; oosF = r_fin["oos"].copy(); oosF["final"] = P20[:K_ENS].mean(0)
if CALIBRACION != "ninguna":
    oosF["final"] = calibrar_adelante(oosF, oosF["final"].values, CALIBRACION)
oosF["antes"] = RES["MLP base U2 (semilla 42)"]["oos"]["p"].values
oosF["antes10"] = RES[f"MLP base U2 (ensamble {K_ENS})"]["oos"]["p"].values
oosF["logistica"] = RES["Logística (mismas variables)"]["oos"]["p"].values
oosF["pers"] = PERS["p"].values
oosF["pers_cal"] = PERS_CAL["p"].values
assert (oosF["y"].values == PERS["y"].values).all() and (oosF["y"].values == PERS_CAL["y"].values).all()
MODELOS_9 = {"Persistencia": "pers", "Persistencia calibrada (rival agregado tras la revisión)": "pers_cal",
             "Antes: MLP base U2 (semilla 42, entregado)": "antes", f"Antes: MLP base U2 (ensamble {K_ENS})": "antes10",
             "Logística (mismas variables)": "logistica", "Después: MLP U3 (ensamble)": "final"}
T91 = pd.DataFrame({k: E.metricas(oosF.y, oosF[c], transicion=oosF.transicion) for k, c in MODELOS_9.items()}).T
tabla(T91[["ROC-AUC", "Log-loss", "Brier", "ECE", "MCC", "Exactitud", "Sensibilidad C", "Especificidad D", "Transiciones acertadas", "Transiciones"]], dec=3)
ic = {k: {m: "[{:.3f} – {:.3f}]".format(*E.bootstrap_ic(oosF.y, oosF[c], fn, grupos=oosF.depto)) for m, fn in [("ROC-AUC", roc_auc_score), ("Log-loss", E.bce)]} for k, c in MODELOS_9.items()}
display(pd.DataFrame(ic).T.rename(columns=lambda c: f"IC95 {c} (por departamento)"))
PAR_91 = pd.DataFrame({f"Después vs. {k}": E.comparar_pareado(oosF.y, oosF.final, oosF[c], grupos=oosF.depto)
                       for k, c in [("persistencia", "pers"), ("persistencia calibrada", "pers_cal"), ("antes (U2, semilla 42)", "antes"),
                                    (f"antes (U2, ensamble {K_ENS})", "antes10"), ("logística", "logistica")]}).T
display(PAR_91[["n", "ΔLog-loss", "IC95 ΔLog-loss", "P(ΔLog-loss mejora)", "ΔBrier", "IC95 ΔBrier", "ΔAUC", "IC95 ΔAUC", "solo A acierta", "solo B acierta", "p McNemar"]]
        .style.format({"ΔLog-loss": "{:+.3f}", "ΔBrier": "{:+.3f}", "ΔAUC": "{:+.3f}", "P(ΔLog-loss mejora)": "{:.2f}", "p McNemar": "{:.3f}"}))
""")

lectura("antes_despues")

md(r"""
## 9.2 Generalización espacial: departamentos nunca vistos

Usamos dos esquemas. El espacial de la U3 agrupa los departamentos en 5 grupos (GroupKFold, años ≤ 2025): para
predecir un departamento en el año t, la red se entrena con los demás, incluidas sus etiquetas del mismo año t y de
años posteriores. La U2 había reportado dejar un departamento fuera, porque con la versión de scikit-learn de entonces
GroupKFold desempataba distinto según el equipo; desde scikit-learn 1.6 el desempate es estable, y usamos 5 grupos para
abaratar el cálculo. El segundo esquema, agregado tras la revisión, bloquea también el tiempo: para cada grupo y cada
año v ∈ {2023, 2024, 2025} entrena con los otros departamentos y solo con t < v. Es el escenario real de un territorio
nuevo al año siguiente, y es el único justo con la persistencia, que no se entrena.
""")

code(r"""
# =============================================================================
# 9.2 Validación espacial (5 pliegues por departamento, años <= 2025)
# =============================================================================
SEM_ENS = tuple(range(1, K_ENS + 1))
evaluar(replace(FINAL, nombre="esp: final U3"), semillas=SEM_ENS, particiones="espacial", clave="esp: final U3")
evaluar(replace(BASE_U2, nombre="esp: base U2 ensamble"), semillas=SEM_ENS, particiones="espacial", clave="esp: base U2 ensamble")
eF, eB = RES["esp: final U3"]["oos"], RES["esp: base U2 ensamble"]["oos"]
esp_tab = pd.DataFrame({
    "Persistencia": E.metricas(PERS_E.y, PERS_E.p),
    "Antes: MLP base U2 (red individual, media)": {c: RES["esp: base U2 ensamble"]["por_semilla"][c].mean() for c in E.COLS_PRINCIPALES},
    f"Antes: MLP base U2 (ensamble {K_ENS})": E.metricas(eB.y, eB.p),
    "Logística (mismas variables)": E.metricas(RES["esp: logística referencia"]["oos"].y, RES["esp: logística referencia"]["oos"].p),
    "Después: MLP U3 (red individual, media)": {c: RES["esp: final U3"]["por_semilla"][c].mean() for c in E.COLS_PRINCIPALES},
    "Después: MLP U3 (ensamble)": E.metricas(eF.y, eF.p)}).T
print("Esquema espacial (5 grupos de departamentos, años <= 2025 mezclados):")
tabla(esp_tab[E.COLS_PRINCIPALES], dec=3)
pers_e_al = PERS_E.set_index("fila").loc[eF["fila"], "p"].values
display(pd.DataFrame({"Después vs. persistencia (espacial)": E.comparar_pareado(eF.y, eF.p, pers_e_al, grupos=eF.depto),
                      f"Después vs. antes (ensamble {K_ENS}, espacial)": E.comparar_pareado(eF.y, eF.p, eB.p.values, grupos=eF.depto)}).T[["n", "ΔLog-loss", "IC95 ΔLog-loss", "ΔAUC", "IC95 ΔAUC"]])

# Bloqueo espacial y temporal: otros departamentos, solo años anteriores al predicho
PERS_ET, PERS_CAL_ET = persistencia_oos("espacio-temporal"), persistencia_oos("espacio-temporal", calibrada=True)
evaluar(replace(FINAL, nombre="et: final U3"), semillas=SEM_ENS, particiones="espacio-temporal", clave="et: final U3")
evaluar(replace(BASE_U2, nombre="et: base U2 ensamble"), semillas=SEM_ENS, particiones="espacio-temporal", clave="et: base U2 ensamble")
evaluar(replace(LOG_F, nombre="et: logística"), particiones="espacio-temporal", clave="et: logística")
etF, etB, etL = (RES[k]["oos"] for k in ("et: final U3", "et: base U2 ensamble", "et: logística"))
assert all((x.y.values == PERS_ET.y.values).all() for x in (etF, etB, etL, PERS_CAL_ET))
ET_TAB = pd.DataFrame({"Persistencia": E.metricas(PERS_ET.y, PERS_ET.p, transicion=PERS_ET.transicion),
                       "Persistencia calibrada": E.metricas(PERS_CAL_ET.y, PERS_CAL_ET.p, transicion=PERS_CAL_ET.transicion),
                       f"Antes: MLP base U2 (ensamble {K_ENS})": E.metricas(etB.y, etB.p, transicion=etB.transicion),
                       "Logística (mismas variables)": E.metricas(etL.y, etL.p, transicion=etL.transicion),
                       "Después: MLP U3 (ensamble)": E.metricas(etF.y, etF.p, transicion=etF.transicion)}).T
print(f"Esquema espacio-temporal (5 grupos × 2023-2025, entrenamiento solo con años anteriores; n = {len(etF)}):")
tabla(ET_TAB[E.COLS_PRINCIPALES + ["Transiciones acertadas"]], dec=3)
PAR_ET = pd.DataFrame({f"Después vs. {k}": E.comparar_pareado(etF.y, etF.p, c_, grupos=etF.depto)
                       for k, c_ in [("persistencia", PERS_ET.p.values), ("persistencia calibrada", PERS_CAL_ET.p.values),
                                     (f"antes (ensamble {K_ENS})", etB.p.values), ("logística", etL.p.values)]}).T
display(PAR_ET[["n", "ΔLog-loss", "IC95 ΔLog-loss", "P(ΔLog-loss mejora)", "ΔAUC", "IC95 ΔAUC"]]
        .style.format({"ΔLog-loss": "{:+.3f}", "ΔAUC": "{:+.3f}", "P(ΔLog-loss mejora)": "{:.2f}"}))
""")

lectura("espacial")

md(r"""
## 9.3 Prueba final 2026 (una sola vez)

El modelo, las variables, la configuración, el ensamble, la calibración y el umbral ya están fijados (§8.1,
`configuracion.json`). Ahora, por única vez, se entrena con 2021–2025 y se predice 2026 (29 departamentos,
etiqueta de 33 semanas; §2.3 estimó un 95 % de coincidencia con la etiqueta anual).
""")

code(r"""
# =============================================================================
# 9.3 Prueba 2026: entrenamiento con 2021-2025, una sola evaluación
# =============================================================================
tr26, te26 = (datos.anio <= 2025).values, (datos.anio == 2026).values
T26 = datos.loc[te26, ["depto", "departamento", "anio", "y", "s_C", "perfil", "perfil_prev", "transicion", "eventos"]].copy()
T26["pers"] = P.persistencia(datos, tr26, te26)
T26["pers_cal"] = P.persistencia_calibrada(datos, tr26, te26)
for nombre, s in [("antes", replace(BASE_U2, semillas=(C.SEED,))), ("antes10", replace(BASE_U2, semillas=SEM_ENS)), ("logistica", LOG_F), ("final", FINAL)]:
    m_, e_ = P.entrenar_especificacion(datos, SEQ, s, tr26)
    T26[nombre] = P.predecir_especificacion(m_, e_, datos, SEQ, s, te26)
    if nombre == "final":
        MODELOS_2025, ESC_2025 = m_, e_
if CALIBRACION != "ninguna":
    cal_26 = M.Calibrador(CALIBRACION).ajustar(P20[:K_ENS].mean(0), oosF.y.values)   # calibrador con 2023-2025 (sin calibrar)
    T26["final"] = cal_26.aplicar(T26["final"])
T93 = pd.DataFrame({k: E.metricas(T26.y, T26[c], transicion=T26.transicion) for k, c in MODELOS_9.items()}).T
spec_emb = replace(RES["Embedding de departamento"]["spec"], semillas=tuple(range(1, K_ENS + 1)))
m_, e_ = P.entrenar_especificacion(datos, SEQ, spec_emb, tr26)
T26["emb"] = P.predecir_especificacion(m_, e_, datos, SEQ, spec_emb, te26)
T93.loc["(alternativa no adoptada) MLP + embedding"] = pd.Series(E.metricas(T26.y, T26.emb, transicion=T26.transicion))
doc26 = pd.read_csv(RAIZ / "referencias_u2" / "metricas_modelos.csv").set_index(["Modelo", "Conjunto"])
T93.loc["(referencia) U2 fase 2 documentado", ["ROC-AUC", "Log-loss", "Brier", "MCC", "Exactitud"]] = doc26.loc[("MLP fase 2", "Prueba"), ["ROC-AUC", "Log-loss", "Brier", "MCC", "Exactitud"]].values
tabla(T93[["ROC-AUC", "Log-loss", "Brier", "ECE", "MCC", "Exactitud", "Sensibilidad C", "Especificidad D", "Transiciones acertadas", "Transiciones"]], dec=3)
ic26 = {k: {m: "{:.3f} [{:.3f} – {:.3f}]".format(fn(T26.y, T26[c]), *E.bootstrap_ic(T26.y, T26[c], fn)) for m, fn in [("ROC-AUC", roc_auc_score), ("Log-loss", E.bce)]} for k, c in MODELOS_9.items()}
display(pd.DataFrame(ic26).T)
PAR_26 = pd.DataFrame({f"Después vs. {k}": E.comparar_pareado(T26.y, T26.final, T26[c])
                       for k, c in [("persistencia", "pers"), ("persistencia calibrada", "pers_cal"), ("antes (U2, semilla 42)", "antes"),
                                    (f"antes (U2, ensamble {K_ENS})", "antes10"), ("logística", "logistica")]}).T
display(PAR_26[["n", "ΔLog-loss", "IC95 ΔLog-loss", "ΔAUC", "IC95 ΔAUC", "solo A acierta", "solo B acierta", "p McNemar"]]
        .style.format({"ΔLog-loss": "{:+.3f}", "ΔAUC": "{:+.3f}", "p McNemar": "{:.3f}"}))
print("Transiciones de 2026:")
display(T26[T26.transicion][["departamento", "perfil_prev", "perfil", "s_C", "pers", "pers_cal", "antes", "antes10", "final"]].round(2).style.hide(axis="index"))
""")

code(r"""
# =============================================================================
# 9.4 Matrices de confusión, ROC y desempeño por año (antes vs. después vs. persistencia)
# =============================================================================
fig = plt.figure(figsize=(16, 8.4)); gs = fig.add_gridspec(2, 4, hspace=.42, wspace=.32)
casos = [("Antes · 2023-2025", oosF.y, oosF.antes), ("Después · 2023-2025", oosF.y, oosF.final), ("Antes · 2026", T26.y, T26.antes), ("Después · 2026", T26.y, T26.final)]
for k, (tit, y_, p_) in enumerate(casos):
    ax = fig.add_subplot(gs[0, k]); cm = confusion_matrix(np.asarray(y_).astype(int), (np.asarray(p_) >= UMBRAL).astype(int), labels=[0, 1])
    ax.imshow(cm, cmap=G.CMAP_P, vmin=0, vmax=cm.max() * 1.3); ax.grid(False)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]}\n({cm[i, j] / max(cm[i].sum(), 1):.0%})", ha="center", va="center", fontsize=10, color="white" if cm[i, j] > cm.max() * .65 else TINTA)
    ax.set_xticks([0, 1], ["D", "C"]); ax.set_yticks([0, 1], ["D", "C"]); ax.set_xlabel("Predicho"); ax.set_ylabel("Real"); ax.set_title(tit, fontsize=9.5)
ax = fig.add_subplot(gs[1, :2])
for nombre, c, color, ls in [("Persistencia", "pers", COL["pers"], "--"), ("Persistencia calibrada", "pers_cal", COL["A"], "-."), ("Antes (U2)", "antes", COL["mlp"], ":"), ("Logística", "logistica", COL["logit"], "-"), ("Después (U3)", "final", COL["u3"], "-")]:
    fpr, tpr, _ = roc_curve(oosF.y, oosF[c]); ax.plot(fpr, tpr, color=color, ls=ls, label=f"{nombre}: 2023-25 AUC {roc_auc_score(oosF.y, oosF[c]):.3f} · 2026 AUC {roc_auc_score(T26.y, T26[c]):.3f}")
ax.plot([0, 1], [0, 1], color=EJE, lw=1); ax.set_xlabel("1 − especificidad"); ax.set_ylabel("Sensibilidad C"); ax.set_title("(e) Curvas ROC · origen móvil 2023-2025"); ax.legend(fontsize=7.8, loc="lower right")
ax = fig.add_subplot(gs[1, 2:])
anual = pd.concat([oosF.assign(particion=oosF.particion.astype(str)), T26.assign(particion="2026")])
for nombre, c, color, ls in [("Persistencia", "pers", COL["pers"], "--"), ("Persistencia calibrada", "pers_cal", COL["A"], "-."), ("Antes (U2)", "antes", COL["mlp"], ":"), ("Después (U3)", "final", COL["u3"], "-")]:
    s_ = anual.groupby("particion").apply(lambda g: E.bce(g.y, g[c]))
    ax.plot(s_.index, s_.values, color=color, ls=ls, marker="o", mec=FONDO, mew=1.5, ms=7, label=nombre)
ax.axvspan(2.6, 3.4, color=COL["test"], alpha=.08, lw=0); ax.text(3, ax.get_ylim()[0] + .005, "prueba única", ha="center", va="bottom", fontsize=8.5)
ax.set_ylabel("Log-loss"); ax.set_title("(f) Log-loss por año (2023-2025 origen móvil, 2026 prueba)"); ax.legend(fontsize=8.5)
guardar(fig, "fig11_antes_despues"); plt.show()
""")

lectura("prueba_2026")
