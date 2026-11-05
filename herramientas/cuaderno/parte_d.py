from nbhelp import md, code, lectura

md(r"""
---
# 10. Análisis crítico del desempeño

## 10.1 ¿Dónde se equivoca el modelo final y qué errores corrigió?
""")

code(r"""
# =============================================================================
# 10.1 Análisis de errores: 2023-2025 (origen móvil) + 2026 (prueba)
# =============================================================================
cols_f = ["departamento", "depto", "anio", "y", "s_C", "perfil", "perfil_prev", "transicion", "eventos", "pers", "pers_cal", "antes", "antes10", "final"]
fuera = pd.concat([oosF[cols_f], T26[cols_f]], ignore_index=True)
fuera["acierto antes"] = (fuera.antes >= UMBRAL) == (fuera.y == 1)
fuera["acierto después"] = (fuera.final >= UMBRAL) == (fuera.y == 1)
fuera["cerca del umbral"] = (fuera.s_C - .5).abs() < .1
err = fuera[~fuera["acierto después"]].sort_values(["anio", "final"])
display(err[["departamento", "anio", "perfil_prev", "perfil", "s_C", "eventos", "pers", "antes", "final", "transicion", "cerca del umbral"]]
        .rename(columns={"pers": "P persist.", "antes": "P antes", "final": "P después"})
        .style.format({"s_C": "{:.2f}", "P persist.": "{:.2f}", "P antes": "{:.2f}", "P después": "{:.2f}"}).hide(axis="index"))
perfil_err = pd.DataFrame({
    "Errores después": [err.transicion.mean(), err["cerca del umbral"].mean(), err.eventos.median(), len(err)],
    "Aciertos después": [fuera.loc[fuera["acierto después"], "transicion"].mean(), fuera.loc[fuera["acierto después"], "cerca del umbral"].mean(),
                         fuera.loc[fuera["acierto después"], "eventos"].median(), int(fuera["acierto después"].sum())]},
    index=["% transiciones", "% a ±10 pp del umbral", "Mediana de eventos", "Filas"])
tabla(perfil_err, dec=2)
cruce = pd.crosstab(fuera["acierto antes"].map({True: "antes acierta", False: "antes falla"}),
                    fuera["acierto después"].map({True: "después acierta", False: "después falla"}))
display(cruce)
tr_f = fuera[fuera.transicion]
tr_ok = lambda c: int(((tr_f[c] >= UMBRAL) == (tr_f.y == 1)).sum())
print(f"Transiciones 2023-2026: {len(tr_f)} | persistencia: {tr_ok('pers')} | persistencia calibrada: {tr_ok('pers_cal')} | "
      f"antes (semilla 42): {tr_ok('antes')} | antes (ensamble {K_ENS}): {tr_ok('antes10')} | después: {tr_ok('final')}")
# Regla de decisión por bandas (P >= 0,7 C consolidado; 0,3-0,7 transición; < 0,3 D consolidado) frente al perfil observado
def banda(p_):
    return np.where(np.asarray(p_) >= .7, "C consolidado", np.where(np.asarray(p_) >= .3, "Transición", "D consolidado"))
filas_b = []
for conj, df_ in [("2023-2025", fuera[fuera.anio <= 2025]), ("2026", fuera[fuera.anio == 2026])]:
    for nombre, c in [("Después: MLP U3", "final"), ("Persistencia", "pers"), ("Persistencia calibrada", "pers_cal")]:
        b_ = banda(df_[c]); cons = b_ != "Transición"
        ok_ = ((b_ == "C consolidado") & (df_.perfil == "C")) | ((b_ == "D consolidado") & (df_.perfil == "D"))
        filas_b.append({"Conjunto": conj, "Modelo": nombre, "Filas": len(df_), "En banda consolidada": int(cons.sum()),
                        "Aciertos en banda consolidada": int(ok_.sum()),
                        "C consolidado (C real / total)": f"{int(((b_ == 'C consolidado') & (df_.perfil == 'C')).sum())}/{int((b_ == 'C consolidado').sum())}",
                        "D consolidado (D real / total)": f"{int(((b_ == 'D consolidado') & (df_.perfil == 'D')).sum())}/{int((b_ == 'D consolidado').sum())}",
                        "En transición": int((~cons).sum())})
BANDAS = pd.DataFrame(filas_b).set_index(["Conjunto", "Modelo"])
print("Regla de decisión por bandas frente al perfil observado:")
display(BANDAS)
seg = fuera[fuera.depto.isin(["arauca", "casanare", "meta", "cesar", "valle del cauca", "bolivar", "choco", "huila"])].pivot_table(index="departamento", columns="anio", values="final").round(2)
obs = fuera.pivot_table(index="departamento", columns="anio", values="perfil", aggfunc="first").reindex(seg.index)
display(pd.concat({"P(C) después": seg, "Perfil observado": obs}, axis=1))

fig, ax = plt.subplots(figsize=(10.5, 6))
for anio_, color in zip(sorted(fuera.anio.unique()), [COL["mlp"], COL["logit"], COL["val"], COL["test"]]):
    s_ = fuera[fuera.anio == anio_]
    ax.scatter(s_.s_C, s_.final, s=40, color=color, edgecolor=FONDO, lw=1.2, label=str(anio_), zorder=3)
for _, r in err.iterrows():
    ax.annotate(f"{r.departamento} {r.anio}", (r.s_C, r.final), xytext=(4, 3), textcoords="offset points", fontsize=7.2, color=TINTA2)
ax.axvline(.5, color=EJE, lw=1); ax.axhline(UMBRAL, color=EJE, lw=1)
ax.axvspan(.4, .6, color=GRID, alpha=.5, lw=0, zorder=0)
ax.text(.02, .97, "falso positivo", fontsize=8.5, color=TENUE, va="top"); ax.text(.98, .03, "falso negativo", fontsize=8.5, color=TENUE, ha="right")
ax.set_xlabel("Proporción observada de eventos C en t ($s^C$); banda gris: ±10 pp del umbral"); ax.set_ylabel("P(C) del modelo final (información de t-1)")
ax.set_title("Errores del modelo final, 2023–2026 (etiquetados)"); ax.legend(fontsize=8.5, title="año", loc="center right")
fig.tight_layout(); guardar(fig, "fig12_errores"); plt.show()
""")

lectura("errores")

md(r"""
## 10.2 ¿Qué aprendió la red? Importancia por permutación e interacción memoria × estructura
""")

code(r"""
# =============================================================================
# 10.2 Importancia por permutación fuera de muestra (por pliegue) y dependencia parcial 2D
# =============================================================================
rng_i = np.random.default_rng(C.SEED); imp = {f: [] for f in VARS_FINAL}
spec_imp = replace(FINAL, semillas=SEM5)
for v, tr, te in E.particiones_origen_movil(datos):
    m_, e_ = P.entrenar_especificacion(datos, SEQ, spec_imp, tr)
    X_te = datos.loc[te, VARS_FINAL].copy(); y_te = datos.loc[te, "y"].values
    base_ll = E.bce(y_te, M.predecir_ensamble(m_, e_.transformar(X_te.values)))
    for f in VARS_FINAL:
        for _ in range(30):
            Xp = X_te.copy(); Xp[f] = rng_i.permutation(Xp[f].values)
            imp[f].append(E.bce(y_te, M.predecir_ensamble(m_, e_.transformar(Xp.values))) - base_ll)
IMP = pd.DataFrame({"ΔLog-loss media": {f: np.mean(v) for f, v in imp.items()}, "Desv. est.": {f: np.std(v) for f, v in imp.items()}}).sort_values("ΔLog-loss media")

def pdp_2d(modelos, esc, fx, fy, n=35):
    base_ = datos.loc[tr26, VARS_FINAL].reset_index(drop=True)
    gx = np.linspace(*datos[fx].quantile([.02, .98]), n); gy = np.linspace(*datos[fy].quantile([.02, .98]), n)
    GX, GY = np.meshgrid(gx, gy); rep_ = pd.concat([base_] * (n * n), ignore_index=True)
    rep_[fx] = np.repeat(GX.ravel(), len(base_)); rep_[fy] = np.repeat(GY.ravel(), len(base_))
    if isinstance(modelos, list) and hasattr(modelos[0], "predecir"):
        z = M.predecir_ensamble(modelos, esc.transformar(rep_[VARS_FINAL].values))
    else:
        z = modelos[0].predict_proba(esc.transformar(rep_[VARS_FINAL].values)[0])[:, 1]
    return gx, gy, z.reshape(n * n, len(base_)).mean(1).reshape(n, n)
LOG_MODS, LOG_ESC = P.entrenar_especificacion(datos, SEQ, LOG_F, tr26)
fig = plt.figure(figsize=(16, 5.2)); gs_ = fig.add_gridspec(1, 3, width_ratios=[1, 1.05, 1.05], wspace=.35)
ax = fig.add_subplot(gs_[0])
ax.barh(range(len(IMP)), IMP["ΔLog-loss media"], xerr=IMP["Desv. est."], color=[COL["u3"] if f not in V.BASE else COL["D"] for f in IMP.index], error_kw=dict(ecolor=TENUE))
ax.set_yticks(range(len(IMP)), [V.ETIQ[f] for f in IMP.index], fontsize=8.5); ax.axvline(0, color=EJE, lw=1)
ax.set_xlabel("Aumento de la log-loss al permutar"); ax.set_title("(a) Importancia fuera de muestra (verde = U3)")
fy_ = IMP.index[-2] if IMP.index[-1] == "m1_sC_t1" else IMP.index[-1]
fy_ = "x2b_letalidad" if fy_ in ("m1_sC_t1", "d1_sC_H2") else fy_
for k, (nombre, mods_, esc_) in enumerate([("MLP U3 (entrenado 2021-2025)", MODELOS_2025, ESC_2025), ("Logística, mismas variables", LOG_MODS, LOG_ESC)]):
    ax = fig.add_subplot(gs_[k + 1])
    gx, gy, Z = pdp_2d(mods_, esc_, "m1_sC_t1", fy_)
    cs = ax.contourf(gx, gy, Z, levels=np.linspace(0, 1, 11), cmap=G.CMAP_P, vmin=0, vmax=1); ax.contour(gx, gy, Z, levels=[.5], colors=TINTA, linewidths=1.6)
    for clase, color in [(0, COL["D"]), (1, COL["C"])]:
        s_ = datos[tr26 & (datos.y == clase).values]; ax.scatter(s_["m1_sC_t1"], s_[fy_], s=16, color=color, edgecolor=TINTA, lw=.4, zorder=3)
    ax.grid(False); ax.set_xlabel(V.ETIQ["m1_sC_t1"]); ax.set_ylabel(V.ETIQ[fy_]); ax.set_title(f"({'bc'[k]}) {nombre}")
cb = fig.colorbar(cs, ax=fig.axes[1:], fraction=.03, pad=.02); cb.set_label("P(C) parcial promedio"); cb.outline.set_visible(False)
guardar(fig, "fig13_importancia_pdp"); plt.show()
tabla(IMP.iloc[::-1].rename(index=V.ETIQ), dec=4)
print("Interacción mostrada en (b)-(c): memoria m1 ×", V.ETIQ[fy_])
""")

lectura("interpretacion")

md(r"""
---
# 11. Integración final: pipeline reproducible

El cuaderno **usa** el pipeline, pero no **es** el pipeline. Todo el proceso vive en el paquete `src/perfil_amenaza/`,
se ejecuta con una sola orden y produce un **artefacto único** de despliegue que no depende del cuaderno.

```
Proyecto_Unidad_3/
├── configuracion.json            ← escrita en §8.1: variables, red, ensamble, calibración, protocolo y decisiones
├── requirements.txt · README.md · pytest.ini
├── datos/acled-data-export.csv · datos/externos/ (coca, bosque y, si se agrega, UCDP)
├── src/perfil_amenaza/
│   ├── config.py      rutas, constantes, mapeo de subtipos, IPM, contigüidad departamental
│   ├── datos.py       contrato y auditoría, panel, bloques trimestrales, fuentes externas, UCDP, control 2026
│   ├── variables.py   predictoras en t-1 (BASE, MEMORIA, DINAMICA, VECINDAD, ACTORES) y secuencias
│   ├── modelo.py      redes Keras (MLP, multitarea, GRU, CNN, embedding), bucle GradientTape, calibración, artefacto
│   ├── evaluacion.py  métricas, ECE, bootstrap por departamento, McNemar, regla de retención, origen móvil, espacial,
│   │                  espacio-temporal, PSI con piso de ruido
│   ├── pipeline.py    orquestación de 8 etapas y línea de comandos
│   └── graficos.py    estilo de figuras
├── tests/test_pipeline.py        10 pruebas: contrato (y su caso negativo), fuga temporal, escalador, artefacto,
│                                 persistencia calibrada, partición espacio-temporal, regla de retención, determinismo, réplica U2
├── Proyecto_U3_Integrador.ipynb  este cuaderno
└── salidas_u3/                   métricas, predicciones, figuras, modelos/, TARJETA_MODELO.md
```
""")

code(r"""
# =============================================================================
# 11.1 Diagrama del pipeline integrado
# =============================================================================
etapas = [("1 · Datos", "contrato + auditoría\npanel depto-año\netiqueta y control 2026", "dataset_modelado_u3.csv"),
          ("2 · Variables", f"{len(VARS_FINAL)} predictoras en t-1\nescalador solo con train", "configuracion.json"),
          ("3 · Evaluación", "origen móvil 2023-2025\nvs. persistencia y U2", "metricas_seleccion.csv"),
          ("4 · Calibración", f"método: {CALIBRACION}\numbral {UMBRAL}", "parámetros en el artefacto"),
          ("5 · Prueba", "2026, una sola vez\nmodelo hasta 2025", "metricas_prueba_2026.csv"),
          ("6 · Despliegue", f"ensamble {K_ENS} redes\n2021-2026 → 1 archivo", "perfil_amenaza_u3.keras"),
          ("7 · Proyección", "P(C) 2027, categoría,\nalerta A/B, PSI", "proyeccion_2027_u3.csv"),
          ("8 · Tarjeta", "uso previsto, límites,\nmétricas, deriva", "TARJETA_MODELO.md")]
fig, ax = plt.subplots(figsize=(16, 4.6)); ax.axis("off"); ax.grid(False)
for i, (t, d_, a) in enumerate(etapas):
    x = i * 2.0
    ax.add_patch(FancyBboxPatch((x - .05, 1.25), 1.8, 1.55, boxstyle="round,pad=0.03,rounding_size=0.12", facecolor="#eef4fc" if i not in (2, 4) else "#e8f6ef", edgecolor=COL["mlp"] if i not in (2, 4) else COL["u3"], lw=1.4))
    ax.text(x + .85, 2.55, t, ha="center", va="center", fontsize=10, fontweight="semibold")
    ax.text(x + .85, 1.85, d_, ha="center", va="center", fontsize=7, color=TINTA2)
    ax.add_patch(FancyBboxPatch((x - .02, .25), 1.74, .55, boxstyle="round,pad=0.02,rounding_size=0.08", facecolor=FONDO, edgecolor=EJE, lw=1, ls="--"))
    ax.text(x + .85, .52, a, ha="center", va="center", fontsize=6.2, color=TINTA2, family="monospace")
    ax.annotate("", xy=(x + .85, .82), xytext=(x + .85, 1.23), arrowprops=dict(arrowstyle="->", color=TENUE, lw=1))
    if i < len(etapas) - 1:
        ax.annotate("", xy=(x + 1.98, 2.03), xytext=(x + 1.72, 2.03), arrowprops=dict(arrowstyle="->", color=TINTA, lw=1.4))
ax.annotate("", xy=(4 * 2 + .85, 3.05), xytext=(2 * 2 + .85, 3.05), arrowprops=dict(arrowstyle="<->", color=COL["u3"], lw=1, connectionstyle="arc3,rad=-0.25"))
ax.text(3 * 2 + .85, 3.5, "las decisiones se congelan antes de la prueba", ha="center", fontsize=8, color=COL["u3"])
ax.text(0, -.25, "python -m perfil_amenaza.pipeline todo      ·      python -m perfil_amenaza.pipeline predecir --anio 2026      ·      python -m pytest -q tests",
        fontsize=8.5, family="monospace", color=TINTA)
ax.set_xlim(-.2, 16); ax.set_ylim(-.5, 3.7); ax.set_title("Pipeline integrado del modelo final (U3): cada etapa deja un artefacto verificable")
guardar(fig, "fig14_pipeline"); plt.show()
""")

code(r"""
# =============================================================================
# 11.2 Pruebas automáticas y ejecución de extremo a extremo por línea de comandos
# =============================================================================
entorno = {**os.environ, "PYTHONPATH": str(RAIZ / "src"), "TF_CPP_MIN_LOG_LEVEL": "3"}
t0 = time.time()
r_test = subprocess.run([sys.executable, "-m", "pytest", "-q", "tests"], cwd=RAIZ, env=entorno, capture_output=True, text=True)
print("$ python -m pytest -q tests"); print("\n".join(l for l in r_test.stdout.strip().splitlines() if "passed" in l or "failed" in l or "PASS" in l or "FAIL" in l or "::" in l))
print(f"({time.time() - t0:.0f} s)\n")
t0 = time.time()
r_cli = subprocess.run([sys.executable, "-m", "perfil_amenaza.pipeline", "todo"], cwd=RAIZ, env=entorno, capture_output=True, text=True)
print("$ python -m perfil_amenaza.pipeline todo"); print(r_cli.stdout.strip())
if r_cli.returncode != 0:
    print(r_cli.stderr[-3000:])
assert r_test.returncode == 0 and r_cli.returncode == 0
""")

code(r"""
# =============================================================================
# 11.3 El artefacto único: carga, estructura y consistencia con lo evaluado
# =============================================================================
ART, META = P.cargar_artefacto()
ruta_art = C.DIR_MODELOS / "perfil_amenaza_u3.keras"
print(f"{ruta_art.name}: {ruta_art.stat().st_size / 1024:.0f} KB | entrada: {tuple(ART.inputs[0].shape)} (variables crudas: {', '.join(META['variables'])}) | "
      f"salidas: {[tuple(o.shape) for o in ART.outputs]} | calibración: {META['calibracion']}")
capas_art = pd.DataFrame([(l.name, type(l).__name__, M.n_parametros(l) if hasattr(l, "trainable_variables") else 0) for l in ART.layers], columns=["Capa", "Tipo", "Parámetros entrenables"])
display(capas_art.style.hide(axis="index"))
met_cli_sel = pd.read_csv(C.DIR_SALIDA / "metricas_seleccion.csv", index_col=0)
met_cli_te = pd.read_csv(C.DIR_SALIDA / "metricas_prueba_2026.csv", index_col=0)
fin_cli = met_cli_sel.index[-1]
consist = pd.DataFrame({"Cuaderno (§9)": [T91.loc["Después: MLP U3 (ensamble)", "Log-loss"], T93.loc["Después: MLP U3 (ensamble)", "Log-loss"]],
                        "Pipeline CLI": [met_cli_sel.loc[fin_cli, "Log-loss"], met_cli_te.loc[fin_cli, "Log-loss"]]}, index=["Log-loss origen móvil 2023-2025", "Log-loss prueba 2026"])
consist["Diferencia"] = consist["Pipeline CLI"] - consist["Cuaderno (§9)"]
tabla(consist, dec=4)
PSI = pd.read_csv(C.DIR_SALIDA / "deriva_psi_2026.csv", index_col=0)
print("Deriva de las predictoras de la proyección (información 2026, los 30 departamentos del entrenamiento) frente a las filas "
      "de entrenamiento; 'p95 sin deriva' es el percentil 95 del PSI de una muestra del mismo tamaño sacada de la referencia:")
tabla(PSI.rename(index=V.ETIQ), dec=3)
display(Markdown((C.DIR_SALIDA / "TARJETA_MODELO.md").read_text(encoding="utf-8")))
""")

lectura("pipeline")

md(r"""
---
# 12. Del modelo a la decisión: priorización territorial 2027

La proyección sale del **artefacto único** (no de objetos en memoria): ensamble de 10 redes reentrenado con todas
las etiquetas 2021–2026 y aplicado a la información de 2026 (33 semanas, anualizada). Se compara con la proyección
de la U2 para ver qué cambió al incorporar la memoria y la dinámica reciente.

> Es un **insumo analítico ilustrativo**, no una recomendación operativa: resume el patrón aprendido de datos públicos
> agregados y debe contrastarse con inteligencia de campo.
""")

code(r"""
# =============================================================================
# 12.1 Proyección 2027 con el artefacto, incertidumbre entre redes y cambio frente a la U2
# =============================================================================
proy = pd.read_csv(C.DIR_SALIDA / "proyeccion_2027_u3.csv")
filas27, _ = V.filas_proyeccion(acled, pext, 2026, actores=ACTORES)
x27 = filas27.set_index("depto").loc[proy.depto, META["variables"]].values.astype("float32")
norm = ART.get_layer("estandarizacion")(x27)
miembros = [l for l in ART.layers if l.name.startswith("miembro_")]
P_m = np.array([np.asarray((lambda o: o[0] if isinstance(o, (list, tuple)) else o)(m(norm))).ravel() for m in miembros])
proy["DE entre redes"] = P_m.std(0)
u2p = pd.read_csv(RAIZ / "referencias_u2" / "proyeccion_2027.csv")[["depto", "P(C) 2027", "Categoría 2027"]].rename(columns={"P(C) 2027": "P(C) U2", "Categoría 2027": "Categoría U2"})
proy = proy.merge(u2p, on="depto", how="left")
# Cercanía a la inercia (no es acierto): ¿la proyección 2027 contradice el perfil observado en 2026?
proy["¿Coincide con el perfil 2026?"] = np.where(proy["perfil_info"].isna(), "–", np.where((proy["P(C) 2027"] >= .5) == (proy["perfil_info"] == "C"), "sí", "revisar"))
cols_p = ["departamento", "perfil_info", "s_C_info", "P(C) 2027", "DE entre redes", "Categoría 2027", "Categoría U2", "¿Coincide con el perfil 2026?", "s_AB observada 2025-26", "Énfasis sugerido", "confiable_info"]
display(proy[cols_p].rename(columns={"perfil_info": "Perfil 2026", "s_C_info": "s_C 2026", "confiable_info": "Datos suficientes"})
        .style.format({"s_C 2026": "{:.2f}", "P(C) 2027": "{:.2f}", "DE entre redes": "{:.3f}", "s_AB observada 2025-26": "{:.1%}"})
        .background_gradient(subset=["P(C) 2027"], cmap=G.CMAP_P, vmin=0, vmax=1).hide(axis="index"))
CAT = proy["Categoría 2027"].value_counts()
print("Categorías 2027 (U3):", CAT.to_dict(), "| U2:", proy["Categoría U2"].value_counts().to_dict())
display(pd.crosstab(proy["Categoría U2"], proy["Categoría 2027"], rownames=["U2"], colnames=["U3"]))
REVISAR = proy.loc[proy["¿Coincide con el perfil 2026?"] == "revisar", "departamento"].tolist()
ALERTA = proy.loc[proy["alerta_AB"].fillna(False), "departamento"].tolist()
print(f"Contradicen el perfil observado en 2026: {len(REVISAR)} ({', '.join(REVISAR)}) | alerta A/B: {', '.join(ALERTA)}")
POCOS = proy.loc[~proy["confiable_info"].astype(bool), ["departamento", "Categoría 2027"]]
print("Departamentos con menos de 10 eventos en 2026 (requieren revisión experta):", POCOS.set_index("departamento")["Categoría 2027"].to_dict())
ab_, nac_ = P.alerta_ab(pext)
print(f"Alerta A/B: proporción A+B 2025-2026 >= 2 × nacional ({nac_:.1%} → umbral {2 * nac_:.1%}) y >= 5 eventos A+B")

# Con la información parcial de 2026, las "últimas 26 semanas" son las semanas 8-33, no el segundo semestre.
# ¿Cuánto cambia d1 por medirlo en esa ventana? Se compara en 2020-2025, donde ambos están disponibles.
b_full = D.bloques_trimestrales(acled); b_part = D.bloques_trimestrales(acled[acled.semana_anio <= N_SEM_2026])
def d1_de(b):
    h = b[b.bloque <= 1].groupby(["depto", "anio"])[["C", "eventos"]].sum()
    return pd.Series(V.contraer(h.C, h.eventos), index=h.index)
d1c = pd.concat([d1_de(b_full).rename("2.º semestre"), d1_de(b_part).rename(f"semanas {N_SEM_2026 - 25}-{N_SEM_2026}")], axis=1).dropna()
d1c = d1c[d1c.index.get_level_values("anio") <= 2025]
D1_CORR, D1_MAE = d1c.corr().iloc[0, 1], (d1c.iloc[:, 0] - d1c.iloc[:, 1]).abs().mean()
print(f"d1 con la ventana de 2026 frente al 2.º semestre (2020-2025, {len(d1c)} celdas): correlación {D1_CORR:.2f}, error absoluto medio {D1_MAE:.3f}")
""")

code(r"""
# =============================================================================
# 12.2 Mapa esquemático de la proyección 2027 (centroides) y transición U2 -> U3
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 9.5), gridspec_kw={"width_ratios": [1.05, 1]})
ax = axes[0]
tam = 40 + 12 * np.sqrt((proy["eventos_info"] / proy["semanas"] * 52).clip(lower=1))
sc = ax.scatter(proy.lon, proy.lat, s=tam, c=proy["P(C) 2027"], cmap=G.CMAP_P, vmin=0, vmax=1, edgecolor=FONDO, lw=2, zorder=3)
al = proy[proy["alerta_AB"].fillna(False)]
ax.scatter(al.lon, al.lat, s=tam[al.index] + 260, facecolor="none", edgecolor="#d03b3b", lw=2, zorder=2)
izq = {"valle del cauca", "risaralda", "quindio", "sucre", "cordoba", "choco"}
for _, r in proy.iterrows():
    i_ = r.depto in izq
    ax.annotate(f"{r.departamento}\n{r['P(C) 2027']:.2f}", (r.lon, r.lat), xytext=(-9 if i_ else 9, -3), textcoords="offset points",
                ha="right" if i_ else "left", fontsize=7.2, color=TINTA if r.confiable_info else TENUE)
cb = fig.colorbar(sc, ax=ax, fraction=.035, pad=.02); cb.set_label("P(perfil C) 2027"); cb.outline.set_visible(False)
ax.set_xlabel("Longitud"); ax.set_ylabel("Latitud"); ax.set_aspect("equal")
ax.set_title("(a) P(C) 2027 por departamento (aro rojo = alerta A/B; gris = pocos datos)")
ax = axes[1]
orden = proy.sort_values("P(C) 2027")
y_ = np.arange(len(orden))
ax.hlines(y_, orden["P(C) U2"], orden["P(C) 2027"], color=EJE, lw=1.5)
ax.scatter(orden["P(C) U2"], y_, color=COL["mlp"], s=30, label="U2 (modelo base, ensamble 5)", zorder=3)
ax.scatter(orden["P(C) 2027"], y_, color=COL["u3"], s=40, label="U3 (modelo final, ensamble 10)", zorder=3)
ax.errorbar(orden["P(C) 2027"], y_, xerr=orden["DE entre redes"], fmt="none", ecolor=COL["u3"], lw=1, alpha=.6)
ax.set_yticks(y_, [f"{d}  [{p if isinstance(p, str) else '–'}]" for d, p in zip(orden.departamento, orden.perfil_info)], fontsize=7.5)
for u in (.3, .5, .7):
    ax.axvline(u, color=TENUE if u != .5 else TINTA, lw=.8, ls="--")
ax.set_xlabel("P(C) 2027  [perfil observado 2026 entre corchetes]"); ax.set_title("(b) Cambio de la proyección U2 → U3"); ax.legend(fontsize=8.5, loc="lower right")
fig.tight_layout(); guardar(fig, "fig15_proyeccion_2027"); plt.show()
""")

lectura("decision")

md(r"""
---
# 13. Conclusiones técnicas y recomendaciones
""")

lectura("conclusiones")

md(r"""
---
# 14. Trabajo en equipo, bitácora y uso de IA
""")

lectura("equipo")

md(r"""
---
# 15. Matriz rúbrica → evidencia
""")

lectura("rubrica")

md(r"""
---
# 16. Artefactos y figuras para el informe
""")

code(r"""
# =============================================================================
# 16.1 Artefactos producidos (cuaderno + pipeline)
# =============================================================================
pd.DataFrame({"tiempo total del cuaderno (min)": [round((time.time() - T0_CUADERNO) / 60, 1)]}).pipe(display)
dec_tab = pd.DataFrame(DECISIONES); dec_tab.to_csv(C.DIR_SALIDA / "decisiones_intervenciones.csv", index=False)
ESC.to_csv(C.DIR_SALIDA / "escalera_mejora.csv"); CTRL.to_csv(C.DIR_SALIDA / "controles_datos.csv"); REJ.to_csv(C.DIR_SALIDA / "rejilla_I6.csv", index=False)
T91.to_csv(C.DIR_SALIDA / "antes_despues_origen_movil.csv"); T93.to_csv(C.DIR_SALIDA / "antes_despues_prueba_2026.csv"); esp_tab.to_csv(C.DIR_SALIDA / "validacion_espacial.csv")
ET_TAB.to_csv(C.DIR_SALIDA / "validacion_espacio_temporal.csv"); BANDAS.to_csv(C.DIR_SALIDA / "bandas_decision.csv")
pd.concat({"origen móvil 2023-2025": PAR_91, "prueba 2026": PAR_26, "espacio-temporal": PAR_ET}).to_csv(C.DIR_SALIDA / "comparaciones_pareadas.csv")
archivos = sorted(p for p in C.DIR_SALIDA.rglob("*") if p.is_file())
display(pd.DataFrame({"Archivo": [str(p.relative_to(C.DIR_SALIDA)) for p in archivos], "KB": [round(p.stat().st_size / 1024, 1) for p in archivos]}).style.hide(axis="index"))
""")

md(r"""
## Anexo A. Figuras del informe en tamaño de impresión

Las figuras del cuaderno están pensadas para pantalla. Esta celda redibuja las que usará el informe (máximo 12
páginas) con el ancho útil de una página carta (6,6 pulgadas), a 300 dpi, **a partir de los resultados ya
calculados**, sin reentrenar nada. Se guardan en `salidas_u3/figuras_informe/`.
""")

code(r"""
# =============================================================================
# Anexo A. Figuras R1-R5 para el informe (carta, 6,6" de ancho, 300 dpi)
# =============================================================================
DIR_INF = C.DIR_SALIDA / "figuras_informe"
with mpl.rc_context({"font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5, "xtick.labelsize": 6.8, "ytick.labelsize": 6.8, "legend.fontsize": 6.8}):
    # R1 escalera
    fig, ax = plt.subplots(figsize=(6.6, 2.6)); x = np.arange(len(ESC))
    ax.bar(x, ESC["LL media"], yerr=ESC["LL DE"], color=[COL["mlp"]] + [COL["u3"]] * (len(ESC) - 1), width=.6, ecolor=TINTA2, capsize=2)
    for xi, v in zip(x, ESC["LL media"]):
        ax.text(xi, v + .005, f"{v:.3f}", ha="center", fontsize=6.5)
    ax.axhline(M_PERS["Log-loss"], color=COL["pers"], ls="--", lw=1); ax.text(len(ESC) - .5, M_PERS["Log-loss"], "persistencia", ha="right", va="bottom", fontsize=6.5)
    ax.set_xticks(x, ESC.index, rotation=12); ax.set_ylabel("Log-loss 2023–2025"); ax.set_ylim(.25, ESC["LL media"].max() * 1.1)
    ax.set_title("Escalera de mejora (media ± DE entre semillas)"); fig.tight_layout(); guardar(fig, "R1_escalera", DIR_INF, 300); plt.close(fig)
    # R2 antes vs después por año
    fig, ax = plt.subplots(figsize=(6.6, 2.5))
    for nombre, c, color, ls in [("Persistencia", "pers", COL["pers"], "--"), ("Antes (U2)", "antes", COL["mlp"], ":"), ("Después (U3)", "final", COL["u3"], "-")]:
        s_ = anual.groupby("particion").apply(lambda g: E.bce(g.y, g[c])); ax.plot(s_.index, s_.values, color=color, ls=ls, marker="o", ms=4, label=nombre)
    ax.set_ylabel("Log-loss"); ax.set_title("Log-loss por año (2026 = prueba única)"); ax.legend(ncol=3, loc="upper left")
    fig.tight_layout(); guardar(fig, "R2_antes_despues_anual", DIR_INF, 300); plt.close(fig)
    # R3 controles
    fig, ax = plt.subplots(figsize=(6.6, 2.4)); o = CTRL["ΔLog-loss"].sort_values().index
    ax.barh(range(len(o)), CTRL.loc[o, "ΔLog-loss"], xerr=CTRL.loc[o, "EE"], color=[COL["A"] if v > 0 else COL["u3"] for v in CTRL.loc[o, "ΔLog-loss"]], height=.6)
    ax.axvline(0, color=TINTA, lw=.8); ax.set_yticks(range(len(o)), [s.replace("ctrl: ", "") for s in o]); ax.set_xlabel("Δ log-loss vs. referencia (+ = peor)")
    ax.set_title("Controles de preparación de datos"); fig.tight_layout(); guardar(fig, "R3_controles", DIR_INF, 300); plt.close(fig)
    # R4 rejilla anidada
    fig, ax = plt.subplots(figsize=(3.3, 2.4)); x = np.arange(3)
    ax.bar(x - .18, ANIDADA.iloc[:3, 1], width=.34, color=COL["D"], label="2023-24 (elección)"); ax.bar(x + .18, ANIDADA.iloc[:3, 2], width=.34, color=COL["val"], label="2025 (fuera)")
    ax.set_xticks(x, ["Mínimo", "1 EE", "Vigente"]); ax.set_ylabel("Log-loss"); ax.set_title("Validación anidada (I6)"); ax.legend(loc="upper left")
    fig.tight_layout(); guardar(fig, "R4_anidada", DIR_INF, 300); plt.close(fig)
    # R5 proyección
    fig, ax = plt.subplots(figsize=(6.6, 4.2)); o = proy.sort_values("P(C) 2027"); yy = np.arange(len(o))
    ax.hlines(yy, o["P(C) U2"], o["P(C) 2027"], color=EJE, lw=1); ax.scatter(o["P(C) U2"], yy, s=10, color=COL["mlp"], label="U2"); ax.scatter(o["P(C) 2027"], yy, s=14, color=COL["u3"], label="U3")
    ax.set_yticks(yy, o.departamento, fontsize=5.6); [ax.axvline(u, color=TENUE, lw=.6, ls="--") for u in (.3, .5, .7)]
    ax.set_xlabel("P(C) 2027"); ax.set_title("Proyección 2027: U2 vs. U3"); ax.legend(loc="lower right"); fig.tight_layout(); guardar(fig, "R5_proyeccion", DIR_INF, 300); plt.close(fig)
print("Figuras del informe:", sorted(p.name for p in DIR_INF.glob("*.png")))
print(f"Figuras del cuaderno ({len(G.FIGURAS)}):", ", ".join(G.FIGURAS))
""")

md(r"""
---
# Referencias

ACLED. (2024). *ACLED codebook*. Armed Conflict Location & Event Data Project. https://acleddata.com

Bergmeir, C., & Benítez, J. M. (2012). On the use of cross-validation for time series predictor evaluation. *Information Sciences, 191*, 192–213. https://doi.org/10.1016/j.ins.2011.12.028

Cawley, G. C., & Talbot, N. L. C. (2010). On over-fitting in model selection and subsequent selection bias in performance evaluation. *Journal of Machine Learning Research, 11*, 2079–2107.

Cho, K., van Merriënboer, B., Gulcehre, C., Bahdanau, D., Bougares, F., Schwenk, H., & Bengio, Y. (2014). Learning phrase representations using RNN encoder–decoder for statistical machine translation. En *Proceedings of EMNLP* (pp. 1724–1734).

Dietterich, T. G. (1998). Approximate statistical tests for comparing supervised classification learning algorithms. *Neural Computation, 10*(7), 1895–1923.

Friedman, J. H. (2001). Greedy function approximation: A gradient boosting machine. *The Annals of Statistics, 29*(5), 1189–1232.

Géron, A. (2017). *Hands-on machine learning with Scikit-Learn and TensorFlow*. O'Reilly Media.

Gneiting, T., & Raftery, A. E. (2007). Strictly proper scoring rules, prediction, and estimation. *Journal of the American Statistical Association, 102*(477), 359–378.

Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep learning*. MIT Press.

Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On calibration of modern neural networks. En *Proceedings of ICML* (pp. 1321–1330).

Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The elements of statistical learning* (2.ª ed.). Springer.

Hegre, H., Bell, C., Colaresi, M., Croicu, M., Hoyles, F., Jansen, R., … Vesco, P. (2021). ViEWS2020: Revising and evaluating the ViEWS political violence early-warning system. *Journal of Peace Research, 58*(3), 599–611.

Kingma, D. P., & Ba, J. (2015). Adam: A method for stochastic optimization. En *ICLR*. https://arxiv.org/abs/1412.6980

Lakshminarayanan, B., Pritzel, A., & Blundell, C. (2017). Simple and scalable predictive uncertainty estimation using deep ensembles. En *Advances in NeurIPS 30*.

Mitchell, M., Wu, S., Zaldivar, A., Barnes, P., Vasserman, L., Hutchinson, B., … Gebru, T. (2019). Model cards for model reporting. En *Proceedings of FAT\** (pp. 220–229).

Platt, J. (1999). Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods. En *Advances in Large Margin Classifiers* (pp. 61–74). MIT Press.

Raleigh, C., Linke, A., Hegre, H., & Karlsen, J. (2010). Introducing ACLED: An armed conflict location and event dataset. *Journal of Peace Research, 47*(5), 651–660.

Roberts, D. R., Bahn, V., Ciuti, S., Boyce, M. S., Elith, J., Guillera-Arroita, G., … Dormann, C. F. (2017). Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure. *Ecography, 40*(8), 913–929.

Ruder, S. (2017). An overview of multi-task learning in deep neural networks. *arXiv:1706.05098*.

Sundberg, R., & Melander, E. (2013). Introducing the UCDP Georeferenced Event Dataset. *Journal of Peace Research, 50*(4), 523–532.

Yurdakul, B. (2018). *Statistical properties of population stability index* (Tesis de doctorado). Western Michigan University.

*(Las referencias de datos de la U2 —SIMCI-UNODC/ODC, Global Forest Watch, DANE— siguen vigentes y se citan en el informe.)*
""")
