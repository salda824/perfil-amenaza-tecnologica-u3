"""
Pipeline integrado de extremo a extremo y línea de comandos.

    python -m perfil_amenaza.pipeline todo                # datos -> modelo -> evaluación -> artefacto -> proyección
    python -m perfil_amenaza.pipeline predecir --anio 2026 # usa solo el artefacto guardado

Etapas (cada una produce un artefacto en salidas_u3/):
  1. datos       contrato y auditoría del export, panel, etiqueta, fuentes externas  -> dataset_modelado_u3.csv
  2. variables   predictoras en t-1 (grupos de configuracion.json)
  3. evaluación  origen móvil 2023-2025 frente a la persistencia (cruda y calibrada) y al modelo base U2
                 (semilla 42 y ensamble de 10 semillas)                            -> metricas_seleccion.csv
  4. calibración solo si config lo pide; en la selección, encadenada hacia adelante -> parámetros en el artefacto
  5. prueba      2026, una sola vez, con el modelo entrenado hasta 2025             -> metricas_prueba_2026.csv
  6. despliegue  ensamble reentrenado con 2021-2026 + calibración en un solo .keras -> modelos/perfil_amenaza_u3.keras
  7. proyección  perfil 2027, alerta A/B y deriva (PSI)                              -> proyeccion_2027_u3.csv
  8. tarjeta     tarjeta del modelo (Markdown)                                       -> TARJETA_MODELO.md
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field, asdict, replace

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from . import config as C
from . import datos as D
from . import variables as V
from . import modelo as M
from . import evaluacion as E

IDS = {d: i for i, d in enumerate(sorted(C.NOMBRE))}   # código fijo de departamento para el embedding


@dataclass
class Especificacion:
    """Qué variables entran y con qué modelo; es la unidad que se compara en las intervenciones."""
    nombre: str
    variables: list
    red: M.ConfigRed | None = None      # None -> regresión logística
    semillas: tuple = (C.SEED,)         # más de una -> ensamble (I7)
    C_logistica: float = 3.0
    escalar: bool = True                # control: False = sin estandarización
    fuga_escalador: bool = False        # control negativo: escalador ajustado también con las filas de prueba

    def a_dict(self):
        d = asdict(self)
        d["red"] = asdict(self.red) if self.red else None
        return d

    @staticmethod
    def de_dict(d):
        red = M.ConfigRed(**{**d["red"], "capas": tuple(d["red"]["capas"])}) if d.get("red") else None
        return Especificacion(d["nombre"], list(d["variables"]), red, tuple(d["semillas"]), d.get("C_logistica", 3.0),
                              d.get("escalar", True), d.get("fuga_escalador", False))


def _entradas(datos, SEQ, filas, spec, esc):
    X = datos.loc[filas, spec.variables].values
    if spec.red is not None and spec.red.tipo in ("gru", "cnn"):
        return esc.transformar(X, SEQ[filas])
    ent = esc.transformar(X)
    if spec.red is not None and spec.red.tipo == "embedding":
        ent.append(datos.loc[filas, "depto"].map(IDS).values.astype("int32").reshape(-1, 1))
    return ent


def entrenar_especificacion(datos, SEQ, spec: Especificacion, tr):
    """Ajusta el escalador y el modelo (o el ensamble) con las filas `tr`. Devuelve (modelos, escalador)."""
    seq = spec.red is not None and spec.red.tipo in ("gru", "cnn")
    filas_esc = np.ones(len(datos), bool) if spec.fuga_escalador else tr     # la fuga solo existe como control
    esc = M.Escalador().ajustar(datos.loc[filas_esc, spec.variables].values, SEQ[filas_esc] if seq else None)
    if not spec.escalar:
        esc.mu, esc.sd = np.zeros_like(esc.mu), np.ones_like(esc.sd)
    ent = _entradas(datos, SEQ, tr, spec, esc)
    if spec.red is None:
        lr = LogisticRegression(C=spec.C_logistica, class_weight="balanced", max_iter=5000)
        return [lr.fit(ent[0], datos.loc[tr, "y"].values)], esc
    cfg = spec.red
    y_obj = datos.loc[tr, "s_C"].values if cfg.objetivo == "blando" else datos.loc[tr, "y"].values
    redes = [M.entrenar(ent, y_obj, cfg, semilla=s, y_duro=datos.loc[tr, "y"].values,
                        eventos=datos.loc[tr, "eventos"].values, comp=datos.loc[tr, V.COMPOSICION].values)
             for s in spec.semillas]
    return redes, esc


def predecir_especificacion(modelos, esc, datos, SEQ, spec, filas, composicion=False):
    ent = _entradas(datos, SEQ, filas, spec, esc)
    if spec.red is None:
        return modelos[0].predict_proba(ent[0])[:, 1]
    if composicion:
        return np.mean([m.composicion(ent) for m in modelos], axis=0)
    return M.predecir_ensamble(modelos, ent)


def ajustar_predecir(datos, SEQ, spec):
    """Cierre para evaluacion.validar: entrena con tr y predice te."""
    def f(tr, te):
        mods, esc = entrenar_especificacion(datos, SEQ, spec, tr)
        return {spec.nombre: predecir_especificacion(mods, esc, datos, SEQ, spec, te)}
    return f


def evaluar_semillas(spec, datos, SEQ, semillas=(1, 2, 3, 4, 5), particiones="movil") -> dict:
    """
    Entrena `spec` con cada semilla por separado en cada pliegue del esquema `particiones` y devuelve:
    "oos" (predicciones fuera de muestra, con "p" = media entre semillas), "por_semilla" (métricas de cada red),
    "P" (matriz semillas × filas) y "spec". Es la unidad de comparación de la regla de retención. Un modelo
    determinista (spec.red is None) se entrena una sola vez.
    """
    semillas = tuple(semillas) if spec.red is not None else (0,)
    part = list(E.particiones(particiones, datos))
    Ps, filas = [], []
    for s in semillas:
        oos = E.validar(datos, ajustar_predecir(datos, SEQ, replace(spec, semillas=(s,))), part)
        Ps.append(oos[spec.nombre].values)
        filas.append({"semilla": s, **E.metricas(oos["y"], oos[spec.nombre], transicion=oos["transicion"])})
    oos = oos.drop(columns=[spec.nombre]); oos["p"] = np.mean(Ps, axis=0)
    return {"oos": oos, "por_semilla": pd.DataFrame(filas).set_index("semilla"), "P": np.array(Ps), "spec": spec}



def persistencia(datos, tr, te):
    """Línea base de la U2: P(C) = s_C del año anterior (prevalencia de entrenamiento si no hubo eventos)."""
    s = datos.loc[te, "s_C_prev"].values.astype(float)
    return np.where(np.isnan(s), datos.loc[tr, "y"].mean(), s)


def _logit(p):
    p = np.clip(np.asarray(p, float), 1e-3, 1 - 1e-3)
    return np.log(p / (1 - p)).reshape(-1, 1)


def persistencia_calibrada(datos, tr, te):
    """
    Rival agregado tras la revisión de la U3: la inercia calibrada. Una logística de dos parámetros sobre
    logit(m1) = logit(s_C contraída de t-1), ajustada solo con las filas de entrenamiento (en el origen móvil, años
    anteriores al validado). Usa la misma información que la persistencia, pero la convierte en una probabilidad.
    """
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(_logit(datos.loc[tr, "m1_sC_t1"]), datos.loc[tr, "y"].values)
    return lr.predict_proba(_logit(datos.loc[te, "m1_sC_t1"]))[:, 1]


def alerta_ab(panel, desde=2025):
    """Alerta A/B de la U2: proporción A+B en desde..2026 >= 2 x promedio nacional y >= 5 eventos A+B."""
    ab = panel[panel["anio"] >= desde].groupby("depto")[["A", "B", "eventos"]].sum()
    ab["s_AB"] = (ab["A"] + ab["B"]) / ab["eventos"]
    nacional = (ab["A"].sum() + ab["B"].sum()) / ab["eventos"].sum()
    ab["alerta_AB"] = (ab["s_AB"] >= 2 * nacional) & ((ab["A"] + ab["B"]) >= 5)
    return ab, nacional


def categoria(p):
    return "C consolidado" if p >= 0.7 else ("Transición" if p >= 0.3 else "D consolidado")


ENFASIS = {"C consolidado": "ISR, detección temprana y movilidad protegida",
           "Transición": "Portafolio flexible y monitoreo trimestral del perfil",
           "D consolidado": "Orden público y capacidades mixtas"}


# ----------------------------------------------------------------------------------------------------------
# Ejecución completa
# ----------------------------------------------------------------------------------------------------------
def ejecutar(ruta_config=C.RUTA_CONFIG, verbose=True) -> dict:
    t0 = time.time()
    log = (lambda *a: print(*a, flush=True)) if verbose else (lambda *a: None)
    C.asegurar_carpetas()
    cfg = C.cargar_config(ruta_config)
    spec = Especificacion.de_dict(cfg["especificacion_final"])
    base_u2 = Especificacion.de_dict(cfg["especificacion_base_u2"])
    # "Antes" justo: la misma arquitectura base con las mismas semillas que el ensamble final
    base_ens = Especificacion(f"MLP base U2 (ensamble {len(spec.semillas)})", base_u2.variables, base_u2.red, spec.semillas)

    # 1-2. Datos y variables
    bruto = D.cargar_acled()
    acled = D.filtrar_horizonte(bruto)
    actores = D.cargar_ucdp() if "x3_actores" in spec.variables else None
    datos, SEQ, pext = V.construir_dataset(acled, actores=actores)
    datos.to_csv(C.DIR_SALIDA / "dataset_modelado_u3.csv", index=False)
    log(f"[1-2] datos: {len(datos)} filas, {datos.depto.nunique()} departamentos, {len(spec.variables)} predictoras")

    # 3. Evaluación con origen móvil 2023-2025 (la selección se hizo en el cuaderno con la misma partición)
    def fn(tr, te):
        out = {"Persistencia": persistencia(datos, tr, te), "Persistencia calibrada": persistencia_calibrada(datos, tr, te)}
        for s in (base_u2, base_ens, spec):
            mods, esc = entrenar_especificacion(datos, SEQ, s, tr)
            out[s.nombre] = predecir_especificacion(mods, esc, datos, SEQ, s, te)
        return out
    oos = E.validar(datos, fn, E.particiones_origen_movil(datos))
    metodo = cfg.get("calibracion", "ninguna")
    # El calibrador del artefacto se ajusta con todas las predicciones 2023-2025 (todas anteriores a la prueba). Para
    # medirlo en la selección sin mirar el futuro, cada pliegue se calibra solo con los anteriores.
    cal = M.Calibrador(metodo).ajustar(oos[spec.nombre], oos["y"])
    modelos_eval = ["Persistencia", "Persistencia calibrada", base_u2.nombre, base_ens.nombre, spec.nombre]
    if metodo != "ninguna":
        oos[spec.nombre + " (calibrado)"] = M.calibrar_adelante(oos["particion"], oos[spec.nombre], oos["y"], metodo)
        modelos_eval.append(spec.nombre + " (calibrado)")
    final = modelos_eval[-1]
    met_sel = E.resumir(oos, modelos_eval)
    met_sel.to_csv(C.DIR_SALIDA / "metricas_seleccion.csv")
    oos.to_csv(C.DIR_SALIDA / "predicciones_origen_movil.csv", index=False)
    log(f"[3-4] origen móvil 2023-2025 (n={len(oos)}): AUC {met_sel.loc[final, 'ROC-AUC']:.3f}, "
        f"log-loss {met_sel.loc[final, 'Log-loss']:.3f} (persistencia {met_sel.loc['Persistencia', 'Log-loss']:.3f}; "
        f"persistencia calibrada {met_sel.loc['Persistencia calibrada', 'Log-loss']:.3f}; "
        f"base U2 {met_sel.loc[base_u2.nombre, 'Log-loss']:.3f}; base U2 ensamble {met_sel.loc[base_ens.nombre, 'Log-loss']:.3f}); "
        f"calibración: {metodo}")

    # 5. Prueba 2026 (una sola vez): modelos entrenados con 2021-2025
    tr, te = (datos["anio"] <= 2025).values, (datos["anio"] == C.ANIO_PRUEBA).values
    prueba = datos.loc[te, ["depto", "departamento", "anio", "y", "s_C", "perfil", "perfil_prev", "transicion", "eventos"]].copy()
    prueba["Persistencia"] = persistencia(datos, tr, te)
    prueba["Persistencia calibrada"] = persistencia_calibrada(datos, tr, te)
    for s in (base_u2, base_ens, spec):
        mods, esc = entrenar_especificacion(datos, SEQ, s, tr)
        prueba[s.nombre] = predecir_especificacion(mods, esc, datos, SEQ, s, te)
    if metodo != "ninguna":
        prueba[final] = cal.aplicar(prueba[spec.nombre])
    met_te = E.resumir(prueba, modelos_eval)
    met_te.to_csv(C.DIR_SALIDA / "metricas_prueba_2026.csv")
    prueba.to_csv(C.DIR_SALIDA / "predicciones_prueba_2026.csv", index=False)
    log(f"[5] prueba 2026 (n={len(prueba)}): AUC {met_te.loc[final, 'ROC-AUC']:.3f}, log-loss {met_te.loc[final, 'Log-loss']:.3f} "
        f"(persistencia {met_te.loc['Persistencia', 'Log-loss']:.3f}; persistencia calibrada "
        f"{met_te.loc['Persistencia calibrada', 'Log-loss']:.3f}; base U2 ensamble {met_te.loc[base_ens.nombre, 'Log-loss']:.3f})")

    # 6. Despliegue: ensamble con todas las etiquetas (2021-2026) + calibración, en un solo artefacto
    todo = np.ones(len(datos), bool)
    redes, esc = entrenar_especificacion(datos, SEQ, spec, todo)
    artefacto = M.empaquetar(redes, esc, cal, len(spec.variables))
    ruta_art = C.DIR_MODELOS / "perfil_amenaza_u3.keras"
    artefacto.save(ruta_art)
    meta = {"variables": spec.variables, "especificacion": spec.a_dict(), "calibracion": metodo, "platt": {"a": cal.a, "b": cal.b},
            "entrenado_con": "2021-2026", "filas": int(len(datos)), "version": "3.0.0"}
    (C.DIR_MODELOS / "perfil_amenaza_u3.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"[6] artefacto: {ruta_art.name} ({ruta_art.stat().st_size / 1024:.0f} KB, {len(redes)} redes)")

    # 7. Proyección 2027 con el artefacto (no con los objetos en memoria) y control de deriva
    proy, _ = predecir_con_artefacto(acled, pext, anio_info=C.ANIO_PRUEBA, actores=actores)
    # Deriva: misma población que el entrenamiento (los 30 departamentos con filas etiquetadas), con su piso de ruido
    psi = E.tabla_psi(datos, proy[proy["depto"].isin(set(datos["depto"]))], spec.variables)
    psi.to_csv(C.DIR_SALIDA / "deriva_psi_2026.csv")
    proy.to_csv(C.DIR_SALIDA / "proyeccion_2027_u3.csv", index=False)
    log(f"[7] proyección 2027: {proy['Categoría 2027'].value_counts().to_dict()} | PSI > 0,25: {int((psi.PSI > .25).sum())}; "
        f"PSI por encima del ruido de muestreo (p95): {int(psi['supera el ruido'].sum())}")

    # 8. Tarjeta del modelo
    escribir_tarjeta(spec, met_sel, met_te, cal, psi, proy, len(datos), final, metodo)
    log(f"[8] tarjeta del modelo escrita. Tiempo total: {time.time() - t0:.0f} s")
    return {"datos": datos, "oos": oos, "met_sel": met_sel, "prueba": prueba, "met_te": met_te, "calibrador": cal,
            "proyeccion": proy, "psi": psi, "artefacto": ruta_art}


def cargar_artefacto(ruta=None):
    import keras
    ruta = ruta or C.DIR_MODELOS / "perfil_amenaza_u3.keras"
    meta = json.loads((ruta.with_suffix(".json")).read_text(encoding="utf-8"))
    return keras.models.load_model(ruta, custom_objects={"CalibracionPlatt": M.CalibracionPlatt}), meta


def predecir_con_artefacto(acled, pext, anio_info=C.ANIO_PRUEBA, actores=None, ruta=None):
    """Proyecta anio_info + 1 usando únicamente el artefacto guardado (prueba de integración)."""
    modelo, meta = cargar_artefacto(ruta)
    filas, _ = V.filas_proyeccion(acled, pext, anio_info, actores=actores)
    out = modelo.predict(filas[meta["variables"]].values.astype("float32"), verbose=0)
    p = np.asarray(out["P_C"]).ravel()
    proy = filas.copy()
    proy[f"P(C) {anio_info + 1}"] = p
    if "composicion_ABCD" in out:
        comp = np.asarray(out["composicion_ABCD"])
        proy[f"s_AB proyectada {anio_info + 1}"] = comp[:, 0] + comp[:, 1]
    panel = pext
    ab, nacional = alerta_ab(panel)
    proy = proy.merge(ab[["s_AB", "alerta_AB"]].rename(columns={"s_AB": "s_AB observada 2025-26"}),
                      left_on="depto", right_index=True, how="left")
    proy[f"Categoría {anio_info + 1}"] = proy[f"P(C) {anio_info + 1}"].map(categoria)
    proy["Énfasis sugerido"] = proy[f"Categoría {anio_info + 1}"].map(ENFASIS) + np.where(
        proy["alerta_AB"].fillna(False), " + evaluar C-UAS / detección de AEI", "")
    return proy.sort_values(f"P(C) {anio_info + 1}", ascending=False).reset_index(drop=True), nacional


def escribir_tarjeta(spec, met_sel, met_te, cal, psi, proy, n_filas, k, metodo):
    fila = lambda m, c: f"| {m} · {c} | {met_sel.loc[m, c]:.3f} | {met_te.loc[m, c]:.3f} |"
    txt = [
        "# Tarjeta del modelo: perfil de amenaza tecnológica (U3)", "",
        "**Uso previsto.** Priorizar, con un año de anticipación, el énfasis tecnológico (ISR, C-UAS, detección de AEI)",
        "por departamento. Insumo analítico; no sustituye la inteligencia de campo.", "",
        "**No usar para:** decisiones sobre personas o municipios individuales; departamentos con menos de 10 eventos",
        "al año (Amazonía y Orinoquía con pocos registros) sin revisión experta.", "",
        f"**Datos.** ACLED semanal agregado a departamento (2020-2026), SIMCI-ODC, Global Forest Watch, DANE. {n_filas} filas.",
        f"**Modelo.** {spec.red.describir() if spec.red else 'logística'}; ensamble de {len(spec.semillas)} semillas; " +
        (f"calibración {metodo} (a = {cal.a:.3f}, b = {cal.b:+.3f})." if metodo != "ninguna" else
         "sin recalibración (la calibración Platt no mejoró la log-loss fuera de muestra; ver §6.10 del cuaderno)."),
        f"**Variables ({len(spec.variables)}).** " + ", ".join(spec.variables), "",
        "| Métrica | Origen móvil 2023-2025 | Prueba 2026 |", "|---|---|---|",
        *[f"| {c} | {met_sel.loc[k, c]:.3f} | {met_te.loc[k, c]:.3f} |" for c in ["ROC-AUC", "Log-loss", "Brier", "MCC", "ECE", "Exactitud"]],
        "", "| Rivales | Origen móvil 2023-2025 | Prueba 2026 |", "|---|---|---|",
        *[fila(m, c) for m in ["Persistencia", "Persistencia calibrada"] for c in ["ROC-AUC", "Log-loss", "Brier"]], "",
        "La persistencia calibrada (logística sobre logit(m1) entrenada con años anteriores) se agregó como rival tras la",
        "revisión; no cambió el modelo adoptado. Frente a ella, la ventaja del modelo no está demostrada a escala departamental.", "",
        "**Limitaciones.** 30 unidades espaciales; etiqueta 2026 con 33 semanas; x3 (actores, UCDP) evaluada y no retenida;",
        "x6 es un proxy departamental; la prevalencia de C cambia entre años (32 % a 62 %); con la información parcial de",
        "2026, d1 resume las semanas 8 a 33 y no el segundo semestre.", "",
        f"**Deriva.** PSI de la información 2026 frente al entrenamiento (mismos 30 departamentos): variables con PSI > 0,25: "
        f"{', '.join(psi.index[psi.PSI > .25]) or 'ninguna'}; por encima del percentil 95 del ruido de muestreo: "
        f"{', '.join(psi.index[psi['supera el ruido']]) or 'ninguna'}.",
        "**Mantenimiento.** Reentrenar cada año con la etiqueta cerrada, revisar el PSI con su piso de ruido y reevaluar la",
        "calibración con encadenamiento hacia adelante (hoy no se calibra porque no mejora sin mirar el futuro).",
    ]
    (C.DIR_SALIDA / "TARJETA_MODELO.md").write_text("\n".join(txt), encoding="utf-8")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Pipeline del perfil de amenaza tecnológica (U3)")
    ap.add_argument("accion", choices=["todo", "predecir"], help="todo: pipeline completo; predecir: solo con el artefacto")
    ap.add_argument("--config", default=str(C.RUTA_CONFIG))
    ap.add_argument("--anio", type=int, default=C.ANIO_PRUEBA, help="año de información para predecir anio+1")
    a = ap.parse_args(argv)
    if a.accion == "todo":
        ejecutar(a.config)
    else:
        acled = D.filtrar_horizonte(D.cargar_acled())
        _, _, pext = V.construir_dataset(acled)
        proy, _ = predecir_con_artefacto(acled, pext, a.anio)
        cols = ["departamento", f"P(C) {a.anio + 1}", f"Categoría {a.anio + 1}", "Énfasis sugerido"]
        print(proy[cols].to_string(index=False, float_format=lambda v: f"{v:.2f}"))


if __name__ == "__main__":
    main()
