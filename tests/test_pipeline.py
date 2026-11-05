"""
Pruebas automáticas del pipeline (pytest). Se ejecutan desde la raíz del proyecto:

    python -m pytest -q tests

1. Contrato de datos: el export cumple el esquema y el dataset de modelado no tiene faltantes ni llaves repetidas.
2. Contrato (caso negativo): un export al que le falta una columna se rechaza con un error explícito.
3. Fuga temporal: alterar los eventos del año t no cambia ninguna predictora de la fila (d, t).
4. Escalador sin fuga: en la validación de origen móvil se ajusta solo con años anteriores al validado.
5. Artefacto: la capa Normalization del .keras guarda las medias y desviaciones del entrenamiento.
6. Persistencia calibrada: se ajusta solo con las filas de entrenamiento (las etiquetas de prueba no la cambian).
7. Validación espacio-temporal: nunca entrena con el año que predice ni con los departamentos de prueba.
8. Regla de retención: una variante determinista se compara con la media de las semillas de la referencia.
9. Determinismo: dos entrenamientos con la misma semilla producen las mismas probabilidades.
10. Reproducción de la U2: el dataset reconstruye las 171 filas y las 7 predictoras de la Unidad 2.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

from perfil_amenaza import config as C, datos as D, variables as V, modelo as M, pipeline as P, evaluacion as E  # noqa: E402


@pytest.fixture(scope="module")
def base():
    acled = D.filtrar_horizonte(D.cargar_acled())
    datos, SEQ, pext = V.construir_dataset(acled)
    return acled, datos, SEQ, pext


def test_contrato_datos(base):
    acled, datos, SEQ, _ = base
    assert set(D.COLUMNAS_ACLED) <= set(D.cargar_acled().columns)
    assert not datos.duplicated(["depto", "anio"]).any(), "llave departamento-año repetida"
    columnas = V.BASE + V.MEMORIA + V.DINAMICA + V.VECINDAD
    assert datos[columnas].notna().all().all(), "hay predictoras faltantes"
    assert set(datos["y"].unique()) <= {0.0, 1.0}
    assert (datos["eventos"] >= C.MIN_EVENTOS).all()
    assert SEQ.shape == (len(datos), 4, len(V.SECUENCIA)) and np.isfinite(SEQ).all()


def test_contrato_rechaza_columna_faltante(tmp_path):
    bruto = pd.read_csv(C.RUTA_ACLED, nrows=50, encoding="utf-8-sig")
    ruta = tmp_path / "export_incompleto.csv"
    bruto.drop(columns=["sub_event_type"]).to_csv(ruta, index=False)
    with pytest.raises(ValueError, match="sub_event_type"):
        D.cargar_acled(ruta)


def test_fuga_temporal(base):
    """Multiplicar por 5 los eventos C del año 2024 no puede cambiar las predictoras de las filas t = 2024."""
    acled, datos, _, _ = base
    alterado = acled.copy()
    m = (alterado["anio"] == 2024) & alterado["sub_event_type"].isin(["Armed clash", "Attack"])
    alterado.loc[m, "events"] *= 5
    alterado.loc[m, "fatalities"] *= 5
    datos2, _, _ = V.construir_dataset(alterado)
    cols = V.BASE + V.MEMORIA + V.DINAMICA + V.VECINDAD
    a = datos[datos.anio == 2024].set_index("depto")[cols]
    b = datos2[datos2.anio == 2024].set_index("depto")[cols].reindex(a.index)
    np.testing.assert_allclose(a.values, b.values, rtol=0, atol=1e-12)
    # y sí cambian las de t = 2025 (que usan 2024 como información): la prueba detectaría una fuga invertida
    c = datos[datos.anio == 2025].set_index("depto")["m1_sC_t1"]
    d = datos2[datos2.anio == 2025].set_index("depto")["m1_sC_t1"].reindex(c.index)
    assert not np.allclose(c.values, d.values)


def test_escalador_sin_fuga(base):
    _, datos, _, _ = base
    for nombre, tr, te in E.particiones_origen_movil(datos):
        v = int(nombre)
        assert datos.loc[tr, "anio"].max() < v and (datos.loc[te, "anio"] == v).all()
        spec = P.Especificacion("lr", V.BASE, None)
        _, esc = P.entrenar_especificacion(datos, None, spec, tr)
        np.testing.assert_allclose(esc.mu, datos.loc[tr, V.BASE].values.mean(0))


def test_artefacto_guarda_estandarizacion(base):
    _, datos, SEQ, _ = base
    tr = (datos.anio <= 2023).values
    spec = P.Especificacion("mlp", V.BASE, M.ConfigRed(pasos=16), semillas=(3,))
    redes, esc = P.entrenar_especificacion(datos, SEQ, spec, tr)
    np.testing.assert_allclose(esc.mu, datos.loc[tr, V.BASE].values.mean(0))
    art = M.empaquetar(redes, esc, M.Calibrador("ninguna"), len(V.BASE))
    norm = art.get_layer("estandarizacion")
    np.testing.assert_allclose(np.asarray(norm.mean).ravel(), esc.mu, rtol=1e-5)
    np.testing.assert_allclose(np.sqrt(np.asarray(norm.variance).ravel()), esc.sd, rtol=1e-4)
    # el artefacto y el cálculo en memoria dan la misma probabilidad
    te = (datos.anio == 2024).values
    p_mem = P.predecir_especificacion(redes, esc, datos, SEQ, spec, te)
    p_art = np.asarray(art.predict(datos.loc[te, V.BASE].values.astype("float32"), verbose=0)["P_C"]).ravel()
    np.testing.assert_allclose(p_art, p_mem, atol=1e-5)


def test_persistencia_calibrada_sin_fuga(base):
    _, datos, _, _ = base
    tr, te = (datos.anio < 2025).values, (datos.anio == 2025).values
    p1 = P.persistencia_calibrada(datos, tr, te)
    alterado = datos.copy()
    alterado.loc[te, "y"] = 1 - alterado.loc[te, "y"]              # cambiar las etiquetas del año predicho
    p2 = P.persistencia_calibrada(alterado, tr, te)
    np.testing.assert_allclose(p1, p2)
    assert ((p1 > 0) & (p1 < 1)).all()


def test_particiones_espacio_temporales(base):
    _, datos, _, _ = base
    n = 0
    for nombre, tr, te in E.particiones_espacio_temporales(datos):
        v = int(nombre.split("·")[-1])
        assert (datos.loc[te, "anio"] == v).all() and datos.loc[tr, "anio"].max() < v
        assert not set(datos.loc[tr, "depto"]) & set(datos.loc[te, "depto"])
        n += int(te.sum())
    assert n == int(datos.anio.isin(C.ANIOS_SELECCION).sum())


def test_regla_retencion_variante_determinista():
    ref = np.array([0.30, 0.31, 0.29, 0.32, 0.33])                 # 5 semillas de la referencia
    r = E.regla_retencion([0.305], ref, [0.95], [0.95] * 5)
    assert abs(r["ΔLog-loss"] - (0.305 - ref.mean())) < 1e-12
    assert abs(r["EE"] - ref.std(ddof=1) / np.sqrt(5)) < 1e-12


def test_determinismo(base):
    _, datos, SEQ, _ = base
    tr, te = (datos.anio <= 2023).values, (datos.anio == 2024).values
    spec = P.Especificacion("mlp", V.BASE + ["m1_sC_t1"], M.ConfigRed(pasos=120), semillas=(7,))
    p1 = P.predecir_especificacion(*P.entrenar_especificacion(datos, SEQ, spec, tr), datos, SEQ, spec, te)
    p2 = P.predecir_especificacion(*P.entrenar_especificacion(datos, SEQ, spec, tr), datos, SEQ, spec, te)
    np.testing.assert_allclose(p1, p2, atol=1e-6)


def test_reproduce_u2(base):
    _, datos, _, _ = base
    ref = RAIZ / "referencias_u2" / "dataset_modelado_departamento_anio.csv"
    if not ref.exists():
        pytest.skip("sin el dataset de referencia de la U2")
    u2 = pd.read_csv(ref)
    m = datos.merge(u2, on=["depto", "anio"], suffixes=("", "_u2"))
    assert len(m) == len(u2) == len(datos) == 171
    for f in V.BASE:
        np.testing.assert_allclose(m[f], m[f + "_u2"], atol=1e-9)
    assert (m["y"] == m["y_u2"]).all()
