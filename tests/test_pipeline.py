"""
Pruebas automáticas del pipeline (pytest). Se ejecutan desde la raíz del proyecto:

    python -m pytest -q tests

1. Contrato de datos: el export cumple el esquema y el dataset de modelado no tiene faltantes ni llaves repetidas.
2. Fuga temporal: alterar los eventos del año t no cambia ninguna predictora de la fila (d, t).
3. Escalador sin fuga: en la validación de origen móvil se ajusta solo con años anteriores al validado.
4. Determinismo: dos entrenamientos con la misma semilla producen las mismas probabilidades.
5. Reproducción de la U2: el dataset reconstruye las 171 filas y las 7 predictoras de la Unidad 2.
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


def test_fuga_temporal(base):
    """Duplicar los eventos C del año 2024 no puede cambiar las predictoras de las filas t = 2024."""
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
