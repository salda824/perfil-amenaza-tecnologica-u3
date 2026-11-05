"""Rutas, constantes del proyecto y lectura de la configuración (configuracion.json)."""
from __future__ import annotations

import json
import os
from pathlib import Path

# Determinismo numérico: deben fijarse antes de importar TensorFlow en cualquier módulo.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("KERAS_BACKEND", "tensorflow")

RAIZ = Path(__file__).resolve().parents[2]          # carpeta Proyecto_Unidad_3/
DIR_DATOS = RAIZ / "datos"
DIR_EXT = DIR_DATOS / "externos"
DIR_SALIDA = RAIZ / "salidas_u3"
DIR_FIG = DIR_SALIDA / "figuras"
DIR_MODELOS = DIR_SALIDA / "modelos"
RUTA_ACLED = DIR_DATOS / "acled-data-export.csv"
RUTA_CONFIG = RAIZ / "configuracion.json"

SEED = 42
ANIO_INICIO = 2020                 # horizonte de la Etapa 1
MIN_EVENTOS = 10                   # D3 (U2): etiqueta confiable con >= 10 eventos
ANIOS_SELECCION = (2023, 2024, 2025)   # validación de origen móvil para seleccionar (U3)
ANIO_PRUEBA = 2026                 # se reporta una sola vez, al final
PASOS_REF = 560                    # D10 (U2): presupuesto de referencia (70 épocas x 8 lotes)

# Mapeo subtipo -> perfil de la Etapa 1 (idéntico; los subtipos ausentes van a D por defecto)
MAPA_PERFIL_U1 = {
    "Remote explosive/landmine/IED": "A", "Grenade": "A", "Suicide bomb": "A", "Chemical weapon": "A",
    "Air/drone strike": "B",
    "Armed clash": "C", "Attack": "C", "Shooting": "C", "Abduction/forced disappearance": "C",
    "Looting/property destruction": "C", "Disrupted weapons use": "C",
    "Mob violence": "D", "Peaceful protest": "D", "Excessive force against protesters": "D", "Arrests": "D",
    "Government regains territory": "D", "Non-state actor overtakes territory": "D",
}
# M8 (U2): mapeo alternativo para el control de sensibilidad de la etiqueta
MAPA_ALTERNATIVO = {**MAPA_PERFIL_U1, "Shelling/artillery/missile attack": "B", "Sexual violence": "C"}

NOMBRE = {
    "amazonas": "Amazonas", "antioquia": "Antioquia", "arauca": "Arauca", "atlantico": "Atlántico",
    "bogota": "Bogotá D.C.", "bolivar": "Bolívar", "boyaca": "Boyacá", "caldas": "Caldas", "caqueta": "Caquetá",
    "casanare": "Casanare", "cauca": "Cauca", "cesar": "Cesar", "choco": "Chocó", "cordoba": "Córdoba",
    "cundinamarca": "Cundinamarca", "guainia": "Guainía", "guaviare": "Guaviare", "huila": "Huila",
    "la guajira": "La Guajira", "magdalena": "Magdalena", "meta": "Meta", "narino": "Nariño",
    "norte de santander": "Norte de Santander", "putumayo": "Putumayo", "quindio": "Quindío",
    "risaralda": "Risaralda", "san andres": "San Andrés", "santander": "Santander", "sucre": "Sucre",
    "tolima": "Tolima", "valle del cauca": "Valle del Cauca", "vaupes": "Vaupés", "vichada": "Vichada"}

# DANE (2019). Pobreza multidimensional por departamentos 2018 (ECV 2018), incidencia total (%). Igual que la U2.
IPM_2018 = {
    "guainia": 65.0, "vaupes": 59.4, "vichada": 55.0, "la guajira": 51.4, "choco": 45.1, "sucre": 39.7,
    "magdalena": 38.6, "cordoba": 36.7, "amazonas": 34.9, "narino": 33.5, "guaviare": 33.5, "cesar": 33.2,
    "bolivar": 32.4, "arauca": 31.8, "norte de santander": 31.5, "caqueta": 28.7, "cauca": 28.7,
    "putumayo": 25.1, "tolima": 23.5, "atlantico": 20.1, "huila": 19.2, "casanare": 19.1, "antioquia": 17.1,
    "boyaca": 16.6, "quindio": 16.2, "meta": 15.6, "caldas": 15.3, "valle del cauca": 13.6, "santander": 12.9,
    "risaralda": 12.5, "cundinamarca": 11.5, "san andres": 8.9, "bogota": 4.4}
BOGOTA = (4.5981, -74.0760)   # Plaza de Bolívar

# Contigüidad departamental (frontera terrestre compartida). Construida a partir del mapa político de Colombia
# (IGAC); se simetriza en datos.vecindad(). San Andrés no tiene vecinos terrestres.
VECINOS = {
    "amazonas": ["caqueta", "putumayo", "vaupes"],
    "antioquia": ["cordoba", "bolivar", "santander", "boyaca", "caldas", "risaralda", "choco"],
    "arauca": ["boyaca", "casanare", "vichada"],
    "atlantico": ["bolivar", "magdalena"],
    "bogota": ["cundinamarca", "meta", "huila"],
    "bolivar": ["atlantico", "magdalena", "cesar", "santander", "antioquia", "cordoba", "sucre"],
    "boyaca": ["santander", "norte de santander", "arauca", "casanare", "cundinamarca", "caldas", "antioquia"],
    "caldas": ["antioquia", "boyaca", "cundinamarca", "tolima", "risaralda"],
    "caqueta": ["meta", "guaviare", "vaupes", "amazonas", "putumayo", "cauca", "huila"],
    "casanare": ["arauca", "vichada", "meta", "boyaca"],
    "cauca": ["valle del cauca", "tolima", "huila", "caqueta", "putumayo", "narino"],
    "cesar": ["la guajira", "magdalena", "bolivar", "santander", "norte de santander"],
    "choco": ["antioquia", "risaralda", "valle del cauca"],
    "cordoba": ["antioquia", "sucre", "bolivar"],
    "cundinamarca": ["boyaca", "caldas", "tolima", "huila", "meta", "bogota"],
    "guainia": ["vichada", "guaviare", "vaupes"],
    "guaviare": ["meta", "vichada", "guainia", "vaupes", "caqueta"],
    "huila": ["tolima", "cundinamarca", "bogota", "meta", "caqueta", "cauca"],
    "la guajira": ["magdalena", "cesar"],
    "magdalena": ["atlantico", "bolivar", "cesar", "la guajira"],
    "meta": ["cundinamarca", "bogota", "casanare", "vichada", "guaviare", "caqueta", "huila"],
    "narino": ["cauca", "putumayo"],
    "norte de santander": ["cesar", "santander", "boyaca"],
    "putumayo": ["narino", "cauca", "caqueta", "amazonas"],
    "quindio": ["risaralda", "tolima", "valle del cauca"],
    "risaralda": ["choco", "antioquia", "caldas", "tolima", "quindio", "valle del cauca"],
    "san andres": [],
    "santander": ["norte de santander", "cesar", "bolivar", "antioquia", "boyaca"],
    "sucre": ["cordoba", "bolivar"],
    "tolima": ["caldas", "cundinamarca", "huila", "cauca", "valle del cauca", "quindio", "risaralda"],
    "valle del cauca": ["choco", "risaralda", "quindio", "tolima", "cauca"],
    "vaupes": ["guaviare", "guainia", "amazonas", "caqueta"],
    "vichada": ["arauca", "casanare", "meta", "guaviare", "guainia"],
}


def cargar_config(ruta: Path | str = RUTA_CONFIG) -> dict:
    """Lee configuracion.json (hiperparámetros del modelo final y del protocolo)."""
    with open(ruta, encoding="utf-8") as f:
        return json.load(f)


def asegurar_carpetas() -> None:
    for c in (DIR_SALIDA, DIR_FIG, DIR_MODELOS, DIR_EXT):
        c.mkdir(parents=True, exist_ok=True)
