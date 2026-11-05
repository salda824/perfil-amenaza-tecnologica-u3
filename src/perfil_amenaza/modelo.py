"""
Redes neuronales del proyecto (Keras 3, backend TensorFlow), bucle de entrenamiento explícito y calibración.

El bucle es el mismo algoritmo verificado en la U2 (§8.2: diferencia máx. 2e-6 frente a model.fit), generalizado
para: objetivo duro o blando (I5), pesos por muestra, salida multitarea de composición A/B/C/D (I9) y entradas
secuenciales o de identificador (comparativos GRU, CNN 1D y embedding).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict, replace, field

import numpy as np

from . import config as C   # noqa: F401  (fija variables de entorno antes de TensorFlow)
import tensorflow as tf
import keras
from keras import layers
from scipy.optimize import minimize

tf.get_logger().setLevel("ERROR")


@dataclass(frozen=True)
class ConfigRed:
    tipo: str = "mlp"              # mlp | multitarea | gru | cnn | embedding
    capas: tuple = (16,)           # neuronas por capa oculta densa
    activacion: str = "relu"       # relu | tanh | elu | gelu
    l2: float = 1e-3               # penalización L2 (λ)
    dropout: float = 0.0
    lr: float = 1e-3               # tasa de aprendizaje de Adam
    batch: int = 16
    pasos: int = 560               # presupuesto de pasos de gradiente (D10, U2)
    pesos_clase: bool = True       # BCE ponderada por clase (U2)
    peso_eventos: str = "ninguno"  # ninguno | raiz: pondera cada fila por sqrt(eventos en t) (I5)
    objetivo: str = "duro"         # duro: y = 1[s_C >= 0,5] | blando: y = s_C (I5)
    lambda_comp: float = 0.0       # peso de la pérdida de composición A/B/C/D (I9, solo multitarea)
    unidades_seq: int = 8          # GRU / CNN
    emb_dim: int = 2               # embedding de departamento
    n_deptos: int = 33

    def describir(self) -> str:
        return (f"{self.tipo} {self.capas} {self.activacion} L2={self.l2:g} do={self.dropout:g} lr={self.lr:g} "
                f"pasos={self.pasos} obj={self.objetivo}" + (f" λc={self.lambda_comp:g}" if self.tipo == "multitarea" else ""))


def n_parametros(modelo) -> int:
    return int(sum(np.prod(v.shape) for v in modelo.trainable_variables))


# ----------------------------------------------------------------------------------------------------------
# Constructores
# ----------------------------------------------------------------------------------------------------------
def _tronco(h, cfg, reg, prefijo="oculta"):
    init = "he_normal" if cfg.activacion in ("relu", "elu", "gelu") else "glorot_uniform"
    for i, u in enumerate(cfg.capas, 1):
        h = layers.Dense(u, activation=cfg.activacion, kernel_initializer=init, kernel_regularizer=reg,
                         name=f"{prefijo}_{i}")(h)
        if cfg.dropout > 0:
            h = layers.Dropout(cfg.dropout, name=f"dropout_{i}")(h)
    return h


def construir_red(n_x: int, cfg: ConfigRed, n_seq: tuple | None = None) -> keras.Model:
    reg = keras.regularizers.L2(cfg.l2) if cfg.l2 > 0 else None
    x = keras.Input(shape=(n_x,), name="x_t_1")
    entradas, h = [x], x
    if cfg.tipo in ("gru", "cnn"):
        s = keras.Input(shape=n_seq, name="secuencia_trimestral_t_1")
        entradas.append(s)
        if cfg.tipo == "gru":
            z = layers.GRU(cfg.unidades_seq, kernel_regularizer=reg, recurrent_regularizer=reg, name="gru")(s)
        else:
            z = layers.Conv1D(cfg.unidades_seq, 2, activation="relu", kernel_regularizer=reg, name="conv1d")(s)
            z = layers.GlobalAveragePooling1D(name="promedio_temporal")(z)
        h = layers.Concatenate(name="estatico_y_secuencia")([x, z])
    elif cfg.tipo == "embedding":
        i = keras.Input(shape=(1,), dtype="int32", name="id_departamento")
        entradas.append(i)
        e = layers.Flatten()(layers.Embedding(cfg.n_deptos, cfg.emb_dim, embeddings_regularizer=reg, name="embedding")(i))
        h = layers.Concatenate(name="x_y_embedding")([x, e])
    h = _tronco(h, cfg, reg)
    p = layers.Dense(1, activation="sigmoid", name="P_C")(h)
    if cfg.tipo == "multitarea":
        comp = layers.Dense(4, activation="softmax", name="composicion_ABCD")(h)
        return keras.Model(entradas, [p, comp], name="MLP_multitarea")
    return keras.Model(entradas, p, name=f"red_{cfg.tipo}")


# ----------------------------------------------------------------------------------------------------------
# Entrenamiento
# ----------------------------------------------------------------------------------------------------------
def _bce_np(y, p, eps=1e-7):
    y = np.asarray(y, float); p = np.clip(np.asarray(p, float), eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


class RedEntrenada:
    """Modelo entrenado + función de predicción en modo inferencia."""

    def __init__(self, modelo, cfg, historial, info):
        self.modelo, self.cfg, self.historial, self.info = modelo, cfg, historial, info
        self._pred = tf.function(lambda xs: self.modelo(xs, training=False), reduce_retracing=True)

    def _salidas(self, entradas):
        xs = [tf.convert_to_tensor(np.asarray(e, "int32" if e.dtype.kind in "iu" else "float32")) for e in entradas]
        out = self._pred(xs)
        return out if isinstance(out, (list, tuple)) else [out]

    def predecir(self, entradas) -> np.ndarray:
        return self._salidas(entradas)[0].numpy().ravel()

    def composicion(self, entradas) -> np.ndarray | None:
        out = self._salidas(entradas)
        return out[1].numpy() if len(out) > 1 else None


def entrenar(entradas, y, cfg: ConfigRed, semilla: int = C.SEED, y_duro=None, eventos=None, comp=None,
             val=None, parada_temprana: bool = False, paciencia: int = 80, diagnostico: bool = False,
             epocas: int | None = None) -> RedEntrenada:
    """
    entradas  lista de arrays [X] o [X, SEQ] o [X, ids] ya escalados.
    y         objetivo (duro 0/1 o blando s_C).
    y_duro    etiqueta 0/1 para los pesos de clase (por defecto, y >= 0,5).
    eventos   eventos en t por fila (para peso_eventos = raiz).
    comp      composición A/B/C/D en t (n, 4) para la salida multitarea.
    val       (entradas_val, y_val_duro) para registrar curvas y, si parada_temprana, detener y restaurar.
    """
    keras.utils.set_random_seed(semilla)
    rng = np.random.default_rng(semilla)
    entradas = [np.asarray(e, "int32" if np.asarray(e).dtype.kind in "iu" else "float32") for e in entradas]
    y = np.asarray(y, "float32").reshape(-1, 1)
    y_duro = (y.ravel() >= .5).astype("float32") if y_duro is None else np.asarray(y_duro, "float32")
    n, B = len(y), cfg.batch
    n_lotes = math.ceil(n / B)
    epocas = epocas or max(1, math.ceil(cfg.pasos / n_lotes))
    modelo = construir_red(entradas[0].shape[1], cfg, entradas[1].shape[1:] if cfg.tipo in ("gru", "cnn") else None)
    opt = keras.optimizers.Adam(learning_rate=cfg.lr)
    opt.build(modelo.trainable_variables)

    w = np.ones(n, "float32")
    if cfg.pesos_clase:
        n1 = float(y_duro.sum())
        w *= np.where(y_duro == 1, n / (2 * n1), n / (2 * (n - n1))).astype("float32")
    if cfg.peso_eventos == "raiz":
        ev = np.sqrt(np.asarray(eventos, "float32")); w *= (ev / ev.mean()).astype("float32")
    multitarea = cfg.tipo == "multitarea"
    comp = np.asarray(comp, "float32") if multitarea else np.zeros((n, 4), "float32")
    lam = float(cfg.lambda_comp)

    @tf.function
    def epoca(xs, yp, wp, cp):
        perdidas = tf.TensorArray(tf.float32, size=n_lotes)
        for i in tf.range(n_lotes):
            xb = [x[i * B:(i + 1) * B] for x in xs]
            yb, wb, cb = yp[i * B:(i + 1) * B], wp[i * B:(i + 1) * B], cp[i * B:(i + 1) * B]
            with tf.GradientTape() as cinta:
                out = modelo(xb, training=True)
                p = out[0] if multitarea else out
                p = tf.clip_by_value(p, 1e-7, 1 - 1e-7)
                bce = -(yb * tf.math.log(p) + (1 - yb) * tf.math.log(1 - p))[:, 0]    # admite objetivo blando
                nb = tf.cast(tf.shape(yb)[0], tf.float32)
                perdida = tf.reduce_sum(wb * bce) / nb                       # misma reducción que Keras (U2)
                if multitarea and lam > 0:
                    q = tf.clip_by_value(out[1], 1e-7, 1.0)
                    kl = tf.reduce_sum(cb * (tf.math.log(tf.clip_by_value(cb, 1e-7, 1.0)) - tf.math.log(q)), axis=1)
                    perdida += lam * tf.reduce_sum(wb * kl) / nb
                if modelo.losses:
                    perdida += tf.add_n(modelo.losses)
            grads = cinta.gradient(perdida, modelo.trainable_variables)
            opt.apply_gradients(zip(grads, modelo.trainable_variables))
            perdidas = perdidas.write(i, perdida * tf.cast(tf.shape(yb)[0], tf.float32))
        return tf.reduce_sum(perdidas.stack()) / n

    H = {"epoca": [], "perdida_obj": [], "bce_tr": [], "bce_va": []}
    red = RedEntrenada(modelo, cfg, H, {})
    mejor, mejor_ep, mejores, espera = np.inf, 0, None, 0
    for ep in range(1, epocas + 1):
        orden = rng.permutation(n)
        l = epoca([tf.constant(x[orden]) for x in entradas], tf.constant(y[orden]), tf.constant(w[orden]),
                  tf.constant(comp[orden]))
        if diagnostico or parada_temprana:
            H["epoca"].append(ep); H["perdida_obj"].append(float(l))
            if diagnostico:
                H["bce_tr"].append(_bce_np(y_duro, red.predecir(entradas)))
            if val is not None:
                H["bce_va"].append(_bce_np(val[1], red.predecir(val[0])))
                if parada_temprana:
                    if H["bce_va"][-1] < mejor - 1e-4:
                        mejor, mejor_ep, mejores, espera = H["bce_va"][-1], ep, modelo.get_weights(), 0
                    else:
                        espera += 1
                        if espera >= paciencia:
                            break
    if parada_temprana and mejores is not None:
        modelo.set_weights(mejores)
    red.info = {"epocas": ep, "mejor_epoca": mejor_ep if parada_temprana else ep, "pasos": ep * n_lotes,
                "parametros": n_parametros(modelo)}
    return red


# ----------------------------------------------------------------------------------------------------------
# Escalado, ensamble y calibración
# ----------------------------------------------------------------------------------------------------------
class Escalador:
    """Estandarización ajustada solo con las filas de entrenamiento (tabular y secuencial)."""

    def ajustar(self, X, SEQ=None):
        X = np.asarray(X, float)
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-9
        if SEQ is not None:
            S = np.asarray(SEQ, float).reshape(-1, SEQ.shape[-1])
            self.mu_s, self.sd_s = S.mean(0), S.std(0) + 1e-9
        return self

    def transformar(self, X, SEQ=None):
        out = [((np.asarray(X, float) - self.mu) / self.sd).astype("float32")]
        if SEQ is not None:
            out.append(((np.asarray(SEQ, float) - self.mu_s) / self.sd_s).astype("float32"))
        return out


def _logit(p, eps=1e-6):
    p = np.clip(np.asarray(p, float), eps, 1 - eps)
    return np.log(p / (1 - p))


class Calibrador:
    """Platt (a·logit(p) + b), temperatura (logit(p)/T) o ninguna (identidad). Se ajusta minimizando la log-loss."""

    def __init__(self, metodo="platt"):
        self.metodo, self.a, self.b = metodo, 1.0, 0.0

    def ajustar(self, p, y):
        if self.metodo == "ninguna":
            return self
        z, y = _logit(p), np.asarray(y, float)

        def nll(th):
            a, b = (th[0], th[1]) if self.metodo == "platt" else (th[0], 0.0)
            q = 1 / (1 + np.exp(-(a * z + b)))
            return _bce_np(y, q)
        x0 = [1.0, 0.0] if self.metodo == "platt" else [1.0]
        r = minimize(nll, x0, method="Nelder-Mead", options={"xatol": 1e-6, "fatol": 1e-9, "maxiter": 2000})
        self.a = float(r.x[0]); self.b = float(r.x[1]) if self.metodo == "platt" else 0.0
        return self

    def aplicar(self, p):
        return 1 / (1 + np.exp(-(self.a * _logit(p) + self.b)))

    @property
    def temperatura(self):
        return 1 / self.a if self.a else np.inf


def calibrar_adelante(particion, p, y, metodo="platt") -> np.ndarray:
    """
    Calibración con encadenamiento hacia adelante: las predicciones del pliegue v se calibran con un calibrador
    ajustado solo con las de los pliegues anteriores. El primer pliegue queda sin calibrar.
    """
    particion, p, y = np.asarray(particion).astype(str), np.asarray(p, float), np.asarray(y, float)
    q = p.copy()
    for v in sorted(np.unique(particion))[1:]:
        m, prev = particion == v, particion < v
        q[m] = Calibrador(metodo).ajustar(p[prev], y[prev]).aplicar(p[m])
    return q


def predecir_ensamble(redes, entradas) -> np.ndarray:
    return np.mean([r.predecir(entradas) for r in redes], axis=0)


# ----------------------------------------------------------------------------------------------------------
# Artefacto único de despliegue: estandarización + ensamble + calibración en un solo .keras
# ----------------------------------------------------------------------------------------------------------
@keras.saving.register_keras_serializable(package="perfil_amenaza")
class CalibracionPlatt(layers.Layer):
    """p' = σ(a·logit(p) + b). Parámetros fijos (no entrenables) estimados fuera de muestra."""

    def __init__(self, a=1.0, b=0.0, **kw):
        super().__init__(**kw)
        self.a, self.b = float(a), float(b)

    def call(self, p):
        p = keras.ops.clip(p, 1e-6, 1 - 1e-6)
        return keras.ops.sigmoid(self.a * keras.ops.log(p / (1 - p)) + self.b)

    def get_config(self):
        return {**super().get_config(), "a": self.a, "b": self.b}


def empaquetar(redes, escalador: Escalador, calibrador: Calibrador, n_x: int) -> keras.Model:
    """Un solo keras.Model: x crudo (t-1) -> estandarización -> K redes -> promedio -> calibración -> P(C)."""
    x = keras.Input(shape=(n_x,), name="x_crudo_t_1")
    norm = layers.Normalization(mean=escalador.mu.astype("float32"), variance=(escalador.sd ** 2).astype("float32"),
                                name="estandarizacion")
    z = norm(x)
    ps, comps = [], []
    for k, r in enumerate(redes):
        m = r.modelo
        m.name = f"miembro_{k + 1}"
        out = m(z)
        if isinstance(out, (list, tuple)):
            ps.append(out[0]); comps.append(out[1])
        else:
            ps.append(out)
    p = layers.Average(name="promedio_ensamble")(ps) if len(ps) > 1 else ps[0]
    p = CalibracionPlatt(calibrador.a, calibrador.b, name="calibracion")(p)
    salidas = {"P_C": p}
    if comps:
        salidas["composicion_ABCD"] = layers.Average(name="composicion_media")(comps) if len(comps) > 1 else comps[0]
    return keras.Model(x, salidas, name="perfil_amenaza_u3")
