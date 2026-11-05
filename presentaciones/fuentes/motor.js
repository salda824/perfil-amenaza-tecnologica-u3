/* =====================================================================================================================
   Motor de diapositivas (sin dependencias). Teclas: → / Espacio avanzar · ← retroceder · O vista general · F pantalla
   completa · T tema claro/oscuro · C cronómetro · S ocultar orador y pie · H ayuda · número + Enter ir a diapositiva.
   Cada <section class="slide"> declara data-titulo, data-orador ("laura" | "juan" | "ambos") y data-seg (segundos meta).
   Al entrar a una diapositiva se emite el evento "diapositiva" en document con { detail: { slide, indice } }.
   ===================================================================================================================== */
(function () {
  "use strict";
  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => Array.from(r.querySelectorAll(s));
  const ORADORES = {
    laura: { nombre: "Laura Díaz", color: "var(--laura)" },
    juan: { nombre: "Juan Diego Saldarriaga", color: "var(--juan)" },
    ambos: { nombre: "Laura y Juan Diego", color: "linear-gradient(90deg, var(--laura) 50%, var(--juan) 50%)" },
  };
  const guardar = (k, v) => { try { localStorage.setItem(k, v); } catch (e) { /* almacenamiento no disponible */ } };
  const leer = (k) => { try { return localStorage.getItem(k); } catch (e) { return null; } };

  const escenario = $("#escenario");
  const slides = $$(".slide", escenario);
  let actual = 0, inicioCrono = null, buffer = "", tBuffer = null;

  // ------------------------------------------------------------------------------------------ HUD
  escenario.insertAdjacentHTML("beforeend", `
    <div id="orador"><i></i><span></span></div>
    <div id="cronometro"><span>⏱ <b id="cr-t">00:00</b></span><span>meta <b id="cr-m">00:00</b></span><span id="cr-d"></span></div>
    <div id="hud-pie"><span class="marca"><i></i>${escenario.dataset.marca || "Perfil de amenaza tecnológica"}</span>
      <span class="der"><span id="hud-n"></span><button id="b-ant" title="Anterior (←)">‹</button><button id="b-sig" title="Siguiente (→)">›</button>
      <button id="b-vista" title="Vista general (O)">▦</button></span></div>
    <div id="progreso"></div>
    <div id="vista-general"><h2>Vista general</h2><div class="rejilla"></div></div>
    <div id="ayuda"><h2>Atajos</h2><table>
      <tr><td><kbd>→</kbd> <kbd>Espacio</kbd></td><td>Avanzar (muestra el siguiente paso o la siguiente diapositiva)</td></tr>
      <tr><td><kbd>←</kbd></td><td>Retroceder</td></tr>
      <tr><td><kbd>O</kbd> / <kbd>Esc</kbd></td><td>Vista general de todas las diapositivas</td></tr>
      <tr><td><kbd>F</kbd></td><td>Pantalla completa (recomendado para grabar el video)</td></tr>
      <tr><td><kbd>C</kbd></td><td>Cronómetro de ensayo con la meta de tiempo de cada diapositiva · <kbd>R</kbd> reinicia</td></tr>
      <tr><td><kbd>S</kbd></td><td>Ocultar o mostrar el orador y el pie</td></tr>
      <tr><td><kbd>T</kbd></td><td>Tema claro (para proyectores con poco contraste) u oscuro</td></tr>
      <tr><td><kbd>3</kbd><kbd>Enter</kbd></td><td>Ir a la diapositiva 3</td></tr>
      <tr><td><kbd>H</kbd></td><td>Esta ayuda</td></tr></table></div>
    <div id="pista">→ avanzar · O vista general · H ayuda</div>`);
  document.body.insertAdjacentHTML("beforeend", `<div id="tooltip"></div>`);
  setTimeout(() => { const p = $("#pista"); if (p) p.style.opacity = 0; }, 4500);

  // metas de tiempo acumuladas
  const metas = []; let acum = 0;
  slides.forEach((s) => { acum += +(s.dataset.seg || 60); metas.push(acum); });
  const fmt = (seg) => `${String(Math.floor(seg / 60)).padStart(2, "0")}:${String(Math.floor(seg % 60)).padStart(2, "0")}`;

  // vista general
  const rej = $("#vista-general .rejilla");
  slides.forEach((s, i) => {
    const o = ORADORES[s.dataset.orador] || ORADORES.ambos;
    const b = document.createElement("button");
    b.innerHTML = `<span class="n"><i style="background:${o.color}"></i>${String(i + 1).padStart(2, "0")} · ${fmt(+(s.dataset.seg || 60))}</span>${s.dataset.titulo || ""}`;
    b.addEventListener("click", () => { cerrarCapas(); ir(i); });
    rej.appendChild(b);
  });

  // ------------------------------------------------------------------------------------------ escala
  function escalar() {
    const k = Math.min(window.innerWidth / 1920, window.innerHeight / 1080);
    escenario.style.transform = `translate(-50%, -50%) scale(${k})`;
    escenario.dataset.escala = k;
  }
  window.addEventListener("resize", escalar); escalar();

  // ------------------------------------------------------------------------------------------ pasos (fragmentos)
  const pasos = (s) => $$(".paso", s).sort((a, b) => (+a.dataset.paso || 0) - (+b.dataset.paso || 0));
  function mostrarPasos(s, todos) { pasos(s).forEach((p) => p.classList.toggle("visible", !!todos)); }

  function ir(i, conPasos = false) {
    i = Math.max(0, Math.min(slides.length - 1, i));
    slides[actual].classList.remove("activa");
    actual = i;
    const s = slides[actual];
    $$(".anim", s).forEach((el, k) => el.style.setProperty("--i", el.dataset.i || k));
    mostrarPasos(s, conPasos);
    s.classList.add("activa");
    const o = ORADORES[s.dataset.orador] || ORADORES.ambos;
    $("#orador i").style.background = o.color; $("#orador span").textContent = o.nombre;
    $("#hud-n").textContent = `${String(actual + 1).padStart(2, "0")} / ${String(slides.length).padStart(2, "0")}`;
    $("#progreso").style.width = `${((actual + 1) / slides.length) * 100}%`;
    $$("#vista-general button").forEach((b, k) => b.classList.toggle("actual", k === actual));
    if (history.replaceState) history.replaceState(null, "", `#${actual + 1}`);
    document.dispatchEvent(new CustomEvent("diapositiva", { detail: { slide: s, indice: actual } }));
  }
  function avanzar() {
    const oc = pasos(slides[actual]).find((p) => !p.classList.contains("visible"));
    if (oc) { oc.classList.add("visible"); document.dispatchEvent(new CustomEvent("paso", { detail: { slide: slides[actual], paso: oc } })); }
    else if (actual < slides.length - 1) ir(actual + 1);
  }
  function retroceder() {
    const vis = pasos(slides[actual]).filter((p) => p.classList.contains("visible"));
    if (vis.length) vis[vis.length - 1].classList.remove("visible");
    else if (actual > 0) ir(actual - 1, true);
  }
  window.Presentacion = { ir, avanzar, retroceder, get actual() { return actual; } };

  // ------------------------------------------------------------------------------------------ capas y utilidades
  const capa = (id) => $("#" + id);
  function cerrarCapas() { ["vista-general", "ayuda"].forEach((id) => capa(id).classList.remove("visible")); }
  function alternar(id) { const v = capa(id).classList.contains("visible"); cerrarCapas(); if (!v) capa(id).classList.add("visible"); }
  function tema(t) { document.documentElement.dataset.theme = t; guardar("pa-tema", t); document.dispatchEvent(new CustomEvent("tema")); }
  function pantallaCompleta() {
    if (!document.fullscreenElement) document.documentElement.requestFullscreen && document.documentElement.requestFullscreen();
    else document.exitFullscreen && document.exitFullscreen();
  }
  const t0 = leer("pa-tema"); if (t0 === "claro") tema("claro");

  setInterval(() => {
    if (!inicioCrono) return;
    const t = (Date.now() - inicioCrono) / 1000, m = metas[actual];
    $("#cr-t").textContent = fmt(t); $("#cr-m").textContent = fmt(m);
    const d = m - t, el = $("#cr-d");
    el.textContent = (d >= 0 ? "−" : "+") + fmt(Math.abs(d)); el.className = d >= 0 ? "adelantado" : "atrasado";
  }, 250);

  // ------------------------------------------------------------------------------------------ eventos
  $("#b-sig").addEventListener("click", avanzar);
  $("#b-ant").addEventListener("click", retroceder);
  $("#b-vista").addEventListener("click", () => alternar("vista-general"));
  document.addEventListener("keydown", (e) => {
    const t = e.target;
    if (t && (t.tagName === "INPUT" || t.tagName === "SELECT" || t.tagName === "TEXTAREA") && e.key !== "Escape") return;
    if (e.ctrlKey || e.metaKey || e.altKey) return;
    const k = e.key;
    if (/^[0-9]$/.test(k)) { buffer += k; clearTimeout(tBuffer); tBuffer = setTimeout(() => (buffer = ""), 1600); return; }
    if (k === "Enter" && buffer) { ir(+buffer - 1); buffer = ""; e.preventDefault(); return; }
    switch (k) {
      case "ArrowRight": case "PageDown": case " ": case "Enter": e.preventDefault(); cerrarCapas(); avanzar(); break;
      case "ArrowLeft": case "PageUp": case "Backspace": e.preventDefault(); cerrarCapas(); retroceder(); break;
      case "Home": ir(0); break;
      case "End": ir(slides.length - 1); break;
      case "o": case "O": alternar("vista-general"); break;
      case "Escape": capa("vista-general").classList.contains("visible") || capa("ayuda").classList.contains("visible") ? cerrarCapas() : alternar("vista-general"); break;
      case "h": case "H": case "?": alternar("ayuda"); break;
      case "f": case "F": pantallaCompleta(); break;
      case "t": case "T": tema(document.documentElement.dataset.theme === "claro" ? "oscuro" : "claro"); break;
      case "s": case "S": $("#orador").classList.toggle("oculto"); $("#hud-pie").classList.toggle("oculto"); break;
      case "c": case "C": $("#cronometro").classList.toggle("visible"); if (!inicioCrono) inicioCrono = Date.now(); break;
      case "r": case "R": inicioCrono = Date.now(); break;
    }
  });
  let x0 = null;
  escenario.addEventListener("touchstart", (e) => { x0 = e.touches[0].clientX; }, { passive: true });
  escenario.addEventListener("touchend", (e) => {
    if (x0 === null) return; const dx = e.changedTouches[0].clientX - x0; x0 = null;
    if (Math.abs(dx) > 60) (dx < 0 ? avanzar : retroceder)();
  });

  document.addEventListener("DOMContentLoaded", () => {
    const h = parseInt((location.hash || "").replace("#", ""), 10);
    ir(Number.isFinite(h) ? h - 1 : 0);
  });
})();
