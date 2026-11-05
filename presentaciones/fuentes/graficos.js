/* =====================================================================================================================
   Componentes interactivos de las presentaciones. Todas las cifras salen de window.DATOS (lo incrusta
   construir_presentaciones.py a partir de salidas_u3/, referencias_u2/ y el export de ACLED).
   Cada componente se declara en el HTML con una clase (g-mapa, g-heatmap, ...) y se anima al entrar a su diapositiva.
   ===================================================================================================================== */
(function () {
  "use strict";
  const D = window.DATOS;
  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => Array.from(r.querySelectorAll(s));
  const NS = "http://www.w3.org/2000/svg";
  const MENOS = "−";

  // ------------------------------------------------------------------------------------------ formato (coma decimal)
  const f = (x, n = 3) => (x === null || x === undefined || Number.isNaN(x)) ? "–"
    : (x < 0 ? MENOS : "") + Math.abs(x).toFixed(n).replace(".", ",");
  const fs = (x, n = 3) => (x > 0 ? "+" : "") + f(x, n);
  const miles = (x) => Math.round(x).toString().replace(/\B(?=(\d{3})+(?!\d))/g, ".");
  const pct = (x, n = 0) => f(x * 100, n) + " %";
  const ic = (a) => a ? `[${f(a[0])}; ${f(a[1])}]` : "–";
  const FORMATOS = { "0": (x) => f(x, 0), "1": (x) => f(x, 1), "2": (x) => f(x, 2), "3": (x) => f(x, 3),
    s3: (x) => fs(x, 3), s2: (x) => fs(x, 2), pct0: (x) => pct(x, 0), pct1: (x) => pct(x, 1), miles, ic, txt: (x) => x };

  function resolver(ruta) {
    let o = D;
    for (const parte of ruta.split("|")) {
      if (o === undefined || o === null) return undefined;
      if (Array.isArray(o) && /^[a-z_]+=/.test(parte)) {
        const conds = parte.split("&").map((c) => c.split("="));
        o = o.find((e) => e && conds.every(([k, v]) => String(e[k]) === v));
      } else if (Array.isArray(o) && !/^\d+$/.test(parte)) {
        o = o.find((e) => e && ["modelo", "nombre", "paso", "var", "id", "conj", "comp"].some((k) => e[k] === parte));
      } else o = o[parte];
    }
    return o;
  }
  function rellenar(raiz = document) {
    $$("[data-v]", raiz).forEach((el) => {
      const v = resolver(el.dataset.v);
      if (v === undefined) { el.textContent = "¿?"; el.style.color = "red"; console.warn("dato no encontrado", el.dataset.v); return; }
      el.textContent = (FORMATOS[el.dataset.f || "3"] || FORMATOS["3"])(v);
    });
  }

  // ------------------------------------------------------------------------------------------ utilidades
  function el(tag, attrs = {}, padre) {
    const e = document.createElementNS(NS, tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (k === "texto") e.textContent = v; else if (k === "estilo") e.setAttribute("style", v); else e.setAttribute(k, v);
    }
    if (padre) padre.appendChild(e);
    return e;
  }
  const svg = (w, h, padre, clase = "") => el("svg", { viewBox: `0 0 ${w} ${h}`, width: w, height: h, class: clase }, padre);
  const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
  function mezclar(a, b, t) {
    const h = (c) => [1, 3, 5].map((i) => parseInt(c.slice(i, i + 2), 16));
    const [x, y] = [h(a), h(b)];
    return "#" + x.map((v, i) => Math.round(v + (y[i] - v) * t).toString(16).padStart(2, "0")).join("");
  }
  const tip = $("#tooltip");
  function mostrarTip(e, html) {
    tip.innerHTML = html; tip.classList.add("visible");
    const r = tip.getBoundingClientRect();
    let x = e.clientX + 16, y = e.clientY + 16;
    if (x + r.width > innerWidth - 8) x = e.clientX - r.width - 16;
    if (y + r.height > innerHeight - 8) y = e.clientY - r.height - 16;
    tip.style.left = x + "px"; tip.style.top = y + "px";
  }
  const ocultarTip = () => tip.classList.remove("visible");
  const filaTip = (a, b) => `<div class="t-fila"><span>${a}</span><span>${b}</span></div>`;

  const alEntrar = [];
  function enDiapositiva(nodo, fn) { alEntrar.push({ slide: nodo.closest(".slide"), fn }); }
  document.addEventListener("diapositiva", (e) => alEntrar.filter((a) => a.slide === e.detail.slide).forEach((a) => a.fn()));
  const alTema = [];
  document.addEventListener("tema", () => alTema.forEach((fn) => fn()));

  const COL_PERFIL = { A: "var(--A)", B: "var(--B)", C: "var(--C)", D: "var(--D)" };
  const NOMBRE_PERFIL = { A: "A · explosivos", B: "B · drones", C: "C · convencional", D: "D · mixto" };

  // ------------------------------------------------------------------------------------------ mapa de Colombia
  function mapa(cont) {
    const M = D.mapa, [, , W, H] = M.viewBox.split(" ").map(Number);
    const alto = +(cont.dataset.alto || 760), ancho = Math.round(alto * W / H);
    let modo = cont.dataset.modo || "2027", anio = 2026, sel = cont.dataset.sel || null;
    const etiquetas = (cont.dataset.etiquetas || "").split(",").filter(Boolean);
    cont.style.display = "flex"; cont.style.gap = "30px"; cont.style.alignItems = "stretch";
    const izq = document.createElement("div"); izq.style.cssText = "display:flex;flex-direction:column;gap:14px;flex:none";
    cont.appendChild(izq);
    if (cont.dataset.controles) {
      const bot = document.createElement("div"); bot.className = "botonera";
      bot.innerHTML = `<button data-m="2027">Proyección 2027</button><button data-m="2026">Perfil observado 2026</button><button data-m="anio">Histórico</button>`;
      izq.appendChild(bot);
      const anios = document.createElement("div"); anios.className = "botonera"; anios.style.display = "none";
      anios.innerHTML = [2021, 2022, 2023, 2024, 2025, 2026].map((a) => `<button data-a="${a}">${a}</button>`).join("");
      izq.appendChild(anios);
      bot.addEventListener("click", (e) => { const m = e.target.dataset.m; if (!m) return; modo = m; pintar(); });
      anios.addEventListener("click", (e) => { const a = e.target.dataset.a; if (!a) return; anio = +a; pintar(); });
      cont._bot = bot; cont._anios = anios;
    }
    const s = svg(ancho, alto, izq, "mapa-svg"); s.setAttribute("viewBox", M.viewBox);
    const defs = el("defs", {}, s);
    const pat = el("pattern", { id: "rayado-" + (cont.id || Math.random().toString(36).slice(2)), width: 9, height: 9, patternUnits: "userSpaceOnUse", patternTransform: "rotate(45)" }, defs);
    el("rect", { width: 9, height: 9, estilo: "fill: var(--panel2)" }, pat);
    el("rect", { width: 4.5, height: 9, estilo: "fill: var(--C); opacity:.85" }, pat);
    const idRayado = pat.getAttribute("id");
    const ins = M.inset; el("rect", { x: ins.x, y: ins.y, width: ins.w, height: ins.h, rx: 8, estilo: "fill:none;stroke:var(--linea2);stroke-dasharray:4 5" }, s);
    el("text", { x: ins.x + 8, y: ins.y + ins.h + 18, "font-size": 15, class: "mono", texto: "San Andrés" }, s);
    const capaD = el("g", {}, s), capaA = el("g", {}, s), capaT = el("g", {}, s);
    const paths = {};
    for (const [k, g] of Object.entries(M.deptos)) {
      const p = el("path", { d: g.d, class: "dpto" }, capaD); paths[k] = p;
      p.addEventListener("mousemove", (e) => mostrarTip(e, tipDepto(k)));
      p.addEventListener("mouseleave", ocultarTip);
      p.addEventListener("click", () => { sel = k; pintar(); });
    }
    for (const [k, d] of Object.entries(D.deptos)) {
      if (!d.alerta) continue;
      const g = M.deptos[k];
      el("circle", { cx: g.cx, cy: g.cy, r: 16, class: "pulso", estilo: "fill:none;stroke:var(--A);stroke-width:4" }, capaA);
      el("circle", { cx: g.cx, cy: g.cy, r: 8, estilo: "fill:var(--A);stroke:var(--bg);stroke-width:2" }, capaA);
    }
    const leyenda = document.createElement("div"); leyenda.className = "leyenda"; if (cont.dataset.leyenda !== "0") izq.appendChild(leyenda);
    let det = null;
    if (cont.dataset.detalle) { det = document.createElement("div"); det.className = "tarjeta"; det.style.cssText = "flex:1;min-width:0;display:flex;flex-direction:column;gap:16px"; cont.appendChild(det); }

    function fila(k) { return (D.deptos[k].hist || []).find((h) => h[0] === anio); }
    function relleno(k) {
      const d = D.deptos[k];
      if (modo === "2027") return d.cat === "C consolidado" ? "var(--C)" : d.cat === "Transición" ? `url(#${idRayado})` : "var(--D)";
      if (modo === "2026") return !d.conf26 ? "var(--panel2)" : d.perf26 === "C" ? "var(--C)" : "var(--D)";
      const h = fila(k); return !h ? "var(--panel2)" : h[2] === "C" ? "var(--C)" : "var(--D)";
    }
    function tipDepto(k) {
      const d = D.deptos[k];
      let h = `<b>${d.n}</b>`;
      if (modo === "2027") {
        h += filaTip("P(C) 2027", f(d.p27, 2)) + filaTip("Categoría", d.cat) + filaTip("A+B 2025–26", pct(d.sab, 1));
        if (d.alerta) h += `<div class="t-fila"><span style="color:var(--A)">⚠ alerta A/B</span><span>C-UAS · AEI</span></div>`;
        if (!d.conf26) h += `<div class="t-fila"><span>* menos de 10 eventos en 2026</span><span></span></div>`;
      } else if (modo === "2026") {
        h += filaTip("Perfil 2026 (33 sem.)", d.conf26 ? d.perf26 : "sin etiqueta") + filaTip("s_C 2026", f(d.sc26, 2)) + filaTip("Eventos", miles(d.ev26));
      } else {
        const r = fila(k);
        h += r ? filaTip("Año", r[0]) + filaTip("Perfil", r[2] + (r[3] ? " (transición)" : "")) + filaTip("s_C", f(r[1], 2)) +
          filaTip("Eventos", miles(r[4])) + (r[5] !== null ? filaTip("P(C) del modelo", f(r[5], 2)) : "")
          : filaTip(String(anio), "menos de 10 eventos");
      }
      return h;
    }
    function pintarDetalle() {
      if (!det || !sel) return;
      const d = D.deptos[sel];
      const color = d.cat === "C consolidado" ? "var(--Cclaro)" : d.cat === "Transición" ? "var(--acento2)" : "var(--tinta2)";
      det.innerHTML = `<div class="fuente">Departamento seleccionado</div>
        <h3 style="font-size:46px;margin:0">${d.n}</h3>
        <div class="stat"><span class="valor" style="color:${color}">${f(d.p27, 2)}</span><span class="etq">P(C) 2027 · <b style="color:${color}">${d.cat}</b></span></div>
        <p class="chico">${d.enf}${d.conf26 ? "" : " · <i>pocos eventos en 2026: revisión experta</i>"}</p>
        ${d.alerta ? `<div class="chip" style="border-color:var(--A);color:var(--A)">⚠ Alerta A/B · ${pct(d.sab, 1)} de eventos A+B en 2025–2026</div>` : ""}
        <div class="fuente" style="margin-top:6px">Proporción de eventos C (s_C) y P(C) del modelo por año</div><div class="spark"></div>`;
      chispa($(".spark", det), d.hist);
    }
    function pintar() {
      for (const [k, p] of Object.entries(paths)) {
        p.style.fill = relleno(k);
        p.classList.toggle("sel", k === sel);
        const h = modo === "anio" ? fila(k) : null;
        p.style.strokeDasharray = (modo === "2027" && !D.deptos[k].conf26) ? "5 4" : "";
        p.style.stroke = h && h[3] ? "var(--tinta)" : ""; p.style.strokeWidth = h && h[3] ? "3" : "";
      }
      capaA.style.display = modo === "2027" ? "" : "none";
      capaT.innerHTML = "";
      etiquetas.forEach((k) => {
        const g = M.deptos[k]; if (!g) return;
        const t = el("text", { x: g.cx, y: g.cy - 22, "text-anchor": "middle", "font-size": 22, estilo: "fill:var(--tinta);font-weight:600;paint-order:stroke;stroke:var(--bg);stroke-width:5px" }, capaT);
        t.textContent = D.deptos[k].n + (modo === "2027" ? " " + f(D.deptos[k].p27, 2) : "");
      });
      if (cont._bot) {
        $$("button", cont._bot).forEach((b) => b.classList.toggle("activo", b.dataset.m === modo));
        cont._anios.style.display = modo === "anio" ? "flex" : "none";
        $$("button", cont._anios).forEach((b) => b.classList.toggle("activo", +b.dataset.a === anio));
      }
      const cuenta = (c) => Object.values(D.deptos).filter((d) => d.cat === c).length;
      leyenda.innerHTML = modo === "2027"
        ? `<span><i style="background:var(--C)"></i>C consolidado (${cuenta("C consolidado")})</span>
           <span><i style="background:repeating-linear-gradient(45deg,var(--C) 0 4px,var(--panel2) 4px 8px)"></i>Transición (${cuenta("Transición")})</span>
           <span><i style="background:var(--D)"></i>D consolidado (${cuenta("D consolidado")})</span>
           <span><i style="background:var(--A);border-radius:50%;width:16px"></i>Alerta A/B</span>`
        : `<span><i style="background:var(--C)"></i>C · convencional</span><span><i style="background:var(--D)"></i>D · mixto</span>
           <span><i style="background:var(--panel2);border:1px solid var(--linea2)"></i>sin etiqueta (&lt; 10 eventos)</span>
           ${modo === "anio" ? `<span><i style="border:3px solid var(--tinta)"></i>transición</span>` : ""}`;
      pintarDetalle();
    }
    pintar();
    cont._pintar = pintar;
  }

  // mini gráfico por departamento (s_C observado y P(C) predicha)
  function chispa(cont, hist) {
    const W = 560, H = 210, m = { l: 46, r: 14, t: 14, b: 34 };
    const s = svg(W, H, cont); s.style.width = "100%"; s.style.height = "auto";
    const x = (a) => m.l + (a - 2021) / 5 * (W - m.l - m.r), y = (v) => m.t + (1 - v) * (H - m.t - m.b);
    const ej = el("g", { class: "eje" }, s);
    [0, 0.5, 1].forEach((v) => { el("line", { x1: m.l, x2: W - m.r, y1: y(v), y2: y(v), estilo: v === .5 ? "stroke:var(--acento);stroke-dasharray:5 5" : "" }, ej);
      el("text", { x: m.l - 8, y: y(v) + 5, "text-anchor": "end", "font-size": 15, texto: f(v, 1) }, ej); });
    [2021, 2022, 2023, 2024, 2025, 2026].forEach((a) => el("text", { x: x(a), y: H - 8, "text-anchor": "middle", "font-size": 15, texto: a }, ej));
    const pts = hist.filter((h) => h[1] !== null);
    if (pts.length > 1) el("path", { d: "M" + pts.map((h) => `${x(h[0])},${y(h[1])}`).join("L"), estilo: "fill:none;stroke:var(--Cclaro);stroke-width:3" }, s);
    pts.forEach((h) => el("circle", { cx: x(h[0]), cy: y(h[1]), r: 6, estilo: `fill:${h[2] === "C" ? "var(--C)" : "var(--D)"};stroke:var(--bg);stroke-width:2` }, s));
    hist.filter((h) => h[5] !== null).forEach((h) => el("rect", { x: x(h[0]) - 6, y: y(h[5]) - 6, width: 12, height: 12, transform: `rotate(45 ${x(h[0])} ${y(h[5])})`, estilo: "fill:none;stroke:var(--u3claro);stroke-width:3" }, s));
    const ley = document.createElement("div"); ley.className = "leyenda"; ley.style.fontSize = "17px";
    ley.innerHTML = `<span><i style="background:var(--Cclaro);height:4px"></i>s_C observado</span><span><i style="border:3px solid var(--u3claro);width:14px;height:14px;transform:rotate(45deg)"></i>P(C) del modelo (fuera de muestra)</span><span><i style="background:var(--acento);height:3px"></i>umbral 0,5</span>`;
    cont.appendChild(ley);
  }

  // ------------------------------------------------------------------------------------------ mapa de calor depto × año
  function heatmap(cont) {
    const anios = [2021, 2022, 2023, 2024, 2025, 2026];
    const filas = Object.entries(D.deptos).filter(([, d]) => d.hist.length)
      .map(([k, d]) => ({ k, n: d.n, h: d.hist, m: d.hist.reduce((a, h) => a + h[1], 0) / d.hist.length }))
      .sort((a, b) => b.m - a.m);
    const W = +(cont.dataset.ancho || 900), H = +(cont.dataset.alto || 820), L = 250, T = 40, B = 70;
    const cw = (W - L - 10) / anios.length, ch = (H - T - B) / filas.length;
    function dibujar() {
      cont.innerHTML = "";
      const s = svg(W, H, cont);
      const c0 = css("--panel2"), c1 = css("--Cclaro"), cD = css("--D");
      anios.forEach((a, j) => el("text", { x: L + j * cw + cw / 2, y: 26, "text-anchor": "middle", "font-size": 19, class: "mono", texto: a + (a === 2026 ? "*" : "") }, s));
      filas.forEach((r, i) => {
        el("text", { x: L - 12, y: T + i * ch + ch / 2 + 6, "text-anchor": "end", "font-size": Math.min(17, ch * .7), texto: r.n }, s);
        anios.forEach((a, j) => {
          const h = r.h.find((x) => x[0] === a);
          const rc = el("rect", { x: L + j * cw + 1.5, y: T + i * ch + 1.5, width: cw - 3, height: ch - 3, rx: 3, class: "celda" }, s);
          if (!h) { rc.setAttribute("style", `fill:none;stroke:${css("--linea")};stroke-dasharray:2 3`); return; }
          rc.setAttribute("style", `fill:${h[1] >= .5 ? mezclar(c0, c1, .35 + .65 * (h[1] - .5) / .5) : mezclar(c0, cD, .25 + .55 * (.5 - h[1]) / .5)};${h[3] ? `stroke:${css("--tinta")};stroke-width:2.5` : ""}`);
          rc.addEventListener("mousemove", (e) => mostrarTip(e, `<b>${r.n} · ${a}</b>` + filaTip("s_C", f(h[1], 2)) + filaTip("Perfil", h[2] === "C" ? "C · convencional" : "D · mixto") +
            filaTip("Eventos", miles(h[4])) + (h[3] ? filaTip("Cambio de perfil", "sí") : "") + (h[5] !== null ? filaTip("P(C) del modelo", f(h[5], 2)) : "")));
          rc.addEventListener("mouseleave", ocultarTip);
        });
      });
      // leyenda: divergente alrededor del umbral 0,5
      const lg = el("g", {}, s), lx = L, ly = H - 40, lw = W - L - 40;
      const grad = el("linearGradient", { id: "hm-grad" }, el("defs", {}, s));
      [[0, mezclar(c0, cD, .8)], [.5, c0], [1, c1]].forEach(([o, c]) => el("stop", { offset: o, "stop-color": c }, grad));
      el("rect", { x: lx, y: ly, width: lw, height: 14, rx: 4, fill: "url(#hm-grad)" }, lg);
      [[0, "s_C = 0 (D)"], [.5, "0,5 umbral"], [1, "1 (C)"]].forEach(([v, t]) => el("text", { x: lx + v * lw, y: ly + 36, "text-anchor": v === 0 ? "start" : v === 1 ? "end" : "middle", "font-size": 16, class: "mono", texto: t }, lg));
      el("rect", { x: lx - 150, y: ly - 2, width: 18, height: 18, rx: 3, estilo: `fill:none;stroke:${css("--tinta")};stroke-width:2.5` }, lg);
      el("text", { x: lx - 124, y: ly + 13, "font-size": 16, class: "mono", texto: "transición" }, lg);
    }
    dibujar(); alTema.push(dibujar);
  }

  // ------------------------------------------------------------------------------------------ composición nacional ACLED
  function composicion(cont) {
    const A = D.acled, W = +(cont.dataset.ancho || 900), H = +(cont.dataset.alto || 560), m = { l: 20, r: 20, t: 46, b: 50 };
    const s = svg(W, H, cont), n = A.anios.length, bw = (W - m.l - m.r) / n * .72, paso = (W - m.l - m.r) / n;
    const orden = ["C", "D", "A", "B"];
    A.anios.forEach((a, j) => {
      const tot = orden.reduce((t, p) => t + A.por_anio[p][j], 0);
      let y0 = H - m.b; const x = m.l + j * paso + (paso - bw) / 2;
      orden.forEach((p) => {
        const v = A.por_anio[p][j], h = v / tot * (H - m.t - m.b);
        const r = el("rect", { x, y: y0 - h, width: bw, height: Math.max(h - 2, 0), rx: 3, estilo: `fill:${COL_PERFIL[p]}`, class: "crecer" }, s);
        r.addEventListener("mousemove", (e) => mostrarTip(e, `<b>${NOMBRE_PERFIL[p]} · ${a}</b>` + filaTip("Eventos", miles(v)) + filaTip("Proporción", pct(v / tot, 1))));
        r.addEventListener("mouseleave", ocultarTip);
        if ((p === "C" || p === "D") && h > 40) el("text", { x: x + bw / 2, y: y0 - h / 2 + 8, "text-anchor": "middle", "font-size": 22, estilo: "fill:#111;font-weight:700", class: "mono", texto: pct(v / tot) }, s);
        y0 -= h;
      });
      const ab = (A.por_anio.A[j] + A.por_anio.B[j]) / tot;
      el("text", { x: x + bw / 2, y: m.t - 14, "text-anchor": "middle", "font-size": 18, class: "mono", estilo: "fill:var(--A)", texto: "A+B " + pct(ab) }, s);
      el("text", { x: x + bw / 2, y: H - m.b + 32, "text-anchor": "middle", "font-size": 20, class: "mono", texto: a + (a === 2026 ? "*" : "") }, s);
    });
  }

  // ------------------------------------------------------------------------------------------ barras HTML genéricas
  function barrasHTML(cont, filas, { dom = [0, 1], fmt = (v) => f(v), clase = () => "", ref = null, bigote = null } = {}) {
    cont.classList.add("barras"); cont.innerHTML = "";
    const esc = (v) => Math.max(0, Math.min(1, (v - dom[0]) / (dom[1] - dom[0]))) * 100;
    filas.forEach((r) => {
      const b = document.createElement("div"); b.className = "barra " + clase(r);
      b.innerHTML = `<span>${r.etq}</span><span class="riel"><span class="relleno"></span></span><span class="v">${fmt(r.v)}</span>`;
      const rel = $(".relleno", b); rel.dataset.w = esc(r.v);
      if (bigote && r.de) { const g = document.createElement("span"); g.className = "bigote"; g.style.left = esc(r.v - r.de) + "%"; g.style.width = (esc(r.v + r.de) - esc(r.v - r.de)) + "%"; $(".riel", b).appendChild(g); }
      if (ref !== null) { const mk = document.createElement("span"); mk.className = "marca-ref"; mk.style.left = esc(typeof ref === "function" ? ref(r) : ref) + "%"; $(".riel", b).appendChild(mk); }
      if (r.tip) { b.addEventListener("mousemove", (e) => mostrarTip(e, r.tip)); b.addEventListener("mouseleave", ocultarTip); }
      cont.appendChild(b);
    });
    const animar = () => { $$(".relleno", cont).forEach((x) => (x.style.width = 0)); requestAnimationFrame(() => setTimeout(() => $$(".relleno", cont).forEach((x) => (x.style.width = x.dataset.w + "%")), 60)); };
    if (!cont._anim) enDiapositiva(cont, () => cont._anim());
    cont._anim = animar; animar();
  }

  function subtipos(cont) {
    const max = D.acled.subtipos[0][1];
    barrasHTML(cont, D.acled.subtipos.map(([s, v, p]) => ({ etq: s, v, p, tip: `<b>${s}</b>` + filaTip("Perfil", NOMBRE_PERFIL[p]) + filaTip("Eventos 2020–2026", miles(v)) })),
      { dom: [0, max], fmt: miles, clase: (r) => "p" + r.p });
    $$(".barra", cont).forEach((b, i) => { $(".relleno", b).style.background = COL_PERFIL[D.acled.subtipos[i][2]]; });
  }

  function importancia(cont) {
    const max = Math.max(...D.importancia.map((r) => r[1] + r[2]));
    barrasHTML(cont, D.importancia.map(([n, v, de]) => ({ etq: n, v, de, tip: `<b>${n}</b>` + filaTip("Δ log-loss al permutar", f(v, 3)) + filaTip("desv. est. entre pliegues", f(de, 3)) })),
      { dom: [0, max], bigote: true, clase: (r) => /d₁|m₁/.test(r.etq) ? "u3" : "" });
  }

  function psi(cont) {
    const NOM = { x1_eventos: "x₁ eventos", x2_fatalidades: "x₂ fatalidades", x2b_letalidad: "x₂b letalidad", x4_coca: "x₄ coca", x5_bosque: "x₅ bosque",
      x6_distancia: "x₆ distancia", x7_ipm: "x₇ IPM", m1_sC_t1: "m₁ memoria", d1_sC_H2: "d₁ semestre" };
    barrasHTML(cont, D.psi.map((r) => ({ etq: NOM[r.var] || r.var, v: r.psi, p95: r.p95, sup: r.supera,
      tip: `<b>${NOM[r.var]}</b>` + filaTip("PSI 2026", f(r.psi, 2)) + filaTip("p95 sin deriva", f(r.p95, 2)) + filaTip("Estado", r.estado) })),
      { dom: [0, .5], fmt: (v) => f(v, 2), ref: (r) => r.p95, clase: (r) => r.sup ? "alerta" : "" });
    $$(".barra.alerta .relleno", cont).forEach((x) => (x.style.background = "var(--A)"));
  }

  // ------------------------------------------------------------------------------------------ escalera de mejora
  function escalera(cont) {
    const E = D.escalera, W = +(cont.dataset.ancho || 820), H = +(cont.dataset.alto || 520), m = { l: 96, r: 200, t: 40, b: 80 };
    const pers = resolver("comparaciones|sel|Persistencia|ll"), cal = resolver("comparaciones|sel|Persistencia calibrada|ll");
    const y0 = .26, y1 = .40, y = (v) => m.t + (1 - (v - y0) / (y1 - y0)) * (H - m.t - m.b);
    const s = svg(W, H, cont), paso = (W - m.l - m.r) / E.length, bw = paso * .58;
    const ej = el("g", { class: "eje" }, s);
    [.26, .30, .34, .38].forEach((v) => { el("line", { x1: m.l, x2: W - m.r, y1: y(v), y2: y(v), estilo: "stroke:var(--linea);stroke-dasharray:2 6" }, ej);
      el("text", { x: m.l - 10, y: y(v) + 6, "text-anchor": "end", "font-size": 18, texto: f(v, 2) }, ej); });
    el("text", { x: 22, y: m.t + (H - m.t - m.b) / 2, "font-size": 18, transform: `rotate(-90 22 ${m.t + (H - m.t - m.b) / 2})`, "text-anchor": "middle", class: "mono", texto: "log-loss 2023–2025" }, ej);
    E.forEach((r, i) => {
      const x = m.l + i * paso + (paso - bw) / 2, yb = y(r.ll), base = y(y0);
      const g = el("g", { class: "crecer-y" }, s);
      el("rect", { x, y: yb, width: bw, height: base - yb, rx: 5, estilo: `fill:${i === 0 ? "var(--gris)" : "var(--u3)"}` }, g);
      if (r.de > 0) { el("line", { x1: x + bw / 2, x2: x + bw / 2, y1: y(r.ll + r.de), y2: y(r.ll - r.de), estilo: "stroke:var(--tinta);stroke-width:2" }, g);
        [r.ll + r.de, r.ll - r.de].forEach((v) => el("line", { x1: x + bw / 2 - 8, x2: x + bw / 2 + 8, y1: y(v), y2: y(v), estilo: "stroke:var(--tinta);stroke-width:2" }, g)); }
      el("text", { x: x + bw / 2, y: y(r.ll + (r.de || 0)) - 12, "text-anchor": "middle", "font-size": 24, class: "mono", estilo: "fill:var(--tinta);font-weight:700", texto: f(r.ll) }, g);
      const etq = r.paso.replace("+ ", "+ ").split(" ");
      el("text", { x: x + bw / 2, y: H - m.b + 30, "text-anchor": "middle", "font-size": 18, texto: etq.slice(0, 2).join(" ") }, s);
      el("text", { x: x + bw / 2, y: H - m.b + 54, "text-anchor": "middle", "font-size": 18, texto: etq.slice(2).join(" ") }, s);
      g.addEventListener("mousemove", (e) => mostrarTip(e, `<b>${r.paso}</b>` + filaTip("log-loss media", f(r.ll)) + filaTip("desv. est. (5 semillas)", f(r.de)) + filaTip("AUC", f(r.auc)) + filaTip("transiciones acertadas (de 9)", f(r.trans, 1))));
      g.addEventListener("mouseleave", ocultarTip);
    });
    [[pers, "persistencia", "var(--acento)"], [cal, "pers. calibrada*", "var(--violeta)"]].forEach(([v, t, c]) => {
      el("line", { x1: m.l, x2: W - m.r + 6, y1: y(v), y2: y(v), estilo: `stroke:${c};stroke-width:2.5;stroke-dasharray:9 7` }, s);
      el("text", { x: W - m.r + 12, y: y(v) - 4, "font-size": 18, class: "mono", estilo: `fill:${c}`, texto: t }, s);
      el("text", { x: W - m.r + 12, y: y(v) + 18, "font-size": 18, class: "mono", estilo: `fill:${c};font-weight:700`, texto: f(v) }, s);
    });
    const animar = () => $$(".crecer-y", cont).forEach((g, i) => { g.style.transition = "none"; g.style.opacity = 0; g.style.transform = "translateY(30px)";
      setTimeout(() => { g.style.transition = "all .7s cubic-bezier(.2,.8,.2,1)"; g.style.opacity = 1; g.style.transform = "none"; }, 150 + i * 260); });
    enDiapositiva(cont, animar);
  }

  // ------------------------------------------------------------------------------------------ comparador antes / después / rivales
  const ORDEN_MOD = ["Persistencia", "Persistencia calibrada", "Base U2 · semilla 42", "Base U2 · 1 red", "Base U2 · ensamble 10",
    "Logística (9 variables)", "MLP U3 · 1 red", "MLP U3", "Embedding (no adoptado)"];
  const CONJ = { sel: "Selección 2023–2025 · n = 87", t26: "Prueba 2026 · n = 29 · una sola vez", et: "Departamentos nuevos y años futuros · n = 87", esp: "Departamentos nuevos (5 grupos)" };
  const MET = { ll: ["Log-loss", "menor es mejor", [0, .56]], auc: ["ROC-AUC", "mayor es mejor · eje desde 0,5", [.5, 1]], ece: ["ECE", "error de calibración · menor es mejor", [0, .25]],
    exact: ["Exactitud", "umbral 0,5 · mayor es mejor", [.5, 1]] };
  function comparador(cont) {
    const conjuntos = (cont.dataset.conjuntos || "sel,t26,et").split(","), metricas = (cont.dataset.metricas || "ll,auc,ece").split(",");
    let cj = conjuntos[0], mt = metricas[0];
    cont.innerHTML = `<div class="botonera b-c" style="margin-bottom:12px">${conjuntos.map((c) => `<button data-c="${c}">${CONJ[c].split(" · ")[0]}</button>`).join("")}</div>
      <div class="botonera b-m" style="margin-bottom:18px">${metricas.map((m) => `<button data-m="${m}">${MET[m][0]}</button>`).join("")}</div>
      <div class="fuente f-c" style="margin-bottom:14px"></div><div class="cuerpo"></div><div class="fuente f-v" style="margin-top:16px;font-size:19px"></div>`;
    $(".b-c", cont).addEventListener("click", (e) => { if (e.target.dataset.c) { cj = e.target.dataset.c; pintar(); } });
    $(".b-m", cont).addEventListener("click", (e) => { if (e.target.dataset.m) { mt = e.target.dataset.m; pintar(); } });
    function pintar() {
      $$(".b-c button", cont).forEach((b) => b.classList.toggle("activo", b.dataset.c === cj));
      $$(".b-m button", cont).forEach((b) => b.classList.toggle("activo", b.dataset.m === mt));
      const filas = ORDEN_MOD.map((n) => D.comparaciones[cj].find((r) => r.modelo === n)).filter((r) => r && r[mt] !== null);
      const [nom, sentido, dom] = MET[mt];
      $(".f-c", cont).textContent = `${CONJ[cj]} · ${nom}: ${sentido}`;
      $(".cuerpo", cont).classList.toggle("grandes", !!cont.dataset.grandes);
      barrasHTML($(".cuerpo", cont), filas.map((r) => ({ etq: r.modelo, v: r[mt], m: r.modelo,
        tip: `<b>${r.modelo}</b>` + filaTip("Log-loss", f(r.ll)) + filaTip("ROC-AUC", f(r.auc)) + filaTip("ECE", f(r.ece)) + filaTip("Exactitud", f(r.exact)) +
          (r.ntrans ? filaTip("Transiciones acertadas", `${f(r.trans, 0)} de ${f(r.ntrans, 0)}`) : "") })),
        { dom, clase: (r) => r.m === "MLP U3" ? "u3" : r.m === "Persistencia calibrada" ? "cal" : r.m.startsWith("Embedding") ? "emb" : "" });
      const mejor = mt === "ll" || mt === "ece" ? Math.min(...filas.map((r) => r[mt])) : Math.max(...filas.map((r) => r[mt]));
      const ganador = filas.find((r) => r[mt] === mejor);
      const u3 = filas.find((r) => r.modelo === "MLP U3");
      $(".f-v", cont).innerHTML = `Mejor en ${nom}: <b style="color:var(--tinta)">${ganador.modelo}</b> (${f(mejor)})` + (u3 && ganador !== u3 ? ` · MLP U3: ${f(u3[mt])}` : "");
    }
    pintar();
  }

  function trio(cont) {
    const conj = [["sel", "Selección 2023–2025"], ["t26", "Prueba 2026"], ["et", "Departamentos nuevos"]];
    const mods = [["Persistencia", "Persistencia"], ["Persistencia calibrada", "Pers. calibrada"], ["MLP U3", "MLP U3"]];
    cont.style.cssText += ";display:grid;grid-template-columns:repeat(3,1fr);gap:34px";
    conj.forEach(([k, t]) => {
      const p = document.createElement("div"); p.innerHTML = `<div class="fuente" style="margin-bottom:12px;font-size:19px">${t} · log-loss</div><div></div>`;
      cont.appendChild(p); const b = p.lastElementChild; b.classList.add("compacta");
      barrasHTML(b, mods.map(([m, e]) => { const r = D.comparaciones[k].find((x) => x.modelo === m); return { etq: e, v: r.ll, m }; }),
        { dom: [0, .5], clase: (r) => r.m === "MLP U3" ? "u3" : r.m === "Persistencia calibrada" ? "cal" : "" });
    });
  }

  // ------------------------------------------------------------------------------------------ gráfico de bosque de las intervenciones
  const ETQ_INT = { "I1 memoria m1": "I1 · memoria m₁", "I1 trayectoria m2, m3": "I1 · trayectoria m₂ m₃", "I2 semestre reciente d1": "I2 · semestre reciente d₁",
    "I2 tendencias d2, d3": "I2 · tendencias d₂ d₃", "I3 actores armados x3": "I3 · actores armados (UCDP)", "I4 vecindad (temporal)": "I4 · vecindad (temporal)",
    "I4 vecindad (espacial)": "I4 · vecindad (espacial)", "I5 objetivo blando (sin calibrar)": "I5 · objetivo blando", "I5 objetivo blando (ambos calibrados)": "I5 · objetivo blando, calibrado",
    "I9 multitarea λ=0.5 (efecto en P(C))": "I9 · multitarea A/B/C/D λ 0,5", "I9 multitarea λ=1 (efecto en P(C))": "I9 · multitarea A/B/C/D λ 1",
    "GRU trimestral": "GRU (4 trimestres)", "CNN 1D trimestral": "CNN 1D (4 trimestres)", "Embedding de departamento": "Embedding de departamento",
    "MLP profundo (16, 8)": "MLP profundo 16-8", "Logística (mismas variables)": "Logística, mismas variables" };
  function bosque(cont) {
    const filas = D.decisiones.filter((r) => r.ic), W = +(cont.dataset.ancho || 1000), H = +(cont.dataset.alto || 700), m = { l: 360, r: 30, t: 40, b: 60 };
    const x0 = -.08, x1 = .2, x = (v) => m.l + (v - x0) / (x1 - x0) * (W - m.l - m.r), rh = (H - m.t - m.b) / filas.length;
    const s = svg(W, H, cont);
    el("rect", { x: x(x0), y: m.t - 10, width: x(0) - x(x0), height: H - m.t - m.b + 10, estilo: "fill:var(--u3);opacity:.08" }, s);
    el("text", { x: x(x0) + 10, y: m.t - 16, "font-size": 18, class: "mono", estilo: "fill:var(--u3claro)", texto: "← mejora" }, s);
    el("text", { x: W - m.r, y: m.t - 16, "font-size": 18, class: "mono", "text-anchor": "end", texto: "empeora →" }, s);
    const ej = el("g", { class: "eje" }, s);
    [-.05, 0, .05, .1, .15].forEach((v) => { el("line", { x1: x(v), x2: x(v), y1: m.t - 10, y2: H - m.b, estilo: v === 0 ? "stroke:var(--tinta2);stroke-width:2" : "stroke:var(--linea);stroke-dasharray:2 6" }, ej);
      el("text", { x: x(v), y: H - m.b + 28, "text-anchor": "middle", "font-size": 17, texto: fs(v, 2) }, ej); });
    el("text", { x: (m.l + W - m.r) / 2, y: H - 6, "text-anchor": "middle", "font-size": 17, class: "mono", texto: "Δ log-loss frente a su referencia · IC 95 % remuestreando departamentos" }, s);
    filas.forEach((r, i) => {
      const yc = m.t + i * rh + rh / 2, r8 = /regla 8/.test(r.decision);
      const c = r.retiene ? "var(--u3claro)" : r8 ? "var(--violeta)" : "var(--gris2)";
      const g = el("g", { class: "fila-bosque", estilo: "cursor:pointer" }, s);
      el("rect", { x: 0, y: yc - rh / 2, width: W, height: rh, estilo: "fill:transparent" }, g);
      el("text", { x: m.l - 16, y: yc + 6, "text-anchor": "end", "font-size": 19, estilo: `fill:${r.retiene || r8 ? "var(--tinta)" : "var(--tinta2)"};font-weight:${r.retiene ? 700 : 400}`, texto: ETQ_INT[r.nombre] || r.nombre }, g);
      el("line", { x1: x(r.ic[0]), x2: x(Math.min(r.ic[1], x1)), y1: yc, y2: yc, estilo: `stroke:${c};stroke-width:4;stroke-linecap:round` }, g);
      el("circle", { cx: x(r.dll), cy: yc, r: 8, estilo: `fill:${c};stroke:var(--bg);stroke-width:2` }, g);
      g.addEventListener("mousemove", (e) => mostrarTip(e, `<b>${ETQ_INT[r.nombre] || r.nombre}</b>` + filaTip("Δ log-loss (5 semillas)", fs(r.dll)) + filaTip("EE entre semillas", f(r.ee, 4)) +
        filaTip("IC 95 % por depto.", ic(r.ic)) + filaTip("Δ AUC", fs(r.dauc)) + `<div style="margin-top:6px">${r.decision}</div>`));
      g.addEventListener("mouseleave", ocultarTip);
    });
    const animar = () => $$(".fila-bosque", cont).forEach((g, i) => { g.style.transition = "none"; g.style.opacity = 0;
      setTimeout(() => { g.style.transition = "opacity .5s"; g.style.opacity = 1; }, 100 + i * 70); });
    enDiapositiva(cont, animar);
  }

  // ------------------------------------------------------------------------------------------ red neuronal animada
  function red(cont) {
    const u3 = (cont.dataset.entradas || "9") === "9";
    const ent = u3 ? ["d₁ semestre reciente", "m₁ memoria t−1", "x₇ IPM", "x₆ distancia", "x₅ bosque", "x₄ coca", "x₂b letalidad", "x₂ fatalidades", "x₁ eventos"]
      : ["x₇ IPM", "x₆ distancia", "x₅ bosque", "x₄ coca", "x₂b letalidad", "x₂ fatalidades", "x₁ eventos"];
    const nuevas = u3 ? 2 : 0, K = +(cont.dataset.ensamble || (u3 ? 10 : 1));
    const W = +(cont.dataset.ancho || 900), H = +(cont.dataset.alto || 640), xe = 250, xo = 140 + (W - 140) * .52, xs = W - 150;
    const ye = (i) => 40 + i * (H - 120) / (ent.length - 1), yo = (j) => 30 + j * (H - 100) / 15, ys = H / 2 - 30;
    const s = svg(W, H, cont);
    for (let k = Math.min(K, 6) - 1; k > 0; k--) {
      el("rect", { x: xo - 26 + k * 9, y: 14 - k * 7, width: 52, height: H - 64, rx: 26, estilo: "fill:none;stroke:var(--linea2);opacity:" + (.6 - k * .08) }, s);
    }
    const aristas = el("g", {}, s), pulsos = el("g", {}, s), nodos = el("g", {}, s);
    const lineas = [];
    ent.forEach((n, i) => { for (let j = 0; j < 16; j++) lineas.push({ a: [xe, ye(i)], b: [xo, yo(j)], c: 0, i, j }); });
    for (let j = 0; j < 16; j++) lineas.push({ a: [xo, yo(j)], b: [xs, ys], c: 1, j });
    lineas.forEach((l) => { l.el = el("line", { x1: l.a[0], y1: l.a[1], x2: l.b[0], y2: l.b[1], estilo: "stroke:var(--linea2);stroke-width:1;opacity:.55" }, aristas); });
    ent.forEach((n, i) => {
      const nueva = i < nuevas;
      el("circle", { cx: xe, cy: ye(i), r: 13, estilo: `fill:${nueva ? "var(--u3)" : "var(--panel2)"};stroke:${nueva ? "var(--u3claro)" : "var(--tinta2)"};stroke-width:2` }, nodos);
      el("text", { x: xe - 26, y: ye(i) + 7, "text-anchor": "end", "font-size": 21, estilo: `fill:${nueva ? "var(--u3claro)" : "var(--tinta2)"};font-weight:${nueva ? 700 : 400}`, texto: n }, nodos);
    });
    for (let j = 0; j < 16; j++) el("circle", { cx: xo, cy: yo(j), r: 10, estilo: "fill:var(--B);stroke:var(--bg);stroke-width:2" }, nodos);
    const salida = el("circle", { cx: xs, cy: ys, r: 26, estilo: "fill:var(--Cclaro);stroke:var(--bg);stroke-width:3" }, nodos);
    el("text", { x: xs, y: ys + 62, "text-anchor": "middle", "font-size": 22, estilo: "fill:var(--tinta);font-weight:700", texto: "P(C | xₜ₋₁)" }, nodos);
    el("text", { x: xs, y: ys + 90, "text-anchor": "middle", "font-size": 17, class: "mono", texto: K > 1 ? `promedio de ${K} redes` : "sigmoide" }, nodos);
    el("text", { x: xe, y: H - 18, "text-anchor": "middle", "font-size": 18, class: "mono", texto: `entrada · ${ent.length}` }, nodos);
    el("text", { x: xo, y: H - 18, "text-anchor": "middle", "font-size": 18, class: "mono", texto: "oculta · 16 ReLU" }, nodos);
    if (K > 1) el("text", { x: xo + 50, y: H / 2, "font-size": 30, class: "mono", estilo: "fill:var(--acento);font-weight:700", texto: `× ${K}` }, nodos);
    const params = ent.length * 16 + 16 + 16 + 1;
    el("text", { x: xs, y: ys + 118, "text-anchor": "middle", "font-size": 17, class: "mono", texto: `${params} parámetros por red` }, nodos);

    let anim = null;
    function correr(atras) {
      cancelAnimationFrame(anim); pulsos.innerHTML = "";
      const sub = lineas.filter((l, k) => l.c === 1 || k % 4 === 0);
      const ps = sub.map((l) => ({ l, e: el("circle", { r: 5, estilo: `fill:${atras ? "var(--A)" : "var(--acento2)"}` }, pulsos), d: (atras ? 1 - l.c : l.c) * .5 + Math.random() * .12 }));
      const t0 = performance.now(), dur = 2300;
      salida.style.transition = "r .3s";
      lineas.forEach((l) => (l.el.style.stroke = atras ? "var(--A)" : "var(--acento)"));
      function cuadro(t) {
        const u = (t - t0) / dur;
        ps.forEach((p) => { const v = Math.max(0, Math.min(1, (u - p.d) / .45)); const q = atras ? 1 - v : v;
          p.e.setAttribute("cx", p.l.a[0] + (p.l.b[0] - p.l.a[0]) * q); p.e.setAttribute("cy", p.l.a[1] + (p.l.b[1] - p.l.a[1]) * q);
          p.e.style.opacity = v <= 0 || v >= 1 ? 0 : 1; });
        lineas.forEach((l) => (l.el.style.opacity = .25 + .45 * Math.sin(Math.min(1, u) * Math.PI)));
        if (!atras && u > .9) salida.setAttribute("r", 34);
        if (u < 1.15) anim = requestAnimationFrame(cuadro);
        else { pulsos.innerHTML = ""; lineas.forEach((l) => { l.el.style.stroke = ""; l.el.style.opacity = .55; }); salida.setAttribute("r", 26); }
      }
      anim = requestAnimationFrame(cuadro);
      $$(".lazo span", cont.parentElement).forEach((x, k) => x.classList.toggle("resalta", atras ? k >= 3 : k < 3));
    }
    const bot = cont.parentElement.querySelector(".botones-red");
    if (bot) bot.addEventListener("click", (e) => { if (e.target.dataset.r) correr(e.target.dataset.r === "atras"); });
    enDiapositiva(cont, () => setTimeout(() => correr(false), 700));
  }

  // ------------------------------------------------------------------------------------------ pipeline de 8 etapas
  const ETAPAS = [
    ["Datos", "Contrato del export, auditoría, panel y etiqueta", "dataset_modelado_u3.csv"],
    ["Variables", "9 predictoras en t−1; escalador ajustado por pliegue", "configuracion.json"],
    ["Evaluación", "Origen móvil 2023–2025 contra las persistencias y la U2", "metricas_seleccion.csv"],
    ["Calibración", "Solo si mejora sin mirar el futuro (no mejoró)", "en el artefacto"],
    ["Prueba", "2026, una sola vez, con todo congelado", "metricas_prueba_2026.csv"],
    ["Despliegue", "10 redes 2021–2025 en un artefacto .keras", "perfil_amenaza_u3.keras"],
    ["Proyección", "P(C) 2027, bandas, alerta A/B y PSI", "proyeccion_2027_u3.csv"],
    ["Tarjeta", "Uso previsto, límites, métricas y deriva", "TARJETA_MODELO.md"]];
  function pipeline(cont) {
    cont.style.cssText += ";display:flex;flex-direction:column;gap:20px";
    const fila = document.createElement("div"); fila.style.cssText = "display:grid;grid-template-columns:repeat(8,1fr);gap:14px;position:relative";
    const det = document.createElement("div"); det.className = "tarjeta sin-marca"; det.style.cssText = "padding:20px 28px;min-height:96px";
    ETAPAS.forEach(([n, d, a], i) => {
      const b = document.createElement("button"); b.className = "etapa";
      b.style.cssText = "background:var(--panel);border:1px solid var(--linea2);border-radius:14px;padding:16px 12px;text-align:left;cursor:pointer;color:var(--tinta);font-family:var(--f-texto);transition:all .25s";
      b.innerHTML = `<div class="mono" style="font-size:17px;color:var(--acento)">${String(i + 1).padStart(2, "0")}</div><div style="font-family:var(--f-titulo);font-size:25px;font-weight:600">${n}</div>`;
      b.addEventListener("mouseenter", () => elegir(i)); b.addEventListener("click", () => { auto = false; elegir(i); });
      fila.appendChild(b);
    });
    cont.appendChild(fila); cont.appendChild(det);
    let auto = true, idx = 0, temp = null;
    function elegir(i) {
      idx = i;
      $$(".etapa", fila).forEach((b, k) => { b.style.borderColor = k === i ? "var(--acento)" : k < i ? "var(--u3)" : "var(--linea2)"; b.style.background = k === i ? "var(--panel2)" : "var(--panel)"; b.style.transform = k === i ? "translateY(-6px)" : ""; });
      det.innerHTML = `<div style="display:flex;gap:26px;align-items:baseline"><span class="mono resalta" style="font-size:20px">Etapa ${i + 1}</span><span style="font-size:28px;color:var(--tinta)">${ETAPAS[i][1]}</span>
        <span class="mono tenue" style="margin-left:auto;font-size:18px">→ ${ETAPAS[i][2]}</span></div>`;
    }
    elegir(0);
    enDiapositiva(cont, () => { auto = true; clearInterval(temp); let k = 0; elegir(0); temp = setInterval(() => { if (!auto || ++k > 7) return clearInterval(temp); elegir(k); }, 1500); });
  }

  // ------------------------------------------------------------------------------------------ terminal con la salida real
  function terminal(cont) {
    const T = D.terminal;
    const pasos = [["python -m pytest -q tests", T.pytest], ["python -m perfil_amenaza.pipeline todo", T.todo],
      ["python -m perfil_amenaza.pipeline predecir --anio 2026", (T.predecir || []).slice(0, +(cont.dataset.lineas || 7)).map((l) => l.replace(/\s+(ISR|Orden|Portafolio|Énfasis).*$/, ""))]];
    cont.innerHTML = `<div class="terminal" style="height:100%"><div class="barra-t"><i style="background:#e66767"></i><i style="background:#f2a900"></i><i style="background:#1fbf8f"></i>
      <span>Proyecto_Unidad_3 · PYTHONPATH=src</span><button class="boton" style="margin-left:auto;font-size:15px;padding:4px 12px">↻ reproducir</button></div><pre></pre></div>`;
    const pre = $("pre", cont); let gen = 0;
    const esc = (t) => t.replace(/&/g, "&amp;").replace(/</g, "&lt;");
    async function correr() {
      const mio = ++gen; pre.innerHTML = "";
      const espera = (ms) => new Promise((r) => setTimeout(r, ms));
      for (const [cmd, out] of pasos) {
        if (mio !== gen) return;
        const linea = document.createElement("div"); linea.innerHTML = `<span class="prompt">$ </span><span class="cmd"></span><span class="cursor"></span>`; pre.appendChild(linea);
        const c = $(".cmd", linea);
        for (const ch of cmd) { if (mio !== gen) return; c.textContent += ch; await espera(28); }
        $(".cursor", linea).remove(); await espera(350);
        for (const o of out || []) { if (mio !== gen) return; const d = document.createElement("div");
          d.innerHTML = esc(o).replace(/(\d+ passed)/, '<span class="ok">$1</span>').replace(/(log-loss [0-9.]+)/, '<span class="ok">$1</span>');
          pre.appendChild(d); await espera(170); }
        await espera(500);
      }
    }
    $(".boton", cont).addEventListener("click", correr);
    enDiapositiva(cont, () => setTimeout(correr, 500));
  }

  // ------------------------------------------------------------------------------------------ simulador de bandas de decisión
  function simulador(cont) {
    cont.innerHTML = `<div style="display:flex;flex-direction:column;gap:26px">
      <div style="display:flex;align-items:flex-end;gap:36px"><div class="stat"><span class="valor s-v">0,50</span><span class="etq">P(C) del departamento</span></div>
        <div class="stat"><span class="valor s-c" style="font-size:56px">Transición</span><span class="etq">categoría</span></div></div>
      <input type="range" class="deslizador" min="0" max="1" step="0.01" value="0.5">
      <div class="sim-escala"><div style="width:30%;background:color-mix(in srgb, var(--D) 55%, transparent)">D consolidado &lt; 0,3</div>
        <div style="width:40%;background:repeating-linear-gradient(45deg, color-mix(in srgb, var(--C) 55%, transparent) 0 10px, transparent 10px 20px)">Transición</div>
        <div style="width:30%;background:var(--C);color:#111">C consolidado ≥ 0,7</div></div>
      <label class="interruptor"><span class="caja"></span>Alerta A/B (A+B ≥ 2 × la nacional y ≥ 5 eventos)</label>
      <div class="tarjeta s-e" style="padding:22px 28px"></div><div class="fuente s-d"></div></div>`;
    const r = $("input", cont), sw = $(".interruptor", cont);
    const ENF = { "C consolidado": "ISR (inteligencia, vigilancia y reconocimiento), detección temprana y movilidad protegida",
      "Transición": "Portafolio flexible y monitoreo trimestral del perfil", "D consolidado": "Orden público y capacidades mixtas" };
    function act() {
      const p = +r.value, cat = p >= .7 ? "C consolidado" : p >= .3 ? "Transición" : "D consolidado", al = sw.classList.contains("on");
      $(".s-v", cont).textContent = f(p, 2);
      const c = $(".s-c", cont); c.textContent = cat; c.style.color = cat === "C consolidado" ? "var(--Cclaro)" : cat === "Transición" ? "var(--acento2)" : "var(--tinta2)";
      $(".s-e", cont).innerHTML = `<div class="fuente">Énfasis tecnológico sugerido</div><p class="grande" style="margin-top:6px">${ENF[cat]}${al ? ' <span style="color:var(--A)">+ C-UAS (antidrones) y detección de AEI</span>' : ""}</p>`;
      const sim = Object.values(D.deptos).filter((d) => d.cat === cat).sort((a, b) => b.p27 - a.p27).map((d) => d.n + (d.alerta ? " ⚠" : ""));
      $(".s-d", cont).innerHTML = `En 2027 caen en esta banda (${sim.length}): ${sim.join(" · ")}`;
    }
    r.addEventListener("input", act); sw.addEventListener("click", () => { sw.classList.toggle("on"); act(); }); act();
  }

  // ------------------------------------------------------------------------------------------ contadores, giros, pestañas, hitos
  function contador(e) {
    const fin = +e.dataset.contar, dec = +(e.dataset.dec || 0), suf = e.dataset.suf || "";
    const pon = (v) => (e.textContent = (dec ? f(v, dec) : miles(v)) + suf);
    pon(fin);
    enDiapositiva(e, () => { const t0 = performance.now(), dur = 1400;
      const paso = (t) => { const u = Math.min(1, (t - t0) / dur); pon(fin * (1 - Math.pow(1 - u, 3))); if (u < 1) requestAnimationFrame(paso); };
      requestAnimationFrame(paso); });
  }
  function pestanas(cont) {
    const bs = $$(".pestanas button", cont), ps = $$(".panel-pestana", cont);
    bs.forEach((b, i) => b.addEventListener("click", () => { bs.forEach((x, k) => x.classList.toggle("activo", k === i)); ps.forEach((x, k) => x.classList.toggle("activo", k === i)); }));
    if (bs[0]) bs[0].click();
  }

  // resultados de la U2 (experimento E1: memoria cruda x8)
  function e1(cont) {
    const filas = D.u2.e1.filter((r) => r.val === (cont.dataset.val || "Temporal"));
    barrasHTML(cont, filas.map((r) => ({ etq: r.modelo.replace(" (x1-x7)", "").replace("ensamble 5 semillas", "ensamble 5"), v: r.ll, m: r.modelo,
      tip: `<b>${r.modelo}</b>` + filaTip("Log-loss", f(r.ll)) + filaTip("ROC-AUC", f(r.auc)) })),
      { dom: [0, .6], clase: (r) => r.m === "Persistencia" ? "cal" : /x8/.test(r.m) ? "u3" : "" });
  }
  // proporción nacional de cada perfil en 2020-2026
  function cuotas() {
    const tot = (p) => D.acled.por_anio[p].reduce((a, b) => a + b, 0), T = ["A", "B", "C", "D"].reduce((a, p) => a + tot(p), 0);
    $$("[data-cuota]").forEach((e) => (e.textContent = pct(tot(e.dataset.cuota) / T, 1)));
  }

  // ------------------------------------------------------------------------------------------ arranque
  const COMP = { "g-mapa": mapa, "g-heatmap": heatmap, "g-composicion": composicion, "g-subtipos": subtipos, "g-escalera": escalera,
    "g-comparador": comparador, "g-bosque": bosque, "g-red": red, "g-pipeline": pipeline, "g-terminal": terminal, "g-simulador": simulador,
    "g-importancia": importancia, "g-psi": psi, "g-trio": trio, "g-e1": e1, "g-pestanas": pestanas };
  $$("[data-contar-v]").forEach((e) => (e.dataset.contar = resolver(e.dataset.contarV)));
  rellenar(); cuotas();
  for (const [clase, fn] of Object.entries(COMP)) $$("." + clase).forEach((n) => { try { fn(n); } catch (err) { console.error(clase, err); } });
  $$("[data-contar]").forEach(contador);
  $$(".giro").forEach((g) => g.addEventListener("click", () => g.classList.toggle("girada")));
  $$("[data-ir]").forEach((h) => h.addEventListener("click", () => window.Presentacion && window.Presentacion.ir(+h.dataset.ir - 1)));
  window.Graficos = { resolver, f, miles };
})();
