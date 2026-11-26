"use strict";
// Shared helpers for every live-stt page: navigation, API calls with the token, time formatting,
// sentence assembly (same rules as live_stt/text/sentences.py:assemble) and sentence rendering.
const END_CHARS = "۔؟?!.…", PAUSE_S = 1.0, MAX_SENT_S = 20.0, LOW_WORD = 0.9;
const pad = (n) => String(n).padStart(2, "0");
const fmtLocal = (t) => { const d = new Date(t * 1000); return `${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`; };
const fmtDateTime = (t) => { const d = new Date(t * 1000); return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${fmtLocal(t)}`; };
const fmtUtc = (t) => new Date(t * 1000).toISOString().replace("T", " ").substr(0, 23) + "Z";
const fmtOffset = (s) => { s = Math.max(0, Math.floor(s)); const h = Math.floor(s / 3600); return (h ? h + ":" : "") + pad(Math.floor(s % 3600 / 60)) + ":" + pad(s % 60); };
const fmtDur = (s) => { s = Math.round(s || 0); return s >= 3600 ? `${Math.floor(s / 3600)} h ${pad(Math.floor(s % 3600 / 60))} m` : s >= 60 ? `${Math.floor(s / 60)} m ${pad(s % 60)} s` : `${s} s`; };
const localTz = () => { try { return Intl.DateTimeFormat().resolvedOptions().timeZone || "UTC"; } catch (e) { return "UTC"; } };
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

// ---- sentences across segment cuts (same rules as live_stt/text/sentences.py:assemble)
function wordOf(x) { return [String(x.w ?? x.word ?? "").trim(), x.s ?? x.start_utc ?? null, x.e ?? x.end_utc ?? null]; }
function assemble(rows) {
  const words = [];
  for (const r of [...rows].sort((a, b) => (a.start_utc || 0) - (b.start_utc || 0))) {
    let ws = (r.words || []).filter(x => x && typeof x === "object" && wordOf(x)[0]);
    if (!ws.length && (r.text_raw || "").trim()) ws = [{ w: r.text_raw.trim(), s: r.start_utc, e: r.end_utc }];
    for (const x of ws) words.push([x, r]);
  }
  const out = [];
  const gap = (a, b) => { const e = wordOf(a)[2], s = wordOf(b)[1]; return (e != null && s != null) ? s - e : 0; };
  const emit = (chunk) => {
    if (!chunk.length) return;
    const ws = chunk.map(c => c[0]);
    const rs = [...new Set(chunk.map(c => c[1]))];
    const starts = ws.map(x => wordOf(x)[1]).filter(v => v != null);
    const ends = ws.map(x => wordOf(x)[2]).filter(v => v != null);
    const ps = ws.map(x => x.p).filter(p => typeof p === "number" && p >= 0 && p <= 1);
    const passes = new Set(rs.map(r => ((r.pass || r.type) === "refined" ? "refined" : "fast")));
    const lats = rs.map(r => r.latency_s ?? (r.timings || {}).latency_s).filter(v => typeof v === "number");
    out.push({ s: starts.length ? Math.min(...starts) : rs[0].start_utc,
               e: ends.length ? Math.max(...ends) : rs[rs.length - 1].end_utc,
               text: ws.map(x => wordOf(x)[0]).join(" "), ws,
               c: ps.length ? ps.reduce((a, b) => a + b, 0) / ps.length : null,
               pass: passes.size === 1 ? [...passes][0] : "mixed",
               lat: lats.length ? Math.max(...lats) : null, segs: rs.map(r => r.segment_id) });
  };
  let cur = [];
  words.forEach(([x, r], i) => {
    cur.push([x, r]);
    const t = wordOf(x)[0], nx = words[i + 1] ? words[i + 1][0] : null;
    if (END_CHARS.includes(t[t.length - 1]) || (nx && gap(x, nx) >= PAUSE_S)) { emit(cur); cur = []; return; }
    let s0 = wordOf(cur[0][0])[1]; const e1 = wordOf(x)[2];
    while (cur.length > 1 && s0 != null && e1 != null && e1 - s0 > MAX_SENT_S) {
      let k = 1, best = -Infinity;
      for (let j = 1; j < cur.length; j++) { const g = gap(cur[j - 1][0], cur[j][0]); if (g >= best) { best = g; k = j; } }
      emit(cur.slice(0, k)); cur = cur.slice(k); s0 = wordOf(cur[0][0])[1];
    }
  });
  emit(cur);
  return out;
}
const confClass = (c) => c == null ? "" : c >= 0.9 ? "hiC" : c >= 0.8 ? "midC" : "loC";

// ---- deployment config, links, token
const STT = { config: null };
async function loadConfig() {
  if (STT.config) return STT.config;
  try { const r = await fetch("/ui/config"); STT.config = r.ok ? await r.json() : {}; } catch (e) { STT.config = {}; }
  return STT.config;
}
function resolveLink(tpl) {
  return String(tpl || "").replaceAll("{scheme}", location.protocol.replace(":", "")).replaceAll("{host}", location.hostname);
}
function getToken() { try { return localStorage.getItem("stt_token") || ""; } catch (e) { return ""; } }
function setToken(v) { try { v ? localStorage.setItem("stt_token", v) : localStorage.removeItem("stt_token"); } catch (e) { /* private mode */ } }
async function api(path, opts = {}) {
  const headers = Object.assign({}, opts.headers || {});
  const tok = getToken();
  if (tok) headers.Authorization = "Bearer " + tok;
  if (opts.json !== undefined) { headers["Content-Type"] = "application/json"; opts.body = JSON.stringify(opts.json); }
  const r = await fetch(path, Object.assign({}, opts, { headers }));
  if (!r.ok) {
    let msg = `${r.status} ${r.statusText}`;
    try { const j = await r.json(); if (j.detail) msg = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail); } catch (e) { /* not json */ }
    if (r.status === 401) msg += " — set the API token (top right, 🔑)";
    throw new Error(msg);
  }
  if (r.status === 204) return null;
  const ct = r.headers.get("content-type") || "";
  return ct.includes("json") ? r.json() : r.text();
}
function toast(msg, bad) {
  let t = document.getElementById("toast");
  if (!t) { t = document.createElement("div"); t.id = "toast"; t.className = "toast"; document.body.appendChild(t); }
  t.textContent = msg; t.style.display = "block"; t.style.borderColor = bad ? "var(--bad)" : "var(--line)";
  clearTimeout(t._h); t._h = setTimeout(() => { t.style.display = "none"; }, bad ? 8000 : 3500);
}

// ---- theme (light by default, dark on request; remembered per browser). Runs before first paint.
// Order: explicit ?theme=light|dark (e.g. Grafana's Light/Dark links) → the viewer's own choice →
// inside Grafana (?embed=1 / ?theme=auto) the computer's light/dark setting → light.
function getTheme() {
  const q = new URLSearchParams(location.search);
  const t = q.get("theme");
  if (t === "light" || t === "dark") return t;
  try { const v = localStorage.getItem("stt_theme"); if (v === "light" || v === "dark") return v; } catch (e) { /* */ }
  if (t === "auto" || q.get("embed") === "1")
    return window.matchMedia && matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  return "light";
}
function setTheme(t) {
  document.documentElement.dataset.theme = t;
  try { localStorage.setItem("stt_theme", t); } catch (e) { /* private mode */ }
  const b = document.getElementById("navTheme") || document.getElementById("embTheme");
  if (b) { b.textContent = t === "dark" ? "☀ Light" : "☾ Dark"; b.title = `Switch to ${t === "dark" ? "light" : "dark"} theme`; }
}
document.documentElement.dataset.theme = getTheme();
// Transcript text size (A− / A+), remembered per browser.
function urduSize(delta) {
  let v = 22;
  try { v = parseInt(localStorage.getItem("stt_urdu_size") || "22", 10) || 22; } catch (e) { /* default */ }
  if (delta) v = Math.max(16, Math.min(34, v + delta));
  document.documentElement.style.setProperty("--urdu-size", v + "px");
  try { localStorage.setItem("stt_urdu_size", String(v)); } catch (e) { /* private mode */ }
  return v;
}
urduSize(0);

// ---- navigation bar (every page; hidden with ?embed=1, e.g. inside Grafana)
const PAGES = [["live", "/", "Live"], ["archive", "/ui/archive", "Find a moment"], ["reports", "/ui/reports", "Reports"],
               ["uploads", "/ui/uploads", "Upload a file"], ["channels", "/ui/channels", "Channel setup"], ["system", "/ui/system", "System & links"]];
const EMBED = new URLSearchParams(location.search).get("embed") === "1";
async function nav(active) {
  if (EMBED) { document.body.classList.add("embed"); return; }
  const n = document.createElement("nav");
  n.className = "top";
  n.innerHTML = `<a class="brand" href="/"><span class="logo">◉</span>Urdu Live Transcription</a>` +
    PAGES.map(([k, href, label]) => `<a class="tab${k === active ? " on" : ""}" href="${href}">${label}</a>`).join("") +
    `<span class="sp"></span><span id="navHealth" class="pill" title="Server health">checking…</span>
     <a id="navGrafana" class="pill" target="_blank" rel="noopener" title="Metrics dashboard">Grafana ↗</a>
     <button id="navTheme" class="pill"></button>
     <button id="navToken" class="pill" title="API token for changes (channel setup, uploads)">🔑</button>`;
  document.body.prepend(n);
  setTheme(getTheme());
  n.querySelector("#navTheme").onclick = () => setTheme(document.documentElement.dataset.theme === "dark" ? "light" : "dark");
  const tokBtn = n.querySelector("#navToken");
  const showTok = () => { tokBtn.textContent = getToken() ? "🔑 token set" : "🔑 token"; };
  showTok();
  tokBtn.onclick = () => {
    const v = prompt("API token (STT_API__TOKEN on the server). Stored in this browser only. Empty = clear.", getToken());
    if (v !== null) { setToken(v.trim()); showTok(); }
  };
  const cfg = await loadConfig();
  const g = n.querySelector("#navGrafana");
  g.href = resolveLink((cfg.links || {}).grafana || "{scheme}://{host}:3000/d/live-stt");
  if (cfg.token_required === false) { tokBtn.style.display = "none"; }
  pollNavHealth();
  setInterval(pollNavHealth, 10000);
}
async function pollNavHealth() {
  const h = document.getElementById("navHealth");
  if (!h) return;
  try {
    const j = await (await fetch("/health")).json();
    h.innerHTML = `<span class="dot"></span>${j.status === "ok" ? "System OK" : "System problem"}`;
    h.className = "pill " + (j.status === "ok" ? "ok" : "bad");
    h.title = `redis ${j.redis ? "ok" : "down"} · database ${j.db ? "ok" : "down"} — details on System & links`;
  } catch (e) { h.textContent = "server unreachable"; h.className = "pill bad"; }
}
// The one channel this deployment shows: ?ch=… if given, else the first enabled channel.
async function mainChannel() {
  const want = new URLSearchParams(location.search).get("ch");
  const chans = await channelList();
  const on = chans.filter(c => c.enabled !== false);
  return { chans, cur: chans.find(c => c.id === want) || on[0] || chans[0] || null };
}

// "239.1.1.1:5000" → "udp://@239.1.1.1:5000" (multicast join), "10.0.0.5:5001" → "udp://10.0.0.5:5001"
function normalizeLink(v) {
  v = v.trim();
  if (!v) return "";
  if (!/^[a-z][a-z0-9+.-]*:\/\//i.test(v)) {
    const host = v.replace(/^@/, "").split(":")[0];
    const first = parseInt(host.split(".")[0], 10);
    v = "udp://" + (first >= 224 && first <= 239 ? "@" : "") + v.replace(/^@/, "");
  }
  return v;
}
// ---- time inputs (local time of the browser)
function toInputs(t) { const d = new Date(t * 1000); return [`${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}`, `${pad(d.getHours())}:${pad(d.getMinutes())}`]; }
function fromInputs(date, time) { if (!date || !time) return null; const t = new Date(`${date}T${time}`).getTime(); return isNaN(t) ? null : t / 1000; }

// ---- sentence list rendering (archive / uploads / search)
function highlightTerms(el, text, terms) {
  if (!terms || !terms.length) { el.textContent = text; return; }
  const low = text.toLowerCase();
  const hits = [];
  for (const t of terms) { const q = t.toLowerCase(); if (!q) continue; let i = low.indexOf(q); while (i >= 0) { hits.push([i, i + q.length]); i = low.indexOf(q, i + q.length); } }
  hits.sort((a, b) => a[0] - b[0]);
  let pos = 0;
  for (const [a, b] of hits) {
    if (a < pos) continue;
    el.appendChild(document.createTextNode(text.slice(pos, a)));
    const m = document.createElement("mark"); m.textContent = text.slice(a, b); el.appendChild(m);
    pos = b;
  }
  el.appendChild(document.createTextNode(text.slice(pos)));
}
function sentenceRows(container, sents, { label, onClick, terms, minConf = 0 } = {}) {
  const frag = document.createDocumentFragment();
  let hidden = 0;
  for (const s of sents) {
    if (minConf > 0 && !(s.c != null && s.c >= minConf)) { hidden++; continue; }
    const row = document.createElement("div");
    row.className = "sent"; row.dataset.t = s.s; row.dataset.e = s.e;
    row.title = `${fmtUtc(s.s)} → ${fmtUtc(s.e)} · ${s.pass} — click to play from here`;
    row.innerHTML = `<span class="t"><span class="tt"></span><small></small></span><span class="txt" dir="auto"></span><span class="c"></span>`;
    row.querySelector(".tt").textContent = label ? label(s) : fmtLocal(s.s);
    row.querySelector("small").textContent = s.pass === "refined" ? "✓ re-checked" : "";
    if (s.pass === "refined") row.querySelector("small").title = "Second, more careful pass done";
    const txt = row.querySelector(".txt");
    if (terms && terms.length) highlightTerms(txt, s.text, terms);
    else s.ws.forEach((w, i) => {
      if (i) txt.appendChild(document.createTextNode(" "));
      const sp = document.createElement("span"); sp.textContent = wordOf(w)[0];
      if (wordOf(w)[1] != null) sp.dataset.s = wordOf(w)[1];
      if (typeof w.p === "number" && w.p < LOW_WORD) { sp.className = "lowp"; sp.title = `word confidence ${Math.round(w.p * 100)}%`; }
      txt.appendChild(sp);
    });
    const c = row.querySelector(".c");
    c.textContent = s.c == null ? "" : `${Math.round(s.c * 100)}%`; c.className = "c " + confClass(s.c);
    if (onClick) row.onclick = () => onClick(s, row);
    frag.appendChild(row);
  }
  container.replaceChildren(frag);
  return hidden;
}
// Highlight the sentence (and word) being played at epoch t; keep it centred in the transcript box
// (scrolls only the box, never the page). opts.scroll=false: highlight only (user is reading).
function markPlaying(container, t, opts = {}) {
  const scroll = opts.scroll !== false;
  let cur = null;
  for (const r of container.querySelectorAll(".sent")) { if (parseFloat(r.dataset.t) <= t + 0.05) cur = r; else break; }
  for (const r of container.querySelectorAll(".sent.hi")) if (r !== cur) r.classList.remove("hi");
  for (const w of container.querySelectorAll(".cw")) w.classList.remove("cw");
  if (!cur) return null;
  cur.classList.add("hi");
  const spoken = t <= parseFloat(cur.dataset.e) + 1.0;   // video is inside this sentence
  cur.classList.toggle("waiting", !spoken);
  if (spoken) {
    let cw = null;
    for (const w of cur.querySelectorAll("[data-s]")) { if (parseFloat(w.dataset.s) <= t) cw = w; else break; }
    if (cw) cw.classList.add("cw");
  }
  if (scroll) {
    const box = container.getBoundingClientRect(), rb = cur.getBoundingClientRect();
    const delta = rb.top - box.top - Math.max(0, (box.height - rb.height) / 2);
    if (Math.abs(delta) > 6) container.scrollTo({ top: container.scrollTop + delta, behavior: opts.instant ? "auto" : "smooth" });
  }
  return cur;
}

// ---- HLS (archive VOD playlists carry PROGRAM-DATE-TIME)
function hlsPlay(video, src, onMsg) {
  if (window.Hls && Hls.isSupported()) {
    const h = new Hls({ maxBufferLength: 30 });
    h.on(Hls.Events.ERROR, (_, d) => { if (d.fatal) onMsg && onMsg("video error: " + (d.details || d.type)); });
    h.loadSource(src); h.attachMedia(video);
    return h;
  }
  if (video.canPlayType("application/vnd.apple.mpegurl")) { video.src = src; return null; }
  onMsg && onMsg("this browser cannot play HLS");
  return null;
}
function hlsTimeToEpoch(h, video) {   // current playback position → epoch seconds
  if (h && h.playingDate) return h.playingDate.getTime() / 1000;
  if (video.getStartDate && !isNaN(video.getStartDate().getTime())) return video.getStartDate().getTime() / 1000 + video.currentTime;
  return null;
}
function hlsSeekEpoch(h, video, t) {
  const ms = t * 1000;
  if (h && h.levels && h.levels.length) {
    const lv = h.levels[Math.max(h.currentLevel, 0)];
    for (const f of (lv && lv.details && lv.details.fragments) || []) {
      if (f.programDateTime == null) continue;
      if (ms >= f.programDateTime && ms < f.programDateTime + f.duration * 1000) { video.currentTime = f.start + (ms - f.programDateTime) / 1000; return true; }
    }
    return false;
  }
  if (video.getStartDate && !isNaN(video.getStartDate().getTime())) { video.currentTime = (ms - video.getStartDate().getTime()) / 1000; return true; }
  return false;
}
async function channelList() {   // live channels for selects: [{id, name}]
  try { return (await api("/channels")).map(c => ({ id: c.id, name: c.name || c.id, enabled: c.enabled })); } catch (e) { return []; }
}
