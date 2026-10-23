'use strict';
// One map, independent layers. Every layer reads from the payload the build inlined;
// the page itself makes no data requests, only tile requests.
const D = JSON.parse(document.getElementById('map-data').textContent);
const places = D.dispatch.places;
const cities = D.cities;
const $ = sel => document.querySelector(sel);

// ---------- Storage: per-viewer conveniences only, every access guarded ----------
const STATE_KEY = 'maybemap_state_v1';
const NOTES_KEY = 'maybemap_notes_v1';
function readStore(key) { try { return JSON.parse(localStorage.getItem(key) || 'null'); } catch (e) { return null; } }
function writeStore(key, value) { try { localStorage.setItem(key, JSON.stringify(value)); return true; } catch (e) { return false; } }

const zoneNames = D.zones.features.map(f => f.properties.label);
const regionNames = D.regions.map(r => r.name);
const stagingTypes = D.staging.map(s => s.type);
const allOn = names => Object.fromEntries(names.map(n => [n, true]));

const DEFAULTS = {
  layers: { zones: true, outlines: true, dots: false, dispatch: false },
  zonesOn: allOn(zoneNames),
  zoneLabels: true,
  zoneOpacity: 'medium',
  buckets: D.buckets.map(() => true),
  mix: allOn(D.serviceTypes.map(t => t.key)),
  regionsOn: allOn(regionNames),
  routeMode: 'neighbors',
  routeCount: 1,
  techNames: true,
  townNames: false,
  stagingOn: Object.fromEntries(stagingTypes.map(t => [t, false])),
  basemap: 'gray',
  placeLabels: true,
};

// Saved state only overrides keys it still has the right shape for, so a zone or
// region added since the last visit starts visible rather than undefined.
function mergeState(saved) {
  const s = structuredClone(DEFAULTS);
  if (!saved || typeof saved !== 'object') return s;
  for (const [k, v] of Object.entries(DEFAULTS)) {
    const got = saved[k];
    if (got === undefined || got === null) continue;
    if (Array.isArray(v)) { if (Array.isArray(got) && got.length === v.length) s[k] = got.map(Boolean); }
    else if (typeof v === 'object') { for (const name of Object.keys(v)) if (typeof got[name] === 'boolean') s[k][name] = got[name]; }
    else if (typeof got === typeof v) s[k] = got;
  }
  return s;
}
const state = mergeState(readStore(STATE_KEY));
function persist() { writeStore(STATE_KEY, state); }

let notes = readStore(NOTES_KEY);
if (!notes || typeof notes !== 'object' || Array.isArray(notes)) notes = {};
const getNote = key => (typeof notes[key] === 'string' ? notes[key] : '');

// ---------- Lookups ----------
const keyOf = (name, st) => name + '|' + st;
const cityByKey = new Map(cities.map(c => [keyOf(c.name, c.state), c]));
const placeIdxByKey = new Map(places.map((p, i) => [keyOf(p.name, p.state), i]));
const zoneByLabel = new Map(D.zones.features.map(f => [f.properties.label, f.properties]));
const regionColor = new Map(D.regions.map(r => [r.name, r.color]));
const allKeys = [...new Set([...cityByKey.keys(), ...placeIdxByKey.keys()])];
let selectedKey = null;

function townInfo(key) {
  const city = cityByKey.get(key) || null;
  const pi = placeIdxByKey.has(key) ? placeIdxByKey.get(key) : -1;
  const place = pi >= 0 ? places[pi] : null;
  const src = city || place;
  return { key, city, pi, place, name: src.name, state: src.state, lat: src.lat, lon: src.lon, techs: place ? place.techs : [] };
}

function esc(s) {
  return String(s).replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;').replaceAll('"', '&quot;').replaceAll("'", '&#39;');
}
function el(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined) e.textContent = text;
  return e;
}
const miles = m => Math.round(m / 1609.344 * 10) / 10;
function duration(s) { const min = Math.round(s / 60); return min >= 60 ? Math.floor(min / 60) + 'h ' + String(min % 60).padStart(2, '0') + 'm' : min + ' min'; }
const fmt = n => (n === null || n === undefined ? '—' : Number(n).toLocaleString());
const mapsPlace = t => 'https://www.google.com/maps/search/?api=1&query=' + encodeURIComponent(t.lat + ',' + t.lon);
function directions(a, b) {
  return 'https://www.google.com/maps/dir/?api=1&origin=' + encodeURIComponent(a.lat + ',' + a.lon) +
    '&destination=' + encodeURIComponent(b.lat + ',' + b.lon) + '&travelmode=driving';
}

// ---------- Filters shared by dots and outlines ----------
function bucketIndex(n) {
  if (n === null || n === undefined) return -1;
  return D.buckets.findIndex(b => n >= b.lo && (b.hi === null || n <= b.hi));
}
function cityPasses(c) {
  if (!state.regionsOn[c.region]) return false;
  // A town the export gave no count, or no known service mix, is never hidden by that filter
  const b = bucketIndex(c.customers);
  if (b >= 0 && !state.buckets[b]) return false;
  if (c.mixKey && state.mix[c.mixKey] === false) return false;
  return true;
}

// ---------- Map, panes and basemaps ----------
const map = L.map('map', { zoomControl: true, zoomSnap: 0.25 });
// Panes stack finest-on-top, so hover and click always resolve to the most
// specific thing under the cursor: pin, then dot, then outline, then zone.
for (const [name, z] of [['zones', 405], ['outlines', 415], ['dots', 425], ['routes', 435], ['basemapLabels', 445], ['highlight', 450]]) {
  const pane = map.createPane(name);
  pane.style.zIndex = z;
  if (name === 'basemapLabels' || name === 'highlight') pane.style.pointerEvents = 'none';
}
const ESRI = 'https://server.arcgisonline.com/ArcGIS/rest/services/';
const BASEMAPS = {
  gray: {
    base: L.tileLayer(ESRI + 'Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}', { attribution: 'Tiles © Esri', maxNativeZoom: 16, maxZoom: 19 }),
    labels: L.tileLayer(ESRI + 'Canvas/World_Light_Gray_Reference/MapServer/tile/{z}/{y}/{x}', { pane: 'basemapLabels', maxNativeZoom: 16, maxZoom: 19 }),
  },
  satellite: {
    base: L.tileLayer(ESRI + 'World_Imagery/MapServer/tile/{z}/{y}/{x}', { attribution: 'Imagery © Esri and contributors', maxZoom: 19 }),
    labels: L.tileLayer(ESRI + 'Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}', { pane: 'basemapLabels', maxZoom: 19 }),
  },
  // OSM bakes its place names into the tiles, so there is no overlay to switch
  streets: {
    base: L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', { attribution: '© OpenStreetMap contributors', maxZoom: 19 }),
    labels: null,
  },
};

// Open on the zones: that is the territory, and towns outside it are context
const zonesBounds = L.geoJSON(D.zones).getBounds();
const startBounds = zonesBounds.isValid() ? zonesBounds : L.latLngBounds(cities.map(c => [c.lat, c.lon]));
map.fitBounds(startBounds, { paddingTopLeft: [20, 140], paddingBottomRight: [20, 20] });

// ---------- Zones ----------
const OPACITY = { light: 0.18, medium: 0.32, strong: 0.5 };
const zoneLayers = new Map();
const zoneLabelLayers = new Map();
for (const f of D.zones.features) {
  const p = f.properties;
  const layer = L.geoJSON(f, { pane: 'zones' });
  layer.bindTooltip(() => '<b>' + esc(p.label) + '</b><small>' + zoneCityCount(p.label) + ' towns</small>', { sticky: true, className: 'hover-tip' });
  zoneLayers.set(p.label, layer);
  zoneLabelLayers.set(p.label, L.marker(p.anchor, {
    pane: 'basemapLabels', interactive: false, keyboard: false,
    icon: L.divIcon({ className: 'zone-label', html: '<span style="background:' + p.edge + 'e6">' + esc(p.label) + '</span>', iconSize: [0, 0] }),
  }));
}
function zoneCityCount(label) { return cities.filter(c => c.zone === label).length; }

function renderZones() {
  for (const [label, layer] of zoneLayers) {
    const p = zoneByLabel.get(label);
    const on = state.layers.zones && state.zonesOn[label];
    layer.setStyle({ color: p.edge, weight: 1.6, opacity: 0.95, fillColor: p.color, fillOpacity: OPACITY[state.zoneOpacity] });
    toggle(layer, on);
    toggle(zoneLabelLayers.get(label), on && state.zoneLabels);
  }
}
function toggle(layer, on) {
  if (on && !map.hasLayer(layer)) layer.addTo(map);
  else if (!on && map.hasLayer(layer)) map.removeLayer(layer);
}

// ---------- City outlines and dots ----------
function hoverTip(c) {
  const note = getNote(keyOf(c.name, c.state)) ? ' 📝' : '';
  const where = [c.regioned ? c.region : 'No region', c.zone].filter(Boolean).join(' · ');
  return '<b>' + esc(c.name) + ', ' + esc(c.state) + note + '</b><small>' + esc(where) + (c.customers !== null ? ' · ' + c.customers + ' customers' : '') + '</small>';
}
const outlineLayers = new Map();
for (const f of D.outlines.features) {
  const c = cities[f.properties.city];
  const layer = L.geoJSON(f, { pane: 'outlines' });
  layer.bindTooltip(() => hoverTip(c), { sticky: true, className: 'hover-tip' });
  layer.on('click', () => select(keyOf(c.name, c.state), false));
  outlineLayers.set(c.id, layer);
}
function outlineColor(c) {
  // Against a zone fill, a darker shade of that zone reads as part of it; on its own
  // the region color is the more useful signal
  if (state.layers.zones && c.zone && state.zonesOn[c.zone]) return zoneByLabel.get(c.zone).edge;
  return c.color;
}
function renderOutlines() {
  for (const [id, layer] of outlineLayers) {
    const c = cities[id];
    const color = outlineColor(c);
    layer.setStyle({ color, weight: 1.2, opacity: 0.95, fillColor: color, fillOpacity: 0.5 });
    toggle(layer, state.layers.outlines && cityPasses(c));
  }
}

const DOT_RADIUS = [5, 6, 7.5, 9];
const dotLayers = cities.map(c => {
  const b = bucketIndex(c.customers);
  const m = L.circleMarker([c.lat, c.lon], {
    pane: 'dots', radius: b >= 0 ? DOT_RADIUS[Math.min(b, DOT_RADIUS.length - 1)] : 4.5,
    color: '#fff', weight: 1.2, fillColor: c.color, fillOpacity: 0.92,
  });
  m.bindTooltip(() => hoverTip(c), { className: 'hover-tip', direction: 'top', offset: [0, -6] });
  m.on('click', () => select(keyOf(c.name, c.state), false));
  return m;
});
function renderDots() {
  cities.forEach((c, i) => toggle(dotLayers[i], state.layers.dots && cityPasses(c)));
}

// ---------- Dispatch ----------
function nearest(i, n, techsOnly) {
  const out = [];
  for (let j = 0; j < places.length; j++) {
    if (techsOnly ? !places[j].techs.length : j === i) continue;
    out.push({ j, m: D.dispatch.meters[i][j], s: D.dispatch.seconds[i][j] });
  }
  return out.sort((a, b) => a.s - b.s || a.m - b.m).slice(0, n);
}
const pinLayer = L.layerGroup();
const routeLayer = L.layerGroup();
const pins = places.map((p, i) => {
  const tech = p.techs.length > 0;
  const mk = L.marker([p.lat, p.lon], {
    icon: L.divIcon({ className: '', html: '<div class="pin' + (tech ? ' tech' : '') + '"></div>', iconSize: tech ? [21, 21] : [17, 17], iconAnchor: tech ? [10, 10] : [8, 8] }),
    zIndexOffset: tech ? 500 : 0,
  }).addTo(pinLayer);
  mk.on('click', () => select(keyOf(p.name, p.state), false));
  return mk;
});
function pinLabel(p) {
  const parts = [];
  const tech = p.techs.length && state.techNames;
  if (tech) parts.push('<span class="name">' + esc(p.techs.join(', ')) + '</span>');
  if (state.townNames || (tech && !compactLabels())) parts.push('<span class="twn">' + esc(p.name) + '</span>');
  return parts.join(' <span>—</span> ');
}
function pinTitle(p) {
  const note = getNote(keyOf(p.name, p.state)) ? ' 📝' : '';
  return (p.techs.length ? p.techs.join(', ') + ' — ' : '') + p.name + note;
}
// Permanent labels are placed greedily: techs first, each taking the first side of
// its pin (right, left, above, below) that does not cover a label already placed.
// A tech is always labeled, on its least-crowded side if every side is taken; a
// plain town that finds no room keeps a hover label instead.
const LABEL_SIDES = [
  { direction: 'right', offset: [8, 0], box: (pt, w, h) => [pt.x + 10, pt.y - h / 2, w, h] },
  { direction: 'left', offset: [-8, 0], box: (pt, w, h) => [pt.x - 10 - w, pt.y - h / 2, w, h] },
  { direction: 'top', offset: [0, -10], box: (pt, w, h) => [pt.x - w / 2, pt.y - 12 - h, w, h] },
  { direction: 'bottom', offset: [0, 10], box: (pt, w, h) => [pt.x - w / 2, pt.y + 12, w, h] },
];
const overlap = (a, b) => Math.max(0, Math.min(a[0] + a[2], b[0] + b[2]) - Math.max(a[0], b[0])) *
  Math.max(0, Math.min(a[1] + a[3], b[1] + b[3]) - Math.max(a[1], b[1]));
// Zoomed out to the whole state, the western techs sit a few pixels apart, so
// their labels drop the town name until there is room for it
const compactLabels = () => map.getZoom() < 8;
function labelText(p) {
  if (p.techs.length && state.techNames) return p.techs.join(', ') + (compactLabels() ? '' : ' — ' + p.name);
  return p.name;
}
function renderPins() {
  const placed = [];
  const order = places.map((p, i) => i).sort((a, b) => places[b].techs.length - places[a].techs.length);
  for (const i of order) {
    const p = places[i];
    const mk = pins[i];
    const node = mk.getElement && mk.getElement();
    if (node && node.firstChild) node.firstChild.classList.toggle('selected', selectedKey === keyOf(p.name, p.state));
    if (mk.getTooltip()) mk.unbindTooltip();
    const label = pinLabel(p);
    let side = null;
    if (label) {
      const pt = map.latLngToContainerPoint([p.lat, p.lon]);
      const w = labelText(p).length * 6.8 + 20, h = 24;
      const crowding = LABEL_SIDES.map(sd => placed.reduce((sum, r) => sum + overlap(r, sd.box(pt, w, h)), 0));
      const best = crowding.indexOf(Math.min(...crowding));
      if (crowding[best] === 0 || p.techs.length) side = LABEL_SIDES[best];
      if (side) placed.push(side.box(pt, w, h));
    }
    if (side) {
      mk.bindTooltip(label, { permanent: true, direction: side.direction, offset: side.offset, className: p.techs.length ? 'techlabel' : 'marketlabel', opacity: 0.97 });
    } else {
      mk.bindTooltip(esc(pinTitle(p)), { direction: 'top', offset: [0, -8], className: 'hover-tip' });
    }
  }
}
function addConnector(i, j, theme) {
  const a = places[i], b = places[j];
  const m = D.dispatch.meters[i][j], s = D.dispatch.seconds[i][j];
  const color = theme === 'staff' ? '#ffa63a' : theme === 'selected' ? '#ffcf3d' : '#55ccea';
  const line = L.polyline([[a.lat, a.lon], [b.lat, b.lon]], {
    pane: 'routes', color, weight: theme === 'selected' ? 5 : 2.7, opacity: 0.9, dashArray: theme === 'selected' ? null : '6 7',
  }).addTo(routeLayer);
  line.bindPopup('<b>' + esc(a.name) + ' → ' + esc(b.name) + '</b><br>' + miles(m) + ' mi · ' + duration(s) +
    '<br><small>Connector, not the road shape</small><br><a href="' + directions(a, b) + '" target="_blank" rel="noopener">Drive directions in Google Maps ↗</a>');
  L.tooltip({ permanent: true, direction: 'center', className: 'edge-label', interactive: false })
    .setLatLng([(a.lat + b.lat) / 2, (a.lon + b.lon) / 2]).setContent(miles(m) + ' mi · ' + duration(s)).addTo(routeLayer);
}
function renderRoutes() {
  routeLayer.clearLayers();
  if (!state.layers.dispatch || state.routeMode === 'none') return;
  const seen = new Set();
  const add = (i, j, theme) => {
    if (i === j) return;
    const k = Math.min(i, j) + '-' + Math.max(i, j);
    if (seen.has(k)) return;
    seen.add(k);
    addConnector(i, j, theme);
  };
  if (state.routeMode === 'selected') {
    const pi = selectedKey && placeIdxByKey.has(selectedKey) ? placeIdxByKey.get(selectedKey) : -1;
    if (pi >= 0) for (const e of nearest(pi, 3, false)) add(pi, e.j, 'selected');
    return;
  }
  const staff = state.routeMode === 'staff';
  for (let i = 0; i < places.length; i++) for (const e of nearest(i, staff ? 1 : state.routeCount, staff)) add(i, e.j, staff ? 'staff' : 'market');
}
function updateEdgeLabelVisibility() {
  // Statewide, a label on every connector is a wall of text; they appear once
  // zoomed in, and always for the handful of lines the selected town draws
  map.getContainer().classList.toggle('hide-edge-labels', map.getZoom() < 9 && state.routeMode !== 'selected');
}
map.on('zoomend', () => { updateEdgeLabelVisibility(); if (state.layers.dispatch) renderPins(); });
function renderDispatch() {
  toggle(pinLayer, state.layers.dispatch);
  toggle(routeLayer, state.layers.dispatch);
  renderPins();
  renderRoutes();
  updateEdgeLabelVisibility();
}

// ---------- Staging ----------
const STAGING_COLORS = { generator: '#b65d03', outage: '#be2b43', warehouse: '#5d51b9' };
const FALLBACK_STAGING = ['#00796b', '#6d4c41', '#c2185b', '#455a64'];
const stagingColor = (t, i) => STAGING_COLORS[t.toLowerCase()] || FALLBACK_STAGING[i % FALLBACK_STAGING.length];
const stagingLayer = L.layerGroup().addTo(map);
function renderStaging() {
  stagingLayer.clearLayers();
  const byPlace = new Map();
  D.staging.forEach((s, i) => {
    if (!state.stagingOn[s.type]) return;
    for (const pi of s.places) {
      if (!byPlace.has(pi)) byPlace.set(pi, []);
      byPlace.get(pi).push({ type: s.type, color: stagingColor(s.type, i) });
    }
  });
  for (const [pi, tags] of byPlace) {
    const p = places[pi];
    const html = '<div class="stage-flag-list">' + tags.map(t =>
      '<div class="stage-flag" style="background:' + t.color + '"><span class="stage-mini">' + esc(t.type.slice(0, 3).toUpperCase()) + '</span> ' + esc(t.type) + ' staging</div>').join('') + '</div>';
    L.marker([p.lat, p.lon], { icon: L.divIcon({ className: 'stage-anchor', html, iconSize: [0, 0], iconAnchor: [0, 0] }), interactive: false, keyboard: false, zIndexOffset: 1100 }).addTo(stagingLayer);
  }
}

// ---------- Basemap ----------
function renderBasemap() {
  for (const [name, bm] of Object.entries(BASEMAPS)) {
    toggle(bm.base, name === state.basemap);
    if (bm.labels) toggle(bm.labels, name === state.basemap && state.placeLabels);
  }
}

// ---------- Selection ----------
const highlight = L.circleMarker([0, 0], { pane: 'highlight', radius: 14, color: '#ffcf3d', weight: 4, fill: false, interactive: false });
function select(key, fly) {
  selectedKey = key;
  const t = townInfo(key);
  highlight.setLatLng([t.lat, t.lon]);
  if (!map.hasLayer(highlight)) highlight.addTo(map);
  if (fly) map.flyTo([t.lat, t.lon], Math.max(map.getZoom(), 10), { duration: 0.5 });
  renderCard();
  renderPins();
  if (state.routeMode === 'selected') renderRoutes();
  updateEdgeLabelVisibility();
  $('#selectedSection').hidden = false;
  $('.sidebar').scrollTo({ top: 0, behavior: 'smooth' });
}
function clearSelection() {
  selectedKey = null;
  if (map.hasLayer(highlight)) map.removeLayer(highlight);
  $('#selectedSection').hidden = true;
  renderPins();
  if (state.routeMode === 'selected') renderRoutes();
}

function fact(value, label) {
  const f = el('div', 'fact');
  f.append(el('b', '', value), el('small', '', label));
  return f;
}
function rowEntry(rank, t, from, e, techRow) {
  const row = el('button', 'rowentry');
  row.type = 'button';
  const num = el('div', 'num' + (techRow ? ' tech' : ''), String(rank));
  const detail = el('div', 'detail');
  const dest = places[e.j];
  detail.append(el('b', '', techRow ? dest.techs.join(', ') : dest.name),
    el('small', '', techRow ? dest.name : dest.techs.length ? 'Tech: ' + dest.techs.join(', ') : 'Dispatch town'));
  const val = el('div', 'value');
  val.append(el('b', '', e.j === from ? 'Here' : miles(e.m) + ' mi'), el('small', '', e.j === from ? 'Based in this town' : duration(e.s)));
  if (e.j !== from) {
    const a = el('a', '', 'Google route ↗');
    a.href = directions(t, dest); a.target = '_blank'; a.rel = 'noopener';
    a.addEventListener('click', ev => ev.stopPropagation());
    val.append(a);
  }
  row.append(num, detail, val);
  row.addEventListener('click', () => select(keyOf(dest.name, dest.state), true));
  return row;
}
function renderCard() {
  const box = $('#selectedCard');
  box.replaceChildren();
  if (!selectedKey) return;
  const t = townInfo(selectedKey);
  const c = t.city;
  box.append(el('h2', 'card-title', t.name + ', ' + t.state));
  const sub = el('p', 'card-sub');
  const where = c ? [c.county ? c.county + ' County' : null, c.regioned ? c.region : 'No region', c.zone].filter(Boolean).join(' · ')
    : 'Tech base · not in the city export';
  sub.append(document.createTextNode(where));
  if (t.techs.length) { sub.append(document.createTextNode(' · ')); sub.append(el('span', 'tech', 'Tech: ' + t.techs.join(', '))); }
  box.append(sub);

  if (c) {
    const facts = el('div', 'facts');
    facts.append(fact(fmt(c.customers), 'Customers'), fact(fmt(c.active), 'Active addresses'), fact(fmt(c.tasks), 'Tasks, 6 months'), fact(c.mix || '—', 'Service mix'));
    if (c.residential !== null && c.business !== null) facts.append(fact(fmt(c.residential), 'Residential'), fact(fmt(c.business), 'Business'));
    box.append(facts);
  }

  if (t.pi >= 0) {
    box.append(el('div', 'field-label', '3 closest dispatch towns · drive time'));
    nearest(t.pi, 3, false).forEach((e, i) => box.append(rowEntry(i + 1, t, t.pi, e, false)));
    const techs = nearest(t.pi, 2, true);
    if (techs.length) {
      box.append(el('div', 'field-label', techs.length === 1 ? 'Nearest technician' : '2 nearest technicians'));
      techs.forEach((e, i) => box.append(rowEntry(i + 1, t, t.pi, e, true)));
    }
  } else {
    box.append(el('p', 'route-note', 'Not in the dispatch set: only towns with a region, and tech bases, have drive times.'));
  }

  const links = el('div', 'btnrow');
  links.style.marginTop = '10px';
  const g = el('a', 'action secondary', 'Open in Google Maps ↗');
  g.href = mapsPlace(t); g.target = '_blank'; g.rel = 'noopener';
  links.append(g);
  box.append(links);

  const label = el('label', 'field-label', 'Notes for this town');
  label.htmlFor = 'townNotes';
  label.style.display = 'block';
  const area = el('textarea');
  area.id = 'townNotes';
  area.placeholder = 'Shop access, special equipment, local contact, coverage tips… (no passwords)';
  area.value = getNote(selectedKey);
  const saveState = el('div', 'save-state', 'Notes save automatically in this browser.');
  area.addEventListener('input', () => {
    if (area.value) notes[selectedKey] = area.value; else delete notes[selectedKey];
    saveState.textContent = writeStore(NOTES_KEY, notes) ? 'Saved in this browser · Export CSV to share' : 'Browser storage unavailable · Export notes CSV to keep changes';
  });
  box.append(label, area, saveState);
}

// ---------- Search ----------
const searchIndex = allKeys.map(key => {
  const t = townInfo(key);
  const c = t.city;
  return { key, t, text: [t.name, t.state, t.techs.join(' '), c ? c.region : '', c && c.zone ? c.zone : ''].join(' ').toLowerCase() };
});
function renderDirectory() {
  const q = $('#townFilter').value.trim().toLowerCase();
  const box = $('#directory');
  box.replaceChildren();
  if (!q) return;
  const hits = searchIndex.filter(s => s.text.includes(q) || getNote(s.key).toLowerCase().includes(q))
    .sort((a, b) => Number(b.t.techs.length > 0) - Number(a.t.techs.length > 0) || a.t.name.localeCompare(b.t.name));
  for (const s of hits.slice(0, 30)) {
    const btn = el('button', 'town' + (s.key === selectedKey ? ' active' : ''));
    btn.type = 'button';
    btn.append(el('i', 'dot' + (s.t.techs.length ? ' tech' : '')),
      el('span', '', (s.t.techs.length ? s.t.techs.join(', ') + ' — ' : '') + s.t.name + (getNote(s.key) ? ' 📝' : '')));
    btn.addEventListener('click', () => select(s.key, true));
    box.append(btn);
  }
  if (hits.length > 30) box.append(el('div', 'more', (hits.length - 30) + ' more · keep typing to narrow it down'));
  if (!hits.length) box.append(el('div', 'more', 'No town, tech or region matches.'));
}
$('#townFilter').addEventListener('input', renderDirectory);

// ---------- Sidebar controls ----------
function swatchToggle(color, title, hint, checked, onChange, letter) {
  const label = el('label', 'swatch-toggle');
  label.style.setProperty('--stage', color);
  const sw = el('span', 'swatch', letter || '');
  sw.setAttribute('aria-hidden', 'true');
  const text = el('span');
  text.append(el('b', '', title));
  if (hint) text.append(el('small', '', hint));
  const input = el('input');
  input.type = 'checkbox';
  input.setAttribute('role', 'switch');
  input.checked = checked;
  input.setAttribute('aria-label', title);
  input.addEventListener('change', () => onChange(input.checked));
  label.append(sw, text, input);
  return { label, input };
}
const zoneInputs = new Map();
for (const f of D.zones.features) {
  const p = f.properties;
  const { label, input } = swatchToggle(p.color, p.label, zoneCityCount(p.label) + ' towns', state.zonesOn[p.label], on => { state.zonesOn[p.label] = on; update(); });
  zoneInputs.set(p.label, input);
  $('#zoneToggles').append(label);
}
const regionInputs = new Map();
for (const r of D.regions) {
  const n = cities.filter(c => c.region === r.name).length;
  const { label, input } = swatchToggle(r.color, r.name, n + (n === 1 ? ' town' : ' towns'), state.regionsOn[r.name], on => { state.regionsOn[r.name] = on; update(); });
  regionInputs.set(r.name, input);
  $('#regionToggles').append(label);
}
const stagingInputs = new Map();
D.staging.forEach((s, i) => {
  const where = s.places.map(pi => places[pi].name).join(' · ');
  const { label, input } = swatchToggle(stagingColor(s.type, i), s.type + ' staging', where, state.stagingOn[s.type], on => { state.stagingOn[s.type] = on; update(); }, s.type[0].toUpperCase());
  stagingInputs.set(s.type, input);
  $('#stagingToggles').append(label);
});
if (!D.staging.length) $('#stagingSection').hidden = true;

D.buckets.forEach((b, i) => {
  const n = cities.filter(c => bucketIndex(c.customers) === i).length;
  const btn = el('button');
  btn.type = 'button';
  btn.append(document.createTextNode(b.label), el('small', '', n + ' towns'));
  btn.addEventListener('click', () => { state.buckets[i] = !state.buckets[i]; update(); });
  $('#bucketButtons').append(btn);
});
for (const t of D.serviceTypes) {
  const n = cities.filter(c => c.mixKey === t.key).length;
  const btn = el('button');
  btn.type = 'button';
  btn.dataset.value = t.key;
  btn.append(document.createTextNode(t.label), el('small', '', n + ' towns'));
  btn.addEventListener('click', () => { state.mix[t.key] = !state.mix[t.key]; update(); });
  $('#mixButtons').append(btn);
}

for (const name of Object.keys(state.layers)) {
  $('#layer-' + name).addEventListener('change', e => { state.layers[name] = e.target.checked; update(); });
}
$('#zoneLabels').addEventListener('change', e => { state.zoneLabels = e.target.checked; update(); });
$('#showTechNames').addEventListener('change', e => { state.techNames = e.target.checked; update(); });
$('#showTownNames').addEventListener('change', e => { state.townNames = e.target.checked; update(); });
$('#placeLabels').addEventListener('change', e => { state.placeLabels = e.target.checked; update(); });
$('#count').addEventListener('change', e => { state.routeCount = Number(e.target.value); update(); });
document.querySelectorAll('#zoneOpacity button').forEach(b => b.addEventListener('click', () => { state.zoneOpacity = b.dataset.value; update(); }));
document.querySelectorAll('#basemapButtons button').forEach(b => b.addEventListener('click', () => { state.basemap = b.dataset.value; update(); }));
document.querySelectorAll('#modeButtons button').forEach(b => b.addEventListener('click', () => {
  state.routeMode = b.dataset.mode;
  if (state.routeMode !== 'none') state.layers.dispatch = true;
  update();
}));
document.querySelectorAll('[data-all]').forEach(b => b.addEventListener('click', () => {
  const on = b.dataset.on === '1';
  const target = { zones: state.zonesOn, regions: state.regionsOn, staging: state.stagingOn }[b.dataset.all];
  for (const k of Object.keys(target)) target[k] = on;
  update();
}));
$('#clearSelection').addEventListener('click', clearSelection);

// Presets only flip the master switches; every option underneath is left as it was
const PRESETS = {
  territory: { zones: true, outlines: true, dots: false, dispatch: false },
  coverage: { zones: true, outlines: false, dots: true, dispatch: false },
  dispatch: { zones: true, outlines: false, dots: false, dispatch: true },
};
document.querySelectorAll('[data-preset]').forEach(b => b.addEventListener('click', () => {
  Object.assign(state.layers, PRESETS[b.dataset.preset]);
  update();
}));

function setSeg(sel, isActive) {
  document.querySelectorAll(sel + ' button').forEach((b, i) => {
    const on = isActive(b, i);
    b.classList.toggle('active', on);
    b.setAttribute('aria-pressed', String(on));
  });
}
function renderControls() {
  for (const name of Object.keys(state.layers)) {
    $('#layer-' + name).checked = state.layers[name];
    document.querySelector('.layer[data-layer="' + name + '"]').classList.toggle('off', !state.layers[name]);
  }
  for (const [k, input] of zoneInputs) input.checked = state.zonesOn[k];
  for (const [k, input] of regionInputs) input.checked = state.regionsOn[k];
  for (const [k, input] of stagingInputs) input.checked = state.stagingOn[k];
  $('#zoneLabels').checked = state.zoneLabels;
  $('#showTechNames').checked = state.techNames;
  $('#showTownNames').checked = state.townNames;
  $('#placeLabels').checked = state.placeLabels;
  $('#placeLabels').disabled = !BASEMAPS[state.basemap].labels;
  $('#count').value = String(state.routeCount);
  $('#count').disabled = state.routeMode !== 'neighbors';
  setSeg('#zoneOpacity', b => b.dataset.value === state.zoneOpacity);
  setSeg('#basemapButtons', b => b.dataset.value === state.basemap);
  setSeg('#modeButtons', b => b.dataset.mode === state.routeMode);
  setSeg('#bucketButtons', (b, i) => state.buckets[i]);
  setSeg('#mixButtons', b => state.mix[b.dataset.value]);
  for (const [name, preset] of Object.entries(PRESETS)) {
    const match = Object.entries(preset).every(([k, v]) => state.layers[k] === v);
    document.querySelector('[data-preset="' + name + '"]').classList.toggle('active', match);
  }

  const visibleZones = zoneNames.filter(z => state.zonesOn[z]).length;
  const filtered = cities.filter(cityPasses);
  const techCount = places.reduce((n, p) => n + p.techs.length, 0);
  $('#zonesHint').textContent = visibleZones + ' of ' + zoneNames.length + ' zones shown';
  $('#outlinesHint').textContent = D.outlines.features.filter(f => cityPasses(cities[f.properties.city])).length + ' of ' + D.outlines.features.length + ' city limits match the filters';
  $('#dotsHint').textContent = filtered.length + ' of ' + cities.length + ' towns match the filters';
  $('#dispatchHint').textContent = places.length + ' towns · ' + techCount + ' techs';
  $('#techNamesHint').textContent = state.techNames ? 'ON · ' + techCount + ' technician labels' : 'OFF · tech pins stay orange';
  $('#townNamesHint').textContent = state.townNames ? 'ON · every dispatch town labeled' : 'OFF · hover a pin for its name';
  $('#placeLabelsHint').textContent = BASEMAP_LABEL_HINT[state.basemap](state.placeLabels);

  const stats = $('#stats');
  stats.replaceChildren();
  for (const [n, label] of [[filtered.length, 'towns'], [D.regions.filter(r => r.name !== D.unassignedLabel).length, 'regions'], [places.length, 'dispatch'], [techCount, 'techs']]) {
    const s = el('span');
    s.append(el('b', '', String(n)), document.createTextNode(label));
    stats.append(s);
  }
}
const BASEMAP_LABEL_HINT = {
  gray: on => (on ? 'ON · Esri place names over the map' : 'OFF · only this map\'s own labels'),
  satellite: on => (on ? 'ON · boundaries and places over the imagery' : 'OFF · bare imagery'),
  streets: () => 'Built into the Streets tiles',
};

function renderLegend() {
  const box = $('#legend');
  box.replaceChildren();
  const item = (swatchHtml, text) => { const s = el('span'); s.innerHTML = swatchHtml; s.append(document.createTextNode(text)); box.append(s); };
  if (state.layers.zones) for (const f of D.zones.features) if (state.zonesOn[f.properties.label]) item('<i class="sw" style="background:' + f.properties.color + '"></i>', f.properties.label);
  if (state.layers.dots || state.layers.outlines) item('<i class="sw" style="background:linear-gradient(90deg,#1f77b4,#e6550d,#2ca02c)"></i>', 'Town colors = region');
  if (state.layers.dots) item('<i class="sw" style="background:#9e9e9e;border-radius:50%"></i>', 'No region');
  if (state.layers.dispatch) {
    item('<i class="dot tech"></i>', 'Technician');
    item('<i class="dot"></i>', 'Dispatch town');
    if (state.routeMode === 'neighbors') item('<i class="ln"></i>', 'Closest town');
    if (state.routeMode === 'staff') item('<i class="ln staff"></i>', 'Nearest tech');
    if (state.routeMode === 'selected') item('<i class="ln" style="border-top:4px solid #ffcf3d"></i>', 'Selected town\'s closest');
  }
}

// ---------- Notes export / import ----------
function escapeCSV(v) { const s = String(v ?? ''); return /[",\n\r]/.test(s) ? '"' + s.replace(/"/g, '""') + '"' : s; }
function csvParse(s) {
  const rows = []; let fields = [], cur = '', quoted = false;
  for (let i = 0; i < s.length; i++) {
    const ch = s[i];
    if (ch === '"') { if (quoted && s[i + 1] === '"') { cur += '"'; i++; } else quoted = !quoted; }
    else if (ch === ',' && !quoted) { fields.push(cur); cur = ''; }
    else if ((ch === '\n' || ch === '\r') && !quoted) {
      if (ch === '\r' && s[i + 1] === '\n') i++;
      fields.push(cur); cur = '';
      if (fields.some(v => v.trim() !== '')) rows.push(fields);
      fields = [];
    } else cur += ch;
  }
  if (quoted) throw new Error('Unclosed quote in CSV');
  fields.push(cur);
  if (fields.some(v => v.trim() !== '')) rows.push(fields);
  return rows;
}
function download(text, name) {
  const url = URL.createObjectURL(new Blob([text], { type: 'text/csv;charset=utf-8' }));
  const a = el('a'); a.href = url; a.download = name;
  document.body.append(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
$('#notesExport').addEventListener('click', () => {
  const rows = [['Town', 'State', 'Technician', 'Notes']];
  for (const key of Object.keys(notes).sort()) {
    if (!getNote(key) || !allKeys.includes(key)) continue;
    const t = townInfo(key);
    rows.push([t.name, t.state, t.techs.join(', '), getNote(key)]);
  }
  download('﻿' + rows.map(r => r.map(escapeCSV).join(',')).join('\r\n'), 'Town_Notes.csv');
});
$('#notesImportBtn').addEventListener('click', () => $('#notesImportInput').click());
$('#notesImportInput').addEventListener('change', async e => {
  const f = e.target.files && e.target.files[0];
  if (!f) return;
  try {
    const table = csvParse((await f.text()).replace(/^﻿/, ''));
    const head = table.shift().map(h => h.trim().toLowerCase());
    // "Market" is the column the old dispatch map exported, so its backups import too
    const ti = head.indexOf('town') >= 0 ? head.indexOf('town') : head.indexOf('market');
    const si = head.indexOf('state'), ni = head.indexOf('notes');
    if (ti < 0 || ni < 0) throw new Error('Expected Town and Notes columns');
    let imported = 0;
    for (const cells of table) {
      const name = String(cells[ti] || '').trim().toLowerCase();
      const st = si >= 0 ? String(cells[si] || '').trim().toUpperCase() : '';
      const key = allKeys.find(k => { const [n, s] = k.split('|'); return n.toLowerCase() === name && (!st || s === st); });
      if (key) { notes[key] = String(cells[ni] || ''); imported++; }
    }
    writeStore(NOTES_KEY, notes);
    renderCard();
    renderPins();
    renderDirectory();
    alert('Imported notes for ' + imported + ' towns.');
  } catch (err) {
    alert('Could not import notes: ' + err.message);
  } finally {
    e.target.value = '';
  }
});

// ---------- Wiring ----------
function update() {
  renderBasemap();
  renderZones();
  renderOutlines();
  renderDots();
  renderDispatch();
  renderStaging();
  renderControls();
  renderLegend();
  persist();
}
$('#title').textContent = D.title;
$('#subtitle').textContent = cities.length + ' serviced towns · ' + zoneNames.length + ' zones · ' + places.length + ' dispatch towns · built ' + D.builtAt;
$('#builtAt').textContent = 'Map data built ' + D.builtAt + '. Drive times are from OSRM at build time.';
update();
