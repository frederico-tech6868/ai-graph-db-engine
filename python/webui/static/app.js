// ---------------------------------------------------------------- helpers
const $ = (s) => document.querySelector(s);
const $$ = (s) => document.querySelectorAll(s);
const api = async (path, opts = {}) => {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...opts,
  });
  if (!res.ok) {
    let msg = res.statusText;
    try { msg = (await res.json()).detail || msg; } catch (e) {}
    throw new Error(msg);
  }
  return res.status === 204 ? null : res.json();
};
function toast(msg, kind = "good") {
  const t = $("#toast");
  t.textContent = msg;
  t.className = `toast show ${kind}`;
  setTimeout(() => (t.className = "toast"), 2600);
}
const shortId = (id) => (id || "").slice(0, 8);
const propsPreview = (p) =>
  Object.entries(p || {})
    .filter(([k]) => k !== "text" && k !== "embedding")
    .slice(0, 3)
    .map(([k, v]) => `${k}=${v}`)
    .join(", ") || "—";

// color palette per label
const PALETTE = ["#5b8def","#7c5cff","#2ea86f","#e0a33e","#e5534b","#3fb6c4","#d76fd0","#8fbf46"];
const labelColors = {};
function colorFor(label) {
  if (!labelColors[label]) {
    labelColors[label] = PALETTE[Object.keys(labelColors).length % PALETTE.length];
  }
  return labelColors[label];
}

let NODES_CACHE = [];

// ---------------------------------------------------------------- navigation
$$(".nav-btn").forEach((btn) =>
  btn.addEventListener("click", () => {
    $$(".nav-btn").forEach((b) => b.classList.remove("active"));
    $$(".view").forEach((v) => v.classList.remove("active"));
    btn.classList.add("active");
    $("#view-" + btn.dataset.view).classList.add("active");
    if (btn.dataset.view === "dashboard") loadDashboard();
    if (btn.dataset.view === "nodes") loadNodes();
    if (btn.dataset.view === "edges") loadEdges();
    if (btn.dataset.view === "search") refreshLabelSelects();
    if (btn.dataset.view === "memory") loadMemory();
  })
);

// ---------------------------------------------------------------- dashboard
let network = null;
async function loadDashboard() {
  const stats = await api("/api/stats");
  $("#stat-nodes").textContent = stats.node_count;
  $("#stat-edges").textContent = stats.edge_count;
  $("#stat-labels").textContent = stats.labels.length;
  $("#stat-memories").textContent = stats.memory.total_memories || 0;
  await refreshLabelSelects(stats.labels.map((l) => l.label));
  await renderGraph();
}

async function renderGraph() {
  const label = $("#graph-label-filter").value;
  const data = await api("/api/graph" + (label ? `?label=${encodeURIComponent(label)}` : ""));
  const nodes = data.nodes.map((n) => ({
    id: n.id,
    label: (n.properties.name || n.properties.text || n.label) + "",
    title: `${n.label} (${shortId(n.id)})`,
    color: colorFor(n.label),
    _raw: n,
  }));
  const edges = data.edges.map((e) => ({
    id: e.id, from: e.src_id, to: e.dst_id, label: e.label,
    arrows: "to", font: { color: "#8b98a8", size: 10, strokeWidth: 0 },
    color: { color: "#3a4657" },
  }));
  const container = $("#graph");
  const vdata = { nodes: new vis.DataSet(nodes), edges: new vis.DataSet(edges) };
  const options = {
    nodes: { shape: "dot", size: 16, font: { color: "#e6edf3", size: 12 }, borderWidth: 0 },
    edges: { smooth: { type: "continuous" }, width: 1.5 },
    physics: { stabilization: true, barnesHut: { gravitationalConstant: -6000, springLength: 130 } },
    interaction: { hover: true },
  };
  network = new vis.Network(container, vdata, options);
  network.on("click", (params) => {
    if (params.nodes.length) showNodeDetail(params.nodes[0]);
  });
  // legend
  const labels = [...new Set(data.nodes.map((n) => n.label))];
  $("#legend").innerHTML = labels
    .map((l) => `<span><i style="background:${colorFor(l)}"></i>${l}</span>`)
    .join("");
}

async function showNodeDetail(id) {
  const n = await api("/api/nodes/" + id);
  const rows = Object.entries(n.properties)
    .filter(([k]) => k !== "embedding")
    .map(([k, v]) => `<div class="k">${k}</div><div>${JSON.stringify(v)}</div>`)
    .join("");
  const edgesHtml = [...n.out_edges.map((e) => `→ ${e.label} → ${shortId(e.dst_id)}`),
                     ...n.in_edges.map((e) => `← ${e.label} ← ${shortId(e.src_id)}`)]
    .map((s) => `<div class="mono">${s}</div>`).join("") || "<div class='mono'>no edges</div>";
  $("#node-detail").innerHTML = `
    <div class="kv"><div class="k">id</div><div class="mono">${n.id}</div>
      <div class="k">label</div><div><span class="tag">${n.label}</span></div>
      <div class="k">embedding</div><div>${n.has_embedding ? "yes" : "no"}</div></div>
    <div class="kv">${rows}</div>
    <h2 style="font-size:13px;margin:8px 0">Edges</h2>${edgesHtml}
    <button class="mini danger" style="margin-top:12px" onclick="deleteNode('${n.id}')">Delete node</button>`;
}

$("#btn-refresh-graph").addEventListener("click", renderGraph);
$("#graph-label-filter").addEventListener("change", renderGraph);

// ---------------------------------------------------------------- labels
async function refreshLabelSelects(labels) {
  if (!labels) {
    const stats = await api("/api/stats");
    labels = stats.labels.map((l) => l.label);
  }
  const opts = '<option value="">All labels</option>' +
    labels.map((l) => `<option value="${l}">${l}</option>`).join("");
  ["#graph-label-filter", "#nodes-label-filter", "#search-label"].forEach((sel) => {
    const el = $(sel); if (!el) return;
    const cur = el.value; el.innerHTML = opts; el.value = cur;
  });
}

// ---------------------------------------------------------------- nodes view
async function loadNodes() {
  await refreshLabelSelects();
  const label = $("#nodes-label-filter").value;
  const nodes = await api("/api/nodes" + (label ? `?label=${encodeURIComponent(label)}` : ""));
  NODES_CACHE = nodes;
  const tb = $("#nodes-table tbody");
  tb.innerHTML = nodes.map((n) => `
    <tr class="clickable" onclick="showNodeDetailModal('${n.id}')">
      <td><span class="tag">${n.label}</span></td>
      <td>${propsPreview(n.properties)}</td>
      <td>${n.has_embedding ? "✓" : "—"}</td>
      <td><button class="del-btn" onclick="event.stopPropagation();deleteNode('${n.id}')">✕</button></td>
    </tr>`).join("") || `<tr><td colspan="4" class="mono">no nodes</td></tr>`;
  populateEdgeSelects(nodes);
}
$("#nodes-label-filter").addEventListener("change", loadNodes);

async function showNodeDetailModal(id) {
  const n = await api("/api/nodes/" + id);
  toast(`${n.label}: ${propsPreview(n.properties)}`);
}

// add property rows
$("#btn-add-prop").addEventListener("click", () => addPropRow());
function addPropRow(k = "", v = "") {
  const row = document.createElement("div");
  row.className = "prop-row";
  row.innerHTML = `<input placeholder="key" value="${k}"/><input placeholder="value" value="${v}"/>
    <button onclick="this.parentElement.remove()">✕</button>`;
  $("#node-props").appendChild(row);
}
function collectProps() {
  const props = {};
  $$("#node-props .prop-row").forEach((r) => {
    const [kEl, vEl] = r.querySelectorAll("input");
    if (kEl.value.trim()) {
      let val = vEl.value;
      if (/^-?\d+$/.test(val)) val = parseInt(val);
      else if (/^-?\d*\.\d+$/.test(val)) val = parseFloat(val);
      else if (val === "true") val = true;
      else if (val === "false") val = false;
      props[kEl.value.trim()] = val;
    }
  });
  return props;
}
$("#btn-add-node").addEventListener("click", async () => {
  const label = $("#node-label").value.trim();
  if (!label) return toast("Label required", "bad");
  try {
    await api("/api/nodes", { method: "POST", body: JSON.stringify({
      label, properties: collectProps(), text: $("#node-text").value.trim() || null }) });
    toast("Node added");
    $("#node-label").value = ""; $("#node-text").value = ""; $("#node-props").innerHTML = "";
    loadNodes();
  } catch (e) { toast(e.message, "bad"); }
});
async function deleteNode(id) {
  try { await api("/api/nodes/" + id, { method: "DELETE" }); toast("Node deleted");
    loadNodes(); if ($("#view-dashboard").classList.contains("active")) loadDashboard();
  } catch (e) { toast(e.message, "bad"); }
}

// ---------------------------------------------------------------- edges view
function populateEdgeSelects(nodes) {
  const opts = nodes.map((n) =>
    `<option value="${n.id}">${n.label}: ${n.properties.name || shortId(n.id)}</option>`).join("");
  ["#edge-src", "#edge-dst"].forEach((s) => { if ($(s)) $(s).innerHTML = opts; });
}
async function loadEdges() {
  const [nodes, edges] = await Promise.all([api("/api/nodes"), api("/api/edges")]);
  NODES_CACHE = nodes;
  populateEdgeSelects(nodes);
  const nameOf = (id) => {
    const n = nodes.find((x) => x.id === id);
    return n ? (n.properties.name || `${n.label}:${shortId(id)}`) : shortId(id);
  };
  $("#edges-table tbody").innerHTML = edges.map((e) => `
    <tr><td><span class="tag">${e.label}</span></td>
      <td>${nameOf(e.src_id)} → ${nameOf(e.dst_id)}</td>
      <td>${e.weight}</td>
      <td><button class="del-btn" onclick="deleteEdge('${e.id}')">✕</button></td></tr>`).join("")
    || `<tr><td colspan="4" class="mono">no edges</td></tr>`;
}
$("#edge-thr").addEventListener("input", (e) => ($("#thr-val").textContent = e.target.value));
$("#btn-add-edge").addEventListener("click", async () => {
  const body = {
    src_id: $("#edge-src").value, dst_id: $("#edge-dst").value,
    label: $("#edge-label").value.trim() || "RELATED",
    weight: parseFloat($("#edge-weight").value) || 1.0,
    similarity_threshold: parseFloat($("#edge-thr").value),
  };
  if (!body.src_id || !body.dst_id) return toast("Pick source and destination", "bad");
  try {
    const res = await api("/api/edges", { method: "POST", body: JSON.stringify(body) });
    const box = $("#edge-scan-result");
    if (res.was_flagged) {
      box.innerHTML = `<div class="flag"><h4>⚠ ${res.similar_edges.length} similar edge(s) found (label-scoped scan)</h4>` +
        res.similar_edges.map((m) =>
          `<div class="match">edge ${shortId(m.existing_edge_id)} — combined ${m.combined_score}
           (src ${m.src_similarity}, dst ${m.dst_similarity})</div>`).join("") +
        `<div class="match">The edge was still added; use this to decide whether to deduplicate.</div></div>`;
    } else {
      box.innerHTML = `<div class="flag ok"><h4>✓ No similar edges — added cleanly</h4></div>`;
    }
    toast("Edge added");
    loadEdges();
  } catch (e) { toast(e.message, "bad"); }
});
async function deleteEdge(id) {
  try { await api("/api/edges/" + id, { method: "DELETE" }); toast("Edge deleted"); loadEdges(); }
  catch (e) { toast(e.message, "bad"); }
}

// ---------------------------------------------------------------- search
$("#btn-search").addEventListener("click", async () => {
  const text = $("#search-text").value.trim();
  if (!text) return toast("Enter a query", "bad");
  try {
    const results = await api("/api/search", { method: "POST", body: JSON.stringify({
      text, label: $("#search-label").value || null, k: parseInt($("#search-k").value) || 5 }) });
    $("#search-results").innerHTML = results.length ? results.map((r) => `
      <div class="result-card">
        <div class="result-head"><span><span class="tag">${r.node.label}</span>
          ${r.node.properties.name || r.node.properties.text || shortId(r.node.id)}</span>
          <b>${r.score.toFixed(3)}</b></div>
        <div class="score-bar"><div class="score-fill" style="width:${Math.max(0, r.score * 100)}%"></div></div>
      </div>`).join("") : `<div class="mono">no results (nodes need embeddings — add nodes with text)</div>`;
  } catch (e) { toast(e.message, "bad"); }
});

// ---------------------------------------------------------------- memory
async function loadMemory() {
  const ents = await api("/api/memory/entities");
  $("#entities-list").innerHTML = ents.map((e) =>
    `<span class="chip">${e.properties.name || shortId(e.id)}</span>`).join("") ||
    "<span class='mono'>no entities yet</span>";
}
$("#btn-remember").addEventListener("click", async () => {
  const text = $("#mem-text").value.trim();
  if (!text) return toast("Enter memory text", "bad");
  const entities = $("#mem-entities").value.split(",").map((s) => s.trim()).filter(Boolean);
  try {
    const res = await api("/api/memory/remember", { method: "POST", body: JSON.stringify({
      text, memory_type: $("#mem-type").value, entities }) });
    const box = $("#remember-result");
    if (res.was_duplicate) {
      box.innerHTML = `<div class="flag"><h4>⚠ Near-duplicate of ${res.similar_existing.length} memory(ies)</h4>` +
        res.similar_existing.map((s) =>
          `<div class="match">${(s.node.properties.text || "").slice(0, 60)}… — ${s.score}</div>`).join("") +
        `</div>`;
    } else {
      box.innerHTML = `<div class="flag ok"><h4>✓ Stored as new memory</h4></div>`;
    }
    toast("Remembered");
    $("#mem-text").value = ""; $("#mem-entities").value = "";
    loadMemory();
  } catch (e) { toast(e.message, "bad"); }
});
$("#btn-recall").addEventListener("click", async () => {
  const query = $("#recall-query").value.trim();
  if (!query) return toast("Enter a query", "bad");
  try {
    const results = await api("/api/memory/recall", { method: "POST", body: JSON.stringify({
      query, k: parseInt($("#recall-k").value) || 5 }) });
    $("#recall-results").innerHTML = results.length ? results.map((r) => `
      <div class="result-card">
        <div class="result-head"><span class="tag">${r.node.properties.memory_type || "memory"}</span>
          <b>${r.score.toFixed(3)}</b></div>
        <div>${r.node.properties.text || r.context_snippet}</div>
        <div class="score-bar"><div class="score-fill" style="width:${Math.max(0, r.score * 100)}%"></div></div>
      </div>`).join("") : `<div class="mono">no memories found</div>`;
  } catch (e) { toast(e.message, "bad"); }
});
$("#btn-reflect").addEventListener("click", async () => {
  try { const r = await api("/api/memory/reflect", { method: "POST" });
    $("#recall-results").innerHTML = `<div class="result-card"><b>Reflection</b><div>${r.reflection}</div></div>`;
    toast("Reflection generated"); loadMemory();
  } catch (e) { toast(e.message, "bad"); }
});

// ---------------------------------------------------------------- admin
$("#btn-seed").addEventListener("click", async () => {
  try { await api("/api/seed", { method: "POST" }); toast("Demo data loaded"); loadDashboard(); }
  catch (e) { toast(e.message, "bad"); }
});
$("#btn-save").addEventListener("click", async () => {
  try { const r = await api("/api/save", { method: "POST" }); toast("Saved to " + r.saved); }
  catch (e) { toast(e.message, "bad"); }
});
$("#btn-load").addEventListener("click", async () => {
  try { await api("/api/load", { method: "POST" }); toast("Loaded from disk"); loadDashboard(); }
  catch (e) { toast(e.message, "bad"); }
});
$("#btn-reset").addEventListener("click", async () => {
  if (!confirm("Clear the entire graph?")) return;
  try { await api("/api/reset", { method: "POST" }); toast("Graph reset"); loadDashboard(); }
  catch (e) { toast(e.message, "bad"); }
});

// ---------------------------------------------------------------- init
(async function init() {
  try {
    const h = await api("/api/health");
    $("#backend").textContent = h.backend;
  } catch (e) { $("#backend").textContent = "offline"; }
  loadDashboard();
})();
