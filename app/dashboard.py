DASHBOARD_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Unicorn JetCarCloud Dashboard</title>
  <style>
    :root {
      --bg: #f6f7fb;
      --panel: #ffffff;
      --text: #202532;
      --muted: #697386;
      --line: #d8dde8;
      --accent: #6b4eff;
      --ok: #168a56;
      --warn: #b56a00;
      --bad: #b42318;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font: 14px/1.45 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    header {
      padding: 18px 22px;
      background: #121826;
      color: #fff;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 18px;
      flex-wrap: wrap;
    }
    h1 {
      margin: 0;
      font-size: 20px;
      font-weight: 650;
      letter-spacing: 0;
    }
    h2 {
      margin: 0 0 12px;
      font-size: 15px;
      font-weight: 650;
    }
    main {
      padding: 18px;
      max-width: 1500px;
      margin: 0 auto;
    }
    .topline {
      color: #c9d2e3;
      display: flex;
      gap: 14px;
      flex-wrap: wrap;
      font-size: 13px;
    }
    .grid {
      display: grid;
      grid-template-columns: repeat(5, minmax(140px, 1fr));
      gap: 12px;
      margin-bottom: 14px;
    }
    .card, .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04);
    }
    .card {
      padding: 14px;
      min-height: 86px;
    }
    .label {
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 6px;
    }
    .value {
      font-size: 24px;
      font-weight: 700;
      line-height: 1.1;
    }
    .sub {
      color: var(--muted);
      font-size: 12px;
      margin-top: 6px;
      overflow-wrap: anywhere;
    }
    .layout {
      display: grid;
      grid-template-columns: minmax(0, 1fr) 420px;
      gap: 14px;
      align-items: start;
    }
    .stack {
      display: grid;
      gap: 14px;
    }
    .panel {
      padding: 14px;
      overflow: hidden;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      table-layout: fixed;
    }
    th, td {
      padding: 8px 7px;
      border-bottom: 1px solid var(--line);
      text-align: left;
      vertical-align: top;
      overflow-wrap: anywhere;
    }
    th {
      color: var(--muted);
      font-size: 12px;
      font-weight: 650;
      background: #fafbfe;
    }
    tr:last-child td { border-bottom: 0; }
    .pill {
      display: inline-flex;
      align-items: center;
      min-height: 22px;
      padding: 2px 8px;
      border-radius: 999px;
      font-size: 12px;
      font-weight: 650;
      background: #eef1f7;
      color: #384152;
    }
    .pill.ok { background: #e7f6ee; color: var(--ok); }
    .pill.warn { background: #fff3df; color: var(--warn); }
    .pill.bad { background: #feeceb; color: var(--bad); }
    .controls {
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 8px;
      margin-bottom: 10px;
    }
    select, button {
      min-height: 34px;
      border-radius: 6px;
      border: 1px solid var(--line);
      background: #fff;
      color: var(--text);
      padding: 6px 9px;
      font: inherit;
    }
    button {
      background: var(--accent);
      border-color: var(--accent);
      color: #fff;
      cursor: pointer;
    }
    .preview {
      width: 100%;
      aspect-ratio: 4 / 3;
      border-radius: 6px;
      border: 1px solid var(--line);
      background: #0c111d;
      display: grid;
      place-items: center;
      overflow: hidden;
    }
    .preview img {
      width: 100%;
      height: 100%;
      object-fit: contain;
      display: block;
    }
    .empty {
      color: var(--muted);
      padding: 16px;
      text-align: center;
    }
    pre {
      margin: 0;
      max-height: 280px;
      overflow: auto;
      padding: 10px;
      border-radius: 6px;
      background: #0c111d;
      color: #d7e1f5;
      font: 12px/1.45 ui-monospace, SFMono-Regular, Consolas, monospace;
    }
    .error { color: var(--bad); }
    @media (max-width: 1100px) {
      .grid { grid-template-columns: repeat(2, minmax(140px, 1fr)); }
      .layout { grid-template-columns: 1fr; }
    }
    @media (max-width: 620px) {
      main { padding: 12px; }
      header { padding: 14px; }
      .grid { grid-template-columns: 1fr; }
      th, td { padding: 7px 5px; font-size: 12px; }
    }
  </style>
</head>
<body>
  <header>
    <div>
      <h1>Unicorn JetCarCloud Dashboard</h1>
      <div class="topline">
        <span id="service">service: loading</span>
        <span id="updated">updated: -</span>
        <span id="endpoint">state: /api/dashboard/state</span>
      </div>
    </div>
    <button id="refreshBtn" type="button">Refresh</button>
  </header>

  <main>
    <section class="grid">
      <div class="card"><div class="label">Streams</div><div id="streamsCount" class="value">-</div><div id="streamsSub" class="sub">registered video streams</div></div>
      <div class="card"><div class="label">Algorithms</div><div id="algorithmsCount" class="value">-</div><div id="algorithmsSub" class="sub">enabled / total</div></div>
      <div class="card"><div class="label">Active Tasks</div><div id="tasksCount" class="value">-</div><div id="tasksSub" class="sub">inference tasks</div></div>
      <div class="card"><div class="label">App Clients</div><div id="clientsCount" class="value">-</div><div id="clientsSub" class="sub">websocket clients</div></div>
      <div class="card"><div class="label">Cached Results</div><div id="resultsCount" class="value">-</div><div id="resultsSub" class="sub">processed frame previews</div></div>
    </section>

    <section class="layout">
      <div class="stack">
        <div class="panel">
          <h2>Video Streams</h2>
          <div id="streamsTable"></div>
        </div>
        <div class="panel">
          <h2>Latest Algorithm Results</h2>
          <div id="resultsTable"></div>
        </div>
        <div class="panel">
          <h2>Algorithm Catalog</h2>
          <div id="algorithmsTable"></div>
        </div>
      </div>

      <aside class="stack">
        <div class="panel">
          <h2>Processed Preview</h2>
          <div class="controls">
            <select id="previewSelect"></select>
            <button id="openPreview" type="button">Open</button>
          </div>
          <div class="preview" id="previewBox"><div class="empty">No processed frame yet</div></div>
          <div id="previewInfo" class="sub"></div>
        </div>
        <div class="panel">
          <h2>Debug Dump</h2>
          <div id="debugTable"></div>
        </div>
        <div class="panel">
          <h2>Raw State</h2>
          <pre id="rawState">{}</pre>
        </div>
      </aside>
    </section>
  </main>

  <script>
    const stateUrl = "/api/dashboard/state";
    let latestState = null;

    function text(id, value) {
      document.getElementById(id).textContent = value;
    }

    function fmtTime(value) {
      if (!value) return "-";
      return new Date(value * 1000).toLocaleTimeString();
    }

    function fmtAge(value) {
      if (value === null || value === undefined) return "-";
      if (value < 1) return value.toFixed(2) + "s";
      if (value < 60) return value.toFixed(1) + "s";
      return Math.round(value) + "s";
    }

    function pill(value, kind) {
      return `<span class="pill ${kind || ""}">${escapeHtml(String(value))}</span>`;
    }

    function escapeHtml(value) {
      return String(value)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
    }

    function table(headers, rows, emptyText) {
      if (!rows.length) return `<div class="empty">${escapeHtml(emptyText)}</div>`;
      return `<table><thead><tr>${headers.map(h => `<th>${escapeHtml(h)}</th>`).join("")}</tr></thead><tbody>${rows.join("")}</tbody></table>`;
    }

    function render(data) {
      latestState = data;
      const streams = data.streams || [];
      const algorithms = data.algorithms || [];
      const results = data.processed_frames || [];
      const connections = data.connections || {};
      const clientCount = Object.values(connections).reduce((sum, item) => sum + (item.app_clients || 0), 0);
      const enabledAlgorithms = algorithms.filter(item => item.enabled).length;

      text("service", `service: ${data.service || "-"}`);
      text("updated", `updated: ${new Date().toLocaleTimeString()}`);
      text("streamsCount", streams.length);
      text("algorithmsCount", `${enabledAlgorithms}/${algorithms.length}`);
      text("tasksCount", data.processing ? data.processing.active_count : 0);
      text("clientsCount", clientCount);
      text("resultsCount", results.length);
      text("tasksSub", data.processing ? `limit ${data.processing.max_concurrent_tasks}, min interval ${data.processing.algorithm_min_interval_ms}ms` : "inference tasks");
      text("clientsSub", `${Object.keys(connections).length} car channel(s)`);
      text("resultsSub", results.length ? `newest age ${fmtAge(results[0].age_seconds)}` : "processed frame previews");

      renderStreams(streams);
      renderResults(results);
      renderAlgorithms(algorithms);
      renderDebug(data.debug || {});
      renderPreviewOptions(results);
      document.getElementById("rawState").textContent = JSON.stringify(data, null, 2);
    }

    function renderStreams(streams) {
      const rows = streams.map(item => {
        const status = item.last_error ? pill("error", "bad") : item.running ? pill("running", "ok") : pill("idle", "warn");
        return `<tr>
          <td>${escapeHtml(item.car_id)}</td>
          <td>${escapeHtml(item.stream_id)}</td>
          <td>${escapeHtml(item.transport)}</td>
          <td>${status}</td>
          <td>${item.frame_count}</td>
          <td>${fmtTime(item.last_frame_at)}</td>
          <td class="${item.last_error ? "error" : ""}">${escapeHtml(item.last_error || "")}</td>
        </tr>`;
      });
      document.getElementById("streamsTable").innerHTML = table(
        ["car", "stream", "transport", "status", "frames", "last frame", "error"],
        rows,
        "No video streams registered yet"
      );
    }

    function renderResults(results) {
      const rows = results.map(item => {
        const status = item.ok ? pill("ok", "ok") : pill("failed", "bad");
        return `<tr>
          <td>${escapeHtml(item.car_id)} / ${escapeHtml(item.stream_id)}</td>
          <td>${escapeHtml(item.algorithm_id)}</td>
          <td>${status}</td>
          <td>${Number(item.latency_ms || 0).toFixed(1)}</td>
          <td>${escapeHtml(String(item.detection_count))}</td>
          <td>${fmtAge(item.age_seconds)}</td>
          <td class="${item.error ? "error" : ""}">${escapeHtml(item.error || item.summary || "")}</td>
        </tr>`;
      });
      document.getElementById("resultsTable").innerHTML = table(
        ["stream", "algorithm", "status", "ms", "detections", "age", "summary"],
        rows,
        "No cached algorithm result yet"
      );
    }

    function renderAlgorithms(algorithms) {
      const rows = algorithms.map(item => {
        const model = item.metadata && item.metadata.model_path ? item.metadata.model_path : "";
        const task = item.metadata && item.metadata.task ? item.metadata.task : "";
        return `<tr>
          <td>${escapeHtml(item.algorithm_id)}</td>
          <td>${escapeHtml(item.runner)}</td>
          <td>${item.enabled ? pill("enabled", "ok") : pill("disabled", "warn")}</td>
          <td>${escapeHtml(task)}</td>
          <td>${escapeHtml(model)}</td>
        </tr>`;
      });
      document.getElementById("algorithmsTable").innerHTML = table(
        ["id", "runner", "enabled", "task", "model"],
        rows,
        "No algorithms loaded"
      );
    }

    function renderDebug(debug) {
      const entries = debug.recent_frames || [];
      const rows = entries.map(item => `<tr>
        <td>${escapeHtml(item.car_id)}</td>
        <td>${escapeHtml(item.stream_id)}</td>
        <td>${escapeHtml(item.frame)}</td>
        <td>${fmtTime(item.modified_at)}</td>
        <td>${escapeHtml((item.algorithms || []).join(", "))}</td>
      </tr>`);
      document.getElementById("debugTable").innerHTML =
        `<div class="sub">enabled: ${debug.enabled ? "true" : "false"} | dir: ${escapeHtml(debug.dir || "")}</div>` +
        table(["car", "stream", "frame", "modified", "algorithms"], rows, "No debug frame dump found");
    }

    function renderPreviewOptions(results) {
      const select = document.getElementById("previewSelect");
      const previous = select.value;
      select.innerHTML = "";
      for (const item of results) {
        const option = document.createElement("option");
        option.value = item.mjpeg_url;
        option.textContent = `${item.car_id}/${item.stream_id} - ${item.algorithm_id}`;
        option.dataset.info = `ok=${item.ok} latency=${Number(item.latency_ms || 0).toFixed(1)}ms age=${fmtAge(item.age_seconds)}`;
        select.appendChild(option);
      }
      if (previous && [...select.options].some(item => item.value === previous)) {
        select.value = previous;
      }
      if (!select.value && select.options.length) {
        select.selectedIndex = 0;
      }
    }

    function openPreview() {
      const select = document.getElementById("previewSelect");
      const box = document.getElementById("previewBox");
      if (!select.value) {
        box.innerHTML = '<div class="empty">No processed frame yet</div>';
        text("previewInfo", "");
        return;
      }
      const cacheBust = select.value.includes("?") ? "&" : "?";
      box.innerHTML = `<img alt="processed preview" src="${escapeHtml(select.value + cacheBust + "t=" + Date.now())}">`;
      text("previewInfo", select.selectedOptions[0] ? select.selectedOptions[0].dataset.info : select.value);
    }

    async function refresh() {
      try {
        const response = await fetch(stateUrl, { cache: "no-store" });
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        render(await response.json());
      } catch (error) {
        document.getElementById("rawState").textContent = String(error);
      }
    }

    document.getElementById("refreshBtn").addEventListener("click", refresh);
    document.getElementById("openPreview").addEventListener("click", openPreview);
    document.getElementById("previewSelect").addEventListener("change", openPreview);
    refresh();
    setInterval(refresh, 1500);
  </script>
</body>
</html>
"""
