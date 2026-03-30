const state = {
  metastudies: [],
  selectedMetastudyPath: null,
  metastudyPayload: null,
  expandedNodes: new Set(),
  selectedTreePath: null,
  autoTail: true,
  inspector: null,
  showHiddenTranscriptItems: false,
};

let inspectorTimer = null;

const metastudySelect = document.getElementById("metastudy-select");
const summaryStrip = document.getElementById("summary-strip");
const fileTree = document.getElementById("file-tree");
const agentTree = document.getElementById("agent-tree");
const inspectorTitle = document.getElementById("inspector-title");
const inspectorActions = document.getElementById("inspector-actions");
const inspectorMeta = document.getElementById("inspector-meta");
const inspectorContent = document.getElementById("inspector-content");
const conversationView = document.getElementById("conversation-view");
const transcriptControls = document.getElementById("transcript-controls");
const autoTailToggle = document.getElementById("auto-tail-toggle");

document.getElementById("refresh-button").addEventListener("click", () => {
  void refreshCurrentMetastudy();
});

document.getElementById("open-run-log").addEventListener("click", () => {
  const runLog = state.metastudyPayload?.paths?.run_log;
  if (!runLog) return;
  void inspectFile(runLog, "Latest run.log tail", { lines: 200, autoTail: true });
});

autoTailToggle.addEventListener("change", (event) => {
  state.autoTail = event.target.checked;
  if (state.inspector) {
    scheduleInspectorRefresh();
  }
});

metastudySelect.addEventListener("change", async (event) => {
  state.selectedMetastudyPath = event.target.value;
  await loadMetastudy(event.target.value);
});

function apiUrl(path, params = {}) {
  const url = new URL(path, window.location.origin);
  Object.entries(params).forEach(([key, value]) => {
    if (value !== undefined && value !== null) {
      url.searchParams.set(key, String(value));
    }
  });
  return url.toString();
}

async function fetchJson(path, params = {}) {
  const response = await fetch(apiUrl(path, params));
  if (!response.ok) {
    const errorPayload = await response.json().catch(() => ({ error: response.statusText }));
    throw new Error(errorPayload.error || response.statusText);
  }
  return response.json();
}

async function loadMetastudies() {
  const payload = await fetchJson("/api/metastudies");
  state.metastudies = payload.metastudies;
  metastudySelect.innerHTML = "";
  payload.metastudies.forEach((metastudy) => {
    const option = document.createElement("option");
    option.value = metastudy.path;
    option.textContent = metastudy.name;
    metastudySelect.append(option);
  });

  if (!state.selectedMetastudyPath && payload.metastudies.length > 0) {
    state.selectedMetastudyPath = payload.metastudies[0].path;
  }

  if (state.selectedMetastudyPath) {
    metastudySelect.value = state.selectedMetastudyPath;
    await loadMetastudy(state.selectedMetastudyPath);
  }
}

async function refreshCurrentMetastudy() {
  await loadMetastudies();
}

async function loadMetastudy(path) {
  state.metastudyPayload = await fetchJson("/api/metastudy", { path });
  renderSummary(state.metastudyPayload.summary);
  renderFileTree();
  renderAgentTree();
  if (!state.inspector) {
    await inspectFile(state.metastudyPayload.paths.run_log, "Latest run.log tail", {
      lines: 200,
      autoTail: true,
    });
  }
}

function renderSummary(summary) {
  summaryStrip.innerHTML = "";
  const cards = [
    ["Studies", String(summary.study_count)],
    ["Succeeded runs", String(summary.status_counts.succeeded || 0)],
    ["Failed runs", String(summary.status_counts.failed || 0)],
    ["Latest activity", summary.latest_run_display || "none"],
  ];
  cards.forEach(([label, value]) => {
    const element = document.createElement("article");
    element.className = "summary-card";
    element.innerHTML = `<span>${label}</span><strong>${escapeHtml(value)}</strong>`;
    summaryStrip.append(element);
  });
}

function renderFileTree() {
  fileTree.innerHTML = "";
  if (!state.metastudyPayload) return;
  const root = state.metastudyPayload.file_tree;
  if (!state.expandedNodes.has(root.path)) {
    state.expandedNodes.add(root.path);
  }
  fileTree.append(buildTreeNode(root, "file"));
}

function renderAgentTree() {
  agentTree.innerHTML = "";
  if (!state.metastudyPayload) return;
  const root = state.metastudyPayload.agent_tree;
  if (!state.expandedNodes.has(root.path)) {
    state.expandedNodes.add(root.path);
  }
  agentTree.append(buildTreeNode(root, "agent"));
}

function buildTreeNode(node, mode) {
  const template = document.getElementById("tree-node-template");
  const fragment = template.content.firstElementChild.cloneNode(true);
  const toggle = fragment.querySelector(".tree-toggle");
  const select = fragment.querySelector(".tree-select");
  const childrenContainer = fragment.querySelector(".tree-children");
  const isExpandable = Array.isArray(node.children) && node.children.length > 0;
  const isExpanded = state.expandedNodes.has(node.path || node.id);

  toggle.textContent = isExpandable ? (isExpanded ? "−" : "+") : "";
  toggle.disabled = !isExpandable;
  toggle.addEventListener("click", () => {
    const key = node.path || node.id;
    if (state.expandedNodes.has(key)) {
      state.expandedNodes.delete(key);
    } else {
      state.expandedNodes.add(key);
    }
    mode === "file" ? renderFileTree() : renderAgentTree();
  });

  select.dataset.path = node.path || node.id;
  if (state.selectedTreePath === select.dataset.path) {
    select.classList.add("active");
  }

  const label = document.createElement("span");
  label.className = "tree-label";
  label.innerHTML = labelMarkup(node, mode);
  select.append(label);

  select.addEventListener("click", async () => {
    state.selectedTreePath = select.dataset.path;
    if (mode === "file") {
      if (node.type === "file") {
        await inspectFile(node.path, node.name, { lines: defaultLines(node.name) });
      } else {
        await inspectDirectory(node);
      }
      renderFileTree();
    } else {
      if (node.kind === "run") {
        await inspectRun(node.invocation);
      } else {
        await inspectDirectory({ path: node.path, name: node.name });
      }
      renderAgentTree();
    }
  });

  if (isExpandable && isExpanded) {
    node.children.forEach((child) => {
      childrenContainer.append(buildTreeNode(child, mode));
    });
  } else if (!isExpandable) {
    childrenContainer.remove();
  }

  return fragment;
}

function labelMarkup(node, mode) {
  if (mode === "file") {
    const kind = node.type === "directory" ? "dir" : "file";
    return `
      <span class="pill role">${kind}</span>
      <span class="tree-name">${escapeHtml(node.name)}</span>
      <small>${formatPathTail(node.path)}</small>
    `;
  }

  if (node.kind === "group") {
    return `
      <span class="pill role">${escapeHtml(node.group_type)}</span>
      <span class="tree-name">${escapeHtml(node.name)}</span>
    `;
  }

  return `
    <span class="pill role">${escapeHtml(node.role)}</span>
    <span class="pill ${escapeHtml(node.status)}">${escapeHtml(node.status)}</span>
    <span class="tree-name">${escapeHtml(node.target_name)}</span>
    <small>${escapeHtml(node.start_display || "pending")}</small>
  `;
}

function importantFileButtons(run) {
  const buttons = [];
  (run.important_files || []).forEach((file) => {
    buttons.push({
      label: file.label,
      onClick: () => inspectFile(file.path, `${run.role}: ${file.label}`, { lines: defaultLines(file.label) }),
    });
  });
  if (run.session_id) {
    buttons.unshift({
      label: "Conversation",
      onClick: () => inspectSession(run.session_id, `${run.role} conversation`),
    });
  }
  buttons.push({
    label: "run.log",
    onClick: () => inspectFile(state.metastudyPayload.paths.run_log, "Latest run.log tail", { lines: 200, autoTail: true }),
  });
  return buttons;
}

async function inspectDirectory(node) {
  clearInspectorTimer();
  state.inspector = { type: "directory", path: node.path, autoTail: false };
  inspectorTitle.textContent = `${node.name || node.path}`;
  inspectorActions.replaceChildren();
  inspectorMeta.replaceChildren(
    metaCard("Path", node.path),
    metaCard("Type", "directory"),
  );
  transcriptControls.classList.add("hidden");
  conversationView.classList.add("hidden");
  inspectorContent.classList.remove("hidden");
  inspectorContent.textContent = `Directory selected:\n${node.path}`;
}

async function inspectRun(run) {
  const metaCards = [
    metaCard("Role", run.role),
    metaCard("Status", run.status),
    metaCard("Target", run.target_path),
    metaCard("Started", run.start_display || run.start_timestamp || "unknown"),
  ];
  if (run.duration_seconds !== null && run.duration_seconds !== undefined) {
    metaCards.push(metaCard("Duration", `${run.duration_seconds}s`));
  }
  if (run.session_id) {
    metaCards.push(metaCard("Session", run.session_id));
  }

  inspectorTitle.textContent = `${run.role} · ${run.target_name}`;
  inspectorMeta.replaceChildren(...metaCards);
  renderInspectorActions(importantFileButtons(run));

  if (run.session_id) {
    await inspectSession(run.session_id, `${run.role} conversation`, { preserveMeta: true });
    return;
  }

  const defaultFile = (run.important_files || [])[0];
  if (defaultFile) {
    await inspectFile(defaultFile.path, `${run.role}: ${defaultFile.label}`, {
      lines: defaultLines(defaultFile.label),
      preserveMeta: true,
    });
    return;
  }

  await inspectFile(state.metastudyPayload.paths.run_log, "Latest run.log tail", {
    lines: 200,
    autoTail: true,
    preserveMeta: true,
  });
}

function metaCard(label, value) {
  const card = document.createElement("article");
  card.className = "meta-card";
  card.innerHTML = `<span>${escapeHtml(label)}</span><strong>${escapeHtml(value)}</strong>`;
  return card;
}

function renderInspectorActions(actions) {
  inspectorActions.replaceChildren();
  actions.forEach((action) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "button button-muted";
    button.textContent = action.label;
    button.addEventListener("click", () => {
      void action.onClick();
    });
    inspectorActions.append(button);
  });
}

async function inspectFile(path, title, options = {}) {
  clearInspectorTimer();
  state.inspector = {
    type: "file",
    path,
    title,
    lines: options.lines || null,
    autoTail: Boolean(options.autoTail),
  };
  if (!options.preserveMeta) {
    inspectorMeta.replaceChildren(metaCard("Path", path));
  }
  inspectorTitle.textContent = title;
  transcriptControls.classList.add("hidden");
  conversationView.classList.add("hidden");
  inspectorContent.classList.remove("hidden");
  const payload = await fetchJson("/api/file", {
    path,
    lines: options.lines || null,
  });
  inspectorContent.textContent = payload.content;
  scheduleInspectorRefresh();
}

async function inspectSession(sessionId, title, options = {}) {
  clearInspectorTimer();
  state.inspector = {
    type: "session",
    sessionId,
    title,
    autoTail: false,
  };
  inspectorTitle.textContent = title;
  if (!options.preserveMeta) {
    inspectorMeta.replaceChildren(metaCard("Session", sessionId));
  }
  inspectorContent.classList.add("hidden");
  conversationView.classList.remove("hidden");
  const payload = await fetchJson("/api/session", { id: sessionId });
  renderSession(payload);
}

function renderSession(payload) {
  inspectorMeta.replaceChildren(
    ...Array.from(inspectorMeta.children),
    metaCard("Transcript file", payload.metadata.path),
    metaCard("Visible items", String(visibleTranscriptItems(payload.items).length)),
    ...(payload.metadata.shell_snapshots || []).slice(0, 2).map((path, index) =>
      metaCard(`Shell snapshot ${index + 1}`, path),
    ),
  );
  transcriptControls.classList.remove("hidden");
  transcriptControls.innerHTML = "";
  const checkbox = document.createElement("input");
  checkbox.type = "checkbox";
  checkbox.checked = state.showHiddenTranscriptItems;
  checkbox.addEventListener("change", (event) => {
    state.showHiddenTranscriptItems = event.target.checked;
    renderTranscriptItems(payload.items);
  });
  const label = document.createElement("label");
  label.className = "toggle";
  label.append(checkbox, document.createTextNode(" Show developer/system transcript items"));
  transcriptControls.append(label);

  const rawButton = document.createElement("button");
  rawButton.type = "button";
  rawButton.className = "button button-muted";
  rawButton.textContent = "Show raw JSONL";
  rawButton.addEventListener("click", () => {
    inspectorContent.classList.remove("hidden");
    inspectorContent.textContent = payload.raw_text;
    conversationView.classList.add("hidden");
  });
  transcriptControls.append(rawButton);

  const transcriptButton = document.createElement("button");
  transcriptButton.type = "button";
  transcriptButton.className = "button button-muted";
  transcriptButton.textContent = "Show parsed transcript";
  transcriptButton.addEventListener("click", () => {
    inspectorContent.classList.add("hidden");
    conversationView.classList.remove("hidden");
    renderTranscriptItems(payload.items);
  });
  transcriptControls.append(transcriptButton);

  renderTranscriptItems(payload.items);
}

function renderTranscriptItems(items) {
  conversationView.replaceChildren();
  visibleTranscriptItems(items).forEach((item) => {
    conversationView.append(buildTranscriptItem(item));
  });
}

function visibleTranscriptItems(items) {
  return items.filter((item) => state.showHiddenTranscriptItems || !item.hidden_by_default);
}

function buildTranscriptItem(item) {
  if (item.kind === "message") {
    return buildMessageItem(item);
  }
  return buildToolItem(item);
}

function buildMessageItem(item) {
  const article = document.createElement("article");
  article.className = `conversation-item conversation-message ${item.role || "unknown"}`;

  const header = document.createElement("header");
  const rolePill = document.createElement("span");
  rolePill.className = "pill role-pill";
  rolePill.textContent = item.role || "message";

  const phase = document.createElement("span");
  phase.className = "phase-tag";
  phase.textContent = item.phase || summarizeMessage(item.text || "");
  header.append(rolePill, phase);

  const body = document.createElement("div");
  body.className = "conversation-message-body";
  body.textContent = item.text || "";

  article.append(header, body);
  return article;
}

function buildToolItem(item) {
  const details = document.createElement("details");
  details.className = "tool-card";

  const summary = document.createElement("summary");
  const left = document.createElement("div");
  left.className = "tool-summary";

  const title = document.createElement("span");
  title.className = "pill role-pill";
  title.textContent = item.kind === "tool_output" ? "tool output" : item.tool_name || "tool";

  const strong = document.createElement("span");
  strong.className = "tool-title";
  strong.textContent =
    item.kind === "tool_output"
      ? "Execution details"
      : item.tool_name || "Tool call";

  const snippet = document.createElement("span");
  snippet.className = "tool-snippet";
  snippet.textContent = summarizeToolItem(item);

  left.append(title, strong, snippet);

  const phase = document.createElement("span");
  phase.className = "phase-tag";
  phase.textContent = item.call_id || "";

  summary.append(left, phase);

  const body = document.createElement("div");
  body.className = "tool-body";
  const pre = document.createElement("pre");
  pre.textContent = prettyToolText(item);
  body.append(pre);

  details.append(summary, body);
  return details;
}

function scheduleInspectorRefresh() {
  clearInspectorTimer();
  if (!state.autoTail || !state.inspector) return;
  if (state.inspector.type !== "file" || !state.inspector.autoTail) return;
  inspectorTimer = window.setTimeout(async () => {
    try {
      const payload = await fetchJson("/api/file", {
        path: state.inspector.path,
        lines: state.inspector.lines || null,
      });
      inspectorContent.textContent = payload.content;
    } catch (error) {
      inspectorContent.textContent = String(error);
    }
    scheduleInspectorRefresh();
  }, 2500);
}

function clearInspectorTimer() {
  if (inspectorTimer !== null) {
    window.clearTimeout(inspectorTimer);
    inspectorTimer = null;
  }
}

function defaultLines(filename) {
  if (filename === "training.log") return 240;
  if (filename === "run.log") return 200;
  return 180;
}

function summarizeMessage(text) {
  const line = text.split("\n").find((candidate) => candidate.trim()) || "";
  return truncate(line.trim(), 120);
}

function summarizeToolItem(item) {
  if (item.kind === "tool_call") {
    if (item.tool_name === "exec_command") {
      const payload = safeJsonParse(item.text);
      if (payload?.cmd) {
        return truncate(payload.cmd, 140);
      }
    }
    return truncate(item.text || "", 140);
  }

  const commandLine = (item.text || "")
    .split("\n")
    .find((line) => line.startsWith("Command: "));
  if (commandLine) {
    return truncate(commandLine.replace("Command: ", ""), 140);
  }
  const outputLine = (item.text || "")
    .split("\n")
    .find((line) => line.trim() && !line.startsWith("Chunk ID:"));
  return truncate(outputLine || "", 140);
}

function prettyToolText(item) {
  if (item.kind === "tool_call") {
    const payload = safeJsonParse(item.text);
    if (payload) {
      return JSON.stringify(payload, null, 2);
    }
  }
  return item.text || "";
}

function safeJsonParse(value) {
  try {
    return JSON.parse(value);
  } catch {
    return null;
  }
}

function truncate(value, maxLength) {
  if (!value) return "";
  if (value.length <= maxLength) return value;
  return `${value.slice(0, maxLength - 1)}…`;
}

function formatPathTail(path) {
  const parts = path.split("/");
  return parts.slice(-3).join("/");
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

window.addEventListener("beforeunload", clearInspectorTimer);

void loadMetastudies().catch((error) => {
  inspectorTitle.textContent = "Failed to load dashboard";
  inspectorContent.textContent = String(error);
});
