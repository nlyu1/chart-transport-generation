const state = {
  metastudies: [],
  selectedMetastudyPath: null,
  metastudyPayload: null,
  expandedNodes: new Set(),
  selectedTreePath: null,
  autoTail: true,
  inspector: null,
  inspectorMetaBase: [],
  showHiddenTranscriptItems: false,
  transcriptMode: "parsed",
};

let inspectorTimer = null;
let workspaceTimer = null;

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
  void refreshCurrentMetastudy({ preserveInspector: true });
});

document.getElementById("open-run-log").addEventListener("click", () => {
  const runLog = state.metastudyPayload?.paths?.run_log;
  if (!runLog) return;
  void inspectFile(runLog, "Latest run.log tail", { lines: 200, autoTail: true });
});

autoTailToggle.addEventListener("change", (event) => {
  state.autoTail = event.target.checked;
  scheduleInspectorRefresh();
  scheduleWorkspaceRefresh();
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

async function refreshCurrentMetastudy(options = {}) {
  if (!state.selectedMetastudyPath) {
    await loadMetastudies();
    return;
  }

  const payload = await fetchJson("/api/metastudies");
  state.metastudies = payload.metastudies;
  renderMetastudyOptions();
  await loadMetastudy(state.selectedMetastudyPath, {
    includeFileTree: true,
    ...options,
  });
}

function renderMetastudyOptions() {
  metastudySelect.innerHTML = "";
  state.metastudies.forEach((metastudy) => {
    const option = document.createElement("option");
    option.value = metastudy.path;
    option.textContent = metastudy.name;
    metastudySelect.append(option);
  });
  if (state.selectedMetastudyPath) {
    metastudySelect.value = state.selectedMetastudyPath;
  }
}

async function loadMetastudy(path, options = {}) {
  const includeFileTree = options.includeFileTree !== false;
  state.metastudyPayload = await fetchJson("/api/metastudy", {
    path,
    include_file_tree: includeFileTree,
  });
  state.selectedMetastudyPath = path;
  renderSummary(state.metastudyPayload.summary);
  if (state.metastudyPayload.file_tree) {
    renderFileTree();
  }
  renderAgentTree();
  scheduleWorkspaceRefresh();
  if (!state.inspector || !options.preserveInspector) {
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
  if (!state.metastudyPayload?.file_tree) return;
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
  state.inspectorMetaBase = [
    ["Path", node.path],
    ["Type", "directory"],
  ];
  inspectorTitle.textContent = `${node.name || node.path}`;
  inspectorActions.replaceChildren();
  renderInspectorMeta(state.inspectorMetaBase);
  transcriptControls.classList.add("hidden");
  conversationView.classList.add("hidden");
  inspectorContent.classList.remove("hidden");
  inspectorContent.textContent = `Directory selected:\n${node.path}`;
}

async function inspectRun(run) {
  state.inspectorMetaBase = [
    ["Role", run.role],
    ["Status", run.status],
    ["Target", run.target_path],
    ["Started", run.start_display || run.start_timestamp || "unknown"],
  ];
  if (run.duration_seconds !== null && run.duration_seconds !== undefined) {
    state.inspectorMetaBase.push(["Duration", `${run.duration_seconds}s`]);
  }
  if (run.session_id) {
    state.inspectorMetaBase.push(["Session", run.session_id]);
  }

  inspectorTitle.textContent = `${run.role} · ${run.target_name}`;
  renderInspectorMeta(state.inspectorMetaBase);
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

function renderInspectorMeta(entries) {
  inspectorMeta.replaceChildren(...entries.map(([label, value]) => metaCard(label, value)));
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
  state.transcriptMode = "parsed";
  if (!options.preserveMeta) {
    state.inspectorMetaBase = [["Path", path]];
    renderInspectorMeta(state.inspectorMetaBase);
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
    autoTail: true,
  };
  inspectorTitle.textContent = title;
  if (!options.preserveMeta) {
    state.inspectorMetaBase = [["Session", sessionId]];
    renderInspectorMeta(state.inspectorMetaBase);
  }
  inspectorContent.classList.add("hidden");
  conversationView.classList.remove("hidden");
  const payload = await fetchJson("/api/session", { id: sessionId });
  renderSession(payload);
  scheduleInspectorRefresh();
}

function renderSession(payload) {
  renderInspectorMeta([
    ...state.inspectorMetaBase,
    ["Transcript file", payload.metadata.path],
    ["Visible items", String(visibleTranscriptItems(payload.items).length)],
    ...(payload.metadata.shell_snapshots || [])
      .slice(0, 2)
      .map((path, index) => [`Shell snapshot ${index + 1}`, path]),
  ]);
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
    state.transcriptMode = "raw";
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
    state.transcriptMode = "parsed";
    inspectorContent.classList.add("hidden");
    conversationView.classList.remove("hidden");
    renderTranscriptItems(payload.items);
  });
  transcriptControls.append(transcriptButton);

  if (state.transcriptMode === "raw") {
    inspectorContent.classList.remove("hidden");
    inspectorContent.textContent = payload.raw_text;
    conversationView.classList.add("hidden");
    return;
  }

  inspectorContent.classList.add("hidden");
  conversationView.classList.remove("hidden");
  renderTranscriptItems(payload.items);
}

function renderTranscriptItems(items) {
  conversationView.replaceChildren();
  groupTranscriptItems(visibleTranscriptItems(items)).forEach((item) => {
    conversationView.append(buildTranscriptItem(item));
  });
}

function visibleTranscriptItems(items) {
  return items.filter((item) => state.showHiddenTranscriptItems || !item.hidden_by_default);
}

function groupTranscriptItems(items) {
  const grouped = [];
  let toolBatch = [];

  for (const item of items) {
    if (item.kind === "message") {
      if (toolBatch.length > 0) {
        grouped.push({ kind: "tool_batch", items: toolBatch });
        toolBatch = [];
      }
      grouped.push(item);
      continue;
    }
    toolBatch.push(item);
  }

  if (toolBatch.length > 0) {
    grouped.push({ kind: "tool_batch", items: toolBatch });
  }

  return grouped;
}

function buildTranscriptItem(item) {
  if (item.kind === "message") {
    return buildMessageItem(item);
  }
  return buildToolBatch(item.items || [item]);
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

function buildToolBatch(items) {
  const details = document.createElement("details");
  details.className = "tool-card tool-batch";

  const summary = document.createElement("summary");
  const left = document.createElement("div");
  left.className = "tool-summary";

  const title = document.createElement("span");
  title.className = "pill role-pill";
  title.textContent = items.length === 1 ? "tool activity" : `${items.length} tool actions`;

  const strong = document.createElement("span");
  strong.className = "tool-title";
  strong.textContent = summarizeToolBatchTitle(items);

  const snippet = document.createElement("span");
  snippet.className = "tool-snippet";
  snippet.textContent = summarizeToolBatch(items);

  left.append(title, strong, snippet);
  summary.append(left);

  const body = document.createElement("div");
  body.className = "tool-body";
  items.forEach((item, index) => {
    body.append(buildToolBatchEntry(item, index));
  });

  details.append(summary, body);
  return details;
}

function buildToolBatchEntry(item, index) {
  const section = document.createElement("section");
  section.className = "tool-entry";

  const header = document.createElement("div");
  header.className = "tool-entry-header";

  const kind = document.createElement("span");
  kind.className = "pill role-pill";
  kind.textContent = item.kind === "tool_output" ? "output" : item.tool_name || "tool";

  const title = document.createElement("strong");
  title.className = "tool-entry-title";
  title.textContent =
    item.kind === "tool_output"
      ? `Result ${index + 1}`
      : summarizeToolEntryTitle(item);

  const snippet = document.createElement("span");
  snippet.className = "tool-entry-snippet";
  snippet.textContent = summarizeToolItem(item);

  header.append(kind, title, snippet);

  const pre = document.createElement("pre");
  pre.textContent = prettyToolText(item);

  section.append(header, pre);
  return section;
}

function scheduleInspectorRefresh() {
  clearInspectorTimer();
  if (!state.autoTail || !state.inspector) return;
  if (!state.inspector.autoTail) return;
  inspectorTimer = window.setTimeout(async () => {
    try {
      if (state.inspector.type === "file") {
        const payload = await fetchJson("/api/file", {
          path: state.inspector.path,
          lines: state.inspector.lines || null,
        });
        inspectorContent.textContent = payload.content;
      } else if (state.inspector.type === "session") {
        const payload = await fetchJson("/api/session", { id: state.inspector.sessionId });
        renderSession(payload);
      }
    } catch (error) {
      inspectorContent.classList.remove("hidden");
      inspectorContent.textContent = `Dashboard server unavailable: ${String(error)}`;
      conversationView.classList.add("hidden");
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

function scheduleWorkspaceRefresh() {
  clearWorkspaceTimer();
  if (!state.autoTail || !state.selectedMetastudyPath) return;
  workspaceTimer = window.setTimeout(async () => {
    try {
      const payload = await fetchJson("/api/metastudy", {
        path: state.selectedMetastudyPath,
        include_file_tree: false,
      });
      state.metastudyPayload = {
        ...state.metastudyPayload,
        ...payload,
        file_tree: state.metastudyPayload?.file_tree,
      };
      renderSummary(state.metastudyPayload.summary);
      renderAgentTree();
    } catch {
      // Leave the current tree visible; the inspector shows the connectivity issue.
    }
    scheduleWorkspaceRefresh();
  }, 4000);
}

function clearWorkspaceTimer() {
  if (workspaceTimer !== null) {
    window.clearTimeout(workspaceTimer);
    workspaceTimer = null;
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

function summarizeToolBatchTitle(items) {
  const labels = [];
  for (const item of items) {
    if (item.kind === "tool_call") {
      labels.push(item.tool_name || "tool");
    } else if (item.kind === "tool_output") {
      labels.push("output");
    }
  }
  return uniqueLabels(labels).slice(0, 4).join(" · ");
}

function summarizeToolBatch(items) {
  const snippets = items
    .slice(0, 3)
    .map((item) => summarizeToolItem(item))
    .filter(Boolean);
  return truncate(snippets.join(" | "), 180);
}

function summarizeToolEntryTitle(item) {
  if (item.tool_name === "exec_command") {
    const payload = safeJsonParse(item.text);
    if (payload?.cmd) {
      return truncate(payload.cmd, 72);
    }
  }
  if (item.tool_name) {
    return item.tool_name;
  }
  return "Tool activity";
}

function uniqueLabels(values) {
  return [...new Set(values)];
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
window.addEventListener("beforeunload", clearWorkspaceTimer);

void loadMetastudies().catch((error) => {
  inspectorTitle.textContent = "Failed to load dashboard";
  inspectorContent.textContent = String(error);
});
