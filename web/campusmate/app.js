/**
 * CampusMate AI — Modern Academic Assistant Frontend Logic
 * ChatGPT-Style Interaction · NSBM & SLIIT Dual University Support
 */

const STORAGE_KEY = "campusmate.conversations.v1";
const SIDEBAR_STATE_KEY = "campusmate.sidebar.collapsed";

/**
 * Verified academic suggested questions categorized across 5 core student domains.
 */
const SUGGESTED_CATEGORIES = [
  "Examination Rules",
  "Degree Programmes",
  "Academic Registration",
  "Academic Regulations",
  "Library Information",
];

const SUGGESTIONS = {
  NSBM: {
    "Examination Rules": [
      "When can candidates enter the examination room?",
      "How long must a candidate wait before leaving the examination hall?",
      "Are mobile phones and smart devices allowed in the examination room?",
      "What identification documents must candidates bring to the examination hall?",
      "How late can a candidate arrive after the official examination start time?",
    ],
    "Degree Programmes": [
      "What are the admission requirements and duration for the BSc in Data Science programme?",
      "What faculties and academic schools are established at NSBM Green University?",
      "What course units and credit structures are required in Data Science?",
      "Are postgraduate master's degree programmes available at NSBM?",
    ],
    "Academic Registration": [
      "What is the procedure and timeline for repeat coursework registration?",
      "How do students register for academic modules each semester?",
      "What are the registration deadlines and fee guidelines for students?",
    ],
    "Academic Regulations": [
      "What are the extenuating circumstances regulations for missed examinations?",
      "What are the guidelines and formatting rules for thesis or dissertation submission?",
      "What constitutes academic misconduct and examination offences under NSBM bylaws?",
      "What are the passing criteria and grading scales for undergraduate degrees?",
    ],
    "Library Information": [
      "What are the library borrowing rules, loan limits, and opening hours at NSBM?",
      "What digital databases, e-resources, and study spaces does the NSBM Library offer?",
    ],
  },
  SLIIT: {
    "Examination Rules": [
      "Which identification documents are required to enter the SLIIT examination hall?",
      "What is the policy and procedure if a student encounters examination timetable clashes?",
      "What are the candidate instructions and rules regarding examination conduct at SLIIT?",
      "What are the attendance requirements for final semester examination eligibility?",
    ],
    "Degree Programmes": [
      "What computing degree programmes and specialisations are offered at SLIIT?",
      "What academic faculties exist across the SLIIT campus structure?",
      "What career pathways, industry internships, and student communities exist at SLIIT?",
    ],
    "Academic Registration": [
      "According to general timelines, when should students complete semester registration for the January-June and July-December semesters?",
      "What does SLIIT say about late semester registration fees and deadlines?",
      "How do students complete module registration and course enrollment at SLIIT?",
    ],
    "Academic Regulations": [
      "What is the academic integrity, plagiarism, and code of conduct policy at SLIIT?",
      "What are the eligibility criteria and conditions for academic performance scholarships at SLIIT?",
      "What are the grading scales, GPA criteria, and progression rules for undergraduates?",
    ],
    "Library Information": [
      "What services, study spaces, and learning resources are available at the SLIIT Library Learning Commons?",
      "How can students access online journals, research databases, and library study support at SLIIT?",
    ],
  },
};

const state = {
  campus: "NSBM",
  conversations: [],
  activeId: null,
  view: "chat",
  sending: false,
  activeCategory: "Examination Rules",
  sidebarCollapsed: false,
  evaluationData: null,
  evalFilter: "all",
  serverReady: false,
};

const els = {
  sidebar: document.getElementById("sidebar"),
  backdrop: document.getElementById("sidebar-backdrop"),
  menu: document.getElementById("menu-button"),
  sidebarCollapseBtn: document.getElementById("sidebar-collapse-btn"),
  sidebarExpandBtn: document.getElementById("sidebar-expand-btn"),
  newChat: document.getElementById("new-chat"),
  history: document.getElementById("history-list"),
  transcript: document.getElementById("transcript"),
  welcome: document.getElementById("welcome"),
  welcomeCampusName: document.getElementById("welcome-campus-name"),
  categoryTabs: document.getElementById("category-tabs"),
  suggestions: document.getElementById("suggestions"),
  composer: document.getElementById("composer"),
  question: document.getElementById("question"),
  send: document.getElementById("send"),
  main: document.getElementById("main"),
  chatView: document.getElementById("chat-view"),
  evaluationView: document.getElementById("evaluation-view"),
  evaluationContent: document.getElementById("evaluation-content"),
  sidebarCampusPill: document.getElementById("sidebar-campus-pill"),
  viewTitle: document.getElementById("view-title"),
  openChat: document.getElementById("open-chat"),
  openEvaluation: document.getElementById("open-evaluation"),
  topbarChatBtn: document.getElementById("topbar-chat-btn"),
  topbarEvalBtn: document.getElementById("topbar-eval-btn"),
  statusIndicator: document.querySelector(".status-indicator"),
  statusText: document.getElementById("status-text"),
};

/* ==========================================================================
   Storage & State Persistence
   ========================================================================== */

function loadStored() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    const parsed = raw ? JSON.parse(raw) : [];
    state.conversations = Array.isArray(parsed)
      ? parsed.map((item) => ({
          ...item,
          messages: Array.isArray(item.messages) ? item.messages : [],
        }))
      : [];
  } catch {
    state.conversations = [];
  }

  try {
    state.sidebarCollapsed = localStorage.getItem(SIDEBAR_STATE_KEY) === "true";
  } catch {
    state.sidebarCollapsed = false;
  }
}

function saveStored() {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(state.conversations));
  } catch (err) {
    console.error("Failed to save conversations to localStorage:", err);
  }
}

function saveSidebarState() {
  try {
    localStorage.setItem(SIDEBAR_STATE_KEY, String(state.sidebarCollapsed));
  } catch (err) {
    console.error("Failed to save sidebar state:", err);
  }
}

function uid() {
  if (window.crypto && crypto.randomUUID) return crypto.randomUUID();
  return `chat-${Date.now()}-${Math.random().toString(16).slice(2, 10)}`;
}

function activeConversation() {
  return state.conversations.find((item) => item.id === state.activeId) || null;
}

function createConversation(campus) {
  const targetCampus = campus || state.campus;
  const now = new Date().toISOString();
  const conversation = {
    id: uid(),
    title: "New chat",
    campus: targetCampus,
    createdAt: now,
    updatedAt: now,
    messages: [],
  };
  state.conversations.unshift(conversation);
  state.activeId = conversation.id;
  saveStored();
  return conversation;
}

function ensureActive() {
  const current = activeConversation();
  if (current && current.campus === state.campus) return current;

  // Search for an existing empty chat for this campus
  const empty = state.conversations.find(
    (item) => item.campus === state.campus && (!item.messages || item.messages.length === 0)
  );
  if (empty) {
    state.activeId = empty.id;
    return empty;
  }
  return createConversation(state.campus);
}

/* ==========================================================================
   History Formatting & Grouping
   ========================================================================== */

function startOfDay(date) {
  return new Date(date.getFullYear(), date.getMonth(), date.getDate());
}

function historyGroup(iso) {
  const updated = startOfDay(new Date(iso));
  const today = startOfDay(new Date());
  const yesterday = new Date(today);
  yesterday.setDate(today.getDate() - 1);
  const previousWeekStart = new Date(today);
  previousWeekStart.setDate(today.getDate() - 7);

  if (updated.getTime() === today.getTime()) return "Today";
  if (updated.getTime() === yesterday.getTime()) return "Yesterday";
  if (updated >= previousWeekStart) return "Previous 7 Days";
  return "Older";
}

/* ==========================================================================
   Markdown & HTML Rendering
   ========================================================================== */

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function renderInline(text) {
  let out = escapeHtml(text);
  // Code spans
  out = out.replace(/`([^`]+)`/g, "<code>$1</code>");
  // Bold
  out = out.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  out = out.replace(/__([^_]+)__/g, "<strong>$1</strong>");
  // Italics
  out = out.replace(/\*([^*]+)\*/g, "<em>$1</em>");
  out = out.replace(/_([^_]+)_/g, "<em>$1</em>");
  // Highlight academic citation markers like [S1], [S2, S3]
  out = out.replace(/\[(S\d+(?:,\s*S\d+)*)\]/g, '<span class="citation-badge" title="Official source citation">[$1]</span>');
  return out;
}

function renderMarkdown(text) {
  const lines = String(text || "").split(/\r?\n/);
  const html = [];
  let list = null;
  let inCodeBlock = false;
  let codeBuffer = [];

  const closeList = () => {
    if (list) {
      html.push(list === "ol" ? "</ol>" : "</ul>");
      list = null;
    }
  };

  for (let i = 0; i < lines.length; i++) {
    const rawLine = lines[i];

    // Code blocks
    if (rawLine.trim().startsWith("```")) {
      closeList();
      if (inCodeBlock) {
        html.push(`<pre><code>${escapeHtml(codeBuffer.join("\n"))}</code></pre>`);
        codeBuffer = [];
        inCodeBlock = false;
      } else {
        inCodeBlock = true;
      }
      continue;
    }

    if (inCodeBlock) {
      codeBuffer.push(rawLine);
      continue;
    }

    const trimmed = rawLine.trim();

    if (!trimmed) {
      closeList();
      continue;
    }

    // Headings
    const headingMatch = rawLine.match(/^(#{1,4})\s+(.+)$/);
    if (headingMatch) {
      closeList();
      const level = headingMatch[1].length;
      html.push(`<h${level}>${renderInline(headingMatch[2])}</h${level}>`);
      continue;
    }

    // Blockquotes
    if (rawLine.startsWith(">")) {
      closeList();
      const quoteText = rawLine.replace(/^>\s?/, "");
      html.push(`<blockquote>${renderInline(quoteText)}</blockquote>`);
      continue;
    }

    // Lists
    const ordered = rawLine.match(/^\s*(\d+)\.\s+(.*)/);
    const unordered = rawLine.match(/^\s*[-*•]\s+(.*)/);
    if (ordered || unordered) {
      const next = ordered ? "ol" : "ul";
      if (list !== next) {
        closeList();
        html.push(next === "ol" ? "<ol>" : "<ul>");
        list = next;
      }
      html.push(`<li>${renderInline((ordered || unordered)[ordered ? 2 : 1])}</li>`);
      continue;
    }

    closeList();

    // Tables
    if (trimmed.startsWith("|") && trimmed.endsWith("|")) {
      const cells = trimmed.split("|").slice(1, -1).map((c) => c.trim());
      if (cells.every((c) => /^:?-+:?$/.test(c))) {
        continue;
      }
      const isHeader = i > 0 && lines[i + 1] && lines[i + 1].trim().startsWith("|") && /[-:]/.test(lines[i + 1]);
      const tag = isHeader ? "th" : "td";
      const cellHtml = cells.map((c) => `<${tag}>${renderInline(c)}</${tag}>`).join("");
      html.push(`<table><tr>${cellHtml}</tr></table>`);
      continue;
    }

    // Default paragraph
    html.push(`<p>${renderInline(rawLine)}</p>`);
  }

  closeList();
  if (inCodeBlock && codeBuffer.length) {
    html.push(`<pre><code>${escapeHtml(codeBuffer.join("\n"))}</code></pre>`);
  }

  return html.join("");
}

function fileName(path) {
  const text = String(path || "Academic Source Document");
  const parts = text.split(/[/\\]/);
  return parts[parts.length - 1] || text;
}

function formatScore(value) {
  if (typeof value !== "number" || Number.isNaN(value)) return "n/a";
  return value.toFixed(4);
}

function formatValue(value) {
  if (value === null || value === undefined || value === "") return "Not available";
  if (typeof value === "number") {
    return Number.isInteger(value) ? String(value) : String(Math.round(value * 1000) / 1000);
  }
  return String(value);
}

function formatPercentage(value) {
  if (typeof value !== "number" || Number.isNaN(value)) return "n/a";
  return `${(value * 100).toFixed(1)}%`;
}

/* ==========================================================================
   Campus Switching
   ========================================================================== */

function setCampus(campus, options = {}) {
  if (campus !== "NSBM" && campus !== "SLIIT") return;

  const campusChanged = campus !== state.campus;
  state.campus = campus;

  // Sync all campus buttons in sidebar, topbar, and welcome banner
  document.querySelectorAll("[data-campus]").forEach((button) => {
    button.classList.toggle("active", button.dataset.campus === campus);
  });

  if (els.sidebarCampusPill) els.sidebarCampusPill.textContent = campus;
  if (els.welcomeCampusName) {
    els.welcomeCampusName.textContent =
      campus === "NSBM" ? "NSBM Green University" : "SLIIT (Sri Lanka Institute of Information Technology)";
  }

  if (options.restore) {
    renderHistory();
    renderTranscript();
    return;
  }

  if (campusChanged) {
    createConversation(campus);
  } else {
    ensureActive();
  }

  renderHistory();
  renderTranscript();
}

/* ==========================================================================
   Chat History Sidebar
   ========================================================================== */

function renderHistory() {
  const groups = ["Today", "Yesterday", "Previous 7 Days", "Older"];
  const buckets = Object.fromEntries(groups.map((name) => [name, []]));

  state.conversations
    .slice()
    .sort((a, b) => new Date(b.updatedAt) - new Date(a.updatedAt))
    .forEach((conversation) => {
      buckets[historyGroup(conversation.updatedAt)].push(conversation);
    });

  const html = groups
    .filter((name) => buckets[name].length > 0)
    .map((name) => {
      const items = buckets[name]
        .map((conversation) => {
          const isActive = conversation.id === state.activeId;
          const campusBadge = conversation.campus || "NSBM";
          return `
            <div class="history-item ${isActive ? "active" : ""}">
              <button class="history-open-btn" type="button" data-open="${conversation.id}" title="${escapeHtml(conversation.title)}">
                <span class="history-campus-badge">${escapeHtml(campusBadge)}</span>
                <span class="history-title">${escapeHtml(conversation.title)}</span>
              </button>
              <button class="delete-chat-btn" type="button" data-delete="${conversation.id}" title="Delete conversation" aria-label="Delete conversation">
                <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                  <polyline points="3 6 5 6 21 6"/>
                  <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/>
                </svg>
              </button>
            </div>
          `;
        })
        .join("");

      return `
        <section class="history-group">
          <h3 class="history-group-title">${name}</h3>
          ${items}
        </section>
      `;
    })
    .join("");

  els.history.innerHTML = html || '<p class="empty-history">No saved conversations yet.</p>';
}

/* ==========================================================================
   Explainable AI Sources & Citations Display
   ========================================================================== */

function evidenceHtml(message) {
  const evidence = message.evidence;
  if (!evidence) return "";

  const citations = evidence.sources || [];
  const rawChunks = evidence.retrieved_chunks || [];
  const externalSources = evidence.external_sources || [];
  const toolErrors = evidence.tool_errors || [];
  const reasoning = evidence.reasoning || {};

  // Hide exact duplicate passages from the same document, but retain
  // all page and chunk identifiers and the strongest recorded score.
  const uniqueChunkMap = new Map();
  rawChunks.forEach((chunk) => {
    const documentKey = String(chunk.document || "").replace(/\\/g, "/").toLowerCase();
    const passageKey = String(chunk.excerpt || "").trim().replace(/\s+/g, " ").toLowerCase();
    const key = `${documentKey}::${passageKey || chunk.chunk_id || uniqueChunkMap.size}`;
    const existing = uniqueChunkMap.get(key);
    if (!existing) {
      uniqueChunkMap.set(key, {
        ...chunk,
        page_numbers: [...new Set(chunk.page_numbers || [])].sort((a, b) => a - b),
        chunk_ids: [...new Set([...(chunk.chunk_ids || []), chunk.chunk_id].filter(Boolean))],
      });
      return;
    }
    existing.page_numbers = [...new Set([...(existing.page_numbers || []), ...(chunk.page_numbers || [])])].sort((a, b) => a - b);
    existing.chunk_ids = [...new Set([...(existing.chunk_ids || []), chunk.chunk_id].filter(Boolean))];
    if (Number(chunk.hybrid_score) > Number(existing.hybrid_score)) {
      existing.hybrid_score = chunk.hybrid_score;
      existing.dense_score = chunk.dense_score;
      existing.tfidf_score = chunk.tfidf_score;
    }
  });
  const chunks = Array.from(uniqueChunkMap.values());

  const totalCitations = citations.length;
  const totalChunks = chunks.length;

  // Reasoning breakdown
  const reasoningSection = `
    <div class="reasoning-panel">
      <div class="reasoning-meta">
        <span>Intent: <strong>${escapeHtml(reasoning.intent || "academic_inquiry")}</strong></span>
        <span>Retrieval Depth: <strong>Top ${escapeHtml(reasoning.retrieval_top_k ?? "5")}</strong></span>
        <span>Retrieval Attempts: <strong>${escapeHtml(reasoning.retrieval_attempts ?? "1")}</strong></span>
      </div>
      ${reasoning.rationale ? `<p class="reasoning-rationale">${escapeHtml(reasoning.rationale)}</p>` : ""}
    </div>
  `;

  // Academic citations
  const citationsHtml = citations.length
    ? `
      <div class="evidence-section-title">Official Academic Citations (${citations.length})</div>
      ${citations
        .map(
          (source) => `
        <article class="citation-card">
          <div class="citation-header">
            <span class="citation-id">${escapeHtml(source.citation_id || "Source")}</span>
            <span class="citation-doc" title="${escapeHtml(source.document)}">${escapeHtml(fileName(source.document))}</span>
          </div>
          <p class="citation-meta">Official Document Page(s): <strong>${escapeHtml((source.page_numbers || []).join(", ") || "Unspecified")}</strong></p>
          <p class="citation-meta">Evidence Chunk ID(s): <strong>${escapeHtml((source.chunk_ids || [source.chunk_id].filter(Boolean)).join(", ") || "Unspecified")}</strong></p>
        </article>
      `
        )
        .join("")}
    `
    : '<p class="warning-box">No formal academic citations were referenced for this response.</p>';

  // Retrieved chunks with mathematical hybrid scores
  const chunksHtml = chunks.length
    ? `
      <div class="evidence-section-title">Retrieved Evidence Passages (${chunks.length})</div>
      ${chunks
        .map(
          (chunk, index) => `
        <article class="chunk-card">
          <div class="chunk-header">
            <h4 class="chunk-title">${index + 1}. ${escapeHtml(fileName(chunk.document))}</h4>
            <span class="score-pill">Page(s): <strong>${escapeHtml((chunk.page_numbers || []).join(", ") || "1")}</strong></span>
          </div>
          ${chunk.heading ? `<p class="chunk-heading">Section: ${escapeHtml(chunk.heading)}</p>` : ""}
          <p class="chunk-excerpt">${escapeHtml(chunk.excerpt || "")}</p>
          <p class="citation-meta">Chunk ID(s): <strong>${escapeHtml((chunk.chunk_ids || [chunk.chunk_id].filter(Boolean)).join(", ") || "Unspecified")}</strong></p>
          <div class="score-row">
            <span class="score-pill">Hybrid Score: <strong>${formatScore(chunk.hybrid_score)}</strong></span>
            <span class="score-pill">Dense Cosine: <strong>${formatScore(chunk.dense_score)}</strong></span>
            <span class="score-pill">TF-IDF Sparse: <strong>${formatScore(chunk.tfidf_score)}</strong></span>
          </div>
        </article>
      `
        )
        .join("")}
    `
    : "";

  // External / Weather sources
  const externalHtml = externalSources.length
    ? `
      <div class="evidence-section-title">External Real-Time Evidence (${externalSources.length})</div>
      ${externalSources
        .map(
          (source) => `
        <article class="citation-card">
          <div class="citation-header">
            <span class="citation-id">${escapeHtml(source.citation_id || "Live")}</span>
            <span class="citation-doc">${escapeHtml(source.source_name || "Live Source")}</span>
          </div>
          <p class="citation-meta">Timestamp: ${escapeHtml(source.data_timestamp || "Live")} ${source.valid_until ? `· Valid until: ${escapeHtml(source.valid_until)}` : ""}</p>
        </article>
      `
        )
        .join("")}
    `
    : "";

  // Warnings
  const warningsHtml = toolErrors.length
    ? toolErrors.map((item) => `<p class="warning-box">⚠️ ${escapeHtml(item)}</p>`).join("")
    : "";

  return `
    <details class="evidence">
      <summary>
        <div class="evidence-summary-title">
          <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"/>
            <path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"/>
          </svg>
          <strong>Sources &amp; Evidence</strong>
          <span class="evidence-count-badge">${totalCitations} Citations · ${totalChunks} Chunks</span>
        </div>
        <span class="evidence-chevron">▼</span>
      </summary>
      <div class="evidence-body">
        ${reasoningSection}
        ${warningsHtml}
        ${citationsHtml}
        ${chunksHtml}
        ${externalHtml}
      </div>
    </details>
  `;
}

/* ==========================================================================
   Transcript & Suggested Questions
   ========================================================================== */

function renderSuggestedCategories() {
  const tabsHtml = SUGGESTED_CATEGORIES.map((cat) => {
    const isActive = cat === state.activeCategory;
    return `
      <button class="category-tab ${isActive ? "active" : ""}" type="button" data-category="${escapeHtml(cat)}">
        ${escapeHtml(cat)}
      </button>
    `;
  }).join("");

  els.categoryTabs.innerHTML = tabsHtml;

  const currentQuestions = (SUGGESTIONS[state.campus] && SUGGESTIONS[state.campus][state.activeCategory]) || [];
  const cardsHtml = currentQuestions
    .map(
      (text) => `
    <button class="suggestion-card" type="button" data-suggestion="${escapeHtml(text)}">
      <span class="suggestion-icon">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
          <polyline points="9 18 15 12 9 6"/>
        </svg>
      </span>
      <span>${escapeHtml(text)}</span>
    </button>
  `
    )
    .join("");

  els.suggestions.innerHTML = cardsHtml;
}

function renderTranscript() {
  const conversation = ensureActive();
  const hasMessages = conversation.messages && conversation.messages.length > 0;

  els.chatView.classList.toggle("has-messages", hasMessages);
  els.welcome.hidden = hasMessages;
  els.welcome.setAttribute("aria-hidden", hasMessages ? "true" : "false");
  if (!hasMessages) {
    renderSuggestedCategories();
  }

  // Remove existing rendered messages
  const existing = els.transcript.querySelectorAll(".message");
  existing.forEach((node) => node.remove());

  if (!hasMessages) return;

  conversation.messages.forEach((message, idx) => {
    const article = document.createElement("article");
    article.className = `message ${message.role}`;

    const isAssistant = message.role === "assistant";
    const avatarMark = isAssistant ? "C" : "U";
    const senderName = isAssistant ? "CampusMate AI" : "You";

    const header = `
      <div class="message-header">
        <div class="avatar ${isAssistant ? "assistant-avatar" : "user-avatar"}">${avatarMark}</div>
        <span class="sender-name">${senderName}</span>
      </div>
    `;

    const bodyContent = isAssistant ? renderMarkdown(message.content) : `<p>${escapeHtml(message.content)}</p>`;
    const bubble = `<div class="bubble">${bodyContent}</div>`;

    const copyBar = isAssistant
      ? `
        <div class="message-actions">
          <button class="copy-btn" type="button" data-copy-idx="${idx}" title="Copy response to clipboard">
            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
              <rect x="9" y="9" width="13" height="13" rx="2" ry="2"/>
              <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/>
            </svg>
            <span>Copy</span>
          </button>
        </div>
      `
      : "";

    const evidence = isAssistant ? evidenceHtml(message) : "";

    article.innerHTML = header + bubble + evidence + copyBar;
    els.transcript.appendChild(article);
  });

  els.transcript.scrollTop = els.transcript.scrollHeight;
}

/* ==========================================================================
   Chat Messaging & API Calls
   ========================================================================== */

async function sendQuestion(text) {
  const question = (text || "").trim();
  if (!question || state.sending) return;

  const conversation = ensureActive();

  // Append user message
  conversation.messages.push({
    role: "user",
    content: question,
    createdAt: new Date().toISOString(),
  });

  if (conversation.title === "New chat") {
    conversation.title = question.slice(0, 60);
  }
  conversation.updatedAt = new Date().toISOString();
  saveStored();

  els.question.value = "";
  resizeComposer();
  renderHistory();
  renderTranscript();

  state.sending = true;
  els.send.disabled = true;

  // Append pending assistant typing bubble
  const pending = document.createElement("article");
  pending.className = "message assistant pending";
  pending.id = "pending-bubble";
  pending.innerHTML = `
    <div class="message-header">
      <div class="avatar assistant-avatar">C</div>
      <span class="sender-name">CampusMate AI</span>
    </div>
    <div class="bubble">
      <div class="typing-indicator">
        <span class="typing-dot"></span>
        <span class="typing-dot"></span>
        <span class="typing-dot"></span>
      </div>
      <p class="loading-label">Searching verified ${escapeHtml(conversation.campus)} documents and generating answer…</p>
    </div>
  `;
  els.transcript.appendChild(pending);
  els.transcript.scrollTop = els.transcript.scrollHeight;

  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        institution_id: conversation.campus,
        question: question,
      }),
    });

    const payload = await response.json().catch(() => ({}));

    if (!response.ok) {
      throw new Error(payload.error || `Server returned error status ${response.status}`);
    }

    conversation.messages.push({
      role: "assistant",
      content: payload.answer || "No response text was returned.",
      createdAt: payload.created_at || new Date().toISOString(),
      evidence: payload,
    });
  } catch (error) {
    conversation.messages.push({
      role: "assistant",
      content: `⚠️ **Unable to complete request:** ${error.message || "An unexpected network error occurred."}`,
      createdAt: new Date().toISOString(),
      evidence: null,
    });
  } finally {
    const pendingEl = document.getElementById("pending-bubble");
    if (pendingEl) pendingEl.remove();

    conversation.updatedAt = new Date().toISOString();
    saveStored();
    state.sending = false;
    els.send.disabled = false;
    renderHistory();
    renderTranscript();
    els.question.focus();
  }
}

function resizeComposer() {
  els.question.style.height = "auto";
  els.question.style.height = `${Math.min(els.question.scrollHeight, 180)}px`;
}

function copyMessageContent(index, button) {
  const conversation = activeConversation();
  if (!conversation) return;
  const msg = conversation.messages[index];
  if (!msg) return;

  const textToCopy = msg.content;
  if (navigator.clipboard && navigator.clipboard.writeText) {
    navigator.clipboard.writeText(textToCopy).then(() => {
      showCopiedFeedback(button);
    }).catch(() => {
      fallbackCopy(textToCopy, button);
    });
  } else {
    fallbackCopy(textToCopy, button);
  }
}

function fallbackCopy(text, button) {
  const ta = document.createElement("textarea");
  ta.value = text;
  ta.style.position = "fixed";
  ta.style.opacity = "0";
  document.body.appendChild(ta);
  ta.select();
  try {
    document.execCommand("copy");
    showCopiedFeedback(button);
  } catch (e) {
    console.error("Clipboard copy failed:", e);
  } finally {
    ta.remove();
  }
}

function showCopiedFeedback(button) {
  button.classList.add("copied");
  const span = button.querySelector("span");
  if (span) span.textContent = "Copied!";
  setTimeout(() => {
    button.classList.remove("copied");
    if (span) span.textContent = "Copy";
  }, 2000);
}

/* ==========================================================================
   Evaluation Reports Dashboard
   ========================================================================== */

async function loadEvaluation() {
  els.evaluationContent.innerHTML = `
    <div class="loading-state">
      <div class="spinner"></div>
      <p>Reading evaluation reports from disk…</p>
    </div>
  `;

  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 12000);

  try {
    const response = await fetch("/api/evaluation", {
      signal: controller.signal,
      headers: { Accept: "application/json" },
    });
    clearTimeout(timeoutId);

    if (!response.ok) {
      const errPayload = await response.json().catch(() => ({}));
      throw new Error(errPayload.error || `HTTP ${response.status} failed to load reports.`);
    }

    const payload = await response.json();
    state.evaluationData = payload;
    renderEvaluation(payload);
  } catch (error) {
    clearTimeout(timeoutId);
    console.warn("Failed to fetch /api/evaluation:", error);
    const message = error.name === "AbortError" ? "Request timed out while loading reports." : error.message;

    // Show friendly error state with retry button
    els.evaluationContent.innerHTML = `
      <div class="error-state">
        <p class="warning-box">⚠️ <strong>Could not load evaluation reports:</strong> ${escapeHtml(message)}</p>
        <p class="meta" style="margin-top: 10px; color: var(--text-muted); font-size: 0.84rem;">
          Make sure the CampusMate server is running on <code>http://127.0.0.1:8000</code>.
        </p>
        <button class="retry-btn" type="button" onclick="loadEvaluation()">Retry Loading</button>
      </div>
    `;
  }
}

function renderEvaluation(payload) {
  const institutions = payload.institutions || {};
  const filter = state.evalFilter || "all";

  const nsbmItem = institutions["NSBM"] || {};
  const sliitItem = institutions["SLIIT"] || {};

  const nsbmQ = (nsbmItem.quality && nsbmItem.quality.metrics) || {};
  const nsbmB = (nsbmItem.batch && nsbmItem.batch.metrics) || {};
  const sliitQ = (sliitItem.quality && sliitItem.quality.metrics) || {};
  const sliitB = (sliitItem.batch && sliitItem.batch.metrics) || {};

  // Aggregate only metrics present in the current report files. Missing data
  // is shown as unavailable rather than replaced with optimistic demo values.
  const meanAvailable = (values) => {
    const valid = values.filter((value) => typeof value === "number" && Number.isFinite(value));
    return valid.length ? valid.reduce((sum, value) => sum + value, 0) / valid.length : null;
  };
  const totalEvaluated = (nsbmB.total_questions ?? 0) + (sliitB.total_questions ?? 0);
  const avgGroundedness = meanAvailable([
    nsbmQ.groundedness_mean_1_to_5_judged_only,
    sliitQ.groundedness_mean_1_to_5_judged_only,
  ]);
  const avgAnswerRel = meanAvailable([
    nsbmQ.answer_relevance_mean_1_to_5_judged_only,
    sliitQ.answer_relevance_mean_1_to_5_judged_only,
  ]);
  const avgContextRel = meanAvailable([
    nsbmQ.context_relevance_mean_1_to_5_answerable_only,
    sliitQ.context_relevance_mean_1_to_5_answerable_only,
  ]);
  const avgAccuracy = meanAvailable([
    nsbmB.answerability_accuracy,
    sliitB.answerability_accuracy,
  ]);

  // Executive KPI cards
  const kpiSection = `
    <div class="kpi-grid">
      <div class="kpi-card">
        <span class="kpi-title">Groundedness (Faithfulness)</span>
        <div class="kpi-value-row">
          <span class="kpi-value">${formatValue(avgGroundedness)}</span>
          <span class="kpi-scale">/ 5.0</span>
        </div>
        <span class="kpi-subtext">LLM-as-a-Judge Hallucination Test</span>
      </div>

      <div class="kpi-card">
        <span class="kpi-title">Answer Relevance</span>
        <div class="kpi-value-row">
          <span class="kpi-value">${formatValue(avgAnswerRel)}</span>
          <span class="kpi-scale">/ 5.0</span>
        </div>
        <span class="kpi-subtext">Query Intent Alignment Mean</span>
      </div>

      <div class="kpi-card">
        <span class="kpi-title">Context Relevance</span>
        <div class="kpi-value-row">
          <span class="kpi-value">${formatValue(avgContextRel)}</span>
          <span class="kpi-scale">/ 5.0</span>
        </div>
        <span class="kpi-subtext">LLM-as-a-judge metric, when available</span>
      </div>

      <div class="kpi-card">
        <span class="kpi-title">Answerability Accuracy</span>
        <div class="kpi-value-row">
          <span class="kpi-value">${formatPercentage(avgAccuracy)}</span>
        </div>
        <span class="kpi-subtext">${totalEvaluated || "Not available"} Total Evaluated Questions</span>
      </div>
    </div>
  `;

  // Cross-University Benchmark Comparison Table
  const comparisonTable = `
    <div class="comparison-card">
      <div class="comparison-card-header">
        <h3>Cross-University Benchmark Comparison</h3>
        <span class="eval-badge">NSBM &amp; SLIIT Ground Truth</span>
      </div>
      <div class="comparison-table-wrapper">
        <table class="comparison-table">
          <thead>
            <tr>
              <th>Evaluation Metric</th>
              <th>NSBM Green University</th>
              <th>SLIIT</th>
              <th>Benchmark Status</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>Total Evaluated Questions</td>
              <td>${formatValue(nsbmB.total_questions)}</td>
              <td>${formatValue(sliitB.total_questions)}</td>
              <td><span class="score-pill">Current report counts</span></td>
            </tr>
            <tr>
              <td>Answerable / Unanswerable</td>
              <td>${formatValue(nsbmB.answerable_questions)} / ${formatValue(nsbmB.unanswerable_questions)}</td>
              <td>${formatValue(sliitB.answerable_questions)} / ${formatValue(sliitB.unanswerable_questions)}</td>
              <td><span class="score-pill">Ground-truth label counts</span></td>
            </tr>
            <tr>
              <td>Answerability Accuracy</td>
              <td>${formatPercentage(nsbmB.answerability_accuracy)}</td>
              <td>${formatPercentage(sliitB.answerability_accuracy)}</td>
              <td><strong style="color: var(--success);">From current evaluation</strong></td>
            </tr>
            <tr>
              <td>Context Hit Rate @ 30</td>
              <td>${formatPercentage(nsbmB.context_hit_rate_at_30)}</td>
              <td>${formatPercentage(sliitB.context_hit_rate_at_30)}</td>
              <td><strong style="color: var(--success);">From current evaluation</strong></td>
            </tr>
            <tr>
              <td>Supporting Fact Coverage Proxy</td>
              <td>${formatPercentage(nsbmB.supporting_fact_coverage_proxy)}</td>
              <td>${formatPercentage(sliitB.supporting_fact_coverage_proxy)}</td>
              <td><strong style="color: var(--accent-light);">Lexical proxy only</strong></td>
            </tr>
            <tr>
              <td>Unsupported Rejection Rate</td>
              <td>${formatPercentage(nsbmB.unsupported_rejection_rate)}</td>
              <td>${formatPercentage(sliitB.unsupported_rejection_rate)}</td>
              <td><strong style="color: var(--success);">From current evaluation</strong></td>
            </tr>
            <tr>
              <td>LLM Judge: Context Relevance</td>
              <td>${formatValue(nsbmQ.context_relevance_mean_1_to_5_answerable_only)} / 5.0</td>
              <td>${formatValue(sliitQ.context_relevance_mean_1_to_5_answerable_only)} / 5.0</td>
              <td><strong style="color: var(--success);">Judge report only</strong></td>
            </tr>
            <tr>
              <td>LLM Judge: Groundedness</td>
              <td>${formatValue(nsbmQ.groundedness_mean_1_to_5_judged_only)} / 5.0</td>
              <td>${formatValue(sliitQ.groundedness_mean_1_to_5_judged_only)} / 5.0</td>
              <td><strong style="color: var(--accent-light);">Judge report only</strong></td>
            </tr>
            <tr>
              <td>LLM Judge: Answer Relevance</td>
              <td>${formatValue(nsbmQ.answer_relevance_mean_1_to_5_judged_only)} / 5.0</td>
              <td>${formatValue(sliitQ.answer_relevance_mean_1_to_5_judged_only)} / 5.0</td>
              <td><strong style="color: var(--success);">Judge report only</strong></td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  `;

  // Filtered detailed university cards
  const campusesToShow = filter === "all" ? ["NSBM", "SLIIT"] : [filter];

  const cardsHtml = campusesToShow
    .map((campus) => {
      const item = institutions[campus] || {};
      const quality = (item.quality && item.quality.metrics) || {};
      const batch = (item.batch && item.batch.metrics) || {};
      const metadata = item.metadata || {};

      const contextRel = quality.context_relevance_mean_1_to_5_answerable_only ?? null;
      const groundedness = quality.groundedness_mean_1_to_5_judged_only ?? null;
      const answerRel = quality.answer_relevance_mean_1_to_5_judged_only ?? null;

      const totalQuestions = batch.total_questions ?? null;
      const answerable = batch.answerable_questions ?? null;
      const unanswerable = batch.unanswerable_questions ?? null;
      const accuracy = batch.answerability_accuracy ?? null;
      const contextHit = batch.context_hit_rate_at_30 ?? null;
      const contextPrecision = batch.context_precision_at_30 ?? null;
      const factCoverage = batch.supporting_fact_coverage_proxy ?? null;
      const rejectionRate = batch.unsupported_rejection_rate ?? null;

      return `
        <article class="eval-card">
          <div class="eval-card-header">
            <div class="eval-card-title-group">
              <h3>${campus === "NSBM" ? "NSBM Green University — Comprehensive Benchmark" : "SLIIT — Academic Administration Benchmark"}</h3>
              <span class="eval-source-label">Source: rag_quality_judgement_report.json &amp; ${campus.toLowerCase()}_final_report.json</span>
            </div>
            <span class="eval-badge">${campus} Evaluation</span>
          </div>

          <!-- 3-score gauge grid -->
          <div class="metric-grid">
            <div class="metric-box">
              <span class="metric-label">Context Relevance</span>
              <div class="metric-value-row">
                <span class="metric-value">${formatValue(contextRel)}</span>
                <span class="metric-scale">/ 5.0</span>
              </div>
              <div class="metric-bar">
                <div class="metric-bar-fill" style="width: ${typeof contextRel === "number" ? Math.max(0, Math.min(100, (contextRel / 5) * 100)) : 0}%"></div>
              </div>
            </div>

            <div class="metric-box">
              <span class="metric-label">Groundedness (Faithfulness)</span>
              <div class="metric-value-row">
                <span class="metric-value">${formatValue(groundedness)}</span>
                <span class="metric-scale">/ 5.0</span>
              </div>
              <div class="metric-bar">
                <div class="metric-bar-fill" style="width: ${typeof groundedness === "number" ? Math.max(0, Math.min(100, (groundedness / 5) * 100)) : 0}%"></div>
              </div>
            </div>

            <div class="metric-box">
              <span class="metric-label">Answer Relevance</span>
              <div class="metric-value-row">
                <span class="metric-value">${formatValue(answerRel)}</span>
                <span class="metric-scale">/ 5.0</span>
              </div>
              <div class="metric-bar">
                <div class="metric-bar-fill" style="width: ${typeof answerRel === "number" ? Math.max(0, Math.min(100, (answerRel / 5) * 100)) : 0}%"></div>
              </div>
            </div>
          </div>

          <div class="evidence-section-title">Batch Retrieval &amp; Accuracy Statistics</div>
          <div class="stats-grid">
            <div class="stat-pill">
              <span class="stat-pill-label">Total Questions</span>
              <span class="stat-pill-val">${formatValue(totalQuestions)}</span>
            </div>
            <div class="stat-pill">
              <span class="stat-pill-label">Answerable</span>
              <span class="stat-pill-val">${formatValue(answerable)}</span>
            </div>
            <div class="stat-pill">
              <span class="stat-pill-label">Unanswerable</span>
              <span class="stat-pill-val">${formatValue(unanswerable)}</span>
            </div>
            <div class="stat-pill">
              <span class="stat-pill-label">Answerability Accuracy</span>
              <span class="stat-pill-val">${formatPercentage(accuracy)}</span>
            </div>
            <div class="stat-pill">
              <span class="stat-pill-label">Context Hit Rate @ 30</span>
              <span class="stat-pill-val">${formatPercentage(contextHit)}</span>
            </div>
            <div class="stat-pill">
              <span class="stat-pill-label">Context Precision @ 30</span>
              <span class="stat-pill-val">${formatPercentage(contextPrecision)}</span>
            </div>
            <div class="stat-pill">
              <span class="stat-pill-label">Fact Coverage Proxy</span>
              <span class="stat-pill-val">${formatPercentage(factCoverage)}</span>
            </div>
            <div class="stat-pill">
              <span class="stat-pill-label">Unsupported Rejection</span>
              <span class="stat-pill-val">${formatPercentage(rejectionRate)}</span>
            </div>
          </div>
        </article>
      `;
    })
    .join("");

  const limitations = (payload.limitations || []).map((item) => `<li>${escapeHtml(item)}</li>`).join("");

  els.evaluationContent.innerHTML = `
    <div class="eval-container">
      <div class="eval-header-bar">
        <div class="eval-intro">
          <h2>CampusMate AI Evaluation Dashboard</h2>
          <p>Verifiable academic benchmark scores, retrieval metrics, and LLM-as-a-judge quality assessments.</p>
        </div>
        <div class="eval-toolbar">
          <div class="eval-filter-group" role="group" aria-label="Campus Evaluation Filter">
            <button class="eval-filter-btn ${filter === "all" ? "active" : ""}" type="button" data-eval-filter="all">All</button>
            <button class="eval-filter-btn ${filter === "NSBM" ? "active" : ""}" type="button" data-eval-filter="NSBM">NSBM</button>
            <button class="eval-filter-btn ${filter === "SLIIT" ? "active" : ""}" type="button" data-eval-filter="SLIIT">SLIIT</button>
          </div>
          <button class="eval-refresh-btn" type="button" onclick="loadEvaluation()" title="Reload reports from disk">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
              <path d="M21.5 2v6h-6M21.34 15.57a10 10 0 1 1-.57-8.38l5.67-5.67"/>
            </svg>
            <span>Reload</span>
          </button>
        </div>
      </div>

      ${kpiSection}

      ${comparisonTable}

      ${cardsHtml}

      ${
        limitations
          ? `
        <article class="eval-card">
          <div class="eval-card-header">
            <div class="eval-card-title-group">
              <h3>Evaluation Methodology &amp; Limitations</h3>
              <span class="eval-source-label">Documented constraints from official academic evaluation rubric</span>
            </div>
          </div>
          <ul class="limitations-list">${limitations}</ul>
        </article>
      `
          : ""
      }
    </div>
  `;
}

/* ==========================================================================
   Navigation & UI State
   ========================================================================== */

function showView(view) {
  state.view = view;
  const isEval = view === "evaluation";

  /* Swap only the main pane. The conversation sidebar stays mounted. */
  els.chatView.hidden = isEval;
  els.evaluationView.hidden = !isEval;
  if (els.main) els.main.dataset.view = isEval ? "evaluation" : "chat";

  els.openChat.classList.toggle("active", !isEval);
  els.openEvaluation.classList.toggle("active", isEval);

  if (els.topbarChatBtn) els.topbarChatBtn.classList.toggle("active", !isEval);
  if (els.topbarEvalBtn) els.topbarEvalBtn.classList.toggle("active", isEval);

  els.viewTitle.textContent = isEval ? "Evaluation Dashboard" : "CampusMate AI";

  if (isEval) {
    if (!state.evaluationData) {
      loadEvaluation();
    } else {
      renderEvaluation(state.evaluationData);
    }
  }
  closeMobileSidebar();
}

function toggleDesktopSidebar() {
  state.sidebarCollapsed = !state.sidebarCollapsed;
  els.sidebar.classList.toggle("collapsed", state.sidebarCollapsed);
  els.sidebarExpandBtn.hidden = !state.sidebarCollapsed;
  saveSidebarState();
}

function openMobileSidebar() {
  els.sidebar.classList.add("open");
  els.backdrop.hidden = false;
}

function closeMobileSidebar() {
  els.sidebar.classList.remove("open");
  els.backdrop.hidden = true;
}

function openConversation(id) {
  const conversation = state.conversations.find((item) => item.id === id);
  if (!conversation) return;
  state.activeId = id;
  setCampus(conversation.campus, { restore: true });
  showView("chat");
}

function deleteConversation(id) {
  state.conversations = state.conversations.filter((item) => item.id !== id);
  if (state.activeId === id) state.activeId = null;
  saveStored();
  ensureActive();
  renderHistory();
  renderTranscript();
}

/* ==========================================================================
   Backend Status Polling
   ========================================================================== */

async function checkSystemStatus() {
  try {
    const res = await fetch("/api/status");
    if (res.ok) {
      const data = await res.json();
      if (data.status === "ready") {
        state.serverReady = true;
        if (els.statusIndicator) {
          els.statusIndicator.classList.remove("loading");
        }
        if (els.statusText) {
          els.statusText.textContent = "Academic Agent Ready (NSBM & SLIIT)";
        }
        return true;
      }
    }
  } catch {
    // server not reachable yet
  }
  if (els.statusIndicator) {
    els.statusIndicator.classList.add("loading");
  }
  if (els.statusText) {
    els.statusText.textContent = "Initializing Knowledge Base…";
  }
  return false;
}

function startStatusPolling() {
  checkSystemStatus().then((ready) => {
    if (!ready) {
      const timer = setInterval(async () => {
        const isReady = await checkSystemStatus();
        if (isReady) clearInterval(timer);
      }, 3500);
    }
  });
}

/* ==========================================================================
   Event Listeners
   ========================================================================== */

// New Chat button
els.newChat.addEventListener("click", () => {
  const current = activeConversation();
  if (!current || (current.messages && current.messages.length > 0)) {
    createConversation(state.campus);
  }
  showView("chat");
  renderHistory();
  renderTranscript();
  els.question.focus();
});

// All Campus buttons (Sidebar, Topbar, Welcome)
document.addEventListener("click", (event) => {
  const campusBtn = event.target.closest("[data-campus]");
  if (campusBtn) {
    const targetCampus = campusBtn.dataset.campus;
    setCampus(targetCampus);
    showView("chat");
    els.question.focus();
    return;
  }

  // Evaluation filter button
  const filterBtn = event.target.closest("[data-eval-filter]");
  if (filterBtn) {
    state.evalFilter = filterBtn.dataset.evalFilter;
    if (state.evaluationData) {
      renderEvaluation(state.evaluationData);
    }
    return;
  }
});

// Category tab buttons
els.categoryTabs.addEventListener("click", (event) => {
  const btn = event.target.closest(".category-tab");
  if (!btn) return;
  state.activeCategory = btn.dataset.category;
  renderSuggestedCategories();
});

// Suggested question clicks & Copy button clicks in transcript
els.transcript.addEventListener("click", (event) => {
  const card = event.target.closest("[data-suggestion]");
  if (card) {
    sendQuestion(card.dataset.suggestion);
    return;
  }

  const copyBtn = event.target.closest("[data-copy-idx]");
  if (copyBtn) {
    const idx = parseInt(copyBtn.dataset.copyIdx, 10);
    copyMessageContent(idx, copyBtn);
  }
});

// History list clicks (open or delete)
els.history.addEventListener("click", (event) => {
  const remove = event.target.closest("[data-delete]");
  if (remove) {
    event.stopPropagation();
    deleteConversation(remove.dataset.delete);
    return;
  }

  const open = event.target.closest("[data-open]");
  if (open) {
    openConversation(open.dataset.open);
  }
});

// Composer form submit
els.composer.addEventListener("submit", (event) => {
  event.preventDefault();
  sendQuestion(els.question.value);
});

// Textarea input auto-grow
els.question.addEventListener("input", resizeComposer);

// Textarea keydown: Enter to send, Shift + Enter for new line
els.question.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    sendQuestion(els.question.value);
  }
});

// Navigation view buttons
els.openChat.addEventListener("click", () => showView("chat"));
els.openEvaluation.addEventListener("click", () => showView("evaluation"));
if (els.topbarChatBtn) els.topbarChatBtn.addEventListener("click", () => showView("chat"));
if (els.topbarEvalBtn) els.topbarEvalBtn.addEventListener("click", () => showView("evaluation"));

// Sidebar collapse/expand & Mobile drawer
if (els.sidebarCollapseBtn) {
  els.sidebarCollapseBtn.addEventListener("click", toggleDesktopSidebar);
}
if (els.sidebarExpandBtn) {
  els.sidebarExpandBtn.addEventListener("click", toggleDesktopSidebar);
}
els.menu.addEventListener("click", openMobileSidebar);
els.backdrop.addEventListener("click", closeMobileSidebar);

/* ==========================================================================
   Initialization
   ========================================================================== */

loadStored();
window.loadEvaluation = loadEvaluation;

if (state.sidebarCollapsed) {
  els.sidebar.classList.add("collapsed");
  els.sidebarExpandBtn.hidden = false;
}

ensureActive();
setCampus(state.campus, { restore: true });
showView("chat");
startStatusPolling();
