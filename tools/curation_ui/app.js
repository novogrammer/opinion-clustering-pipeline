const REPRESENTATIVE_COLUMNS = ["topic_id", "topic_size", "response_id", "question_id", "answer_text", "topic_probability", "representative_rank"];
const TOPIC_CATEGORY_MAPPING_COLUMNS = ["topic_id", "category_id"];
const CATEGORY_MASTER_COLUMNS = ["category_id", "category_name", "category_definition"];
const OUTLIER_TOPIC_ID = "-1";

const state = {
  representatives: [],
  topics: [],
  topicMap: new Map(),
  mappings: new Map(),
  categories: [],
  search: "",
  sort: "size_desc",
  unassignedOnly: false,
};

const elements = {
  representativesFile: document.getElementById("representativesFile"),
  mappingFile: document.getElementById("mappingFile"),
  masterFile: document.getElementById("masterFile"),
  searchInput: document.getElementById("searchInput"),
  sortSelect: document.getElementById("sortSelect"),
  unassignedOnly: document.getElementById("unassignedOnly"),
  topics: document.getElementById("topics"),
  categoryTable: document.getElementById("categoryTable"),
  status: document.getElementById("status"),
  normalTopicCount: document.getElementById("normalTopicCount"),
  unassignedCount: document.getElementById("unassignedCount"),
  categoryCount: document.getElementById("categoryCount"),
  outlierCount: document.getElementById("outlierCount"),
  downloadMappingButton: document.getElementById("downloadMappingButton"),
  downloadMasterButton: document.getElementById("downloadMasterButton"),
  validateButton: document.getElementById("validateButton"),
  addCategoryButton: document.getElementById("addCategoryButton"),
};

function setStatus(message, isError = false) {
  elements.status.textContent = message;
  elements.status.classList.toggle("error", isError);
}

function normalizeText(value) {
  return String(value ?? "").trim();
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function parseCsvFile(file) {
  return new Promise((resolve, reject) => {
    Papa.parse(file, {
      header: true,
      skipEmptyLines: true,
      complete(results) {
        if (results.errors.length > 0) {
          reject(new Error(results.errors.map((error) => error.message).join("\n")));
          return;
        }
        resolve(results.data);
      },
      error(error) {
        reject(error);
      },
    });
  });
}

function requireColumns(rows, columns, label) {
  if (!rows.length) return;
  const first = rows[0];
  const missing = columns.filter((column) => !(column in first));
  if (missing.length) {
    throw new Error(`${label} に必須列がありません: ${missing.join(", ")}`);
  }
}

function rebuildTopics() {
  const groups = new Map();
  for (const row of state.representatives) {
    const topicId = normalizeText(row.topic_id);
    if (!groups.has(topicId)) {
      groups.set(topicId, {
        topic_id: topicId,
        topic_size: Number.parseInt(normalizeText(row.topic_size), 10) || 0,
        representatives: [],
      });
    }
    groups.get(topicId).representatives.push({
      response_id: normalizeText(row.response_id),
      question_id: normalizeText(row.question_id),
      answer_text: normalizeText(row.answer_text),
      topic_probability: normalizeText(row.topic_probability),
      representative_rank: Number.parseInt(normalizeText(row.representative_rank), 10) || 0,
    });
  }
  const topics = Array.from(groups.values()).map((topic) => {
    topic.representatives.sort((a, b) => a.representative_rank - b.representative_rank);
    return topic;
  });
  state.topics = topics;
  state.topicMap = new Map(topics.map((topic) => [topic.topic_id, topic]));
}

function renderTopics() {
  const search = state.search.toLowerCase();
  const topics = state.topics
    .filter((topic) => {
      const assigned = normalizeText(state.mappings.get(topic.topic_id));
      if (state.unassignedOnly && topic.topic_id !== OUTLIER_TOPIC_ID && !assigned) return true;
      if (state.unassignedOnly) return false;
      if (!search) return true;
      if (topic.topic_id.toLowerCase().includes(search)) return true;
      return topic.representatives.some((rep) => rep.answer_text.toLowerCase().includes(search));
    })
    .sort((a, b) => {
      if (state.sort === "topic_asc") return Number(a.topic_id) - Number(b.topic_id);
      if (b.topic_size !== a.topic_size) return b.topic_size - a.topic_size;
      return Number(a.topic_id) - Number(b.topic_id);
    });

  if (!topics.length) {
    elements.topics.innerHTML = '<div class="empty">表示対象の topic がありません。</div>';
    return;
  }

  const categoryOptions = [
    '<option value="">未割当</option>',
    ...state.categories.map((category) => `<option value="${escapeHtml(category.category_id)}">${escapeHtml(category.category_id)} | ${escapeHtml(category.category_name || "(名称未設定)")}</option>`),
  ].join("");

  elements.topics.innerHTML = topics.map((topic) => {
    const isOutlier = topic.topic_id === OUTLIER_TOPIC_ID;
    const assignedCategoryId = normalizeText(state.mappings.get(topic.topic_id));
    return `
      <article class="topic-card ${isOutlier ? "outlier" : ""}">
        <div class="topic-head">
          <div>
            <h3>topic ${escapeHtml(topic.topic_id)}</h3>
            <div class="topic-meta">
              <span class="pill">${escapeHtml(String(topic.topic_size))} responses</span>
              ${isOutlier ? '<span class="pill warn">OTHER 扱い / mapping不要</span>' : ""}
              ${!isOutlier && assignedCategoryId ? `<span class="pill">${escapeHtml(assignedCategoryId)}</span>` : ""}
            </div>
          </div>
        </div>
        <div class="rep-list">
          ${topic.representatives.map((rep) => `
            <div class="rep-item">
              <small>#${rep.representative_rank} / response_id=${escapeHtml(rep.response_id)}${rep.topic_probability ? ` / p=${escapeHtml(rep.topic_probability)}` : ""}</small>
              <div>${escapeHtml(rep.answer_text)}</div>
            </div>
          `).join("")}
        </div>
        <div class="topic-actions">
          <div class="field">
            <label>割当 category_id</label>
            <select data-topic-id="${escapeHtml(topic.topic_id)}" class="topic-category-select" ${isOutlier ? "disabled" : ""}>
              ${categoryOptions}
            </select>
          </div>
          <button type="button" class="secondary suggest-category-button" data-topic-id="${escapeHtml(topic.topic_id)}" ${isOutlier ? "disabled" : ""}>新規カテゴリ雛形</button>
        </div>
      </article>
    `;
  }).join("");

  for (const select of elements.topics.querySelectorAll(".topic-category-select")) {
    const topicId = select.dataset.topicId;
    select.value = normalizeText(state.mappings.get(topicId));
    select.addEventListener("change", (event) => {
      const value = normalizeText(event.target.value);
      if (value) state.mappings.set(topicId, value);
      else state.mappings.delete(topicId);
      renderAll();
    });
  }

  for (const button of elements.topics.querySelectorAll(".suggest-category-button")) {
    button.addEventListener("click", () => createCategoryDraftFromTopic(button.dataset.topicId));
  }
}

function getCategoryUsage() {
  const usage = new Map();
  for (const category of state.categories) usage.set(category.category_id, 0);
  for (const [topicId, categoryId] of state.mappings.entries()) {
    if (topicId === OUTLIER_TOPIC_ID || !categoryId) continue;
    usage.set(categoryId, (usage.get(categoryId) || 0) + 1);
  }
  return usage;
}

function renderCategories() {
  const usage = getCategoryUsage();
  if (!state.categories.length) {
    elements.categoryTable.innerHTML = '<div class="empty">まだカテゴリがありません。</div>';
    return;
  }
  elements.categoryTable.innerHTML = state.categories.map((category, index) => `
    <div class="category-row" data-category-index="${index}">
      <div class="category-grid">
        <div class="field">
          <label>category_id</label>
          <input type="text" class="category-id-input" value="${escapeHtml(category.category_id)}">
        </div>
        <div class="field">
          <label>category_name</label>
          <input type="text" class="category-name-input" value="${escapeHtml(category.category_name)}">
        </div>
        <div class="field">
          <label>category_definition</label>
          <textarea class="category-definition-input">${escapeHtml(category.category_definition)}</textarea>
        </div>
        <div class="category-usage">
          <div class="hint">割当 topic</div>
          <div>${usage.get(category.category_id) || 0}</div>
        </div>
        <div class="button-row">
          <button type="button" class="warn delete-category-button">削除</button>
        </div>
      </div>
    </div>
  `).join("");

  for (const row of elements.categoryTable.querySelectorAll(".category-row")) {
    const index = Number(row.dataset.categoryIndex);
    row.querySelector(".category-id-input").addEventListener("input", (event) => {
      const previousId = state.categories[index].category_id;
      const nextId = normalizeText(event.target.value);
      state.categories[index].category_id = nextId;
      if (previousId !== nextId) {
        for (const [topicId, categoryId] of state.mappings.entries()) {
          if (categoryId === previousId) {
            if (nextId) state.mappings.set(topicId, nextId);
            else state.mappings.delete(topicId);
          }
        }
      }
      renderAll();
    });
    row.querySelector(".category-name-input").addEventListener("input", (event) => {
      state.categories[index].category_name = normalizeText(event.target.value);
    });
    row.querySelector(".category-definition-input").addEventListener("input", (event) => {
      state.categories[index].category_definition = normalizeText(event.target.value);
    });
    row.querySelector(".delete-category-button").addEventListener("click", () => deleteCategory(index));
  }
}

function updateSummary() {
  const normalTopics = state.topics.filter((topic) => topic.topic_id !== OUTLIER_TOPIC_ID);
  const outlierTopics = state.topics.filter((topic) => topic.topic_id === OUTLIER_TOPIC_ID);
  const unassigned = normalTopics.filter((topic) => !normalizeText(state.mappings.get(topic.topic_id)));
  elements.normalTopicCount.textContent = String(normalTopics.length);
  elements.unassignedCount.textContent = String(unassigned.length);
  elements.categoryCount.textContent = String(state.categories.length);
  elements.outlierCount.textContent = String(outlierTopics.length);
}

function renderAll() {
  updateSummary();
  renderTopics();
  renderCategories();
}

function suggestCategoryId() {
  let counter = 1;
  const existing = new Set(state.categories.map((category) => category.category_id));
  while (true) {
    const candidate = `CAT${String(counter).padStart(3, "0")}`;
    if (!existing.has(candidate)) return candidate;
    counter += 1;
  }
}

function createCategoryDraftFromTopic(topicId) {
  const topic = state.topicMap.get(topicId);
  if (!topic) return;
  const categoryId = suggestCategoryId();
  const firstAnswer = topic.representatives[0]?.answer_text || "";
  state.categories.push({
    category_id: categoryId,
    category_name: "",
    category_definition: firstAnswer.slice(0, 80),
  });
  state.mappings.set(topicId, categoryId);
  renderAll();
  setStatus(`topic ${topicId} から ${categoryId} の雛形を追加しました。`);
}

function deleteCategory(index) {
  const category = state.categories[index];
  if (!category) return;
  const assignedCount = Array.from(state.mappings.values()).filter((value) => value === category.category_id).length;
  if (assignedCount > 0) {
    setStatus(`category_id=${category.category_id} は ${assignedCount} 件の topic に割り当て済みのため削除できません。`, true);
    return;
  }
  state.categories.splice(index, 1);
  renderAll();
}

function buildMappingRows() {
  return state.topics
    .filter((topic) => topic.topic_id !== OUTLIER_TOPIC_ID)
    .map((topic) => ({
      topic_id: topic.topic_id,
      category_id: normalizeText(state.mappings.get(topic.topic_id)),
    }))
    .filter((row) => row.category_id)
    .sort((a, b) => Number(a.topic_id) - Number(b.topic_id));
}

function buildCategoryRows() {
  return state.categories
    .map((category) => ({
      category_id: normalizeText(category.category_id),
      category_name: normalizeText(category.category_name),
      category_definition: normalizeText(category.category_definition),
    }))
    .filter((row) => row.category_id || row.category_name || row.category_definition);
}

function validateState() {
  const errors = [];
  if (!state.topics.length) errors.push("cluster_representatives.csv が未読込です。");
  const normalTopics = state.topics.filter((topic) => topic.topic_id !== OUTLIER_TOPIC_ID);
  const mappingRows = buildMappingRows();
  const mappedTopicIds = new Set(mappingRows.map((row) => row.topic_id));
  const categoryRows = buildCategoryRows();
  const categoryIdSet = new Set();

  for (const topic of normalTopics) {
    const categoryId = normalizeText(state.mappings.get(topic.topic_id));
    if (!categoryId) errors.push(`topic_id=${topic.topic_id} が未割当です。`);
  }
  if (mappedTopicIds.size !== mappingRows.length) {
    errors.push("topic_category_mapping.csv に topic_id 重複があります。");
  }
  for (const row of categoryRows) {
    if (!row.category_id) errors.push("category_master.csv に空の category_id があります。");
    else if (categoryIdSet.has(row.category_id)) errors.push(`category_master.csv に重複 category_id があります: ${row.category_id}`);
    categoryIdSet.add(row.category_id);
  }
  for (const row of mappingRows) {
    if (!categoryIdSet.has(row.category_id)) {
      errors.push(`mapping に存在する category_id=${row.category_id} が category_master.csv にありません。`);
    }
  }
  return { errors, mappingRows, categoryRows };
}

function downloadCsv(filename, columns, rows) {
  const csv = Papa.unparse({ fields: columns, data: rows.map((row) => columns.map((column) => row[column] ?? "")) });
  const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  document.body.appendChild(anchor);
  anchor.click();
  document.body.removeChild(anchor);
  URL.revokeObjectURL(url);
}

async function loadRepresentatives(file) {
  const rows = await parseCsvFile(file);
  requireColumns(rows, REPRESENTATIVE_COLUMNS, "cluster_representatives.csv");
  state.representatives = rows.map((row) => ({
    topic_id: normalizeText(row.topic_id),
    topic_size: normalizeText(row.topic_size),
    response_id: normalizeText(row.response_id),
    question_id: normalizeText(row.question_id),
    answer_text: normalizeText(row.answer_text),
    topic_probability: normalizeText(row.topic_probability),
    representative_rank: normalizeText(row.representative_rank),
  }));
  rebuildTopics();
  state.mappings = new Map();
  state.categories = [];
  const normalTopics = state.topics.filter((topic) => topic.topic_id !== OUTLIER_TOPIC_ID);
  setStatus(`代表回答を読み込みました。通常 topic ${normalTopics.length} 件、outlier ${state.topics.length - normalTopics.length} 件です。`);
  renderAll();
}

async function loadMappings(file) {
  const rows = await parseCsvFile(file);
  requireColumns(rows, TOPIC_CATEGORY_MAPPING_COLUMNS, "topic_category_mapping.csv");
  state.mappings = new Map(
    rows
      .map((row) => [normalizeText(row.topic_id), normalizeText(row.category_id)])
      .filter(([topicId]) => topicId)
  );
  setStatus("topic_category_mapping.csv を読み込みました。");
  renderAll();
}

async function loadCategories(file) {
  const rows = await parseCsvFile(file);
  requireColumns(rows, CATEGORY_MASTER_COLUMNS, "category_master.csv");
  state.categories = rows
    .map((row) => ({
      category_id: normalizeText(row.category_id),
      category_name: normalizeText(row.category_name),
      category_definition: normalizeText(row.category_definition),
    }))
    .filter((row) => row.category_id || row.category_name || row.category_definition);
  setStatus("category_master.csv を読み込みました。");
  renderAll();
}

elements.representativesFile.addEventListener("change", async (event) => {
  const file = event.target.files[0];
  if (!file) return;
  try {
    await loadRepresentatives(file);
  } catch (error) {
    setStatus(error.message, true);
  }
});

elements.mappingFile.addEventListener("change", async (event) => {
  const file = event.target.files[0];
  if (!file) return;
  try {
    await loadMappings(file);
  } catch (error) {
    setStatus(error.message, true);
  }
});

elements.masterFile.addEventListener("change", async (event) => {
  const file = event.target.files[0];
  if (!file) return;
  try {
    await loadCategories(file);
  } catch (error) {
    setStatus(error.message, true);
  }
});

elements.searchInput.addEventListener("input", (event) => {
  state.search = normalizeText(event.target.value);
  renderTopics();
});

elements.sortSelect.addEventListener("change", (event) => {
  state.sort = event.target.value;
  renderTopics();
});

elements.unassignedOnly.addEventListener("change", (event) => {
  state.unassignedOnly = event.target.checked;
  renderTopics();
});

elements.addCategoryButton.addEventListener("click", () => {
  state.categories.push({ category_id: suggestCategoryId(), category_name: "", category_definition: "" });
  renderAll();
});

elements.validateButton.addEventListener("click", () => {
  const { errors, mappingRows, categoryRows } = validateState();
  if (errors.length) {
    setStatus(errors.join("\n"), true);
    return;
  }
  setStatus(`検査OKです。mapping ${mappingRows.length} 件、category ${categoryRows.length} 件。`);
});

elements.downloadMappingButton.addEventListener("click", () => {
  const { errors, mappingRows } = validateState();
  if (errors.length) {
    setStatus(errors.join("\n"), true);
    return;
  }
  downloadCsv("topic_category_mapping.csv", TOPIC_CATEGORY_MAPPING_COLUMNS, mappingRows);
  setStatus("topic_category_mapping.csv をダウンロードしました。");
});

elements.downloadMasterButton.addEventListener("click", () => {
  const { errors, categoryRows } = validateState();
  if (errors.length) {
    setStatus(errors.join("\n"), true);
    return;
  }
  downloadCsv("category_master.csv", CATEGORY_MASTER_COLUMNS, categoryRows);
  setStatus("category_master.csv をダウンロードしました。");
});
