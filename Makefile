PYTHON ?= python
PROJECT_NAME ?= your_project_name
QUESTION_ID ?= Q1
MODEL ?= text-embedding-3-small
BATCH_SIZE ?= 100
MAX_RETRIES ?= 3
RETRY_BASE_SECONDS ?= 1.0
FORCE ?= false
UMAP_N_NEIGHBORS ?= 15
UMAP_N_COMPONENTS ?= 5
HDBSCAN_MIN_CLUSTER_SIZE ?= 10
HDBSCAN_MIN_SAMPLES ?= 5
RANDOM_STATE ?= 42
LOG_NAME ?= pipeline.log
RAW_INPUT ?= projects/$(PROJECT_NAME)/00_raw/source.csv
RESPONSE_ID_COL ?= 回答ID
QUESTION_ID_COL ?= 設問ID
QUESTION_TEXT_COL ?= 質問文
ANSWER_TEXT_COL ?= 自由回答

.PHONY: help init-project init-question normalize validate-processed validate-mapping screening duplicate-check validate-screening validate-duplicates embeddings embeddings-prepare validate-embedding-requests validate-embedding-metadata validate-embedding-failures validate-embeddings-array clustering validate-clusters validate-cluster-summary validate-clustering-metadata scaffold-category-master category-conflicts validate-category-master validate-override-rules validate-override-candidates validate-override-rule-hits validate-override-rule-summary validate-category-conflicts classification validate-final-labels review review-summary review-priorities review-samples review-corrections override-candidates promote-override-candidates override-rule-hits override-rule-summary validate-review-log validate-review-summary validate-review-priorities validate-review-samples validate-review-corrections validate-question validate-project validate-log

help:
	@echo "Available targets:"
	@echo "  init-project"
	@echo "  init-question"
	@echo "  normalize"
	@echo "  validate-processed"
	@echo "  validate-mapping"
	@echo "  screening"
	@echo "  duplicate-check"
	@echo "  validate-screening"
	@echo "  validate-duplicates"
	@echo "  embeddings"
	@echo "  embeddings-prepare"
	@echo "  validate-embedding-requests"
	@echo "  validate-embedding-metadata"
	@echo "  validate-embedding-failures"
	@echo "  validate-embeddings-array"
	@echo "  clustering"
	@echo "  validate-clusters"
	@echo "  validate-cluster-summary"
	@echo "  validate-clustering-metadata"
	@echo "  scaffold-category-master"
	@echo "  category-conflicts"
	@echo "  validate-category-master"
	@echo "  validate-override-rules"
	@echo "  validate-override-candidates"
	@echo "  validate-override-rule-hits"
	@echo "  validate-override-rule-summary"
	@echo "  validate-category-conflicts"
	@echo "  classification"
	@echo "  validate-final-labels"
	@echo "  review"
	@echo "  review-summary"
	@echo "  review-priorities"
	@echo "  review-samples"
	@echo "  review-corrections"
	@echo "  override-candidates"
	@echo "  promote-override-candidates"
	@echo "  override-rule-hits"
	@echo "  override-rule-summary"
	@echo "  validate-review-log"
	@echo "  validate-review-summary"
	@echo "  validate-review-priorities"
	@echo "  validate-review-samples"
	@echo "  validate-review-corrections"
	@echo "  validate-question"
	@echo "  validate-project"
	@echo "  validate-log"

init-project:
	$(PYTHON) scripts/pipeline.py init-project --project-name $(PROJECT_NAME)

init-question:
	$(PYTHON) scripts/pipeline.py init-question --project-name $(PROJECT_NAME) --question-id $(QUESTION_ID)

normalize:
	$(PYTHON) scripts/pipeline.py normalize --project-name $(PROJECT_NAME) --input $(RAW_INPUT) --response-id-col $(RESPONSE_ID_COL) --question-id-col $(QUESTION_ID_COL) --question-text-col $(QUESTION_TEXT_COL) --answer-text-col $(ANSWER_TEXT_COL)

validate-processed:
	$(PYTHON) scripts/pipeline.py validate-processed --project-name $(PROJECT_NAME)

validate-mapping:
	$(PYTHON) scripts/pipeline.py validate-mapping --project-name $(PROJECT_NAME)

screening:
	$(PYTHON) scripts/pipeline.py screening --project-name $(PROJECT_NAME)

duplicate-check:
	$(PYTHON) scripts/pipeline.py duplicate-check --project-name $(PROJECT_NAME)

validate-screening:
	$(PYTHON) scripts/pipeline.py validate-screening --project-name $(PROJECT_NAME)

validate-duplicates:
	$(PYTHON) scripts/pipeline.py validate-duplicates --project-name $(PROJECT_NAME)

embeddings:
	$(PYTHON) scripts/pipeline.py embeddings --project-name $(PROJECT_NAME) --question-id $(QUESTION_ID) --model $(MODEL) --batch-size $(BATCH_SIZE) --max-retries $(MAX_RETRIES) --retry-base-seconds $(RETRY_BASE_SECONDS) $(if $(filter true,$(FORCE)),--force)

embeddings-prepare:
	$(PYTHON) scripts/pipeline.py embeddings --project-name $(PROJECT_NAME) --question-id $(QUESTION_ID) --model $(MODEL) --batch-size $(BATCH_SIZE) --max-retries $(MAX_RETRIES) --retry-base-seconds $(RETRY_BASE_SECONDS) $(if $(filter true,$(FORCE)),--force) --prepare-only

validate-embedding-requests:
	$(PYTHON) scripts/pipeline.py validate-embedding-requests --project-name $(PROJECT_NAME) --question-id $(QUESTION_ID)

validate-embedding-metadata:
	$(PYTHON) scripts/pipeline.py validate-embedding-metadata --project-name $(PROJECT_NAME) --question-id $(QUESTION_ID)

validate-embedding-failures:
	$(PYTHON) scripts/pipeline.py validate-embedding-failures --project-name $(PROJECT_NAME) --question-id $(QUESTION_ID)

validate-embeddings-array:
	$(PYTHON) scripts/pipeline.py validate-embeddings-array --project-name $(PROJECT_NAME) --question-id $(QUESTION_ID)

clustering:
	$(PYTHON) scripts/pipeline.py clustering --project-name $(PROJECT_NAME) --question-id $(QUESTION_ID) --umap-n-neighbors $(UMAP_N_NEIGHBORS) --umap-n-components $(UMAP_N_COMPONENTS) --hdbscan-min-cluster-size $(HDBSCAN_MIN_CLUSTER_SIZE) --hdbscan-min-samples $(HDBSCAN_MIN_SAMPLES) --random-state $(RANDOM_STATE) $(if $(filter true,$(FORCE)),--force)

validate-clusters:
	$(PYTHON) scripts/pipeline.py validate-clusters --project-name $(PROJECT_NAME) --question-id $(QUESTION_ID)

validate-cluster-summary:
	$(PYTHON) scripts/pipeline.py validate-cluster-summary --project-name $(PROJECT_NAME) --question-id $(QUESTION_ID)

validate-clustering-metadata:
	$(PYTHON) scripts/pipeline.py validate-clustering-metadata --project-name $(PROJECT_NAME) --question-id $(QUESTION_ID)

scaffold-category-master:
	$(PYTHON) scripts/pipeline.py scaffold-category-master --project-name $(PROJECT_NAME) --question-id $(QUESTION_ID)

category-conflicts:
	$(PYTHON) scripts/pipeline.py category-conflicts --project-name $(PROJECT_NAME) --question-id $(QUESTION_ID)

validate-category-master:
	$(PYTHON) scripts/pipeline.py validate-category-master --project-name $(PROJECT_NAME) --question-id $(QUESTION_ID)

validate-override-rules:
	$(PYTHON) scripts/pipeline.py validate-override-rules --project-name $(PROJECT_NAME) --question-id $(QUESTION_ID)

validate-override-candidates:
	$(PYTHON) scripts/pipeline.py validate-override-candidates --project-name $(PROJECT_NAME) --question-id $(QUESTION_ID)

validate-override-rule-hits:
	$(PYTHON) scripts/pipeline.py validate-override-rule-hits --project-name $(PROJECT_NAME) --question-id $(QUESTION_ID)

validate-override-rule-summary:
	$(PYTHON) scripts/pipeline.py validate-override-rule-summary --project-name $(PROJECT_NAME) --question-id $(QUESTION_ID)

validate-category-conflicts:
	$(PYTHON) scripts/pipeline.py validate-category-conflicts --project-name $(PROJECT_NAME) --question-id $(QUESTION_ID)

classification:
	$(PYTHON) scripts/pipeline.py classification --project-name $(PROJECT_NAME) --question-id $(QUESTION_ID)

validate-final-labels:
	$(PYTHON) scripts/pipeline.py validate-final-labels --project-name $(PROJECT_NAME) --question-id $(QUESTION_ID)

review:
	$(PYTHON) scripts/pipeline.py review --project-name $(PROJECT_NAME) --question-id $(QUESTION_ID)

review-summary:
	$(PYTHON) scripts/pipeline.py review-summary --project-name $(PROJECT_NAME) --question-id $(QUESTION_ID)

review-priorities:
	$(PYTHON) scripts/pipeline.py review-priorities --project-name $(PROJECT_NAME) --question-id $(QUESTION_ID)

review-samples:
	$(PYTHON) scripts/pipeline.py review-samples --project-name $(PROJECT_NAME) --question-id $(QUESTION_ID)

review-corrections:
	$(PYTHON) scripts/pipeline.py review-corrections --project-name $(PROJECT_NAME) --question-id $(QUESTION_ID)

override-candidates:
	$(PYTHON) scripts/pipeline.py override-candidates --project-name $(PROJECT_NAME) --question-id $(QUESTION_ID)

promote-override-candidates:
	$(PYTHON) scripts/pipeline.py promote-override-candidates --project-name $(PROJECT_NAME) --question-id $(QUESTION_ID)

override-rule-hits:
	$(PYTHON) scripts/pipeline.py override-rule-hits --project-name $(PROJECT_NAME) --question-id $(QUESTION_ID)

override-rule-summary:
	$(PYTHON) scripts/pipeline.py override-rule-summary --project-name $(PROJECT_NAME) --question-id $(QUESTION_ID)

validate-review-log:
	$(PYTHON) scripts/pipeline.py validate-review-log --project-name $(PROJECT_NAME) --question-id $(QUESTION_ID)

validate-review-summary:
	$(PYTHON) scripts/pipeline.py validate-review-summary --project-name $(PROJECT_NAME) --question-id $(QUESTION_ID)

validate-review-priorities:
	$(PYTHON) scripts/pipeline.py validate-review-priorities --project-name $(PROJECT_NAME) --question-id $(QUESTION_ID)

validate-review-samples:
	$(PYTHON) scripts/pipeline.py validate-review-samples --project-name $(PROJECT_NAME) --question-id $(QUESTION_ID)

validate-review-corrections:
	$(PYTHON) scripts/pipeline.py validate-review-corrections --project-name $(PROJECT_NAME) --question-id $(QUESTION_ID)

validate-question:
	$(PYTHON) scripts/pipeline.py validate-question --project-name $(PROJECT_NAME) --question-id $(QUESTION_ID)

validate-project:
	$(PYTHON) scripts/pipeline.py validate-project --project-name $(PROJECT_NAME)

validate-log:
	$(PYTHON) scripts/pipeline.py validate-log --project-name $(PROJECT_NAME) --log-name $(LOG_NAME)
