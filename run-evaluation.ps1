Set-Location $PSScriptRoot

$env:POSTGRES_HOST = "127.0.0.1"
$env:NEO4J_URI = "neo4j://127.0.0.1:7687"
$env:MILVUS_URI = "http://127.0.0.1:19530"
$env:LOG_LEVEL = "INFO"
$env:PYTHONUNBUFFERED = "1"


##################################################################
# HotpotQA
##################################################################

.\.venv\Scripts\python.exe -u -m lightrag_wrapper.evaluation `
    --dataset hotpotqa `
    --data data/eval/datasets/hotpot_dev_distractor_v1.json `
    --limit 100 `
    --seed 0 `
    --repeat 1 `
    --workspace hotpotqa `
    --working-dir data/eval/hotpotqa/rag_storage `
    --disable-rerank `
    --ingest-only

##################################################################
# Musique
##################################################################
# .\.venv\Scripts\python.exe -u -m lightrag_wrapper.evaluation `
#     --dataset musique `
#     --data data/eval/datasets/musique_ans_v1.0_dev.jsonl `
#     --limit 100 `
#     --seed 0 `
#     --repeat 1 `
#     --workspace musique `
#     --working-dir data/eval/musique/rag_storage `
#     --disable-rerank `
#     --ingest-only
