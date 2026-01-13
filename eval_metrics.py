import json
from collections import defaultdict

# Load eval results
with open("eval_results.json", "r", encoding="utf-8") as f:
    results = json.load(f)

# Accumulators
metrics_sum = defaultdict(float)
metrics_count = defaultdict(int)

for r in results:
    # Retrieval metrics
    for k, v in r["retrieval"].items():
        metrics_sum[k] += v
        metrics_count[k] += 1

    # LLM-based metrics
    metrics_sum["context_relevance"] += r["context_relevance"]
    metrics_count["context_relevance"] += 1

    metrics_sum["faithfulness"] += r["faithfulness"]
    metrics_count["faithfulness"] += 1

    metrics_sum["correctness"] += r["correctness"]
    metrics_count["correctness"] += 1

# Compute mean
metrics_mean = {
    k: round(metrics_sum[k] / metrics_count[k], 4)
    for k in metrics_sum
}

# Pretty print
print("📊 MEAN EVALUATION METRICS")
for k, v in metrics_mean.items():
    print(f"{k:20s}: {v}")
