import json
import os
import time
import logging
from typing import List

from langchain_core.prompts import ChatPromptTemplate

from query.core import get_llm, get_embedding_model, get_cross_encoder
from query.medical.medical_rag import MedicalRerankRAG
from query.core import LLMEvalResult
from query.prompt_templates import EVAL_PROMPT


# =========================
# CONFIG
# =========================

CHECKPOINT_FILE = "eval_results_rerank_checkpoint.json"
FINAL_RESULT_FILE = "eval_results_rerank.json"


# =========================
# LOGGING
# =========================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("rag-eval-rerank")


# =========================
# Eval data loader
# =========================

class EvalData:
    def __init__(self, path: str):
        with open(path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

    def __iter__(self):
        for item in self.data:
            yield item


# =========================
# Metrics: Retrieval
# =========================

def recall_at_k(retrieved_ids: List[str], relevant_ids: List[str]) -> float:
    return float(any(rid in retrieved_ids for rid in relevant_ids))


def precision_at_k(retrieved_ids: List[str], relevant_ids: List[str]) -> float:
    if not retrieved_ids:
        return 0.0
    return len(set(retrieved_ids) & set(relevant_ids)) / len(retrieved_ids)


def mrr(retrieved_ids: List[str], relevant_ids: List[str]) -> float:
    for i, rid in enumerate(retrieved_ids):
        if rid in relevant_ids:
            return 1.0 / (i + 1)
    return 0.0


# =========================
# Main Evaluation Pipeline
# =========================

def evaluate_rag(
    eval_path: str,
    k1: int = 20,
    k2: int = 5,
    sleeping_time: int = 10,
):
    logger.info("🚀 START RERANK EVALUATION")

    # ===== LOAD CHECKPOINT =====
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
            results = json.load(f)
        done_queries = set(r["query"] for r in results)
        logger.info(f"🔁 Resume from checkpoint: {len(results)} samples done")
    else:
        results = []
        done_queries = set()
        logger.info("🆕 No checkpoint found, start fresh")

    # ===== INIT MODELS =====
    logger.info("[INIT] Load models")

    # 🔹 Answer LLM (TEXT ONLY)
    answer_llm = get_llm("llama3")

    # 🔹 Retrieval + Rerank
    embedding_model = get_embedding_model()
    reranking_model = get_cross_encoder()
    medical_rag = MedicalRerankRAG(
        embedder=embedding_model,
        reranker=reranking_model,
    )

    # 🔹 Eval LLM (structured output OK)
    eval_llm = (
        get_llm("openai-oss", temperature=0.0)
        .with_structured_output(LLMEvalResult)
    )

    eval_data = EvalData(eval_path)

    # ===== PROMPTS =====

    answer_prompt = ChatPromptTemplate.from_messages([
        ("system", "Bạn là trợ lý y khoa. Chỉ sử dụng thông tin trong ngữ cảnh."),
        (
            "human",
            "Câu hỏi: {query}\n\n"
            "Ngữ cảnh:\n{context}\n\n"
            "Trả lời ngắn gọn, chính xác."
        )
    ])

    eval_prompt = ChatPromptTemplate.from_messages([
        ("system", "Bạn là chuyên gia đánh giá hệ thống hỏi–đáp y khoa."),
        ("human", EVAL_PROMPT)
    ])

    answer_chain = answer_prompt | answer_llm
    eval_chain = eval_prompt | eval_llm

    # ===== MAIN LOOP =====

    for idx, sample in enumerate(eval_data):
        query = sample["question"]

        if query in done_queries:
            logger.info(f"⏭️ SKIP [{idx}] already evaluated")
            continue

        relevant_chunks = sample.get("ground_truth_chunk_ids", [])

        logger.info(f"\n🧪 SAMPLE {idx}")
        logger.info(f"[QUERY] {query}")

        try:
            # ---- 1. RETRIEVAL + RERANK ----
            logger.info("[STAGE 1] Retrieval + Rerank")

            hits = medical_rag.query(query, limit1=k1, limit2=k2)
            time.sleep(1)  # giảm áp lực vector DB

            retrieved_ids = [hit.id for hit in hits]
            context = "\n\n".join(hit.payload["text"] for hit in hits)

            retrieval_metrics = {
                "recall@k": recall_at_k(retrieved_ids, relevant_chunks),
                "precision@k": precision_at_k(retrieved_ids, relevant_chunks),
                "mrr": mrr(retrieved_ids, relevant_chunks),
            }

            logger.info(f"[RETRIEVAL METRIC] {retrieval_metrics}")

            # ---- 2. ANSWER GENERATION ----
            logger.info("[STAGE 2] Answer Generation")

            answer_msg = answer_chain.invoke({
                "query": query,
                "context": context
            })
            answer_text = answer_msg.content.strip()

            logger.info(f"[ANSWER] {answer_text}")

            # ---- 3. LLM JUDGE ----
            logger.info("[STAGE 3] LLM-based Evaluation")

            results_eval = eval_chain.invoke({
                "query": query,
                "context": context,
                "answer": answer_text
            })

            logger.info(
                "[EVAL SCORE] "
                f"context={results_eval.context_relevance.context_relevance}, "
                f"faithfulness={results_eval.faithfulness.faithfulness}, "
                f"correctness={results_eval.correctness.correctness}"
            )

            # ---- SAVE RESULT ----
            results.append({
                "query": query,
                "retrieval": retrieval_metrics,
                "context_relevance": results_eval.context_relevance.context_relevance,
                "faithfulness": results_eval.faithfulness.faithfulness,
                "correctness": results_eval.correctness.correctness,
                "answer": answer_text,
                "retrieved_ids": retrieved_ids,
                "reason_context_relevance": results_eval.context_relevance.reason,
                "reason_faithfulness": results_eval.faithfulness.reason,
                "reason_correctness": results_eval.correctness.reason,
            })

            # 💾 SAVE CHECKPOINT
            with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

            logger.info("💾 Checkpoint saved")

            # ⏱️ SLEEP
            logger.info(f"⏳ Sleep {sleeping_time}s")
            time.sleep(sleeping_time)

        except Exception as e:
            logger.exception(f"❌ ERROR at sample {idx}: {e}")
            logger.info("➡️ Skip this sample and continue")
            continue

    logger.info("✅ FINISH RERANK EVALUATION")
    return results


# =========================
# Run
# =========================

if __name__ == "__main__":
    eval_results = evaluate_rag(
        eval_path="data/hybrid_eval_set_openai.json",
        k1=20,
        k2=5,
        sleeping_time=10,
    )

    with open(FINAL_RESULT_FILE, "w", encoding="utf-8") as f:
        json.dump(eval_results, f, ensure_ascii=False, indent=2)

    print("✅ Evaluation finished. Results saved.")
