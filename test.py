from query.core import HistoryManager, get_llm, get_embedding_model, get_cross_encoder
from query.medical.medical_rag import MedicalNativeRAG, MedicalRerankRAG
from langchain_openai import ChatOpenAI

# medical_search = MedicalSearch(max_results=3)
# query = "thông tin thuốc Paracetamol?"
# results = medical_search.answer(query)
# print("Kết quả tìm kiếm và trả lời:")
# print(results.answer)
# print("Nguồn:")
# print(results.source)

# history = HistoryManager(get_llm())
# history.put_history("user1", "user", "Hello, how are you?")
# history.put_history("user1", "assistant", "I'm fine, thank you!")
# history.put_history("user1", "user", "Can you tell me about Paracetamol?")
# history.put_history("user1", "assistant", "Paracetamol is a common pain reliever and fever reducer.")
# history.put_history("user1", "user", "What are its side effects?")
# history.put_history("user1", "assistant", "Common side effects include nausea and allergic reactions.")
# history.put_history("user1", "user", "Thank you for the information!")
# history.put_history("user1", "assistant", "You're welcome! If you have any more questions, feel free to ask.")
# history.put_history("user1", "user", "Actually, can you summarize our conversation?")
# history.put_history("user1", "assistant", "Sure! We discussed Paracetamol, its uses, and side effects.")
# history.put_history("user1", "user", "Great, thanks!")
# history.put_history("user1", "assistant", "No problem!")
# history.put_history("user1", "user", "One more thing, what is the recommended dosage?")
# history.put_history("user1", "assistant", "The recommended dosage for adults is usually 500 mg to 1000 mg every 4 to 6 hours, not exceeding 4000 mg per day.")
# history.put_history("user1", "user", "Got it, thanks for the help!")

# print(history.get_history("user1"))
# print(history.get_history("user1"))

embedder = get_embedding_model()
reranker = get_cross_encoder()
rag = MedicalRerankRAG(embedder, reranker)

context = rag.query("Các loại thuốc tương tự Paracetamol")
for con in context:
    print(con)
