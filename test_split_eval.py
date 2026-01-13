"""
Test file để kiểm tra Split Query và Eval Answer components
"""
from query.split_query import SplitQueryHandler
from query.eval_answer import EvalAnswerHandler

# Test Split Query
print("=" * 50)
print("TEST SPLIT QUERY")
print("=" * 50)

split_handler = SplitQueryHandler()

# Test case 1: Câu hỏi đơn giản
query1 = "Tôi bị đau bụng thì nên dùng thuốc gì?"
result1 = split_handler.split(query1)
print(f"\nCâu hỏi gốc: {query1}")
print(f"Số câu hỏi con: {len(result1.queries)}")
print(f"Các câu hỏi con: {result1.queries}")
print(f"Lý do: {result1.reasoning}")

# Test case 2: Câu hỏi phức tạp
query2 = "Tôi muốn biết về thuốc Paracetamol: công dụng, liều lượng, tác dụng phụ và giá bán"
result2 = split_handler.split(query2)
print(f"\nCâu hỏi gốc: {query2}")
print(f"Số câu hỏi con: {len(result2.queries)}")
print(f"Các câu hỏi con: {result2.queries}")
print(f"Lý do: {result2.reasoning}")

# Test Eval Answer
print("\n" + "=" * 50)
print("TEST EVAL ANSWER")
print("=" * 50)

eval_handler = EvalAnswerHandler(max_tries=3)

# Test case 1: Câu trả lời tốt
query = "Tôi bị đau bụng thì nên dùng thuốc gì?"
good_answer = "Để giảm đau bụng, bạn có thể sử dụng các thuốc như Paracetamol (500-1000mg mỗi 4-6 giờ) hoặc Ibuprofen (200-400mg mỗi 4-6 giờ). Tuy nhiên, nếu đau bụng kéo dài hoặc nghiêm trọng, bạn nên tham khảo ý kiến bác sĩ."
eval_result1 = eval_handler.evaluate(query, good_answer, try_count=1)
print(f"\nCâu hỏi: {query}")
print(f"Câu trả lời: {good_answer[:100]}...")
print(f"Đạt yêu cầu: {eval_result1.is_satisfactory}")
print(f"Điểm số: {eval_result1.score:.2f}")
print(f"Nên thử lại: {eval_result1.should_retry}")
print(f"Lý do: {eval_result1.reasoning}")

# Test case 2: Câu trả lời không tốt
bad_answer = "Tôi không biết."
eval_result2 = eval_handler.evaluate(query, bad_answer, try_count=1)
print(f"\nCâu hỏi: {query}")
print(f"Câu trả lời: {bad_answer}")
print(f"Đạt yêu cầu: {eval_result2.is_satisfactory}")
print(f"Điểm số: {eval_result2.score:.2f}")
print(f"Nên thử lại: {eval_result2.should_retry}")
print(f"Lý do: {eval_result2.reasoning}")

# Test case 3: Đã thử quá nhiều lần
eval_result3 = eval_handler.evaluate(query, bad_answer, try_count=3)
print(f"\nĐã thử 3 lần:")
print(f"Đạt yêu cầu: {eval_result3.is_satisfactory}")
print(f"Nên thử lại: {eval_result3.should_retry} (sẽ là False vì đã đạt max_tries)")
print(f"Lý do: {eval_result3.reasoning}")

print("\n" + "=" * 50)
print("TEST HOÀN TẤT")
print("=" * 50)

