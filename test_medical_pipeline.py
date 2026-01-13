"""
Test file để kiểm tra Medical Query Pipeline hoàn chỉnh
"""
from query.medical_query_pipeline import MedicalQueryPipeline

# Khởi tạo pipeline
print("=" * 70)
print("KHỞI TẠO MEDICAL QUERY PIPELINE")
print("=" * 70)
pipeline = MedicalQueryPipeline(max_retries=3)

# Test case 1: Câu hỏi đơn giản về thuốc
print("\n" + "=" * 70)
print("TEST CASE 1: Câu hỏi đơn giản về thuốc")
print("=" * 70)
query1 = "Tôi bị đau bụng thì nên dùng thuốc gì?"
print(f"\nCâu hỏi: {query1}")
print("\nĐang xử lý...")
result1 = pipeline.process_query(query1)
print(f"\n Câu trả lời cuối cùng:")
print(f"{result1.answer}")
print(f"\n Nguồn tham khảo: {', '.join(result1.sources) if result1.sources else 'Không có'}")
print(f" Độ tin cậy: {result1.confidence:.2f}")

# Test case 2: Câu hỏi phức tạp về nhiều thông tin
print("\n" + "=" * 70)
print("TEST CASE 2: Câu hỏi phức tạp về nhiều thông tin")
print("=" * 70)
query2 = "Tôi muốn biết về thuốc Paracetamol: công dụng, liều lượng và tác dụng phụ"
print(f"\nCâu hỏi: {query2}")
print("\nĐang xử lý...")
result2 = pipeline.process_query(query2)
print(f"\n Câu trả lời cuối cùng:")
print(f"{result2.answer}")
print(f"\n Nguồn tham khảo: {', '.join(result2.sources) if result2.sources else 'Không có'}")
print(f" Độ tin cậy: {result2.confidence:.2f}")

# Test case 3: Câu hỏi về triệu chứng
print("\n" + "=" * 70)
print("TEST CASE 3: Câu hỏi về triệu chứng")
print("=" * 70)
query3 = "Tôi bị sốt và đau đầu, nên dùng thuốc gì?"
print(f"\nCâu hỏi: {query3}")
print("\nĐang xử lý...")
result3 = pipeline.process_query(query3)
print(f"\n Câu trả lời cuối cùng:")
print(f"{result3.answer}")
print(f"\n Nguồn tham khảo: {', '.join(result3.sources) if result3.sources else 'Không có'}")
print(f" Độ tin cậy: {result3.confidence:.2f}")

print("\n" + "=" * 70)
print("TEST HOÀN TẤT")
print("=" * 70)

