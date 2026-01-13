"""
Pipeline tích hợp hoàn chỉnh cho nhánh phía trên (Medical Knowledge path).
Luồng: User Query -> Split Query -> K Queries -> Router -> RAG + Answer -> Eval Answer 
-> (loop back hoặc Web search + Answer) -> Summary -> Final Answer
"""
from typing import Optional
from .split_query import SplitQueryHandler
from .router.router import Router
from .medical.medical_pipeline import MedicalPipeline
from .medical.medical_search import MedicalSearch
from .eval_answer import EvalAnswerHandler
from .summary import SummaryHandler
from .final_answer import FinalAnswerHandler
from .core import AnswerQuery, FinalAnswer

import logging

logger = logging.getLogger(__name__)


class MedicalQueryPipeline:
    """
    Pipeline hoàn chỉnh xử lý câu hỏi y tế theo kiến trúc:
    User Query -> Split Query -> K Queries -> Router -> RAG + Answer -> Eval Answer 
    -> (loop back hoặc Web search + Answer) -> Summary -> Final Answer
    """
    
    def __init__(self, max_retries: int = 3):
        """
        Khởi tạo pipeline.
        
        Args:
            max_retries: Số lần thử tối đa (M) cho RAG + Answer trước khi chuyển sang web search
        """
        self.split_handler = SplitQueryHandler()
        self.router = Router()
        self.medical_pipeline = MedicalPipeline()
        self.medical_search = MedicalSearch(max_results=3)
        self.eval_handler = EvalAnswerHandler(max_tries=max_retries)
        self.summary_handler = SummaryHandler()
        self.final_handler = FinalAnswerHandler()
        self.max_retries = max_retries
    
    def process_query(self, user_query: str) -> FinalAnswer:
        """
        Xử lý câu hỏi người dùng theo pipeline hoàn chỉnh.
        
        Args:
            user_query: Câu hỏi từ người dùng
            
        Returns:
            FinalAnswer: Câu trả lời cuối cùng
        """
        logger.info(f"Processing query: {user_query}")
        
        # Bước 1: Split Query thành K Queries
        split_result = self.split_handler.split(user_query)
        k_queries = split_result.queries
        logger.info(f"Split into {len(k_queries)} sub-queries")
        
        # Bước 2: Xử lý từng query trong K queries
        all_answers = []
        
        for query in k_queries:
            logger.info(f"Processing sub-query: {query}")
            
            # Bước 3: Router để xác định datasource
            route_result = self.router.route(query)
            logger.info(f"Routed to: {route_result.datasource}")
            
            # Chỉ xử lý nếu route đến medical_knowledge
            if route_result.datasource == "medical_knowledge":
                answer = self._process_medical_query(query)
                if answer:
                    all_answers.append(answer)
            else:
                # Nếu route đến store_database, bỏ qua (nhánh khác xử lý)
                logger.info(f"Skipping query routed to {route_result.datasource}")
        
        # Bước 4: Summary tất cả các answers
        if not all_answers:
            # Nếu không có answer nào, trả về câu trả lời mặc định
            return FinalAnswer(
                answer="Xin lỗi, tôi không tìm thấy thông tin phù hợp để trả lời câu hỏi của bạn.",
                sources=[],
                confidence=0.0
            )
        
        summary = self.summary_handler.summarize(user_query, all_answers)
        logger.info("Summarized all answers")
        
        # Bước 5: Final Answer
        final_answer = self.final_handler.generate(user_query, summary)
        logger.info("Generated final answer")
        
        return final_answer
    
    def _process_medical_query(self, query: str) -> Optional[AnswerQuery]:
        """
        Xử lý một câu hỏi y tế: RAG + Answer -> Eval Answer -> (loop back hoặc Web search).
        
        Args:
            query: Câu hỏi y tế
            
        Returns:
            AnswerQuery: Câu trả lời từ RAG hoặc Web search
        """
        # Thử RAG + Answer với retry logic
        for try_count in range(1, self.max_retries + 1):
            logger.info(f"Attempt {try_count}/{self.max_retries} for RAG + Answer")
            
            # RAG + Answer
            rag_answer = self._get_rag_answer(query)
            
            if not rag_answer:
                # Nếu không có kết quả từ RAG, chuyển sang web search ngay
                logger.info("No RAG results, switching to web search")
                return self._get_web_search_answer(query)
            
            # Eval Answer
            eval_result = self.eval_handler.evaluate(query, rag_answer.answer, try_count)
            logger.info(f"Evaluation: satisfactory={eval_result.is_satisfactory}, score={eval_result.score:.2f}")
            
            # Nếu đạt yêu cầu, trả về
            if eval_result.is_satisfactory:
                logger.info("Answer is satisfactory, returning RAG answer")
                return rag_answer
            
            # Nếu không đạt và không nên retry, chuyển sang web search
            if not eval_result.should_retry:
                logger.info("Should not retry, switching to web search")
                return self._get_web_search_answer(query)
            
            # Nếu nên retry và chưa đạt max, tiếp tục loop
            logger.info(f"Retrying RAG + Answer (try {try_count}/{self.max_retries})")
        
        # Nếu đã thử hết max_retries mà vẫn không đạt, chuyển sang web search
        logger.info("Max retries reached, switching to web search")
        return self._get_web_search_answer(query)
    
    def _get_rag_answer(self, query: str) -> Optional[AnswerQuery]:
        """
        Lấy câu trả lời từ RAG.
        
        Args:
            query: Câu hỏi
            
        Returns:
            AnswerQuery hoặc None nếu không tìm thấy
        """
        try:
            # Query RAG để lấy documents
            results = self.medical_pipeline.medical_rag.query(query)
            thresholded_results = [
                res.payload["text"] 
                for res in results 
                if res.score >= self.medical_pipeline.similarity_threshold
            ]
            
            if thresholded_results:
                context = "\n\n".join(thresholded_results)
                answer = self.medical_pipeline.process_medical_answer(query, context=context)
                return answer
            
            return None
        except Exception as e:
            logger.error(f"Error in RAG query: {e}")
            return None
    
    def _get_web_search_answer(self, query: str) -> AnswerQuery:
        """
        Lấy câu trả lời từ Web search.
        
        Args:
            query: Câu hỏi
            
        Returns:
            AnswerQuery: Câu trả lời từ web search
        """
        try:
            answer = self.medical_search.answer(query)
            return answer
        except Exception as e:
            logger.error(f"Error in web search: {e}")
            return AnswerQuery(
                answer="Xin lỗi, không thể tìm kiếm thông tin trên web.",
                source="Lỗi hệ thống"
            )

