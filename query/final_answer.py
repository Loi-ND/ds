from langchain_core.prompts import ChatPromptTemplate
from .core import get_llm, FinalAnswer, SummaryAnswer
from .prompt_templates import FINAL_ANSWER_SYSTEM_PROMPT, FINAL_ANSWER_HUMAN_PROMPT

import logging

logger = logging.getLogger(__name__)


class FinalAnswerHandler:
    """
    Tạo câu trả lời cuối cùng cho người dùng từ Summary.
    Dựa trên kiến trúc: Summary -> Final Answer
    """
    
    def __init__(self):
        self.llm = get_llm()
        self.structured_llm = self.llm.with_structured_output(FinalAnswer)
        self.prompt = self._create_prompt()
        self.final_chain = self.prompt | self.structured_llm
    
    def _create_prompt(self):
        """Tạo prompt template cho câu trả lời cuối cùng."""
        return ChatPromptTemplate.from_messages([
            ("system", FINAL_ANSWER_SYSTEM_PROMPT),
            ("human", FINAL_ANSWER_HUMAN_PROMPT),
        ])
    
    def generate(self, query: str, summary: SummaryAnswer) -> FinalAnswer:
        """
        Tạo câu trả lời cuối cùng từ Summary.
        
        Args:
            query: Câu hỏi gốc của người dùng
            summary: Câu trả lời đã được tổng hợp từ Summary
            
        Returns:
            FinalAnswer: Câu trả lời cuối cùng với confidence score
        """
        try:
            sources_text = ", ".join(summary.sources) if summary.sources else "Không có nguồn"
            
            result = self.final_chain.invoke({
                "query": query,
                "summary": summary.summary,
                "sources": sources_text
            })
            
            logger.info(f"Generated final answer with {len(summary.sources)} sources")
            return result
        except Exception as e:
            logger.error(f"Error in generating final answer: {e}")
            # Fallback: trả về summary trực tiếp
            return FinalAnswer(
                answer=summary.summary,
                sources=summary.sources,
                confidence=0.7
            )
    
    def generate_simple(self, query: str, summary: SummaryAnswer) -> str:
        """
        Tạo câu trả lời cuối cùng dạng đơn giản (chỉ trả về text).
        Phương thức tiện ích khi không cần structured output.
        
        Args:
            query: Câu hỏi gốc của người dùng
            summary: Câu trả lời đã được tổng hợp
            
        Returns:
            str: Câu trả lời cuối cùng dạng text
        """
        final_result = self.generate(query, summary)
        return final_result.answer

