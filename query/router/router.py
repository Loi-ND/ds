from typing import Literal
from langchain_core.prompts import ChatPromptTemplate
from ..core import get_llm, RouteQuery
from ..prompt_templates import ROUTER_SYSTEM_PROMPT, ROUTER_HUMAN_PROMPT

import logging

logger = logging.getLogger(__name__)

class Router:
    def __init__(self):
        self.llm = get_llm()
        self.structured_llm = self.llm.with_structured_output(RouteQuery)
        self.prompt = self._create_prompt()
        self.router_chain = self.prompt | self.structured_llm

    def _create_prompt(self):

        return ChatPromptTemplate.from_messages(
            [
                ("system", ROUTER_SYSTEM_PROMPT),
                ("human", ROUTER_HUMAN_PROMPT),
            ]
        )

    def route(self, question: str) -> RouteQuery:
        try:
            result = self.router_chain.invoke({"question": question})
            logger.info(f"Routed question to: {result.datasource} - {result.reasoning}")
            return result
        except Exception as e:
            logger.error(f"Error in routing: {e}")
            # Default to medical knowledge if routing fails
            return RouteQuery(
                datasource="medical_knowledge",
                reasoning="Fallback due to routing error",
            )
