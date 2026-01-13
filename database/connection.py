from langchain_community.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain

db = SQLDatabase.from_uri(
    "mysql+mysqlconnector://root:123456@localhost:3306/clinic_management"
)

from langchain.agents import create_sql_agent
from langchain.agents.agent_types import AgentType
from ..query.core import get_llm

llm = get_llm()

agent = create_sql_agent(
    llm=llm,
    db=db,
    agent_type=AgentType.OPENAI_FUNCTIONS,
    verbose=True
)

