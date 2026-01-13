from query.core import get_llm
from pydantic import BaseModel
from typing import List, Optional
import ast

class QueryPlan(BaseModel):
    sql: str
    need_chart: bool
    chart_type: Optional[str] = None   # line | bar | pie
    x: Optional[str] = None
    y: Optional[str] = None
    title: Optional[str] = None

from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template("""
Bạn là một Data Analyst chuyên SQL MySQL.

Schema database:
{schema}

Nhiệm vụ:
- Sinh câu SQL hợp lệ dựa trên schema trên
- KHÔNG bịa bảng hoặc cột không tồn tại
- CHỈ dùng SELECT

Sau đó quyết định:
- Có cần vẽ biểu đồ không
- Nếu có: loại biểu đồ, trục x, y, tiêu đề

Câu hỏi người dùng:
{question}
""")


llm = get_llm('openai-oss')
llm = llm.with_structured_output(QueryPlan)

from langchain_community.utilities import SQLDatabase

db = SQLDatabase.from_uri(
    "mysql+mysqlconnector://root:123456@localhost:3306/datait3170"
)

from langchain_core.tools import tool
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import io
import cv2

@tool
def plot_chart(payload: dict):
    """
    Vẽ biểu đồ từ dữ liệu SQL.

    payload:
      chart_type: line | bar | pie
      data: list[dict]
      x: cột trục x
      y: cột trục y
      title: tiêu đề (optional)
    """
    df = pd.DataFrame(payload["data"])
    plt.figure(figsize=(8, 5))

    if payload["chart_type"] == "line":
        plt.plot(df[payload["x"]], df[payload["y"]])
    elif payload["chart_type"] == "bar":
        plt.bar(df[payload["x"]], df[payload["y"]])
    elif payload["chart_type"] == "pie":
        plt.pie(df[payload["y"]], labels=df[payload["x"]], autopct="%1.1f%%")

    plt.title(payload.get("title", "Biểu đồ"))

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close()

    return cv2.imdecode(
        np.frombuffer(buf.getvalue(), np.uint8),
        cv2.IMREAD_COLOR
    )
    
from sqlalchemy import text


def run(question: str):
    # 1. LLM lập kế hoạch
    chain = prompt | llm
    plan = chain.invoke({
        "question": question,
        "schema": db.get_table_info()
    })

    with db._engine.connect() as conn:
        result = conn.execute(text(plan.sql))
        df = pd.DataFrame(result.fetchall(), columns=result.keys())

    if not plan.need_chart:
        return df

    return plot_chart.invoke({
        "payload": {
            "chart_type": plan.chart_type,
            "data": df.to_dict("records"),
            "x": df.columns[0],
            "y": df.columns[1],
            "title": plan.title
        }}
    )

result = run("Vẽ thống kê ngày đến phòng khám của khách sử dụng phòng khám nhiều nhất")

if isinstance(result, np.ndarray):
    cv2.imshow("Chart", result)
    cv2.waitKey(0)
else:
    print(result)
