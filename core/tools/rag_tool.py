# core/tools/rag_tool.py
from langchain_core.tools import tool
from core import qwen_utils
from core.retriever import get_retriever_by_strategy

llm = qwen_utils.get_qwen_llm()


@tool
async def search_local_files(query: str) -> str:
    """当用户询问关于公司内部文档、技术方案或已上传的私有资料时，必须使用此工具。"""
    print(f"\n[🛠️ 工具] 正在异步调用本地 RAG 检索: '{query}'...")

    retriever = get_retriever_by_strategy("ultimate", llm)

    # 核心改造：使用 ainvoke 进行异步检索
    docs = await retriever.ainvoke(query)

    if not docs:
        return "本地知识库中未找到相关内容。"

    context = "\n\n".join([f"文档片段:\n{doc.page_content}" for doc in docs])
    return f"本地检索结果:\n{context}"