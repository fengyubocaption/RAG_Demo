# core/agent.py
from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain.messages import HumanMessage, SystemMessage
from langchain_community.tools import DuckDuckGoSearchRun

from core import qwen_utils
from core.retriever import get_retriever_by_strategy

llm = qwen_utils.get_qwen_llm()


@tool
def search_local_files(query: str) -> str:
    """
    当用户询问关于公司内部文档、私有技术方案、或刚上传的资料时，必须使用此工具。
    输入应是高度概括的搜索关键词。
    """
    print(f"\n[🛠️ Agent] 正在调用本地 RAG 检索: '{query}'...")

    # 完美复用你昨天的终极策略！
    retriever = get_retriever_by_strategy("ultimate", llm)
    docs = retriever.invoke(query)

    if not docs:
        return "本地知识库中未找到相关内容。"

    # 将高亮文档块打包给大模型看
    context = "\n\n".join([f"文档片段:\n{doc.page_content}" for doc in docs])
    return f"本地检索结果:\n{context}"


@tool
def web_search(query: str) -> str:
    """
    当用户询问最新新闻、当下日期、产品最新评价、或本地文档中没有的通用外部知识时，使用此工具。
    """
    print(f"\n[🛠️ Agent] 正在进行全网搜索: '{query}'...")
    search = DuckDuckGoSearchRun()
    try:
        result = search.invoke(query)
        return f"网络搜索结果:\n{result}"
    except Exception as e:
        return f"网络搜索失败: {str(e)}"


tools = [search_local_files, web_search]


async def run_research_agent(question: str) -> str:
    """
    组装并运行研究助手 Agent，返回最终综合解答
    """
    system_msg = SystemMessage(
        "你是一个高级研究分析师。任务是综合本地私有文档和互联网信息来回答问题。\n"
        "【决策逻辑】\n"
        "1. 若问题偏向私有知识，优先调用 search_local_files。\n"
        "2. 若问题涉及外部时效性事实，调用 web_search。\n"
        "3. 若需对比，可依次调用两者。\n"
        "【输出要求】\n"
        "务必在回答末尾明确标注信息来源，例如：[来源：本地文档] 或 [来源：网络搜索]。"
    )

    agent = create_agent(model=llm, tools=tools)
    inputs = {"messages": [system_msg, HumanMessage(question)]}

    print(f">>> [业务层] 研究助手启动，目标任务: {question}")

    # 注意：API 接口通常需要直接返回完整结果，所以这里用 invoke 而不是 stream
    response = await agent.ainvoke(inputs)

    # 从状态机最后一条消息中提取大模型的最终回答
    final_answer = response["messages"][-1].content
    return final_answer