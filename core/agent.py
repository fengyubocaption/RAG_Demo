# core/agent.py
from langchain.agents import create_agent
from langchain.messages import HumanMessage, SystemMessage

from core import qwen_utils
from core.tools import AGENT_TOOLS
from core.memory import get_chat_history, add_messages_to_history

llm = qwen_utils.get_qwen_llm()


# (全局字典 sessions_history 已经被彻底删除了)

async def run_research_agent(question: str, session_id: str = "default_user") -> str:
    """
    组装并运行研究助手 Agent，返回最终综合解答。
    具备多轮对话的上下文记忆能力 (Redis 驱动)。
    """
    # 1. 异步从 Redis 拉取记忆
    history = await get_chat_history(session_id)

    # 如果是全新的会话，手动塞入一条 SystemMessage 设为基座
    if not history:
        sys_msg = SystemMessage(
            "你是一个高级研究分析师。任务是综合本地私有文档和互联网信息来回答问题。\n"
            "【决策逻辑】\n"
            "1. 若问题偏向私有知识，优先调用 search_local_files。\n"
            "2. 若问题涉及外部事实，调用 web_search。\n"
            "【输出要求】\n"
            "务必在回答末尾明确标注信息来源。\n"
            "【特殊要求】\n"
            "你会记住之前的对话上下文，如果用户的问题缺乏主语，请根据历史记忆进行推理补充。"
        )
        history = [sys_msg]
        # 把这条系统设定立刻写入 Redis，防止并发穿透
        await add_messages_to_history(session_id, [sys_msg])

    # 2. 构造本次输入
    current_input = {"messages": history + [HumanMessage(content=question)]}

    # 3. 创建并执行 Agent
    agent = create_agent(model=llm, tools=AGENT_TOOLS)
    print(f">>> [Agent] 启动 | 会话ID: {session_id} | 当前记忆长度: {len(history)} | 提问: {question}")

    response = await agent.ainvoke(current_input)

    # 4. 提取增量消息并异步回写 Redis
    new_messages = response["messages"][len(history):]
    await add_messages_to_history(session_id, new_messages)

    return response["messages"][-1].content