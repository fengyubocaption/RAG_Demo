from langsmith import evaluate, Client
from langchain_core.output_parsers import StrOutputParser

import qwen_utils
from rag_service import (
    vector_retriever,
    create_multi_query_retriever,
    create_hyde_retriever,
    create_hybrid_rerank_retriever,
    prompt,
    format_docs
)

client = Client()
dataset_name = "RAG-Advanced-Dataset"


# ==========================================
# 核心魔法：使用工厂函数动态生成“答题机器”
# ==========================================
def create_predict_function(strategy_name: str):
    """
    这是一个工厂函数：你告诉它要什么策略，它就返回一个专门用该策略答题的函数
    """

    def predict_rag_answer(inputs: dict) -> dict:
        question = inputs["question"]
        llm = qwen_utils.get_qwen_llm()

        # 根据闭包传进来的 strategy_name 动态选择检索器
        if strategy_name == "naive":
            retriever = vector_retriever
        elif strategy_name == "multi_query":
            retriever = create_multi_query_retriever(llm, vector_retriever)
        elif strategy_name == "hyde":
            retriever = create_hyde_retriever(llm, vector_retriever)
        elif strategy_name == "hybrid":
            retriever = create_hybrid_rerank_retriever()
        else:
            raise ValueError(f"未知的策略: {strategy_name}")

        # 统一的执行逻辑
        docs = retriever.invoke(question)
        context_str = format_docs(docs)

        chain = prompt | llm | StrOutputParser()
        answer = chain.invoke({"context": context_str, "question": question})

        return {"answer": answer, "context": context_str}

    # 返回组装好的真实答题函数
    return predict_rag_answer


# ==========================================
# 自动化批量执行
# ==========================================
if __name__ == "__main__":
    # 定义你要参加大考的 4 位选手
    strategies_to_test = ["naive", "multi_query", "hyde", "hybrid"]
    # strategies_to_test = ["naive"]

    print("\n🚀 启动全自动化批量评估，正在按顺序交卷...\n")

    for strategy in strategies_to_test:
        print(f"========== 正在测试选手: [{strategy}] ==========")

        # 1. 动态获取当前策略的答题函数
        current_predict_func = create_predict_function(strategy)

        # 2. 提交给 LangSmith
        evaluate(
            current_predict_func,
            data=dataset_name,
            experiment_prefix=f"{strategy}-eval",  # 让每次实验名字都不一样
            metadata={"strategy": strategy}
        )
        print(f"✅ 选手 [{strategy}] 试卷已上交！\n")

    print("🎉 所有 4 种策略已全部跑完！请前往 LangSmith 网页端查看对比图表！")