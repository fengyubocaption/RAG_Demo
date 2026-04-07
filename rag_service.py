# rag_service.py
import os
import jieba
from langchain_classic.retrievers import MultiQueryRetriever, ContextualCompressionRetriever, EnsembleRetriever
from langchain_community.document_compressors import DashScopeRerank
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings.dashscope import DashScopeEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.globals import set_debug

import qwen_utils
from document_loader import load_and_split_directory

# 打开debug模式
# set_debug(True)

# ==========================================
# 第一部分：全局初始化 (整个服务器生命周期只执行一次)
# ==========================================
print(">>> [系统] 正在初始化 RAG 知识库 (全局构建)...")

# 1. 预先切分文档并构建向量库
split_docs = load_and_split_directory("file")
embeddings = DashScopeEmbeddings(model="text-embedding-v2")

# 向量检索器 (召回 Top 6，撒大网)
vectorstore = FAISS.from_documents(split_docs, embeddings)
vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 6})

# BM25 检索器 (纯关键词召回 Top 6，兜底专有名词)
bm25_retriever = BM25Retriever.from_documents(
    split_docs,
    preprocess_func=jieba.lcut # 直接注入底层函数方法
)
bm25_retriever.k = 6

# 2. 全局复用的大模型和 Prompt
llm = qwen_utils.get_qwen_llm()
prompt = ChatPromptTemplate.from_template(
    "请严格根据以下背景资料回答问题。如果资料中没有相关信息，请明确回答不知道。\n\n背景资料:\n{context}\n\n用户问题: {question}"
)

print(">>> [系统] 基础组件初始化完毕！")


# ==========================================
# 第二部分：策略工厂函数
# ==========================================
def create_multi_query_retriever(llm, base_retriever):
    return MultiQueryRetriever.from_llm(retriever=base_retriever, llm=llm)


def create_hyde_retriever(llm, base_retriever):
    hyde_prompt = ChatPromptTemplate.from_template(
        """你是一个专业的文档撰写助手。请针对用户提出的问题，写一段字数约 200 字的伪造回答。
        要求：使用陈述句，包含专业术语，逻辑自洽。

        用户问题: {question}
        伪造回答:"""
    )
    hyde_chain = hyde_prompt | llm | StrOutputParser()
    return hyde_chain | base_retriever


def create_hybrid_rerank_retriever():
    """
    创建 混合检索 + DashScopeRerank 重排 检索器
    """
    # 步骤 A：将两路检索合并 (权重各占 50%)
    # 它会自动执行 RRF 算法，去除两路召回的重复项，最终输出综合排名
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.5, 0.5]
    )

    # 步骤 B：初始化 DashScopeRerank
    # top_n=3 表示经过精细打分后，只保留最最相关的前 3 个文档块给大模型
    dashscope_api_key = os.getenv("DASHSCOPE_API_KEY")
    compressor = DashScopeRerank(
        model="qwen3-vl-rerank",
        top_n=3
    )

    # 步骤 C：使用 ContextualCompressionRetriever 包装合并后的检索器
    # 数据流向：Ensemble召回约 12 个 -> Compressor打分过滤 -> 输出最精准的 3 个
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=ensemble_retriever
    )

    return compression_retriever


def create_ultimate_retriever():
    """
    终极检索策略：
    1. Multi-Query: 扩展问题维度
    2. Hybrid Search: 向量(FAISS) + 关键词(BM25) 双路召回
    3. Rerank: 阿里 DashScope 重排模型精选 Top 3
    """

    # --- 步骤 1: 构建基础的混合检索器 (Hybrid) ---
    # 同时从语义和关键字两个维度捞数据
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.4, 0.6]  # 向量检索通常权重略高
    )

    # --- 步骤 2: 在混合检索之上叠加 Multi-Query ---
    # 让 LLM 生成多个衍生问题，每个问题都去跑一遍 Hybrid 检索
    # 这极大地提高了召回率，防止用户提问太模糊
    mq_hybrid_retriever = MultiQueryRetriever.from_llm(
        retriever=ensemble_retriever,
        llm=llm
    )

    # --- 步骤 3: 接入 Rerank 重排过滤器 ---
    # 前面捞回来的文档可能非常多（Multi-Query 会导致倍增）
    # 使用 Rerank 模型强力打分，只留下最精华的 3 个
    compressor = DashScopeRerank(
        model="qwen-reranker",  # 请确保模型名正确
        top_n=3
    )

    # 使用压缩器包装前面的复杂检索链
    ultimate_compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=mq_hybrid_retriever
    )

    return ultimate_compression_retriever


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# ==========================================
# 第三部分：核心业务调用 (每次 API 请求都会执行)
# ==========================================
async def process_question(question: str, strategy: str = "naive") -> str:
    """
    接收每次用户的提问，根据策略动态组装轻量级的 LCEL 链并执行。
    """
    # 1. 根据传入的策略，动态选择检索器（复用全局的 vector_retriever）
    if strategy == "multi_query":
        current_retriever = create_multi_query_retriever(llm, vector_retriever)
    elif strategy == "hyde":
        current_retriever = create_hyde_retriever(llm, vector_retriever)
    elif strategy == "hybrid":
        current_retriever = create_hybrid_rerank_retriever()
    elif strategy == "ultimate":
        current_retriever = create_ultimate_retriever()
    else:
        current_retriever = vector_retriever

    # 2. 动态组装 RAG 链条
    chain = (
            {"context": current_retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
    )

    # 3. 真正执行发问请求
    return await chain.ainvoke(question)