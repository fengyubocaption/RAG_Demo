# core/retriever.py
import jieba
import os
import pickle
from langchain_community.vectorstores import Milvus
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import MultiQueryRetriever, ContextualCompressionRetriever, EnsembleRetriever
from langchain_community.document_compressors import DashScopeRerank
from langchain_community.embeddings.dashscope import DashScopeEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from config import settings
from .document_loader import load_and_split_path

# ==========================================
# 全局初始化：秒级启动架构
# ==========================================
BM25_CACHE_PATH = os.path.join(settings.DATA_DIR, "bm25_cache.pkl")

print(">>> [检索层] 正在初始化 BM25 内存词表...")

if os.path.exists(BM25_CACHE_PATH):
    print(">>> [检索层] 发现 BM25 硬盘缓存，正在执行【秒级反序列化加载】...")
    with open(BM25_CACHE_PATH, "rb") as f:
        # 只读取剥离出来的纯数据字典
        cache_data = pickle.load(f)

        # 用纯数据重新组装一个干净的、带新锁的 Retriever 对象！
        bm25_retriever = BM25Retriever(
            docs=cache_data["docs"],
            vectorizer=cache_data["vectorizer"],
            preprocess_func=jieba.lcut
        )
        bm25_retriever.k = 6
else:
    print(">>> [检索层] 未发现缓存，需读取全量文件构建词表 (仅需一次)...")
    split_docs = load_and_split_path(settings.FILE_DIR)
    bm25_retriever = BM25Retriever.from_documents(split_docs, preprocess_func=jieba.lcut)
    bm25_retriever.k = 6

    # 【核心解法】只抽取它核心的文档和算法模型，避开 LangChain 外壳的线程锁！
    cache_data = {
        "docs": bm25_retriever.docs,
        "vectorizer": bm25_retriever.vectorizer
    }
    with open(BM25_CACHE_PATH, "wb") as f:
        pickle.dump(cache_data, f)
    print(">>> [检索层] BM25 纯数据缓存已生成！")

# 2. 核心改变：直接实例化连接现有的 Milvus 数据库，而不是每次重建！
embeddings = DashScopeEmbeddings(model="text-embedding-v2")
vectorstore = Milvus(
    embedding_function=embeddings,
    connection_args={"host": "127.0.0.1", "port": "19530"},
    collection_name="rag_collection",
    auto_id=True
)
vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 6})

print(">>> [检索层] 基础双路检索器就绪！(Milvus 直连成功)")

# ==========================================
# 策略工厂函数
# ==========================================
def get_retriever_by_strategy(strategy: str, llm):
    """
    根据策略名称，动态返回组装好的高级检索器
    """
    if strategy == "multi_query":
        return MultiQueryRetriever.from_llm(retriever=vector_retriever, llm=llm)

    elif strategy == "hyde":
        hyde_prompt = ChatPromptTemplate.from_template(
            "你是一个专业的文档撰写助手。请针对用户提出的问题，写一段字数约 200 字的伪造回答。\n要求：使用陈述句，包含专业术语，逻辑自洽。\n\n用户问题: {question}\n伪造回答:"
        )
        hyde_chain = hyde_prompt | llm | StrOutputParser()
        return hyde_chain | vector_retriever

    elif strategy == "hybrid":
        ensemble = EnsembleRetriever(retrievers=[bm25_retriever, vector_retriever], weights=[0.5, 0.5])
        compressor = DashScopeRerank(model="qwen3-vl-rerank", top_n=3)
        return ContextualCompressionRetriever(base_compressor=compressor, base_retriever=ensemble)

    elif strategy == "ultimate":
        ensemble = EnsembleRetriever(retrievers=[bm25_retriever, vector_retriever], weights=[0.4, 0.6])
        mq_hybrid = MultiQueryRetriever.from_llm(retriever=ensemble, llm=llm)
        compressor = DashScopeRerank(model="qwen-reranker", top_n=3)
        return ContextualCompressionRetriever(base_compressor=compressor, base_retriever=mq_hybrid)

    return vector_retriever