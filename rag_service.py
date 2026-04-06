# rag_service.py
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings.dashscope import DashScopeEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

import qwen_utils
# 💡 引入我们刚刚抽离出来的数据层方法
from document_loader import load_and_split_document

print(">>> [系统] 正在初始化 RAG 知识库与检索链...")

# 1. 从数据层获取切分好的文档
split_docs = load_and_split_document("sample.md")

# 2. 初始化向量库和检索器
embeddings = DashScopeEmbeddings(model="text-embedding-v2")
vectorstore = FAISS.from_documents(split_docs, embeddings)
retriever = vectorstore.as_retriever()

# 3. 初始化模型与 Prompt
llm = qwen_utils.get_qwen_llm()
prompt = ChatPromptTemplate.from_template(
    "请严格根据以下背景资料回答问题。如果资料中没有相关信息，请明确回答不知道。\n\n背景资料:\n{context}\n\n用户问题: {question}"
)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# 4. 组装全局 LCEL 链
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
print(">>> [系统] RAG 核心服务初始化完毕！")

# 5. 核心业务函数
async def process_question(question: str) -> str:
    """供 main.py 调用的核心业务函数"""
    return await rag_chain.ainvoke(question)