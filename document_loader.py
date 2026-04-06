# document_loader.py
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_and_split_document(file_path: str):
    """
    负责读取本地文档，并进行语义切分，返回标准的 Document 对象列表。
    """
    print(f">>> [数据层] 开始加载文件: {file_path}")

    # 1. 加载文档
    try:
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        print(f">>> [数据层] 文件加载成功，共 {len(documents)} 页。")
    except Exception as e:
        print(f">>> [数据层] 文件加载失败: {e}")
        raise e

    # 2. 语义切分
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", "。", "！", "？", ".", "?", "!", " ", ""]
    )

    split_docs = text_splitter.split_documents(documents)
    print(f">>> [数据层] 文档切分完毕，共产生 {len(split_docs)} 个文本块。")

    return split_docs