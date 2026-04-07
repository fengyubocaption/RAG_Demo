import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_and_split_directory(dir_path: str):
    """
    通用目录加载器：递归扫描目录下所有的 PDF, TXT, MD 文件并统一加载切分
    """
    if not os.path.exists(dir_path) or not os.path.isdir(dir_path):
        raise FileNotFoundError(f"找不到目录或该路径不是文件夹: {dir_path}")

    all_documents = []
    print(f">>> [数据层] 开始扫描目录: {dir_path}")

    # 1. 递归遍历目录下的所有文件
    for root, _, files in os.walk(dir_path):
        for file in files:
            file_path = os.path.join(root, file)
            file_extension = os.path.splitext(file_path)[1].lower()

            # 2. 根据后缀名选择对应的加载器
            if file_extension == ".pdf":
                loader = PyPDFLoader(file_path)
            elif file_extension == ".md":
                loader = UnstructuredMarkdownLoader(file_path)
            elif file_extension == ".txt":
                loader = TextLoader(file_path, encoding="utf-8")
            else:
                # 遇到不支持的文件（比如图片、mp4）直接跳过，不报错
                print(f"  [跳过] 不支持的文件格式: {file}")
                continue

            # 3. 尝试加载文件内容
            try:
                docs = loader.load()
                all_documents.extend(docs)
                print(f"  [成功] 已加载: {file}")
            except Exception as e:
                print(f"  [失败] 无法解析 {file}, 错误: {str(e)}")

    if not all_documents:
        raise ValueError(f"在目录 {dir_path} 中没有找到任何支持的文档！")

    print(f">>> [数据层] 文件全部读取完毕，准备统一进行语义切分...")

    # 4. 统一语义切分 (针对所有合并后的文档)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", "。", "！", "？", ".", "?", "!", " ", ""]
    )

    split_docs = text_splitter.split_documents(all_documents)
    print(f">>> [数据层] 全部切分完毕，共产生 {len(split_docs)} 个文本块。")

    return split_docs