# test_loader.py
from document_loader import load_and_split_document

# 可以是 .md, .txt, 或者 .pdf
test_file = "file/sample.md"

try:
    # 调用刚才写好的加载与切分逻辑
    chunks = load_and_split_document(test_file)

    print(f"\n🎉 恭喜！成功将文件切分为 {len(chunks)} 个数据块 (Chunks)。")
    print("-" * 50)

    # 3. 打印前 3 个块来观察细节
    for i, chunk in enumerate(chunks[:3]):
        print(f"📦 [第 {i + 1} 块]")

        # 打印元数据 (Metadata)：比如文件来源、所在页码
        print(f"🏷️ 元数据: {chunk.metadata}")

        # 打印字符长度
        print(f"📏 字数: {len(chunk.page_content)}")

        # 打印真正的正文内容
        print(f"📄 内容:\n{chunk.page_content}")

        print("-" * 50)

except Exception as e:
    print(f"❌ 测试出错: {e}")