# core/memory.py
import json
import redis.asyncio as redis
from typing import List
from langchain_core.messages import BaseMessage, messages_from_dict, messages_to_dict

# 1. 建立全局异步连接池 (decode_responses=True 让我们拿到的直接是字符串)
redis_pool = redis.ConnectionPool.from_url("redis://127.0.0.1:6379/0", decode_responses=True)
redis_client = redis.Redis(connection_pool=redis_pool)

# 配置项
SESSION_TTL = 86400  # 记忆存活时间：24小时 (超过一天不聊，自动忘掉，节省内存)
MAX_HISTORY_LEN = 11  # 历史消息最大保留条数 (防止上下文爆炸)


async def get_chat_history(session_id: str) -> List[BaseMessage]:
    """从 Redis 异步获取反序列化后的历史消息"""
    key = f"chat_history:{session_id}"

    # 获取列表中的所有元素
    raw_data = await redis_client.lrange(key, 0, -1)
    if not raw_data:
        return []

    # 将 JSON 字符串解析为字典
    dicts = [json.loads(msg) for msg in raw_data]
    # 使用 LangChain 内置方法将字典转回 Message 对象
    return messages_from_dict(dicts)


async def add_messages_to_history(session_id: str, messages: List[BaseMessage]):
    """将新消息序列化并异步存入 Redis，同时维护长度和过期时间"""
    if not messages:
        return

    key = f"chat_history:{session_id}"

    # 序列化
    dicts = messages_to_dict(messages)
    json_msgs = [json.dumps(d, ensure_ascii=False) for d in dicts]

    # 2. 使用 Pipeline 打包命令，减少网络 RTT 开销
    async with redis_client.pipeline(transaction=True) as pipe:
        # 将新消息推入列表尾部
        await pipe.rpush(key, *json_msgs)
        # 滑动窗口：利用 ltrim 只保留最后 MAX_HISTORY_LEN 条！(这步太关键了)
        await pipe.ltrim(key, -MAX_HISTORY_LEN, -1)
        # 刷新过期时间
        await pipe.expire(key, SESSION_TTL)
        # 一次性执行所有命令
        await pipe.execute()