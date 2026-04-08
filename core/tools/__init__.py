# core/tools/__init__.py

# 1. 把各个子文件里的工具导入进来
from .rag_tool import search_local_files
from .web_tool import web_search
from .weather_tool import get_weather_advanced
from .api_tool import fetch_external_user_profile

# 2. 打包成一个总的工具箱列表
AGENT_TOOLS = [
    search_local_files,
    web_search,
    get_weather_advanced,
    fetch_external_user_profile,
]