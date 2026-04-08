# core/tools/weather_tool.py
import aiohttp
import requests
from pydantic import BaseModel, Field
from langchain_core.tools import tool

class WeatherInput(BaseModel):
    location: str = Field(..., description="需要查询天气的城市名称，例如：'Beijing', 'Shanghai'")
    unit: str = Field(default="celsius", description="温度单位，必须是 'celsius' (摄氏度) 或 'fahrenheit' (华氏度)")

@tool(args_schema=WeatherInput)
async def get_weather_advanced(location: str, unit: str) -> str:
    """查询指定城市的实时天气。务必提供城市名称。"""
    print(f"\n[🛠️ 工具] 正在查询 {location} 的天气...")
    try:
        format_param = "%C,+%t" if unit == "celsius" else "%C,+%f"
        url = f"https://wttr.in/{location}?format={format_param}"

        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=5) as response:
                if response.status == 200:
                    weather_text = await response.text()
                    return f"{location} 的当前天气: {weather_text.strip()}"
                return f"获取失败，状态码: {response.status}"
    except Exception as e:
        return f"网络请求异常: {str(e)}"