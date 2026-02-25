import os
from pydantic import BaseModel

class Config(BaseModel):
    dashscope_api_key: str = os.environ.get("DASHSCOPE_API_KEY")
    dingtalk_webhook: str = os.environ.get("DINGTALK_WEBHOOK")
    country: str = "CN"
    top_n: int = 20
    keywords: list = ["AI 陪伴", "虚拟恋人", "AI 聊天", "角色扮演", "智能女友", "AI 朋友", "情感倾诉", "树洞","AI聊天", "AI角色", "AI男友", "AI女友", "虚拟人", "AI对话", "Character", "Roleplay"]
    exclude_sellers: list = ["Tencent", "Baidu", "Alibaba", "NetEase"]

    class Config:
        populate_by_name = True

config = Config()
