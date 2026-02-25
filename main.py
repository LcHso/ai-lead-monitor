import os
import json
import time
import logging
import requests
import re
from datetime import datetime
from typing import List, Dict, Any
from urllib.parse import quote
from dashscope import Generation
from config import config

# === 日志增强：同时输出到文件 + 控制台 ===
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("scan.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

STATE_FILE = "state.json"

# === 1. 状态管理 ===
def load_state() -> Dict[str, Any]:
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            # 兼容旧版本
            if "notified_ids" not in data:
                data["notified_ids"] = []
            return data
    return {"notified_ids": [], "last_run": None}

def save_state(state: Dict[str, Any]):
    state["last_run"] = datetime.now().isoformat()
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    logger.info("State saved locally.")

# === 2. 数据抓取（App Store CN） ===
def fetch_apps() -> List[Dict]:
    all_apps = []
    seen_ids = set()
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
    }
    
    for keyword in config.keywords:
        url = "https://itunes.apple.com/cn/search"
        params = {
            "term": keyword,
            "country": config.country,
            "entity": "software",
            "limit": 100  # 提升单次采集量
        }
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=15)
            if resp.status_code != 200:
                logger.warning(f"Fetch failed with status {resp.status_code} for {keyword}")
                continue
            
            results = resp.json().get("results", [])
            for app in results:
                track_id = str(app['trackId'])
                if track_id in seen_ids:
                    continue
                
                seller = app.get('sellerName', '')
                if seller in config.exclude_sellers:
                    continue
                
                seen_ids.add(track_id)
                all_apps.append({
                    "id": track_id,
                    "name": app['trackName'],
                    "desc": app.get('description', '')[:1200],  # 增加描述长度
                    "url": app['trackViewUrl'],
                    "seller": seller,
                    "genre": app.get('primaryGenreName', '')
                })
            time.sleep(0.3)  # 礼貌请求
        except Exception as e:
            logger.error(f"Fetch failed for {keyword}: {e}")
    logger.info(f"Total apps fetched: {len(all_apps)}")
    return all_apps

# === 3. AI 批量分析（增强 JSON 解析） ===
def analyze_batch(apps: List[Dict]) -> List[Dict]:
    if not apps:
        return []
    
    qualified_apps = []
    batch_size = 3  # 小批量保证质量
    
    for i in range(0, len(apps), batch_size):
        batch = apps[i:i+batch_size]
        prompt_content = "作为阿里云大模型专家，分析以下 App 列表。判断标准：高频对话、长期角色记忆、情感陪伴。\n\n"
        for idx, app in enumerate(batch):
            prompt_content += f"[{idx}] 名称:{app['name']}, 开发商:{app['seller']}, 描述:{app['desc']}\n"
        prompt_content += """
直接返回一个 JSON 数组，不要包含任何解释文字、markdown 标记或其他内容。每个元素包含：
- index: 整数，对应上面列表的索引
- is_companion: 布尔值，是否为高频 AI 陪伴/社交类
- token_level: 字符串，Token 消耗等级 (High/Medium/Low)
- pain_point: 字符串，客户最核心的成本痛点
- score: 整数，商业价值评分 (1-10)
- pitch: 字符串，针对阿里云产品的具体销售切入点
示例：[{"index":0,"is_companion":true,"token_level":"High","pain_point":"上下文太长","score":9,"pitch":"推荐 Context Cache"}]
"""
        try:
            response = Generation.call(
                model="qwen-turbo",
                api_key=config.dashscope_api_key,
                messages=[
                    {'role': 'system', 'content': '你是一个精通云计算销售和 AI 架构的专家，只输出纯 JSON 格式数据，不要任何额外文字。'},
                    {'role': 'user', 'content': prompt_content}
                ],
                result_format='message'
            )
            content = response.output.choices[0].message.content.strip()
            
            # === 增强 JSON 清理逻辑 ===
            # 移除 markdown 代码块
            if content.startswith("```"):
                parts = content.split("```")
                content = parts[1] if len(parts) > 1 else content
                if content.startswith("json"):
                    content = content[4:]
            content = content.strip()
            
            # 提取第一个 [ 到最后一个 ] 之间的内容
            start_idx = content.find("[")
            end_idx = content.rfind("]")
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                content = content[start_idx:end_idx+1]
            
            results = json.loads(content)
            
            for res in results:
                if res.get("is_companion") and res.get("score", 0) >= 6:
                    app = batch[res['index']]
                    app['analysis'] = res
                    qualified_apps.append(app)
                    
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode failed: {e}, content: {content[:200]}...")
        except Exception as e:
            logger.error(f"Batch analysis failed: {e}")
        time.sleep(0.5)  # 避免请求过快
    
    qualified_apps.sort(key=lambda x: x['analysis'].get('score', 0), reverse=True)
    logger.info(f"Qualified apps after AI analysis: {len(qualified_apps)}")
    return qualified_apps

# === 4. 背景调查：生成找人路径（新增功能） ===
def research_contact_path(app_name: str, seller_name: str) -> str:
    """
    利用 AI 推演该公司的联系线索寻找方案
    返回：Markdown 格式的搜索链接 + 破冰话术
    """
    encoded_seller = quote(seller_name)
    encoded_app = quote(app_name)
    
    prompt = f"""
App 名称: {app_name}
开发商: {seller_name}

你是一名资深 B2B 销售顾问，请为这个 App 生成"找人路径"：

1. 【备案查询】生成天眼查/企查查搜索链接
2. 【社交搜索】生成即刻/小红书搜索链接（搜产品名看用户反馈）
3. 【技术溯源】生成 GitHub 搜索链接（搜 App 名找开源项目或开发者）
4. 【职业网络】生成领英搜索链接（搜"公司名 + CTO/创始人"）
5. 【破冰话术】写一句针对 AI 陪伴赛道的技术型破冰话术（突出 Token 成本/上下文长度/并发优化）

返回格式（严格 Markdown，不要其他文字）：
- [🔍 查备案/电话](https://www.tianyancha.com/search?key={encoded_seller})
- [💬 即刻搜产品](https://web.okjike.com/search?keyword={encoded_app})
- [💻 GitHub 搜代码](https://github.com/search?q={encoded_app}&type=repositories)
- [👔 领英搜 CTO](https://www.linkedin.com/search/results/people/?keywords={encoded_seller}%20CTO)
- 💡 **破冰金句**：`你的话术`
"""
    try:
        response = Generation.call(
            model="qwen-turbo",
            api_key=config.dashscope_api_key,
            messages=[{'role': 'user', 'content': prompt}],
            result_format='message'
        )
        result = response.output.choices[0].message.content.strip()
        # 二次清理，确保是纯 Markdown
        if result.startswith("```"):
            result = result.split("```", 2)[1].strip()
        return result
    except Exception as e:
        logger.warning(f"Contact research failed for {app_name}: {e}")
        # 兜底：返回基础搜索链接
        return (
            f"- [🔍 查备案/电话](https://www.tianyancha.com/search?key={encoded_seller})\n"
            f"- [💬 即刻搜产品](https://web.okjike.com/search?keyword={encoded_app})\n"
            f"- [💻 GitHub 搜代码](https://github.com/search?q={encoded_app}&type=repositories)\n"
            f"- [👔 领英搜 CTO](https://www.linkedin.com/search/results/people/?keywords={encoded_seller}%20CTO)\n"
            f"- 💡 **破冰金句**：`您好，关注到贵司 {app_name} 在 AI 陪伴赛道的创新，我们有针对长上下文场景的优化方案，方便交流吗？`"
        )

# === 5. 去重逻辑 ===
def filter_new_leads(apps: List[Dict], state: Dict) -> List[Dict]:
    notified_ids = set(state.get("notified_ids", []))
    new_leads = []
    for app in apps:
        if app['id'] not in notified_ids:
            new_leads.append(app)
            notified_ids.add(app['id'])
    # 状态保留最近 2000 个，防止 state.json 过大
    state["notified_ids"] = list(notified_ids)[-2000:]
    logger.info(f"New leads after deduplication: {len(new_leads)}")
    return new_leads

# === 6. 钉钉推送（集成找人路径） ===
def send_report(apps: List[Dict]):
    if not apps:
        logger.info("No new high-value leads found today.")
        return
    
    if not config.dingtalk_webhook:
        logger.warning("DingTalk webhook not configured.")
        return
    
    today = datetime.now().strftime("%m 月%d 日")
    title = f"🚩 发现 {len(apps)} 个 AI 陪伴高价值新客户"
    
    content = []
    for i, app in enumerate(apps[:config.top_n], 1):
        a = app['analysis']
        # 消耗等级可视化
        level_map = {"High": "🔴 极高", "Medium": "🟡 中等", "Low": "🟢 较低"}
        token_ui = level_map.get(a['token_level'], a['token_level'])
        
        # 🔥 新增：调用背景调查（每个 App 只查一次）
        contact_path = research_contact_path(app['name'], app['seller'])
        
        # 组装富文本
        text = (
            f"### {i}. 【{app['name']}】\n"
            f"- **消耗等级**：{token_ui} (评分: {a['score']})\n"
            f"- **开发主体**：`{app['seller']}`\n"
            f"- **技术痛点**：{a.get('pain_point', '未知')}\n"
            f"- **🎯 销售策略**：{a['pitch']}\n"
            f"- **🎯 一键找人**：\n{contact_path}\n"
            f"- [📱 App Store]({app['url']})\n"
            f"---"
        )
        content.append(text)
        time.sleep(0.5)  # 避免 AI 调用过快
    
    full_text = f"# {title}\n\n" + "\n".join(content) + f"\n\n> **战役提示**：请优先联系 High 等级的客户，主打 Context Cache 降本方案。"
    
    body = {
        "msgtype": "markdown",
        "markdown": {
            "title": title,
            "text": full_text
        }
    }
    
    try:
        r = requests.post(config.dingtalk_webhook, json=body, timeout=15, headers={"Content-Type": "application/json"})
        if r.status_code == 200:
            logger.info("Report sent to DingTalk successfully.")
        else:
            logger.error(f"DingTalk API error: {r.status_code}, {r.text}")
    except Exception as e:
        logger.error(f"Failed to send report: {e}")

# === 主流程 ===
if __name__ == "__main__":
    start_time = time.time()
    logger.info("=== 430 战役：C 端社交雷达启动 ===")
    
    # 1. 加载状态
    state = load_state()
    
    # 2. 获取数据
    raw_apps = fetch_apps()
    logger.info(f"App Store 扫描完成，获取原始数据: {len(raw_apps)} 条")
    
    # 3. AI 分析
    qualified_apps = analyze_batch(raw_apps)
    logger.info(f"AI 深度研判完成，筛选高价值目标: {len(qualified_apps)} 条")
    
    # 4. 去重
    new_leads = filter_new_leads(qualified_apps, state)
    logger.info(f"过滤重复，新增潜客: {len(new_leads)} 条")
    
    # 5. 发送报告（含背景调查）
    send_report(new_leads)
    
    # 6. 保存状态
    save_state(state)
    
    duration = time.time() - start_time
    logger.info(f"=== 扫描结束，耗时 {duration:.2f}s ===")
