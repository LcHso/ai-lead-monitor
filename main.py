import os
import json
import time
import logging
import requests
from datetime import datetime
from typing import List, Dict, Any
from dashscope import Generation
from config import config

# 日志增强：增加文件记录，方便回溯
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("scan.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

STATE_FILE = "state.json"

def load_state() -> Dict[str, Any]:
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"notified_ids": [], "last_run": None}

def save_state(state: Dict[str, Any]):
    state["last_run"] = datetime.now().isoformat()
    # 状态文件增加缩进，方便人工复核
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    logger.info("State saved locally.")

def fetch_apps() -> List[Dict]:
    all_apps = []
    headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
    # 去重集合
    seen_ids = set()
    
    for keyword in config.keywords:
        url = "https://itunes.apple.com/cn/search"
        params = {"term": keyword, "country": config.country, "entity": "software", "limit": 100} # 提升单次采集量
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=15)
            if resp.status_code != 200: continue
            
            results = resp.json().get("results", [])
            for app in results:
                track_id = str(app['trackId'])
                if track_id in seen_ids: continue
                
                seller = app.get('sellerName', '')
                if seller in config.exclude_sellers: continue
                
                seen_ids.add(track_id)
                all_apps.append({
                    "id": track_id, 
                    "name": app['trackName'],
                    "desc": app.get('description', '')[:1200], # 稍微增加描述长度提高AI准确度
                    "url": app['trackViewUrl'],
                    "seller": seller, 
                    "genre": app.get('primaryGenreName', '')
                })
        except Exception as e:
            logger.error(f"Fetch failed for {keyword}: {e}")
    return all_apps

def analyze_batch(apps: List[Dict]) -> List[Dict]:
    if not apps: return []
    qualified_apps = []
    batch_size = 3 # 减小 batch 以获得更精细的回复
    
    for i in range(0, len(apps), batch_size):
        batch = apps[i:i+batch_size]
        prompt_content = "作为阿里云大模型专家，分析以下 App 列表。判断标准：高频对话、长期角色记忆、情感陪伴。\n\n"
        for idx, app in enumerate(batch):
            prompt_content += f"[{idx}] 名称:{app['name']}, 开发商:{app['seller']}, 描述:{app['desc']}\n"
        
        prompt_content += """
        返回 JSON 数组格式，包含：
        - index: 索引
        - is_companion: 是否为高频 AI 陪伴/社交类 (true/false)
        - token_level: Token 消耗等级 (High/Medium/Low)
        - pain_point: 客户最核心的成本痛点 (如：上下文太长、记忆丢失、并发波动)
        - score: 商业价值评分 (1-10)
        - pitch: 针对阿里云产品的具体销售切入点 (如：推荐 Context Cache、新客补贴、百炼集成)
        """

        try:
            # 升级为 qwen-max 以获得更强的逻辑推理（如果额度允许）
            response = Generation.call(
                model="qwen-turbo", 
                api_key=config.dashscope_api_key,
                messages=[{'role': 'system', 'content': '你是一个精通云计算销售和AI架构的专家。'},
                          {'role': 'user', 'content': prompt_content}], 
                result_format='message'
            )
            content = response.output.choices[0].message.content.replace("```json", "").replace("```", "").strip()
            results = json.loads(content)
            
            for res in results:
                if res.get("is_companion") and res.get("score", 0) >= 6: # 过滤低分 App
                    app = batch[res['index']]
                    app['analysis'] = res
                    qualified_apps.append(app)
        except Exception as e:
            logger.error(f"Batch analysis failed: {e}")
            
    # 按评分降序排列
    qualified_apps.sort(key=lambda x: x['analysis'].get('score', 0), reverse=True)
    return qualified_apps

def filter_new_leads(apps: List[Dict], state: Dict) -> List[Dict]:
    notified_ids = set(state.get("notified_ids", []))
    new_leads = []
    for app in apps:
        if app['id'] not in notified_ids:
            new_leads.append(app)
            notified_ids.add(app['id'])
    
    # 状态保留最近 2000 个，防止 state.json 过大
    state["notified_ids"] = list(notified_ids)[-2000:]
    return new_leads

def send_report(apps: List[Dict]):
    if not apps: 
        logger.info("No new high-value leads found today.")
        return
        
    today = datetime.now().strftime("%m月%d日")
    title = f"🚩 发现 {len(apps)} 个 AI 陪伴高价值新客户"
    
    content = []
    for i, app in enumerate(apps[:config.top_n], 1):
        a = app['analysis']
        # 消耗等级可视化
        level_map = {"High": "🔴 极高", "Medium": "🟡 中等", "Low": "🟢 较低"}
        token_ui = level_map.get(a['token_level'], a['token_level'])
        
        # 组装富文本
        text = (
            f"### {i}. 【{app['name']}】\n"
            f"- **消耗等级**：{token_ui} (评分: {a['score']})\n"
            f"- **开发主体**：`{app['seller']}`\n"
            f"- **技术痛点**：{a.get('pain_point', '未知')}\n"
            f"- **🎯 销售策略**：{a['pitch']}\n"
            f"- [点此快速调研]({app['url']})\n"
            f"---"
        )
        content.append(text)
    
    full_text = f"# {title}\n\n" + "\n".join(content) + f"\n\n> **战役提示**：请优先联系 High 等级的客户，主打 Context Cache 降本方案。"
    
    body = {
        "msgtype": "markdown", 
        "markdown": {"title": title, "text": full_text}
    }
    
    try:
        r = requests.post(config.dingtalk_webhook, json=body, timeout=10)
        logger.info(f"Report sent to DingTalk: {r.text}")
    except Exception as e:
        logger.error(f"Failed to send report: {e}")

if __name__ == "__main__":
    start_time = time.time()
    logger.info("=== 430 战役：C端社交雷达启动 ===")
    
    state = load_state()
    raw_apps = fetch_apps()
    logger.info(f"App Store 扫描完成，获取原始数据: {len(raw_apps)} 条")
    
    qualified_apps = analyze_batch(raw_apps)
    logger.info(f"AI 深度研判完成，筛选高价值目标: {len(qualified_apps)} 条")
    
    new_leads = filter_new_leads(qualified_apps, state)
    logger.info(f"过滤重复，新增潜客: {len(new_leads)} 条")
    
    send_report(new_leads)
    save_state(state)
    
    duration = time.time() - start_time
    logger.info(f"=== 扫描结束，耗时 {duration:.2f}s ===")
