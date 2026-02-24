import os
import json
import time
import logging
import requests
from datetime import datetime
from typing import List, Dict, Any
from dashscope import Generation
from config import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

STATE_FILE = "state.json"

def load_state() -> Dict[str, Any]:
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"notified_ids": [], "last_run": None}

def save_state(state: Dict[str, Any]):
    state["last_run"] = datetime.now().isoformat()
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    logger.info("State saved locally.")

def fetch_apps() -> List[Dict]:
    all_apps = []
    headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
    for keyword in config.keywords:
        url = "https://itunes.apple.com/cn/search"
        params = {"term": keyword, "country": config.country, "entity": "software", "limit": 50}
        try:
            for attempt in range(3):
                resp = requests.get(url, params=params, headers=headers, timeout=15)
                if resp.status_code == 200: break
                time.sleep(2)
            results = resp.json().get("results", [])
            for app in results:
                seller = app.get('sellerName', '')
                if seller in config.exclude_sellers: continue
                all_apps.append({
                    "id": str(app['trackId']), "name": app['trackName'],
                    "desc": app.get('description', '')[:1000], "url": app['trackViewUrl'],
                    "seller": seller, "genre": app.get('primaryGenreName', '')
                })
            time.sleep(0.5)
        except Exception as e:
            logger.error(f"Fetch failed for {keyword}: {e}")
    return all_apps

def analyze_batch(apps: List[Dict]) -> List[Dict]:
    if not apps: return []
    qualified_apps = []
    batch_size = 5
    for i in range(0, len(apps), batch_size):
        batch = apps[i:i+batch_size]
        prompt_content = "请分析以下 App 列表，返回 JSON 数组：\n"
        for idx, app in enumerate(batch):
            prompt_content += f"[{idx}] 名称:{app['name']}, 开发商:{app['seller']}, 描述:{app['desc']}\n"
        prompt_content += """
        要求：1. 判断是否为“高频交互 AI 陪伴类”(排除纯工具)。2. 预估 Token 消耗等级 (High/Medium/Low)。3. 商业评分 (1-10)。4. 销售切入话术。
        返回格式 (纯 JSON 数组): [{"index": 0, "is_companion": true, "token_level": "High", "score": 9, "pitch": "话术..."}]
        """
        try:
            response = Generation.call(model="qwen-turbo", api_key=config.dashscope_api_key,
                messages=[{'role': 'user', 'content': prompt_content}], result_format='message')
            content = response.output.choices[0].message.content.replace("```json", "").replace("```", "").strip()
            results = json.loads(content)
            for res in results:
                if res.get("is_companion"):
                    app = batch_data[res['index']]
                    app['analysis'] = res
                    qualified_apps.append(app)
        except Exception as e:
            logger.error(f"Batch analysis failed: {e}")
    qualified_apps.sort(key=lambda x: x['analysis'].get('score', 0), reverse=True)
    return qualified_apps

# 修正 batch_data 引用问题
def analyze_batch(apps: List[Dict]) -> List[Dict]:
    if not apps: return []
    qualified_apps = []
    batch_size = 5
    for i in range(0, len(apps), batch_size):
        batch = apps[i:i+batch_size]
        prompt_content = "请分析以下 App 列表，返回 JSON 数组：\n"
        for idx, app in enumerate(batch):
            prompt_content += f"[{idx}] 名称:{app['name']}, 开发商:{app['seller']}, 描述:{app['desc']}\n"
        prompt_content += """
        要求：1. 判断是否为“高频交互 AI 陪伴类”(排除纯工具)。2. 预估 Token 消耗等级 (High/Medium/Low)。3. 商业评分 (1-10)。4. 销售切入话术。
        返回格式 (纯 JSON 数组): [{"index": 0, "is_companion": true, "token_level": "High", "score": 9, "pitch": "话术..."}]
        """
        try:
            response = Generation.call(model="qwen-turbo", api_key=config.dashscope_api_key,
                messages=[{'role': 'user', 'content': prompt_content}], result_format='message')
            content = response.output.choices[0].message.content.replace("```json", "").replace("```", "").strip()
            results = json.loads(content)
            for res in results:
                if res.get("is_companion"):
                    app = batch[res['index']] # 修正这里
                    app['analysis'] = res
                    qualified_apps.append(app)
        except Exception as e:
            logger.error(f"Batch analysis failed: {e}")
    qualified_apps.sort(key=lambda x: x['analysis'].get('score', 0), reverse=True)
    return qualified_apps

def filter_new_leads(apps: List[Dict], state: Dict) -> List[Dict]:
    notified_ids = set(state.get("notified_ids", []))
    new_leads = []
    for app in apps:
        if app['id'] not in notified_ids:
            new_leads.append(app)
            notified_ids.add(app['id'])
    if len(notified_ids) > 1000: notified_ids = set(list(notified_ids)[-1000:])
    state["notified_ids"] = list(notified_ids)
    return new_leads

def send_report(apps: List[Dict]):
    if not apps: return
    if not config.dingtalk_webhook: return
    today = datetime.now().strftime("%Y-%m-%d")
    title = f"🇨🇳 新增 AI 陪伴潜客 {len(apps)} 个"
    content = []
    for i, app in enumerate(apps[:config.top_n], 1):
        a = app['analysis']
        level_color = "🔴" if a['token_level'] == "High" else "🟡"
        text = f"{i}. **{app['name']}** (评分：{a['score']}) {level_color}\n   - 开发商：{app['seller']}\n   - 💡 **销售切入**：{a['pitch']}\n   - [查看 App]({app['url']})\n"
        content.append(text)
    body = {"msgtype": "markdown", "markdown": {"title": title, "text": f"### {title}\n\n" + "\n".join(content) + f"\n\n> 生成时间：{today}\n> 数据源：App Store CN"}}
    try:
        requests.post(config.dingtalk_webhook, json=body, headers={"Content-Type": "application/json"})
    except Exception as e:
        logger.error(f"Failed to send report: {e}")

if __name__ == "__main__":
    logger.info("=== Start Production Scan ===")
    state = load_state()
    raw_apps = fetch_apps()
    logger.info(f"Fetched {len(raw_apps)} raw apps.")
    qualified_apps = analyze_batch(raw_apps)
    logger.info(f"Qualified {len(qualified_apps)} apps after AI analysis.")
    new_leads = filter_new_leads(qualified_apps, state)
    logger.info(f"Found {len(new_leads)} new leads.")
    send_report(new_leads)
    save_state(state)
    logger.info("=== Scan Finished ===")
