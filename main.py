import os
import json
import time
import logging
import requests
import re
from datetime import datetime
from typing import List, Dict, Any, Tuple
from urllib.parse import quote
from dashscope import Generation
from config import config

# === 日志配置 ===
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
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                if "app_history" not in data:
                    data["app_history"] = {}
                if "notified_ids" not in data:
                    data["notified_ids"] = []
                return data
        except Exception as e:
            logger.warning(f"Load state failed: {e}")
            return {"app_history": {}, "notified_ids": [], "last_run": None}
    return {"app_history": {}, "notified_ids": [], "last_run": None}

def save_state(state: Dict[str, Any]):
    state["last_run"] = datetime.now().isoformat()
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    logger.info("State saved.")

# === 2. 数据抓取 ===
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
            "limit": 100
        }
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=15)
            if resp.status_code != 200:
                logger.warning(f"Fetch failed with status {resp.status_code} for {keyword}")
                continue
            
            results = resp.json().get("results", [])
            for rank, app in enumerate(results, 1):
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
                    "desc": app.get('description', '')[:1200],
                    "url": app['trackViewUrl'],
                    "seller": seller,
                    "genre": app.get('primaryGenreName', ''),
                    "keyword": keyword,
                    "rank": rank
                })
            time.sleep(0.3)
        except Exception as e:
            logger.error(f"Fetch failed for {keyword}: {e}")
    
    logger.info(f"Total apps fetched: {len(all_apps)}")
    return all_apps

# === 3. AI 批量分析 ===
def analyze_batch(apps: List[Dict]) -> List[Dict]:
    if not apps:
        return []
    
    qualified_apps = []
    batch_size = 3
    
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
            
            if content.startswith("```"):
                parts = content.split("```", 2)
                if len(parts) > 1:
                    content = parts[1].strip()
                    if content.startswith("json"):
                        content = content[4:].strip()
            
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
                    
        except Exception as e:
            logger.error(f"Batch analysis failed: {e}")
        time.sleep(0.5)
    
    qualified_apps.sort(key=lambda x: x['analysis'].get('score', 0), reverse=True)
    logger.info(f"Qualified apps after AI analysis: {len(qualified_apps)}")
    return qualified_apps

# === 4. 排名变化分析 ===
def analyze_ranking_changes(apps: List[Dict], state: Dict) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    history = state.get("app_history", {})
    new_leads = []
    rising_leads = []
    falling_leads = []
    
    for app in apps:
        app_id = app['id']
        current_rank = app.get('rank', 999)
        current_keyword = app.get('keyword', 'unknown')
        
        if app_id not in history:
            new_leads.append(app)
            history[app_id] = {
                "rank": current_rank,
                "keyword": current_keyword,
                "score": app['analysis'].get('score', 0),
                "level": app['analysis'].get('token_level', 'Low'),
                "date": datetime.now().strftime("%Y-%m-%d"),
                "name": app['name']
            }
        else:
            old_rank = history[app_id].get("rank", 999)
            old_keyword = history[app_id].get("keyword", "")
            
            if current_keyword == old_keyword:
                rank_diff = old_rank - current_rank
                
                if rank_diff >= 5:
                    app['ranking_info'] = {
                        "old_rank": old_rank,
                        "new_rank": current_rank,
                        "diff": rank_diff,
                        "trend": "rising"
                    }
                    rising_leads.append(app)
                elif rank_diff <= -5:
                    app['ranking_info'] = {
                        "old_rank": old_rank,
                        "new_rank": current_rank,
                        "diff": rank_diff,
                        "trend": "falling"
                    }
                    falling_leads.append(app)
            
            history[app_id] = {
                "rank": current_rank,
                "keyword": current_keyword,
                "score": app['analysis'].get('score', 0),
                "level": app['analysis'].get('token_level', 'Low'),
                "date": datetime.now().strftime("%Y-%m-%d"),
                "name": app['name']
            }
    
    state["app_history"] = history
    logger.info(f"New: {len(new_leads)}, Rising: {len(rising_leads)}, Falling: {len(falling_leads)}")
    return new_leads, rising_leads, falling_leads

# === 5. 背景调查：生成找人路径 ===
def research_contact_path(app_name: str, seller_name: str) -> str:
    encoded_seller = quote(seller_name)
    encoded_app = quote(app_name)
    
    contact_lines = [
        f"- [🔍 ICP 备案查询](https://icp.chinaz.com/?query={encoded_seller}) **（推荐：查官网 + 公司名）**",
        f"- [🏢 天眼查/企查查](https://www.tianyancha.com/search?key={encoded_seller})",
        f"- [💬 即刻搜产品](https://web.okjike.com/search?keyword={encoded_app})",
        f"- [📱 小红书搜评测](https://www.xiaohongshu.com/search_result?keyword={encoded_app})",
        f"- [💻 GitHub 搜代码](https://github.com/search?q={encoded_app}&type=repositories)",
        f"- [👔 领英搜 CTO](https://www.linkedin.com/search/results/people/?keywords={encoded_seller}%20CTO)",
        f"- [💼 脉脉搜公司](https://maimai.cn/web/search?query={encoded_seller})",
    ]
    
    domain_guess = seller_name.lower().replace(' ', '').replace('(', '').replace(')', '').replace('科技', '').replace('公司', '')
    contact_lines.append(f"- 📧 **邮箱猜测**：`contact@{domain_guess}.com` | `bd@{domain_guess}.com`")
    
    pitch = f"\"您好，我是阿里云大模型团队的。关注到贵司{app_name}在 AI 陪伴赛道的创新，我们的 Context Cache 方案可帮助长对话场景降低 40% Token 成本，已有类似客户案例，方便安排 15 分钟技术交流吗？\""
    contact_lines.append(f"- 💡 **破冰话术**：{pitch}")
    
    return "\n".join(contact_lines)

# === 6. 钉钉推送 ===
def send_report(new_leads: List[Dict], rising_leads: List[Dict], falling_leads: List[Dict]):
    if not new_leads and not rising_leads and not falling_leads:
        logger.info("No significant changes found today.")
        return
    
    if not config.dingtalk_webhook:
        logger.warning("DingTalk webhook not configured.")
        return
    
    today = datetime.now().strftime("%m 月%d 日")
    title = f"🚩 {today} AI 陪伴情报：新增{len(new_leads)} | 上升{len(rising_leads)} | 下降{len(falling_leads)}"
    
    content = []
    
    if rising_leads:
        content.append("## 🚀 排名上升榜单 (下载量可能增长)")
        for i, app in enumerate(rising_leads[:10], 1):
            r = app.get('ranking_info', {})
            old_r = r.get('old_rank', 0)
            new_r = r.get('new_rank', 0)
            diff = r.get('diff', 0)
            a = app['analysis']
            text = (
                f"{i}. **【{app['name']}】** [📱 App Store]({app['url']})\n"
                f"   - 排名：{old_r} → {new_r} ⬆️{diff}\n"
                f"   - 关键词：`{app.get('keyword', 'unknown')}`\n"
                f"   - 等级：{a.get('token_level', 'Unknown')} | 评分：{a.get('score', 0)}\n"
                f"   - 策略：{a.get('pitch', '持续跟进')}\n"
                f"   - **🎯 一键找人**：\n{research_contact_path(app['name'], app['seller'])}\n"
            )
            content.append(text)
        content.append("---")
    
    if new_leads:
        content.append("## 🆕 新增高价值潜客")
        for i, app in enumerate(new_leads[:10], 1):
            a = app['analysis']
            level_map = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}
            token_ui = level_map.get(a.get('token_level', 'Low'), "⚪")
            text = (
                f"{i}. **【{app['name']}】** {token_ui} (评分：{a.get('score', 0)}) [📱 App Store]({app['url']})\n"
                f"   - 排名：#{app.get('rank', 999)} | 关键词：`{app.get('keyword', 'unknown')}`\n"
                f"   - 主体：`{app['seller']}`\n"
                f"   - 策略：{a.get('pitch', '暂无')}\n"
                f"   - **🎯 一键找人**：\n{research_contact_path(app['name'], app['seller'])}\n"
            )
            content.append(text)
        content.append("---")
    
    if falling_leads:
        content.append("## ⚠️ 排名下降监控 (可能遇到问题)")
        for i, app in enumerate(falling_leads[:5], 1):
            r = app.get('ranking_info', {})
            old_r = r.get('old_rank', 0)
            new_r = r.get('new_rank', 0)
            diff = r.get('diff', 0)
            text = (
                f"{i}. **【{app['name']}】** [📱 App Store]({app['url']})\n"
                f"   - 排名：{old_r} → {new_r} ⬇️{abs(diff)}\n"
                f"   - 机会点：用户可能流失，可推竞品替代方案\n"
            )
            content.append(text)
    
    full_text = "# " + title + "\n\n" + "\n".join(content)
    full_text += f"\n\n> **提示**：排名上升 = 下载量增长信号，建议优先跟进上升榜单客户。"
    
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
    
    state = load_state()
    raw_apps = fetch_apps()
    logger.info(f"App Store 扫描完成，获取原始数据：{len(raw_apps)} 条")
    
    qualified_apps = analyze_batch(raw_apps)
    logger.info(f"AI 深度研判完成，筛选高价值目标：{len(qualified_apps)} 条")
    
    new_leads, rising_leads, falling_leads = analyze_ranking_changes(qualified_apps, state)
    
    send_report(new_leads, rising_leads, falling_leads)
    
    save_state(state)
    
    duration = time.time() - start_time
    logger.info(f"=== 扫描结束，耗时 {duration:.2f}s ===")
