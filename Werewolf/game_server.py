# wolf_api_deepseek.py
import os
import logging
from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv

# ========== 加载 .env ==========
load_dotenv()

# ========== 日志 ==========
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ========== 全局客户端 ==========
client: OpenAI = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global client
    try:
        api_key = os.getenv("DEEPSEEK_API_KEY") or "sk-5571c6bd730142a49093c262409a5b08"
        if not api_key:
            raise RuntimeError("请设置环境变量 DEEPSEEK_API_KEY")
        client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com"
        )
        logger.info("DeepSeek客户端初始化完成")
        yield
    except Exception as e:
        logger.error(f"初始化失败: {e}")
        raise

app = FastAPI(lifespan=lifespan)

# ========== CORS ==========
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========== 请求体 ==========
class QueryRequest(BaseModel):
    prompt: str
    role: str = "villager"
    phase: str = "day"

# ========== 工具函数 ==========
def build_prompt(prompt: str, role: str, phase: str) -> str:
    """
    根据前端发送的游戏上下文构建给大模型的完整提示。

    参数说明：
        prompt: 前端提供的游戏上下文字符串，其中包含当前回合、阶段、存活玩家以及所有历史公共日志。
        role: 当前 AI 扮演的身份（werewolf/seer/villager 等）。
        phase: 游戏阶段（day/night/vote 等）。

    为了提升 AI 的发言质量，我们在提示中加入更加详细的角色指令和阶段描述，并明确要求 AI 参考全部历史公共日志进行推理。
    """
    # 角色指令：为不同身份定制侧重点
    role_instructions = {
        "werewolf": "你正在扮演狼人角色，请隐藏自己的身份，注意观察并分析其他玩家的发言，适当混淆视听，引导好人错误怀疑目标。",
        "seer": "你正在扮演预言家角色，你是好人阵营的核心，请通过逻辑分析历史公共信息和其他玩家的发言推断出狼人，并引导大家投票正确的目标。",
        "villager": "你正在扮演普通村民角色，你应该积极分析历史公共发言，努力找出可疑的狼人并提醒其他玩家。"
    }
    # 阶段描述：帮助模型理解当前阶段的上下文
    phase_descriptions = {
        "day": "现在是白天讨论阶段，所有存活玩家会轮流发言讨论谁是狼人。请合理发言，不要直接透露自己的身份。",
        "night": "现在是夜晚行动阶段，狼人、预言家和女巫会进行各自的夜间行动。请不要在夜晚公开身份。",
        "vote": "现在是投票阶段，玩家需要根据已有信息投票决定出局的人。请提供投票理由或分析。"
    }
    # 取角色和阶段的中文说明，如果不存在则为空字符串
    role_desc = role_instructions.get(role, "")
    phase_desc = phase_descriptions.get(phase, f"当前阶段：{phase}")

    # 构建提示文本
    prompt_text = f"[角色指令]\n{role_desc}\n{phase_desc}\n\n" \
                  f"[输入内容]\n" \
                  "以下是本局游戏的公共信息，包括系统提示和所有玩家的历史发言：\n" \
                  f"{prompt}\n\n" \
                  "[响应要求]\n" \
                  "请综合所有历史公共日志，结合你的角色视角，用不超过两句中文进行发言或表达你的看法，并注意不要直接暴露自己的身份或阵营。"
    return prompt_text

def clean_response(text: str) -> str:
    return text.replace("<|im_start|>", "").replace("<|im_end|>", "").strip()

# ========== 健康检查 ==========
@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": client is not None}

# ========== 推理接口 ==========
@app.post("/api/query")
async def query(request: QueryRequest):
    if client is None:
        raise HTTPException(status_code=503, detail="DeepSeek客户端未初始化")

    prompt = build_prompt(request.prompt, request.role, request.phase)
    try:
        resp = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "你是一个专业的游戏AI助手，你的任务是根据游戏规则和上下文进行推理和决策。"},
                {"role": "user", "content": prompt}
            ],
            max_tokens=512,
            stream=False,
            temperature=0.7
        )
        return {
            "response": clean_response(resp.choices[0].message.content),
            "status": "success"
        }
    except Exception as e:
        logger.error(f"推理错误: {e}")
        raise HTTPException(status_code=500, detail=f"推理错误: {e}")

# ========== 启动 ==========
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10021)