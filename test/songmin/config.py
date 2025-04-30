"""
EMA 시스템 설정 파일
"""
import os
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# API 키 설정
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# 모델 설정
LLM_MODEL = "gemini-2.0-flash"

# 로깅 설정
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"