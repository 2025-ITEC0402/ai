"""
EMA 시스템의 로깅 설정
"""
import logging
import sys
from config import LOG_LEVEL, LOG_FORMAT

def setup_logger(name):
    """
    지정된 이름으로 로거를 설정하고 반환합니다.
    
    Args:
        name (str): 로거 이름
        
    Returns:
        logging.Logger: 설정된 로거 인스턴스
    """
    logger = logging.getLogger(name)
    
    # 로그 레벨 설정
    level = getattr(logging, LOG_LEVEL)
    logger.setLevel(level)
    
    # 핸들러가 이미 있는지 확인
    if not logger.handlers:
        # 콘솔 핸들러 추가
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(LOG_FORMAT))
        logger.addHandler(handler)
    
    return logger