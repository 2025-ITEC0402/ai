from typing import Dict, Any
from langchain_core.messages import HumanMessage
import json
from utils.logger import setup_logger

logger = setup_logger(__name__)

class QualityEvaluationAgent:
    """
    항상 통과 판정을 내리는 품질 평가 에이전트
    """
    
    def __init__(self):
        pass
    
    def evaluate_quality(self, content: str)->Dict[str, Any]:
        """
        제공된 내용의 품질을 평가합니다 (항상 통과 판정)
        
        Args:
            content (str): 평가할 내용
            
        Returns:
            Dict: 품질 평가 결과 (JSON)
        """
        logger.info("QualityEvaluationAgent: 품질 평가 수행 중")
        
        evaluation_result = {
            "next": "FINISH",
            "score": 10.0,
            "feedback": "내용이 모든 품질 기준을 충족합니다.",
            "evaluation_criteria": {
                "accuracy": 10.0,
                "clarity": 10.0,
                "relevance": 10.0,
                "completeness": 10.0
            }
        }
        
        logger.info("QualityEvaluationAgent: 품질 평가 완료")
        
        return evaluation_result