import os
import json
import numpy as np
from typing import List, Dict, Any, Tuple
import google.generativeai as genai

class EmbeddingSearcher:
    """임베딩을 사용하여 유사한 내용을 검색하는 클래스"""
    
    def __init__(self, embeddings_path: str, chunks_path: str, api_key: str, model_name: str = "models/text-embedding-004"):
        """
        초기화
        
        Args:
            embeddings_path: 임베딩 JSON 파일 경로
            chunks_path: 청크 JSON 파일 경로
            api_key: Google API 키
            model_name: 임베딩 모델 이름
        """
        self.embeddings_path = embeddings_path
        self.chunks_path = chunks_path
        self.api_key = api_key
        self.model_name = model_name
        
        # 임베딩과 청크 로드
        self.embeddings = self._load_embeddings(embeddings_path)
        self.chunks = self._load_chunks(chunks_path)
        
        # Google Generative AI 초기화
        genai.configure(api_key=api_key)
        self.genai = genai
    
    def _load_embeddings(self, embeddings_path: str) -> Dict[str, Any]:
        """
        임베딩 파일 로드
        
        Args:
            embeddings_path: 임베딩 파일 경로
            
        Returns:
            임베딩 데이터
        """
        print(f"임베딩 로드 중: {embeddings_path}")
        if not os.path.exists(embeddings_path):
            raise FileNotFoundError(f"임베딩 파일을 찾을 수 없습니다: {embeddings_path}")
        
        with open(embeddings_path, "r", encoding="utf-8") as f:
            return json.load(f)
    
    def _load_chunks(self, chunks_path: str) -> List[Dict[str, Any]]:
        """
        청크 파일 로드
        
        Args:
            chunks_path: 청크 파일 경로
            
        Returns:
            청크 데이터 목록
        """
        print(f"청크 로드 중: {chunks_path}")
        if not os.path.exists(chunks_path):
            raise FileNotFoundError(f"청크 파일을 찾을 수 없습니다: {chunks_path}")
        
        with open(chunks_path, "r", encoding="utf-8") as f:
            return json.load(f)
    
    def preprocess_query(self, query: str) -> str:
        """
        검색 쿼리 전처리
        
        Args:
            query: 처리할 쿼리
            
        Returns:
            처리된 쿼리
        """
        # 수학 기호 정규화 및 공백 처리
        import re
        
        # 수학 기호 정규화
        replacements = {
            "\u2212": "-",    # 마이너스 기호
            "\u2260": "!=",   # 같지 않음
            "\u2264": "<=",   # 작거나 같음
            "\u2265": ">=",   # 크거나 같음
            "\u221E": "infinity", # 무한대
            "\u222B": "integral", # 적분
            "\u2211": "sum",  # 시그마
            "\u220F": "product", # 파이 (곱)
            "\u2248": "approximately", # 근사
            "\u221A": "sqrt", # 제곱근
        }
        
        for symbol, replacement in replacements.items():
            query = query.replace(symbol, replacement)
        
        # 줄바꿈을 공백으로 변환
        query = re.sub(r'\n+', ' ', query)
        
        # 여러 공백을 하나로 줄이기
        query = re.sub(r'\s+', ' ', query)
        
        return query.strip()
    
    def _compute_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        두 벡터 간 코사인 유사도 계산
        
        Args:
            vec1: 첫 번째 벡터
            vec2: 두 번째 벡터
            
        Returns:
            코사인 유사도 (-1에서 1 사이의 값, 1이 가장 유사)
        """
        # 벡터를 numpy 배열로 변환
        vec1_np = np.array(vec1)
        vec2_np = np.array(vec2)
        
        # 벡터의 크기 계산
        norm1 = np.linalg.norm(vec1_np)
        norm2 = np.linalg.norm(vec2_np)
        
        # 두 벡터가 모두 영벡터가 아닌 경우에만 코사인 유사도 계산
        if norm1 > 0 and norm2 > 0:
            return np.dot(vec1_np, vec2_np) / (norm1 * norm2)
        else:
            return 0.0
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        쿼리와 가장 유사한 청크 검색
        
        Args:
            query: 검색할 쿼리
            top_k: 반환할 결과 수
            
        Returns:
            가장 유사한 청크 목록 (유사도 점수 포함)
        """
        print(f"쿼리 검색 중: {query}")
        
        # 쿼리 전처리
        processed_query = self.preprocess_query(query)
        
        # 쿼리 임베딩 생성
        try:
            embedding_result = self.genai.embed_content(
                model=self.model_name,
                content="Calculus textbook content: " + processed_query,
                # 중요 변경: task_type을 retrieval_query에서 retrieval_document로 변경
                task_type="retrieval_document",
                # title 파라미터 제거 또는 다음과 같이 사용
                # title=None  # title을 제거하거나
            )
            
            # 임베딩 추출
            if hasattr(embedding_result, "embedding") and not callable(embedding_result.embedding):
                query_vector = embedding_result.embedding
            elif hasattr(embedding_result, "values") and not callable(embedding_result.values):
                query_vector = embedding_result.values
            elif isinstance(embedding_result, dict):
                if "embedding" in embedding_result:
                    query_vector = embedding_result["embedding"]
                elif "values" in embedding_result:
                    query_vector = embedding_result["values"]
                else:
                    raise ValueError(f"임베딩 결과에서 벡터를 찾을 수 없습니다: {embedding_result}")
            else:
                raise ValueError(f"임베딩 결과에서 벡터를 찾을 수 없습니다: {type(embedding_result)}")
        except Exception as e:
            raise Exception(f"쿼리 임베딩 생성 중 오류 발생: {e}")
    
    # 이하 코드는 동일...
        
        # 모든 청크와의 유사도 계산
        similarities = []
        
        for chunk_id, chunk_data in self.embeddings.items():
            chunk_vector = chunk_data.get("embedding")
            if not chunk_vector:
                continue
            
            # 청크 ID에 해당하는 텍스트 찾기
            chunk_text = ""
            chunk_metadata = {}
            
            for chunk in self.chunks:
                if chunk["id"] == chunk_id:
                    chunk_text = chunk["text"]
                    chunk_metadata = chunk["metadata"]
                    break
            
            # 유사도 계산
            similarity = self._compute_similarity(query_vector, chunk_vector)
            
            similarities.append({
                "chunk_id": chunk_id,
                "similarity": similarity,
                "text": chunk_text,
                "metadata": chunk_metadata
            })
        
        # 유사도에 따라 정렬 (내림차순)
        sorted_similarities = sorted(similarities, key=lambda x: x["similarity"], reverse=True)
        
        # 상위 k개 결과 반환
        return sorted_similarities[:top_k]
    
    def search_and_format(self, query: str, top_k: int = 5) -> str:
        """
        쿼리와 가장 유사한 청크를 검색하고 결과를 포맷팅
        
        Args:
            query: 검색할 쿼리
            top_k: 반환할 결과 수
            
        Returns:
            포맷팅된 검색 결과
        """
        results = self.search(query, top_k)
        
        if not results:
            return "검색 결과가 없습니다."
        
        formatted_results = f"쿼리: {query}\n\n검색 결과 (상위 {len(results)}개):\n\n"
        
        for i, result in enumerate(results, 1):
            formatted_results += f"결과 {i} (유사도: {result['similarity']:.4f})\n"
            formatted_results += f"페이지: {result['metadata'].get('page', '알 수 없음')}\n"
            formatted_results += f"내용: {result['text']}\n\n"
            formatted_results += "-" * 50 + "\n\n"
        
        return formatted_results


def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description="임베딩을 사용한 검색")
    parser.add_argument("--embeddings", default="output/embeddings/embeddings.json", help="임베딩 파일 경로")
    parser.add_argument("--chunks", default="output/chunks/all_chunks.json", help="청크 파일 경로")
    parser.add_argument("--api-key", required=True, help="Google API 키")
    parser.add_argument("--query", help="검색할 쿼리")
    parser.add_argument("--top-k", type=int, default=5, help="반환할 결과 수")
    parser.add_argument("--interactive", action="store_true", help="대화형 모드 활성화")
    
    args = parser.parse_args()
    
    try:
        # 검색기 초기화
        searcher = EmbeddingSearcher(args.embeddings, args.chunks, args.api_key)
        
        if args.interactive:
            print("대화형 검색 모드 (종료하려면 'exit' 또는 'quit' 입력)")
            while True:
                query = input("\n검색할 쿼리 입력: ")
                if query.lower() in ["exit", "quit"]:
                    break
                
                try:
                    results = searcher.search_and_format(query, args.top_k)
                    print(results)
                except Exception as e:
                    print(f"검색 중 오류 발생: {e}")
        elif args.query:
            results = searcher.search_and_format(args.query, args.top_k)
            print(results)
        else:
            print("쿼리를 입력하거나 대화형 모드를 활성화하세요.")
    except Exception as e:
        print(f"오류 발생: {e}")


if __name__ == "__main__":
    main()