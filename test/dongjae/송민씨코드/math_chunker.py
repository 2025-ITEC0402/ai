import os
import re
import json
import numpy as np
from typing import List, Dict, Any
import pdfplumber
from tqdm import tqdm

class SimpleTextChunker:
    """텍스트를 작은 청크 단위로 강제로 나누는 클래스"""
    
    def __init__(self, pdf_path: str, output_dir: str = "output", chunk_size: int = 500):
        """
        초기화
        
        Args:
            pdf_path: PDF 파일 경로
            output_dir: 출력 디렉토리
            chunk_size: 최대 청크 크기 (기본값 500으로 감소)
        """
        self.pdf_path = pdf_path
        self.output_dir = output_dir
        self.chunk_size = chunk_size
        
        # 출력 디렉토리 생성
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "chunks"), exist_ok=True)
    
    def extract_text_from_pdf(self) -> List[str]:
        """
        PDF에서 텍스트 추출 (페이지별로 반환)
        
        Returns:
            페이지별 텍스트 목록
        """
        print(f"PDF 파일에서 텍스트 추출 중: {self.pdf_path}")
        pages = []
        
        try:
            with pdfplumber.open(self.pdf_path) as pdf:
                for i, page in enumerate(tqdm(pdf.pages, desc="페이지 처리 중")):
                    text = page.extract_text()
                    if text:
                        pages.append(text)
        except Exception as e:
            print(f"PDF 파일 처리 중 오류 발생: {e}")
        
        print(f"{len(pages)}개 페이지가 추출되었습니다.")
        return pages
    
    def clean_text(self, text: str) -> str:
        """
        텍스트 정리 및 정규화
        
        Args:
            text: 정리할 텍스트
            
        Returns:
            정리된 텍스트
        """
        # 페이지 번호와 헤더/푸터 제거
        text = re.sub(r'\d+\s*Copyright(?:.+?)require it\.', '', text, flags=re.DOTALL)
        
        # 줄바꿈 하이픈 수정
        text = re.sub(r'(\w+)-\s+(\w+)', r'\1\2', text)
        
        # 여러 공백 정규화
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def has_math_content(self, text: str) -> bool:
        """
        수학 표현식 포함 여부 확인
        
        Args:
            text: 확인할 텍스트
            
        Returns:
            수학 표현식 포함 여부
        """
        # 수학 표현식 패턴
        patterns = [
            r'[a-z]\s*\([a-z]\)\s*=', # 함수 표기: f(x) =
            r'[a-z]\s*=\s*[^.,;:]+', # 방정식: y = 2x + 1
            r'\$[^$]+\$',            # LaTeX: $x^2$
            r'\\frac{[^}]+}{[^}]+}', # LaTeX 분수
            r'\\int',                # 적분
            r'\\sum',                # 시그마
            r'\\prod',               # 프로덕트
            r'\\lim',                # 극한
            r'\|\s*[a-z]\s*\|',      # 절대값: |x|
            r'(?<![a-zA-Z])[a-z]\^\d+(?![a-zA-Z])' # 지수: x^2
        ]
        
        for pattern in patterns:
            if re.search(pattern, text):
                return True
        
        return False
    
    def force_chunk_text(self, text: str, chunk_size: int) -> List[str]:
        """
        텍스트를 강제로 작은 청크로 나눔
        
        Args:
            text: 분할할 텍스트
            chunk_size: 청크 크기
            
        Returns:
            청크 목록
        """
        # 텍스트를 문장으로 분할
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            # 너무 긴 문장은 그 자체로 청크가 될 수 있음
            if len(sentence) > chunk_size:
                # 기존 청크가 있으면 먼저 추가
                if current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = ""
                
                # 긴 문장을 강제로 분할
                words = sentence.split()
                temp_sentence = ""
                
                for word in words:
                    if len(temp_sentence) + len(word) + 1 > chunk_size:
                        chunks.append(temp_sentence)
                        temp_sentence = word
                    else:
                        if temp_sentence:
                            temp_sentence += " " + word
                        else:
                            temp_sentence = word
                
                if temp_sentence:
                    current_chunk = temp_sentence
            else:
                # 현재 청크에 문장 추가 시 크기 초과 여부 확인
                if len(current_chunk) + len(sentence) + 1 > chunk_size:
                    chunks.append(current_chunk)
                    current_chunk = sentence
                else:
                    if current_chunk:
                        current_chunk += " " + sentence
                    else:
                        current_chunk = sentence
        
        # 마지막 청크 추가
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def process(self):
        """전체 처리 과정 실행"""
        # PDF에서 텍스트 추출 (페이지별)
        page_texts = self.extract_text_from_pdf()
        
        all_chunks = []
        chunk_index = 1
        
        # 각 페이지를 처리
        for i, page_text in enumerate(page_texts):
            # 텍스트 정리
            clean_text = self.clean_text(page_text)
            
            # 페이지가 비어있으면 건너뜀
            if not clean_text:
                continue
            
            # 페이지 텍스트를 강제로 작은 청크로 나눔
            page_chunks = self.force_chunk_text(clean_text, self.chunk_size)
            
            # 청크 추가
            for chunk_text in page_chunks:
                # 너무 짧은 청크는 건너뜀 (30자 미만)
                if len(chunk_text) < 30:
                    continue
                    
                has_math = self.has_math_content(chunk_text)
                
                all_chunks.append({
                    "id": f"chunk_{chunk_index}",
                    "text": chunk_text,
                    "metadata": {
                        "page": i + 1,
                        "has_math": has_math
                    }
                })
                
                chunk_index += 1
        
        # 청크 저장
        chunks_path = os.path.join(self.output_dir, "chunks", "all_chunks.json")
        with open(chunks_path, "w", encoding="utf-8") as f:
            json.dump(all_chunks, f, indent=2)
        
        print(f"{len(all_chunks)}개의 청크가 저장되었습니다: {chunks_path}")
        
        # 각 청크에 수학 표현식이 포함되어 있는지 추가 분석
        math_chunks = [chunk for chunk in all_chunks if chunk["metadata"].get("has_math", False)]
        math_ratio = len(math_chunks) / len(all_chunks) if all_chunks else 0
        
        print(f"수학 표현식이 포함된 청크: {len(math_chunks)} ({math_ratio:.1%})")
        
        return all_chunks


class SimpleEmbedder:
    """텍스트 청크의 임베딩을 생성하는 클래스"""
    
    def __init__(self, api_key: str, model_name: str = "models/text-embedding-004"):
        """
        초기화
        
        Args:
            api_key: Google API 키
            model_name: 임베딩 모델 이름
        """
        self.api_key = api_key
        self.model_name = model_name
        
        # Google Generative AI 초기화
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            self.genai = genai
        except ImportError:
            print("Error: google-generativeai 패키지가 설치되지 않았습니다.")
            print("설치 명령어: pip install google-generativeai")
            raise
    
    def preprocess_text(self, text: str) -> str:
        """
        임베딩 전 텍스트 전처리
        
        Args:
            text: 처리할 텍스트
            
        Returns:
            처리된 텍스트
        """
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
            text = text.replace(symbol, replacement)
        
        # 줄바꿈을 공백으로 변환
        text = re.sub(r'\n+', ' ', text)
        
        # 여러 공백을 하나로 줄이기
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def create_embeddings(self, chunks: List[Dict[str, Any]], output_dir: str):
        """
        청크의 임베딩 생성
        
        Args:
            chunks: 임베딩을 생성할 청크 목록
            output_dir: 출력 디렉토리
        """
        print(f"{len(chunks)}개 청크의 임베딩 생성 중...")
        
        # 임베딩 디렉토리 생성
        embeddings_dir = os.path.join(output_dir, "embeddings")
        os.makedirs(embeddings_dir, exist_ok=True)
        
        embeddings = {}
        
        # 각 청크에 대한 임베딩 생성
        for chunk in tqdm(chunks, desc="임베딩 생성 중"):
            chunk_id = chunk["id"]
            
            # 텍스트 전처리
            processed_text = self.preprocess_text(chunk["text"])
            
            # 청크가 너무 짧으면 건너뜀
            if len(processed_text) < 20:
                print(f"청크 {chunk_id}가 너무 짧아 건너뜁니다: {len(processed_text)}자")
                continue
            
            # 임베딩 생성
            try:
                embedding_result = self.genai.embed_content(
                    model=self.model_name,
                    content="Calculus textbook content: " + processed_text,
                    task_type="retrieval_document",
                    title=f"Chunk {chunk_id}"
                )
                
                # 임베딩 추출 (여러 형태 지원)
                vector = None
                
                # Check if embedding_result is a method or function before accessing attributes
                if callable(embedding_result):
                    print(f"Warning: embedding_result is callable for chunk {chunk_id}")
                    continue
                    
                if hasattr(embedding_result, "embedding") and not callable(embedding_result.embedding):
                    vector = embedding_result.embedding
                elif hasattr(embedding_result, "values") and not callable(embedding_result.values):
                    vector = embedding_result.values
                else:
                    # 딕셔너리로 반환될 수도 있음
                    if isinstance(embedding_result, dict):
                        if "embedding" in embedding_result:
                            vector = embedding_result["embedding"]
                        elif "values" in embedding_result:
                            vector = embedding_result["values"]
                
                # If we still couldn't find a vector, print a warning and skip
                if vector is None:
                    print(f"임베딩 결과에서 벡터를 찾을 수 없습니다: {type(embedding_result)}")
                    # Print embedding_result structure to help debugging
                    print(f"embedding_result structure: {dir(embedding_result)}")
                    continue
                
                # Check if vector is callable before trying to convert it
                if callable(vector):
                    print(f"Warning: vector is callable for chunk {chunk_id}")
                    continue
                    
                # 벡터를 리스트로 변환 (안전한 방법)
                vector_list = None
                
                if isinstance(vector, (list, tuple)):
                    vector_list = list(vector)
                elif hasattr(vector, "tolist") and callable(getattr(vector, "tolist")):
                    vector_list = vector.tolist()
                elif isinstance(vector, np.ndarray):
                    vector_list = vector.tolist()
                else:
                    # Print vector type for debugging
                    print(f"Vector type: {type(vector)}")
                    # Only try to convert to list if it's iterable
                    try:
                        if hasattr(vector, "__iter__"):
                            vector_list = list(vector)
                        else:
                            print(f"벡터가 반복 가능한 객체가 아닙니다: {type(vector)}")
                            continue
                    except Exception as e:
                        print(f"벡터를 리스트로 변환할 수 없습니다: {e}")
                        continue
                
                if vector_list is None:
                    continue
                    
                embeddings[chunk_id] = {
                    "embedding": vector_list,
                    "metadata": chunk["metadata"]
                }
            except Exception as e:
                print(f"청크 {chunk_id} 임베딩 생성 중 오류 발생: {e}")
        
        # 임베딩 저장
        embeddings_path = os.path.join(embeddings_dir, "embeddings.json")
        with open(embeddings_path, "w", encoding="utf-8") as f:
            json.dump(embeddings, f)
        
        print(f"임베딩이 저장되었습니다: {embeddings_path}")
        return embeddings

def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description="단순 텍스트 청킹 및 임베딩")
    parser.add_argument("pdf_path", help="PDF 파일 경로")
    parser.add_argument("--output-dir", default="output", help="출력 디렉토리")
    parser.add_argument("--chunk-size", type=int, default=500, help="최대 청크 크기")
    parser.add_argument("--api-key", help="Google API 키 (임베딩 생성용)")
    parser.add_argument("--skip-embeddings", action="store_true", help="임베딩 생성 건너뛰기")
    
    args = parser.parse_args()
    
    # 청크 생성
    chunker = SimpleTextChunker(args.pdf_path, args.output_dir, args.chunk_size)
    chunks = chunker.process()
    
    # 임베딩 생성 (API 키가 제공된 경우)
    if chunks and args.api_key and not args.skip_embeddings:
        embedder = SimpleEmbedder(args.api_key)
        embedder.create_embeddings(chunks, args.output_dir)
    elif not chunks:
        print("생성된 청크가 없어 임베딩을 생성하지 않습니다.")
    elif not args.api_key:
        print("API 키가 제공되지 않아 임베딩을 생성하지 않습니다.")
    
    print("처리가 완료되었습니다!")


if __name__ == "__main__":
    main()