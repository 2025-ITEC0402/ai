import os
import re
import time
from pathlib import Path
from dotenv import load_dotenv

import pdfplumber
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain

# 1) .env에서 API 키 로드
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# 2) Gemini LLM 초기화
llm = ChatGoogleGenerativeAI(
    model="models/gemini-1.5-pro",
    temperature=0.0,
    api_key=api_key
)

# 3) Markdown 변환용 프롬프트 템플릿
md_prompt = PromptTemplate.from_template(
    """You receive the following engineering math textbook excerpt, including LaTeX formulas.
Convert it into well-formatted Markdown:
- Use `#` for chapter headings and `##` for section headings.
- Wrap formulas in `$$...$$`.
- Preserve bullet lists.
- Output only Markdown.

# Excerpt:
{excerpt}

# Markdown Output:"""
)

# 4) LLMChain 생성
chain = LLMChain(llm=llm, prompt=md_prompt)

def extract_chapters(pdf_path: str) -> dict:
    """
    PDF의 각 페이지 첫 줄에서 'Chapter X. Title' 패턴을 찾아 챕터별로 텍스트를 분리합니다.
    반환값: {안전한챕터파일명: 챕터텍스트}
    """
    chapters = {}
    current_title = None
    current_lines = []

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            lines = text.splitlines()
            if not lines:
                continue

            # Chapter 헤더 패턴
            header_match = re.match(r"^(?:Chapter|CHAPTER)\s+(\d+)\.?\s*(.+)", lines[0].strip())
            if header_match:
                # 이전 챕터 저장
                if current_title:
                    chapters[current_title] = "\n".join(current_lines).strip()
                # 새 챕터 제목
                ch_num, ch_title = header_match.groups()
                safe_title = re.sub(r"[^\w\-]", "_", ch_title.strip())
                current_title = f"{int(ch_num):02d}_{safe_title}"
                # 본문 초기화 (첫 줄 제외)
                current_lines = lines[1:]
            else:
                if current_title:
                    current_lines.extend(lines)

        # 마지막 챕터 저장
        if current_title:
            chapters[current_title] = "\n".join(current_lines).strip()

    return chapters

def split_text(text: str, max_len: int = 2000) -> list:
    """
    문단 단위로 묶어서 max_len 이하의 청크 리스트 반환
    """
    paragraphs = text.split("\n\n")
    chunks = []
    current = ""
    for p in paragraphs:
        p_block = p.strip() + "\n\n"
        if len(current) + len(p_block) <= max_len:
            current += p_block
        else:
            if current:
                chunks.append(current.strip())
            if len(p_block) <= max_len:
                current = p_block
            else:
                # 너무 긴 문단은 강제로 분할
                for i in range(0, len(p_block), max_len):
                    chunks.append(p_block[i : i + max_len].strip())
                current = ""
    if current:
        chunks.append(current.strip())
    return chunks

def convert_and_save(chapters: dict, output_dir: str = "chapters_md"):
    """
    각 챕터를 Markdown으로 변환해 파일로 저장
    """
    Path(output_dir).mkdir(exist_ok=True)
    for title, content in chapters.items():
        if int(title[:2]) < 9:
            continue
        # 청크 분할
        chunks = split_text(content)
        md_parts = []
        for idx, chunk in enumerate(chunks, 1):
            print(f"[{title}] processing chunk {idx}/{len(chunks)}...")
            try:
                part = chain.run(excerpt=chunk)
            except Exception as e:
                print(f"Error on chunk {idx}: {e}, retrying in 2s...")
                time.sleep(2)
                part = chain.run(excerpt=chunk)
            md_parts.append(part)

        md_full = "\n\n".join(md_parts)
        filename = f"{title}.md"
        filepath = Path(output_dir) / filename
        with open(filepath, "w", encoding="utf-8") as f:
            human_title = title.replace("_", " ", 1)
            f.write(f"# Chapter {human_title}\n\n")
            f.write(md_full)

    print(f"✅ Markdown files saved to '{output_dir}'")

if __name__ == "__main__":
    pdf_path = "James Stewart - Calculus, Early Transcendentals, International Metric Edition-CENGAGE Learning (2016).pdf"
    chapters = extract_chapters(pdf_path)
    convert_and_save(chapters)
