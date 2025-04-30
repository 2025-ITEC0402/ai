# EMA (Engineering Mathematics Assistant)

공학수학 학습을 위한 슈퍼바이저 기반 AI 에이전트 시스템

## 프로젝트 개요

EMA는 공학수학 학습을 지원하기 위한 다중 에이전트 시스템으로, 사용자의 학습 이력을 분석하고 맞춤형 문제와 설명을 제공하는 지능형 학습 도우미입니다.

## 주요 기능

- **이론 설명**: 공학수학 특정 챕터에 대한 이론 자료 제공
- **학습 이력 분석**: 사용자의 학습 이력을 통해 학습 취약점 분석
- **문제 생성**: 주어진 문맥에 맞는 정확한 수학 문제 생성
- **문제 풀이**: 주어진 문제에 대한 정확한 답과 풀이 생성
- **외부 검색**: 특정 주제에 대한 외부 정보 검색
- **답변 퀄리티 평가**: 사용자에게 제시할 답변의 질을 자체 평가
- **외부 공유**: 서비스 데이터를 PDF 문서로 변환하여 공유

## 시스템 구조

ema/
├── .env                           # 환경 변수 설정
├── agents/                        # 에이전트 모듈
│   ├── __init__.py
│   ├── external_search.py          # 외부 검색 에이전트
│   ├── external_sharing.py         # 외부 공유 에이전트
│   ├── learning_history.py         # 학습 이력 에이전트
│   ├── problem_generation.py       # 문제 생성 에이전트
│   ├── problem_solving.py           # 문제 풀이 에이전트
│   ├── quality_evaluation.py       # 답변 퀄리티 평가 에이전트
│   ├── task_manager.py              # 태스크 매니저
│   └── theory_explanation.py       # 이론 설명 에이전트
├── data/                           # 데이터 파일
│   ├── topics.json                 # topic json파일(calculus 챕터)
├── graph/                          # 워크플로우 그래프
│   ├── __init__.py
│   └── workflow.py                 # 워크플로우 정의
├── utils/                          # 유틸리티 함수
│   ├── __init__.py
├── .dockerignore                   
├── Dockerfile                      
├── docker-compose.yml              
├── config.py                       # 설정 파일
├── main.py                         # 메인 실행 파일
├── poetry.lock                     
├── pyproject.toml                  # Poetry 설정 파일
├── README.md                       
└── requirements.txt                # 의존성 목록

## 워크플로우

EMA는 다음과 같은 워크플로우를 통해 작동합니다:

1. **문제 생성 워크플로우**:
   - Task Manager가 학습 이력 조회
   - 적절한 문제 난이도와 단원 확인
   - 내부/외부 자료 수집
   - 문제 생성 및 퀄리티 평가

2. **풀이 생성 워크플로우**:
   - 학습 이력 기반 취약 개념 파악
   - 내부/외부 자료 수집
   - 맞춤형 풀이 생성 및 퀄리티 평가

3. **Q&A 워크플로우**:
   - 사용자 요청 해석
   - 필요한 에이전트 호출
   - 답변 생성 및 퀄리티 평가
   - 필요시 외부 공유 형식으로 변환

## 환경 설정

1. `.env` 파일을 생성하고 필요한 API 키를 설정합니다:

```
OPENAI_API_KEY=your_openai_api_key
GOOGLE_API_KEY=your_google_api_key
TAVILY_API_KEY=your_tavily_api_key
```

## 실행 방법

```bash
docker-compose run -it ema
```

## 개발 정보

- Python 버전: 3.11
- 주요 의존성:
  - LangChain: 에이전트 시스템 프레임워크
  - LangGraph: 에이전트 워크플로우 관리