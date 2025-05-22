# EMA - 수학 검색 에이전트

LangChain과 Google Gemini를 사용한 수학 개념 검색 AI 에이전트

## 환경 설정

### 1. API 키 설정

`.env` 파일을 생성하고 다음 내용 추가:

```env
GOOGLE_API_KEY=구글_API_키
TAVILY_API_KEY=타빌리_API_키
```

### 2. 도커 실행

```bash
# 빌드(처음)
docker-compose --build
# 실행(이후, ema 위치는 이름 지은거, ema 말고도 아무거나 가능)
docker-compose run ema
# 실행(실행 후 컨테이너 자동 삭제)
docker-compose run --rm -it ema
```

## 패키지 추가 방법

### Poetry로 패키지 추가

```bash
# 새 패키지 추가
poetry add 패키지명

# 개발용 패키지 추가
poetry add --group dev 패키지명

# 특정 버전 추가
poetry add "패키지명==1.2.3"
```

### 도커 환경에서 패키지 추가

```bash
# 1. 패키지 추가
poetry add 새로운패키지

# 2. 도커 재빌드
docker-compose down
docker-compose up --build
```