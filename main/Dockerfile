FROM python:3.11-slim

WORKDIR /app

RUN pip install poetry==2.1.2

# Poetry 가상환경 생성하지 않도록 설정
RUN poetry config virtualenvs.create false

COPY pyproject.toml poetry.lock* /app/

RUN poetry install --no-interaction --no-ansi

COPY . /app/

CMD ["python", "main.py"]