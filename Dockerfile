FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

COPY pyproject.toml ./
COPY src ./src
RUN pip install ".[ui]"

COPY configs ./configs
COPY app ./app
COPY examples ./examples

EXPOSE 8000 8501
# Par défaut : l'API. docker-compose lance aussi l'interface Streamlit.
CMD ["uvicorn", "ragassist.api:app", "--host", "0.0.0.0", "--port", "8000"]
