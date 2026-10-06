FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 BROWSER_PATH=/usr/bin/chromium
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 chromium fonts-dejavu-core && rm -rf /var/lib/apt/lists/*
RUN useradd -m appuser
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY --chown=appuser:appuser . .
USER appuser
EXPOSE 8501
# Render and similar hosts inject $PORT
CMD ["sh", "-c", "streamlit run app.py --server.port=${PORT:-8501} --server.address=0.0.0.0 --server.headless=true"]
