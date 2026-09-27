FROM python:3.12-slim

# Copie des exécutables uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Installation de libgomp1 (requis pour XGBoost / OpenMP)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copie des fichiers de dépendances
COPY pyproject.toml uv.lock ./

# Installation déterministe via le lockfile
RUN uv sync --frozen --no-cache

# Copie du reste de l'application
COPY . .

# Lancement de Streamlit sur le port attribué par l'environnement
CMD ["sh", "-c", ".venv/bin/streamlit run app.py --server.port=${PORT:-8501} --server.address=0.0.0.0"]