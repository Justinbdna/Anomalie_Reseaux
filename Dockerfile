# Image de départ : Python 3.12 en version "slim" (légère).
FROM python:3.12-slim

# Réglages Python : pas de fichiers .pyc, affichage des logs immédiat, pas de cache pip (image plus petite).
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Dossier de travail dans le conteneur ; on installe d'abord les dépendances.
# (Docker garde cette étape en cache tant que requirements.txt ne change pas : builds suivants plus rapides.)
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

# Le CSV UCI doit être téléchargé localement avant la construction (voir README).
# On copie les données et les scripts, puis on entraîne le modèle PENDANT la construction de l'image
# (ça crée artifacts/model.joblib à l'intérieur de l'image).
COPY data/log2.csv data/log2.csv
COPY train.py features.py ./
RUN python train.py

# On copie l'API, l'interface et les tests.
COPY api.py app.py ./
COPY tests/ tests/
# Sécurité : on crée un utilisateur normal et on n'exécute pas l'application en administrateur (root).
RUN useradd --create-home appuser
USER appuser

# Ports utilisés : 8000 pour l'API, 8501 pour Streamlit.
# Commande par défaut : lancer l'API (docker-compose.yml la remplace pour le service "app").
EXPOSE 8000 8501
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
