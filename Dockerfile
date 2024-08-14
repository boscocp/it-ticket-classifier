FROM python:3.12

ENV PYTHONPATH=/

WORKDIR /app

COPY src/machine_learning_training/helpers.py .
COPY src/classification_service/main.py .
COPY src/classification_service/ticket_model.py .
COPY src/classification_service/__init__.py .
COPY src/classification_service/requirements.txt .

COPY src/machine_learning_training/model.pkl ./machine_learning_training/model.pkl
COPY src/machine_learning_training/__init__.py ./machine_learning_training/__init__.py

RUN pip install -r requirements.txt

USER nobody

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]