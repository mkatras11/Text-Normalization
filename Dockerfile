
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt /app
COPY ./src /app
COPY get_model.py /app

RUN pip install --upgrade pip wheel setuptools
RUN pip install --no-cache-dir -r requirements.txt

RUN python get_model.py

RUN pip install ./model.tar.gz

EXPOSE 8080

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]