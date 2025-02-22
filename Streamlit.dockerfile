
FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir streamlit==1.42.2

COPY ./src/client.py /app

EXPOSE 8501

CMD ["streamlit", "run", "client.py", "--server.port=8501", "--server.address=0.0.0.0"]