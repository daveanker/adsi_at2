FROM tiangolo/uvicorn-gunicorn-fastapi:python3.7

COPY requirements.txt .

RUN pip3 install -r requirements.txt

COPY ./app /app

COPY ./models /models

COPY ./src /src

CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "-c", "/gunicorn_conf.py", "main:app"]

WORKDIR /