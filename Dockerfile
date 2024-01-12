FROM tiangolo/uvicorn-gunicorn-fastapi:python3.11

WORKDIR /app

COPY ./api/middleware /app/api/middleware
COPY ./api/pipeline /app/api/pipeline
COPY ./api/router /app/api/router
COPY ./api/schemas /app/api/schemas
COPY ./api/custom_response.py /app/api/custom_response.py
COPY ./api/download_model.py /app/api/download_model.py
COPY ./api/requirements.txt /app/api/requirements.txt

RUN if [ -f models ]; then \
        cp models /app/models; \
    else \
        true; \
    fi

COPY ../server.py /app/server.py
COPY static /app/static
COPY templates /app/templates
COPY ../entrypoint.sh /app/entrypoint.sh

RUN apt-get update && \
    apt-get install -y \
    build-essential \
    python3-dev \
    python3-setuptools \
    gcc \
    make \
    wget

WORKDIR /app


RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install --no-cache-dir -r /app/api/requirements.txt


RUN if [ ! -d /app/models ]; then \
        mkdir -p /app/models && \
        wget -O /app/models/replit-v2-codeinstruct-3b.q4_1.bin https://huggingface.co/abacaj/Replit-v2-CodeInstruct-3B-ggml/resolve/main/replit-v2-codeinstruct-3b.q4_1.bin; \
    fi

RUN apt-get remove -y --purge make gcc build-essential \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV CPU_THREADS=$(nproc)

ENV WORKERS=$((CPU_THREADS*2+1))

CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000", "--worker-connections", "1000","--timeout", "10000", "server:app"]
