#!/bin/bash

APP_PORT=${PORT:-8000}

CPU_THREADS=$(nproc)

WORKERS=$((CPU_THREADS*2+1))

gunicorn -k uvicorn.workers.UvicornWorker --workers 5 --worker-connections 1000 --worker-class gevent server:app --bind "0.0.0.0:${APP_PORT}" --timeout 10000
