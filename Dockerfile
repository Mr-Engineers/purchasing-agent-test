FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY purchasing_agent ./purchasing_agent

# AGENT_MODE=direct: backend-2's load balancer uses a certificate from our own CA (no public
# domain). Trust it on top of the public CAs (certifi), which Bedrock still needs.
# httpx (apps and the OpenAI client) reads SSL_CERT_FILE.
COPY certs/backend-2-ca.crt /tmp/backend-2-ca.crt
RUN cat "$(python -c 'import certifi; print(certifi.where())')" /tmp/backend-2-ca.crt > /etc/ssl/ca-bundle.pem \
    && rm /tmp/backend-2-ca.crt
ENV SSL_CERT_FILE=/etc/ssl/ca-bundle.pem

RUN useradd --create-home --uid 1000 appuser
USER appuser

CMD ["python", "-m", "purchasing_agent"]
