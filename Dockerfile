# syntax=docker/dockerfile:1.7

# ---------------------------------------------------------------------------
# mind-nerve daemon image
#
# Stage 1 (builder) installs the wheel into a throwaway venv so we keep
# build tools out of the final image. Stage 2 copies just the venv plus
# the package sources into a slim runtime.
#
# Healthcheck pings the daemon's UNIX socket through mind-nerve-routed-ensure;
# the daemon's own bind is what makes the socket responsive, so this is a
# real readiness signal, not a port-listen tautology.
# ---------------------------------------------------------------------------

FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

COPY pyproject.toml README.md ./
COPY python ./python

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

RUN pip install --upgrade pip \
    && pip install ".[mcp]"


FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PATH="/opt/venv/bin:${PATH}" \
    MIND_NERVE_RUNTIME_DIR=/var/lib/mind-nerve/runtime \
    MIND_NERVE_SOCKET=/var/run/mind-nerve/mind-nerve.sock \
    XDG_CACHE_HOME=/var/cache/mind-nerve \
    HF_HOME=/var/cache/mind-nerve/huggingface

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system --gid 1000 mindnerve \
    && useradd --system --uid 1000 --gid mindnerve --home-dir /home/mindnerve \
        --create-home --shell /usr/sbin/nologin mindnerve \
    && mkdir -p /var/lib/mind-nerve/runtime /var/run/mind-nerve /var/cache/mind-nerve \
    && chown -R mindnerve:mindnerve /var/lib/mind-nerve /var/run/mind-nerve /var/cache/mind-nerve

COPY --from=builder /opt/venv /opt/venv

USER mindnerve
WORKDIR /home/mindnerve

EXPOSE 8765

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD mind-nerve-routed-ensure && \
        mind-nerve --version || exit 1

ENTRYPOINT ["mind-nerve-routed"]
