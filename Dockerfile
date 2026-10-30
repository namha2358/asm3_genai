FROM python:3.12-slim-bookworm

# Install curl (needed to fetch uv) and certificates, then delete the apt cache to shrink the image
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*
    
ADD https://astral.sh/uv/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh
ENV PATH="/root/.local/bin/:$PATH"

WORKDIR /code

COPY pyproject.toml uv.lock /code/
RUN uv sync --frozen

COPY ./app /code/app
COPY ./helper_lib /code/helper_lib
COPY main.py /code/

CMD ["uv", "run", "fastapi", "run", "main.py", "--port", "80"]