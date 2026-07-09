FROM python:3.11-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1
COPY pyproject.toml requirements.txt README.md ./
COPY src ./src
RUN pip install --no-cache-dir -e .
ENTRYPOINT ["atlasforecast"]
CMD ["plan"]
