# syntax=docker/dockerfile:1
#
# 校园课程知识库 RAG 问答系统 —— 运行镜像
#
# 构建：
#   docker build -t campus-course-kb .
# 运行（推荐用 compose，见 docker-compose.yml）：
#   docker run --rm -p 8000:8000 --env-file .env \
#     -v "$PWD/data:/app/data" -v "$PWD/models:/app/models:ro" campus-course-kb
#
# 三条刻意的取舍：
#
# 1. **不把模型权重和数据打进镜像**。requirements 里的 torch 已经有几百 MB，
#    再把 95MB 的 BGE 模型与向量库塞进去，镜像会变得又大又难更新；模型与数据都
#    用 volume 挂载（`models/` 只读挂进去即可）。
# 2. **不复制 .env**。密钥通过 `--env-file` / compose 的 `env_file` 在运行时注入，
#    镜像里永远不含密钥，镜像推送出去也不会泄露。
# 3. **以非 root 用户运行**。容器逃逸的代价太大，没有理由让应用以 root 跑。

FROM python:3.11-slim AS runtime

# PYTHONDONTWRITEBYTECODE：容器里没必要留 .pyc
# PYTHONUNBUFFERED：日志实时输出（否则 `docker logs` 会攒着不显示）
# PIP_NO_CACHE_DIR：不把 pip 缓存留在镜像层里
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app \
    APP_ENV=production \
    LOG_LEVEL=INFO \
    CHROMA_PERSIST_DIR=/app/data/chroma \
    FAQ_PATH=/app/data/faq.json \
    EMBEDDING_MODEL=/app/models/bge-small-zh-v1.5

WORKDIR /app

# ---- 依赖层：单独一层，改代码不会触发重装依赖 ----
# 先只复制 requirements.txt，让这一层能被 Docker 的层缓存复用；
# 每次改业务代码都重装一遍 torch 是不可接受的。
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# ---- 代码层 ----
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY tests/ ./tests/
COPY pyproject.toml README.md ./
COPY data/faq.json ./data/faq.json
COPY data/raw/ ./data/raw/

# ---- 运行期目录与非 root 用户 ----
# data/chroma 是挂载点：这里先建好目录并交给 appuser，容器第一次启动时
# （还没跑过 scripts/ingest.py）才不会因为权限问题起不来。
RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /app/data/chroma /app/data/processed /app/models \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

# 健康检查用 python 自带的 urllib，不额外装 curl（少一个包就少一处漏洞面）。
# start-period 给 40 秒：首次请求要加载 Embedding 模型，太早判失败会把健康容器标成不健康。
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import sys, urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=3).status == 200 else 1)"

# 单 worker：每个 worker 都会各自加载一份 Embedding 模型（几百 MB 内存），
# 需要更高并发时应先在容器外扩副本，而不是在这里加 workers。
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
