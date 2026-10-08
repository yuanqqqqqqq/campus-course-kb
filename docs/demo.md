# 演示流程

从零到"能问答"的完整演示，命令可以直接复制执行。每一步都标了**是否需要 DeepSeek Key**——
没有 Key 也能演示 FAQ 快路径与拒答机制，这两条恰好是最能说明设计取舍的部分。

```text
准备（1~3）→ 不需要 Key     演示问答（4~8）→ 其中 5、6 需要 Key
```

---

## 0. 前置：安装依赖与配置

```bash
# 创建虚拟环境并安装依赖
python -m venv .venv
source .venv/Scripts/activate        # Windows Git Bash
# source .venv/bin/activate          # macOS / Linux
pip install -r requirements.txt

# 生成配置文件（.env 已被 .gitignore 忽略）
cp .env.example .env
```

`.env` 里至少要确认两项：

```ini
# 1) 本地已下载的 Embedding 模型目录（否则会去联网下载，国内通常失败）
EMBEDDING_MODEL=./models/bge-small-zh-v1.5

# 2) DeepSeek Key（不填也能起服务，只是涉及生成的请求会返回 500 CONFIGURATION_ERROR）
DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxxxxx
```

---

## 1. 导入课程数据（不需要 Key）

语料放在 `data/raw/`：`courses.json`（结构化课程目录）与 `course_syllabus.md`（教学大纲）。
先试运行，确认语料能解析：

```bash
python scripts/ingest.py --dry-run
```

```text
语料解析结果：17 个切片（course=7、syllabus=10），涉及 7 门课程
  课程：CS101、CS201、CS202、CS301、CS302、CS401、MA101

试运行完成：共解析出 17 个文档切片，未写入向量库。
```

## 2. 创建向量索引（不需要 Key）

```bash
python scripts/ingest.py
```

```text
建库完成：17 个文档切片写入集合 campus_courses（.../data/chroma），耗时 9.8s。
```

首次运行会加载 Embedding 模型（约 5 秒），这一步**不调用 LLM**，所以不花 API 费用。
建库是**全量重建**（先删同名集合再写入）：删除的课程不会残留。

## 3. 启动服务（不需要 Key）

```bash
uvicorn src.main:app --reload
```

```text
INFO | 应用启动 | name=Campus Course Knowledge Base version=0.1.0 log_level=INFO
INFO | 关键配置 | llm_model=deepseek-chat base_url=https://api.deepseek.com ...
INFO | FAQ 加载完成 | entries=7 patterns=20 threshold=0.85
INFO | 问答链路就绪 | faq_entries=7 collection=campus_courses top_k=5 relevance_threshold=0.50
INFO | Uvicorn running on http://127.0.0.1:8000
```

## 4. 健康检查（不需要 Key）

```bash
curl http://localhost:8000/api/health
# {"status":"ok"}
```

## 5. 查询课程（**需要 Key**）

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "数据结构的先修课程是什么？", "use_faq": true, "top_k": 5}'
```

```json
{
  "answer": "根据《数据结构与算法》课程资料（CS201），该课程的先修课程是程序设计基础（CS101）。",
  "sources": [
    { "course_id": "CS201", "name": "数据结构与算法", "type": "syllabus",
      "source": "course_syllabus.md", "section": "课程基本信息", "score": 0.7836 }
  ],
  "route": "course_query",
  "cached": false,
  "latency_ms": 1873.4
}
```

日志里可以顺着一个 `request_id` 看到整条链路：

```text
INFO | request_id=a1b2c3d4 | src.routing.classifier | 意图分类：规则短路，不调用 LLM | intent=faq confidence=0.95
INFO | request_id=a1b2c3d4 | src.services.pipeline | 问答路由 | route=faq 未命中 FAQ，继续走 RAG
INFO | request_id=a1b2c3d4 | src.services.pipeline | 问答路由 | route=faq 进入生成 | retrieval_count=5 best_score=0.7836
INFO | request_id=a1b2c3d4 | src.services.pipeline | 问答完成 | route=faq cached=false top_k=5 retrieval_count=5 relevance=relevant best_score=0.7836 llm_called=true llm_latency_ms=1621.3 latency_ms=1873.4 tokens=512 question=<已脱敏 len=14>
INFO | request_id=a1b2c3d4 | src.main | 请求完成 | method=POST path=/api/chat status=200 latency_ms=1876.0
```

> 注意 `question=<已脱敏 len=14>`：默认不把用户问题写进日志。需要排查具体问题时设
> `LOG_REQUEST_CONTENT=true`。

## 6. 查询 FAQ（不需要 Key）

FAQ 快路径命中时不碰 Embedding / Chroma / LLM：

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "CS101几学分"}'
```

```json
{
  "answer": "CS101《程序设计基础》为 4 学分，授课教师李明，开课学期 2026-2027-1。",
  "sources": [{ "course_id": "CS101", "name": null, "type": "faq",
                "source": "faq.json", "section": "faq_001", "score": null }],
  "route": "faq",
  "cached": true,
  "latency_ms": 0.4
}
```

**0.4 毫秒（链路内；HTTP 层约 2.2 毫秒）vs 上一步的 1873 毫秒** —— 这就是 FAQ Cache 的
意义。同一句话关掉快路径（`"use_faq": false`）之后要走完整检索，实测延迟相差约 5 倍。

> 如果服务端配置了 `FAQ_ADMIN_TOKEN`，下面的写入请求需要加上
> `-H "X-Admin-Token: <令牌>"`，否则返回 401。

新增一条 FAQ，然后立刻再问：

```bash
curl -X POST http://localhost:8000/api/faq \
  -H "Content-Type: application/json" \
  -d '{"patterns": ["MA101多少学分"], "answer": "MA101《高等数学（上）》为 5 学分。", "course_id": "MA101"}'
# 201 {"id":"faq_008","patterns":["MA101多少学分"],"answer":"...","course_id":"MA101","saved":true}

curl -X POST http://localhost:8000/api/chat -H "Content-Type: application/json" \
  -d '{"question": "MA101多少学分"}'
# cached=true，来源指向 faq_008
```

## 7. 查询教务流程（**需要 Key**）

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "缓考怎么申请？"}'
```

当前语料里**没有**教务流程文档，所以正确答案是拒答——系统应当先说"知识库中暂未找到"，
而不是用常识编一套流程出来。日志会显示：

```text
INFO | 意图分类：退化为规则候选 | intent=process_query rule=process_admin
INFO | 问答路由 | route=process_query 证据不足，拒答 | 最高向量分数 0.3944 低于阈值 0.5
INFO | 问答完成 | route=process_query cached=false retrieval_count=5 relevance=insufficient llm_called=false
```

`llm_called=false`：拒答**没有**调用生成模型。要让它能回答，需要把教务文档放进
`data/raw/`，然后重新建库。

## 8. 知识库不存在的问题（不需要 Key）

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "食堂几点开门？"}'
```

```json
{
  "answer": "知识库中暂未找到与该问题相关的课程资料，无法回答。可以换个说法，或确认一下课程编号（例如 CS201）。",
  "sources": [],
  "route": "course_query",
  "cached": false,
  "latency_ms": 12.2
}
```

这条演示的是 **RAG 拒答**：证据不足时宁可明说没找到，也不给模型编造的机会。

## 9. 流式输出（不需要 Key）

```bash
curl -N -X POST http://localhost:8000/api/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"question": "CS101几学分"}'
```

```text
event: meta
data: {"route": "faq", "cached": true, "sources": [...]}

event: delta
data: {"text": "CS101《程序设计基础》为 4 学分，授课教师李明，开课学期 2026-2027-1。"}

event: done
data: {"latency_ms": 1.775}
```

## 10. 查看来源与 Swagger

```bash
curl -X POST http://localhost:8000/api/chat -H "Content-Type: application/json" \
  -d '{"question": "数据结构的先修课程是什么？"}' | python -m json.tool
```

响应里的 `sources` 就是"这个回答是从哪来的"：课程编号、课程名、来源文件、章节、相关性分数。

打开 <http://localhost:8000/docs> 可以在页面上直接调这四个接口（Swagger UI 由 FastAPI
按 Pydantic 模型自动生成，与本文档不会有分歧）；<http://localhost:8000/redoc> 是更适合
阅读的版本。

---

## 附：一次完整的"改数据 → 看效果"循环

```bash
# 1. 往 FAQ 里加一条（也可以直接编辑 data/faq.json）
curl -X POST http://localhost:8000/api/faq -H "Content-Type: application/json" \
  -d '{"patterns": ["CS302谁教"], "answer": "CS302《操作系统》的授课教师是赵磊。", "course_id": "CS302"}'

# 2. 立刻问一次，确认命中（cached=true）
curl -X POST http://localhost:8000/api/chat -H "Content-Type: application/json" \
  -d '{"question": "CS302谁教"}'

# 3. 跑一遍评测，确认改动没有让别的指标变差
python scripts/evaluate.py

# 4. 跑测试，确认没有破坏既有行为
pytest -q
```

## 附：Docker 演示（已在 CI 实测，未在本机执行）

> ✅ 2026-10-08 在 GitHub Actions 上实测：镜像构建成功，容器启动后 `/api/health`
> 返回 `{"status":"ok"}`，`docker compose config` 通过。
> ⚠️ 开发这个项目的机器上没有 Docker，所以下面这几条命令**没有在本地执行过**——
> 实测环境是 GitHub 的 runner，本地首次运行请以自己的环境为准。

```bash
cp .env.example .env       # 填好 Key 与 EMBEDDING_MODEL
docker compose up --build
# 另开终端
curl http://localhost:8000/api/health
```

容器内 `EMBEDDING_MODEL` 被 compose 覆盖为 `/app/models/bge-small-zh-v1.5`，
对应宿主的 `./models`（只读挂载）；向量库则是 `./data:/app/data`。
