# API 文档

本文是接口的**人读版本**；机器可读版本由 FastAPI 自动生成，两者不会有分歧——
所有 Schema 都来自 `src/schemas/` 里的 Pydantic 模型，没有手写的接口定义。

| 入口 | 地址（本地默认） |
| --- | --- |
| Swagger UI（可交互） | <http://localhost:8000/docs> |
| ReDoc | <http://localhost:8000/redoc> |
| OpenAPI JSON | <http://localhost:8000/openapi.json> |

统一约定：

- 所有业务接口在 `/api` 前缀下；
- 请求与响应都是 UTF-8 JSON（流式接口除外）；
- 响应头带 `X-Request-ID`（调用方也可以自己传，用于链路追踪）；
- 错误响应结构统一为 `{"code", "message", "details"}`。

---

## GET /api/health

**用途**：探活。给部署探针与本地自检用，**不触碰任何业务组件**（不连 Chroma、不加载模型）。
这也是它能在"没配 Key、没建库"的环境里照常返回 200 的原因。

**请求**：无参数。

**响应** `200`：

```json
{ "status": "ok" }
```

**curl**

```bash
curl http://localhost:8000/api/health
```

---

## POST /api/chat

**用途**：完整问答链路（意图分类 → FAQ / 检索 → 相关性校验 → 生成），一次性返回。

**请求体**（`ChatRequest`）

| 字段 | 类型 | 必填 | 默认 | 说明 |
| --- | --- | --- | --- | --- |
| `question` | string | 是 | — | 用户问题，1~1000 字符；纯空白会被拒绝（422） |
| `use_faq` | boolean | 否 | `true` | 是否允许走 FAQ 快路径；`false` 时即使意图是 `faq` 也走 RAG（用于对比实验） |
| `top_k` | integer | 否 | `null` → 用服务端 `TOP_K`（默认 5） | 本次检索条数，1~50。**留空表示用服务端配置**，这样运维改 `TOP_K` 才真的生效 |

**响应体**（`ChatResponse`）

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `answer` | string | 回答文本；可能是生成的答案、拒答话术或引导话术 |
| `sources` | array | 来源列表，见下表；拒答与闲聊时为空，FAQ 命中时是 FAQ 条目 |
| `route` | string | 命中的意图：`faq` / `course_query` / `process_query` / `chitchat` |
| `cached` | boolean | `true` 表示由 FAQ 直接命中，**没有**调用 Embedding / Chroma / LLM |
| `latency_ms` | number | 端到端耗时（毫秒），`perf_counter` 测量 |

**`sources[]` 字段**（`Source`）

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `course_id` | string \| null | 课程编号，如 `CS201` |
| `name` | string \| null | 课程名称 |
| `type` | string \| null | 来源类型：`course`（结构化目录）/ `syllabus`（教学大纲）/ `faq` |
| `source` | string \| null | 来源文件名，如 `course_syllabus.md`；FAQ 命中时为 `faq.json` |
| `section` | string \| null | 章节标题；FAQ 命中时为条目 id |
| `score` | number \| null | 该来源的 relevance_score（`[0,1]`，越大越相关）；FAQ 命中时为 `null` |

**响应示例**

```json
{
  "answer": "根据《数据结构与算法》课程资料（CS201），该课程的先修课程是程序设计基础（CS101）。",
  "sources": [
    {
      "course_id": "CS201",
      "name": "数据结构与算法",
      "type": "syllabus",
      "source": "course_syllabus.md",
      "section": "课程基本信息",
      "score": 0.7836
    }
  ],
  "route": "course_query",
  "cached": false,
  "latency_ms": 1873.4
}
```

**拒答响应**（HTTP 仍是 200 —— 这是业务分支，不是错误）

```json
{
  "answer": "知识库中暂未找到与该问题相关的课程资料，无法回答。可以换个说法，或确认一下课程编号（例如 CS201）。",
  "sources": [],
  "route": "course_query",
  "cached": false,
  "latency_ms": 12.9
}
```

**curl**

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "数据结构的先修课程是什么？", "use_faq": true, "top_k": 5}'
```

> Windows 的 Git Bash 里直接写中文会有编码问题（body 变成非法 UTF-8，服务端返回 400）。
> 把 JSON 写进文件再用 `--data-binary @payload.json` 即可。

---

## POST /api/chat/stream

**用途**：与 `/api/chat` 完全相同的链路，答案以 **SSE** 逐片段下发。

**请求体**：同 `ChatRequest`。

**响应**：`200`，`Content-Type: text/event-stream`。

| 事件 | 何时发 | `data` 载荷 |
| --- | --- | --- |
| `meta` | 最先，固定一条 | `{"route": ..., "cached": ..., "sources": [...]}` |
| `delta` | 每段文本一条 | `{"text": "..."}`，按顺序拼接即完整回答 |
| `done` | 正常结束 | `{"latency_ms": 1234.5}` |
| `error` | 流中途失败 | `{"code": "GENERATION_ERROR", "message": "..."}` |

闲聊、FAQ 命中、拒答这三条分支不调用 LLM，会把整段答案放在**一个** `delta` 里——
客户端不需要为它们写第二套逻辑。

**准备阶段与流式阶段的分界**：分类、FAQ、检索、相关性都在响应头发出**之前**完成，
所以它们的错误仍然返回正常的 HTTP 状态码 + JSON 错误体；一旦开始写 SSE 帧，就只能发
`error` 事件了（状态码已经发出，改不了）。

**示例**

```text
event: meta
data: {"route": "faq", "cached": true, "sources": [{"course_id": "CS101", ...}]}

event: delta
data: {"text": "CS101《程序设计基础》为 4 学分，授课教师李明，开课学期 2026-2027-1。"}

event: done
data: {"latency_ms": 1.775}
```

**curl**

```bash
curl -N -X POST http://localhost:8000/api/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"question": "CS101几学分"}'
```

---

## POST /api/faq

**用途**：动态新增一条 FAQ。新条目写进**业务链路正在使用的那一份缓存**（下一个请求
立刻命中），同时原子落盘到 `FAQ_PATH`（重启后仍在）。

**请求头**

| 头 | 何时需要 | 说明 |
| --- | --- | --- |
| `X-Admin-Token` | 服务端配置了 `FAQ_ADMIN_TOKEN` 时**必需** | 值必须与服务端令牌一致，否则 401。比较用 `secrets.compare_digest`（定长比较，避免时序侧信道） |

**请求体**（`FAQCreateRequest`）

| 字段 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| `patterns` | string[] | 是 | 唤起这条答案的问法，1~20 条，每条非空且不超过 200 字符；归一化后重复的会被去掉 |
| `answer` | string | 是 | 命中后**原样返回**的答案（不经过 LLM 改写） |
| `course_id` | string \| null | 否 | 关联课程编号 |

**响应** `201 Created`（`FAQCreateResponse`）

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `id` | string | 自动分配的 id，形如 `faq_008` |
| `patterns` | string[] | 实际写入的问法（已去重、去空白） |
| `answer` | string | 实际写入的答案 |
| `course_id` | string \| null | 关联课程编号 |
| `saved` | boolean | 是否已落盘 |

**curl**

```bash
curl -X POST http://localhost:8000/api/faq \
  -H "Content-Type: application/json" \
  -d '{"patterns": ["MA101多少学分", "高等数学几学分"], "answer": "MA101《高等数学（上）》为 5 学分。", "course_id": "MA101"}'
```

> 新增的问法要能被意图分类判成 `faq` 才会走快路径。常见的"学分 / 考核方式 / 教材 /
> 上课时间"类问法命中规则表；生僻问法会先落到 LLM 分类（或默认的 `course_query`），
> 那就会走 RAG 而绕过 FAQ 库。

---

## 错误码与状态码

响应体统一为：

```json
{ "code": "INDEX_NOT_READY", "message": "向量库为空，无法检索。请先运行 scripts/ingest.py 建库。", "details": { "collection": "campus_courses" } }
```

| 状态码 | `code` | 含义 | 该怎么办 | 重试有用吗 |
| --- | --- | --- | --- | --- |
| 422 | `VALIDATION_ERROR` | 请求参数不合法 | 看 `details.errors` 改请求 | 改完再说 |
| 401 | `UNAUTHORIZED` | `POST /api/faq` 缺少或不匹配 `X-Admin-Token`（仅在服务端配置了 `FAQ_ADMIN_TOKEN` 时出现） | 带上正确的请求头 | ✗ |
| 500 | `CONFIGURATION_ERROR` | 没配 Key、鉴权失败、模型名不存在 | 改配置 | ✗ |
| 500 | `FAQ_DATA_ERROR` | FAQ 文件损坏 / 结构不对 | 修数据 | ✗ |
| 500 | `LLM_FORMAT_ERROR` | 上游响应的**结构**解析不了（不是空回答，是结构不对） | 查上游/SDK 版本 | ✗（结构错重试无意义，代码里显式不重试） |
| 500 | `ROUTING_ERROR` | 路由层配置问题（如阈值越界） | 改配置 | ✗ |
| 500 | `DOCUMENT_LOAD_ERROR` | 语料解析失败（建库脚本用） | 修语料 | ✗ |
| 502 | `GENERATION_ERROR` | 上游（DeepSeek）调用失败 | 稍后重试 | ✓ |
| 504 | `LLM_TIMEOUT` | 上游超时且重试用尽 | 稍后重试 | ✓ |
| 503 | `INDEX_NOT_READY` | 向量库为空（没建库） | 跑 `scripts/ingest.py` | 建完再说 |
| 503 | `RETRIEVAL_ERROR` | 向量库不可用 / 距离空间不一致 | 查持久化目录 | ✗ |
| 503 | `SERVICE_NOT_READY` | 应用没走 lifespan 就被调用（服务未就绪） | 检查启动方式 | ✗ |

SSE 中途失败时，`error` 事件的 `code` 可能是上表里的任意一个；**未预期的内部错误**
用 `INTERNAL_ERROR` 表示（message 不含内部细节，堆栈只进日志）。

映射规则在 `src/main.py`，按"**调用方该怎么办**"分类，而不是按"哪个模块抛的"：

- 4xx 只有请求问题（**拒答与闲聊都是 200**）；
- 502 / 504 是**上游**的问题，重试有意义；
- 503 是**自己**暂时不可用（索引没建好），修好即可；
- 500 是**配置或数据**错了，重试没有意义。

未在 `AppError` 体系内的异常（真正的代码 bug）走 FastAPI 默认处理：**500 且不回显内部细节**，
堆栈只进日志。
