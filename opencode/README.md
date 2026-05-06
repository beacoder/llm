# AI 编程助手本质上是上下文窗口内的模式匹配与推理引擎
## 固定 prompt 和 tool 只是“发动机”，上下文注入是方向盘，但能决定你能开多稳的是“导航系统”（任务拆解+验证闭环）和“车载电脑”（项目规则沉淀）。

# 下面是一个最佳实践的例子：Redis 限流开发工作流示例

## 📜 Step 1：项目级规则文件（静态上下文）

```markdown
# Project Rules
- Stack: Node 20, Express, TS5, Prisma, Jest, Redis
- Structure: routes → controllers → middleware; tests co-located as `*.test.ts`
- Style: `airbnb` ESLint, strict null checks, `async/await` only, no `any`
- Testing: Jest + Supertest, 90% coverage, mock all external services
- Infra: Dockerized, rate limiting via Redis, 429 on limit exceeded
```

## 🎯 Step 2：项目级规则文件（静态上下文）

```markdown
## Task
为 `POST /api/v1/register` 实现滑动窗口 Redis 限流（5 req/min per IP）。

## Context
- Target Route: `src/routes/auth.routes.ts` (line 12-24)
- Reference Middleware: `src/middleware/validate.ts` (follow same export pattern)
- Redis Client: `src/utils/redis.ts` → exported as `redisClient`
- Error Utility: `src/utils/errors.ts` → `AppError(code, message, details?)`

## Constraints
- DO NOT modify existing middleware or route logic
- Use `INCR` + `EXPIRE` atomic pipeline
- Return `429` with `{ error: "Rate limit exceeded", retryAfter: <seconds> }`
- Keep implementation under 60 lines
- Tests must use `redis-mock` (see `jest.setup.js`)

## Expected Output Format
1. New file: `src/middleware/rateLimiter.ts`
2. Diff for `auth.routes.ts`
3. Test file: `src/middleware/rateLimiter.test.ts` (cover pass, block, reset)
```

## 📦 Step 3：AI 预期输出示例（生成结果）

```markdown
......
```

## 🔍 Step 4：迭代验证与修复 Prompt（失败回喂）

```markdown
## Test Output
FAIL src/middleware/rateLimiter.test.ts
● Redis client not initialized in test environment

## Context
- Jest uses `redis-mock` via `jest.setup.js`
- Expected: Mock `redisClient` in test, skip real connection
- Implementation should remain unchanged

## Request
Fix test file only. Keep `rateLimiter.ts` intact.
```

## 🔍 Step 5：上下文资产沉淀（规则更新）

```markdown
## Middleware Patterns
- Rate limiting: always use `src/middleware/rateLimiter.ts`
- Validation: use Zod + `validate.ts` pattern
- Error handling: wrap in `try/catch`, throw `AppError`
```
