# AI 编程助手本质上是上下文窗口内的模式匹配与推理引擎

# 下面就是用户需要提供的上下文的例子

## 📜 Step 1：项目级规则文件（静态上下文）

```markdown
# Project Rules
- Stack: Node 20, Express, TS5, Prisma, Jest, Redis
- Structure: routes → controllers → middleware; tests co-located as `*.test.ts`
- Style: `airbnb` ESLint, strict null checks, `async/await` only, no `any`
- Testing: Jest + Supertest, 90% coverage, mock all external services
- Infra: Dockerized, rate limiting via Redis, 429 on limit exceeded
```

## 🎯 Step 2：任务拆解（静态上下文）

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
```
