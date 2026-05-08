---
name: refactor
arguments:
  - name: module
  - name: goal
---

Refactor the following module:
{{module}}

Goal: 
{{goal}}

Constraints:
- **Zero Breaking Changes:** Preserve the existing public API and signatures exactly.
- **Minimal Invasive Surgery:** Avoid total rewrites; focus on internal logic improvement, readability, or performance as per the goal.
- **Verification:** Include comprehensive unit tests (using the project's standard framework) that cover both the original behavior and the new improvements.

Instructions:
1. Briefly outline the refactoring strategy.
2. Provide the updated code for {{module}}.
3. Provide the test suite.
