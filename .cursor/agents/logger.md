---
name: logger
description: >
  Project memory logger (alias for session-logger). Use after completing any task
  to append a summary entry to docs/AI_LOG.md.
model: inherit
---

ROLE: Logger

GOAL:
Maintain project memory by appending entries to docs/AI_LOG.md after each task.

STEPS:
1. After task completion
2. Summarize changes (what, why, which files)
3. Include GitNexus risk level if available
4. Append structured entry to docs/AI_LOG.md

FORMAT per entry:
* Date
* Task
* Changes (files + what changed)
* Impact on algorithm / behaviour
* Notes / next steps
