---
name: code-review
description: Reviews code changes or existing files for bugs, design issues, style, and performance.
---

# Code Review Skill

Use this skill when the user asks for a code review or when you are self-reviewing critical code.

## 1. Review Checklist

### Functionality & Correctness
- [ ] **Logic Errors**: Are there any off-by-one errors, infinite loops, or incorrect conditional logic?
- [ ] **Requirements**: Does the code meet the apparent or stated requirements?
- [ ] **Edge Cases**: Are empty inputs, null values, or large datasets handled gracefully?
- [ ] **Error Handling**: Are exceptions caught and logged properly? Are error messages clear?

### Code Quality & Style
- [ ] **Readability**: Is the code easy to understand? Are variable/function names descriptive?
- [ ] **Modularity**: Are functions too long or doing too much? Should code be refactored into helpers?
- [ ] **DRY (Don't Repeat Yourself)**: Is there duplicated code that can be consolidated?
- [ ] **Type Hints**: (Python) Are type hints used and accurate?

### Performance
- [ ] **Efficiency**: Are there any evident O(n^2) or worse algorithms that could be O(n)?
- [ ] **Resources**: Are file handles, database connections, or network resources managed properly?

### Security
- [ ] **Input Validation**: Is outside input sanitized?
- [ ] **Credentials**: Are there hardcoded secrets? (Flag immediately!)

## 2. Feedback Format

Provide your feedback in the following Markdown format:

```markdown
## Code Review: [Filename]

### Summary
[Brief high-level summary of the code and its quality]

### Critical Issues 🔴
- **[Line X]**: [Description of a bug or major logic flaw]

### Suggestions 🟡
- **[Line Y]**: [Suggestion for improvement, refactoring, or better naming]

### Nitpicks 🟢
- **[Line Z]**: [Minor style or formatting comment]

### Positive Notes
- [What was done well?]
```

## 3. Action Plan
If there are critical issues, offer to fix them immediately.
