# AGENTS.md

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

## 5. Windows / PowerShell / Unicode Safety

**Avoid Unicode-related failures by keeping machine-facing interfaces ASCII-only.**

When running scripts on Windows or through PowerShell:

* Do not pass Chinese dataset names, non-ASCII labels, or Unicode paths as command-line arguments.
* Use stable ASCII identifiers such as `profile_id`, `dataset_id`, or `slug` for CLI arguments.
* Store display names, Unicode paths, plot titles, and dataset metadata in UTF-8 config files such as JSON or YAML.
* Run Python in UTF-8 mode when possible:

  * `python -X utf8 ...`
  * or set `PYTHONUTF8=1`.
* In Python, always read and write text files with explicit `encoding="utf-8"` unless the dataset profile explicitly specifies another encoding.
* Use `pathlib.Path` for filesystem paths.
* Do not build shell command strings by concatenation.
* Prefer `subprocess.run([...], shell=False, check=True, text=True, encoding="utf-8")` over shell string execution.
* Never delete an existing output directory before regeneration.
* Generate outputs into a temporary directory, validate them, then replace the final directory.
* If generation fails, preserve the previous successful outputs.

For datasets with Chinese names, do not pass the Chinese name directly through PowerShell or shell command-line arguments.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.
