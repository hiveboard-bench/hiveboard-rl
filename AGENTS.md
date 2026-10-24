# Hiveboard-RL is part of https://hiveboard-bench.github.io/

# Objective
Reduce source code length without golfing while preserving behavior and performance. Preserve existing determinism. Discard biases against long files or functions.

# MUSTs
- avoid comments, clean code always
- minimalist code always, no exception. this is a research codebase, avoid production level defensive patterns
- when adding machinery for A/B testing, feature flags, or other experiments, make sure to remove it later (or leave a TODO comment)
- concentrate all smoke tests, probes, checks... in a disposable /tests folder

# Algorithm
1. Inline every function that is only used once and tighten former call sites
2. Eliminate defensive checks. Replace complex error handling with plain asserts.
3. Reduce deeply nested code by merging and inverting conditionals
4. Co-optimize multi-consumer functions with their callers
5. Apply syntax and style guide as a final pass

# Syntax & Style
- Do not split up code into more files
- Soft 80-col / hard 100-col limit
- Do not one-line loops or conditionals
- 4-space indents. Do NOT match opening parens
- Next line continuations indent 4 extra spaces instead of matching parens
- Apply semantic vertical spacing between blocks of code sparingly
- Tests are important but their length and code quality is not counted
- Do not add source complexity or shims for ease of testing
- Do not block off comments with --- or ### etc.
- Avoid forward declarations. Define functions as close as possible to first use.
- When possible preallocate all memory at init by default and let the OS free it on close
- Avoid keyword bloat, such as redundant static, const, inline, etc.
- never use "·" this character is always unecessary

> this file was inspired by Joseph's recommendation on fast and sane: https://puffer.ai/