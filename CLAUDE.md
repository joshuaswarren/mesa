# Aurora Silicon assistant policy

This is Aurora Silicon's own policy for AI coding assistants working in
this repository. It is the only assistant instruction file this tree
carries. Upstream projects' instruction files are never carried here.

AGENTS.md, CLAUDE.md, GEMINI.md and .github/copilot-instructions.md are
copies of .github/aurora/AGENTS.md. The aurora-agent-policy workflow
puts them back if anything else replaces them, including an upstream
merge. To change the policy, edit .github/aurora/AGENTS.md.

## Rules

- Only public code goes in. Never copy code from proprietary sources.
- Changes are tested on the hardware they affect before they are merged.

## Hardware benchmarks for pull requests

Every agent-authored Mesa pull request must include before-and-after benchmark
results from physical hardware affected by the change. Run the same relevant
workload against the target-branch baseline and the proposed Mesa revision on
the same hardware with matching test conditions. Verify that the benchmark
exercises the affected path on each build as applicable. If hardware results
are unavailable, report the blocker and wait for evidence before opening the PR.

In the PR description, include the tested hardware and software configuration,
both Mesa commit IDs, the benchmark and workload versions, and the exact
commands and settings. Report baseline and proposed measurements side by side
with units and percentage changes. Include the run count and describe any
variability that affects the comparison. Link the benchmark source and any test
scripts, and attach or link the raw output for both runs somewhere reviewers
can access. Call out regressions and explain any trade-offs for review.

For feature enablement, benchmark an application that uses the feature. Use
extension and conformance checks as additional evidence of correctness.

## To be expanded

Build, test and review conventions for this repository will go here.
