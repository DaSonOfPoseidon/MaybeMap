# False positives

Findings reviews raised here that were judged wrong. Each entry is one narrow pattern, why it's fine, and where it
came up. The gate shows this file to its reviewers. Never add a whole class of bug.

## The codex-review caller passes no `with: pr:`

- **Pattern:** `.github/workflows/codex-review.yml` calls
  `DaSonOfPoseidon/claude-ship/.github/workflows/codex-status.yml@v1` with no `with: pr:`, so a reviewer
  concludes a `workflow_dispatch` run has no PR number.
- **Why it's fine:** the called workflow reads the number as `github.event.inputs.pr || inputs.pr`, and a
  reusable workflow sees its caller's `github.event`, so a dispatch's `pr` input reaches it unchanged. The file
  is claude-ship's own template, used as is.
- **Where:** the pre-push gate on #12 (user waived).
