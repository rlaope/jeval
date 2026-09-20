<!-- Keep it short. Reviewers read the diff first. -->

## What changed

<!-- One paragraph. Why, not just what. -->

## How this was verified

<!-- Paste the command(s) you ran and their result. -->

```
uv run pytest
uv run ruff check .
uv run mypy jeval
```

## Acceptance criteria addressed

<!-- Which criterion does this close, or which behavior does it fix? -->

## Checklist

- [ ] Statistics changes are covered by a synthetic test that fails without the change
- [ ] No test tolerance was widened to make this pass
- [ ] Nothing from the non-goals list in `CLAUDE.md` was added
- [ ] No user data, cost figures, or generated reports are committed
- [ ] Artifacts are in English; commits follow Conventional Commits
