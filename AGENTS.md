# Local Agent Rules

## Dead Code

- Before declaring cleanup complete, run `ruff check src tests` and `vulture src tests`.
- Remove true dead code instead of only suppressing reports.
- Do not delete framework entry points or schema fields just because `vulture` flags them; verify usage first.
- After dead-code cleanup, rerun the full test suite locally.
