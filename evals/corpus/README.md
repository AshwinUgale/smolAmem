# Corpus

Labelled conversations used by `python -m evals` to measure memory-layer
performance. Each `.json` file is one conversation; the harness loads
every JSON in this directory.

## The 30-conversation target

The master doc calls for **30 multi-turn conversations** at v1.0. We ship
**5 starter conversations** that demonstrate every label pattern:

| File | Pattern |
|------|---------|
| `001_typescript_preference.json` | Preference established early, tested after distractor turns |
| `002_react_to_vue_switch.json` | Fact established then contradicted; newer-wins test |
| `003_project_facts.json` | Multiple semantic facts about the same project |
| `004_time_bounded_request.json` | Time-sensitive context (an in-flight task) |
| `005_multi_fact_recall.json` | Single query expected to recall multiple facts |

Adding the remaining 25 is a labelling project, not a coding one. The
schema is stable; new files dropped here are picked up automatically.

## Schema

Every conversation file matches `evals/schema.py`:

```json
{
  "schema_version": "1",
  "id": "001_typescript_preference",
  "description": "User establishes TypeScript preference at turn 3; tested at turn 12.",
  "turns": [
    {"role": "user", "content": "hi"},
    {"role": "assistant", "content": "hello, how can I help?"},
    {"role": "user", "content": "I work mostly in TypeScript these days"},
    {"role": "user", "content": "what's the weather?", "test_at": null},

    /* ... distractor turns ... */

    {
      "role": "user",
      "content": "what language do I usually use?",
      "test_at": {
        "question": "what language does the user prefer?",
        "expected_fact": "user prefers TypeScript",
        "expected_keywords": ["TypeScript"]
      }
    }
  ]
}
```

Notes:

- `test_at` runs `retrieve(question, k=k)` **before** the turn is added to memory.
- `expected_keywords` are case-insensitive substring matches.
- A conversation can have multiple test points.

## Why this format

Plain JSON, one file per conversation. Easy to label by hand in any text
editor. Easy to version with git. Easy to fork for project-specific
corpora. No proprietary tooling.

## Running the eval

From the repo root:

```bash
# Cheap retrieval-only metrics (no LLM calls at eval time)
python -m evals --corpus evals/corpus --runner mneme --output results.json

# End-to-end with real LLM-generated answers + LLM-judged accuracy
python -m evals --corpus evals/corpus --runner mneme --with-answers --judge
```

Without `--with-answers`, the eval reads every conversation as a ground-truth
transcript and only measures what the memory layer retrieves at test points.
With it, the runner generates the next assistant turn at every test point
using a real LLM and the memory layer's context, then checks the answer.
