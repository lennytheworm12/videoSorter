# Phase 2K Production-Model Selection Report

| Condition | Strict | Unsupported | Grounding | Gate | Calls/target |
|---|---:|---:|---:|---|---:|
| OX Alpha baseline | 109/110 | 0 | 0 | baseline | 1 |
| P (deepseek-v4-pro) | 92/110 | 4 | 0 | FAIL | 1 |
| F (deepseek-v4-flash) | 103/110 | 1 | 0 | FAIL | 1 |
| FV (deepseek-v4-flash) | 103/110 | 1 | 0 | CONDITIONAL_PASS | 6 |

Verifier gate: **VERIFIER_SCALING_NOT_JUSTIFIED** (delta +0, +3/-3 targets)

Production recommendation: **NO_DEEPSEEK_CONFIGURATION_MEETS_PRODUCTION_GATE**

## Per-field strict successes (of 10)

| Field | OX | P | F | FV |
|---|---:|---:|---:|---:|
| actors_entities | 10 | 7 | 10 | 9 |
| reference_bindings | 10 | 7 | 7 | 8 |
| abilities_resources | 10 | 9 | 9 | 10 |
| events_actions | 9 | 9 | 9 | 9 |
| states | 10 | 9 | 9 | 8 |
| conditions | 10 | 10 | 10 | 10 |
| recommended_advice | 10 | 9 | 10 | 10 |
| consequences_outcomes | 10 | 8 | 9 | 9 |
| explicit_relationships | 10 | 8 | 10 | 10 |
| uncertainty_unresolved | 10 | 6 | 10 | 10 |
| supporting_source_spans | 10 | 10 | 10 | 10 |

## Cost accounting

- P: 10 calls, 195.0s/target, retries 0, parse failures 0, transports deepseek_api
- F: 10 calls, 231.9s/target, retries 0, parse failures 0, transports deepseek_api
- FV: 60 calls, 1183.9s/target, retries 4, parse failures 4, transports deepseek_api
