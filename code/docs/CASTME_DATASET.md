# CastMe synthetic dataset

`scripts/generate_castme_dataset.py` deterministically creates the 60,000-row
CastMe audition dataset. The original ten classifier inputs and `callback`
labels use generation seed `2026` and are preserved by the metadata redesign.

Each row is one audition submission. `audition_id` is unique and
`audition_round` orders repeated attempts for an actor-role pair. Actors are
performers, managers are agencies or representation accounts, films are
projects and roles belong to films.

## Audit-only metadata

| Field | Definition | Model use |
| --- | --- | --- |
| `audition_date` | ISO date from 2021-01-01 through 2025-12-31; repeated rounds are seven days apart | Excluded |
| `withdrawal_requested` | Deterministic SHA-256 selection representing recorded submission-level withdrawals | Excluded |

Dates and withdrawals use stable SHA-256 scores independent of callback labels.
They do not affect preprocessing, training or threshold selection.

## Frozen seed-42 deletion scenarios

Percentages are measured against the 42,000-row baseline training partition.

| Scenario | Frozen condition | Target |
| --- | --- | ---: |
| Actor withdrawal | `actor_id == "A034"` | 1% |
| Submission withdrawal | `withdrawal_requested == True` | 5% |
| Agency withdrawal | `manager_id == "M20"` | 10% |
| Project closure | `film_id == "F50"` | 15% |
| Retention expiry | `audition_date < "2021-09-20"` | 15% |

`A034` is selected deterministically from seed-42 training metadata: it is close
to 1%, spans many films and roles, is not managed by M20, and is not contained
in F50. Lexical actor ID resolves ties. The date cutoff is a literal rather
than a rerun-time quantile. Scenarios are evaluated independently and may
overlap; they are not cumulative requests.

These are synthetic operational structures for unlearning research. They do
not claim that every condition represents a particular legal right or that
behavioural unlearning demonstrates regulatory compliance.
