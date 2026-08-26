"""Generate the deterministic CastMe audition dataset.

The original feature and callback values use seed 2026. Two independently
seeded audit-only fields support realistic deletion-request experiments:
audition dates and recorded submission-level withdrawal requests.
"""

from __future__ import annotations

from collections import defaultdict
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from sklearn.model_selection import train_test_split


GENERATION_SEED = 2026
SPLIT_SEED = 42
ROW_COUNT = 60_000
SELECTED_ACTOR_ID = "A034"
AGENCY_WITHDRAWAL_ID = "M20"
PROJECT_CLOSURE_ID = "F50"
DATE_START = pd.Timestamp("2021-01-01")
DATE_END = pd.Timestamp("2025-12-31")
ROUND_SPACING_DAYS = 7
RETENTION_CUTOFF = pd.Timestamp("2021-09-20")
WITHDRAWAL_RATE = 0.05

MODEL_AND_TARGET_COLUMNS = [
    "audition_id",
    "actor_id",
    "manager_id",
    "film_id",
    "role_id",
    "audition_round",
    "experience_years",
    "role_match_score",
    "accent_match",
    "availability_match",
    "line_accuracy_score",
    "special_skill_match",
    "self_tape_quality_score",
    "manager_rating",
    "prior_collaboration",
    "days_before_deadline",
    "callback",
]

COLUMNS = [
    *MODEL_AND_TARGET_COLUMNS[:6],
    "audition_date",
    "withdrawal_requested",
    *MODEL_AND_TARGET_COLUMNS[6:],
]


def actor_manager_map() -> dict[str, str]:
    """Return the coherent, intentionally uneven CastMe agency structure."""
    mapping: dict[str, str] = {}
    ordinary_actors = [f"A{number:03d}" for number in range(1, 91)]
    cursor = 0
    for manager_number in range(1, 15):
        for actor_id in ordinary_actors[cursor : cursor + 5]:
            mapping[actor_id] = f"M{manager_number:02d}"
        cursor += 5
    for manager_number in range(15, 20):
        for actor_id in ordinary_actors[cursor : cursor + 4]:
            mapping[actor_id] = f"M{manager_number:02d}"
        cursor += 4
    for actor_number in range(91, 101):
        mapping[f"A{actor_number:03d}"] = AGENCY_WITHDRAWAL_ID
    assert len(mapping) == 100
    return mapping


def actor_row_counts() -> dict[str, int]:
    """Assign row volumes without inspecting outcomes or split performance."""
    counts = {f"A{number:03d}": 0 for number in range(1, 101)}
    counts["A001"] = 3_000
    for actor_number in range(11, 26):
        counts[f"A{actor_number:03d}"] = 600
    for actor_number in range(91, 101):
        counts[f"A{actor_number:03d}"] = 600

    remaining = [actor_id for actor_id, count in counts.items() if count == 0]
    for actor_id in remaining[:42]:
        counts[actor_id] = 568
    for actor_id in remaining[42:]:
        counts[actor_id] = 567

    assert sum(counts.values()) == ROW_COUNT
    assert sum(counts[f"A{number:03d}"] for number in range(91, 101)) == 6_000
    assert sum(counts[f"A{number:03d}"] for number in range(11, 26)) == 9_000
    return counts


def generate_dataset() -> pd.DataFrame:
    """Create a stochastic but learnable callback dataset from seed 2026."""
    rng = np.random.default_rng(GENERATION_SEED)
    actors = [f"A{number:03d}" for number in range(1, 101)]
    managers = actor_manager_map()
    row_counts = actor_row_counts()
    films = [f"F{number:02d}" for number in range(1, 51)]
    roles_by_film = {
        film_id: [f"RL{(film_position - 1) * 24 + role_position:04d}" for role_position in range(1, 25)]
        for film_position, film_id in enumerate(films, start=1)
    }

    actor_experience = {
        actor_id: round(float(np.clip(rng.normal(7.5, 4.3), 0, 20)), 1)
        for actor_id in actors
    }
    actor_talent = {actor_id: float(rng.normal(0, 0.18)) for actor_id in actors}
    actor_collaboration = {actor_id: float(rng.uniform()) for actor_id in actors}
    # This explicit order reproduces the original saved dataset while removing
    # the legacy process-dependent iteration over ``set(managers.values())``.
    manager_generation_order = [
        "M01", "M02", "M11", "M12", "M13", "M05", "M19", "M16", "M10", "M14",
        "M06", "M03", "M18", "M04", "M20", "M17", "M09", "M07", "M08", "M15",
    ]
    assert set(manager_generation_order) == set(managers.values())
    manager_effect = {
        manager_id: float(rng.normal(0, 0.11))
        for manager_id in manager_generation_order
    }
    film_difficulty = {film_id: float(rng.normal(0, 0.12)) for film_id in films}
    role_requirements = {
        role_id: float(rng.normal(0, 0.14))
        for role_ids in roles_by_film.values()
        for role_id in role_ids
    }

    rows: list[dict[str, object]] = []
    audition_number = 1
    for actor_id in actors:
        actor_films = [PROJECT_CLOSURE_ID] if 11 <= int(actor_id[1:]) <= 25 else films[:-1]
        role_rounds: defaultdict[str, int] = defaultdict(int)
        for local_index in range(row_counts[actor_id]):
            film_id = actor_films[local_index % len(actor_films)]
            role_id = roles_by_film[film_id][(local_index // len(actor_films)) % 24]
            role_rounds[role_id] += 1

            manager_id = managers[actor_id]
            role_match = float(
                np.clip(
                    rng.beta(3.3, 2.5)
                    + actor_talent[actor_id]
                    + role_requirements[role_id]
                    - film_difficulty[film_id],
                    0,
                    1,
                )
            )
            accent_match = int(rng.random() < np.clip(0.52 + 0.25 * (role_match - 0.5), 0.15, 0.9))
            availability_match = int(rng.random() < np.clip(0.60 + 0.22 * (role_match - 0.5), 0.2, 0.92))
            line_accuracy = float(np.clip(rng.beta(3.4, 2.3) + 0.15 * actor_talent[actor_id], 0, 1))
            special_skill = int(rng.random() < np.clip(0.28 + 0.20 * (role_match - 0.5), 0.05, 0.75))
            self_tape = float(np.clip(rng.beta(3.1, 2.5) + 0.10 * actor_talent[actor_id], 0, 1))
            manager_rating = float(np.clip(0.57 + manager_effect[manager_id] + rng.normal(0, 0.12), 0, 1))
            prior_collaboration = int(rng.random() < 0.12 + 0.34 * actor_collaboration[actor_id])
            days_before_deadline = int(rng.integers(0, 31))

            logit = (
                -1.90
                + 3.10 * (role_match - 0.5)
                + 0.62 * availability_match
                + 0.48 * accent_match
                + 1.00 * (line_accuracy - 0.5)
                + 0.44 * (self_tape - 0.5)
                + 0.58 * (manager_rating - 0.5)
                + 0.34 * special_skill
                + 0.31 * prior_collaboration
                + 0.18 * ((actor_experience[actor_id] / 20) - 0.5)
                + 0.22 * ((days_before_deadline / 30) - 0.5)
                + rng.normal(0, 0.30)
            )
            callback = int(rng.random() < 1 / (1 + np.exp(-logit)))
            rows.append(
                {
                    "audition_id": f"AU{audition_number:06d}",
                    "actor_id": actor_id,
                    "manager_id": manager_id,
                    "film_id": film_id,
                    "role_id": role_id,
                    "audition_round": role_rounds[role_id],
                    "experience_years": actor_experience[actor_id],
                    "role_match_score": round(role_match, 6),
                    "accent_match": accent_match,
                    "availability_match": availability_match,
                    "line_accuracy_score": round(line_accuracy, 6),
                    "special_skill_match": special_skill,
                    "self_tape_quality_score": round(self_tape, 6),
                    "manager_rating": round(manager_rating, 6),
                    "prior_collaboration": prior_collaboration,
                    "days_before_deadline": days_before_deadline,
                    "callback": callback,
                }
            )
            audition_number += 1

    dataset = pd.DataFrame(rows, columns=MODEL_AND_TARGET_COLUMNS)
    dataset = add_audit_metadata(dataset)
    assert dataset.shape == (ROW_COUNT, len(COLUMNS))
    assert dataset["audition_id"].is_unique
    assert not dataset.duplicated().any()
    assert dataset.groupby("actor_id")["manager_id"].nunique().eq(1).all()
    assert dataset.groupby("role_id")["film_id"].nunique().eq(1).all()
    assert not dataset.duplicated(["actor_id", "role_id", "audition_round"]).any()
    assert dataset["callback"].mean() > 0.35 and dataset["callback"].mean() < 0.40
    validate_audit_metadata(dataset)
    return dataset


def stable_score(namespace: str, value: str) -> float:
    """Return a process-independent score in [0, 1) from SHA-256."""
    payload = f"{GENERATION_SEED}|{namespace}|{value}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") / 2**64


def add_audit_metadata(dataset: pd.DataFrame) -> pd.DataFrame:
    """Add metadata without inspecting features or callback labels."""
    max_round = int(dataset["audition_round"].max())
    latest_base = DATE_END - pd.Timedelta(days=(max_round - 1) * ROUND_SPACING_DAYS)
    available_days = (latest_base - DATE_START).days + 1
    pair_keys = dataset[["actor_id", "role_id"]].astype(str).agg("|".join, axis=1)
    pair_offsets = pair_keys.map(
        lambda key: int(stable_score("audition_date", key) * available_days)
    )
    dates = (
        DATE_START
        + pd.to_timedelta(pair_offsets, unit="D")
        + pd.to_timedelta(
            (dataset["audition_round"] - 1) * ROUND_SPACING_DAYS,
            unit="D",
        )
    )
    withdrawals = dataset["audition_id"].map(
        lambda audition_id: stable_score("withdrawal_requested", audition_id)
        < WITHDRAWAL_RATE
    )
    augmented = dataset.copy()
    augmented.insert(6, "audition_date", dates.dt.strftime("%Y-%m-%d"))
    augmented.insert(7, "withdrawal_requested", withdrawals.astype(bool))
    return augmented[COLUMNS]


def seed_42_split(dataset: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reproduce the established 70/15/15 stratified row split."""
    indices = dataset.index.to_numpy()
    train_indices, temporary_indices = train_test_split(
        indices,
        test_size=0.30,
        random_state=SPLIT_SEED,
        stratify=dataset["callback"],
    )
    validation_indices, test_indices = train_test_split(
        temporary_indices,
        test_size=0.50,
        random_state=SPLIT_SEED,
        stratify=dataset.loc[temporary_indices, "callback"],
    )
    return train_indices, validation_indices, test_indices


def validate_audit_metadata(dataset: pd.DataFrame) -> None:
    """Validate metadata integrity and the frozen scenario configuration."""
    parsed_dates = pd.to_datetime(dataset["audition_date"], format="%Y-%m-%d", errors="raise")
    assert parsed_dates.between(DATE_START, DATE_END).all()
    assert dataset["withdrawal_requested"].dtype == bool

    ordered = dataset.assign(_date=parsed_dates).sort_values(
        ["actor_id", "role_id", "audition_round"]
    )
    chronological = ordered.groupby(["actor_id", "role_id"])["_date"].apply(
        lambda values: values.is_monotonic_increasing and values.is_unique
    )
    assert chronological.all()

    train_indices, _, _ = seed_42_split(dataset)
    train = dataset.loc[train_indices]
    # Selection uses seed-42 training metadata only: size, manager, films and roles.
    actor_profiles = train.groupby("actor_id").agg(
        training_rows=("audition_id", "size"),
        manager_id=("manager_id", "first"),
        film_count=("film_id", "nunique"),
        role_count=("role_id", "nunique"),
    )
    actor_profiles["training_percentage"] = actor_profiles["training_rows"] / len(train)
    actor_profiles["f50_share"] = train["film_id"].eq(PROJECT_CLOSURE_ID).groupby(train["actor_id"]).mean()
    eligible = actor_profiles.loc[
        actor_profiles["training_percentage"].sub(0.01).abs().le(0.0025)
        & actor_profiles["film_count"].ge(10)
        & actor_profiles["role_count"].ge(50)
        & actor_profiles["f50_share"].lt(1.0)
        & actor_profiles["manager_id"].ne(AGENCY_WITHDRAWAL_ID)
    ].copy()
    eligible["distance"] = eligible["training_percentage"].sub(0.01).abs()
    selected_actor = eligible.reset_index().sort_values(
        ["distance", "actor_id"]
    ).iloc[0]["actor_id"]
    assert selected_actor == SELECTED_ACTOR_ID
    assert abs(actor_profiles.loc[SELECTED_ACTOR_ID, "training_percentage"] - 0.01) <= 0.0025
    actor_rows = train.index[train["actor_id"].eq(SELECTED_ACTOR_ID)]
    project_rows = train.index[train["film_id"].eq(PROJECT_CLOSURE_ID)]
    assert not set(actor_rows).issubset(project_rows)
    assert abs(train["withdrawal_requested"].mean() - 0.05) <= 0.0025
    assert abs(train["manager_id"].eq(AGENCY_WITHDRAWAL_ID).mean() - 0.10) <= 0.0025
    assert abs(train["film_id"].eq(PROJECT_CLOSURE_ID).mean() - 0.15) <= 0.0025
    assert abs((pd.to_datetime(train["audition_date"]) < RETENTION_CUTOFF).mean() - 0.15) <= 0.0025

    withdrawal_rows = dataset.loc[dataset["withdrawal_requested"]]
    assert withdrawal_rows["actor_id"].nunique() >= 90
    assert withdrawal_rows["manager_id"].nunique() >= 15
    assert withdrawal_rows["film_id"].nunique() >= 40
    assert withdrawal_rows["role_id"].nunique() >= 500


def main() -> None:
    output_path = Path(__file__).resolve().parents[1] / "data" / "raw" / "castme" / "castme_auditions.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    previous = pd.read_csv(output_path) if output_path.is_file() else None
    dataset = generate_dataset()
    if previous is not None and set(MODEL_AND_TARGET_COLUMNS).issubset(previous.columns):
        assert_frame_equal(
            previous[MODEL_AND_TARGET_COLUMNS],
            dataset[MODEL_AND_TARGET_COLUMNS],
            check_dtype=False,
        )
    dataset.to_csv(output_path, index=False)
    train_indices, _, _ = seed_42_split(dataset)
    train = dataset.loc[train_indices]
    print(f"Wrote {len(dataset):,} rows to {output_path}")
    print(f"Callback rate: {dataset['callback'].mean():.2%}")
    print(f"Selected 1% actor: {SELECTED_ACTOR_ID} ({train['actor_id'].eq(SELECTED_ACTOR_ID).mean():.2%})")
    print(f"Submission withdrawals: {train['withdrawal_requested'].mean():.2%} of training rows")
    print(f"Retention cutoff: {RETENTION_CUTOFF.date()} ({(pd.to_datetime(train['audition_date']) < RETENTION_CUTOFF).mean():.2%})")


if __name__ == "__main__":
    main()
