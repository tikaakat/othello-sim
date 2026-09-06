import json
import os

from .individual import Individual


def load_checkpoint(data_dir):
    individuals_path = os.path.join(data_dir, "individuals.json")
    matches_path = os.path.join(data_dir, "matches.json")

    if not (os.path.exists(individuals_path) and os.path.exists(matches_path)):
        return None, None

    with open(individuals_path, "r", encoding="utf-8") as f:
        all_individuals = {d["id"]: Individual.from_dict(d) for d in json.load(f)}
    with open(matches_path, "r", encoding="utf-8") as f:
        matches_log = json.load(f)

    return all_individuals, matches_log


def export_population(all_individuals, matches_log, data_dir):
    os.makedirs(data_dir, exist_ok=True)
    individuals_path = os.path.join(data_dir, "individuals.json")
    matches_path = os.path.join(data_dir, "matches.json")

    with open(individuals_path, "w", encoding="utf-8") as f:
        json.dump([ind.to_dict() for ind in all_individuals], f, ensure_ascii=False, indent=2)

    with open(matches_path, "w", encoding="utf-8") as f:
        json.dump(matches_log, f, ensure_ascii=False, indent=2)

    print(f"Exported: {individuals_path}, {matches_path}")


def load_training_state(data_dir):
    path = os.path.join(data_dir, "training_state.json")
    if not os.path.exists(path):
        return {"benchmark_depth": None, "win_rate_history": [], "active_population_ids": None}
    with open(path, "r", encoding="utf-8") as f:
        state = json.load(f)
        state.setdefault("active_population_ids", None)
        return state


def save_training_state(data_dir, state):
    os.makedirs(data_dir, exist_ok=True)
    path = os.path.join(data_dir, "training_state.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
