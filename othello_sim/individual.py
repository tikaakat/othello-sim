import random


class Individual:
    def __init__(self, ind_id, generation, parent_a_id=None, parent_b_id=None, params=None, family_label=None):
        self.id = ind_id
        self.generation = generation
        self.parent_a_id = parent_a_id
        self.parent_b_id = parent_b_id
        self.params = params or self._random_params()
        self.elo = 1500.0
        self.match_history = []
        self.family_label = family_label or ind_id

    def _random_params(self):
        # リミットは今回設けず、0〜10の範囲で自由にランダム初期化する（傾向を見てから上限を検討する）
        return {
            "corner_weight": random.uniform(0.5, 10.0),
            "danger_zone_weight": random.uniform(0.5, 10.0),
            "mobility_weight": random.uniform(0.5, 10.0),
            "edge_stability_weight": random.uniform(0.5, 10.0),
            "frontier_weight": random.uniform(0.5, 10.0),
            "disc_weight": random.uniform(0.5, 10.0),
            "parity_weight": random.uniform(0.5, 10.0),
            "center_weight": random.uniform(0.5, 10.0),
        }

    def to_dict(self):
        return {
            "id": self.id,
            "generation": self.generation,
            "parent_a_id": self.parent_a_id,
            "parent_b_id": self.parent_b_id,
            "params": self.params,
            "elo": self.elo,
            "match_history": self.match_history,
            "family_label": self.family_label,
        }

    @staticmethod
    def from_dict(d):
        ind = Individual(
            d["id"], d["generation"], d.get("parent_a_id"), d.get("parent_b_id"),
            d["params"], family_label=d.get("family_label"),
        )
        ind.elo = d.get("elo", 1500.0)
        ind.match_history = d.get("match_history", [])
        return ind
