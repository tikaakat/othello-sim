import random

from .individual import Individual
from .elo import update_elo
from .play import play_individual_vs_individual, play_vs_benchmark
from .family_names import random_immigrant_family_name

MUTATION_RATE = 0.2
MUTATION_STRENGTH = 0.25

PARAM_KEYS = [
    "corner_weight", "danger_zone_weight", "mobility_weight", "edge_stability_weight",
    "frontier_weight", "disc_weight", "parity_weight", "center_weight",
]


def _new_id(generation, suffix=""):
    return f"G{generation}-{format(random.randint(0, 46655), 'x').upper()}{suffix}"


# ============================================================
# 無性生殖：親の変異版を1体作る
# ============================================================
def mutate_clone(parent, generation):
    child_params = {}
    for key in PARAM_KEYS:
        val = parent.params[key]
        if random.random() < MUTATION_RATE:
            val *= random.uniform(1 - MUTATION_STRENGTH, 1 + MUTATION_STRENGTH)
        child_params[key] = max(0.01, val)

    child = Individual(
        _new_id(generation), generation, parent_a_id=parent.id, parent_b_id=None,
        params=child_params, family_label=parent.family_label,
    )
    child.elo = parent.elo
    return child


# ============================================================
# 移民との交配イベント：パラメータごとに、どちらかの親から丸ごと継承する「組み換え」方式。
# 平均化（ブレンド）は多様性を減らす方向に働くため採用しない。家名は合成表記。
# ============================================================
def crossover_main_sub(parent_main, parent_sub, new_id, generation, family_label=None):
    child_params = {}
    for key in PARAM_KEYS:
        # パラメータごとに、どちらか一方の親の値をそのまま受け継ぐ（平均は取らない）
        source = parent_main if random.random() < 0.5 else parent_sub
        val = source.params[key]
        if random.random() < MUTATION_RATE:
            val *= random.uniform(1 - MUTATION_STRENGTH, 1 + MUTATION_STRENGTH)
        child_params[key] = max(0.01, val)

    child = Individual(
        new_id, generation, parent_main.id, parent_sub.id, child_params,
        family_label=family_label or parent_main.family_label,
    )
    child.elo = (parent_main.elo + parent_sub.elo) / 2  # 実力の初期値は両親の中間からスタート
    return child



def generate_immigrant(generation):
    return Individual(
        ind_id=_new_id(generation, suffix="i"), generation=generation,
        family_label=random_immigrant_family_name(),
    )


def crossover_hybrid(parent_top, immigrant, generation):
    combined_label = f"{parent_top.family_label}×{immigrant.family_label}"
    return crossover_main_sub(parent_top, immigrant, _new_id(generation, suffix="h"), generation, family_label=combined_label)


def manual_breed(population_by_id, parent_a_id, parent_b_id, generation):
    parent_a = population_by_id[parent_a_id]
    parent_b = population_by_id[parent_b_id]
    return crossover_main_sub(parent_a, parent_b, _new_id(generation, suffix="m"), generation)


# ============================================================
# スイス方式トーナメント
# ============================================================
def swiss_pairing(ranked_ids, played_pairs):
    unpaired = list(ranked_ids)
    pairs = []
    while len(unpaired) >= 2:
        a = unpaired.pop(0)
        matched_idx = None
        for i, b in enumerate(unpaired):
            if frozenset((a, b)) not in played_pairs:
                matched_idx = i
                break
        if matched_idx is None:
            matched_idx = 0
        b = unpaired.pop(matched_idx)
        pairs.append((a, b))
    return pairs


def run_swiss_tournament(population, generation, matches_log, rounds=4, depth=4):
    by_id = {ind.id: ind for ind in population}
    score = {ind.id: 0.0 for ind in population}
    played_pairs = set()

    for rnd_idx in range(rounds):
        ranked_ids = sorted(by_id.keys(), key=lambda i: (-score[i], -by_id[i].elo))
        pairs = swiss_pairing(ranked_ids, played_pairs)
        print(f"  --- スイス方式 ラウンド{rnd_idx + 1}/{rounds}（{len(pairs)}局）---")

        for pair_idx, (a_id, b_id) in enumerate(pairs):
            played_pairs.add(frozenset((a_id, b_id)))
            ind_a, ind_b = by_id[a_id], by_id[b_id]

            outcome_a, moves, black, white = play_individual_vs_individual(ind_a, ind_b, depth=depth)
            ind_a.elo, ind_b.elo = update_elo(ind_a.elo, ind_b.elo, outcome_a)

            if outcome_a == "win":
                score[a_id] += 1.0
            elif outcome_a == "loss":
                score[b_id] += 1.0
            else:
                score[a_id] += 0.5
                score[b_id] += 0.5

            print(f"    局{pair_idx + 1}/{len(pairs)}: {a_id} vs {b_id} → {outcome_a}（{black}-{white}）")

            outcome_b = {"win": "loss", "loss": "win", "draw": "draw"}[outcome_a]
            ind_a.match_history.append({"opponent": b_id, "result": outcome_a, "generation": generation})
            ind_b.match_history.append({"opponent": a_id, "result": outcome_b, "generation": generation})
            matches_log.append({
                "individual_a_id": a_id, "individual_b_id": b_id, "opponent_type": "individual",
                "result": outcome_a, "moves": moves, "generation": generation,
                "final_score": {"black": black, "white": white},
                "individual_a_color": "black",
            })

    return score


# ============================================================
# タイトル戦：温度計の壁を極限まで登り切った首位だけが挑戦できる、
# 「タイトルホルダー」（暫定：自前の深い探索。将来的に本物のEdaxへ差し替え予定）
# ============================================================
TITLE_HOLDER_DEPTH = 7  # 暫定のタイトルホルダーの強さ（Edax接続までの仮の壁。エンジン高速化後に引き上げ予定）
TITLE_MATCH_GAMES = 5


def run_title_challenge(champion, generation, matches_log, individual_depth=4):
    """首位がタイトルホルダーに挑戦する。Eloには影響させない特別な記録として残す"""
    wins = 0
    for _ in range(TITLE_MATCH_GAMES):
        outcome, moves, color, black, white = play_vs_benchmark(
            champion, individual_depth=individual_depth, benchmark_depth=TITLE_HOLDER_DEPTH,
        )
        if outcome == "win":
            wins += 1
        champion.match_history.append({
            "opponent": "title_holder", "result": outcome, "generation": generation, "color": color,
        })
        matches_log.append({
            "individual_a_id": champion.id, "opponent_type": "title_holder",
            "result": outcome, "moves": moves, "generation": generation,
            "final_score": {"black": black, "white": white},
            "individual_a_color": color,
        })
    won_title = wins > TITLE_MATCH_GAMES / 2
    return won_title, wins


# ============================================================
# ベンチマーク「温度計」：Eloには影響させず、現在のレベルを監視するためだけの対局
# ============================================================
def run_benchmark_thermometer(top_individual, generation, matches_log, games=5,
                               individual_depth=4, benchmark_depth=5):
    for _ in range(games):
        outcome, moves, color, black, white = play_vs_benchmark(
            top_individual, individual_depth=individual_depth, benchmark_depth=benchmark_depth,
        )
        top_individual.match_history.append({
            "opponent": "benchmark", "result": outcome, "generation": generation, "color": color,
        })
        matches_log.append({
            "individual_a_id": top_individual.id, "opponent_type": "benchmark",
            "result": outcome, "moves": moves, "generation": generation,
            "final_score": {"black": black, "white": white},
            "individual_a_color": color,
        })


# ============================================================
# 新規参入個体の生成：基本は変異クローンで埋め、うち1枠は毎世代必ず新しい移民に譲る。
# 移民は次世代のトーナメントで実力を証明できれば生き残り、弱ければそのまま消える。
# ============================================================
def generate_new_entrants(survivors, generation, population_size, immigrant_count=1):
    slots = population_size - len(survivors)
    new_entrants = []

    actual_immigrant_count = min(immigrant_count, slots)
    for _ in range(actual_immigrant_count):
        immigrant = generate_immigrant(generation)
        new_entrants.append(immigrant)
        print(f"  → 移民 {immigrant.id}［{immigrant.family_label}流］が現れました（実力を証明できれば生き残ります）")

    # 残りの枠は、生存者の変異クローン（無性生殖）で埋める
    ranked_survivors = sorted(survivors, key=lambda ind: -ind.elo) if survivors else []
    idx = 0
    while len(new_entrants) < slots and ranked_survivors:
        parent = ranked_survivors[idx % len(ranked_survivors)]
        new_entrants.append(mutate_clone(parent, generation))
        idx += 1

    return new_entrants[:slots]


def select_survivors_with_protection(ranked, population_size, protected_families, max_slots_per_family_ratio=0.5):
    """
    保護対象（恒久保証＋猶予期間中の移民）の流派には、優先的に1枠を確保する。
    保護対象が生存枠を超える場合は、代表の順位が低い流派から優先度を落とす
    （＝恒久保証側が一時的に席を譲る形になる。移民の猶予は必ず守られる）。
    残りの枠は成績順で埋めるが、1流あたりの取得枠数には上限を設ける。
    """
    slots = max(1, population_size // 2)
    max_per_family = max(1, int(slots * max_slots_per_family_ratio))

    rank_index = {ind.id: i for i, ind in enumerate(ranked)}

    family_reps = {}
    for ind in ranked:
        label = ind.family_label
        if label not in family_reps or rank_index[ind.id] < rank_index[family_reps[label].id]:
            family_reps[label] = ind

    present_protected = [fam for fam in protected_families if fam in family_reps]
    present_protected_sorted = sorted(present_protected, key=lambda fam: rank_index[family_reps[fam].id])
    guaranteed_labels = present_protected_sorted[:slots]

    guaranteed = [family_reps[fam] for fam in guaranteed_labels]
    guaranteed_ids = {ind.id for ind in guaranteed}

    survivors = list(guaranteed)
    family_counts = {}
    for ind in survivors:
        family_counts[ind.family_label] = family_counts.get(ind.family_label, 0) + 1

    for ind in ranked:
        if len(survivors) >= slots:
            break
        if ind.id in guaranteed_ids:
            continue
        if family_counts.get(ind.family_label, 0) >= max_per_family:
            continue
        survivors.append(ind)
        family_counts[ind.family_label] = family_counts.get(ind.family_label, 0) + 1

    if len(survivors) < slots:
        chosen_ids = {ind.id for ind in survivors}
        for ind in ranked:
            if len(survivors) >= slots:
                break
            if ind.id in chosen_ids:
                continue
            survivors.append(ind)

    return survivors


# ============================================================
# 1世代分の処理
# ============================================================
def run_generation(population, generation, matches_log,
                    population_size=16, swiss_rounds=4, search_depth=4,
                    benchmark_depth=5, benchmark_games=5,
                    immigrant_count=1,
                    guaranteed_families=None, grace_info=None,
                    grace_period=3, reset_interval=10):
    """
    guaranteed_families: 恒久保証されている流派名のリスト（reset_intervalごとに引き直す）
    grace_info: {流派名: 保護終了世代} の辞書。新規移民に猶予期間として付与する
    """
    guaranteed_families = list(guaranteed_families) if guaranteed_families else []
    grace_info = dict(grace_info) if grace_info else {}

    score = run_swiss_tournament(
        population, generation, matches_log, rounds=swiss_rounds, depth=search_depth,
    )

    ranked = sorted(population, key=lambda ind: (-score[ind.id], -ind.elo))

    if ranked:
        run_benchmark_thermometer(
            ranked[0], generation, matches_log, games=benchmark_games,
            individual_depth=search_depth, benchmark_depth=benchmark_depth,
        )

    slots = max(1, population_size // 2)
    rank_index = {ind.id: i for i, ind in enumerate(ranked)}

    def top_family_labels(n):
        family_reps = {}
        for ind in ranked:
            label = ind.family_label
            if label not in family_reps or rank_index[ind.id] < rank_index[family_reps[label].id]:
                family_reps[label] = ind
        return sorted(family_reps.keys(), key=lambda fam: rank_index[family_reps[fam].id])[:n]

    # 初回（恒久保証リストが空）は、その時点の上位流派で初期化する
    if not guaranteed_families:
        guaranteed_families = top_family_labels(slots)
        print(f"  恒久保証リストを初期化しました: {', '.join(guaranteed_families)}")

    # 現在保護されている流派 = 恒久保証 ∪ 猶予期間中の流派（移民は必ずフル猶予を得られる）
    active_grace = {fam for fam, expiry in grace_info.items() if generation <= expiry}
    protected_families = set(guaranteed_families) | active_grace

    survivors = select_survivors_with_protection(ranked, population_size, protected_families)

    families_present = sorted(set(ind.family_label for ind in survivors))
    grace_note = f"（猶予中: {', '.join(sorted(active_grace))}）" if active_grace else ""
    print(f"  現在の流派（生存者内）: {', '.join(families_present)} {grace_note}")

    # 定期リセット：恒久保証リストを、その時点の実力順で引き直す
    if reset_interval and generation > 0 and generation % reset_interval == 0:
        new_guaranteed = top_family_labels(slots)
        print(f"  ★ 恒久保証リストを更新: {', '.join(guaranteed_families)} → {', '.join(new_guaranteed)}")
        guaranteed_families = new_guaranteed
        # 期限切れの猶予情報は掃除する
        grace_info = {fam: exp for fam, exp in grace_info.items() if generation <= exp}

    # 毎世代必ず1体の移民枠を確保し、猶予期間を新規付与する
    new_entrants = generate_new_entrants(survivors, generation + 1, population_size, immigrant_count)
    for ind in new_entrants:
        is_fresh_immigrant = (ind.parent_a_id is None and ind.parent_b_id is None)
        if is_fresh_immigrant and ind.family_label not in grace_info and ind.family_label not in guaranteed_families:
            grace_info[ind.family_label] = generation + 1 + grace_period
            print(f"  → {ind.family_label}流に猶予期間を付与（世代{generation + 1 + grace_period}まで保護）")

    return survivors + new_entrants, guaranteed_families, grace_info
