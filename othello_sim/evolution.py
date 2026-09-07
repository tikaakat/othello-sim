import random

from .individual import Individual
from .elo import update_elo
from .play import play_individual_vs_individual, play_vs_benchmark
from .family_names import random_immigrant_family_name

MUTATION_RATE = 0.2
MUTATION_STRENGTH = 0.25

PARAM_KEYS = ["corner_weight", "mobility_weight", "stability_weight", "disc_weight"]


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
# 移民との交配イベント：両親の値をブレンドして「尖りを均す」。家名は合成表記
# ============================================================
def crossover_main_sub(parent_main, parent_sub, new_id, generation, family_label=None):
    child_params = {}
    for key in PARAM_KEYS:
        # 単純な二者択一ではなく加重平均でブレンドする。
        # 進化（変異）で尖った個体同士でも、交配すればバランスの取れた値に寄る。
        main_weight = random.uniform(0.4, 0.6)  # 主にやや比重を残しつつ、ほぼ半々でブレンド
        blended = parent_main.params[key] * main_weight + parent_sub.params[key] * (1 - main_weight)
        if random.random() < MUTATION_RATE:
            blended *= random.uniform(1 - MUTATION_STRENGTH, 1 + MUTATION_STRENGTH)
        child_params[key] = max(0.01, blended)

    child = Individual(
        new_id, generation, parent_main.id, parent_sub.id, child_params,
        family_label=family_label or parent_main.family_label,
    )
    child.elo = (parent_main.elo + parent_sub.elo) / 2  # ブレンドなので実力も両親の中間からスタート
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
# 新規参入個体の生成：通常は変異クローン、移民が来た世代だけ交配イベント
# ============================================================
def generate_new_entrants(survivors, generation, population_size, immigrant_count=0):
    slots = population_size - len(survivors)
    new_entrants = []

    for _ in range(immigrant_count):
        if len(new_entrants) >= slots:
            break
        immigrant = generate_immigrant(generation)
        new_entrants.append(immigrant)
        print(f"  → 移民 {immigrant.id}［{immigrant.family_label}流］が現れました")

        if len(new_entrants) < slots and survivors:
            top = max(survivors, key=lambda ind: ind.elo)
            hybrid = crossover_hybrid(top, immigrant, generation)
            new_entrants.append(hybrid)
            print(f"  → {top.id}［{top.family_label}流］と交配し、{hybrid.id}［{hybrid.family_label}流］が誕生しました")

    # 残りの枠は、生存者の変異クローン（無性生殖）で埋める
    ranked_survivors = sorted(survivors, key=lambda ind: -ind.elo) if survivors else []
    idx = 0
    while len(new_entrants) < slots and ranked_survivors:
        parent = ranked_survivors[idx % len(ranked_survivors)]
        new_entrants.append(mutate_clone(parent, generation))
        idx += 1

    return new_entrants[:slots]


def select_survivors_with_species_guarantee(ranked, population_size, max_slots_per_family_ratio=0.5):
    """
    純粋な成績順だけで選ぶと、強い流派が枠を独占して他の流派が根絶やしになる
    （NEATの「種分化」の考え方を採用し、各流に最低1枠を保証する）。
    さらに、1つの流派が枠を独占しすぎないよう、1流あたりの上限も設ける。

    手順:
    1. 現在の集団に存在する流派ごとに、その中の最高成績の個体を「代表」とする
    2. 代表をランキング順に並べ、生存枠が許す限り「1流1枠」を優先的に確保する
    3. 残った枠は成績順で埋めるが、1流あたりの取得枠数が上限（既定：生存枠の半分）に
       達した流派はスキップし、他の流派に機会を譲る
    """
    slots = max(1, population_size // 2)
    max_per_family = max(1, int(slots * max_slots_per_family_ratio))

    rank_index = {ind.id: i for i, ind in enumerate(ranked)}

    families = {}
    for ind in ranked:
        label = ind.family_label
        if label not in families or rank_index[ind.id] < rank_index[families[label].id]:
            families[label] = ind

    representatives = sorted(families.values(), key=lambda ind: rank_index[ind.id])
    guaranteed = representatives[:slots]
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
            continue  # この流派は既に上限枠を取得済み。他の流派に譲る
        survivors.append(ind)
        family_counts[ind.family_label] = family_counts.get(ind.family_label, 0) + 1

    # 上限キャップのせいで枠が余ってしまった場合（多様な流派で埋め切れなかった場合）は、
    # キャップを無視してでも残り枠を成績順で埋める（空席を残さない）
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
                    immigrant_count=0, immigrant_interval=5):
    score = run_swiss_tournament(
        population, generation, matches_log, rounds=swiss_rounds, depth=search_depth,
    )

    ranked = sorted(population, key=lambda ind: (-score[ind.id], -ind.elo))

    if ranked:
        run_benchmark_thermometer(
            ranked[0], generation, matches_log, games=benchmark_games,
            individual_depth=search_depth, benchmark_depth=benchmark_depth,
        )

    survivors = select_survivors_with_species_guarantee(ranked, population_size)

    families_present = sorted(set(ind.family_label for ind in survivors))
    print(f"  現在の流派（生存者内）: {', '.join(families_present)}")

    this_gen_immigrants = immigrant_count
    if immigrant_interval and generation > 0 and generation % immigrant_interval == 0:
        this_gen_immigrants += 1

    new_entrants = generate_new_entrants(survivors, generation + 1, population_size, this_gen_immigrants)

    return survivors + new_entrants
