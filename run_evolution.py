import argparse
import random

from othello_sim.individual import Individual
from othello_sim.evolution import run_generation, run_title_challenge
from othello_sim.io_utils import load_checkpoint, export_population, load_training_state, save_training_state
from othello_sim.family_names import assign_initial_family_names

BENCHMARK_DEPTH_MAX = 6
BENCHMARK_DEPTH_STEP = 1
WIN_RATE_UPGRADE_THRESHOLD = 0.65
WIN_RATE_DOWNGRADE_THRESHOLD = 0.25


def compute_win_rate_vs_benchmark(matches_log, generation):
    relevant = [m for m in matches_log if m.get("opponent_type") == "benchmark" and m.get("generation") == generation]
    if not relevant:
        return None
    wins = sum(1 for m in relevant if m["result"] == "win")
    draws = sum(1 for m in relevant if m["result"] == "draw")
    return (wins + 0.5 * draws) / len(relevant)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--generations", type=int, default=3)
    parser.add_argument("--population-size", type=int, default=16)
    parser.add_argument("--swiss-rounds", type=int, default=4)
    parser.add_argument("--search-depth", type=int, default=4, help="個体の探索深さ")
    parser.add_argument("--benchmark-depth", type=int, default=5,
                         help="ベンチマーク相手の初期探索深さ。2回目以降は training_state.json の値が優先")
    parser.add_argument("--benchmark-games", type=int, default=5)
    parser.add_argument("--immigrant-count", type=int, default=1, help="毎世代必ず投入する移民の数")
    parser.add_argument("--grace-period", type=int, default=3, help="新規移民が無条件で保護される世代数")
    parser.add_argument("--reset-interval", type=int, default=10, help="恒久保証リストを実力順で引き直す世代間隔")
    args = parser.parse_args()

    all_individuals, matches_log = load_checkpoint(args.data_dir)
    training_state = load_training_state(args.data_dir)

    benchmark_depth = training_state.get("benchmark_depth") or args.benchmark_depth
    win_rate_history = training_state.get("win_rate_history", [])
    active_population_ids = training_state.get("active_population_ids")
    title_holder_defeats = training_state.get("title_holder_defeats", [])
    guaranteed_families = training_state.get("guaranteed_families", [])
    grace_info = training_state.get("grace_info", {})

    if all_individuals is None:
        random.seed()
        all_individuals = {}
        matches_log = []
        population = []
        family_names = assign_initial_family_names(args.population_size)
        for i in range(args.population_size):
            ind = Individual(ind_id=f"G0-{i:03d}", generation=0, family_label=family_names[i])
            population.append(ind)
            all_individuals[ind.id] = ind
        start_gen = 0
    else:
        start_gen = max(ind.generation for ind in all_individuals.values()) + 1
        if active_population_ids:
            population = [all_individuals[i] for i in active_population_ids if i in all_individuals]
            missing = len(active_population_ids) - len(population)
            if missing > 0:
                print(f"  ※ 前回の現役個体のうち{missing}体が見つかりませんでした")
        else:
            print("  ※ 現役個体リストが見つからないため、Elo上位から復元します")
            ranked = sorted(all_individuals.values(), key=lambda ind: ind.elo, reverse=True)
            population = ranked[:args.population_size]

    for gen in range(start_gen, start_gen + args.generations):
        print(f"=== Generation {gen} (benchmark_depth={benchmark_depth}, population={len(population)}) ===")
        population, guaranteed_families, grace_info = run_generation(
            population, gen, matches_log,
            population_size=args.population_size,
            swiss_rounds=args.swiss_rounds,
            search_depth=args.search_depth,
            benchmark_depth=benchmark_depth,
            benchmark_games=args.benchmark_games,
            immigrant_count=args.immigrant_count,
            guaranteed_families=guaranteed_families,
            grace_info=grace_info,
            grace_period=args.grace_period,
            reset_interval=args.reset_interval,
        )
        for ind in population:
            all_individuals[ind.id] = ind

        ranked_now = sorted(population, key=lambda ind: -ind.elo)
        for ind in ranked_now[:5]:
            print(f"  {ind.id} [{ind.family_label}流]: Elo={ind.elo:.1f}")

        export_population(list(all_individuals.values()), matches_log, args.data_dir)

        win_rate = compute_win_rate_vs_benchmark(matches_log, gen)
        if win_rate is not None:
            win_rate_history.append({"generation": gen, "win_rate": round(win_rate, 3), "benchmark_depth": benchmark_depth})
            print(f"  温度計勝率(首位 vs ベンチマーク, 世代{gen}): {win_rate:.1%}")

            if win_rate >= WIN_RATE_UPGRADE_THRESHOLD and benchmark_depth < BENCHMARK_DEPTH_MAX:
                benchmark_depth = min(BENCHMARK_DEPTH_MAX, benchmark_depth + BENCHMARK_DEPTH_STEP)
                print(f"  → 勝率が高いため、ベンチマークの探索深さを {benchmark_depth} に引き上げました")
            elif win_rate <= WIN_RATE_DOWNGRADE_THRESHOLD and benchmark_depth > args.benchmark_depth:
                benchmark_depth = max(args.benchmark_depth, benchmark_depth - BENCHMARK_DEPTH_STEP)
                print(f"  → 勝率が低いため、ベンチマークの探索深さを {benchmark_depth} に引き下げました")

            # --- タイトル挑戦：温度計の壁を極限まで登り切った首位だけが挑戦できる ---
            if benchmark_depth >= BENCHMARK_DEPTH_MAX and win_rate >= WIN_RATE_UPGRADE_THRESHOLD:
                champion = ranked_now[0]
                print(f"  ★ {champion.id}［{champion.family_label}流］がタイトルホルダーに挑戦します！")
                won_title, wins = run_title_challenge(champion, gen, matches_log, individual_depth=args.search_depth)
                print(f"    結果: {wins}/5勝 → {'タイトル奪取！' if won_title else '敗退'}")
                if won_title:
                    title_holder_defeats.append({
                        "individual_id": champion.id,
                        "family_label": champion.family_label,
                        "generation": gen,
                        "wins": wins,
                    })

        save_training_state(args.data_dir, {
            "benchmark_depth": benchmark_depth,
            "win_rate_history": win_rate_history,
            "active_population_ids": [ind.id for ind in population],
            "title_holder_defeats": title_holder_defeats,
            "guaranteed_families": guaranteed_families,
            "grace_info": grace_info,
        })


if __name__ == "__main__":
    main()
