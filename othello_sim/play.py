from . import board as B
from . import engine as E

# ベンチマーク（温度計）用の固定パラメータ：バランス型の手堅い設定
BENCHMARK_PARAMS = {
    "corner_weight": 3.0,
    "danger_zone_weight": 2.0,
    "mobility_weight": 2.0,
    "edge_stability_weight": 2.0,
    "frontier_weight": 1.5,
    "disc_weight": 1.0,
    "parity_weight": 1.0,
    "center_weight": 1.0,
}


def _play(params_black, params_white, depth_black, depth_white, max_moves=64):
    """1局対局し、(黒石数, 白石数, 着手履歴) を返す"""
    bd = B.initial_board()
    color = B.BLACK
    move_history = []

    for _ in range(max_moves):
        if B.is_game_over(bd):
            break
        moves = B.legal_moves(bd, color)
        if not moves:
            color = B.opponent(color)
            continue

        params = params_black if color == B.BLACK else params_white
        depth = depth_black if color == B.BLACK else depth_white
        mv = E.choose_move(bd, color, params, depth=depth)
        B.apply_move(bd, mv, color)
        move_history.append({"pos": mv, "color": color})
        color = B.opponent(color)

    black, white = B.count_discs(bd)
    return black, white, move_history


def _outcome_from_score(my_score, opp_score):
    if my_score > opp_score:
        return "win"
    elif my_score < opp_score:
        return "loss"
    return "draw"


def play_individual_vs_individual(ind_a, ind_b, depth=4):
    """個体同士の対局。ind_a=黒、ind_b=白で固定"""
    black, white, moves = _play(ind_a.params, ind_b.params, depth, depth)
    outcome_a = _outcome_from_score(black, white)
    return outcome_a, moves, black, white


def play_vs_benchmark(individual, individual_depth=4, benchmark_depth=5):
    """個体 vs ベンチマーク（温度計）。個体側の色はランダム"""
    import random
    individual_color = random.choice([B.BLACK, B.WHITE])

    if individual_color == B.BLACK:
        black, white, moves = _play(individual.params, BENCHMARK_PARAMS, individual_depth, benchmark_depth)
        my_score, opp_score = black, white
    else:
        black, white, moves = _play(BENCHMARK_PARAMS, individual.params, benchmark_depth, individual_depth)
        my_score, opp_score = white, black

    outcome = _outcome_from_score(my_score, opp_score)
    color_str = "black" if individual_color == B.BLACK else "white"
    return outcome, moves, color_str, black, white
