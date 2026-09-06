"""
オセロの評価関数とミニマックス（アルファベータ枝刈り）探索。

評価関数は「個性パラメータ」で重み付けされた複数の指標の合計：
  - 角の確保（corner_weight）
  - 着手可能数＝機動力（mobility_weight）
  - 安定石の簡易近似（stability_weight）
  - 石数（disc_weight）※終盤ほど重要度が上がる

序盤・中盤・終盤で重みの掛け方を変えることで、
「終盤の粘り強さ」のような個性も表現できるようにしてある。
"""
import math
import random

from . import board as B

CORNERS = {0, 7, 56, 63}
X_SQUARES = {9, 14, 49, 54}       # 角の斜め隣（危険マス）
C_SQUARES = {1, 6, 8, 15, 48, 55, 57, 62}  # 角の辺隣（やや危険なマス）

# 手の並び替え用の静的なマス評価（角を高く、危険マスを低く）。
# 探索そのものの評価には使わず、アルファベータの枝刈り効率を上げるためだけに使う。
_STATIC_WEIGHTS = [0] * 64
for _sq in CORNERS:
    _STATIC_WEIGHTS[_sq] = 100
for _sq in X_SQUARES:
    _STATIC_WEIGHTS[_sq] = -50
for _sq in C_SQUARES:
    _STATIC_WEIGHTS[_sq] = -20


def _order_moves(moves):
    """良さそうな手を先に並べる（枝刈り効率を上げるためだけの処理）"""
    return sorted(moves, key=lambda mv: -_STATIC_WEIGHTS[mv])


def evaluate(bd, color, params):
    """color視点での評価値（正が有利）。paramsは個性パラメータの辞書"""
    opp = B.opponent(color)
    phase = B.game_phase(bd)

    # --- 角の確保 ---
    my_corners = sum(1 for c in CORNERS if bd[c] == color)
    opp_corners = sum(1 for c in CORNERS if bd[c] == opp)
    corner_score = (my_corners - opp_corners) * 25

    # --- 危険マス（角の隣、まだ角が取られていない場合はペナルティ）---
    danger_score = 0
    for sq in X_SQUARES:
        if bd[sq] == color:
            danger_score -= 1
        elif bd[sq] == opp:
            danger_score += 1
    for sq in C_SQUARES:
        if bd[sq] == color:
            danger_score -= 0.5
        elif bd[sq] == opp:
            danger_score += 0.5

    # --- 機動力（着手可能数の差） ---
    my_moves = len(B.legal_moves(bd, color))
    opp_moves = len(B.legal_moves(bd, opp))
    total_moves = my_moves + opp_moves
    mobility_score = 0
    if total_moves > 0:
        mobility_score = 100 * (my_moves - opp_moves) / total_moves

    # --- 石数（終盤ほど重要） ---
    black, white = B.count_discs(bd)
    my_discs = black if color == B.BLACK else white
    opp_discs = white if color == B.BLACK else black
    total_discs = my_discs + opp_discs
    disc_score = 0
    if total_discs > 0:
        disc_score = 100 * (my_discs - opp_discs) / total_discs

    # --- フェーズごとの重み調整 ---
    phase_disc_multiplier = {"opening": 0.1, "midgame": 0.4, "endgame": 1.5}[phase]

    score = (
        params["corner_weight"] * corner_score
        + params["mobility_weight"] * mobility_score
        + params["stability_weight"] * danger_score
        + params["disc_weight"] * disc_score * phase_disc_multiplier
    )
    return score


def minimax(bd, color, depth, alpha, beta, maximizing_color, params):
    moves = B.legal_moves(bd, color)

    if depth == 0:
        return evaluate(bd, maximizing_color, params), None

    if not moves:
        opp_moves = B.legal_moves(bd, B.opponent(color))
        if not opp_moves:
            return evaluate(bd, maximizing_color, params), None  # 両者とも打てず終局
        value, _ = minimax(bd, B.opponent(color), depth - 1, alpha, beta, maximizing_color, params)
        return value, None

    moves = _order_moves(moves)
    best_move = None
    is_maximizing = (color == maximizing_color)

    if is_maximizing:
        value = -math.inf
        for mv in moves:
            new_bd = bd.copy()
            B.apply_move(new_bd, mv, color)
            child_value, _ = minimax(new_bd, B.opponent(color), depth - 1, alpha, beta, maximizing_color, params)
            if child_value > value:
                value, best_move = child_value, mv
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        return value, best_move
    else:
        value = math.inf
        for mv in moves:
            new_bd = bd.copy()
            B.apply_move(new_bd, mv, color)
            child_value, _ = minimax(new_bd, B.opponent(color), depth - 1, alpha, beta, maximizing_color, params)
            if child_value < value:
                value, best_move = child_value, mv
            beta = min(beta, value)
            if alpha >= beta:
                break
        return value, best_move


def choose_move(bd, color, params, depth=5):
    """個体のパラメータを使って1手選ぶ"""
    moves = B.legal_moves(bd, color)
    if not moves:
        return None
    _, move = minimax(bd, color, depth, -math.inf, math.inf, color, params)
    if move is None:
        move = random.choice(moves)
    return move
