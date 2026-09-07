import random
import string

ALPHABET = string.ascii_uppercase


def _index_to_label(i):
    label = ""
    i += 1
    while i > 0:
        i, rem = divmod(i - 1, 26)
        label = ALPHABET[rem] + label
    return label


def assign_initial_family_names(count):
    """初期個体数分、A, B, C... の順に別々の流派名を割り当てる（26を超えたらAA, AB...）"""
    return [_index_to_label(i) for i in range(count)]


def random_immigrant_family_name():
    first = random.choice(ALPHABET)
    second = random.choice(ALPHABET).lower()
    return f"{first}{second}"
