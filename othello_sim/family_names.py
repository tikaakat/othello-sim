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


def random_immigrant_family_name():
    first = random.choice(ALPHABET)
    second = random.choice(ALPHABET).lower()
    return f"{first}{second}"
