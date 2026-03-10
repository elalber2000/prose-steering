import json
import random
import math
from collections import defaultdict

K = 32
TOL = 2.0
WINDOW = 200
MAX_MATCHES = 10000


def load_samples(path):
    data = json.load(open(path))

    samples = []

    for alpha, texts in data["steering"].items():
        for t in texts:
            samples.append((f"steer_{alpha}", t))

    for beta, texts in data["spec_dec"].items():
        for t in texts:
            samples.append((f"spec_{beta}", t))

    return samples


def expected(a, b):
    return 1 / (1 + 10 ** ((b - a) / 400))


def update(rA, rB, scoreA):
    eA = expected(rA, rB)
    eB = expected(rB, rA)

    rA += K * (scoreA - eA)
    rB += K * ((1 - scoreA) - eB)

    return rA, rB


def judge(a, b):
    print("\nA:\n", a)
    print("\nB:\n", b)
    x = input("\nWhich is better? (a/b/t): ").strip().lower()
    if x == "a":
        return 1
    if x == "b":
        return 0
    return 0.5


def elo_sample(path):

    samples = load_samples(path)

    ratings = defaultdict(lambda: 1000.0)

    history = []

    for step in range(MAX_MATCHES):

        (mA, tA), (mB, tB) = random.sample(samples, 2)

        scoreA = judge(tA, tB)

        ratings[mA], ratings[mB] = update(ratings[mA], ratings[mB], scoreA)

        history.append(dict(ratings))

        if step > WINDOW:

            delta = 0
            prev = history[-WINDOW]

            for k in ratings:
                delta += abs(ratings[k] - prev.get(k, 1000))

            delta /= len(ratings)

            if delta < TOL:
                print("\nConverged after", step, "matches")
                break

    print("\nFinal Elo:\n")

    for k, v in sorted(ratings.items(), key=lambda x: -x[1]):
        print(f"{k:12} {v:.1f}")


if __name__ == "__main__":
    elo_sample("dataset.json")