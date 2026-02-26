from __future__ import annotations

from dataclasses import dataclass

from prose_steering.utils import load_json


@dataclass(frozen=True)
class Axis:
    feature: str
    positive_dir: str
    negative_dir: str
    prompts: list[str]


def load_axis(path: str) -> Axis:
    d = load_json(path)
    return Axis(
        feature=d["feature"],
        positive_dir=d["positive_dir"],
        negative_dir=d["negative_dir"],
        prompts=list(d["prompts"]),
    )