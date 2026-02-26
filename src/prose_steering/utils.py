import json
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn.functional as F


def load_json(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8"))


def save_tensor(path: str | Path, t: torch.Tensor) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(t.detach().cpu(), p)


def load_tensor(path: str | Path, device: torch.device) -> torch.Tensor:
    t = torch.load(Path(path), map_location="cpu")
    return t.to(device)


def top_p_sample(logits: torch.Tensor, temperature: float = 0.8, top_p: float = 0.9) -> int:
    if logits.dim() == 2:
        logits = logits[0]
    if temperature <= 0:
        return int(torch.argmax(logits).item())

    logits = logits / temperature
    probs = F.softmax(logits, dim=-1)

    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cumsum = torch.cumsum(sorted_probs, dim=-1)

    cutoff = torch.searchsorted(cumsum, torch.tensor(top_p, device=logits.device))
    cutoff = max(1, int(cutoff.item()) + 1)

    filtered_probs = sorted_probs[:cutoff]
    filtered_idx = sorted_idx[:cutoff]
    filtered_probs = filtered_probs / filtered_probs.sum()

    next_local = torch.multinomial(filtered_probs, num_samples=1)
    next_id = filtered_idx[next_local].item()
    return int(next_id)