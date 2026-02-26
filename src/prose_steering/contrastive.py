from __future__ import annotations

import torch
import torch.nn.functional as F

from prose_steering.hf import decode_new_tokens, format_dialogue


@torch.no_grad()
def generate_contrastive(
    model,
    tokenizer,
    device,
    stop_ids: set[int],
    system_pos: str,
    system_neg: str,
    user_question: str,
    beta: float = 1.0,
    max_new_tokens: int = 180,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 200,
    clamp: float = 10.0,
) -> str:
    prompt_pos = format_dialogue(tokenizer, system_pos, user_question)
    prompt_neg = format_dialogue(tokenizer, system_neg, user_question)

    ids_pos = tokenizer(prompt_pos, return_tensors="pt").input_ids.to(device)
    ids_neg = tokenizer(prompt_neg, return_tensors="pt").input_ids.to(device)

    past_pos = None
    past_neg = None
    gen_pos = ids_pos
    gen_neg = ids_neg

    for _ in range(max_new_tokens):
        out_pos = model(
            input_ids=gen_pos[:, -1:] if past_pos is not None else gen_pos,
            past_key_values=past_pos,
            use_cache=True,
            return_dict=True,
        )
        out_neg = model(
            input_ids=gen_neg[:, -1:] if past_neg is not None else gen_neg,
            past_key_values=past_neg,
            use_cache=True,
            return_dict=True,
        )
        past_pos = out_pos.past_key_values
        past_neg = out_neg.past_key_values

        logits_pos = out_pos.logits[:, -1, :][0]
        logits_neg = out_neg.logits[:, -1, :][0]

        topk_vals, topk_idx = torch.topk(logits_pos, k=top_k, dim=-1)

        lp_pos = F.log_softmax(logits_pos, dim=-1)
        lp_neg = F.log_softmax(logits_neg, dim=-1)

        delta = lp_pos[topk_idx] - beta * lp_neg[topk_idx]
        delta = torch.clamp(delta, -clamp, clamp)

        delta = delta / temperature
        probs = F.softmax(delta, dim=-1)

        sorted_probs, sorted_i = torch.sort(probs, descending=True)
        cumsum = torch.cumsum(sorted_probs, dim=-1)
        cutoff = torch.searchsorted(cumsum, torch.tensor(top_p, device=device))
        cutoff = max(1, int(cutoff.item()) + 1)

        filt_probs = sorted_probs[:cutoff]
        filt_i = sorted_i[:cutoff]
        filt_probs = filt_probs / filt_probs.sum()

        next_local = torch.multinomial(filt_probs, 1).item()
        next_id = topk_idx[filt_i[next_local]].item()

        next_tok = torch.tensor([[next_id]], device=device)
        gen_pos = torch.cat([gen_pos, next_tok], dim=1)
        gen_neg = torch.cat([gen_neg, next_tok], dim=1)

        if next_id in stop_ids:
            break

    return decode_new_tokens(tokenizer, ids_pos, gen_pos)