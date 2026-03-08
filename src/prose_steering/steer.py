from __future__ import annotations

import torch
import math

from prose_steering.hf import decode_new_tokens, format_dialogue, get_decoder_layers
from prose_steering.utils import top_p_sample


@torch.no_grad()
def capture_layer_output_mean(model, tokenizer, device, axis, layer_idx: int, K: int = 32) -> torch.Tensor:
    layers = get_decoder_layers(model)
    layer = layers[layer_idx]

    vecs = []
    for q in axis.prompts:
        text = format_dialogue(axis_prompt_prefix, q)  # set below by closure
        inputs = tokenizer(text, return_tensors="pt").to(device)

        captured = {}

        def capture_hook(_, __, out):
            h = out[0] if isinstance(out, (tuple, list)) else out
            captured["h"] = h.detach()

        hndl = layer.register_forward_hook(capture_hook)
        try:
            _ = model(**inputs, use_cache=False, return_dict=True)
        finally:
            hndl.remove()

        h = captured["h"][0]
        h_tail = h[-K:, :].mean(dim=0)
        vecs.append(h_tail)

    return torch.stack(vecs, dim=0).mean(dim=0)


def capture_layer_rms(model, tokenizer, device, axis, layers, system_prefix: str, layer_idx: int, K: int = 32) -> float:
    """
    Returns mean RMS of hidden state at layer output over last K prompt tokens.
    RMS is computed per token as ||h||/sqrt(D), then averaged.
    """
    layer = layers[layer_idx]
    rms_vals = []

    for q in axis["prompts"]:
        text = format_dialogue(system_prefix, q)
        inputs = tokenizer(text, return_tensors="pt").to(device)

        captured = {}

        def capture_hook(module, inp, out):
            h = out[0] if isinstance(out, (tuple, list)) else out  # [B, S, D]
            captured["h"] = h.detach()

        hndl = layer.register_forward_hook(capture_hook)
        try:
            _ = model(**inputs, use_cache=False, return_dict=True)
        finally:
            hndl.remove()

        h = captured["h"][0]            # [S, D]
        h_tail = h[-K:, :]              # [K, D]
        D = h_tail.shape[-1]
        token_rms = h_tail.norm(dim=-1) / math.sqrt(D)   # [K]
        rms_vals.append(token_rms.mean().item())

    return float(sum(rms_vals) / len(rms_vals))


@torch.no_grad()
def compute_steer_vector(model, tokenizer, device, axis, layer_idx: int, K: int = 32) -> torch.Tensor:
    layers = get_decoder_layers(model)
    n_layers = len(layers)
    if not (0 <= layer_idx < n_layers):
        raise ValueError(f"layer_idx {layer_idx} out of range [0, {n_layers-1}]")

    global axis_prompt_prefix

    axis_prompt_prefix = axis.positive_dir
    pos_mean = capture_layer_output_mean(model, tokenizer, device, axis, layer_idx, K=K)

    axis_prompt_prefix = axis.negative_dir
    neg_mean = capture_layer_output_mean(model, tokenizer, device, axis, layer_idx, K=K)

    steer = pos_mean - neg_mean
    steer = steer / (steer.norm() + 1e-8)
    return steer


@torch.no_grad()
def generate_midlayer_steered(
    model,
    tokenizer,
    device,
    stop_ids: set[int],
    steer: torch.Tensor,
    system_prefix: str,
    user_question: str,
    alpha: float,
    layer_idx: int,
    max_new_tokens: int = 180,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> str:
    prompt = format_dialogue(system_prefix, user_question)
    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    generated = prompt_ids

    layers = get_decoder_layers(model)
    layer = layers[layer_idx]

    steer_vec = steer.to(dtype=next(model.parameters()).dtype)

    def steer_hook(module, inp, out):
        h = out[0] if isinstance(out, (tuple, list)) else out
        sv = steer_vec.to(device=h.device, dtype=h.dtype).view(1, 1, -1)
        return ((h + alpha * sv),) + tuple(out[1:]) if isinstance(out, (tuple, list)) else (h + alpha * sv)

    hndl = layer.register_forward_hook(steer_hook)
    try:
        out = model(input_ids=prompt_ids, use_cache=True, return_dict=True)
        past = out.past_key_values
        logits = out.logits[:, -1, :]

        for _ in range(max_new_tokens):
            next_id = top_p_sample(logits, temperature=temperature, top_p=top_p)
            next_tok = torch.tensor([[next_id]], device=prompt_ids.device)
            generated = torch.cat([generated, next_tok], dim=1)

            if next_id in stop_ids:
                break

            out = model(input_ids=next_tok, past_key_values=past, use_cache=True, return_dict=True)
            past = out.past_key_values
            logits = out.logits[:, -1, :]
    finally:
        hndl.remove()

    return decode_new_tokens(prompt_ids, generated)