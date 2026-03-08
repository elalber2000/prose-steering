from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model_and_tokenizer(
    model_name: str,
    torch_dtype: torch.dtype = torch.float16,
):
    torch.set_grad_enabled(False)

    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    mdl = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        device_map="auto",
    ).eval()

    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    device = next(mdl.parameters()).device

    im_end_id = tok.convert_tokens_to_ids("<|im_end|>")
    stop_ids = {tok.eos_token_id}
    if im_end_id is not None:
        stop_ids.add(im_end_id)

    return mdl, tok, device, stop_ids


def format_dialogue(system_prefix: str, user_question: str) -> str:
    messages = []
    if system_prefix and system_prefix.strip():
        messages.append({"role": "system", "content": system_prefix.strip()})
    messages.append({"role": "user", "content": user_question.strip()})
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def decode_new_tokens(tokenizer, prompt_ids: torch.Tensor, full_ids: torch.Tensor) -> str:
    new_ids = full_ids[0, prompt_ids.shape[1] :]
    return tokenizer.decode(new_ids, skip_special_tokens=True)


def get_decoder_layers(m: torch.nn.Module):
    base = getattr(m, "model", None) or getattr(m, "transformer", None) or m
    for attr in ["layers", "h", "decoder_layers", "decoder", "blocks"]:
        obj = getattr(base, attr, None)
        if obj is None:
            continue
        if isinstance(obj, torch.nn.ModuleList):
            return obj
        if hasattr(obj, "layers") and isinstance(obj.layers, torch.nn.ModuleList):
            return obj.layers
    if (
        hasattr(base, "decoder")
        and hasattr(base.decoder, "layers")
        and isinstance(base.decoder.layers, torch.nn.ModuleList)
    ):
        return base.decoder.layers
    raise RuntimeError("Could not locate decoder layers on this model.")