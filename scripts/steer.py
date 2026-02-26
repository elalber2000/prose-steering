from itertools import product

import torch

from prose_steering.axis import load_axis
from prose_steering.config import config
from prose_steering.contrastive import generate_contrastive
from prose_steering.hf import get_decoder_layers, load_model_and_tokenizer
from prose_steering.steer import compute_steer_vector, generate_midlayer_steered


torch.set_grad_enabled(False)


if __name__ == "__main__":
    config = config()

    axis = load_axis("dataset/high_prose.json")

    model, tokenizer, device, stop_ids = load_model_and_tokenizer(config.default_model)

    layers = get_decoder_layers(model)
    n_layers = len(layers)
    hook_layer_idx = n_layers // 2

    steer = compute_steer_vector(
        model=model,
        tokenizer=tokenizer,
        device=device,
        axis=axis,
        layer_idx=hook_layer_idx,
        K=32,
    )

    neutral_sys_prompt = config.default_neutral_prompt
    test_question = "Write exactly 5 lines. Each line must be 6-12 words. No title."

    print(f"Using layer {hook_layer_idx} / {n_layers-1} for mid-layer steering.")

    print("\n=== Baseline (no steering) ===")
    print(
        generate_midlayer_steered(
            model=model,
            tokenizer=tokenizer,
            device=device,
            stop_ids=stop_ids,
            steer=steer,
            system_prefix=neutral_sys_prompt,
            user_question=test_question,
            alpha=0.0,
            layer_idx=hook_layer_idx,
            temperature=config.temperature,
            top_p=config.top_p,
        )
    )

    for alpha, i in product([2, 4, 9], range(3)):
        beta = alpha / 2

        print(f"\n=== Mid-layer steered toward {axis.feature} (alpha=+{alpha}) ===")
        print(
            generate_midlayer_steered(
                model=model,
                tokenizer=tokenizer,
                device=device,
                stop_ids=stop_ids,
                steer=steer,
                system_prefix=neutral_sys_prompt,
                user_question=test_question,
                alpha=+alpha,
                layer_idx=hook_layer_idx,
                temperature=config.temperature,
                top_p=config.top_p,
            )
        )

        print(f"\n=== Mid-layer steered away from {axis.feature} (alpha=-{alpha}) ===")
        print(
            generate_midlayer_steered(
                model=model,
                tokenizer=tokenizer,
                device=device,
                stop_ids=stop_ids,
                steer=steer,
                system_prefix=neutral_sys_prompt,
                user_question=test_question,
                alpha=-alpha,
                layer_idx=hook_layer_idx,
                temperature=config.temperature,
                top_p=config.top_p,
            )
        )

        if i > 0:
            continue

        print(f"\n=== Contrastive decoding (pos vs neg, beta={beta}) ===")
        print(
            generate_contrastive(
                model=model,
                tokenizer=tokenizer,
                device=device,
                stop_ids=stop_ids,
                system_pos=axis.positive_dir,
                system_neg=axis.negative_dir,
                user_question=test_question,
                beta=beta,
                temperature=config.temperature,
                top_p=config.top_p,
                top_k=config.top_k,
                clamp=config.clamp,
            )
        )
