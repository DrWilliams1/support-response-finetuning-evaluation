# Sources and license scope

This project's SFT-to-preference-tuning structure was informed by Chapter 12, “Fine-tuning Generation Models,” of *Hands-On Large Language Models* by Jay Alammar and Maarten Grootendorst (O'Reilly, 2024). See the [official repository and Chapter 12 notebook](https://github.com/HandsOnLLM/Hands-On-Large-Language-Models/tree/main/chapter12). The uploaded source experiment records completed training; its logs are not results for this new support benchmark.

Method and model sources:

- Hu et al., [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685), 2021.
- Dettmers et al., [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314), 2023.
- Rafailov et al., [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290), 2023.
- TinyLlama team, [TinyLlama-1.1B-Chat-v1.0](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0), pinned to [revision fe8a4ea](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0/tree/fe8a4ea1ffedaf415f4da2f062534de366a451e6).

The MIT license in this repository covers its original code, documentation, and synthetic dataset. It does not relicense model weights, third-party dependencies, or upstream materials. TinyLlama and the book's official code repository carry their own Apache-2.0 licenses. Model weights and the source PDF are not distributed in this project. The support cases describe a fictional store and contain no customer records.
