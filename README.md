# Group 13 - Robot Instruction Ambiguity Detection

## Project Overview

This project investigates whether Large Language Models (LLMs) can identify ambiguous instructions in robotic environments.

We use the AmbiK dataset and Qwen Plus to compare three prompting strategies:

- Zero-shot Prompting
- Few-shot Prompting
- Checklist Prompting

The model classifies each instruction as either `CLEAR` or `NEEDS_CLARIFICATION`.

## Team Members

| Member | Responsibilities |
|---|---|
| CAI YU | Data preparation and prompt construction |
| CAO YUNFEI | Model API integration and experiment execution |
| WANG HONGFENG | Evaluation, visualization and case analysis |

## Experimental Design

- Dataset: AmbiK
- Model: Qwen Plus (fixed version, non-thinking mode)
- Test samples: 200 instructions (100 pairs)
- Prompting strategies: 3
- Total experimental requests: 600
- Evaluation: Accuracy, Macro-F1, missed ambiguity rate, unnecessary clarification rate, pairwise accuracy and valid output rate

## Project Structure

```text
Group13-Robot-Ambiguity-Detection/
├── data/
├── prompts/
├── configs/
├── src/
│   ├── data_prepare.py
│   ├── prompts.py
│   ├── model_client.py
│   ├── run_experiment.py
│   ├── evaluate.py
│   └── analysis.py
├── results/
├── figures/
├── report/
├── requirements.txt
└── README.md
```

## Project Status

In development. Experimental results have not yet been generated.