# Interpretable Contrastive Monte Carlo Tree Search Reasoning

<a href="https://arxiv.org/abs/2410.01707" alt="arXiv">
    <img src="https://img.shields.io/badge/arXiv-2410.01707-b31b1b.svg?style=flat" /></a>


## Install Environment
```bash
conda create -n mcts python=3.10 -y
conda activate mcts
pip install -e .
```

## Code Index
```plaintext
SC_MCTS/
├── LLMs-Planning/
├── blocksworld/
│   ├── data/
│   │   ├── split_v1/           # Hard mode blocksworld dataset
│   │   └── split_v2/           # Easy mode blocksworld dataset
│   └── prompts/
├── reasoners/
│   ├── base.py                 # Experiment dataset control flow.
│   ├── blocksworld.py          # Blocksworld result validator, ICL prompt construction, etc.
│   ├── bw_utils.py             # Blocksworld action and plan extractor, etc.
│   ├── exllamav2_model.py      # ExllamaV2 inference framework.
│   ├── mcts.py                 # Main program for MCTS.
│   ├── search_config.py        # MCTS search configuration.
│   └── world_model.py          # Blocksworld step validator, state updates, etc.
├── scripts/
│   └── mcts_inference.py       # MCTS experiment framework.
│   └── mcts.sh                 # MCTS blocksworld experiment script.
│   └── cot_inference.py        # CoT blocksworld experiment framework.
│   └── cot.sh                  # CoT blocksworld experiment script.
├── README.md
├── setup.py
└── requirements.txt
```


## Acknowledgement
In our code we referenced some implementation from [llm-reasoners](https://github.com/maitrix-org/llm-reasoners). We are very grateful for their outstanding contributions!


### Citation
```
@misc{gao2024interpretablecontrastivemontecarlo,
      title={Interpretable Contrastive Monte Carlo Tree Search Reasoning}, 
      author={Zitian Gao and Boye Niu and Xuzheng He and Haotian Xu and Hongzhang Liu and Aiwei Liu and Xuming Hu and Lijie Wen},
      year={2024},
      eprint={2410.01707},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2410.01707}, 
}
```