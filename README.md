## Install Conda Environment
```bash
conda create -n reasoners python=3.10 -y

conda activate reasoners

pip install exllamav2 torch scikit-build fschat pddl matplotlib protobuf accelerate sentencepiece torch requests psutil numpy transformers shortuuid accelerate optimum

git clone https://github.com/PanQiWei/AutoGPTQ.git && cd AutoGPTQ && MAX_JOBS=40 pip install -vvv --no-build-isolation -e .

git clone https://github.com/Dao-AILab/flash-attention.git && cd flash-attention && MAX_JOBS=40 pip install -e . --no-build-isolation
```


## Install Reasoners
Ref: https://github.com/maitrix-org/llm-reasoners
```bash
pip install -e .
```


## Code Index
Bash script for the blocksworld MCTS experiment:
[mcts.sh](scripts/blocksworld/mcts.sh)


ExllamaV2 inference framework:
[exllamav2_model.py](reasoners/exllamav2_model.py)


MCTS experiment framework:
[mcts_inference.py](scripts/mcts_inference.py)


Main program for MCTS:
[mcts.py](reasoners/mcts.py)


MCTS search configuration:
[search_config.py](reasoners/search_config.py)


Blocksworld step validator, state updates, etc.:
[world_model.py](reasoners/world_model.py)


Blocksworld result validator, ICL prompt construction, etc.:
[blocksworld.py](reasoners/blocksworld.py)


Blocksworld action and plan extractor, etc.:
[bw_utils.py](reasoners/bw_utils.py)


Experiment dataset control flow:
[base.py](reasoners/base.py)


## Acknowledgement
In our code, we referenced some implementation from llm-reasoners (URL: https://github.com/maitrix-org/llm-reasoners). We are very grateful for their outstanding contributions!


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