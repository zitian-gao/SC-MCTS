import pickle
from typing import Type, Callable, Optional
import numpy as np
from tqdm import tqdm
from datetime import datetime
from reasoners import Reasoner, SearchAlgorithm
from reasoners.blocksworld import BWEvaluator
from reasoners.mcts import MCTS
from reasoners.world_model import BlocksWorldModel
from reasoners.search_config import BWConfig
from reasoners.exllamav2_model import ExLlamaModelV2
import argparse
import os
import json
import random
import torch
import time


def RAP_bw(
    base_model,
    search_algo: Type[SearchAlgorithm] = MCTS,
    data_path: str = "data",
    reward_model: str = "logll",
    prompt_path: str = "data",
    mode: str = "sc",
    start_idx: int = 0,
    end_idx: int = 1,
    depth_limit: int = 6,
    reward_alpha: float = 0.5,
    batch_size=1,
    goal_reached_reward=100,
    goal_reward_default=0.0,
    w_exp=1.0,
    simulate_strategy="sample",
    cum_reward: Callable[[list[float]], float] = sum,
    calc_q: Callable[[list[float]], float] = np.mean,
    log_dir: Optional[str] = None,
    disable_log: bool = False,
    pickle_path=None,
    **search_algo_params,
):

    with open(prompt_path) as f:
        prompt = json.load(f)

    config_file = "blocksworld/data/bw_config.yaml"
    domain_file = "blocksworld/data/generated_domain.pddl"

    search_algo_params |= {
        "cum_reward": cum_reward,
        "w_exp": w_exp,
        "mode": mode,
        "simulate_strategy": simulate_strategy,
        "calc_q": calc_q,
        "depth_limit": depth_limit,
        "disable_tqdm": False,
    }

    world_model = BlocksWorldModel(
        base_model=base_model,
        prompt=prompt,
        batch_size=batch_size,
        max_steps=depth_limit,
    )

    config = BWConfig(base_model=base_model,
                      prompt=prompt,
                      mode=mode,
                      batch_size=batch_size,
                      reward_alpha=reward_alpha,
                      goal_reached_reward=goal_reached_reward,
                      goal_reward_default=goal_reward_default,
                      reward_model=reward_model,
                      pickle_path=pickle_path)

    search_algo = search_algo(**search_algo_params)

    reasoner = Reasoner(world_model=world_model,
                        search_config=config,
                        search_algo=search_algo)

    evaluator = BWEvaluator(
        config_file=config_file,
        domain_file=domain_file,
        data_path=data_path,
        init_prompt=prompt,
        disable_log=disable_log,
    )

    accuracy = evaluator.evaluate(reasoner,
                                  shuffle_prompt=True,
                                  num_shot=4,
                                  start_idx=start_idx,
                                  end_idx=end_idx,
                                  log_dir=log_dir,
                                  pickle_path=pickle_path)

    print(f'accuracy: {accuracy:.4f}')


if __name__ == "__main__":
    start_time = time.time()

    parser = argparse.ArgumentParser(description="Run RAP Inference")
    parser.add_argument("--llama_path", type=str, required=True)
    parser.add_argument("--depth_limit", type=int, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--prompt_path", type=str, required=True)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=0)

    parser.add_argument("--exllama_version", type=int, default=1)
    parser.add_argument("--reward_model", type=str, default="logll")
    parser.add_argument("--draft_model_path", type=str, default=None)
    parser.add_argument("--pickle_path", type=str, default=None)
    parser.add_argument("--w_exp", type=float, default=1.0)
    parser.add_argument("--mode", type=str, default="sc")
    parser.add_argument("--simulate_strategy", type=str, default="max")

    args = parser.parse_args()
    device = torch.device('cuda')
    np.random.seed(1)
    random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    torch.backends.cudnn.deterministic = True

    if args.exllama_version == 1:
        llama_model = ExLlamaModel(model_path=args.llama_path,
                                   draft_model_path=args.draft_model_path,
                                   lora_dir=None,
                                   device="cuda",
                                   max_batch_size=1,
                                   max_new_tokens=200,
                                   max_seq_len=16384)

    elif args.exllama_version == 2:
        llama_model = ExLlamaModelV2(model_path=args.llama_path,
                                     draft_model_path=args.draft_model_path,
                                     lora_dir=None,
                                     device="cuda",
                                     max_batch_size=1,
                                     max_new_tokens=200,
                                     max_seq_len=16384)

    RAP_bw(
        llama_model,
        data_path=args.data_path,
        prompt_path=args.prompt_path,
        start_idx=args.start_idx,
        depth_limit=args.depth_limit,
        reward_model=args.reward_model,
        pickle_path=args.pickle_path,
        w_exp=args.w_exp,
        mode=args.mode,
        simulate_strategy=args.simulate_strategy,
    )

    print(f"运行时间: {time.time()-start_time:.2f} 秒")
