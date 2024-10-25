#!/bin/bash

export VAL="LLMs-Planning/planner_tools/VAL"

run_blocksworld() {
    local gpu=""
    local step=""
    local reward_model=""
    local llama_path=""
    local draft_model_path=""
    local version=2             # 默认为2，2代表blocksworld的简单模式，1代表困难模式
    local exllama_version=2
    local start_idx=""
    local mode="sc"

    for arg in "$@"
    do
    case $arg in
        --gpu=*)
        gpu="${arg#*=}"
        shift
        ;;
        --step=*)
        step="${arg#*=}"
        shift
        ;;
        --reward_model=*)
        reward_model="${arg#*=}"
        shift
        ;;
        --mode=*)
        mode="${arg#*=}"
        shift
        ;;
        --llama_path=*)
        llama_path="${arg#*=}"
        shift
        ;;
        --draft_model_path=*)
        draft_model_path="${arg#*=}"
        shift
        ;;
        --version=*)
        version="${arg#*=}"
        shift
        ;;
        --exllama_version=*)
        exllama_version="${arg#*=}"
        shift
        ;;
        --start_idx=*)
        start_idx="${arg#*=}"
        shift
        ;;
        *)
        echo "Unknown option: ${arg}"
        return 1
        ;;
    esac
    done

    if [ -z "$gpu" ] || [ -z "$step" ] || [ -z "$reward_model" ] || [ -z "$llama_path" ] || [ -z "$draft_model_path" ]; then
        echo "Error: Missing required parameters."
        echo "Usage: run_blocksworld --gpu= --step= --reward_model= --llama_path= --draft_model_path= [--version=] [--exllama_version=] [--start_idx=]"
        return 1
    fi

    local model_name=$(echo "$llama_path" | sed -n 's|.*hy-tmp/\(.*\)|\1|p')
    local log_name="${model_name}/v${version}_step${step}_${reward_model}"

    mkdir -p log/${model_name}
    mkdir -p pickles/blocksworld/rap/${model_name}

    local prompt_path=""
    if [ "$version" -eq 1 ]; then
        prompt_path="blocksworld/prompts/pool_prompt_v1.json"
    else
        prompt_path="blocksworld/prompts/pool_prompt_v${version}_step_${step}.json"
    fi

    export CUDA_VISIBLE_DEVICES=$gpu

    local log_file="log/${log_name}.log"

    if [ -n "$start_idx" ]; then
        python scripts/mcts_inference.py --llama_path "$llama_path" \
                                                     --reward_model "$reward_model" \
                                                     --mode "$mode" \
                                                     --draft_model_path "$draft_model_path" \
                                                     --depth_limit "$step" \
                                                     --exllama_version "$exllama_version" \
                                                     --data_path "blocksworld/data/split_v${version}/split_v${version}_step_${step}_data.json" \
                                                     --prompt_path "$prompt_path" \
                                                     --pickle_path "pickles/blocksworld/rap/${model_name}/v${version}/step${step}/${reward_model}" \
                                                     --start_idx "$start_idx" \
                                                     >> "$log_file" 2>&1 &
    else
        python scripts/mcts_inference.py --llama_path "$llama_path" \
                                                     --reward_model "$reward_model" \
                                                     --mode "$mode" \
                                                     --draft_model_path "$draft_model_path" \
                                                     --depth_limit "$step" \
                                                     --exllama_version "$exllama_version" \
                                                     --data_path "blocksworld/data/split_v${version}/split_v${version}_step_${step}_data.json" \
                                                     --prompt_path "$prompt_path" \
                                                     --pickle_path "pickles/blocksworld/rap/${model_name}/v${version}/step${step}/${reward_model}" \
                                                     > "$log_file" 2>&1 &
    fi

    echo "$log_file"
}




################################################ llama3-70b ##################################################################
llama_path="/home/gzt/models/Llama-3.2-3B-Instruct"
draft_model_path="/home/gzt/models/Llama-3.2-1B-Instruct"

# v2
version=2
step=2
run_blocksworld --gpu=0 \
                --reward_model="sc_mcts" \
                --mode="sc" \
                --step=$step \
                --version=$version \
                --llama_path=$llama_path \
                --draft_model_path=$draft_model_path

# step=4
# run_blocksworld --gpu=1 \
#                 --reward_model="sc_mcts" \
#                 --mode="sc" \
#                 --step=$step \
#                 --version=$version \
#                 --llama_path=$llama_path \
#                 --draft_model_path=$draft_model_path

# step=6
# run_blocksworld --gpu=2 \
#                 --reward_model="sc_mcts" \
#                 --mode="sc" \
#                 --step=$step \
#                 --version=$version \
#                 --llama_path=$llama_path \
#                 --draft_model_path=$draft_model_path

# step=8
# run_blocksworld --gpu=3 \
#                 --reward_model="sc_mcts" \
#                 --mode="sc" \
#                 --step=$step \
#                 --version=$version \
#                 --llama_path=$llama_path \
#                 --draft_model_path=$draft_model_path

# step=10
# run_blocksworld --gpu=4 \
#                 --reward_model="sc_mcts" \
#                 --mode="sc" \
#                 --step=$step \
#                 --version=$version \
#                 --llama_path=$llama_path \
#                 --draft_model_path=$draft_model_path

# step=12
# run_blocksworld --gpu=5 \
#                 --reward_model="sc_mcts" \
#                 --mode="sc" \
#                 --step=$step \
#                 --version=$version \
#                 --llama_path=$llama_path \
#                 --draft_model_path=$draft_model_path


# # v1
# version=1

# step=2
# run_blocksworld --gpu=6 \
#                 --reward_model="sc_mcts" \
#                 --mode="sc" \
#                 --step=$step \
#                 --version=$version \
#                 --llama_path=$llama_path \
#                 --draft_model_path=$draft_model_path

# step=4
# run_blocksworld --gpu=7 \
#                 --reward_model="sc_mcts" \
#                 --mode="sc" \
#                 --step=$step \
#                 --version=$version \
#                 --llama_path=$llama_path \
#                 --draft_model_path=$draft_model_path

# step=6
# run_blocksworld --gpu=8 \
#                 --reward_model="sc_mcts" \
#                 --mode="sc" \
#                 --step=$step \
#                 --version=$version \
#                 --llama_path=$llama_path \
#                 --draft_model_path=$draft_model_path

# step=8
# run_blocksworld --gpu=9 \
#                 --reward_model="sc_mcts" \
#                 --mode="sc" \
#                 --step=$step \
#                 --version=$version \
#                 --llama_path=$llama_path \
#                 --draft_model_path=$draft_model_path

# step=10
# run_blocksworld --gpu=10 \
#                 --reward_model="sc_mcts" \
#                 --mode="sc" \
#                 --step=$step \
#                 --version=$version \
#                 --llama_path=$llama_path \
#                 --draft_model_path=$draft_model_path

# step=12
# run_blocksworld --gpu=11 \
#                 --reward_model="sc_mcts" \
#                 --mode="sc" \
#                 --step=$step \
#                 --version=$version \
#                 --llama_path=$llama_path \
#                 --draft_model_path=$draft_model_path
# ################################################ llama3-70b ##################################################################









# ################################################ llama3.1-70b ##################################################################
# llama_path="/hy-tmp/Meta-Llama-3.1-70B-Instruct-GPTQ-INT4"
# draft_model_path="/hy-tmp/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4"

# # v2
# version=2
# step=2
# run_blocksworld --gpu=12 \
#                 --reward_model="sc_mcts" \
#                 --mode="sc" \
#                 --step=$step \
#                 --version=$version \
#                 --llama_path=$llama_path \
#                 --draft_model_path=$draft_model_path

# step=4
# run_blocksworld --gpu=13 \
#                 --reward_model="sc_mcts" \
#                 --mode="sc" \
#                 --step=$step \
#                 --version=$version \
#                 --llama_path=$llama_path \
#                 --draft_model_path=$draft_model_path

# step=6
# run_blocksworld --gpu=14 \
#                 --reward_model="sc_mcts" \
#                 --mode="sc" \
#                 --step=$step \
#                 --version=$version \
#                 --llama_path=$llama_path \
#                 --draft_model_path=$draft_model_path

# step=8
# run_blocksworld --gpu=15 \
#                 --reward_model="sc_mcts" \
#                 --mode="sc" \
#                 --step=$step \
#                 --version=$version \
#                 --llama_path=$llama_path \
#                 --draft_model_path=$draft_model_path

# step=10
# run_blocksworld --gpu=16 \
#                 --reward_model="sc_mcts" \
#                 --mode="sc" \
#                 --step=$step \
#                 --version=$version \
#                 --llama_path=$llama_path \
#                 --draft_model_path=$draft_model_path

# step=12
# run_blocksworld --gpu=17 \
#                 --reward_model="sc_mcts" \
#                 --mode="sc" \
#                 --step=$step \
#                 --version=$version \
#                 --llama_path=$llama_path \
#                 --draft_model_path=$draft_model_path


# # v1
# version=1

# step=2
# run_blocksworld --gpu=18 \
#                 --reward_model="sc_mcts" \
#                 --mode="sc" \
#                 --step=$step \
#                 --version=$version \
#                 --llama_path=$llama_path \
#                 --draft_model_path=$draft_model_path

# step=4
# run_blocksworld --gpu=19 \
#                 --reward_model="sc_mcts" \
#                 --mode="sc" \
#                 --step=$step \
#                 --version=$version \
#                 --llama_path=$llama_path \
#                 --draft_model_path=$draft_model_path

# step=6
# run_blocksworld --gpu=20 \
#                 --reward_model="sc_mcts" \
#                 --mode="sc" \
#                 --step=$step \
#                 --version=$version \
#                 --llama_path=$llama_path \
#                 --draft_model_path=$draft_model_path

# step=8
# run_blocksworld --gpu=21 \
#                 --reward_model="sc_mcts" \
#                 --mode="sc" \
#                 --step=$step \
#                 --version=$version \
#                 --llama_path=$llama_path \
#                 --draft_model_path=$draft_model_path

# step=10
# run_blocksworld --gpu=22 \
#                 --reward_model="sc_mcts" \
#                 --mode="sc" \
#                 --step=$step \
#                 --version=$version \
#                 --llama_path=$llama_path \
#                 --draft_model_path=$draft_model_path

# step=12
# run_blocksworld --gpu=23 \
#                 --reward_model="sc_mcts" \
#                 --mode="sc" \
#                 --step=$step \
#                 --version=$version \
#                 --llama_path=$llama_path \
#                 --draft_model_path=$draft_model_path
################################################ llama3.1-70b ##################################################################






# ################################################ llama3.1-405b ##################################################################
# llama_path="/hy-tmp/Meta-Llama-3.1-405B-Instruct-GPTQ-INT4"
# draft_model_path="/hy-tmp/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4"

# # v2
# version=2
# step=2
# run_blocksworld --gpu=12 \
#                 --reward_model="sc_mcts" \
#                 --mode="sc" \
#                 --step=$step \
#                 --version=$version \
#                 --llama_path=$llama_path \
#                 --draft_model_path=$draft_model_path

# step=4
# run_blocksworld --gpu=13 \
#                 --reward_model="sc_mcts" \
#                 --mode="sc" \
#                 --step=$step \
#                 --version=$version \
#                 --llama_path=$llama_path \
#                 --draft_model_path=$draft_model_path

# step=6
# run_blocksworld --gpu=14 \
#                 --reward_model="sc_mcts" \
#                 --mode="sc" \
#                 --step=$step \
#                 --version=$version \
#                 --llama_path=$llama_path \
#                 --draft_model_path=$draft_model_path

# step=8
# run_blocksworld --gpu=15 \
#                 --reward_model="sc_mcts" \
#                 --mode="sc" \
#                 --step=$step \
#                 --version=$version \
#                 --llama_path=$llama_path \
#                 --draft_model_path=$draft_model_path

# step=10
# run_blocksworld --gpu=16 \
#                 --reward_model="sc_mcts" \
#                 --mode="sc" \
#                 --step=$step \
#                 --version=$version \
#                 --llama_path=$llama_path \
#                 --draft_model_path=$draft_model_path

# step=12
# run_blocksworld --gpu=17 \
#                 --reward_model="sc_mcts" \
#                 --mode="sc" \
#                 --step=$step \
#                 --version=$version \
#                 --llama_path=$llama_path \
#                 --draft_model_path=$draft_model_path


# # v1
# version=1

# step=2
# run_blocksworld --gpu=18 \
#                 --reward_model="sc_mcts" \
#                 --mode="sc" \
#                 --step=$step \
#                 --version=$version \
#                 --llama_path=$llama_path \
#                 --draft_model_path=$draft_model_path

# step=4
# run_blocksworld --gpu=19 \
#                 --reward_model="sc_mcts" \
#                 --mode="sc" \
#                 --step=$step \
#                 --version=$version \
#                 --llama_path=$llama_path \
#                 --draft_model_path=$draft_model_path

# step=6
# run_blocksworld --gpu=20 \
#                 --reward_model="sc_mcts" \
#                 --mode="sc" \
#                 --step=$step \
#                 --version=$version \
#                 --llama_path=$llama_path \
#                 --draft_model_path=$draft_model_path

# step=8
# run_blocksworld --gpu=21 \
#                 --reward_model="sc_mcts" \
#                 --mode="sc" \
#                 --step=$step \
#                 --version=$version \
#                 --llama_path=$llama_path \
#                 --draft_model_path=$draft_model_path

# step=10
# run_blocksworld --gpu=22 \
#                 --reward_model="sc_mcts" \
#                 --mode="sc" \
#                 --step=$step \
#                 --version=$version \
#                 --llama_path=$llama_path \
#                 --draft_model_path=$draft_model_path

# step=12
# run_blocksworld --gpu=23 \
#                 --reward_model="sc_mcts" \
#                 --mode="sc" \
#                 --step=$step \
#                 --version=$version \
#                 --llama_path=$llama_path \
#                 --draft_model_path=$draft_model_path
# ################################################ llama3.1-405b ############################################################
