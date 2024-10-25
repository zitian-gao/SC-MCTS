#!/bin/bash

export VAL="LLMs-Planning/planner_tools/VAL"

run_blocksworld() {
    local gpus=""              # 修改为gpus，支持多个gpu
    local step=""
    local llama_path=""
    local draft_model_path=""
    local version=2             
    local exllama_version=2
    local start_idx=""

    for arg in "$@"
    do
    case $arg in
        --gpus=*)
        gpus="${arg#*=}"  # 支持传递多个 GPU，例如 "0,1,2"
        shift
        ;;
        --step=*)
        step="${arg#*=}"
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
        *)
        echo "Unknown option: ${arg}"
        return 1
        ;;
    esac
    done

    local model_name=$(echo "$llama_path" | sed -n 's|.*hy-tmp/\(.*\)|\1|p')
    local log_name="${model_name}/e${exllama_version}_v${version}_step${step}"

    mkdir -p log/blocksworld/cot/${model_name}
    mkdir -p pickles/blocksworld/cot/${model_name}

    local prompt_path=""
    if [ "$version" -eq 1 ]; then
        prompt_path="datasets/blocksworld/prompts/pool_prompt_v1.json"
    else
        prompt_path="datasets/blocksworld/prompts/pool_prompt_v${version}_step_${step}.json"
    fi

    # 修改 CUDA_VISIBLE_DEVICES，以支持多个 GPU，例如 "0,1,2"
    export CUDA_VISIBLE_DEVICES=$gpus

    local log_file="log/blocksworld/cot/${log_name}.log"

    if [ -n "$start_idx" ]; then
        python scripts/blocksworld/cot_inference.py --llama_path "$llama_path" \
                                                    --draft_model_path "$draft_model_path" \
                                                    --depth_limit "$step" \
                                                    --exllama_version "$exllama_version" \
                                                    --data_path "datasets/blocksworld/data/split_v${version}/split_v${version}_step_${step}_data.json" \
                                                    --prompt_path "$prompt_path" \
                                                    --start_idx "$start_idx" \
                                                    >> "$log_file" 2>&1 &
    else
        python scripts/blocksworld/cot_inference.py --llama_path "$llama_path" \
                                                    --draft_model_path "$draft_model_path" \
                                                    --depth_limit "$step" \
                                                    --exllama_version "$exllama_version" \
                                                    --data_path "datasets/blocksworld/data/split_v${version}/split_v${version}_step_${step}_data.json" \
                                                    --prompt_path "$prompt_path" \
                                                    > "$log_file" 2>&1 &
    fi

    echo "$log_file"
}


check_gpu_free() {
    local gpu_id=$1
    local pid=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits -i "$gpu_id")

    if [ -z "$pid" ]; then
        return 0  # GPU 空闲
    else
        return 1  # GPU 正在使用
    fi
}


# 检测多个 GPU 是否空闲
check_gpus_free() {
    local gpus=($1)  # 将逗号分隔的 GPU ID 转换为数组
    for gpu in "${gpus[@]}"; do
        if ! check_gpu_free "$gpu"; then
            return 1  # 只要有一个 GPU 被占用，返回 1
        fi
    done
    return 0  # 所有 GPU 都空闲，返回 0
}

# 循环检测多个 GPU 是否空闲
wait_for_gpus() {
    local gpus=($1)  # 将 GPU 列表转换为数组
    while ! check_gpus_free "$gpus"; do
        echo "GPUs (${gpus[@]}) 正在使用，等待中..."
        sleep 5  # 每隔5秒检测一次
    done
    echo "GPUs (${gpus[@]}) 空闲，开始执行任务。"
}

# 使用多个 GPU 执行任务
execute_task_on_gpus() {
    local gpus=$1
    local step=$2
    local version=$3
    local llama_path=$4
    local draft_model_path=$5

    wait_for_gpus "$gpus"
    run_blocksworld --gpus="$gpus" --step="$step" --version="$version" --llama_path="$llama_path" --draft_model_path="$draft_model_path"
}

# 动态分配 GPU 任务
schedule_tasks() {
    local -n task_queue=$1  # 引用传递任务队列
    local gpus=("$@")  # 提供的 GPU 列表

    while [ "${#task_queue[@]}" -gt 0 ]; do
        for gpu_set in "${gpus[@]:1}"; do
            # 检测是否有任务要执行
            if check_gpus_free "$gpu_set"; then
                task="${task_queue[0]}"
                IFS=" " read -r version step <<< "$task"  # 将任务分成 version 和 step
                echo "在 GPUs $gpu_set 上执行 version=$version step=$step"

                # 执行任务并移出队列
                execute_task_on_gpus "$gpu_set" "$step" "$version" "$llama_path" "$draft_model_path" &
                task_queue=("${task_queue[@]:1}")  # 更新任务队列
                sleep 10  # 防止频繁切换导致冲突
            fi
        done
        sleep 5  # 每隔5秒检测一次是否有可用 GPU
    done
}


# 任务排队（顺序为 version 1 的所有任务和 version 2 的所有任务）
tasks=(
    "2 6"
    "2 8"
    "2 10"
    "2 4"
    "2 12"
)

# 可用的 GPU 列表（更新以支持多个 GPU 分配）
gpus=(
    "0,1,2"  # 需要三个 GPU 的任务
    "3,4,5"  # 需要三个 GPU 的任务
)

# llama_path 和 draft_model_path 的设置
llama_path="/hy-tmp/Meta-Llama-3.1-405B-Instruct-GPTQ-INT4"  # 需要三个 GPU 的任务
draft_model_path="/hy-tmp/Llama-3.2-1B-4bit-128g"

# 调度任务
schedule_tasks tasks "${gpus[@]}"
