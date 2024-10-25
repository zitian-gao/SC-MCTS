# touch ~/.no_auto_tmux
# apt install htop pigz vim -y
# rm -rf /opt/conda
# wget https://repo.anaconda.com/archive/Anaconda3-2024.02-1-Linux-x86_64.sh
# bash Anaconda3-2024.02-1-Linux-x86_64.sh -b
# echo "export PATH=/root/anaconda3/bin:$PATH" >> ~/.bashrc
# source ~/.bashrc
# conda init
conda create -n reasoners python=3.10 -y
# source ~/.bashrc
# echo "export PATH=/root/anaconda3/bin:$PATH" >> ~/.bashrc
# echo "conda activate reasoners" >> ~/.bashrc
conda activate reasoners
# pip install "huggingface_hub[cli]" nvitop
pip install exllamav2 torch scikit-build fschat pddl matplotlib openai protobuf accelerate sentencepiece torch requests anthropic psutil numpy transformers shortuuid accelerate optimum
# conda install -c anaconda scipy==1.9.1 conda-forge gcc=12.1.0 conda-forge libgcc=5.2.0 -y
git clone https://github.com/PanQiWei/AutoGPTQ.git && cd AutoGPTQ && MAX_JOBS=40 pip install -vvv --no-build-isolation -e .
git clone https://github.com/Dao-AILab/flash-attention.git && cd flash-attention && MAX_JOBS=40 pip install -e . --no-build-isolation
# wget https://gpucloud-static-public-prod.gpushare.com/installation/oss/oss_linux_x86 && mv oss_linux_x86 oss && chmod +x ./oss

# ./oss cp oss://reasoners_0916.tar.gz ./ && tar -xf reasoners_0916.tar.gz && cd reasoners && pip install -e .

# huggingface-cli download --resume-download double7/vicuna-68m --local-dir /hy-tmp/vicuna-68m
# huggingface-cli download --resume-download lmsys/vicuna-7b-v1.3 --local-dir /hy-tmp/vicuna-7b-v1.3
# huggingface-cli download --resume-download TheBloke/vicuna-7B-v1.3-GPTQ --local-dir /hy-tmp/vicuna-7B-v1.3-GPTQ

# huggingface-cli download --resume-download Qwen/Qwen2-0.5B-Instruct-GPTQ-Int4 --local-dir /hy-tmp/Qwen2-0.5B-Instruct-GPTQ-Int4
# huggingface-cli download --resume-download Qwen/Qwen2-1.5B-Instruct-GPTQ-Int4 --local-dir /hy-tmp/Qwen2-1.5B-Instruct-GPTQ-Int4
# huggingface-cli download --resume-download Qwen/Qwen2-7B-Instruct-GPTQ-Int4 --local-dir /hy-tmp/Qwen2-7B-Instruct-GPTQ-Int4
# huggingface-cli download --resume-download Qwen/Qwen2-72B-Instruct-GPTQ-Int4 --local-dir /hy-tmp/Qwen2-72B-Instruct-GPTQ-Int4

# huggingface-cli download --resume-download yuhuili/EAGLE-LLaMA3-Instruct-8B --local-dir /hy-tmp/EAGLE-LLaMA3-Instruct-8B
# huggingface-cli download --resume-download TechxGenus/Meta-Llama-3-8B-GPTQ --local-dir /hy-tmp/Meta-Llama-3-8B-GPTQ
# huggingface-cli download --resume-download TechxGenus/Meta-Llama-3-8B-Instruct-GPTQ --local-dir /hy-tmp/Meta-Llama-3-8B-Instruct-GPTQ-INT4

# huggingface-cli download --resume-download TheBloke/Llama-2-7B-Chat-GPTQ --local-dir /hy-tmp/Llama-2-7B-Chat-GPTQ

# huggingface-cli download --resume-download hugging-quants/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4 --local-dir /hy-tmp/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4
# huggingface-cli download --resume-download hugging-quants/Meta-Llama-3.1-70B-Instruct-GPTQ-INT4 --local-dir /hy-tmp/Meta-Llama-3.1-70B-Instruct-GPTQ-INT4
# huggingface-cli download --resume-download hugging-quants/Meta-Llama-3.1-405B-Instruct-GPTQ-INT4 --local-dir /hy-tmp/Meta-Llama-3.1-405B-Instruct-GPTQ-INT4

# huggingface-cli download --resume-download GAIR/autoj-13b-GPTQ-4bits --local-dir /hy-tmp/autoj-13b-GPTQ-4bits

