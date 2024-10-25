from typing import Union, Optional
import warnings
import time
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from . import LanguageModel, GenerateOutput
import torch.nn.functional as F
from datetime import datetime
# sys.path.append(os.path.join(os.path.dirname(reasoners.__file__), os.path.pardir, 'exllamav2'))

from exllamav2 import (
    ExLlamaV2,
    ExLlamaV2Config,
    ExLlamaV2Cache,
    ExLlamaV2Tokenizer,
)
from exllamav2.generator import ExLlamaV2Sampler, ExLlamaV2DynamicGenerator

class ExLlamaModelV2(LanguageModel):
    def __init__(self,
                 model_path,
                 lora_dir,
                 draft_model_path,
                 max_batch_size,
                 max_new_tokens,
                 max_seq_len=32768,
                 device='cuda:0',
                 mem_map: list[int] = None,
                 log_time=False,
                 log_output=False,
                 temperature=1.0,
                 top_k=1.0,
                 top_p=1,
                 top_a=0,
                 typical=0,
                 critic=False,
                 token_repetition_penalty=1):

        if draft_model_path is not None:
            self.draft_config = ExLlamaV2Config(draft_model_path)
            self.draft_config.arch_compat_overrides()
            self.draft_model = ExLlamaV2(self.draft_config)
            self.draft_cache = ExLlamaV2Cache(self.draft_model, max_seq_len=max_seq_len, lazy=True)
            self.draft_model.load_autosplit(self.draft_cache, progress=False)
        else:
            self.draft_model = None
            self.draft_cache = None

        # self.student_config = ExLlamaV2Config(draft_model_path)
        # self.student_config.set_low_mem()
        # # self.student_config.arch_compat_overrides()
        # self.student_config.max_seq_len=max_seq_len,
        # self.student_config.prepare()
        # self.student_model = ExLlamaV2(self.student_config)
        # self.student_cache = ExLlamaV2Cache(self.student_model, max_seq_len=max_seq_len, lazy=True)
        # # self.student_cache = None
        # self.student_model.load_autosplit(self.student_cache, progress=False)
        # self.student_tokenizer = ExLlamaV2Tokenizer(self.student_config)
        # self.student_generator = ExLlamaV2DynamicGenerator(model=self.student_model,
        #                                                    cache=self.student_cache,
        #                                                    tokenizer=self.student_tokenizer)
        # self.student_settings = ExLlamaV2Sampler.Settings()
        # self.student_settings.temperature = temperature
        # self.student_settings.top_k = top_k
        # self.student_settings.top_p = top_p
        # self.student_settings.top_a = top_a
        # self.student_settings.typical = typical
        # self.student_settings.token_repetition_penalty = token_repetition_penalty
        # self.student_generator.warmup()

        self.config = ExLlamaV2Config(model_path)
        self.config.arch_compat_overrides()
        self.config.max_seq_len = max_seq_len
        self.config.prepare()
        self.model = ExLlamaV2(self.config)
        self.cache = ExLlamaV2Cache(self.model, max_seq_len=max_seq_len, lazy=True)
        self.model.load_autosplit(self.cache, progress=False)
        self.tokenizer = ExLlamaV2Tokenizer(self.config)
        self.generator = ExLlamaV2DynamicGenerator(model=self.model,
                                                   cache=self.cache,
                                                   draft_model=self.draft_model,
                                                   draft_cache=self.draft_cache,
                                                   tokenizer=self.tokenizer,
                                                   max_chunk_size=2048,
                                                   paged=False)
        
        self.settings = ExLlamaV2Sampler.Settings()
        self.settings.temperature = temperature
        self.settings.top_k = top_k
        self.settings.top_p = top_p
        self.settings.top_a = top_a
        self.settings.typical = typical
        self.settings.token_repetition_penalty = token_repetition_penalty
        self.generator.warmup()

        if critic:
            # critic_model_dir = "/hy-tmp/autoj-13b-GPTQ-4bits"
            # critic_model_dir = "/hy-tmp/Qwen2-7B-Instruct-GPTQ-Int4"
            # critic_model_dir = "/hy-tmp/Qwen2-1.5B-Instruct-GPTQ-Int4"
            # critic_model_dir = "/hy-tmp/Qwen2-0.5B-Instruct-GPTQ-Int4"
            self.critic_config = ExLlamaV2Config(critic_model_dir)
            self.critic_config.max_seq_len=4096,
            self.critic_config.prepare()
            self.critic_config.set_low_mem()
            self.critic_config.arch_compat_overrides()
            self.critic_model = ExLlamaV2(self.critic_config)
            self.critic_cache = ExLlamaV2Cache(self.critic_model, max_seq_len=max_seq_len, lazy=True)
            self.critic_model.load_autosplit(self.critic_cache, progress=False)
            self.critic_tokenizer = ExLlamaV2Tokenizer(self.critic_config)
            self.critic_generator = ExLlamaV2DynamicGenerator(model=self.critic_model,
                                                            cache=self.critic_cache,
                                                            tokenizer=self.critic_tokenizer)
            self.critic_generator.warmup()
        else:
            self.critic_model = None
            self.critic_generator = None
            self.critic_tokenizer = None
            self.critic_cache = None


        self.device = device
        self.max_batch_size = max_batch_size
        self.max_new_tokens = max_new_tokens
        self.max_seq_length = max_seq_len
        self.critic = critic


    def generate(
        self,
        inputs: list[str],
        max_length: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
        do_sample: bool = False,
        temperature=1,
        top_k=1,
        top_p=1.0,
        top_a=0,
        typical=0,
        token_repetition_penalty=1.15,
        num_return_sequences: int = 1,
        eos_token_id: Union[None, str, int, list[str, int]] = None,
        hide_input: bool = True,
        output_log_probs: bool = False,
        strategy="greedy",
        eos_list=None,
        critic=False,
        **kwargs,
    ) -> GenerateOutput:
        start_time = time.time()

        print(f"\ninput: {inputs}")
        decoded_list = []

        if eos_list is None:
            # eos_list = ["\n", "<|eot_id|>", 128001, 128008, 128009]  # mcts: llama3, llama3.1
            eos_list = ["\n["]  # cot: llama3, llama3.1
        #   eos_list = ["<\n>", 13]  # mcts: llama2
        #   eos_list = ["\n["]  # cot: llama2

        if strategy == "greedy":
            self.settings.temperature = 1.0
            self.settings.top_k = 1.0
            self.settings.top_p = 1
            self.settings.top_a = 0
            self.settings.typical = 0
            self.settings.token_repetition_penalty = 1

            with torch.inference_mode():
                self.generator.reset_page_table()
                decoded = self.generator.generate(prompt=inputs,
                                                  max_new_tokens=200,
                                                  min_new_tokens=50,
                                                  stop_conditions=eos_list,
                                                  decode_special_tokens=True,
                                                  add_eos=True,
                                                  completion_only=True)
                

        generate_time = time.time() - start_time
        print(f"generate takes: {generate_time}s, n_tokens: {len(decoded[0])}, speed: {generate_time/len(decoded[0])}s/token")

        # elif strategy == "contrastive":
        #     def contrastive_decoding(input_sequence, max_length):
        #         current_sequence = input_sequence
        #         for _ in range(max_length):
        #             strong_probs = self.model.get_next_token_logits(current_sequence)
        #             weak_probs = self.draft_model.get_next_token_logits(current_sequence)
        #             # 计算对比差异度，根据论文指定具体的差异计算方式
        #             contrastive_scores = strong_probs - weak_probs
        #             # 选择对比差异度最大的词
        #             next_word = select_word_with_max_contrastive_score(contrastive_scores)
        #             # 更新当前序列
        #             current_sequence = current_sequence + next_word
        #             for e in eos_list:
        #                 if next_word == e:
        #                     break
        #         return current_sequence
        #     def select_word_with_max_contrastive_score(contrastive_scores):
        #         # 选择差异分数最大的词进行输出
        #         max_score_idx = contrastive_scores.argmax()
        #         return vocabulary[max_score_idx]  # vocabulary是一个索引到单词的映射
        #     decoded = contrastive_decoding(inputs, max_length=20)


        if not isinstance(decoded, list):
            decoded = [decoded]

        decoded = [text.strip() for text in decoded]
        decoded_list.extend(decoded)

        print(f"output: {decoded_list}")

        return GenerateOutput(decoded_list, None)

    
    @torch.no_grad()
    def get_next_token_logits(
            self, 
            prompt: Union[str, list[str]],
            candidates: Union[list[str], list[list[str]]]) -> list[np.ndarray]:
        
        if isinstance(prompt, str):
            prompt = [prompt]

        if isinstance(candidates[0], str):
            candidates = [candidates] * len(prompt)

        cand_tokens = []
        for candidate in candidates:
            cand_tokens.append([])
            for cand in candidate:
                token = self.tokenizer.encode(cand,
                                              add_bos=False,
                                              add_eos=False)
                
                if len(token) != 1:
                    warnings.warn(f'candidate {cand} corresponds to {len(token)} instead of 1')

                cand_tokens[-1].append(token[1].item() if len(token) > 1 else token[0].item())

        bsz = len(prompt)
        assert bsz <= self.max_batch_size, (bsz, self.max_batch_size)

        with torch.no_grad():
            # self.cache.current_seq_len = 0

            tokens = self.tokenizer.encode(prompt,
                                           add_bos=True,
                                           add_eos=False)
            
            all_logits = self.model.forward(tokens,
                                            self.cache,
                                            last_id_only=True).squeeze(1)

        assert all_logits.shape[0] == bsz, (all_logits.shape[0], bsz)

        logits = []

        for case_logits, cand in zip(all_logits, cand_tokens):
            logits.append(case_logits[cand].cpu().numpy())

        return logits
    

    @torch.no_grad()
    def get_loglikelihood(self, 
                          prefix: str, 
                          contents: list[str],
                          **kwargs) -> np.ndarray:

        contents_length = len(contents)
        assert contents_length <= self.max_batch_size, (contents_length, self.max_batch_size)

        # critic=False

        loglikelihood_model = self.critic_model if self.critic else self.model
        loglikelihood_tokenizer = self.critic_tokenizer if self.critic else self.tokenizer
        loglikelihood_cache = self.critic_cache if self.critic else self.cache

        prefix_tokens = loglikelihood_tokenizer.encode(prefix,
                                                       add_bos=True,
                                                       add_eos=False).squeeze(0)
        
        contents_tokens = loglikelihood_tokenizer.encode(contents,
                                                         add_bos=True,
                                                         add_eos=False)

        # 确保每个提示tokens都以给定的前缀tokens开头
        for content_tokens in contents_tokens:
            assert torch.all(content_tokens[:len(prefix_tokens)] == prefix_tokens)  

        with torch.no_grad():
            loglikelihood_cache.current_seq_len = 0
            contents_logits = loglikelihood_model.forward(contents_tokens,
                                                          loglikelihood_cache,
                                                          last_id_only=False,
                                                          preprocess_only=False)


        # 创建一个形状为 (批次大小) 的张量，用于累积对数概率，初始值为0
        acc_loglikelihood = torch.zeros(contents_length).to(self.device)  

        # 从前缀tokens的长度开始遍历到最大提示长度
        for i in range(len(prefix_tokens), contents_tokens.shape[1]):  
            probs = torch.softmax(contents_logits[:, i - 1, :], dim=-1)  # 对第i-1位置的logits计算softmax以得到概率分布
            for j in range(contents_length):  # 对于每个批次中的样本
                if contents_tokens[j, i] != self.tokenizer.pad_token_id:  # 如果当前token不是填充符
                    acc_loglikelihood[j] += torch.log(probs[j, contents_tokens[j, i]])  # 将该token的对数概率累加到acc_loglikelihood中

        return acc_loglikelihood.cpu().numpy()


    @torch.no_grad()
    def reward(self, 
               prefix: str, 
               full_contents: list[str],
               reward_model: str,
               pickle_path=None,
               self_eval_prompt=None,
               node_id=None,
               **kwargs) -> np.ndarray:
        
        student_model = self.draft_model

        # student_model = self.draft_model

        continue_contents = full_contents[0][len(prefix):]
        prefix_ids = self.tokenizer.encode(prefix, add_bos=True, add_eos=False)
        full_contents_ids = self.tokenizer.encode(full_contents, add_bos=True, add_eos=False)
        prefix_length = prefix_ids.shape[-1]
        continue_ids = full_contents_ids[0, prefix_length:]
        prefix = full_contents[0][:len(prefix)]

        self.cache.current_seq_len = 0
        full_content_logits = self.model.forward(full_contents_ids,
                                                 self.cache,
                                                 last_id_only=False,
                                                 preprocess_only=False).to("cuda:0")

        target_logits = full_content_logits[:, prefix_length - 1:-1, :]

        self.cache.current_seq_len = 0
        student_logits = student_model.forward(full_contents_ids,
                                               self.draft_cache,
                                               last_id_only=False,
                                               preprocess_only=False
                                               )[:, prefix_length - 1:-1, :].to("cuda:0")
        
        softmax_target = F.softmax(target_logits, dim=-1)
        softmax_draft = F.softmax(student_logits, dim=-1)

        log_softmax_target = F.log_softmax(target_logits, dim=-1)
        log_softmax_draft = F.log_softmax(student_logits, dim=-1)

        log_softmax_target_clamp = torch.clamp(log_softmax_target, min=1e-5, max=1)
        log_softmax_draft_clamp = torch.clamp(log_softmax_draft, min=1e-5, max=1)

        M = 0.5 * (softmax_target + softmax_draft)

        kl_div_batchmean = F.kl_div(log_softmax_target_clamp, log_softmax_draft_clamp, reduction='batchmean').cpu().numpy()
        kl_div_mean = F.kl_div(log_softmax_target_clamp, log_softmax_draft_clamp, reduction='none').mean(-1).sum().item()

        kl_div_target_M_batchmean = F.kl_div(log_softmax_target, M, reduction='batchmean')
        kl_div_target_M_mean = F.kl_div(log_softmax_target, M, reduction='none').mean(-1)

        kl_div_target_M_clamp_batchmean = F.kl_div(log_softmax_target_clamp, M, reduction='batchmean')
        kl_div_target_M_clamp_mean = F.kl_div(log_softmax_target_clamp, M, reduction='none').mean(-1)

        kl_div_draft_M_batchmean = F.kl_div(log_softmax_draft, M, reduction='batchmean')
        kl_div_draft_M_mean = F.kl_div(log_softmax_draft, M, reduction='none').mean(-1)

        kl_div_draft_M_clamp_batchmean = F.kl_div(log_softmax_draft_clamp, M, reduction='batchmean')
        kl_div_draft_M_clamp_mean = F.kl_div(log_softmax_draft_clamp, M, reduction='none').mean(-1)

        js_div_clamp_batchmean = 0.5 * (kl_div_target_M_clamp_batchmean + kl_div_draft_M_clamp_batchmean).cpu().numpy()
        js_div_clamp_mean = 0.5 * (kl_div_target_M_clamp_mean + kl_div_draft_M_clamp_mean).sum().item()

        js_div_batchmean = 0.5 * (kl_div_target_M_batchmean + kl_div_draft_M_batchmean).cpu().numpy()
        js_div_mean = 0.5 * (kl_div_target_M_mean + kl_div_draft_M_mean).sum().item()

        diff_logits = log_softmax_target - log_softmax_draft

        # if post_softmax:
        diff_logits = diff_logits.log_softmax(dim=-1)

        # log_probs = diff_logits[0, range(diff_logits.shape[1]), continue_ids].sum().item()

        acc_diff_logits = torch.zeros(full_contents_ids.shape[0]).to("cuda:0")

        # 计算 diff_logits 的累加值
        for i in range(diff_logits.shape[1]):  # 遍历 diff_logits 的时间步
            for j in range(full_contents_ids.shape[0]):  # 对于每个批次中的样本
                if continue_ids[i] != self.tokenizer.pad_token_id:  # 如果当前token不是填充符
                    acc_diff_logits[j] += diff_logits[j, i, continue_ids[i]].to("cuda:0")
                else:
                    print(f"padding: {continue_ids[i]}")

        cd_logprobs_diff = acc_diff_logits.cpu().numpy()[0]


        ################################################# loglikelihood ##################################################
        # # 创建一个形状为 (批次大小) 的张量，用于累积对数概率，初始值为0
        # full_contents_length = len(full_contents)
        # acc_loglikelihood = torch.zeros(full_contents_length).to("cuda:0")

        # # 从前缀tokens的长度开始遍历到最大提示长度
        # for i in range(len(prefix_ids), full_contents_ids.shape[1]):  
        #     probs = torch.softmax(full_content_logits[:, i - 1, :], dim=-1).to("cuda:0")  # 对第i-1位置的logits计算softmax以得到概率分布
        #     for j in range(full_contents_length):  # 对于每个批次中的样本
        #         if full_contents_ids[j, i] != self.tokenizer.pad_token_id:  # 如果当前token不是填充符
        #             acc_loglikelihood[j] += torch.log(probs[j, full_contents_ids[j, i]]).to("cuda:0")  # 将该token的对数概率累加到acc_loglikelihood中

        # loglikelihood = acc_loglikelihood.cpu().numpy()[0]

        full_contents_length = len(full_contents)
        acc_loglikelihood = torch.zeros(full_contents_length).to("cuda:0")

        # 用于记录对数似然值的列表和累积的对数似然
        loglikelihood_values = []
        accumulated_loglikelihoods = []  # 新增累加的对数似然值列表

        # 从前缀tokens的长度开始遍历到最大提示长度
        for i in range(len(prefix_ids), full_contents_ids.shape[1]):  
            probs = torch.softmax(full_content_logits[:, i - 1, :], dim=-1).to("cuda:0")  # 对第i-1位置的logits计算softmax以得到概率分布
            step_loglikelihoods = []  # 用于记录每个批次中当前token的对数似然值
            
            for j in range(full_contents_length):  # 对于每个批次中的样本
                if full_contents_ids[j, i] != self.tokenizer.pad_token_id:  # 如果当前token不是填充符
                    log_prob = torch.log(probs[j, full_contents_ids[j, i]]).to("cuda:0")
                    acc_loglikelihood[j] += log_prob  # 将该token的对数概率累加到acc_loglikelihood中
                    step_loglikelihoods.append(log_prob.cpu().item())  # 保存当前token的对数似然值到列表

            loglikelihood_values.append(step_loglikelihoods)  # 将当前步骤的对数似然值添加到总列表中
            
            # 保存累积的对数似然值的平均值
            accumulated_loglikelihood_mean = acc_loglikelihood.mean().cpu().item()
            accumulated_loglikelihoods.append(accumulated_loglikelihood_mean)

        # 将累积的对数似然转换为 numpy 数组
        js_div_clamp_batchmean = 0
        loglikelihood = float(acc_loglikelihood.cpu().numpy()[0])

        if self.critic:
            with torch.inference_mode():
                try:
                    self.critic_generator.reset_page_table()
                    critic_output = self.critic_generator.generate(prompt=build_autoj_input(prefix, continue_contents),
                                                                    max_new_tokens=500,
                                                                    min_new_tokens=50,
                                                                    stop_conditions=["]]", "<|eot_id|>"],
                                                                    add_eos=True,
                                                                    completion_only=True)
                    critic_score = int(critic_output[-1])
                    
                except:
                    print(f"\ncritic score error at output: \n{critic_output}")
                    critic_score = 2
        else:
            critic_score = 1
            critic_eval = 1

        # print(f'self_eval_prompt: """\n{self_eval_prompt}\n"""')

        self_eval = self.get_loglikelihood(self_eval_prompt, 
                                           [self_eval_prompt + "good"])[0]

        critic_eval = self.get_loglikelihood(self_eval_prompt, 
                                             [self_eval_prompt + "good"],
                                             critic=True)[0]

        # cd_logprobs_diff = cd_logprobs_diff * 100
        # norm_cd_logprobs_diff = (cd_logprobs_diff - (-180)) / (-110 - (-180))

        # with open('quantile_transformer.pkl', 'rb') as f:
        #     cd_logprobs_diff_quantile_transformer = pickle.load(f)

        # transformed_cd_logprobs_diff = cd_logprobs_diff_quantile_transformer.transform([[cd_logprobs_diff]])[0][0] * 100


        # # 定义函数加载QuantileTransformer
        # def load_quantile_transformer(filename):
        #     with open(filename, 'rb') as f:
        #         quantile_transformer= pickle.load(f)

        # # 定义函数根据区间应用QuantileTransformer
        # def quantile_transformer(value, range_label):
        #     if range_label == "2000_3500":
        #         transformer = load_quantile_transformer("quantile_transformer_2000_3500.pkl")
        #     elif range_label == "5500_6800":
        #         transformer = load_quantile_transformer("quantile_transformer_5500_6800.pkl")
        #     elif range_label == "6800_8500":
        #         transformer = load_quantile_transformer("quantile_transformer_6800_8500.pkl")
        #     else:
        #         raise ValueError(f"Invalid range label: {range_label}")

        #     # 使用加载的 transformer 对单个数值进行转换
        #     return transformer.transform([[value]])[0][0]

        # js_div_batchmean = js_div_batchmean * 100

        # # 根据数据值所属的区间应用不同的 transformer
        # if 2000 <= js_div_batchmean <= 3500:
        #     norm_js_div_batchmean = quantile_transformer(js_div_batchmean, "2000_3500")
        # elif 5500 <= js_div_batchmean <= 6800:
        #     norm_js_div_batchmean = quantile_transformer(js_div_batchmean, "5500_6800")
        # elif 6800 <= js_div_batchmean <= 8500:
        #     norm_js_div_batchmean = quantile_transformer(js_div_batchmean, "6800_8500")
        # else:
        #     # 如果值不在任何区间内，选择最接近的
        #     print(f"OOD js_div_batchmean={js_div_batchmean}")
        #     if abs(js_div_batchmean - 3500) < abs(js_div_batchmean - 5500):
        #         norm_js_div_batchmean = quantile_transformer(js_div_batchmean, "2000_3500")
        #     elif abs(js_div_batchmean - 6800) < abs(js_div_batchmean - 8500):
        #         norm_js_div_batchmean = quantile_transformer(js_div_batchmean, "5500_6800")
        #     else:
        #         norm_js_div_batchmean = quantile_transformer(js_div_batchmean, "6800_8500")

        # print(f"Normalized js_div_batchmean: {norm_js_div_batchmean}")
        # print(f"kl_div_mean: {kl_div_mean}")

        reward_dict = {
            "control": 1,
            "verifier": 0,
            "sc_mcts": loglikelihood+self_eval+js_div_batchmean,
            "loglikelihood": loglikelihood,
            "self_eval": self_eval,
            "cd_logprobs_diff": cd_logprobs_diff,
            "kl_div_mean": kl_div_mean,
            "kl_div_batchmean": kl_div_batchmean,
            "js_div_clamp_batchmean": js_div_clamp_batchmean,
            "js_div_clamp_mean": js_div_clamp_mean,
            "js_div_batchmean": js_div_batchmean,
            "js_div_mean": js_div_mean,
        }

        print(f"\nnode_id: {node_id}")
        print(f"action: {full_contents[0][len(prefix):]}")

        for label, value in reward_dict.items():
            print(f"{label}: {value}")

        reward_dict["intuition"] = reward_dict[reward_model]

        print(f"reward_model: {reward_model}")

        return reward_dict

    