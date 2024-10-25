from reasoners.lm import ExLlamaModelV2
import json
from reasoners.benchmark import BWEvaluator
import argparse
import torch


class CoTReasoner():

    def __init__(self, base_model, temperature=0.8, model_type="completion"):
        self.base_model = base_model
        self.temperature = temperature
        self.model_type = model_type

    def __call__(self, example, prompt=None):
        inputs = prompt["icl"].replace("<init_state>", example["init"])\
            .replace("<goals>", example["goal"]).replace("<action>", "")

        if self.model_type == "completion":
            output = self.base_model.generate(
                [inputs],
                hide_input=True,
                do_sample=True,
                temperature=self.temperature,
                eos_token_id=[13, "<\n>"]).text[0].strip()

        elif self.model_type == "chat":
            output = self.base_model.generate(
                [inputs],
                hide_input=True,
                do_sample=True,
                temperature=self.temperature).text[0].replace(
                    "[PLAN END]", "").strip()
            
        return output


def main(base_model,
         data_path,
         prompt_path,
         start_idx=0,
         end_idx=0,
         temperature=0,
         log_dir=None,
         disable_log=False):

    config_file = "examples/CoT/blocksworld/data/bw_config.yaml"
    domain_file = "examples/CoT/blocksworld/data/generated_domain.pddl"

    with open(prompt_path) as f:
        prompt = json.load(f)

    reasoner = CoTReasoner(base_model,
                           temperature=temperature,
                           model_type="completion")  # if openai, use "chat"

    evaluator = BWEvaluator(
        config_file=config_file,
        domain_file=domain_file,
        data_path=data_path,
        init_prompt=prompt,
        disable_log=disable_log,
        output_extractor=lambda x: x,
        sample_prompt_type="rap")  # rap prompt includes cot

    accuracy = evaluator.evaluate(reasoner,
                                  shuffle_prompt=True,
                                  num_shot=4,
                                  start_idx=start_idx,
                                  end_idx=end_idx,
                                  log_dir=log_dir)

    print(f'accuracy: {accuracy:.4f}')


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run COT Inference")
    parser.add_argument("--start_idx", type=int, default=None)
    parser.add_argument("--end_idx", type=int, default=None)
    parser.add_argument("--llama_path", type=str, required=True)
    parser.add_argument("--draft_model_path", type=str, default=None)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--prompt_path", type=str, required=True)
    parser.add_argument("--exllama_version", type=int, required=True)

    # optional
    parser.add_argument("--depth_limit", type=int, default=8)
    args = parser.parse_args()
    device = torch.device('cuda')

    if args.exllama_version == 1:
        llama_model = ExLlamaModel(model_dir=args.llama_path,
                                    student_lm_dir=None,
                                    lora_dir=None,
                                    device="cuda",
                                    max_batch_size=1,
                                    max_new_tokens=200,
                                    max_seq_length=4096)
        
    elif args.exllama_version == 2:
        llama_model = ExLlamaModelV2(model_path=args.llama_path,
                                     draft_model_path=args.draft_model_path,
                                     lora_dir=None,
                                     device="cuda",
                                     max_batch_size=1,
                                     max_new_tokens=200,
                                     max_seq_len=16384)

    main(base_model=llama_model,
         prompt_path=args.prompt_path,
         data_path=args.data_path,
         start_idx=args.start_idx,
         end_idx=args.end_idx,
         temperature=0,
         log_dir=f"log/blocksworld/output/{args.start_idx}_{args.end_idx}.log")
