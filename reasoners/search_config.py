import numpy as np

import reasoners.bw_utils as utils
from reasoners.world_model import BWState, BWAction
from reasoners import SearchConfig, LanguageModel

class BWConfig(SearchConfig):
    def __init__(self,
                 base_model: LanguageModel,
                 prompt: dict,
                 batch_size=2,
                 reward_alpha=0.5,
                 goal_reward_default=0.,
                 goal_reached_reward=100,
                 reward_model="logll",
                 mode="sc",
                 pickle_path=None) -> None:
        
        super().__init__()
        self.base_model = base_model
        self.example = None
        self.mode = mode
        self.prompt = prompt
        self.batch_size = batch_size
        self.reward_alpha = reward_alpha
        self.goal_reward_default = goal_reward_default
        self.goal_reached_reward = goal_reached_reward
        self.reward_model = reward_model
        self.pickle_path = pickle_path


    def get_actions(self, state: BWState) -> list[BWAction]:
        blocks_state = state.blocks_state
        return utils.generate_all_actions(blocks_state)


    def fast_reward(self, 
                    state: BWState, 
                    action: BWAction,
                    node_id) -> tuple[float, dict]:
        
        if state.buffered_action == "":
            # if no action buffered
            current_blocks_state = state.blocks_state
        else:
            # if action buffered
            current_blocks_state = state.last_blocks_state
        previous_action = state.buffered_action + "\n" if state.buffered_action != "" else ""
        
        # every two step, we will deduct the icl prompt
        # so that the distribution of step length is more reasonable
        icl_template = self.prompt["icl_list"][state.step_idx // 2]
        
        inputs = icl_template.replace("<init_state>", current_blocks_state)\
            .replace("<goals>", utils.extract_goals(self.example, return_raw=True)).replace("<action>", previous_action)
        
        self_eval_prompt = self.prompt["self-eval"].replace("<init_state>", current_blocks_state)\
            .replace("<goals>", utils.extract_goals(self.example, return_raw=True)).replace("<action>", action)

        reward_dict = self.base_model.reward(inputs, 
                                             [inputs + action], 
                                             self.reward_model, 
                                             self_eval_prompt=self_eval_prompt,
                                             node_id=node_id)

        intuition = reward_dict.get(self.reward_model)
        
        self_eval = reward_dict.get("self_eval")

        return self.calculate_reward(intuition, self_eval), reward_dict


    def calculate_reward(self, intuition, self_eval, goal_reached=None):
        # to provide a unified interface for reward and fast_reward
        if goal_reached is None:
            goal_reward = self.goal_reward_default

        elif goal_reached[0] == True:
            goal_reward = self.goal_reached_reward

        else:
            goal_reward = goal_reached[1]

        print(f"goal_reward: {goal_reward}")

        if self.mode == "baseline":
            return (intuition + self_eval) * self.reward_alpha + goal_reward * 20 * (1 - self.reward_alpha)
        else:
            return (intuition + self_eval) * self.reward_alpha + goal_reward * (1 - self.reward_alpha)
        

    def reward(self, 
               state: BWState, 
               action: BWAction,
               reward_dict : dict,
               goal_reached: tuple[bool, float] = None) -> float:
        
        intuition = reward_dict.get("intuition")
        self_eval = reward_dict.get("self_eval")
        reward_dict["goal_reached"] = goal_reached

        assert intuition is not None, "intuition is required to calculate reward in this search config, consider passing it in fast_reward"
        assert self_eval is not None, "self_eval is required to calculate reward in this search config, consider passing it in fast_reward"
        assert goal_reached is not None, "goal_reached is required to calculate reward in this search config, consider passing it in world model's step"

        return (self.calculate_reward(intuition, self_eval, goal_reached), reward_dict)

    def update_example(self, example, prompt=None) -> None:
        super().update_example(example, prompt=prompt)
