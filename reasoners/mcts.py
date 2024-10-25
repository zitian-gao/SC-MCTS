import math
from copy import deepcopy
from typing import Generic, Optional, NamedTuple, Callable, Hashable
import itertools
from abc import ABC
from abc import ABC
from collections import defaultdict

import numpy as np
from tqdm import trange

from . import SearchAlgorithm, WorldModel, SearchConfig, State, Action, Example, Trace


class MCTSNode(Generic[State, Action, Example]):
    id_iter = itertools.count()  # 初始化节点ID迭代器，从0开始递增。

    @classmethod
    def reset_id(cls):
        cls.id_iter = itertools.count()  # 静态方法，用于重置节点ID迭代器。

    # 初始化节点对象，包含状态、动作、父节点、快速奖励、终端状态标识等
    def __init__(
        self,
        state: Optional[State],
        action: Optional[Action],
        parent: "Optional[MCTSNode]" = None,
        fast_reward: float = 0.0,
        fast_reward_details=None,
        is_terminal: bool = False,
        calc_q: Callable[[list[float]], float] = np.mean,
    ):
        """
        MCTS搜索树中的一个节点。

        :param state: 当前状态
        :param action: 上一步的动作，即从父节点到当前节点的动作
        :param parent: 父节点，如果是树的根节点则为None
        :param fast_reward: 上一步奖励的估计值
        :param is_terminal: 当前状态是否为终端状态
        :param calc_q: 从历史记录计算Q值的方法。默认使用np.mean
        """
        if fast_reward_details is None:
            fast_reward_details = {}  # 如果没有提供快速奖励的详细信息，则初始化为空字典

        self.id = next(MCTSNode.id_iter)  # 为节点分配唯一ID
        self.cum_rewards: list[float] = []  # 初始化累积奖励列表
        self.fast_reward = self.reward = fast_reward  # 初始化快速奖励和实际奖励
        self.fast_reward_details = fast_reward_details  # 存储快速奖励的详细信息
        self.is_terminal = is_terminal  # 标识当前状态是否为终端状态
        self.action = action  # 存储到达当前节点的动作
        self.state = state  # 存储当前状态
        self.parent = parent  # 存储父节点
        self.children: "Optional[list[MCTSNode]]" = None  # 初始化子节点列表为空
        self.calc_q = calc_q  # 存储用于计算Q值的方法

        if parent is None:
            self.depth = 0  # 如果没有父节点，深度为0（根节点）
        else:
            self.depth = parent.depth + 1  # 否则深度为父节点深度加1

    # 定义Q值属性，返回节点的Q值
    @property
    def Q(self) -> float:
        if self.state is None:  # 如果状态为空，返回快速奖励
            return self.fast_reward

        else:  # 否则，使用指定的方法计算Q值
            return self.calc_q(self.cum_rewards)


# 定义MCTS结果类，包含终端状态、累积奖励、路径等信息
class MCTSResult(NamedTuple):
    terminal_state: State  # 终端状态
    cum_reward: float  # 累积奖励
    trace: Trace  # 路径
    trace_of_nodes: list[MCTSNode]  # 节点的路径列表
    tree_state: MCTSNode  # 树的状态
    trace_in_each_iter: list[list[MCTSNode]] = None  # 每次迭代中的路径
    tree_state_after_each_iter: list[MCTSNode] = None  # 每次迭代后的树状态
    aggregated_result: Optional[Hashable] = None  # 聚合结果


# 定义MCTS聚合类，用于聚合多个答案
class MCTSAggregation(Generic[State, Action, Example], ABC):

    def __init__(self,
                 retrieve_answer: Callable[[State], Hashable],
                 weight_policy: str = "edge"):

        assert weight_policy in [
            "edge",
            "edge_inverse_depth",
            "uniform",
        ]

        self.retrieve_answer = retrieve_answer  # 存储答案提取方法
        self.weight_policy = weight_policy  # 存储权重策略

    def __call__(
            self, tree_state: MCTSNode[State, Action,
                                       Example]) -> Optional[Hashable]:

        answer_dict = defaultdict(lambda: 0)  # 使用默认值为0的字典存储答案

        def visit(cur: MCTSNode[State, Action, Example]):
            if cur.state is None:
                return []  # 如果当前节点状态为空，返回空列表

            if cur.is_terminal:
                answer = self.retrieve_answer(cur.state)  # 检索终端节点的答案

                if answer is None:
                    print("MCTSAggregation: no answer retrieved.")
                    return []

                if self.weight_policy == "edge":
                    answer_dict[answer] += cur.reward  # 按边权重更新答案字典

                elif self.weight_policy == "edge_inverse_depth":
                    answer_dict[answer] += (cur.reward / cur.depth
                                            )  # 按边反深度更新答案字典

                elif self.weight_policy == "uniform":
                    answer_dict[answer] += 1.0  # 按均匀权重更新答案字典

                return [(answer, cur.depth)]  # 返回答案及其深度

            depth_list = defaultdict(list)  # 创建用于存储深度的字典
            cur_list = []  # 初始化当前列表为空

            for child in cur.children:
                cur_list.extend(child_info := visit(child))  # 递归访问子节点并扩展当前列表
                for answer, depth in child_info:
                    depth_list[answer].append(depth)  # 将深度添加到深度列表中

            for answer, depths in depth_list.items():
                if self.weight_policy == "edge":
                    answer_dict[answer] += cur.reward  # 按边权重更新答案字典

                elif self.weight_policy == "edge_inverse_depth":
                    answer_dict[answer] += cur.reward / np.mean(
                        depths)  # 按边反深度平均值更新答案字典

            return cur_list

        visit(tree_state)  # 开始访问树的根节点

        if len(answer_dict) == 0:
            return None  # 如果没有答案，返回None
        return max(answer_dict,
                   key=lambda answer: answer_dict[answer])  # 返回具有最大权重的答案


# 定义MCTS算法类，用于实现蒙特卡洛树搜索
class MCTS(SearchAlgorithm, Generic[State, Action, Example]):

    def __init__(
        self,
        output_trace_in_each_iter: bool = True,
        w_exp: float = 1.0,
        depth_limit: int = 5,
        n_iters: int = 10,
        cum_reward: Callable[[list[float]], float] = sum,
        calc_q: Callable[[list[float]], float] = np.mean,
        simulate_strategy: str | Callable[[list[float]], int] = "max",
        output_strategy: str = "max_reward",
        mode: str = "sc",
        uct_with_fast_reward: bool = True,
        aggregator: Optional[MCTSAggregation] = None,
        disable_tqdm: bool = True,
        node_visualizer: Callable[[MCTSNode], dict] = lambda x: x.__dict__,
    ):
        """
        MCTS算法

        :param output_trace_in_each_iter: 是否在每次迭代中输出选择的轨迹；轨迹将被深度拷贝
                                          还将输出*tree_state_after_each_iter*，即深度拷贝的根节点
        :param w_exp: 在UCT中的探索权重
        :param cum_reward: 计算每步累积奖励的方法。默认使用sum
        :param calc_q: 从历史记录计算Q值的方法。默认使用np.mean
        :param simulate_strategy: 模拟策略。选项有：'max'、'sample'、'random'，或使用自定义函数
        :param output_strategy: 输出结果的方式。节点不被深度拷贝，因此信息在所有迭代后仍保持
                                选项：'max_reward': 使用:param cum_reward:对最终树进行深度优先搜索，找到最大奖励的轨迹
                                         'follow_max': 从根节点开始，在每一步选择最大奖励的子节点。如果死胡同可能输出非终端节点
                                         'max_visit': 拥有最多访问次数的终端节点
                                         'max_iter': 每次迭代中拥有最大奖励的终端节点轨迹
                                         'last_iter': 最后一次迭代的轨迹。如果最后一次迭代导致死胡同可能输出非终端节点
                                         'last_terminal_iter': 最后一条包含终端节点的轨迹
                                如果没有包含终端节点的轨迹但有必要，输出*None*
        :param uct_with_fast_reward: 如果为True，使用fast_reward代替未访问子节点的奖励
                                     否则，先访问具有最大fast_reward的未访问子节点
        """
        super().__init__()
        self.world_model = None  # 初始化世界模型为空
        self.search_config = None  # 初始化搜索配置为空

        self.output_trace_in_each_iter = (
            output_trace_in_each_iter  # 设置是否在每次迭代中输出轨迹
        )

        self.w_exp = w_exp  # 设置探索权重
        self.mode = mode  # 设置探索权重
        self.depth_limit = depth_limit  # 设置深度限制
        self.n_iters = n_iters  # 设置迭代次数
        self.cum_reward = cum_reward  # 设置累积奖励计算方法
        self.calc_q = calc_q  # 设置Q值计算方法

        default_simulate_strategies: dict[str, Callable[
            [list[float]], int]] = {
                "max": lambda x: np.argmax(x),  # 使用最大值策略
                "sample": lambda x: np.random.choice(len(x), p=x),  # 使用抽样策略
                "random": lambda x: np.random.choice(len(x)),  # 使用随机选择策略
            }

        self.simulate_choice: Callable[[list[float]],
                                       int] = (default_simulate_strategies.get(
                                           simulate_strategy,
                                           simulate_strategy))  # 选择模拟策略

        assert output_strategy in [
            "max_reward",
            "follow_max",
            "max_visit",
            "max_iter",
            "last_iter",
            "last_terminal_iter",
        ]  # 验证输出策略的有效性

        self.output_strategy = output_strategy  # 设置输出策略

        self.uct_with_fast_reward = (
            uct_with_fast_reward  # 设置是否使用快速奖励进行UCT选择
        )

        self._output_iter: list[MCTSNode] = None  # 初始化输出迭代列表为空
        self._output_cum_reward = -math.inf  # 初始化输出累积奖励为负无穷大

        self.trace_in_each_iter: list[list[MCTSNode]] = (
            None  # 初始化每次迭代的轨迹列表为空
        )

        self.root: Optional[MCTSNode] = None  # 初始化根节点为空
        self.disable_tqdm = disable_tqdm  # 设置是否禁用进度条显示
        self.node_visualizer = node_visualizer  # 设置节点可视化方法
        self.aggregator = aggregator  # 设置聚合器
        self.node_visualizer = node_visualizer  # 设置节点可视化方法
        self.aggregator = aggregator  # 设置聚合器

    def iterate(self, node: MCTSNode) -> list[MCTSNode]:
        """
        执行一次MCTS迭代过程，从选择节点到模拟和回溯更新。

        :param node: MCTS树的根节点
        :return: 执行路径上的节点列表
        """
        path = self._select(node)  # 选择要扩展的路径（从根节点开始）

        if not self._is_terminal_with_depth_limit(
                path[-1]):  # 检查路径的最后一个节点是否为终端状态或达到深度限制
            self._expand(path[-1])  # 如果不是终端状态或未达到深度限制，则扩展该节点
            self._simulate(path)  # 模拟新的路径

        cum_reward = self._back_propagate(path)  # 回溯传播累积奖励到路径上的每个节点

        # 根据不同的输出策略更新输出
        if (self.output_strategy == "max_iter" and path[-1].is_terminal
                and cum_reward > self._output_cum_reward):
            self._output_cum_reward = cum_reward  # 如果是最大迭代策略并且路径最后一个节点是终端且累积奖励更大，更新输出的累积奖励
            self._output_iter = path  # 更新输出的路径

        if self.output_strategy == "last_iter":  # 如果是最后一次迭代策略
            self._output_cum_reward = cum_reward  # 更新输出的累积奖励
            self._output_iter = path  # 更新输出的路径

        if (self.output_strategy == "last_terminal_iter"
                and path[-1].is_terminal):  # 如果是最后终端节点策略并且路径最后一个节点是终端
            self._output_cum_reward = cum_reward  # 更新输出的累积奖励
            self._output_iter = path  # 更新输出的路径

        return path  # 返回选择和扩展的路径

    def _is_terminal_with_depth_limit(self, node: MCTSNode) -> bool:
        """
        检查节点是否为终端状态或已达到深度限制。

        :param node: MCTS树中的节点
        :return: 布尔值，表示节点是否为终端或达到深度限制
        """
        # 返回节点是否为终端状态或其深度超过预设的深度限制
        return (node.is_terminal or node.depth >= self.depth_limit)

    def _select(self, node: MCTSNode) -> list[MCTSNode]:
        """
        选择一条路径进行扩展。

        :param node: MCTS树的根节点
        :return: 路径上的节点列表
        """
        path = []  # 初始化路径为空列表
        while True:  # 无限循环，直到找到需要扩展的节点
            path.append(node)  # 将当前节点添加到路径中

            # 如果节点没有子节点，或者所有子节点都访问过，或者节点是终端状态或达到深度限制
            if (node.children is None or len(node.children) == 0
                    or self._is_terminal_with_depth_limit(node)):
                return path  # 返回已选择的路径

            node = self._uct_select(node)  # 使用UCT算法选择下一个节点

    def _uct(self, node: MCTSNode) -> float:
        """
        计算节点的UCT值，用于选择最佳的子节点。

        :param node: MCTS树中的节点
        :return: 节点的UCT值
        """
        # 获取Q值（节点的平均奖励）
        Q_value = node.Q

        if self.mode == "sc":
            self.w_exp = 50
        else:
            self.w_exp = 1

        # 计算探索项的分子：父节点的访问次数的对数
        exploration_numerator = np.log(len(node.parent.cum_rewards))

        # 计算探索项的分母：当前节点的访问次数（至少为1）
        exploration_denominator = max(1, len(node.cum_rewards))

        # 计算探索项的值
        exploration_term = self.w_exp * np.sqrt(
            exploration_numerator / exploration_denominator)

        print(
            f"Q_value: {Q_value}, w_exp: {self.w_exp}, exploration_term: {exploration_term}"
        )

        uct_value = Q_value + exploration_term

        return uct_value

    def _uct_select(self, node: MCTSNode) -> MCTSNode:
        """
        使用UCT算法从节点的子节点中选择最佳节点。

        :param node: 当前节点
        :return: 选择的子节点
        """

        print("=" * 30 + " node selection " + "=" * 30)
        for x in node.children:
            uct_value = self._uct(x)  # 计算当前节点的UCT值
            print(f"Child ID: {x.id}, UCT Value: {uct_value}")
        print("=" * 30 + " node selection " + "=" * 30)

        if self.uct_with_fast_reward or all(x.state is not None
                                            for x in node.children):
            return max(node.children, key=self._uct)  # 返回UCT值最大的子节点

        else:
            unvisited_children = filter(lambda x: x.state is None,
                                        node.children)  # 找到所有未访问的子节点
            return max(unvisited_children,
                       key=lambda x: x.fast_reward)  # 返回快速奖励最大的未访问子节点

    def _expand(self, node: MCTSNode):
        """
        扩展一个节点，生成其所有可能的子节点。

        :param node: MCTS树中的节点
        """

        if node.state is None:  # 如果节点的状态为空
            node.state, aux = self.world_model.step(
                node.parent.state, node.action)  # 使用世界模型获取新的状态和辅助信息

            # 奖励在状态更新后计算，以便信息可以从世界模型缓存并传递给奖励函数，避免重复计算
            node.reward, node.reward_details = self.search_config.reward(
                node.parent.state,
                node.action,
                reward_dict=node.fast_reward_details,
                **aux)

            node.is_terminal = self.world_model.is_terminal(
                node.state)  # 更新节点的终端状态标识

        if node.is_terminal:  # 如果节点是终端状态
            return  # 不再扩展，直接返回

        children = []  # 初始化子节点列表

        actions = self.search_config.get_actions(node.state)  # 获取当前状态的所有可能动作

        for action in actions:  # 对每个动作
            fast_reward, fast_reward_details = self.search_config.fast_reward(
                node.state, action, node_id=node.id)

            # 创建新的子节点，并添加到子节点列表中
            child = MCTSNode(
                state=None,
                action=action,
                parent=node,
                fast_reward=fast_reward,
                fast_reward_details=fast_reward_details,
                calc_q=self.calc_q,
            )

            children.append(child)

        node.children = children  # 将生成的子节点列表分配给当前节点

    def _simulate(self, path: list[MCTSNode]):
        """
        模拟路径上的后续步骤，直至到达终端状态或达到深度限制。

        :param path: 当前路径上的节点列表
        """
        node = path[-1]  # 获取路径中的最后一个节点
        while True:
            if node.state is None:  # 如果节点状态为空
                self._expand(node)  # 扩展节点以生成子节点
            if (self._is_terminal_with_depth_limit(node)
                    or len(node.children) == 0):  # 如果节点是终端状态或达到深度限制
                return  # 结束模拟
            fast_rewards = [child.fast_reward
                            for child in node.children]  # 获取所有子节点的快速奖励
            node = node.children[self.simulate_choice(
                fast_rewards)]  # 根据模拟策略选择下一个子节点
            path.append(node)  # 将选择的子节点添加到路径中

    def _back_propagate(self, path: list[MCTSNode]) -> float:
        """
        进行回溯更新路径上的每个节点的累积奖励。

        :param path: 路径上的节点列表
        :return: 累积奖励
        """
        rewards = []  # 初始化奖励列表
        cum_reward = -math.inf  # 初始化累积奖励为负无穷大

        for node in reversed(path):  # 从路径末端向前遍历节点
            rewards.append(node.reward)  # 将节点的奖励添加到奖励列表中
            cum_reward = self.cum_reward(rewards[::-1])  # 计算反向奖励的累积值
            node.cum_rewards.append(cum_reward)  # 更新节点的累积奖励列表

        return cum_reward  # 返回累积奖励

    def _dfs_max_reward(self,
                        path: list[MCTSNode]) -> tuple[float, list[MCTSNode]]:
        """
        使用深度优先搜索找到具有最大累积奖励的路径。

        :param path: 当前路径上的节点列表
        :return: 最大累积奖励和对应的路径
        """
        cur = path[-1]  # 获取路径中的最后一个节点

        if cur.is_terminal:  # 如果当前节点是终端状态
            # 返回累积奖励和路径
            return (self.cum_reward([node.reward for node in path[1:]]), path)

        if cur.children is None:  # 如果当前节点没有子节点
            return -math.inf, path  # 返回负无穷大和当前路径

        visited_children = [x for x in cur.children
                            if x.state is not None]  # 获取所有已访问的子节点

        if len(visited_children) == 0:  # 如果没有已访问的子节点
            return -math.inf, path  # 返回负无穷大和当前路径

        return max((self._dfs_max_reward(path + [child])
                    for child in visited_children),
                   key=lambda x: x[0])  # 递归查找最大奖励路径

    def search(self):
        """
        执行MCTS搜索，找到最优解路径。
        """
        self._output_cum_reward = -math.inf  # 初始化输出累积奖励为负无穷大
        self._output_iter = None  # 初始化输出路径为空

        self.root = MCTSNode(state=self.world_model.init_state(),
                             action=None,
                             parent=None,
                             calc_q=self.calc_q)  # 创建根节点

        if self.output_trace_in_each_iter:
            self.trace_in_each_iter = []  # 如果需要输出每次迭代的轨迹，则初始化轨迹列表为空

        # 运行n_iters次迭代
        for i in trange(self.n_iters, disable=True, desc="MCTS iteration"):

            path = self.iterate(self.root)  # 执行一次迭代，获取路径

            if self.output_trace_in_each_iter:
                self.trace_in_each_iter.append(
                    deepcopy(path))  # 如果需要输出轨迹，深拷贝路径并存储

            print(f"\niteration: {i+1}/{self.n_iters}")

        if self.output_strategy == "follow_max":  # 如果输出策略是跟随最大
            self._output_iter = []  # 初始化输出路径为空列表
            cur = self.root  # 从根节点开始

            while True:
                self._output_iter.append(cur)  # 将当前节点添加到输出路径

                if cur.is_terminal:  # 如果当前节点是终端状态
                    break

                visited_children = [
                    x for x in cur.children if x.state is not None
                ]  # 获取已访问的子节点

                if len(visited_children) == 0:  # 如果没有已访问的子节点
                    break

                cur = max(visited_children,
                          key=lambda x: x.reward)  # 选择奖励最高的子节点

            self._output_cum_reward = self.cum_reward(
                [node.reward for node in self._output_iter[1::-1]])  # 更新输出累积奖励

        if self.output_strategy == "max_reward":  # 如果输出策略是最大奖励
            self._output_cum_reward, self._output_iter = self._dfs_max_reward(
                [self.root])  # 使用深度优先搜索找到最大奖励路径

            if self._output_cum_reward == -math.inf:
                self._output_iter = None  # 如果没有找到路径，输出路径设为空

    def __call__(
        self,
        world_model: WorldModel[State, Action, Example],
        search_config: SearchConfig[State, Action, Example],
        log_file: Optional[str] = None,
        **kwargs,
    ) -> MCTSResult:
        """
        运行MCTS算法，返回搜索结果。

        :param world_model: 世界模型
        :param search_config: 搜索配置
        :param log_file: 日志文件（可选）
        :return: MCTS搜索结果
        """
        MCTSNode.reset_id()  # 重置节点ID计数器
        self.world_model = world_model  # 设置世界模型
        self.search_config = search_config  # 设置搜索配置

        self.search()  # 执行搜索

        if self._output_iter is None:  # 如果没有找到终端路径
            terminal_state = trace = None  # 终端状态和路径都设为空
        else:
            terminal_state = self._output_iter[-1].state  # 获取终端状态

            trace = (
                [node.state for node in self._output_iter],
                [node.action for node in self._output_iter[1:]],
            )  # 获取路径中的状态和动作

        # 如果需要输出每次迭代的轨迹
        if self.output_trace_in_each_iter:
            # 获取每次迭代的轨迹
            trace_in_each_iter = self.trace_in_each_iter

            # 获取每次迭代后的树状态
            tree_state_after_each_iter = [
                trace[0] for trace in trace_in_each_iter
            ]
        else:
            # 否则，轨迹和树状态都设为空
            trace_in_each_iter = tree_state_after_each_iter = (None)

        result = MCTSResult(
            terminal_state=terminal_state,
            cum_reward=self._output_cum_reward,
            trace=trace,
            trace_of_nodes=self._output_iter,
            tree_state=self.root,
            trace_in_each_iter=trace_in_each_iter,
            tree_state_after_each_iter=tree_state_after_each_iter,
        )  # 创建MCTS结果对象

        if self.aggregator is not None:  # 如果存在聚合器
            result = MCTSResult(
                terminal_state=result.terminal_state,
                cum_reward=result.cum_reward,
                trace=result.trace,
                trace_of_nodes=result.trace_of_nodes,
                tree_state=result.tree_state,
                trace_in_each_iter=result.trace_in_each_iter,
                tree_state_after_each_iter=result.tree_state_after_each_iter,
                aggregated_result=self.aggregator(
                    result.tree_state),  # 使用聚合器对结果进行聚合
            )

        return result  # 返回MCTS搜索结果
