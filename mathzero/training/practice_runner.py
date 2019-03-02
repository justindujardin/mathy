import time
from multiprocessing import Array, Pool, Process, Queue, cpu_count
from random import shuffle
from sys import stdin
import numpy
from ..environment_state import MathEnvironmentState
from ..util import (
    LOSE_REWARD,
    WIN_REWARD,
    discount_rewards,
    is_terminal_reward,
    normalize_rewards,
)
from .mcts import MCTS
from ..math_game import MathGame
from ..core.expressions import MathExpression
from ..model.math_model import MathModel


class RunnerConfig:
    """
    Set configuration options for the episode runner that control how the 
    episodes play out.
    """

    def __init__(
        self,
        num_wokers=cpu_count(),
        num_mcts_sims=15,
        temperature_threshold=0.5,
        cpuct=1.0,
        model_dir=None,
    ):
        self.num_wokers = num_wokers
        self.num_mcts_sims = num_mcts_sims
        self.temperature_threshold = temperature_threshold
        self.cpuct = cpuct
        self.model_dir = model_dir


class PracticeRunner:
    """
    Instance that controls how episodes are executed. By default this class executes episodes serially
    in a single process. This is great for debugging problems in an interactive debugger or running locally
    but is not ideal for machines with many processors available. For multiprocessing swap out the default 
    `PracticeRunner` class for the `ParallelPracticeRunner` class that is defined below.    
    """

    def __init__(self, config):
        if config is None or not isinstance(config, RunnerConfig):
            raise ValueError("configuration must be an instance of RunnerConfig")
        self.config = config

    def get_game(self):
        raise NotImplementedError("game implementation must be provided by subclass")

    def get_predictor(self, game, all_memory=False):
        raise NotImplementedError(
            "predictor implementation must be provided by subclass"
        )

    def act(
        self,
        game: MathGame,
        env_state: MathEnvironmentState,
        mcts: MCTS,
        model: MathModel,
        move_count,
        history,
    ):
        """Take an action, criticize it and return the next state. If the next state
        is terminal, return training examples for the episode feature, policy, and focus
        values.

        The process follows these steps:
           1. Simulate an action roll-out using MCTS and return a probability
              distribution over the actions that the agent can take.
           2. Choose an action and produce a next_state environment.

        returns: A tuple of (new_env_state, terminal_results_or_none)
        
        NOTE: This is an attempt to combine Actor-Critic with MCTS. I'm not sure if it's 
        correct, but I suppose the training will tell us pretty quickly. If anyone else 
        sees this message, let's talk about how it turned out, and what kind of docstring 
        should go here.
        """

        # Hold on to the episode example data for training the neural net
        state = env_state.clone()
        example_data = state.to_input_features()

        # If the move_count is less than threshold, set temp = 1 else 0
        temp = int(move_count < self.config.temperature_threshold)
        pi = mcts.getActionProb(state, temp=temp)
        action = numpy.random.choice(len(pi), p=pi)

        # Calculate focus value for output into examples file. This is a
        # "forced" focus based on the selected action from the MCTS search.
        expression = state.parser.parse(state.agent.problem)
        node_index, _ = game.get_focus_at_index(env_state, action, expression)
        state.agent.focus = node_index / expression.countNodes()

        # Calculate the next state based on the selected action
        next_state, next_state_reward, is_done = game.get_next_state(state, action)

        # Where to focus for the next comes from the network predictions
        next_state.agent.focus = mcts.getFocusProb(next_state)

        example_text = next_state.agent.problem
        r = game.get_state_reward(next_state)
        is_term = is_terminal_reward(r)
        is_win = True if is_term and r > 0 else False
        if is_term:
            r = abs(r - WIN_REWARD) if is_win else r - LOSE_REWARD
        history.append([example_data, pi, r, example_text, state.agent.focus])

        # Keep going if the reward signal is not terminal
        if not is_term:
            return next_state, None

        rewards = [x[2] for x in history]
        discounts = list(discount_rewards(rewards))

        # Note that we insert negative(max_discounted_reward) so that normalizing
        # the reward will never make a WIN value that is less than 0 or a lose that is
        # greater than 0.
        if is_win:
            anchor_discount = -numpy.max(discounts)
        else:
            anchor_discount = abs(numpy.min(discounts))
        # Compute the normalized values, and slice off the last (anchor) element
        rewards = normalize_rewards(discounts + [anchor_discount])[:-1]

        # If we're losing, reverse the reward values so they get more negative as
        # the agent approaches the losing move.
        if not is_win:
            numpy.flip(rewards)

        # Floating point is tricky, so sometimes the values near zero will flip
        # their signs, so force the appropriate sign. This is okay(?) because it's
        # only small value changes, e.g. (-0.001 to 0.001)
        if is_win:
            rewards = [min(abs(r) + 1e-3, 1.0) for r in rewards]
        else:
            rewards = [max(-abs(r) - 1e-3, -1) for r in rewards]
        examples = []
        for i, x in enumerate(history):
            examples.append(
                {
                    "reward": float(rewards[i]),
                    "focus": float(x[4]),
                    "before": x[3],
                    "policy": x[1],
                    "inputs": x[0],
                }
            )
        episode_reward = rewards[0]
        return next_state, (examples, episode_reward, is_win)

    def execute_episodes(self, episode_args_list):
        """
        Execute (n) episodes of self-play serially. This is mostly useful for debugging, and
        when you cannot fit multiple copies of your model in GPU memory
        """
        examples = []
        results = []

        game = self.get_game()
        predictor = self.get_predictor(game)
        predictor.start()
        for i, args in enumerate(episode_args_list):
            start = time.time()
            episode_examples, episode_reward, is_win, episode_complexity = self.execute_episode(
                i, game, predictor, **args
            )
            duration = time.time() - start
            examples.extend(episode_examples)
            episode_summary = dict(
                solved=bool(is_win),
                complexity=episode_complexity,
                reward=episode_reward,
                duration=duration,
            )
            results.append(episode_summary)
            self.episode_complete(i, episode_summary)
        predictor.stop()
        return examples, results

    def execute_episode(self, episode, game, predictor, model):
        """
        This function executes one episode.
        As the game is played, each turn is added as a training example to
        trainExamples. The game continues until get_state_reward returns a non-zero
        value, then the outcome of the game is used to assign values to each example
        in trainExamples.
        """
        if game is None:
            raise NotImplementedError("PracticeRunner.get_game returned None type")
        if predictor is None:
            raise NotImplementedError("PracticeRunner.get_predictor returned None type")

        env_state, complexity = game.get_initial_state()

        episode_history = []
        move_count = 0
        mcts = MCTS(game, predictor, self.config.cpuct, self.config.num_mcts_sims)
        while True:
            move_count += 1
            env_state, result = self.act(
                game, env_state, mcts, predictor, move_count, episode_history
            )
            if result is not None:
                return result + (complexity,)

    def episode_complete(self, episode, summary):
        """Called after each episode completes. Useful for things like updating progress indicators"""
        pass

    def process_trained_model(
        self, updated_model, iteration, train_examples, model_path
    ):
        if updated_model is None:
            return False
        # updated_model.save_checkpoint(model_path)
        return True

    def train(self, iteration, train_examples, model_path=None):
        """
        Train the model at the given checkpoint path with the training examples and return
        the updated model or None if there was an error.
        """
        return self.process_trained_model(
            self.train_with_examples(iteration, train_examples, model_path),
            iteration,
            train_examples,
            model_path,
        )

    def train_with_examples(self, iteration, train_examples, model_path=None):
        game = self.get_game()
        new_net = self.get_predictor(game, True)
        # Train the model with the examples
        if new_net.train(train_examples) == False:
            print(
                "There are not at least batch-size examples for training, more self-play is required..."
            )
            return None
        return new_net


class ParallelPracticeRunner(PracticeRunner):
    """Run (n) parallel self-play or training processes in parallel."""

    def execute_episodes(self, episode_args_list):
        def worker(work_queue, result_queue):
            """Pull items out of the work queue and execute episodes until there are no items left"""
            game = self.get_game()
            predictor = self.get_predictor(game)
            predictor.start()
            while work_queue.empty() == False:
                episode, args = work_queue.get()
                start = time.time()
                try:
                    episode_examples, episode_reward, is_win, episode_complexity = self.execute_episode(
                        episode, game, predictor, **args
                    )
                except Exception as e:
                    print(
                        "ERROR: self practice thread threw an exception: {}".format(
                            str(e)
                        )
                    )
                    print(e)
                    result_queue.put((i, [], {"error": str(e)}))
                    continue
                duration = time.time() - start
                episode_summary = dict(
                    complexity=episode_complexity,
                    reward=episode_reward,
                    solved=bool(is_win),
                    duration=duration,
                )
                result_queue.put((i, episode_examples, episode_summary))
            predictor.stop()
            return 0

        # Fill a work queue with episodes to be executed.
        work_queue = Queue()
        result_queue = Queue()
        for i, args in enumerate(episode_args_list):
            work_queue.put((i, args))
        processes = [
            Process(target=worker, args=(work_queue, result_queue))
            for i in range(self.config.num_wokers)
        ]
        for proc in processes:
            proc.start()

        # Gather the outputs
        examples = []
        results = []
        count = 0
        while count != len(episode_args_list):
            i, episode_examples, summary = result_queue.get()
            self.episode_complete(i, summary)
            count += 1
            if "error" not in summary:
                examples.extend(episode_examples)
                results.append(summary)

        # Wait for the workers to exit completely
        for proc in processes:
            proc.join()

        return examples, results

    def train(self, iteration, train_examples, model_path=None):
        def train_and_save(output, i, examples, out_path):
            update_model = self.train_with_examples(i, examples, out_path)
            output.put(self.process_trained_model(update_model, i, examples, out_path))

        result_queue = Queue()
        proc = Process(
            target=train_and_save,
            args=(result_queue, iteration, train_examples, model_path),
        )
        proc.start()
        result = result_queue.get()
        proc.join()
        return result
