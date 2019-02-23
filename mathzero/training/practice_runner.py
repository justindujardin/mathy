import time
from multiprocessing import Array, Pool, Process, Queue, cpu_count
from random import shuffle
from sys import stdin

import numpy

from ..util import (
    LOSE_REWARD,
    WIN_REWARD,
    discount_rewards,
    is_terminal_reward,
    normalize_rewards,
)
from .mcts import MCTS


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
            episode_examples, episode_reward, episode_complexity, is_win = self.execute_episode(
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

        episode_examples = []
        env_state, complexity = game.get_initial_state()

        move_count = 0
        mcts = MCTS(game, predictor, self.config.cpuct, self.config.num_mcts_sims)
        while True:
            move_count += 1
            # If the move_count is less than threshold, set temp = 1 else 0
            temp = int(move_count < self.config.temperature_threshold)

            pi = mcts.getActionProb(env_state.clone(), temp=temp)
            # Store the episode example data for training the neural net
            example_data = env_state.to_input_features()
            action = numpy.random.choice(len(pi), p=pi)
            env_state = game.get_next_state(env_state, action)
            example_text = env_state.agent.problem
            r = game.get_state_reward(env_state)
            is_term = is_terminal_reward(r)
            is_win = True if is_term and r > 0 else False
            if is_term:
                r = abs(r - WIN_REWARD) if is_win else r - LOSE_REWARD
            episode_examples.append([example_data, pi, r, example_text])
            if is_term:
                rewards = [x[2] for x in episode_examples]
                if not is_win:
                    rewards.reverse()
                discounts = list(discount_rewards(rewards))

                if is_win:
                    anchor_discount = -numpy.max(discounts)
                else:
                    anchor_discount = abs(numpy.min(discounts))

                max_discount = [anchor_discount]
                # Note that we insert negative(max_discounted_reward) so that normalizing
                # the reward will never make a WIN value that is less than 0.
                rewards = normalize_rewards(discounts + max_discount)
                # Floating point is tricky, so sometimes the values near zero will flip
                # their signs, so force the appropriate sign. This is okay(?) because it's
                # only small value changes, e.g. (-0.001 to 0.001)
                if is_win:
                    rewards = [min(abs(r) + 1e-3, 1.0) for r in rewards]
                else:
                    rewards = [max(-abs(r) - 1e-3, -1) for r in rewards]
                examples = []
                for i, x in enumerate(episode_examples):
                    examples.append(
                        {
                            "reward": float(rewards[i]),
                            "text": x[3],
                            "policy": x[1],
                            "inputs": x[0],
                        }
                    )
                    pass
                return examples, rewards[0], complexity, is_win

        return [], -1, complexity, False

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
                    episode_examples, episode_reward, episode_complexity, is_win = self.execute_episode(
                        episode, game, predictor, **args
                    )
                except Exception as e:
                    print(
                        "ERROR: self practice thread threw an exception: {}".format(
                            str(e)
                        )
                    )
                    print(e)
                    result_queue.put((i, [], {"input": game.problem, "error": str(e)}))
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
