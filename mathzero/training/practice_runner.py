import time
from multiprocessing import Array, Pool, Process, Queue, cpu_count
from random import shuffle
from sys import stdin

import numpy

from ..embeddings.actor_mcts import ActorMCTS
from ..core.expressions import MathExpression
from ..environment_state import MathEnvironmentState
from ..embeddings.math_game import MathGame
from ..model.math_model import MathModel
from ..util import is_terminal_transition, normalize_rewards
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
        num_exploration_moves=5,
        cpuct=1.0,
        model_dir=None,
    ):
        self.num_wokers = num_wokers
        self.num_mcts_sims = num_mcts_sims
        self.num_exploration_moves = num_exploration_moves
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
        self.game = None
        self.predictor = None

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

        if self.game is None:
            self.game = self.get_game()
        if self.predictor is not None:
            self.predictor.stop()
        self.predictor = self.get_predictor(self.game)
        self.predictor.start()
        for i, args in enumerate(episode_args_list):
            start = time.time()
            episode_examples, episode_reward, is_win, episode_complexity = self.execute_episode(
                i, self.game, self.predictor, **args
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
        return examples, results

    def execute_episode(self, episode, game, predictor, model):
        """
        This function executes one episode.
        As the game is played, each turn is added as a training example to
        trainExamples. The game continues until get_state_value returns a non-zero
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
        actor = ActorMCTS(mcts, self.config.num_exploration_moves)
        while True:
            move_count += 1
            env_state, result = actor.step(game, env_state, predictor, episode_history)
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

    def train(
        self, iteration, short_term_examples, long_term_examples, model_path=None
    ):
        """
        Train the model at the given checkpoint path with the training examples and return
        the updated model or None if there was an error.
        """
        return self.train_with_examples(
            iteration, short_term_examples, long_term_examples, model_path
        )

    def train_with_examples(
        self, iteration, short_term_examples, long_term_examples, model_path=None
    ):
        if self.predictor is None:
            raise ValueError("predictor must be initialized before training")
        # Train the model with the examples
        if self.predictor.train(short_term_examples, long_term_examples) == False:
            print(
                "There are not at least batch-size examples for training, more self-play is required..."
            )
            return None
        return self.predictor


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

    def train(
        self, iteration, short_term_examples, long_term_examples, model_path=None
    ):
        def train_and_save(output, i, st_examples, lt_examples, out_path):
            update_model = self.train_with_examples(
                i, st_examples, lt_examples, out_path
            )
            output.put(self.process_trained_model(update_model, i, examples, out_path))

        result_queue = Queue()
        proc = Process(
            target=train_and_save,
            args=(
                result_queue,
                iteration,
                short_term_examples,
                long_term_examples,
                model_path,
            ),
        )
        proc.start()
        result = result_queue.get()
        proc.join()
        return result
