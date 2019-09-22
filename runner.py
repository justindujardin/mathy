import time
from multiprocessing import Array, Pool, Process, Queue, cpu_count
from pickle import Pickler
from random import shuffle
from typing import Any, List, Optional, Tuple

from pydantic import BaseModel

from mathy import (
    MathyEnv,
    deprecated_MathyEnvEpisodeResult,
    deprecated_MathyEnvObservation,
    MathyEnvState,
)
from mathy.agent.controller import MathModel
from mathy.agent.training.actor_mcts import ActorMCTS
from mathy.agent.training.mcts import MCTS
from mathy.envs.polynomial_grouping import MathyPolynomialGroupingEnv


class RunnerConfig(BaseModel):
    num_workers: int = cpu_count()
    num_mcts_sims: int = 150
    eval: bool = False


# The tuple returned from ExecuteEpisode
ExecuteEpisodeResult = Tuple[
    MathyEnvState,
    deprecated_MathyEnvObservation,
    Optional[deprecated_MathyEnvEpisodeResult],
]


class EpisodeRunner:
    config: RunnerConfig

    def __init__(self, config: RunnerConfig):
        if config is None or not isinstance(config, RunnerConfig):
            raise ValueError("configuration must be an instance of RunnerConfig")
        self.config = config

    def get_env(self):
        raise NotImplementedError("env implementation must be provided by subclass")

    def get_model(self, env: MathyEnv, all_memory=False):
        raise NotImplementedError(
            "neural net implementation must be provided by subclass"
        )

    def execute_episodes(self, num_episodes: int):
        """Execute (n) episodes of self-play serially. This is mostly useful for
        debugging, and when you cannot fit multiple copies of your model in
        GPU memory
        """
        results: List[ExecuteEpisodeResult] = []

        env = self.get_env()
        model = self.get_model(env)
        model.start()
        for i in range(num_episodes):
            start = time.time()
            results.append(self.execute_episode(i, env, model, **args))
            duration = time.time() - start
            self.episode_complete(i, duration)
        model.stop()
        return results

    def execute_episode(
        self, episode: int, env: MathyEnv, model: MathModel, **kwargs
    ) -> ExecuteEpisodeResult:
        if env is None:
            raise NotImplementedError("EpisodeRunner.get_env returned None type")
        if model is None:
            raise NotImplementedError("EpisodeRunner.get_model returned None type")
        env_name = str(env.__class__.__name__)
        print(f"{env_name}")
        # generate a new problem
        env_state, prob = env.get_initial_state()

        # Configure MCTS options for train/eval
        if self.config.eval:
            num_rollouts = 500
            num_exploration_moves = 0
            epsilon = 0.0
        else:
            num_rollouts = 250
            num_exploration_moves = int(env.max_moves * 0.8)
            epsilon = 0.9

        # Execute episode
        mcts = MCTS(env, model, epsilon, num_rollouts)
        actor = ActorMCTS(mcts, num_exploration_moves)
        final_result = None
        time_steps: List[deprecated_MathyEnvObservation] = []
        episode_steps = 0
        while final_result is None:
            episode_steps = episode_steps + 1
            env_state, train_example, final_result = actor.step(
                env, env_state, model, time_steps
            )
        # episode_examples, episode_reward, is_win = final_result
        return final_result

    def episode_complete(self, episode, duration):
        """Called after each episode completes. Useful for things like updating
        progress indicators"""
        pass

    def process_trained_model(
        self, updated_model, iteration, train_examples, model_path
    ):
        if updated_model is None:
            return False
        updated_model.save_checkpoint(model_path)
        examples_file = "{}.examples".format(model_path)
        with open(examples_file, "wb+") as f:
            Pickler(f).dump(train_examples)
        return True

    def train(self, iteration, train_examples, model_path=None):
        """Train the model at the given checkpoint path with the training examples
        and return the updated model or None if there was an error."""
        return self.process_trained_model(
            self.train_with_examples(iteration, train_examples, model_path),
            iteration,
            train_examples,
            model_path,
        )

    def train_with_examples(self, iteration, train_examples, model_path=None):
        env = self.get_env()
        new_net = self.get_model(env, True)
        has_best = new_net.can_load_checkpoint(model_path)
        if has_best:
            new_net.load_checkpoint(model_path)

        # shuffle examlpes before training
        shuffle(train_examples)

        # Train the model with the examples
        new_net.start()
        if new_net.train(train_examples) is False:
            print(f"Need more training examples...")
            return None
        new_net.stop()
        return new_net


class ParallelEpisodeRunner(EpisodeRunner):
    """Run (n) parallel self-play or training processes in parallel."""

    def execute_episodes(self, num_episodes: int):
        def worker(work_queue, result_queue):
            """Pull items out of the work queue and execute episodes until there
            are no items left"""
            env = self.get_env()
            model = self.get_model(env)
            model.start()
            while work_queue.empty() is False:
                episode = work_queue.get()
                start = time.time()
                result = self.execute_episode(episode, env, model)
                duration = time.time() - start
                result_queue.put((i, result, duration))
            model.stop()
            return 0

        # Fill a work queue with episodes to be executed.
        work_queue: Queue = Queue()
        result_queue: Queue = Queue()
        for i in range(num_episodes):
            work_queue.put((i))
        processes = [
            Process(target=worker, args=(work_queue, result_queue))
            for i in range(self.config.num_workers)
        ]
        for proc in processes:
            proc.start()

        # Gather the outputs
        results: List[Tuple[int, Any, float]] = []
        count = 0
        while count < num_episodes:
            i, result, duration = result_queue.get()
            self.episode_complete(i, duration)
            count += 1
            results.extend(result)

        # Wait for the workers to exit completely
        for proc in processes:
            proc.join()

        return results

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


eps = 100
temp = int(eps * 0.5)

args = {
    "training_iterations": 100,
    "self_play_iterations": eps,
    "max_training_examples": 200000,
    "checkpoint": "/mnt/gcs/mzc/agent_1/",
    # "checkpoint": "./training/latest/",
    "best_model_name": "latest",
}

# Single-process implementation for debugging and development
dev_mode = False

BaseEpisodeRunner = EpisodeRunner if dev_mode else ParallelEpisodeRunner


class MathEpisodeRunner(BaseEpisodeRunner):  # type: ignore
    def get_env(self):
        return MathyPolynomialGroupingEnv()

    def get_model(self, env: MathyEnv, all_memory=False):
        return MathModel(env.action_size, "training/dev", all_memory)


if __name__ == "__main__":
    config = RunnerConfig()
    runner = MathEpisodeRunner(config)
    runner.execute_episodes(32)
    print("done, bye")
