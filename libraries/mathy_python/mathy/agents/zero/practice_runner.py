import os
import queue
import time
from multiprocessing import Array, Pool, Process, Queue, cpu_count
from random import shuffle
from sys import stdin
from typing import Any, List, Tuple

import numpy as np
from pydantic import BaseModel
from wasabi import msg

from ...agents.mcts import MCTS
from ...envs.gym.mathy_gym_env import MathyGymEnv
from ...state import (
    MathyEnvState,
    MathyObservation,
    MathyWindowObservation,
    observations_to_window,
)
from ...util import discount, is_terminal_transition, pad_array, print_error
from ..policy_value_model import PolicyValueModel
from .config import SelfPlayConfig
from .trainer import SelfPlayTrainer
from .types import EpisodeHistory, EpisodeSummary


class PracticeRunner:
    """
    Instance that controls how episodes are executed. By default this class executes episodes serially
    in a single process. This is great for debugging problems in an interactive debugger or running locally
    but is not ideal for machines with many processors available. For multiprocessing swap out the default 
    `PracticeRunner` class for the `ParallelPracticeRunner` class that is defined below.    
    """

    def __init__(self, config: SelfPlayConfig):
        if config is None or not isinstance(config, SelfPlayConfig):
            raise ValueError(
                f"configuration must be an instance of {type(SelfPlayConfig)}"
            )
        self.config = config

    def get_env(self):
        raise NotImplementedError("game implementation must be provided by subclass")

    def get_model(self, game) -> PolicyValueModel:
        raise NotImplementedError(
            "predictor implementation must be provided by subclass"
        )

    def step(
        self,
        game: MathyGymEnv,
        env_state: MathyEnvState,
        mcts: MCTS,
        model: PolicyValueModel,
        move_count,
        history: List[EpisodeHistory],
        is_verbose_worker: bool,
    ):
        import tensorflow as tf

        # Hold on to the episode example data for training the neural net
        valids = game.mathy.get_valid_moves(env_state)
        last_observation: MathyObservation = env_state.to_observation(valids)

        # If the move_count is less than threshold, set temp = 1 else 0
        temp = int(
            move_count < self.config.temperature_threshold * game.mathy.max_moves
        )
        mcts_state_copy = env_state.clone()
        predicted_policy, value = mcts.estimate_policy(mcts_state_copy, temp=temp)
        action = np.random.choice(len(predicted_policy), p=predicted_policy)
        observation, reward, done, meta = game.step(action)
        next_state = game.state
        assert next_state is not None
        transition = meta["transition"]
        if is_verbose_worker and self.config.print_training is True:
            game.render()
        example_text = next_state.agent.problem
        r = float(transition.reward)
        is_term = is_terminal_transition(transition)
        is_win = True if is_term and r > 0 else False
        history.append(
            EpisodeHistory(
                text=example_text,
                action=action,
                reward=r,
                # NOTE: we have to update this when the episode is complete
                discounted=r,
                terminal=is_term,
                observation=last_observation,
                pi=predicted_policy,
                value=float(value),
            )
        )

        # Keep going if the reward signal is not terminal
        if not is_term:
            return next_state, None

        rewards = [x.reward for x in history]
        discounts = list(discount(rewards))
        episode_reward = np.sum(discounts)
        # Build a final history with discounted rewards set
        final_history: List[EpisodeHistory] = []
        for i, h in enumerate(history):
            final_history.append(
                EpisodeHistory(
                    text=h.text,
                    action=h.action,
                    reward=h.reward,
                    discounted=float(discounts[i]),
                    terminal=h.terminal,
                    observation=h.observation,
                    pi=h.pi,
                    value=h.value,
                )
            )

        return next_state, (final_history, episode_reward, is_win)

    def execute_episodes(
        self, episode_args_list
    ) -> Tuple[List[EpisodeHistory], List[EpisodeSummary]]:
        """
        Execute (n) episodes of self-play serially. This is mostly useful for debugging, and
        when you cannot fit multiple copies of your model in GPU memory
        """
        examples = []
        results: List[EpisodeSummary] = []
        if self.config.profile:
            import cProfile

            pr = cProfile.Profile()
            pr.enable()

        try:
            game = self.get_env()
            predictor = self.get_model(game)
            for i, args in enumerate(episode_args_list):
                start = time.time()
                (
                    episode_examples,
                    episode_reward,
                    is_win,
                    problem,
                ) = self.execute_episode(i, game, predictor, **args)
                duration = time.time() - start
                examples.extend(episode_examples)
                episode_summary = EpisodeSummary(
                    solved=bool(is_win),
                    text=problem.text,
                    complexity=problem.complexity,
                    reward=episode_reward,
                    duration=duration,
                )
                results.append(episode_summary)
                self.episode_complete(i, episode_summary)
        except KeyboardInterrupt:
            print("Interrupt received. Exiting.")

        if self.config.profile:
            profile_name = f"worker_0.profile"
            profile_path = os.path.join(self.config.model_dir, profile_name)
            pr.disable()
            pr.dump_stats(profile_path)
            if self.config.verbose:
                print(f"PROFILER: saved {profile_path}")
        return examples, results

    def execute_episode(
        self,
        episode: int,
        game: MathyGymEnv,
        predictor: PolicyValueModel,
        model_dir: str,
        is_verbose_worker: bool = False,
    ):
        """
        This function executes one episode.
        As the game is played, each turn is added as a training example to
        trainExamples. The game continues until get_state_reward returns a non-zero
        value, then the outcome of the game is used to assign values to each example
        in trainExamples.
        """
        if game is None:
            raise ValueError("PracticeRunner.get_env returned None type")
        if predictor is None:
            raise ValueError("PracticeRunner.get_model returned None type")
        game.reset()
        if game.state is None:
            raise ValueError("Cannot start self-play practice with a None game state.")
        env_state = game.state
        episode_history: List[Any] = []
        move_count = 0
        mcts = MCTS(game.mathy, predictor, self.config.cpuct, self.config.mcts_sims)
        if is_verbose_worker and self.config.print_training is True:
            game.render()

        while True:
            move_count += 1
            env_state, result = self.step(
                game=game,
                env_state=env_state,
                mcts=mcts,
                model=predictor,
                move_count=move_count,
                history=episode_history,
                is_verbose_worker=is_verbose_worker,
            )
            if result is not None:
                if is_verbose_worker and self.config.print_training is True:
                    game.render()

                return result + (game.problem,)

    def episode_complete(self, episode: int, summary: EpisodeSummary):
        """Called after each episode completes. Useful for things like updating
        progress indicators"""
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
        Train the model at the given checkpoint path with the training examples and
        return the updated model or None if there was an error."""
        return self.process_trained_model(
            self.train_with_examples(iteration, train_examples, model_path),
            iteration,
            train_examples,
            model_path,
        )

    def train_with_examples(self, iteration, train_examples, model_path=None):
        game = self.get_env()
        new_net = self.get_model(game)
        trainer = SelfPlayTrainer(self.config, new_net, action_size=new_net.predictions)
        if trainer.train(train_examples, new_net):
            new_net.save()


class ParallelPracticeRunner(PracticeRunner):
    """Run (n) parallel self-play or training processes in parallel."""

    request_quit = False

    def execute_episodes(
        self, episode_args_list
    ) -> Tuple[List[EpisodeHistory], List[EpisodeSummary]]:
        def worker(worker_idx: int, work_queue: Queue, result_queue: Queue):
            """Pull items out of the work queue and execute episodes until there are
            no items left """
            game = self.get_env()
            predictor = self.get_model(game)
            msg.good(f"Worker {worker_idx} started.")

            while (
                ParallelPracticeRunner.request_quit is False
                and work_queue.empty() is False
            ):
                episode, args = work_queue.get()
                start = time.time()
                try:
                    (
                        episode_examples,
                        episode_reward,
                        is_win,
                        problem,
                    ) = self.execute_episode(
                        episode,
                        game,
                        predictor,
                        is_verbose_worker=worker_idx == 0,
                        **args,
                    )
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    err = print_error(e, f"Self-practice episode threw")
                    result_queue.put((i, [], {"error": err}))
                    continue
                duration = time.time() - start
                episode_summary = EpisodeSummary(
                    complexity=problem.complexity,
                    text=problem.text,
                    reward=episode_reward,
                    solved=bool(is_win),
                    duration=duration,
                )
                result_queue.put((i, episode_examples, episode_summary))
            return 0

        # Fill a work queue with episodes to be executed.
        work_queue: Queue = Queue()
        result_queue: Queue = Queue()
        for i, args in enumerate(episode_args_list):
            work_queue.put((i, args))
        processes = [
            Process(target=worker, args=(i, work_queue, result_queue), daemon=True)
            for i in range(self.config.num_workers)
        ]
        for proc in processes:
            proc.start()

        # Gather the outputs
        examples = []
        results = []
        count = 0
        while ParallelPracticeRunner.request_quit is False and count != len(
            episode_args_list
        ):
            try:
                i, episode_examples, summary = result_queue.get_nowait()
                self.episode_complete(i, summary)
                count += 1
                if "error" not in summary:
                    examples.extend(episode_examples)
                    results.append(summary)
            except queue.Empty:
                pass

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
