import time
from multiprocessing import Array, Pool, Process, Queue, cpu_count
from random import shuffle
from sys import stdin
from typing import Any, List, Tuple, NamedTuple

import numpy as np
from pydantic import BaseModel

from ...agents.mcts import MCTS
from ...agents.episode_memory import rnn_weighted_history
from ...envs.gym.mathy_gym_env import MathyGymEnv
from ...state import (
    MathyEnvState,
    MathyObservation,
    observations_to_window,
    MathyWindowObservation,
)
from ...util import EnvRewards, discount, is_terminal_transition, pad_array
from ..policy_value_model import PolicyValueModel
from .config import SelfPlayConfig
from .types import EpisodeHistory, EpisodeSummary

from .trainer import SelfPlayTrainer


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

    def get_game(self):
        raise NotImplementedError("game implementation must be provided by subclass")

    def get_predictor(self, game):
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
    ):
        # Hold on to the episode example data for training the neural net
        valids = game.mathy.get_valid_moves(env_state)
        rnn_states = [
            model.embedding.state_h.numpy().tolist(),
            model.embedding.state_c.numpy().tolist(),
        ]
        rnn_history = rnn_weighted_history(
            [o.observation for o in history], len(rnn_states[0][0])
        )
        last_observation: MathyObservation = env_state.to_observation(
            valids, rnn_state=rnn_states, rnn_history=rnn_history
        )

        # If the move_count is less than threshold, set temp = 1 else 0
        temp = int(
            move_count < self.config.temperature_threshold * game.mathy.max_moves
        )
        mcts_state_copy = env_state.clone()
        predicted_policy, value = mcts.estimate_policy(
            mcts_state_copy, rnn_states, temp=temp
        )
        action = np.random.choice(len(predicted_policy), p=predicted_policy)

        next_state, transition, change = game.mathy.get_next_state(env_state, action)
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

        game = self.get_game()
        predictor = self.get_predictor(game)
        for i, args in enumerate(episode_args_list):
            start = time.time()
            (episode_examples, episode_reward, is_win, problem,) = self.execute_episode(
                i, game, predictor, **args
            )
            duration = time.time() - start
            examples.append(episode_examples)
            episode_summary = EpisodeSummary(
                solved=bool(is_win),
                text=problem.text,
                complexity=problem.complexity,
                reward=episode_reward,
                duration=duration,
            )
            results.append(episode_summary)
            self.episode_complete(i, episode_summary)
        return examples, results

    def execute_episode(
        self, episode, game: MathyGymEnv, predictor: PolicyValueModel, model_dir: str
    ):
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
        game.reset()
        env_state = game.state
        episode_history: List[Any] = []
        move_count = 0
        mcts = MCTS(game.mathy, predictor, self.config.cpuct, self.config.mcts_sims)
        while True:
            move_count += 1
            env_state, result = self.step(
                game, env_state, mcts, predictor, move_count, episode_history
            )
            if result is not None:
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
        import tensorflow as tf

        game = self.get_game()
        new_net = self.get_predictor(game)
        trainer = SelfPlayTrainer(self.config, new_net, action_size=new_net.predictions)
        trainer.train(train_examples, new_net)
        new_net.save()

        # def _lazy_examples():
        #     for ex in train_examples:
        #         yield ex

        # class BatchedDataset(tf.keras.utils.Sequence):
        #     def __init__(self, inputs, inputs_len, batch_size=32):
        #         self.inputs = inputs
        #         self.inputs_len = inputs_len
        #         self.batch_size = batch_size

        #     def __len__(self):
        #         return int(np.ceil(self.inputs_len / float(self.batch_size)))

        #     def __getitem__(self, idx):
        #         batch_x = []
        #         batch_y = []
        #         for ex in self.inputs:
        #             text, action, reward, discounted, terminal, observation, pi = ex
        #             batch_x.append(MathyObservation(*observation))
        #             batch_y.append(np.array(pi))
        #             # Stop at terminal states and use a smaller batch
        #             if terminal is True:
        #                 break
        #             if len(batch_x) >= self.batch_size:
        #                 break

        #         batch_x = observations_to_window(batch_x)
        #         # Convert to tensor
        #         batch_x = MathyWindowObservation(
        #             nodes=np.asarray(batch_x.nodes),
        #             mask=np.asarray(batch_x.mask),
        #             type=np.asarray(batch_x.type),
        #             time=np.asarray(batch_x.time),
        #             values=np.asarray(batch_x.values),
        #             rnn_state=np.asarray(batch_x.rnn_state),
        #             rnn_history=np.asarray(batch_x.rnn_history),
        #         )
        #         batch_y = tf.keras.preprocessing.sequence.pad_sequences(batch_y)
        #         return batch_x, batch_y

        # generator = BatchedDataset(_lazy_examples(), len(train_examples))
        # new_net.fit_generator(
        #     callbacks=[
        #         # tf.keras.callbacks.TensorBoard(
        #         #     log_dir=log_dir, histogram_freq=1, write_graph=True,
        #         # ),
        #         # ModelBurnInState(args, model)
        #     ],
        #     epochs=10,
        #     shuffle=False,
        #     generator=generator,
        #     verbose=1,
        #     workers=4,
        # )

        # # if new_net.train(train_examples) is False:
        # #     print("Need batch-size examples before training...")
        # #     return None
        # return new_net


class ParallelPracticeRunner(PracticeRunner):
    """Run (n) parallel self-play or training processes in parallel."""

    def execute_episodes(
        self, episode_args_list
    ) -> Tuple[List[EpisodeHistory], List[EpisodeSummary]]:
        def worker(work_queue, result_queue):
            """Pull items out of the work queue and execute episodes until there are
            no items left"""
            game = self.get_game()
            predictor = self.get_predictor(game)
            while work_queue.empty() is False:
                episode, args = work_queue.get()
                start = time.time()
                try:
                    (
                        episode_examples,
                        episode_reward,
                        is_win,
                        problem,
                    ) = self.execute_episode(episode, game, predictor, **args)
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
        work_queue = Queue()
        result_queue = Queue()
        for i, args in enumerate(episode_args_list):
            work_queue.put((i, args))
        processes = [
            Process(target=worker, args=(work_queue, result_queue))
            for i in range(self.config.num_workers)
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
                examples.append(episode_examples)
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
