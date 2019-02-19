from sys import stdin
from multiprocessing import Pool, Array, Process, Queue, cpu_count
from .NeuralNet import NeuralNet
from .MCTS import MCTS
from .Game import Game
import time
import numpy
from random import shuffle
from pickle import Pickler


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


class EpisodeRunner:
    """
    Instance that controls how episodes are executed. By default this class executes episodes serially
    in a single process. This is great for debugging problems in an interactive debugger or running locally
    but is not ideal for machines with many processors available. For multiprocessing swap out the default 
    `EpisodeRunner` class for the `ParallelEpisodeRunner` class that is defined below.    
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
        for i, args in enumerate(episode_args_list):
            start = time.time()
            episode_examples, episode_reward, episode_complexity = self.execute_episode(
                i, game, predictor, **args
            )
            duration = time.time() - start
            examples.extend(episode_examples)
            episode_summary = dict(
                complexity=episode_complexity, reward=episode_reward, duration=duration
            )
            results.append(episode_summary)
            self.episode_complete(i, episode_summary)
        predictor.destroy()
        return examples, results

    def execute_episode(self, episode, game, predictor, model, **kwargs):
        """
        This function executes one episode.
        As the game is played, each turn is added as a training example to
        trainExamples. The game continues until getGameEnded returns a non-zero
        value, then the outcome of the game is used to assign values to each example
        in trainExamples.

        It uses a temp=1 if episodeStep < temperature_threshold, and thereafter
        uses temp=0.

        Returns:
            trainExamples: a list of examples of the form (canonicalBoard,pi,v)
                            pi is the MCTS informed policy vector, v is +1 if
                            the player eventually won the game, else -1.
        """
        if game is None:
            raise NotImplementedError("EpisodeRunner.get_game returned None type")
        if predictor is None:
            raise NotImplementedError("EpisodeRunner.get_predictor returned None type")
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
            episode_examples.append([example_data, pi, None])
            action = numpy.random.choice(len(pi), p=pi)
            env_state = game.get_next_state(env_state, action)
            # TODO: support scalar reward that's not {1,0,-1,0.0000001}
            r = game.getGameEnded(env_state)

            if r != 0:
                examples = [
                    {"reward": r, "inputs": x[0], "policy": x[1]}
                    for x in episode_examples
                ]
                return examples, r, complexity

        return [], -1, complexity

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


class ParallelEpisodeRunner(EpisodeRunner):
    """Run (n) parallel self-play or training processes in parallel."""

    def execute_episodes(self, episode_args_list):
        def worker(work_queue, result_queue):
            """Pull items out of the work queue and execute episodes until there are no items left"""
            game = self.get_game()
            nnet = self.get_predictor(game)
            while work_queue.empty() == False:
                episode, args = work_queue.get()
                start = time.time()
                episode_examples, episode_reward, episode_complexity = self.execute_episode(
                    episode, game, nnet, **args
                )
                duration = time.time() - start
                episode_summary = dict(
                    complexity=episode_complexity,
                    reward=episode_reward,
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
