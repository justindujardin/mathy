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
    ):
        self.num_wokers = num_wokers
        self.num_mcts_sims = num_mcts_sims
        self.temperature_threshold = temperature_threshold
        self.cpuct = cpuct


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

    def get_nnet(self, game, all_memory=False):
        raise NotImplementedError(
            "neural net implementation must be provided by subclass"
        )

    def execute_episodes(self, episode_args_list):
        """
        Execute (n) episodes of self-play serially. This is mostly useful for debugging, and
        when you cannot fit multiple copies of your model in GPU memory
        """
        results = []

        game = self.get_game()
        nnet = self.get_nnet(game)
        for i, args in enumerate(episode_args_list):
            start = time.time()
            results.extend(self.execute_episode(i, game, nnet, **args))
            duration = time.time() - start
            self.episode_complete(i, duration)
        return results

    def execute_episode(self, episode, game, nnet, player, model, **kwargs):
        """
        This function executes one episode of self-play, starting with player 1.
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
        if nnet is None:
            raise NotImplementedError("EpisodeRunner.get_nnet returned None type")
        if model is not None:
            if nnet.can_load_checkpoint(model):
                nnet.load_checkpoint(model)
        episode_examples = []
        env_state = game.get_initial_state()
        current_player = player
        move_count = 0
        mcts = MCTS(game, nnet, self.config.cpuct, self.config.num_mcts_sims)
        while True:
            move_count += 1
            canonical_state = game.getCanonicalForm(env_state, current_player)
            temp = int(move_count < self.config.temperature_threshold)

            pi = mcts.getActionProb(canonical_state, temp=temp)
            # Store the episode example data for training the neural net
            example_data = canonical_state
            if hasattr(example_data, 'to_numpy'):
                example_data = example_data.to_numpy()
            episode_examples.append([example_data, current_player, pi, None])
            action = numpy.random.choice(len(pi), p=pi)
            env_state, current_player = game.get_next_state(env_state, current_player, action)
            r = game.getGameEnded(env_state, current_player)

            if r != 0:
                return [
                    (x[0], x[2], r * ((-1) ** (x[1] != current_player)))
                    for x in episode_examples
                ]

        return []

    def episode_complete(self, episode, duration):
        """Called after each episode completes. Useful for things like updating progress indicators"""
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
        new_net = self.get_nnet(game, True)
        has_best = new_net.can_load_checkpoint(model_path)
        if has_best:
            new_net.load_checkpoint(model_path)

        # shuffle examlpes before training
        shuffle(train_examples)

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
            nnet = self.get_nnet(game)
            while work_queue.empty() == False:
                episode, args = work_queue.get()
                start = time.time()
                result = self.execute_episode(episode, game, nnet, **args)
                duration = time.time() - start
                result_queue.put((i, result, duration))
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
        results = []
        count = 0
        while count != len(episode_args_list):
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
