from sys import stdin
from multiprocessing import Pool, Array, Process, Queue, cpu_count
from .NeuralNet import NeuralNet
from .MCTS import MCTS
from .Game import Game
import time
import numpy


class RunnerConfig:
    """
    Set configuration options for the episode runner that control how the 
    episodes play out.
    """

    def __init__(
        self,
        max_workers=cpu_count(),
        num_mcts_sims=15,
        temperature_threshold=0.5,
        cpuct=1.0,
    ):
        self.max_workers = max_workers
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

    def get_nnet(self, game):
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
            results.append(self.execute_episode(i, game, nnet, **args))
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
        episode_examples = []
        board = game.getInitBoard()
        current_player = player
        move_count = 0
        mcts = MCTS(game, nnet, self.config.cpuct, self.config.num_mcts_sims)
        while True:
            move_count += 1
            canonical_state = game.getCanonicalForm(board, current_player)
            temp = int(move_count < self.config.temperature_threshold)

            pi = mcts.getActionProb(canonical_state, temp=temp)
            sym = game.getSymmetries(canonical_state, pi)
            for b, p in sym:
                episode_examples.append([b, current_player, p, None])
            action = numpy.random.choice(len(pi), p=pi)
            board, current_player = game.getNextState(board, current_player, action)
            r = game.getGameEnded(board, current_player)

            if r != 0:
                return [
                    (x[0], x[2], r * ((-1) ** (x[1] != current_player)))
                    for x in episode_examples
                ]

        return []

    def episode_complete(self, episode, duration):
        """Called after each episode completes. Useful for things like updating progress indicators"""
        pass


class ParallelEpisodeRunner(EpisodeRunner):
    """Run (n) parallel self-play processes in parallel."""

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
            for i in range(self.config.max_workers)
        ]
        for proc in processes:
            proc.start()

        # Gather the outputs
        results = []
        while len(results) != len(episode_args_list):
            i, result, duration = result_queue.get()
            self.episode_complete(i, duration)
            results.append(result)

        # Wait for the workers to exit completely
        for proc in processes:
            proc.join()

        return results
