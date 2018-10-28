import numpy as np
from .pytorch_classification.utils import Bar, AverageMeter
import time


class Arena:
    """
    An Arena class where any 2 agents can be pit against each other.
    """

    def __init__(self, player, game, display=None):
        self.player = player
        self.game = game
        self.display = display

    def playGame(self, verbose=False):
        env_state = self.game.get_initial_state()
        it = 0
        next_state = self.game.getGameEnded(env_state)
        while next_state == 0:
            it += 1
            if verbose and self.display:
                self.display(env_state)
            action = self.player(env_state)

            valids = self.game.getValidMoves(env_state)

            if valids[action] == 0:
                print(action)
                assert valids[action] > 0
            env_state = self.game.get_next_state(env_state, action)
            next_state = self.game.getGameEnded(env_state)

        # Display the final move
        if verbose:
            assert self.display
            self.display(env_state)

        is_win = next_state == 1
        if verbose:
            if is_win:
                outcome_str = "Problem Solved"
            else:
                outcome_str = "Failed"
            print("\n\t\tResult: {}\n\n".format(outcome_str))
        return is_win

    def playGames(self, num, verbose=False):
        """
        Plays num games in which player1 starts num/2 games and player2 starts
        num/2 games.

        Returns:
            solved: number of problems solved
            failed: number of problems unsolved
        """
        eps_time = AverageMeter()
        bar = Bar("Problem.solve", max=num)
        end = time.time()
        eps = 0
        maxeps = int(num)

        solved = 0
        failed = 0
        for _ in range(num):
            is_win = self.playGame(verbose=verbose)
            if is_win:
                solved += 1
            else:
                failed += 1

            # bookkeeping + plot progress
            eps += 1
            eps_time.update(time.time() - end)
            end = time.time()
            bar.suffix = "({eps}/{maxeps}) Eps Time: {et:.3f}s | Total: {total:} | ETA: {eta:}".format(
                eps=eps + 1,
                maxeps=maxeps,
                et=eps_time.avg,
                total=bar.elapsed_td,
                eta=bar.eta_td,
            )
            bar.next()
        bar.finish()

        return solved, failed
