import time

import numpy as np

from .lib.average_meter import AverageMeter
from .lib.progress.bar import Bar

from ...util import is_lose_reward, is_terminal_transition, is_win_reward


class ExaminationRunner:
    def __init__(self, agent, game, display=None):
        self.agent = agent
        self.game = game
        self.display = display

    def attempt(self, verbose=False):
        steps = []
        env_state, complexity = self.game.get_initial_state()
        it = 0
        next_state = self.game.get_state_reward(env_state)
        while not is_terminal_transition(next_state):
            it += 1
            if verbose and self.display:
                self.display(env_state)
            steps.append(env_state.agent.problem)
            action = self.agent(env_state)
            valids = self.game.getValidMoves(env_state)
            if valids[action] == 0:
                print(action)
                assert valids[action] > 0
            env_state = self.game.get_next_state(env_state, action)
            next_state = self.game.get_state_reward(env_state)

        # Display the final move
        if verbose:
            assert self.display
            self.display(env_state)
        # Final state
        steps.append(env_state.agent.problem)

        is_win = is_win_reward(next_state)
        if verbose:
            if is_win:
                outcome_str = "Problem Solved"
            else:
                outcome_str = "Failed"
            print("\n\t\tResult: {}\n\n".format(outcome_str))
        return is_win, steps

    def playGames(self, num, verbose=False):
        """
        Evaluate a model by having it attempt to solve num problems.

        Returns:
            solved: number of problems solved
            failed: number of problems unsolved
            details: the problems attempted with each step as ascii math expressions
        """
        details = {"solved": [], "failed": []}
        eps_time = AverageMeter()
        bar = Bar("Problem.solve", max=num)
        end = time.time()
        eps = 0
        maxeps = int(num)

        solved = 0
        failed = 0
        for _ in range(num):
            is_win, steps = self.attempt(verbose=verbose)
            if is_win:
                solved += 1
                details["solved"].append(steps)
            else:
                failed += 1
                details["failed"].append(steps)

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

        return solved, failed, details
