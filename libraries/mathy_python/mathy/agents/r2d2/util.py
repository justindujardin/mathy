from colr import color
from ..base_config import BaseConfig
from typing import Any
import datetime
from multiprocessing import Process
from threading import Thread

is_debug = True

MPClass: Any = Process if is_debug is False else Thread


# From openai baselines: https://bit.ly/30EvCzy
def cat_entropy(logits):
    import tensorflow as tf

    a0 = logits - tf.reduce_max(logits, 1, keepdims=True)
    ea0 = tf.exp(a0)
    z0 = tf.reduce_sum(ea0, 1, keepdims=True)
    p0 = ea0 / z0
    return tf.reduce_sum(p0 * (tf.math.log(z0) - a0), 1)


def truncate(value):
    return float("%.3f" % (float(value)))


global_ep_reward = 0.0


def record(episode_reward, worker_idx, num_steps, env_name, exp_buffer_full: bool):
    global global_ep_reward

    now = datetime.datetime.now().strftime("%H:%M:%S")

    global_ep_reward = global_ep_reward * 0.95 + episode_reward * 0.05

    fore = "green" if episode_reward > 0.0 else "red"

    if exp_buffer_full:
        print(
            color(
                f"{now} "
                f"reward(avg:{truncate(global_ep_reward)} ep:{truncate(episode_reward)}) "
                f"steps({num_steps}) "
                f"worker{worker_idx}: {env_name}",
                fore=fore,
                style="bright",
            )
        )
    return global_ep_reward


def record_losses(step, pi_loss, value_loss, entropy_loss, aux_losses, total_loss):
    now = datetime.datetime.now().strftime("%H:%M:%S")

    def truncate(value):
        return float("%.3f" % (float(value)))

    fore = "blue"
    losses = [
        f"step: {step} total: {truncate(total_loss)} "
        f"pi: {truncate(pi_loss)} "
        f"value: {truncate(value_loss)} "
        f"entropy: {truncate(entropy_loss)} "
    ]
    if isinstance(aux_losses, dict):
        for k in aux_losses.keys():
            losses.append(f"{k}: {truncate(aux_losses[k])} ")
    loss_str = f"losses( {' '.join(losses)}) "

    print(color(f"{now} {loss_str}", fore=fore, style="bright"))
