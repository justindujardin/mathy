from colr import color
from .config import A3CArgs
import datetime


# From openai baselines: https://bit.ly/30EvCzy
def cat_entropy(logits):
    import tensorflow as tf

    a0 = logits - tf.reduce_max(logits, 1, keepdims=True)
    ea0 = tf.exp(a0)
    z0 = tf.reduce_sum(ea0, 1, keepdims=True)
    p0 = ea0 / z0
    return tf.reduce_sum(p0 * (tf.math.log(z0) - a0), 1)


def record(
    episode,
    episode_reward,
    worker_idx,
    global_ep_reward,
    result_queue,
    pi_loss,
    value_loss,
    entropy_loss,
    total_loss,
    num_steps,
    env_name,
):
    """Helper function to store score and print statistics.
  Arguments:
    episode: Current episode
    episode_reward: Reward accumulated over the current episode
    worker_idx: Which thread (worker)
    global_ep_reward: The moving average of the global reward
    result_queue: Queue storing the moving average of the scores
    total_loss: The total loss accumualted over the current episode
    num_steps: The number of steps the episode took to complete
  """

    now = datetime.datetime.now().strftime("%H:%M:%S")

    def truncate(value):
        return float("%.3f" % (float(value)))

    global_ep_reward = global_ep_reward * 0.95 + episode_reward * 0.05

    fore = "green" if episode_reward > 0.0 else "red"
    print(
        color(
            f"{now} ep{episode} "
            f"reward(avg:{truncate(global_ep_reward)} ep:{truncate(episode_reward)}) "
            f"loss(total: {truncate(total_loss)} "
            f"pi: {truncate(pi_loss)} "
            f"value: {truncate(value_loss)} "
            f"entropy: {truncate(entropy_loss)}) "
            f"steps({num_steps}) "
            f"worker{worker_idx}: {env_name}",
            fore=fore,
            style="bright",
        )
    )
    result_queue.put(global_ep_reward)
    return global_ep_reward
