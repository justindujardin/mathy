from colr import color
import datetime


def truncate(value):
    return float("%.3f" % (float(value)))


def record(
    episode,
    episode_reward,
    worker_idx,
    global_ep_reward,
    result_queue,
    pi_loss,
    value_loss,
    entropy_loss,
    aux_losses,
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

    global_ep_reward = global_ep_reward * 0.99 + episode_reward * 0.01

    fore = "green" if episode_reward > 0.0 else "red"

    losses = [
        "total: {:<8}".format(truncate(total_loss)),
        "pi: {:<8}".format(truncate(pi_loss)),
        "v: {:<8}".format(truncate(value_loss)),
        "h: {:<8}".format(truncate(entropy_loss)),
    ]
    if isinstance(aux_losses, dict):
        for k in aux_losses.keys():
            losses.append("{:<4}: {:<8}".format(k, truncate(aux_losses[k])))

    heading = "{:<8} {:<3} {:<8} {:<10}".format(
        now, f"w{worker_idx}", f"ep: {episode}", f"steps: {num_steps}"
    )
    rewards = "r_avg: {:<6} r_ep: {:<6}".format(
        truncate(global_ep_reward), truncate(episode_reward)
    )
    loss_str = " ".join(losses)
    print(
        color(f"{heading} {rewards} {loss_str} [{env_name}]", fore=fore, style="bright")
    )
    result_queue.put(global_ep_reward)
    return global_ep_reward
