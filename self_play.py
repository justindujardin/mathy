import plac
from typing import Optional
from multiprocessing import cpu_count
from mathy.agents.zero import SelfPlayConfig, self_play_runner


@plac.annotations(
    model_dir=("The folder to train in.", "positional", None, str),
    transfer_from=("An existing model to load weights from.", "positional", None, str),
    workers=(
        "Number of worker threads to use. More increases diversity of exp",
        "option",
        None,
        int,
    ),
    profile=("Set to gather profiler outputs for the A3C workers", "flag", False, bool),
    show=("Set to gather profiler outputs for the A3C workers", "flag", False, bool),
    max_eps=("Maximum number of episodes to run", "option", None, int),
    evaluate=("Set when evaluation is desired", "flag", False, bool),
)
def main(
    model_dir: str,
    transfer_from: Optional[str] = None,
    workers: int = cpu_count(),
    profile: bool = False,
    show: bool = False,
    max_eps: int = cpu_count(),
    evaluate: bool = False,
):
    args = SelfPlayConfig(
        verbose=True,
        topics=["poly"],
        model_dir=model_dir,
        init_model_from=transfer_from,
        num_workers=workers,
        profile=profile,
        print_training=show,
        evaluate=evaluate,
        max_eps=max_eps,
    )

    self_play_runner(args)


if __name__ == "__main__":
    plac.call(main)
