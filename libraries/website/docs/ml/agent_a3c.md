Mathy provides an on-policy learning agent (a3c) that can be trained on a modern desktop CPU.

# A3C

Asynchronous Advantage Actor-Critic (A3C) is an algorithm that uses multiple workers to train a shared model.
Each worker 

!!! info "Help Wanted - Parallelizing A3C updates"

        One thing that is challenging about A3C is that the workers all need to push their gradients
        to a shared "global" model. Tensorflow doesn't make this easy to do across process boundaries
        so the A3C implementation is strictly multi-threaded and has limited ability to scale.

        As a workaround for the inability to use multiprocessing, a hyperparameter "worker_wait" is
        defined by the agent configuration, and each worker that isn't the main (worker 0) will wait
        that many milliseconds between each action it attempts to take. This allows you to run more
        workers than you have cores. The overall number of examples gained is not greated using this
        trick, but the diversity of the data gathered should be.

        If you would like to help out with making the A3C implementation scaling using multiprocessing
        [open an issue here](https://github.com/justindujardin/mathy/issues/new?title=A3CMultiprocessing){target=\_blank}
