class NeuralNet:
    """
    This class specifies the base NeuralNet class. To define your own neural
    network, subclass this class and implement the functions below.
    """

    def __init__(self, game):
        pass

    def train(self, examples):
        """
        This function trains the neural network with examples obtained from
        self-play.

        Input:
            examples: a list of training examples, where each example is of form
                      (env_state, pi, v). pi is the MCTS informed policy vector for
                      the given env_state, and v is its value. The examples has
                      env_state in its canonical form.
        """
        pass

    def predict(self, env_state):
        """
        Input:
            env_state: current env_state in its canonical form.

        Returns:
            pi: a policy vector for the current env_state- a numpy array of length
                game.get_agent_actions_count
            v: a float in [-1,1] that gives the value of the current env_state
        """
        pass
