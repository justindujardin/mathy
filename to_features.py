from typing import Optional
import uuid
import plac
import tensorflow as tf
import numpy as np

from mathy.mathy_env import MathyEnv
from mathy.agent.layers.math_embedding import MathEmbedding
from mathy.envs.polynomial_simplification import MathyPolynomialSimplificationEnv
from mathy.envs.binomial_distribution import MathyBinomialDistributionEnv


@plac.annotations(
    model_dir=(
        "The name of the model to train. This changes the output folder.",
        "positional",
        None,
        str,
    ),
    input_text=("The text to parse into features.", "positional", None, str),
    prior_text=(
        "The prior text parse into previous features.",
        "positional",
        None,
        str,
    ),
)
def main(model_dir: str, input_text: str, prior_text: Optional[str] = None):
    import tensorflow as tf

    print(f"Input Text: {input_text}")
    print(f"Prior Text: {prior_text}")

    num_problem_type_buckets = 4
    num_problem_buckets = 1024

    def env_to_hash(env: MathyEnv):
        """Convert a MathyEnv class name into a hash bucket entry. Used for
        encoding the problem type into observations."""
        return tf.strings.to_hash_bucket_fast(
            foil_env.__class__.__name__, num_problem_type_buckets
        )

    junk_hash = tf.strings.to_hash_bucket_fast(
        "JUNK", num_buckets=num_problem_type_buckets
    )
    foil_env = MathyBinomialDistributionEnv()
    binomial_hash = env_to_hash(foil_env)
    poly_env = MathyPolynomialSimplificationEnv()
    poly_hash = env_to_hash(poly_env)
    problem_args = {"difficulty": 3}

    train_examples = 128

    x_train = []
    y_train = []
    for i in range(train_examples // 3):
        # one example for each type
        _, f_prob = foil_env.get_initial_state(problem_args)
        x_train.append(tf.strings.to_hash_bucket_fast(f_prob.text, num_problem_buckets))
        y_train.append(binomial_hash)

        _, p_prob = poly_env.get_initial_state(problem_args)
        x_train.append(tf.strings.to_hash_bucket_fast(p_prob.text, num_problem_buckets))
        y_train.append(poly_hash)

        x_train.append(
            tf.strings.to_hash_bucket_fast(uuid.uuid4().hex, num_problem_buckets)
        )
        y_train.append(junk_hash)

    x_train = np.array(x_train)
    y_train = np.array(y_train)
    model = tf.keras.models.Sequential(
        [
            # MathEmbedding(name="embed"),
            tf.keras.layers.Flatten(input_shape=(1,)),
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(num_problem_type_buckets, activation="softmax"),
        ]
    )

    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )
    model.fit(x_train, y_train, epochs=500)

    # model.evaluate(x_test, y_test)
    print("Complete. Bye!")


if __name__ == "__main__":
    plac.call(main)
