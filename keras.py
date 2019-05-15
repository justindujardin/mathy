from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop
import numpy

batch_size = 10

headline_data = numpy.random.uniform(size=[batch_size, 10])
print(headline_data)
headline_labels = numpy.random.uniform(size=[batch_size, 10])
print(headline_labels)

aux_data = numpy.ones([batch_size, 5])
aux_labels = numpy.zeros([batch_size, 5])

main_input = Input(shape=(batch_size,), dtype="int32", name="main_input")
auxiliary_input = Input(shape=(5,), name="aux_input")

x = Embedding(output_dim=6, input_dim=50, input_length=1)(main_input)
# A LSTM will transform the vector sequence into a single vector,
# containing information about the entire sequence
lstm_out = LSTM(32)(x)

auxiliary_output = Dense(1, activation="tanh", name="aux_output")(lstm_out)

x = concatenate([lstm_out, auxiliary_input])
# We stack a deep densely-connected network on top
x = Dense(64, activation="relu")(x)

# And finally we add the main logistic regression layer
main_output = Dense(1, activation="sigmoid", name="main_output")(x)

model = Model(
    inputs=[main_input, auxiliary_input], outputs=[main_output, auxiliary_output]
)
model.compile(
    optimizer="adam",
    loss={"main_output": "mean_squared_error", "aux_output": "mean_squared_error"},
)
model.summary()

# And trained it via:
model.fit(
    {"main_input": headline_data, "aux_input": aux_data},
    {"main_output": headline_labels, "aux_output": aux_labels},
    epochs=50,
    batch_size=batch_size,
)
# from mathy.agent.model import MathSharedModel


# shared = MathSharedModel(6)
# model = shared.policy_model
# model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
# model.summary()

