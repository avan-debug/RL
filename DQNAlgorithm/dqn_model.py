import tensorflow as tf
from tensorflow.keras import layers, Sequential

class Model:

    def __init__(self, obs_n, act_dim):
        self.obs_n = obs_n
        self.act_dim = act_dim
        self.build_model()

    def build_model(self):
        hid1_size = 128
        hid2_size = 128

        model = Sequential([
            layers.Dense(hid1_size, activation="relu", name="l1"),
            layers.Dense(hid2_size, activation="relu", name="l2"),
            layers.Dense(self.act_dim, name="l3")
        ])

        model.build(input_shape=(None, self.obs_n))
        # model.summary()
        self.model = model

        target_model = Sequential([
            layers.Dense(hid1_size, activation="relu", name="l1"),
            layers.Dense(hid2_size, activation="relu", name="l2"),
            layers.Dense(self.act_dim, name="l3")
        ])

        target_model.build(input_shape=(None, self.obs_n))
        # target_model.summary()
        self.target_model = target_model


if __name__ == '__main__':
    a = Model(2, 2)

