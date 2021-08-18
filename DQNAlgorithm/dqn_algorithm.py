import numpy as np
import tensorflow as tf
from tensorflow.keras import optimizers
import dqn_model as dm

class DQN:
    def __init__(self, model, gamma, lr_rate):
        self.model = model.model
        self.target_model = model.target_model
        self.gamma = gamma
        self.lr_rate = lr_rate

        # 优化器和损失函数
        self.model.optimizer = optimizers.Adam(learning_rate=self.lr_rate)
        self.model.loss_fn = tf.losses.MSE

        #记录
        self.global_step = 0
        self.update_step = 200

    def predict(self, obs):
        return self.model(obs)

    def _train_step(self, action, features, labels):
        # action dim => [b, 1]
        with tf.GradientTape() as tape:
            # [b, action_dim]
            predictions = self.model(features, training=True)
            enum_act = list(enumerate(action))
            enum_act = tf.convert_to_tensor(np.array(enum_act), dtype=tf.int32)
            # print("predictions: ", predictions)
            # print("enum_act: ", enum_act)
            pred_action_value = tf.gather_nd(predictions, indices=enum_act)
            loss = self.model.loss_fn(pred_action_value, labels)
            loss = tf.reduce_sum(loss)
            # print(loss)

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.model.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

    def _train_model(self, actions, features, labels, epochs = 1):

        for i in range(epochs):
            self._train_step(actions, features, labels)


    # 更新model参数
    # obs=[b, obs_n] actions=[b, act_] terminal=[b, 1]
    def learn(self, obs, actions, reward, next_obs, terminal):

        # 更新目标网络
        if self.global_step % self.update_step == 0:
            self.replace_target()

        target_val = self.target_model(next_obs)
        best_v = tf.reduce_max(target_val, axis=1)
        terminal = tf.cast(terminal, tf.float32)
        labels = reward + 1.0 * self.gamma * (1.0 - terminal) * best_v

        self._train_model(actions, obs, labels, epochs=1)

        self.global_step += 1



    def replace_target(self):
        #self.target_model.set_weights(self.model.get_weights())
        self.target_model.get_layer(name='l1').set_weights(self.model.get_layer(name='l1').get_weights())
        self.target_model.get_layer(name='l2').set_weights(self.model.get_layer(name='l2').get_weights())
        self.target_model.get_layer(name='l3').set_weights(self.model.get_layer(name='l3').get_weights())



if __name__ == '__main__':
    pass
