# Deep Learning Libraries
import numpy as np
np.set_printoptions(suppress=True)
from keras.models import load_model

import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import keras.backend as K

from what.utils.proj_lp import proj_lp

class PCBAttack:
    def __init__(self, model, attack_type, monochrome, classes):
        self.classes = len(classes)
        self.epsilon = 1
        self.graph = tf.compat.v1.get_default_graph()
        self.monochrome = monochrome
        self.use_filter = False

        if self.monochrome:
            self.noise = np.zeros((416, 416))
        else:
            self.noise = np.zeros((416, 416, 3))

        self.adv_patch_boxes = []
        self.fixed = True

        self.model = load_model(model)
        self.model.summary()
        self.attack_type = attack_type

        self.delta = 0
        loss = 0

        for out in self.model.output:
            # Targeted One Box
            if attack_type == "one_targeted":
                loss = K.max(K.sigmoid(K.reshape(out, (-1, 5 + self.classes))[:, 4]) * K.sigmoid(K.reshape(out, (-1, 5 + self.classes))[:, 5]))

            # Targeted Multi boxes
            if attack_type == "multi_targeted":
                loss = K.sigmoid(K.reshape(out, (-1, 5 + self.classes))[:, 4]) * K.sigmoid(K.reshape(out, (-1, 5 + self.classes))[:, 5])

            # Untargeted Multi boxes
            if attack_type == "multi_untargeted":
                for i in range(0, self.classes):
                    loss = loss + tf.reduce_sum(K.sigmoid(K.reshape(out, (-1, 5 + self.classes))[:, 4]) * K.sigmoid(K.reshape(out, (-1, 5 + self.classes))[:, i+5]))

        grads = K.gradients(loss, self.model.input)
        self.delta = self.delta + K.sign(grads[0])

        self.sess = tf.compat.v1.keras.backend.get_session()

    def attack(self, input_cv_image):
        with self.graph.as_default():
            # Draw each adversarial patch on the input image
            input_cv_image = input_cv_image + self.noise
            input_cv_image = np.clip(input_cv_image, 0.0, 1.0).astype(np.float32)

            if not self.fixed:
                outputs, grads = self.sess.run([self.model.output, self.delta], feed_dict={self.model.input:np.array([input_cv_image])})

                self.noise = self.noise + 2 / 255.0 * grads[0, :, :, :]

                self.noise = np.clip(self.noise, -1.0, 1.0)

                self.noise = proj_lp(self.noise, xi=8/255.0, p = np.inf)
            else:
                outputs = self.sess.run(self.model.output, feed_dict={self.model.input:np.array([input_cv_image])})

            return input_cv_image, outputs