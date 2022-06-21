# Deep Learning Libraries
import numpy as np
np.set_printoptions(suppress=True)
from keras.models import load_model

import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import keras.backend as K

from what.utils.proj_lp import proj_lp

class TOGAttack:
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
        
        self.c_h = [tf.compat.v1.placeholder(dtype=tf.float32,
                    shape=self.model.output[layer].shape) for layer in range(3)]
        
        for layer in range(3):
            loss += K.sum(K.binary_crossentropy(self.c_h[layer][..., 4:5], self.model.output[layer][..., 4:5], from_logits=True))

        grads = K.gradients(loss, self.model.input)
        self.delta = self.delta + K.sign(grads[0])

        self.sess = tf.compat.v1.keras.backend.get_session()

    def attack(self, input_cv_image):
        with self.graph.as_default():
            c_h = []
            c_h.append(np.zeros((1, 13, 13, 3, 85)))
            c_h.append(np.zeros((1, 26, 26, 3, 85)))
            c_h.append(np.zeros((1, 52, 52, 3, 85)))
            c_h[0][..., 4] = 1.0
            c_h[1][..., 4] = 1.0
            c_h[2][..., 4] = 1.0
            c_h[0] = c_h[0].reshape((1, 13, 13, 255))
            c_h[1] = c_h[1].reshape((1, 26, 26, 255))
            c_h[2] = c_h[2].reshape((1, 52, 52, 255))
            # Draw each adversarial patch on the input image
            input_cv_image = input_cv_image + self.noise
            input_cv_image = np.clip(input_cv_image, 0.0, 1.0).astype(np.float32)

            if not self.fixed:
                outputs, grads = self.sess.run([self.model.output, self.delta], 
                                                feed_dict={self.model.input:np.array([input_cv_image]),
                                                self.c_h[0]: c_h[0],
                                                self.c_h[1]: c_h[1],
                                                self.c_h[2]: c_h[2],
                                                })

                self.noise = self.noise - 2 / 255.0 * grads[0, :, :, :]

                self.noise = np.clip(self.noise, -1.0, 1.0)

                self.noise = proj_lp(self.noise, xi=8/255.0, p = np.inf)

            return input_cv_image, outputs
