from keras import backend as K
from keras.engine.topology import Layer, initializers
import tensorflow as tf


class Capsule(Layer):
    def __init__(
            self, num_capsule, dim_capsule, routing_iterations,
            kernel_initializer='random_uniform', **kwargs
    ):
        assert num_capsule > 0 and dim_capsule > 0 and routing_iterations > 0
        super(Capsule, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routing_iterations = routing_iterations
        self.kernel_initializer = initializers.get(kernel_initializer)

    def build(self, input_shape):
        assert len(input_shape) == 3  # [batch_size, input_num_capsule, input_num_dim]
        self.input_num_capsule = input_shape[1]
        self.input_dim_capsule = input_shape[2]
        self.W = self.add_weight(
            shape=[1, self.input_num_capsule, self.num_capsule, self.dim_capsule, self.input_dim_capsule],
            initializer=self.kernel_initializer,
            name='transformation_matrix'
        )
        super(Capsule, self).build(input_shape)

    def call(self, prev_capsule):
        # prev_capsule size: [batch_size, input_num_capsule, input_dim_capsule]
        prev_capsule_expanded = K.expand_dims(prev_capsule, -1)
        prev_capsule_expanded = K.expand_dims(prev_capsule_expanded, 2)
        prev_capsule_tiled = K.tile(prev_capsule_expanded, [1, 1, self.num_capsule, 1, 1])
        # prev_capsule_tiled size: [batch_size, input_num_capsule, num_capsule, input_dim_capsule, 1]
        batch_size = tf.shape(prev_capsule)[0]
        W_tiled = K.tile(self.W, [batch_size, 1, 1, 1, 1])
        # tf matrix production, keras.dot doesn't work here
        caps_predicted = W_tiled @ prev_capsule_tiled
        caps_predicted = K.squeeze(caps_predicted, -1)  # [batch_size, input_num_capsule, num_capsule, dim_capsule]
        b = tf.zeros(shape=(batch_size, self.input_num_capsule, self.num_capsule, 1))
        for i in range(self.routing_iterations):
            c = tf.nn.softmax(b, dim=2)  # keras softmax doesn't work here
            weighted_predictions = c * caps_predicted  # [batch_size, input_num_capsule, num_capsule, dim_capsule]
            weighted_sum = K.sum(weighted_predictions, axis=1,
                                 keepdims=True)  # [batch_size, 1, num_capsule, dim_capsule]
            output = squash(weighted_sum)
            if i == self.routing_iterations - 1:
                break
            output_tiled = K.tile(output, [1, self.input_num_capsule, 1, 1])
            # output size: [batch_size, input_num_capsule, num_capsule, dim_capsule]
            output_flattened = K.reshape(output_tiled, [-1, self.dim_capsule])
            caps_predicted_flattened = K.reshape(caps_predicted, [-1, self.dim_capsule])
            agreement = K.reshape(
                K.sum(output_flattened * caps_predicted_flattened, axis=-1),
                [batch_size, self.input_num_capsule, self.num_capsule, 1])
            b += agreement
        return K.squeeze(output, 1)

    def compute_output_shape(self, input_shape):
        return tuple([None, self.num_capsule, self.dim_capsule])


class Length(Layer):
    def call(self, inputs, **kwargs):
        return K.sqrt(K.sum(K.square(inputs), -1))

    def compute_output_shape(self, input_shape):
        return input_shape[:-1]


def squash(s, axis=-1):
    squared_norm = K.sum(K.square(s), axis=axis, keepdims=True)
    safe_norm = K.sqrt(squared_norm + K.epsilon())  # avoid infinite gradient
    squash_factor = squared_norm / (1. + squared_norm)
    unit_vector = s / safe_norm
    return squash_factor * unit_vector
