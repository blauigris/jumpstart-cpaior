from keras import backend as K, regularizers
from keras.engine.topology import Layer
from keras.utils.generic_utils import to_list




class JumpstartReLU(Layer):
    """
    Implementation of Separating Constraints for ReLU
    From the paper: Superseding Model Scaling by Penalizing Dead Units and Points with Separation Constraints

    """

    def __init__(self, separability_regularizer=None, relu_clipping_value=None, balance=None, axis=None,
                  **kwargs):
        """

        :param separability_regularizer: keras.regularizers instance
        :param relu_clipping_value: clipping value for ReLU, disabled by default
        :param balance: tradeoff between positive and negative slacks. \rho in the paper. [0,1]
        :param axis: axis used to compute the maxes and mins, equal to setting the type of the constraints:
                         [0,-1] -> Unit Point based
                         -1 -> Point based
                         0 -> Unit based

        :param kwargs:
        """
        super().__init__(**kwargs)
        self.axis = axis
        self.balance = balance if balance is not None else 0.5
        self.relu_clipping_value = relu_clipping_value
        self.separability_regularizer = separability_regularizer if separability_regularizer else regularizers.l2(1)

        def separating_relu(x):
            return K.identity(x, name='separating_relu')

        self.activation = separating_relu

    def _get_constraint_for_axis(self, x, axis):
        constraint = self.separability_regularizer(
            self.balance * K.relu(1 - K.max(x, axis=axis)) +
            (1 - self.balance) * K.relu(1 + K.min(x, axis=axis)))

        return constraint

    def call(self, x):
        separabiliy_regularizers = []

        if self.separability_regularizer:
            try:
                for ax in self.axis:
                    separabiliy_regularizers.append(self._get_constraint_for_axis(x, ax))

            except TypeError:  # self.axis is not iterable, thus is either a single number or None
                separabiliy_regularizers.append(self._get_constraint_for_axis(x, self.axis))

        if len(separabiliy_regularizers) > 0:
            self.add_loss(separabiliy_regularizers, to_list(x))

        return K.relu(x, max_value=self.relu_clipping_value)

