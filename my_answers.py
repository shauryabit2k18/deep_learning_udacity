import numpy as np
import math

class NeuralNetwork(object):
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # Set number of nodes in input, hidden and output layers.
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # Initialize weights
        self.weights_input_to_hidden = np.random.normal(0.0, self.input_nodes ** -0.5,
                                                        (self.input_nodes, self.hidden_nodes))

        self.weights_hidden_to_output = np.random.normal(0.0, self.hidden_nodes ** -0.5,
                                                         (self.hidden_nodes, self.output_nodes))
        self.lr = learning_rate

        #### TODO: Set self.activation_function to your implemented sigmoid function ####
        #
        # Note: in Python, you can define a function with a lambda expression,
        # as shown below.
        # self.activation_function =
        # lambda x : 1/(1+np.exp(-x))  # Replace 0 with your sigmoid calculation.

        ### If the lambda code above is not something you're familiar with,
        # You can uncomment out the following three lines and put your
        # implementation there instead.
        #
        def sigmoid(x):
            sig_x = (1 / (1 + np.exp(-x)))
            # print("In Sigmoid: ", sig_x)
            return sig_x  # Replace 0 with your sigmoid calculation here

        self.activation_function = sigmoid

    def train(self, features, targets):
        ''' Train the network on batch of features and targets.
            Arguments
            ---------
            features: 2D array, each row is one data record, each column is a feature
            targets: 1D array of target values
        '''
        n_records = features.shape[0]
        delta_weights_i_h = np.zeros(self.weights_input_to_hidden.shape)
        delta_weights_h_o = np.zeros(self.weights_hidden_to_output.shape)
        # print("features", features)
        for X, y in zip(features, targets):
            # print("X", X)
            # print("target - y", y)
            final_outputs, hidden_outputs = self.forward_pass_train(X)
            # Implement the backproagation function below
            delta_weights_i_h, delta_weights_h_o = self.backpropagation(final_outputs, hidden_outputs, X, y, delta_weights_i_h, delta_weights_h_o)


            self.update_weights(delta_weights_i_h, delta_weights_h_o, n_records)

    def forward_pass_train(self, X):
        ''' Implement forward pass here
            Arguments
            ---------
            X: features batch
        '''
        #### Implement the forward pass here ####
        ### Forward pass ###
        # TODO: Hidden layer - Replace these values with your calculations.
        # Harpreet: Weights given to the initial layer. They start randomized
        # Harpreet: This is sigma (wnxn+b)
        # print("Weights.input_to_hidden\n", self.weights_input_to_hidden)
        # print("Weights.input_to_hidden.shape", self.weights_input_to_hidden.shape)
        #
        # print("X", X)
        # print("X\n", X.shape)
        hidden_inputs = np.dot(X, self.weights_input_to_hidden)
        # signals into hidden layer
        # print("hidden_inputs", hidden_inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        # print("hidden_outputs", hidden_outputs)
        # print(hidden_outputs.shape)

        # TODO: Output layer - Replace these values with your calculations.
        # Harpreet. final_outputs.shape should 1x56
        final_inputs = np.dot(hidden_outputs, self.weights_hidden_to_output)  # signals into final output layer
        # Harpreet: 2nd activation is a straight up pass through
        final_outputs = final_inputs * 1 # signals from final output layer
        # print("final_outputs", final_outputs)
        # print("hidden_outputs", hidden_outputs)

        return final_outputs, hidden_outputs

    def backpropagation(self, final_outputs, hidden_outputs, X, y, delta_weights_i_h, delta_weights_h_o):
        ''' Implement backpropagation
            Arguments
            ---------
            final_outputs: output from forward pass
            y: target (i.e. label) batch
            delta_weights_i_h: change in weights from input to hidden layers
            delta_weights_h_o: change in weights from hidden to output layers
        '''
        #### Implement the backward pass here ####
        ### Backward pass ###

        # TODO: Output error - Replace this value with your calculations.
        # Harpreet: this should be SSE
        # None  # Output layer error is the difference between desired target and actual output.

        error = y - final_outputs
        # print("Final Outputs ", error)

        # TODO: Backpropagated error terms - Replace these values with your calculations.
        # Since we have sigmoid(x) = x eat output; derivative is 1
        output_error_term = error

        # TODO: Calculate the hidden layer's contribution to the error
        hidden_error = np.dot(self.weights_hidden_to_output, output_error_term)

        hidden_error_term = hidden_error * hidden_outputs * (1.0 - hidden_outputs)

        # Weight step (input to hidden)
        delta_weights_i_h += hidden_error_term * X[:, None]
        # Weight step (hidden to output)
        delta_weights_h_o += output_error_term * hidden_outputs[:, None]
        return delta_weights_i_h, delta_weights_h_o


    def update_weights(self, delta_weights_i_h, delta_weights_h_o, n_records):
        ''' Update weights on gradient descent step
            Arguments
            ---------
            delta_weights_i_h: change in weights from input to hidden layers
            delta_weights_h_o: change in weights from hidden to output layers
            n_records: number of records
        '''
        self.weights_hidden_to_output +=  self.lr * delta_weights_h_o / n_records  # update hidden-to-output weights with gradient descent step
        self.weights_input_to_hidden += self.lr * delta_weights_i_h / n_records

    def run(self, features):
        ''' Run a forward pass through the network with input features
            Arguments
            ---------
            features: 1D array of feature values
        '''

        #### Implement the forward pass here ####
        # TODO: Hidden layer - replace these values with the appropriate calculations.
        hidden_inputs = np.dot(features, self.weights_input_to_hidden)  # signals into hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)

        # TODO: Output layer - Replace these values with the appropriate calculations.
        final_inputs = np.dot(hidden_outputs, self.weights_hidden_to_output)  # signals into final output layer
        final_outputs = final_inputs  # signals from final output layer

        return final_outputs


#########################################################
# Set your hyperparameters here
##########################################################
iterations = 700
learning_rate = 0.1
hidden_nodes = 5
output_nodes = 1