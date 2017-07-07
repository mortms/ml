function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
% Theta1 dimensions should be hidden_layer_size x (input_layer_size + 1)
% Theta2 dimensinos should be num_labels x (hidden_layer_size + 1)
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);

% y as passed is single-value per sample.  Convert to class labels
% with each sample in a column
training_labels = zeros(num_labels, m);
for i = 1:m
  training_labels(y(i), i) = 1;
endfor
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network   and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m

% We're going to forward-propagate all examples at once.
% Input X comes with examples in each row, but that confilcts with 
% normal representation of activation layers, so switch to columns
printf("Dimensions of X: %d x %d\n", size(X));
a1 = X';

% Add bias inputs in as we multiply
z2 = Theta1 * [ones(1, size(a1, 2)); a1];
a2 = sigmoid(z2);

z3 = Theta2 * [ones(1, size(a2, 2)); a2];
a3 = sigmoid(z3);

% Sum up cost across each output label
for k = 1:num_labels
  % Grab k-th row of output and labels - both row vectors 
  h_k = a3(k,:);
  y_k = training_labels(k,:);
  
  % Basic cost function
  part1 = -y_k * log(h_k)';
  part2 = (1 .- y_k) * log(1 .- h_k)';
  J = J + (part1 - part2) / m;
endfor
  
% Add regularization.  Note that we extract the first column of each
% theta to remove params applied to bias
reg = lambda / (2 * m) * (sumsq(Theta1(:, 2:end)(:)) + sumsq(Theta2(:, 2:end)(:)));
J = J + reg;

%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

grad = [Theta1_grad(:); Theta2_grad(:)];

















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
