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
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
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

%处理真实值矩阵y
Y = [];
E = eye(num_labels);

for i= 1:num_labels
  Y0 = find(y == i);
  Y(Y0, :) = repmat(E(i, :), size(Y0, 1), 1);
end

%利用前向传播算法计算各层的激活值以及输出值
X = [ones(m, 1) X];

a2 = sigmoid(X * Theta1');

a2 = [ones(m, 1) a2];

a3 = sigmoid(a2 * Theta2');

%去除theta（0）,不参与正则化
temp1 = [zeros(size(Theta1, 1), 1) Theta1(:, 2:end)];

temp2 = [zeros(size(Theta2, 1), 1) Theta2(:, 2:end)];
%计算正则项
temp1 = sum(temp1 .^2);

temp2 = sum(temp2 .^2);

%计算cost
cost = Y .* log(a3) + (1 - Y) .* log(1 - a3);

%cost 是m*k的结果矩阵

J = - 1 / m * sum(cost(:)) + lambda / (2 * m) * (sum(temp1(:)) + sum(temp2(:)));

 %利用反向传播算法计算gradient
 %初始化grad矩阵
 delta_1 = zeros(size(Theta1));
 delta_2 = zeros(size(Theta2));

for t = 1:m
    %step 1 计算出每层的激活值
    a_1 = X(t, :)';
    z_2 = Theta1 * a_1;
    a_2 = sigmoid(z_2);
    a_2 = [1; a_2];
    z_3 = Theta2 * a_2;
    a_3 = sigmoid(z_3);

    %step 2 计算输出层的残差
    err_3 = zeros(num_labels, 1);

    for k = 1:num_labels
      err_3(k) = a_3(k) - (y(t) == k);
    end

    %step 3 计算剩余的隐藏层的残差
    err_2 = Theta2' * err_3;
    %去掉第一个残差，减少为25行。sigmoidGradient（z_2）or g'(z_2)只有25行
    err_2 = err_2(2:end) .* sigmoidGradient(z_2);

    delta_2 = delta_2 + err_3 * a_2';
    delta_1 = delta_1 + err_2 * a_1';
end

% step 5
% theta(0) 不参与正则化
Theta1_temp = [zeros(size(Theta1, 1), 1) Theta1(:, 2:end)];
Theta2_temp = [zeros(size(Theta2, 1), 1) Theta2(:, 2:end)];

Theta1_grad = 1 / m * (delta_1 + lambda * Theta1_temp);
Theta2_grad = 1 / m * (delta_2 + lambda * Theta2_temp);

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
