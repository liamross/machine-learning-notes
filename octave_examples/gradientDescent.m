% Applies gradient descent to learn theta.
%
% usage: gradientDescent (X, y, theta, alpha, num_iters)
%   X = matrix of features
%   y = matrix of target variables
%   theta = initial theta values
%   alpha = the learning rate
%   num_iters = number of iterations
%
% returns: [theta, J_history]
%   theta = the final theta values
%   J_history = a vector of the cost at each iteration

function [theta, J_history] = gradientDescent (X, y, theta, alpha, num_iters)

    m = length(y);                   % number of training examples
    J_history = zeros(num_iters, 1); % initialize J_history

    % For each iteration, set theta using vectorized gradient descent algorithm,
    % and store the resulting cost in the J history.
    for iter = 1:num_iters

        theta = theta - alpha * (1 / m) * ((X * theta - y)' * X)';
        J_history(iter) = computeCost(X, y, theta);

    endfor

endfunction
