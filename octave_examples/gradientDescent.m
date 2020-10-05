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
    % and store the resulting cost of the new theta in the J history.
    for iter = 1:num_iters

        % First, calculate the result of the linear hypothesis for each item in
        % the training set. This will create a vector with the output value for
        % each item.
        hypothesis = X * theta;

        % Next, calculate alpha times the partial derivative of the cost
        % function, which will generate a vector of values the same length as
        % theta. Finally, subtract these from the previous theta vector to get
        % the new values of theta.
        theta = theta - (alpha / m) * X' * (hypothesis - y)

        % Finally, calculate the actual cost from the cost function in order to
        % store the values and determine whether the cost is decreasing with
        % each run.
        J_history(iter) = linearCostFunction(X, y, theta);

    end

end
