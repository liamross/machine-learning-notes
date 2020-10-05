% Compute the cost for logistic hypothesis in a binary classification problem.
%
% usage: logisticCostFunction (X, y, theta)
%   X = matrix of features
%   y = matrix of target variables
%   theta = chosen theta values
%
% returns: [J, grad]
%   J = the cost for logistic regression
%   grad = the partial derivative gradients

function [J, grad] = logisticCostFunction (X, y, theta)

    m = length(y);

    % First, calculate the result of the logistic hypothesis for each item in
    % the training set. This will create a vector with the output value for
    % each item.
    hypothesis = sigmoid(X * theta);

    % Calculate the error sum for the case when y=0 or y=1.
    err = (-y)' * log(hypothesis) - (1 - y)' * log(1 - hypothesis);

    % Calculate average error by dividing by m (same as 1/m * err).
    J = err / m;

    % Get vector of gradients by getting partial derivative for each theta.
    grad = (1 / m) * X' * (hypothesis - y)

end
