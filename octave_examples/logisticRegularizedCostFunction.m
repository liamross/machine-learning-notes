% Compute the cost for logistic hypothesis in a binary classification problem.
%
% usage: logisticRegularizedCostFunction (X, y, theta, lambda)
%   X = matrix of features
%   y = matrix of target variables
%   theta = chosen theta values
%   lambda = the regularization parameter
%
% returns: [J, grad]
%   J = the cost for logistic regression
%   grad = the partial derivative gradients

function [J, grad] = logisticRegularizedCostFunction (X, y, theta, lambda)

    m = length(y);

    % --- Cost function ---

    % First, calculate the result of the logistic hypothesis for each item in
    % the training set. This will create a vector with the output value for
    % each item.
    hypothesis = sigmoid(X * theta);

    % Calculate the error sum for the case when y=0 or y=1.
    err = (-y)' * log(hypothesis) - (1 - y)' * log(1 - hypothesis);

    % Calculate average error by dividing by m (same as 1/m * err).
    J = 1 / m;

    % Get vector of gradients by getting partial derivative for each theta.
    grad = (1 / m) * X' * (hypothesis - y)

    % --- Regularization ---

    % Next, add the result of theta(2:end) (we don't calculate for theta_1) to
    % the base J value to calculate the cost.
    theta_2_end = theta(2:end);
    J = JBase + (lambda / (2 * m)) * theta_2_end' * theta_2_end;

    % Finally, add the vector of regularized values to the rest of grad.
    grad(2:end) = grad(2:end) + ((lambda / m) * theta_2_end);

end
