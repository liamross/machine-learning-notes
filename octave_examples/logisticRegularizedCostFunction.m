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

    % First, get the base values of the normal logistic cost function.
    [JBase, gradBase] = costFunction(theta, X, y);

    % Next, add the result of theta(2:end) (we don't calculate for theta_1) to
    % the base J value to calculate the cost.
    theta_2_end = theta(2:end);
    J = JBase + (lambda / (2 * m)) * theta_2_end' * theta_2_end;

    % Assign the first value of base to grad since it's skipped by
    % the regularization.
    grad(1) = gradBase(1);

    % Finally, add the vector of regularized values to the rest of grad.
    grad(2:end) = gradBase(2:end) + ((lambda * theta_2_end) / m);

end
