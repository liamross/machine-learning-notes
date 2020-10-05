% Compute the cost for linear hypothesis.
%
% usage: linearCostFunction (X, y, theta)
%   X = matrix of features
%   y = matrix of target variables
%   theta = chosen theta values
%
% returns: the cost J for linear regression

function J = linearCostFunction (X, y, theta)

    m = length(y);

    % First, calculate the result of the linear hypothesis for each item in
    % the training set. This will create a vector with the output value for
    % each item.
    hypotheses = X * theta;

    % Calculate the error for each item by subtracting the actual value from
    % the hypothesis value.
    err = hypotheses - y;

    % Square and sum the values, then times by 1/2m to get the sum cost across
    % all items in the training set. This could also be done as sum(err .^ 2).
    J = (1 / (2 * m)) * err' * err;

end
