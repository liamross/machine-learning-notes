% Compute the cost for linear regression.
%
% usage: costFunction (X, y, theta)
%   X = matrix of features
%   y = matrix of target variables
%   theta = initial theta values
%
% returns: the cost J for linear regression

function J = costFunction (X, y, theta)

    m = length(y);
    J = (1 / (2 * m)) * (X * theta - y)' * (X * theta - y);

endfunction
