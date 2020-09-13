% Compute the cost for linear regression.
%
% usage: costFunction (X, y, theta)
%   X = matrix of features
%   y = matrix of target variables
%   theta = chosen theta values
%
% returns: the cost J for linear regression

function J = costFunction (X, y, theta)

    m = length(y);
    err = X * theta - y;
    J = (1 / (2 * m)) * err' * err;

endfunction
