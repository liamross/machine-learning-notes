% Applies normal equation to learn theta.
%
% usage: normalEquation (X, y)
%   X = matrix of features
%   y = matrix of target variables
%
% returns: the final theta values

function theta = normalEquation (X, y)

    X_tran = X';
    theta = pinv(X_tran * X) * X_tran * y;

endfunction
