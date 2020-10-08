% Computes the sigmoid function.
%
% usage: sigmoid (z)
%   z = scalar or matrix input
%
% returns: the values with signmoid applied to each

function g = sigmoid (z)

    g = 1.0 ./ (1.0 + exp(-z));

end
