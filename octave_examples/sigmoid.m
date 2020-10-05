% Computes the sigmoid function.
%
% usage: sigmoid (z)
%   z = scalar or matrix input
%
% returns: the values with signmoid applied to each

function g = sigmoid (z)

    g = 1 ./ (1 + e .^ -z);

end
