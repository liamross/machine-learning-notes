% Normalizes features by subtracting mean and dividing by standard deviation.
%
% usage: meanStandardNormalize (X)
% usage to check new values: meanStandardNormalize (X, mu, sigma)
%   X = matrix of features
%   mu = Optional. The mean of the features
%   sigma = Optional. The std deviation of the features
% 
%
% returns: [X_norm, mu, sigma]
%   X_norm = X with each value normalized
%   mu = a matrix of the mean value for each column
%   sigma = a matrix of the standard deviation for each column

function [X_norm, mu, sigma] = meanStandardNormalize (X, mu = mean(X), sigma = std(X))

    X_norm = (X - mu) ./ sigma;

end
