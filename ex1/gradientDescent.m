function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %







    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);
    
    scalar = alpha*(1/m);

    e = 0;
    for i = 1:m
        h = theta(1) + (theta(2) * X(i,2));
        a = h - y(i);
        b = a*X(i,1);
        e = e + b;
    end;

    f = scalar * e;
    g = theta(1) - f;
    

    e = 0;
    for i = 1:m
        h = theta(1) + (theta(2) * X(i,2));
        a = h - y(i);
        b = a*X(i,2);
        e = e + b;
    end;

    f = scalar * e;
    h = theta(2) - f;

    theta(1) = g;
    theta(2) = h;

end

end
