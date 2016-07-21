function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
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
    %       of the cost function (computeCostMulti) and gradient here.
    %



	
	tempTheta = theta;

	%sizeFeatures = size(X,2);
	
	%for n = 1:sizeFeatures
	
		%e = 0;
		%h = 0;
		%for i = 1:m
		%	h = 0;
		%	for t = 1:sizeFeatures
		%		h = h + (theta(t) * X(i,t));
		%	end;			
		%	a = h - y(i);
		%	b = a*X(i,n);
		%	e = a*b;
		%end;

		%f = scalar * e;
		%g = tempTheta(n) - f;
		scalar = alpha/m;
		a = X*theta;
		b = a - y;
		c = X' * b;
		d = scalar*c;
		e = theta-d;
		
		
		
		
	%	tempTheta(n) = g;
	%end;
	
	theta = e;







    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);


end

end
