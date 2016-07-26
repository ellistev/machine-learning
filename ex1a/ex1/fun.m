
clear ; close all; clc

fprintf('Loading data ...\n');

%% Load Data
data = load('Housing.txt');
X = data(:, 1:4);
y = data(:, 5);
m = length(y);

% Scale features and set them to zero mean
fprintf('Normalizing Features ...\n');

[X mu sigma] = featureNormalize(X);

% Add intercept term to X
X = [ones(m, 1) X];


fprintf('Running gradient descent ...\n');

% Choose some alpha value
alpha = 0.03;
num_iters = 50000;

% Init Theta and Run Gradient Descent 
theta = zeros(5, 1);
[theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters);

% Plot the convergence graph
figure;
plot(1:numel(J_history), J_history, '-b', 'LineWidth', 2);
xlabel('Number of iterations');
ylabel('Cost J');

% Display gradient descent's result
fprintf('Theta computed from gradient descent: \n');
fprintf(' %f \n', theta);
fprintf('\n');

% Estimate the price of a 1650 sq-ft, 3 br house
% ====================== YOUR CODE HERE ======================
% Recall that the first column of X is all-ones. Thus, it does
% not need to be normalized.
price = 0; % You should change this

X = [1650, 3 , 3, 2];


[X] = featureNormalizeOldSigma(X, mu, sigma);
X = [ones(1, 1) X];

price = X * theta;


% ============================================================

fprintf(['Predicted price of a 1650 sq-ft, 3 br house, 1 bathroom, 1 story home ' ...
         '(using gradient descent):\n $%f\n'], price);

fprintf('Program paused. Press enter to continue.\n');
pause;

%% ================ Part 3: Normal Equations ================

fprintf('Solving with normal equations...\n');


%% Load Data
data = csvread('Housing.txt');
X = data(:, 1:4);
y = data(:, 5);
m = length(y);

% Add intercept term to X
X = [ones(m, 1) X];

% Calculate the parameters from the normal equation
theta = normalEqn(X, y);

% Display normal equation's result
fprintf('Theta computed from the normal equations: \n');
fprintf(' %f \n', theta);
fprintf('\n');


% Estimate the price of a 1650 sq-ft, 3 br house 3 bath 2 story
% ====================== YOUR CODE HERE ======================
price = 0; % You should change this
X = [1650, 3 , 3, 2];
X = [ones(1, 1) X];
price = X * theta;


% ============================================================

fprintf(['Predicted price of a 1650 sq-ft, 3 br house, 1 bathroom, 1 story home ' ...
         '(using gradient descent):\n $%f\n'], price);

