function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

C_vals = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
sigma_vals = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];

best_C_index = 1;
best_sigma_index = 1;
best_error = Inf;


# Try out combinations of C and sigma to see what performs best on the
# cross-validation data set.
for i = 1:length(C_vals)
  for j = 1:length(sigma_vals)
    # Train model w/ choice of C and sigma
    C = C_vals(i);
    sigma = sigma_vals(j);
    model = svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
    
    # See how we do
    predictions = svmPredict(model, Xval);
    error = mean(double(predictions ~= yval));
    if error < best_error
      best_C_index = i;
      best_sigma_index = j;
      best_error = error;
    endif
    
  endfor
endfor

C = C_vals(best_C_index);
sigma = sigma_vals(best_sigma_index);
fprintf("Best values are C = %f, sigma = %f\n", C, sigma);

% =========================================================================

end
