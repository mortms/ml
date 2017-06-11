function plotData(X, y)
%PLOTDATA Plots the data points X and y into a new figure 
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Mx2 matrix.

% Create New Figure
figure; hold on;

% ====================== YOUR CODE HERE ======================
% Instructions: Plot the positive and negative examples on a
%               2D plot, using the option 'k+' for the positive
%               examples and 'ko' for the negative examples.
%

pos = find(y==1);
neg = find(y==0);

# Extraact positive samples into separate nx2 array
pos_points = zeros(size(pos)(1),2);
i = 1;
for l = pos'
  assert(y(l) == 1);
  pos_points(i,:) = [X(l,1), X(l,2)];
  i = i + 1;
endfor
assert(i == size(pos)(1) + 1)
plot(pos_points(:,1), pos_points(:,2), "k+")

# Extract negative samples into separate nx2 array
i = 1;
for l = neg'
  assert(y(l(1)) == 0)
  neg_points(i,:) = [X(l,1), X(l,2)];
  i = i + 1;
endfor
assert(i == size(neg)(1) + 1)
plot(neg_points(:,1), neg_points(:,2), "ko")




% =========================================================================



hold off;

end
