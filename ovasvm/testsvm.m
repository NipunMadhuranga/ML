
%% ---------------- Read Dataset ----------------
data = load('./Dataset/wine.txt');

data(any(isnan(data), 2), :) = []; 

%% ---------------- Remap Class Labels to 1..N ----------------
% The Wine dataset has labels 1, 2, 3, which are already in 1..N format.
% This step is redundant for standard wine dataset but kept for safety.
originalLabels = unique(data(:, end));
for i = 1:numel(originalLabels)
    data(data(:, end) == originalLabels(i), end) = i;
end
N = numel(originalLabels); % Number of classes
    
%% ---------------- Split Data ----------------
X = data(:,2:end);
Y = data(:,1);
numSamples = length(Y);

% (i) 70%-30% split
rng(1); % For reproducibility
cv70 = cvpartition(numSamples,'HoldOut',0.3);
X_train_70 = X(training(cv70),:);
Y_train_70 = Y(training(cv70),:);
X_test_30  = X(test(cv70),:);
Y_test_30  = Y(test(cv70),:);

TrainData70 = [X_train_70 Y_train_70];
TestData30 = [X_test_30 Y_test_30];

% (i) 90%-10% split
cv90 = cvpartition(numSamples,'HoldOut',0.1);
X_train_90 = X(training(cv90),:);
Y_train_90 = Y(training(cv90),:);
X_test_10  = X(test(cv90),:);
Y_test_10  = Y(test(cv90),:);

TrainData90 = [X_train_90 Y_train_90];
TestData10 = [X_test_10 Y_test_10];


%% ---------------- Run OVA SVM Classification ----------------
% (iv) Call OVASVMs with the specific train/test sets and C=22
C_value = 4; % Required Misclassification cost

% 70%-30% split
[accuracy_70, ~] = OVASVMs(TrainData70, TestData30, N, C_value);

% 90%-10% split
[accuracy_90, ~] = OVASVMs(TrainData90, TestData10, N, C_value);

%% ---------------- Display Accuracy ----------------
fprintf('\nFinal Accuracy (70%% Train - 30%% Test) = %.2f%%\n', accuracy_70*100);
fprintf('Final Accuracy (90%% Train - 10%% Test) = %.2f%%\n', accuracy_90*100);