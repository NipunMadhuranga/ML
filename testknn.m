%-------------Question 01-------------
data = load('iris.txt');
X = data(:,1:end-1);
Y = data(:,end);

N = length(Y);


%-------------Question 02-------------
rng(1);
c = cvpartition(N,'Holdout',0.3);

train_X = X(training(c),:);
test_X = X(test(c),:);

train_Y = Y(training(c),:);0
test_Y = Y(test(c),:);



%-------------Question 03-------------

k = 5;

Y_pred_30 = myKNN(train_X, train_Y, test_X, k);


accuracy_30 = sum(Y_pred_30 == test_Y) / length(test_Y) * 100;
fprintf('Accuracy for 70-30 split with k=5: %.2f%%\n', accuracy_30);

accuracy = OVASVMs(data,N);
fprintf('\nFinal Accuracy (70%% Train - 30%% Test) = %.2f%%\n', accuracy*100);

