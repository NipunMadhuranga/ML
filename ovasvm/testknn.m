
%--------------------------Question 01--------------------------
data = load('./Dataset/wine.txt');
X = data(:,2:end);
Y = data(:,1);

N = length(Y);


% -- 70% - 30% split --
rng(1) 
C = cvpartition(N,'HoldOut',0.3);% test(c) = 0.3 , training (c) = 0.7

X_train_70 = X(training(C),:);
Y_train_70 = Y(training(C),:);

X_test_30 = X(test(C),:);
Y_test_30 = Y(test(C),:);


% -- 90% - 10% split --
C = cvpartition(N,'HoldOut',0.1); % test(C) = 0.1 %training(C) = 0.9
X_train_90 = X(training(C),:);
Y_train_90 = Y(training(C),:);

X_test_10 = X(test(C),:);
Y_test_10 = Y(test(C),:);



%--------------------------Question 02--------------------------
X_max_70 = max(X_train_70);
X_min_70 = min(X_train_70);

X_max_90 = max(X_train_90);
X_min_90 = min(X_train_90);


X_train_70_scaled = 2*((X_train_70-X_min_70)./(X_max_70-X_min_70))-1;
X_test_30_scaled =  2*((X_test_30-X_min_70)./(X_max_70-X_min_70))-1;



X_train_90_scaled = 2*((X_train_90-X_min_90)./(X_max_90-X_min_90))-1;
X_test_10_scaled =  2*((X_test_10-X_min_90)./(X_max_90-X_min_90))-1;



%--------------------------Question 03--------------------------
k = 3;
Y_pred_30 = myKNN(X_train_70_scaled, Y_train_70, X_test_30_scaled, k);

% Measure accuracy
accuracy_30 = sum(Y_pred_30 == Y_test_30) / length(Y_test_30) * 100;
fprintf('Accuracy for 70-30 split with k=3: %.2f%%\n', accuracy_30);

% Example using 90%-10% split (scaled data)
Y_pred_10 = myKNN(X_train_90_scaled, Y_train_90, X_test_10_scaled, k);
accuracy_10 = sum(Y_pred_10 == Y_test_10) / length(Y_test_10) * 100;
fprintf('Accuracy for 90-10 split with k=3: %.2f%%\n', accuracy_10);

