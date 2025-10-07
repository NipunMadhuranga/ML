function predicted_labels = myKNN(X_train, Y_train, X_test, k)
    nTest = size(X_test,1);                % number of test samples
    predicted_labels = zeros(nTest,1);     % store predicted labels

    for i = 1:nTest
        % Compute Euclidean distance to all training points
        distances = sqrt(sum((X_train - X_test(i,:)).^2, 2));

        % Sort distances and get indices of k nearest neighbors
        [~, idx] = sort(distances, 'ascend');
        nearest_labels = Y_train(idx(1:k));

        % Majority vote
        predicted_labels(i) = mode(nearest_labels);
    end
end
