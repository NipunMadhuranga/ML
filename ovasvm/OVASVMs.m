function [accuracy, test_predictions] = OVASVMs(trainData, testData, N, C)

rng(1); 

[trData, teData] = scaleData(trainData, testData);

A = 1:N; 

options = svmlopt('C', C, 'Kernel', 0);

predict = [];

for class_idx = 1:N
    current_class_label = A(class_idx);


    xTrain = invertData(trData, current_class_label);
    yTrain = xTrain(:, end);
    xTrain(:, end) = [];
    svmlwrite('SVMTrain', xTrain, yTrain);

  
    modelFile = ['Model_', int2str(current_class_label), '_VsAll'];
    svm_learn(options, 'SVMTrain', modelFile);

    xTest = teData;
    yTest = xTest(:, end); 
    xTest(:, end) = []; 
    svmlwrite('SVMTest', xTest, yTest);


    outputFile = ['ModelOutput_', int2str(current_class_label), '_VsAll'];
    svm_classify(options, 'SVMTest', modelFile, outputFile);


    svmPred = svmlread(outputFile); 


    predict = [predict, svmPred];
end


[~, predicted_idx] = max(predict, [], 2); 


true_idx = teData(:, end);
accuracy = mean(predicted_idx == true_idx);
test_predictions = predicted_idx;

end