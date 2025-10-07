function rate = KNN(trData,teData)

ntr = size(trData,1); 
nte = size(teData,1); 
predict = [];

for j = 1:nte 
    d = [];
    y = teData(j,1:end-1);
    for i = 1:ntr 
        d(i) = norm(y-trData(i,1:end-1));
    end
    [elt,ind] = min(d);
    predict(j) = trData(ind,end);
end

actual = teData(:,end); 
rate = 100*sum(actual == predict')/nte;