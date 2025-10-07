function invData= invertData(Data,class)

invData=Data;
n=size(invData,2);

indices=(invData(:,n)==class);
invData(indices,n)=1;


ind= ~indices;
invData(ind,n)=-1;




