function [Data,Target]=Make_Data(x,N,D,L)
Data=zeros(N-D-L,D+1);
% Target=zeros(N-D-L,1);
for i= 1:N-D-L
Data(i,1:end-1)=x(i:D-1+i,1);
Data(i,end)=x(D-1+i+L+1,1);
end
Target=Data(:,D+1);
Data=Data(:,1:D);
end