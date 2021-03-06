function [stats,preds,proj,cent]=test_ZVD_V1(w,test,classMeans)
%w: matrix with columns are disrciminant vectors
%test:test data
%ClassMeans: means of each class in the training data: R'
%mus: means/std of the training set
%stats: list containing misclassified obs, l0 and l1 norms of discriminant
%preds: predicted labels for test data according to nearest centroid and
%the discriminants
%Extract class labels and obs
test_labels=test(:,1);
test_obs=test(:,2:end);
%Get number of test obs
N=size(test,1);
K=length(unique(test_labels));
%test_obs=normalize_test(test_obs,mu,sig);
%Project the test data to the lower dimensional space by ZVDs
proj=w'*test_obs';
%Compute the centroids and projected distances
cent=w'*classMeans;
%Compute the distances to the centroid for test data, i.e the distance of
%every projected point to every cent:
dist=zeros(N,K);
for i=1:N
    for j=1:K
        dist(i,j)=norm(proj(:,i)-cent(:,j));
    end
end
%Label test_obs accoring to the closest centroid to its projection
[~,predicted_labels]=min(dist, [], 2);
%Compute misclassed, l0 and l1 norms of the classifiers
%Compute fraction of misclassified observations
misclassed=sum(abs(test_labels-predicted_labels)>0)/N;
%l0
A=(abs(w)>1e-3);
sum(A)
l0=sum(abs(w)>1e-3); %l0 is the number of non-zero entries
%l1
l1=sum(abs(w));

stats.mc=misclassed;
stats.l0=l0;
stats.l1=l1;
preds=predicted_labels;



    

