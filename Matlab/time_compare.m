function [t1,t2]=time_compare(p,r,k,N,T)
%p: number of features
%r: value of constant covariance between features
%k: number of classes
%N: vector of number of observations per class
%T: number of trials for p

%prepare the data set
[obs,~,~]=type1_data(p,r,k,N);
[train,test]=train_test_split(obs,0.8);
[~,p]=size(train); 
p=p-1;
gamma=0.03;
penalty=0;
scaling=1;
beta=3;
tol.rel = 1e-3;
tol.abs= 1e-3;
maxits=100;
quiet=1;
D=eye(p);

%time the new version

%standaize train and test data 
train=[train(:,1),normalize(train(:,2:end))];
train_obs=train(:,2:end);
mu_train=mean(train_obs);
sig_train=std(train_obs);
test_class=test(:,1);
test_obs=test(:,2:end);
test_obs=normalize_test(test_obs,mu_train,sig_train);
test=[test_class,test_obs];
mc_1=zeros(1,T);
tic;
for t=1:T
    [DVs,its,pen_scal,N,classMeans]=SZVD_V4(train,D,penalty,tol,maxits,beta,quiet,gamma);
    [stats,preds,proj,cent]=test_ZVD_V1(DVs,test,classMeans);
    
   %preds;
end
t1=toc;


%time the old version
tic;
mc_2=zeros(1,T);
for t=1:T
    [DVs,its,pen_scal,N,classMeans,mus]=SZVD_00(train,gamma,D,penalty,scaling,tol,maxits,beta,quiet);
    [stats, preds] = test_ZVD(DVs, test, classMeans, mus, scaling);
end
t2=toc;





