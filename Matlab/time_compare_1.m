function [time1,time2]=time_compare_1(p,r,k,N,T)
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

%Initialize matrices for storing results
time1 = zeros(T, length(p));
time2=time1;NumErr1=time1;NumErr2=time1;NumFeat1=time1;NumFeat2 = time1;
for i=1:length(p)
    for j=1:T
        tic;
        %Train discriminant vectors using new method
        train=[train(:,1),normalize(train(:,2:end))];
        train_obs=train(:,2:end);
        mu_train=mean(train_obs);
        sig_train=std(train_obs);
        [DVs,~,~,~,classMeans]=SZVD_V4(train,D,penalty,tol,maxits,beta,quiet,gamma);
        time1(j, i) =toc; % Stop timer after training is finished.
        
        test_class=test(:,1);
        test_obs=test(:,2:end);
        test_obs=normalize_test(test_obs,mu_train,sig_train);
        test=[test_class,test_obs];
        [stats,~,~,~]=test_ZVD_V1(DVs,test,classMeans);
        NumErr1(j,i)=stats.mc;
        NumFeat1(j,i)=sum(stats.l0);
        
        %Repeat using the old code and save results to remaining matrices.
        tic;
        [DVs,~,~,~,classMeans,mus]=SZVD_00(train,gamma,D,penalty,scaling,tol,maxits,beta,quiet);
        time2(j, i) =toc;
        
        [stats, ~] = test_ZVD(DVs, test, classMeans, mus, scaling);
        NumErr2(j,i)=stats.mc;
        NumFeat2(j,i)=sum(stats.l0);
    end
end





