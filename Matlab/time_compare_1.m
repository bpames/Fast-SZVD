function [time1,time2]=time_compare_1(p,r,k,N,T)
%p: number of features
%r: value of constant covariance between features
%k: number of classes
%N: vector of number of training observations per class.
%Ntest: vector of number of testing observations per class.
%T: number of trials for p

%prepare the data set
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
        
        % Generate  and process (i,j)th training data.                        
        train = type1_data(p(i),r,k,N(i)); 
        [train_obs, mu_train, sig_train] = normalize(train(:,2:(p(i)+1)));
        train=[train(:,1), train_obs];
        
%         train=[train(:,1),normalize(train(:,2:end))];
%         train_obs=train(:,2:end);
%         mu_train=mean(train_obs);
%         sig_train=std(train_obs);
        
        % Solve using new version and record cpu time.
        tic;
        [DVs,~,~,~,classMeans] = SZVD_V4(train,D,penalty,tol,maxits,beta,quiet,gamma);
        time1(j, i) =toc; % Stop timer after training is finished.
        
        % Sample and normalize test data.
        test = type1_data(p(i),r,k,Ntest(i)); 
        test_obs=test(:,2:(p(i)+1));
        test_obs=normalize_test(test_obs,mu_train,sig_train);
        test(2,(p(i)+1)) = test_obs ;
        
        % Check classification and feature selection performance.
        [stats,~,~,~]=test_ZVD_V1(DVs,test,classMeans);
        NumErr1(j,i)=stats.mc;
        NumFeat1(j,i)=sum(stats.l0);
        
        %Repeat using the old code and save results to remaining matrices.
        tic;
        [DVs2,~,~,~,~,~]=SZVD_00(train,gamma,D,penalty,scaling,tol,maxits,beta,quiet);
        time2(j, i) =toc;
        
        [stats, ~] = test_ZVD_V1(DVs2,test,classMeans);
        NumErr2(j,i)=stats.mc;
        NumFeat2(j,i)=sum(stats.l0);
    end
end





