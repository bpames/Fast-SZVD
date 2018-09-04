% TESTING TYPE 1 DATA

p = 500;
r = 0.25;
k = 4;
N = 100*ones(k,1);

[obs,mu,sigma]=type1_data(p,r,k,N);

%%
size(obs)
%%
plot(obs(1:100,2:p+1)')

%% TEST TIME COMPARE.
T = 3;
[t1,t2]=time_compare(p,r,k,N,T);

%% TEST TIME COMPARE 1.
k = 4;
p = [1500];
N = 50*ones(k,length(p));
Ntest = 100*ones(k, length(p));
T = 1;

[time1,time2, err1, err2, feat1, feat2]=time_compare_1(p,r,k,N,Ntest, T);

%% Debug old SZVD.
% Generate  and process (i,j)th training data.                        
train = type1_data(p(i),r,k,N(:, i)); 
[train_obs, mu_train, sig_train] = normalize(train(:,2:(p(i)+1)));
train=[train(:,1), train_obs];

% Process using steps in SZVD_V6
classes=train(:,1);
[n,p]=size(train);
X=train(:,2:p);
%X=normalize(X);
%Extract observations 
labels=unique(classes); 
K=length(labels);
%Initiate matrix of within-class means
p=p-1;
classMeans=zeros(p,K);
ClassMeans=zeros(p,K);
w=zeros(p,K-1);
%for each class, make an object in the list containing only the obs of that
%class and update the between and within-class sample
M=zeros(p,n);

for i=1:K    
    class_obs=X(classes==labels(i),:);
    %Get the number of obs in that class (the number of rows)
    ni=size(class_obs,1);
    %Compute within-class mean
    classMeans(:,i)=mean(class_obs);
    %Update W 
    xj=class_obs-ones(ni,1)*classMeans(:,i)';
    M(:,classes == labels(i)) =xj';
    ClassMeans(:,i)=mean(class_obs)*sqrt(ni);
end

%Symmetrize W and B
R=ClassMeans';
%Find ZVDs 
N=null(M');
if (get_DVs==1)
    %Compute k-1 nontrivial e-vectors of N'*B*N
    RN = R*N;
    size(RN)
    [~,sigma,w]=svd(R*N);
    w=w(:,1);
    %calculate gamma
    R=R/sigma(1,1);
    %gamma=0.5/norm((D*N*w),1);
end
