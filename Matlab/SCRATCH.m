% TESTING TYPE 1 DATA

p = 500;
r = 0.25;
k = 3;
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
p = 500;
N = 50*ones(k,length(p));
Ntest = 100*ones(k, length(p));
T = 1;

[time1,time2, err1, err2, feat1, feat2]=time_compare_1(p,r,k,N,Ntest, T);

%% Debug old SZVD.

% Generate  and process (i,j)th training data.                        
p = 500;
k = 3;
N = 50*ones(k,length(p));
Ntest = 100*ones(k, length(p));
r = 0.5;

train = type1_data(p,r,k,N); 
[train_obs, mu_train, sig_train] = normalize(train(:,2:(p+1)));
train=[train(:,1), train_obs];

%% Set up problem using new code.

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

%Compute k-1 nontrivial e-vectors of N'*B*N
RN = R*N;
%size(RN)
[~,sigma,w]=svd(RN);
w=w(:,1);
%calculate gamma
R=R/sigma(1,1);
RN = RN/sigma(1,1);
%gamma=0.5/norm((D*N*w),1);


Anew = RN'*RN;

%% Set up problem with old code.
scaling = 1;
get_DVs = 1;

w0 = ZVD(train, scaling, get_DVs);
classMeans=w0.means;
mus=w0.mu;

s=ones(p,1);
w0.s=s;
%w0.B = 1/2*(w0.B + w0.B');
B0 = (w0.N' * w0.B * w0.N)/(d1'*w0.B*d1);

%B0 = (B0+B0')/2;
N = w0.N;
K=w0.k;

%% Problem parameters.
gammascale=0.5;
penalty=0;
scaling=1;
beta=3;
tol.rel = 1e-5;
tol.abs= 1e-5;
maxits=100;
quiet=0;
D = eye(p);

%% Call new solver.
[DVs,x,~,~,~,classMeans,gamma] = SZVD_V6(train,D,penalty,tol,maxits,beta,quiet,gammascale);
plot(DVs)
nnz(DVs)
%% Call old solver.
[DVs,~,~,~,~,~]=SZVD_01(train,gamma,D,penalty,scaling,tol,maxits,beta,1);

plot(DVs2)
nnz(DVs2)

%% Compare.
errs = zeros(k-1,1);
for i = 1:(k-1)
    errs(i) = min([norm(DVs(:,i) - DVs2(:,i));norm(DVs(:,i) + DVs2(:,i))] );
end
errs