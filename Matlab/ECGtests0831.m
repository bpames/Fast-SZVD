% Applies SZVD_V4 to ECG data.

% Load data.
load('ECGdata.mat')
train = ECGtrain;
test = ECGtest;

%% Set problem parameters.
[n,p] = size(train); 
D=eye(p-1);               % Basis for the sparse solution
penalty = 0;
%tol = 1e-4;
maxits = 500;
beta = 3;
quiet = 0;
tol.rel = 1e-3;
tol.abs= 1e-3;
%%
gamma = 0.10

[DVs,its,pen_scal,N,classMeans]=SZVD_V4(train,D,penalty,tol,maxits,beta,quiet,gamma);
plot(DVs)
nnz(DVs)
norm(DVs)

%%
%gamma = 0.1

[DVs2,its,pen_scal,N,classMeans]=SZVD_V5(train,D,penalty,tol,maxits,beta,quiet,gamma);
plot(DVs2)
nnz(DVs2)
norm(DVs2)