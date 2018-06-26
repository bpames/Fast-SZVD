% TESTING TYPE 1 DATA

p = 500;
r = 0.5;
k = 4;
N = 100*ones(k,1);

[obs,mu,sigma]=type1_data(p,r,k,N);

%%
size(obs)
%%
plot(obs(101:200,2:p+1)')

%% TEST TIME COMPARE.
T = 3;
[t1,t2]=time_compare(p,r,k,N,T)