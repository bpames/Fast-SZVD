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
p = [500];
N = 50*ones(k,length(p));
Ntest = 100*ones(k, length(p));
T = 1;

[time1,time2]=time_compare_1(p,r,k,N,Ntest, T);