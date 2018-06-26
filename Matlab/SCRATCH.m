% TESTING TYPE 1 DATA

p = 500;
r = 0.5;
k = 4;
N = 100;

[obs,mu,sigma]=type1_data(p,r,k,N);

%%
plot(obs)