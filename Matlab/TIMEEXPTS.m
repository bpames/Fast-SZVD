%% Set up problem parameters.
k = 4;
p = [500, 1000];
N = [50*ones(k,1), 100*ones(k,1)];
Ntest = 500*ones(k, length(p));
T = 2;
savemat = true;
r = 0.5;

%% Run experiment.
[time1,time2, err1, err2, feat1, feat2]=time_compare_1(p,r,k,N,Ntest, T, savemat);

%% Test boolean.

if savemat
    x = 1;
end