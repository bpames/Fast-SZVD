%% Set up problem parameters.
k = 4;
%p = [500, 1000];

p = 25*2.^(0:8);
N = [50*ones(k,1), 100*ones(k,1)];
Ntest = 500*ones(k, length(p));
T = 2;
savemat = true;
r = 0.5;

%% Run experiment.
[time1,time2, err1, err2, feat1, feat2]=time_compare_1(p,r,k,N,Ntest, T, savemat);

%% SCRATCH.

p = [50:50:500, 600:100:1000, 1250: 250:2500, 3000:500:5000];
