function [val_w, DVs, gamma,gammas, max_gamma,its, w, x0, scaler, val_score, classMeans] = SZVD_Val_V2(Atrain, Aval,D,num_gammas,gmults, sparsity_level, penalty, beta, tol, maxits,quiet)
%Atrain: training data
%D: Penalty dictionary basis matrix
%Aval: validation set
%k: #of classes within the training and validation sets
%num_gammas: number of gammas to train on
%g_mults:(c_min, c_max): parameters defining range of gammas to train g_max*(c_min, c_max)

%Call ZVD to solve the unpenalized problem
get_DVs=1;
classes=Atrain(:,1);
[~,p]=size(Atrain);
X=Atrain(:,2:p);
%mu=mean(X);
%sig=std(X);
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
M=[];
for i=1:K
    class_obs=X(classes==labels(i),:);
    %Get the number of obs in that class (the number of rows)
    ni=size(class_obs,1);
    %Compute within-class mean
    classMeans(:,i)=mean(class_obs);
    %Update W 
    xj=class_obs-ones(ni,1)*classMeans(:,i)';
    M=[M,xj'];
    ClassMeans(:,i)=mean(class_obs)*sqrt(ni);
end
%Symmetrize W and B
R=ClassMeans';
N=null(M');
%Find ZVDs 
if (get_DVs==1)
    %Compute k-1 nontrivial e-vectors of N'*B*N
    [~,sigma,w]=svd(R*N);
    w=w(:,1);
    %calculate gamma
    R=R/sigma(1,1);
    %Project back to the originial space
end
R0=R;
w0=w;
%Extract scaling vector for weighted l1 penalty and diagonal penalty matrix
if (penalty==1)
    s=sqrt(diag(M*M'));
else
    s=ones(1,p);
end
%Initialize the validation scores
val_score=zeros(num_gammas,1);
%mc_ind=1;
%l0_ind=1;
best_ind=1;
%min_mc=1;
%min_l0=p;
triv=0;
%Initialize DVs and iterations
N0=N;
DVs=zeros(p,K-1,num_gammas);
its=zeros(K-1,num_gammas);
%For each gamma, calculate ZVDs and corresponding validation score
gammas=zeros(num_gammas,K-1);
for i=1:num_gammas
    N=N0;
    x0=w0;
    R=R0;
    %Get DVs 
    for j=1:K-1
        % Initial solutions.
        max_gamma=0.5/norm((D*N*x0),1);
        gammas(i,j)=gmults(i)*max_gamma;
        sols0.x = x0;
        sols0.y = D*N*x0;
        sols0.z = zeros(p,1);
        quietADMM=1;
        %Call ADMM
        [tmpx,~,~,tmpits]=SZVD_ADMM_V(R,N,D,sols0,s,gammas(i,j),beta,tol,maxits,quietADMM);
        DVs(:,j,i)=D*N*tmpx;
        its(j,i)=tmpits;
        %Update N and B 
        if j< K-1
            x=DVs(:,j,i);
            x=x/norm(x);
            N=Nupdate(N,x);
            [~,sigma,w]=svd(R*N);
            w=w(:,1);
            R=R/sigma(1,1);
            x0=w;
        end
    end
    %Get performance scores on the validation set
    %Call test ZVD to get predictions
    [stats,~,proj,cent]=test_ZVD_V1(DVs(:,:,i), Aval, classMeans);%
    %%%%%%%%%%%%
    %If gamma gives trivial sol, give it a large penalty
    if (sum(stats.l0)<3)
        val_score(i)=100*size(Aval,1);
        triv=1;
    else
        if (sum(stats.l0)>sparsity_level*size(DVs(:,:,i),1))
            val_score(i)=sum(stats.l0);
        else
            val_score(i)=stats.mc;
        end
    end
    %Update the best gamma so far
    if (val_score(i)<=val_score(best_ind))
        best_ind=i;
    end
    %Record sparest nontrivial sol 
    %if (min(stats.l0)>3 && stats.l0<min_l0)
    %    l0_ind=1;
    %    l0_x=DVs(:,:,i);
    %    min_10=stats.l0;
    %end
    %Record best misclassificartion error
    %if(stats.mc<=min_mc)
    %    min_ind=i;
    %    mc_x=DVs(:,:,i);
    %    min_mc=stats.mc;
    %end
    %Terminate if a trivial solution has been found
    if (quiet==0)
       fprintf('it = %g, val_score= %g, mc=%g, l0=%g, its=%g \n', i, val_scores(i), stats.mc, sum(stats.l0), mean(its(:,i)));
    end
    if(triv==1)
        break;
    end
end
%Export DVs found using validation
val_w=DVs(:,:,best_ind);
gamma=gammas(best_ind,:);
scaler=gmults(best_ind);
val_score=val_score(best_ind);
