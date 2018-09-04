function [DVs,its,pen_scal,N,classMeans]=SZVD_V5(train,D,penalty,tol,maxits,beta,quiet,gamma)
%Normalize the training data
get_DVs=1;
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

size(M)

%Symmetrize W and B
R=ClassMeans';
%Find ZVDs 
N=null(M');
if (get_DVs==1)
    %Compute k-1 nontrivial e-vectors of N'*B*N
    [~,sigma,w]=svd(R*N);
    w=w(:,1);
    %calculate gamma
    R=R/sigma(1,1);
    %gamma=0.5/norm((D*N*w),1);
end
%Set sigma--penalty parameter
if (penalty==1)
    s=sqrt(diag(M*M'));
else
    s=ones(1,p);
end
%Initialization for the output
DVs=zeros(p,K-1);
its=zeros(1,K-1);
%Call ADMM
for i=1:(K-1)
    %Initial solutions.
    sols0.x = w;
    sols0.y = D*N*w;
    sols0.z = zeros(p,1);
    [x,y,~,its]=SZVD_ADMM_V(R,N,D,sols0,s,gamma,beta,tol,maxits,quiet);
    DVs(:,i) = y/norm(y);
    %DVs(:,i)=D*N*x;
    its(i)=its;
    if (quiet == 0)          
        fprintf('Found SZVD %g after %g its \n', i, its(i));
    end
    if(i<(K-1))
        %Project N onto orthogonal complement of Nx 
        x=D*(N*x);
        x=x/norm(x);
        N=Nupdate(N,x);
        [~,sigma,w]=svd(R*N);
        w=w(:,1);
        R=R/sigma(1,1);
        %gamma=0.5/norm((D*N*w),1);
    end
end
pen_scal=s;
end