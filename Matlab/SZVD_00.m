function [DVs,its,pen_scal,N,classMeans,mus]=SZVD_00(train,gamma,D,penalty,scaling,tol,maxits,beta,quiet)
%Prepare the training data (ZVD step)
get_DVs=1;
[~,p]=size(train); 
p=p-1;
tic
w0 = ZVD(train, scaling, get_DVs);
classMeans=w0.means;
mus=w0.mu;
if (penalty==1)
    s=sqrt(diag(w0.W));
else
    s=ones(p,1);
end
w0.s=s;
%w0.B = 1/2*(w0.B + w0.B');
B0 = w0.N' * w0.B * w0.N;
d1 = w0.dvs(:,1);
B0 = (w0.N' * w0.B * w0.N)/(d1'*w0.B*d1);
B0 = (B0+B0')/2;
N = w0.N;
K=w0.k;
%Initialization for the output
DVs=zeros(p,K-1);
its=zeros(1,K-1);

ppt = toc;
fprintf('ppt %1.4d \n', ppt)
%Call ADMM

ntime =0;
st = 0;

for i=1:(K-1)
    %Initial solutions.
    tic
    sols0.x = N'*D'*w0.dvs(:,i);
    sols0.y = w0.dvs(:,i);
    sols0.z = zeros(p,1);
    [x,~,~,its]=SZVD_ADMM(B0,N,D,sols0,s,gamma(i),beta,tol,maxits,quiet);
    st =  st + toc;
    fprintf('solve time %1.4d \n', st)
    DVs(:,i)=D*N*x;
    DVs(:,i) = DVs(:,i)/norm(DVs(:,i));
    its(i)=its;
    if (quiet == 0)          
        fprintf('Found SZVD %g after %g its \n', i, its(i));
    end
    tic
    if(i<(K-1))
        %Project N onto orthogonal complement of Nx 
        x=DVs(:,i);
        x=x/norm(x);
        Ntmp=N-x*(x'*N);%null complement of W
        %QR factorization of Ntmp
        [Q,R]=qr(Ntmp);
        %Extract non-zero rows of R to get columns of Q to update N
        R_rows=(abs(diag(R))>1e-6);
        N=Q(:,R_rows);
        B0=N'*w0.B*N;
        B0=0.5*(B0+B0');
    end
    ntime = ntime + toc;
    fprintf('Nt %1.4d \n', ntime)
end
pen_scal=s;
end

        
        
    
    



