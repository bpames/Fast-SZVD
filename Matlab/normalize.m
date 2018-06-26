function x=normalize(x)
[n,~]=size(x);
x=x-ones(n,1)*mean(x);
sig=std(x);
%sig(~sig(:))=1;
x=x*diag(1./sig);
end
