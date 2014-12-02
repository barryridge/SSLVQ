% randgen(mu,mu1,mu2,cov1,cov2,cov3) = Random generation of Gaussian Samples
% in d-dimensions where d = 2
% mu, mu1, mu2 = (x,y) coordinates(means) that the gaussian samples are centered around
% cov1, cov2, cov3 are the covariance matrices and will vary changing the
% shape of the distribution, example: cov = sigma^2*Identity Matrix, where sigma^2 = a scalar
% N = the number of gaussian samples used are provided as user input,
% A test set of N/2 and a training set of N/2 gaussian samples is also generated 
% Output is directed to the command window and a plot of the distributions are generated 
% example1: randgen([4 5],[9 0;0 9],[10 15],[6 0;0 6],[15 10],[4 0;0 4]) or
% example2: randgen([4 5],[9, 0; 0 9],[10 15],[5, 1.5; 1, 5.5],[15 10],[6, -1; -1, 4]) 
% by John Shell

function x1 = rand2DGaussian(mu1,cov1,N)

d = 2;

z = randn(N, d);                    %  N x d, 
meanz = mean(z)';                   %  d x 1
covz = cov(z);                      %  1 x d
meanz2 = [meanz meanz];             %  d x d
mu11 = mu1';
mu1 = [mu11 mu11];                  %  d x d

[row, col] = size(z');
[row1, col1] = size(meanz2);

while col1 ~= col
    meanz2 = [meanz2 meanz];
    mu1 = [mu1 mu11];
    
    col1 = col1 + 1;
end

[zvects, zeigs] = eig(covz);        % eigenvectors = d x d , eigenvalues = d x d, (lambdas=diagonals)
[x1vects, x1eig] = eig(cov1);

yp = (zvects * zeigs ^ (-.5))'*z'- meanz2;            % A Whitening Transform for zero means correction
x1 = (((x1vects * x1eig^(-.5))')^-1) * yp + mu1;      % d x N, the inverse Whitening to center the dist. to means 

x1p = x1';
meanx1 = mean(x1p);                         % d x d
covx1 = cov(x1p);
