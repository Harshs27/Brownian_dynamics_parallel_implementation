function a = laplacian2d(m,n) 
% a = laplacian2d(m,n) 
%  returns 2D 5-point laplacian matrix for m by n grid

e = -ones(m*n,1);
b1 = e;
b2 = e;
for i = m:m:n*m
   b1(i) = 0;
   b2(i-m+1) = 0;
end
a = spdiags([e b1 -4*e b2 e], [-m,-1,0,1,m], m*n, m*n);
