m = 3;
p = 2;
q = 1;

grow = 1; % number of grid points to grow, must be less than m

% each subdomain is m by m
% we have p by q processor grid

aglobal = laplacian2d(p*m,q*m);
size(aglobal)
%full(aglobal)

rng(0);
rhs    = rand(p*m,q*m)-.5
oldsol = zeros(p*m,q*m);
newsol = zeros(p*m,q*m);

% zero sources and sinks, random initial approximation
%rhs    = zeros(p*m,q*m); NOTE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
oldsol = rand(p*m,q*m)-.5; 
newsol = oldsol;

% exact solution
sol = aglobal \ rhs(:);
sol = reshape(sol, p*m, q*m);

% solver sweeps
for sweep = 1:1%500

  % loop over subdomains
  % for project, subdomains are processed in parallel
  for i=0:p-1
  for j=0:q-1
    colset = i*m+1:(i+1)*m;
    rowset = j*m+1:(j+1)*m;

    colset_expanded = colset;
    rowset_expanded = rowset;
    i
    j

    if (i > 0)   colset_expanded = [-grow+i*m+1:i*m colset_expanded];  end
    if (i < p-1) colset_expanded = [colset_expanded (i+1)*m+1:(i+1)*m+grow];  end

    if (j > 0)   rowset_expanded = [-grow+j*m+1:j*m rowset_expanded];  end
    if (j < q-1) rowset_expanded = [rowset_expanded (j+1)*m+1:(j+1)*m+grow];  end
   
%    colset_expanded
%    rowset_expanded 
    rhslocal = rhs(colset_expanded,rowset_expanded)

    % modify the rhs with the current solution
    
    % top boundary
    if (i > 0)
      disp('top boundary')
      oldsol
      ind = i*m-grow;
      oldsol(ind, rowset_expanded)
      rhslocal(1,:) = rhslocal(1,:) + oldsol(ind,rowset_expanded);
      rhslocal
    end

    % left boundary
    if (j > 0)
      disp('left boundary')
      oldsol
      ind = j*m-grow;
      rhslocal(:,1) = rhslocal(:,1) + oldsol(colset_expanded,ind);
      rhslocal
    end

    % bottom boundary
    if (i < p-1)
      disp('bottom boundary')
      oldsol
      ind = (i+1)*m+1+grow;
      rhslocal(end,:) = rhslocal(end,:) + oldsol(ind,rowset_expanded);
      rhslocal
    end

    % right boundary
    if (j < q-1)
      disp('right boundary')
      oldsol
      ind = (j+1)*m+1+grow;
      rhslocal(:,end) = rhslocal(:,end) + oldsol(colset_expanded,ind);
      rhslocal
    end

    % solve local problem
    alocal  = laplacian2d(length(colset_expanded),length(rowset_expanded));
    length(colset_expanded)
    length(rowset_expanded)
    full(alocal)
    x = alocal \ rhslocal(:);
    temp = reshape(x,length(colset_expanded),length(rowset_expanded));
    
%    colset(1)
%    colset_expanded(1)
    starti = colset(1)-colset_expanded(1)+1
    startj = rowset(1)-rowset_expanded(1)+1

    % write back to part of the solution
    newsol
    temp
    colset 
    rowset
    newsol(colset,rowset) = temp(starti:starti+m-1,startj:startj+m-1);
    newsol

  end%j
  end%i

  % print residual norm and error norm
  fprintf('%3d  %e  %e\n', sweep, norm(rhs(:) - aglobal*newsol(:)), ...
                              norm(newsol(:)-sol(:)) );

  % get ready for next sweep
  oldsol = newsol;
end%sweep
