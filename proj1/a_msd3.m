function out = a_msd3(filename, numframes)
% Example: out = a_msd3('output.xyz');

if nargin < 2
  numframes = 1000000;
end

% frame 2 is used for interval 1
% frame 3 is used for interval 1
%       4                      1 and 2
%       5                      1
%       6                      1, 2, 3

maxinterv = 9;
accum = zeros(maxinterv,1);
denom = zeros(maxinterv,1);

% read one frame at a time
% for each interval, save the last position

fid = fopen(filename, 'r');
i = 0;
while (1)
  [npos count] = fscanf(fid, '%d\n', 1); if (count < 1), break, end
  tline = fgets(fid);
  [pos count] = fscanf(fid, '%f', [4 npos]); if (count < 4*npos), break, end
  i = i + 1;
  if mod(i,1000)==0, fprintf('%d ', i); end

  % new: remove center of mass motion
  center = mean(pos(:,1:256),2);
  pos = pos - repmat(center, 1, npos);

  pos = pos(2:4,:);

  if (i == 1)
    % save copy of initial position
    pos0 = zeros(3,npos,maxinterv);
    for j=1:maxinterv
      pos0(:,:,j) = pos;
    end
    continue
  end

  % loop over all number of intervals
  for interv = 1:maxinterv
    if mod(i-1,interv) ~= 0
      continue
    end

    r = pos - pos0(:,:,interv);
    r = r';
    s_pro = r(:,1).*r(:,1) + r(:,2).*r(:,2) + r(:,3).*r(:,3);

    pos0(:,:,interv) = pos;

    accum(interv) = accum(interv) + mean(s_pro);
    denom(interv) = denom(interv) + 1;
  end

  if (i >= numframes)
    break;
  end
end
fclose(fid);
fprintf('\n');

out = accum ./ denom;

y = (1:maxinterv)';
x = 0.1*y; % 0.1 is the interval length in time
p = polyfit(x,out(y),1); 
fprintf('Diffusion constant: %f\n', p(1)/6);

