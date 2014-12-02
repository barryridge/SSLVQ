function [DB,CH,Dunn,KL,Han,st] = valid_internal_deviation(data,labels,dtype)
% cluster validity indices based on deviation & sum of squares

[nrow,nc] = size(data);
labels = double(labels);
k=max(labels);
if dtype == 1
   [st,sw,sb,cintra,cinter] = valid_sumsqures(data,labels,k);
else
   [st,sw,sb,cintra,cinter] = valid_sumpearson(data,labels,k);
end
ssw = trace(sw);
ssb = trace(sb);

if k > 1
% Davies-Bouldin & Dunn based on centroid diameter & linkage distance
  [DB, Dunn] = valid_DBDunn(cintra, cinter, k);
  CH = ssb/(k-1); 
else
  CH =ssb; 
  DB = NaN;
  Dunn = NaN; 
end

CH = (nrow-k)*CH/ssw;    % Calinski-Harabasz
Han = ssw;                        % component of Hartigan  
KL = (k^(2/nc))*ssw;         % component of Krzanowski-Lai
