function [Hom,Sep,Cindex,Wintp,Dunn,DB] = valid_internal_intra(Smatrix,U,dtype,id)
% indices base on intra and inter similarity

Hom = 0;
Sep = 0;
Wintp = 0;
Dunn = 0;
DB = 0;
k = length(U);
nrow = size(Smatrix,1);

[avintra, avinter, intra, inter, nintra, ninter] = valid_intrainter(Smatrix,U);
% DB & Dunn index based on average diameter & linkage distance
%[DB, Dunn] = valid_DbDunn(avintra, avinter, k);

ns = nrow*(nrow-1);         % number of all pairs of elements
for i = 1:nrow
   Smatrix(i,i) = 0;
end
cut = (sum(sum(Smatrix))-nrow)/ns;
ns = sum(nintra);
S = sum(intra);
[wintra, dbs] = find_nearpoint(Smatrix, cut, ns, 2);
for i = 1:nrow
   Smatrix(i,i) = inf;
end
[sinter, ni] = find_nearpoint(Smatrix, cut, ns, 1);
if ni < dbs
   wintra = wintra(1:ni);
elseif ni > dbs
   sinter = sinter(1:dbs);
end
R = Smatrix(wintra);
Smax = sum(R);
R = Smatrix(sinter);
Smin = sum(R);
Cindex = (S-Smin)/(Smax-Smin);        % C index, Hubert and Levin

if id
   return;
end

ns = k*(k-1)/2;                                    % mates of clusters
[avintra, avinter, intra, inter, nintra, ninter] = valid_intrainter(1-Smatrix,U);
% Homogeneity & Separation
Hom = sum(intra)/sum(nintra);          % average intra similarity
Sep = Hom;
if ns > 0
  Sep = (sum(sum(inter)))/(sum(sum(ninter)));    % average inter
end
if dtype == 1
   Hom = 1-Hom;
   Sep = 1-Sep;
elseif dtype == 2
   Hom = Hom+Hom -1;
   Sep = Sep+Sep -1;
end

% weight_intertra
wintra = zeros(1,k);
Sinter = zeros(1,k);
sinter = zeros(1,k);
Inter = inter+inter';
for i = 1:k
  ind = U{i};
  ni = length(ind);
  Sinter(i) = sum(Inter(i,:))/(nrow-ni);
  sinter(i) = sum(inter(i,:))/(nrow-ni);
  if ni ==1
     ni = 2;
  end
  wintra(i)=2*intra(i)/(ni-1);
end
if k == 2
   sinter(2) = 0.5*Sinter(2);
end
Sintra = sum(wintra);
%Sinter = sum(Sinter);
Sinter = sum(sinter);
Wint = 1-Sinter/Sintra;
Wintp = (1-2*k/nrow)*Wint;    % penalized
