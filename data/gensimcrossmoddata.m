%
% 
%
function Data = gensimcrossmoddata(varargin)

    %% Defaults...    
    sample_size = 1000;
    % Clusters = {{1st Cluster Class Num,
    %              Mod 1 Position, Mod 1 Cov matrix,
    %              Mod 2 Position, Mod 2 Cov matrix,
    %              Num of samples (optional)},...
    %             {2nd Cluster Class Num,
    %              Mod 1 Position, Mod 1 Cov matrix,
    %              Mod 2 Position, Mod 2 Cov matrix,
    %              Num of samples (optional)},...
    %             {3rd Cluster...etc...}         };
    Clusters = {{1, [0.50 0.25],[0.0005 0;0 0.0005], [0.50 0.75],[0.0005 0;0 0.0005], round(sample_size/3)},...
                {2, [0.25 0.75],[0.0005 0;0 0.0005], [0.25 0.25],[0.0005 0;0 0.0005], round(sample_size/3)},...
                {3, [0.75 0.75],[0.0005 0;0 0.0005], [0.75 0.75],[0.0005 0;0 0.0005], round(sample_size/3)}};            
    add_noise = false;
    Mod1NoiseParams = [];
    Mod2NoiseParams = [];
    Mod1XDims = 1:2;
    Mod2XDims = 3:4;

    % Loop through arguments...
    i = 1;
    while i <= length(varargin), 
        argok = 1;
        if isnumeric(varargin{i})
            sample_size = varargin{i};
            
        elseif iscell(varargin{i})
            Clusters = varargin{i};

        elseif ischar(varargin{i}), 
            switch lower(varargin{i}), 
                case {'noise', 'addnoise', 'add_noise'},
                    i=i+1;
                    if islogical(varargin{i})
                        add_noise = varargin{i};
                        Mod1NoiseParams = sqrt([0.005 0.05 0.5]);
                        Mod2NoiseParams = sqrt([0.005 0.05 0.5]);
                        
                    elseif isnumeric(varargin{i}) && length(varargin{i}) == 2 && ~any(varargin{i} < 0)
                        add_noise = true;
                        NumNoiseDims = varargin{i};                        
                        Mod1NoiseParams = sqrt(0.05:(1-0.05)/(NumNoiseDims(1)-1):1);                        
                        Mod2NoiseParams = sqrt(0.05:(1-0.05)/(NumNoiseDims(2)-1):1);                        
                    end
            
                otherwise
                    argok = 0;
            end
        else
            argok = 0;
        end

        if ~argok, 
            disp(['loadsim1data(): Ignoring invalid argument #' num2str(i)]);
            fprintf(obj.UsageMessage);
        end

        i = i + 1;
    end

    %% Generate the data...
    % Seed the random number generator...
    v = ver('matlab');
    if str2double(v.Version) < 7.7
        rand('twister', 42);
    else
        RandStream.setDefaultStream(RandStream('mrg32k3a', 'seed', 42));
    end

    Mod1X = [];
    Mod2X = [];
    X = [];
    Y = [];
    
    % Generate clusters for each class in 2 dimensions...
    for iCluster = 1:length(Clusters)
        
        Cluster = Clusters{iCluster};
        
        if size(Cluster,2) < 6
            cluster_sample_size = round(sample_size/length(Clusters));
        else
            cluster_sample_size = Cluster{6};
        end
        
        Mod1X = [Mod1X rand2DGaussian(Cluster{2}, Cluster{3}, cluster_sample_size)];
        Mod2X = [Mod2X rand2DGaussian(Cluster{4}, Cluster{5}, cluster_sample_size)];

        Y = [Y (Cluster{1} * ones(1,cluster_sample_size))];
    end
    
    % Add noise if required...
    if add_noise
        
        % Modality 1...
        % Add noise to the first dimension, progressively more noise for
        % each dimension required...
        for iDim = 1:length(Mod1NoiseParams)
            Mod1X(end+1,:) = Mod1X(1,:) + randn(size(Mod1X(1,:))) * Mod1NoiseParams(iDim);
        end
        
        Mod1XDims = 1:size(Mod1X,1);
        
        % Modality 2...
        % Add noise to the first dimension, progressively more noise for
        % each dimension required...
        for iDim = 1:length(Mod2NoiseParams)            
            Mod2X(end+1,:) = Mod2X(1,:) + randn(size(Mod2X(1,:))) * Mod2NoiseParams(iDim);
        end
        
        Mod2XDims = Mod1XDims(end)+1 : Mod1XDims(end) + size(Mod2X,1);
    end
    
    %% Build the Data struct...
    X = [Mod1X; Mod2X]';                                           
    
    Data.FeatureVectors = X';

    for iDim = 1:size(X,2)
        Data.FeatureNames{iDim} = ['Dim ' num2str(iDim)];
    end

    UniqueYs = unique(Y);
    Data.ClassLabels = zeros(size(UniqueYs,2), size(Y,2));
    for iY = 1:size(UniqueYs,2)
        Data.ClassNames{iY} = ['Class ' num2str(UniqueYs(iY))];
        Data.ClassLabels(iY, Y==UniqueYs(iY)) = 1;
    end           

    %% Restrict the training class set...
    Data.AllowedTrainingClassIndices = 1:size(Data.ClassNames,2);
    Data.GroundTruthClassIndices = 1:size(Data.ClassNames,2);

    %% Split the feature set into multiple modalities...
    % Modality 1:
    Data.Modalities{1}.FeatureIndices = Mod1XDims;
    Data.Modalities{1}.FeatureMask = zeros(1,size(Data.FeatureVectors,1));
    Data.Modalities{1}.FeatureMask(Mod1XDims) = 1;
    % Modality 2:
    Data.Modalities{2}.FeatureIndices = Mod2XDims;
    Data.Modalities{2}.FeatureMask = zeros(1,size(Data.FeatureVectors,1));
    Data.Modalities{2}.FeatureMask(Mod2XDims) = 1;
    
end

function Cluster = gencluster(Center, Bandwidth, UpperBound, LowerBound, nPoints, iClass)
    for iDim = 1:length(Center)
        Cluster(:,iDim) = (rand([nPoints 1]) * Bandwidth(iDim)) + (Center(iDim) - (Bandwidth(iDim) / 2));
        Cluster(:,iDim) = min(Cluster(:,iDim), repmat(UpperBound(iDim), length(Cluster(:,iDim)), 1));
        Cluster(:,iDim) = max(Cluster(:,iDim), repmat(LowerBound(iDim), length(Cluster(:,iDim)), 1));
    end
    Cluster(:,end+1) = ones(1,nPoints) * iClass;
end