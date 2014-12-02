%
% 1 class, regular bi-modal configuration...
%
function Data = loadsim2crossmoddata(varargin)

    %% Defaults...
    add_noise = false;
    sample_size = 1000;

    % Loop through arguments...
    i = 1;
    while i <= length(varargin), 
        argok = 1;
        if isnumeric(varargin{i})
            sample_size = varargin{i};

        elseif ischar(varargin{i}), 
            switch lower(varargin{i}), 
                case {'noise', 'addnoise', 'add_noise'},
                    i=i+1; add_noise = varargin{i};
            
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

    % Generate clusters for each class in 2 dimensions...
    Mod1XClass1 = rand2DGaussian([0.5 0.5],[0.01 0;0 0.01],sample_size)';    
    % Mod1XClass2 = rand2DGaussian([0.75 0.5],[0.0005 0;0 0.01],sample_size/2)';
    Mod2XClass1 = rand2DGaussian([0.5 0.5],[0.01 0;0 0.01],sample_size)';
    % Mod2XClass2 = rand2DGaussian([0.5 0.75],[0.01 0;0 0.0005],sample_size/2)';
    YClass1 = ones(sample_size,1);    
            
    %% Build the Data struct...
    X = [Mod1XClass1'; Mod2XClass1']';
    Y = [YClass1']';        

    for iDim = 1:size(X,2)
        Data.FeatureNames{iDim} = ['Dim ' num2str(iDim)];
    end

    for iY = 1:size(Y,1)
        if Y(iY) == 1
            StrY{iY} = 'Class One';
        elseif Y(iY) == 2
            StrY{iY} = 'Class Two';
        end
    end
    Y = StrY;

    Data.ClassNames = unique(Y);

    Data.FeatureVectors = X';

    for iClass = 1:size(Data.ClassNames,2)
        Data.ClassLabels(iClass,:) = strcmp(Y, Data.ClassNames{iClass})';
    end

    %% Restrict the training class set...
    Data.AllowedTrainingClassIndices = 1:size(Data.ClassNames,2);
    Data.GroundTruthClassIndices = 1:size(Data.ClassNames,2);

    %% Split the feature set into multiple modalities...
    % Modality 1:
    Data.Modalities{1}.FeatureIndices = 1:2;
    Data.Modalities{1}.FeatureMask = zeros(1,size(Data.FeatureVectors,1));
    Data.Modalities{1}.FeatureMask(1:2) = 1;
    % Modality 2:
    Data.Modalities{2}.FeatureIndices = 3:4;
    Data.Modalities{2}.FeatureMask = zeros(1,size(Data.FeatureVectors,1));
    Data.Modalities{2}.FeatureMask(3:4) = 1;
    
end

function Cluster = gencluster(Center, Bandwidth, UpperBound, LowerBound, nPoints, iClass)
    for iDim = 1:length(Center)
        Cluster(:,iDim) = (rand([nPoints 1]) * Bandwidth(iDim)) + (Center(iDim) - (Bandwidth(iDim) / 2));
        Cluster(:,iDim) = min(Cluster(:,iDim), repmat(UpperBound(iDim), length(Cluster(:,iDim)), 1));
        Cluster(:,iDim) = max(Cluster(:,iDim), repmat(LowerBound(iDim), length(Cluster(:,iDim)), 1));
    end
    Cluster(:,end+1) = ones(1,nPoints) * iClass;
end