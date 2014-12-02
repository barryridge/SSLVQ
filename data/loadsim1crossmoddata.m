%
% This artificial data should be a close match to that described in
% Bojer et al, "Relevance determination in learning vector quantization" in
% Proceedings of European Symposium on Artificial Neural Networks 2001.
%
function Data = loadsim1crossmoddata(varargin)

    %% Arguments...
    add_noise = true;

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
    Foo = gencluster([0.28 0.15], [0.2, 0.2], [1, 1], [0, 0], 15, 1);
    Foo = [Foo' gencluster([0.65 0.075], [0.1, 0.1], [1, 1], [0, 0], 15, 2)']';
    Foo = [Foo' gencluster([0.1 0.65], [0.2, 0.15], [1, 1], [0, 0], 15, 2)']';
    Foo = [Foo' gencluster([0.55 0.95], [0.2, 0.15], [1, 1], [0, 0], 15, 3)']';
    Foo = [Foo' gencluster([0.75 0.6], [0.15, 0.075], [1, 1], [0, 0], 15, 1)']';
    Foo = [Foo' gencluster([0.85 0.65], [0.15, 0.06], [1, 1], [0, 0], 15, 3)']';        
    
    % Add noise dimensions if desired...
    if add_noise
        
        % Temporarily grab class dimension...
        ClassDim = Foo(:,end);

        % Add dimensions by adding noise to first dimension...
        Foo(:,end) = Foo(:,1) + (randn(size(Foo(:,end))) * sqrt(0.05));
        Foo(:,end+1) = Foo(:,1) + (randn(size(Foo(:,end))) * sqrt(0.1));
        Foo(:,end+1) = Foo(:,1) + (randn(size(Foo(:,end))) * sqrt(0.2));
        Foo(:,end+1) = Foo(:,1) + (randn(size(Foo(:,end))) * sqrt(0.5));

        % Add dimensions by adding pure noise...
        Foo(:,end+1) = rand(size(Foo(:,end))) - 0.5;
        Foo(:,end+1) = (rand(size(Foo(:,end))) * 0.4) - 0.2;
        Foo(:,end+1) = randn(size(Foo(:,end))) * sqrt(0.5);
        Foo(:,end+1) = randn(size(Foo(:,end))) * sqrt(0.2);   

        % Add the class dimension back...
        Foo(:,end+1) = ClassDim;
        
    end
    
    %% Build the Data struct...
    Y = Foo(:,end);
    X = Foo(:,1:end-1);

    for iDim = 1:size(X,2)
        Data.FeatureNames{iDim} = ['Dim ' num2str(iDim)];
    end

    for iY = 1:size(Y,1)
        if Y(iY) == 1
            StrY{iY} = 'Class One';
        elseif Y(iY) == 2
            StrY{iY} = 'Class Two';
        elseif Y(iY) == 3
            StrY{iY} = 'Class Three';
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
    Data.Modalities{1}.FeatureIndices = 1:size(Data.FeatureVectors,1);
    Data.Modalities{1}.FeatureMask = ones(1,size(Data.FeatureVectors,1));
    % Modality 2:
    Data.Modalities{2}.FeatureIndices = 1:size(Data.FeatureVectors,1);
    Data.Modalities{2}.FeatureMask = ones(1,size(Data.FeatureVectors,1));
    
end

function Cluster = gencluster(Center, Bandwidth, UpperBound, LowerBound, nPoints, iClass)
    for iDim = 1:length(Center)
        Cluster(:,iDim) = (rand([nPoints 1]) * Bandwidth(iDim)) + (Center(iDim) - (Bandwidth(iDim) / 2));
        Cluster(:,iDim) = min(Cluster(:,iDim), repmat(UpperBound(iDim), length(Cluster(:,iDim)), 1));
        Cluster(:,iDim) = max(Cluster(:,iDim), repmat(LowerBound(iDim), length(Cluster(:,iDim)), 1));
    end
    Cluster(:,end+1) = ones(1,nPoints) * iClass;
end
