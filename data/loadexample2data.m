%
% 
%
function Data = loadexample2data(varargin)

    %% Arguments...
    add_noise = false;

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
    Foo = rand2DGaussian([0.2 0.5],[0.0005 0;0 0.01],250)';
    Foo(:,3) = 1;
    Bar = rand2DGaussian([0.4 0.5],[0.0005 0;0 0.01],250)';
    Bar(:,3) = 2;
    Foo1 = rand2DGaussian([0.6 0.5],[0.0005 0;0 0.01],250)';
    Foo1(:,3) = 1;
    Bar1 = rand2DGaussian([0.8 0.5],[0.0005 0;0 0.01],250)';
    Bar1(:,3) = 2;
    Foo = [Foo' Bar' Foo1' Bar1']';
    
    %     Foo = gencluster([0.2 0.5], [0.1, 1], [1, 1], [0, 0], 250, 1);
    %     Foo = [Foo' gencluster([0.4 0.5], [0.1, 1], [1, 1], [0, 0], 250, 2)']';
    %     Foo = [Foo' gencluster([0.6 0.5], [0.1, 1], [1, 1], [0, 0], 250, 1)']';
    %     Foo = [Foo' gencluster([0.8 0.5], [0.1, 1], [1, 1], [0, 0], 250, 2)']';
    % Foo = [Foo' gencluster([0.75 0.6], [0.15, 0.075], [1, 1], [0, 0], 15, 1)']';
    % Foo = [Foo' gencluster([0.85 0.65], [0.15, 0.06], [1, 1], [0, 0], 15, 3)']';
    
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

    for iDim = 1:size(Foo,2)
        Data.FeatureNames{iDim} = ['Dim ' num2str(iDim)];
    end

    for iY = 1:size(Y,1)
        StrY{iY} = num2str(Y(iY));
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
    
    %% Plot the data...
    C1 = Data.FeatureVectors(:,Data.ClassLabels(1,:));
    C2 = Data.FeatureVectors(:,Data.ClassLabels(2,:));
    
    figure;
    plot(C1(1,:), C2(2,:), '+b');
    hold;
    plot(C2(1,:), C2(2,:), 'or');
    axis([0,1,0,1]);
    
    % Fisher criterion scores...
    FCS = var([mean(C1,2) mean(C2,2)]',1) ./ (var(C1') + var(C2'));
    xscore = ['Fisher criterion score for X-dimension = ' num2str(FCS(1), '%3.4f')];
    xlabel(xscore);
    yscore = ['Fisher criterion score for Y-dimension = ' num2str(FCS(2), '%3.4f')];
    ylabel(yscore);
    
end

function Cluster = gencluster(Center, Bandwidth, UpperBound, LowerBound, nPoints, iClass)
    for iDim = 1:length(Center)
        Cluster(:,iDim) = (rand([nPoints 1]) * Bandwidth(iDim)) + (Center(iDim) - (Bandwidth(iDim) / 2));
        Cluster(:,iDim) = min(Cluster(:,iDim), repmat(UpperBound(iDim), length(Cluster(:,iDim)), 1));
        Cluster(:,iDim) = max(Cluster(:,iDim), repmat(LowerBound(iDim), length(Cluster(:,iDim)), 1));
    end
    Cluster(:,end+1) = ones(1,nPoints) * iClass;
end