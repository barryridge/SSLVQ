%
% 
%
function Data = loadexample3data(varargin)

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
    
    %% Train a classifier...
    LearnerObject = UniLayeredLearner('Data', Data,...
                                      'normalization', 'train',...
                                      'randomize_train', true,...
                                       'name', 'GLVQ',...
                                       'type', 'unilayer',...
                                       'modality_types', {'codebook'},...
                                       'codebook_sizes', {[4 1]},...
                                       'codebook_neighs', {'bubble'},...
                                       'codebook_lattices', {'hexa'},...D'
                                       'codebook_init_method', {'sample'},...
                                       'phase_shifts', {{NaN}},...
                                       'updaters', {{'GLVQ'}},...
                                       'alpha_types', {{'constant'}},...
                                       'alpha_inits', {{0.1}},...
                                       'radius_types', {{'linear'}},...
                                       'radius_inits', {{5}},...
                                       'radius_fins', {{1}},...
                                       'window_sizes', {{NaN}},...
                                       'alpha_feature_types', {{'constant'}},...
                                       'alpha_feature_inits', {{NaN}},...
                                       'feature_selection', 'fuzzy',...
                                       'feature_selection_max', NaN,...
                                       'feature_selection_feedback', true,...
                                       'metric', {'euclidean'},...
                                       'classification_method', 'node');                                 
    
    %% Plot the data...
    C1 = Data.FeatureVectors(:,Data.ClassLabels(1,:));
    C2 = Data.FeatureVectors(:,Data.ClassLabels(2,:));
%     C1 = LearnerObject.TrainingData1Epoch.Modalities{1}.NormedFeatureVectors(:,LearnerObject.TrainingData1Epoch.Modalities{1}.ClassLabels(1,:));
%     C2 = LearnerObject.TrainingData1Epoch.Modalities{1}.NormedFeatureVectors(:,LearnerObject.TrainingData1Epoch.Modalities{1}.ClassLabels(2,:));
    
    figure1 = figure;
    plot(C1(1,:), C2(2,:), '+b', 'LineWidth', 1);
    hold;
    plot(C2(1,:), C2(2,:), 'or');
    axis([0,1,0,1]);
    
    ClustA_1 = C1(:,1:250);
    ClustA_2 = C1(:,251:500);
    ClustB_1 = C2(:,1:250);
    ClustB_2 = C2(:,251:500);
    
    Prot1 = mean(ClustA_1,2);
    Prot2 = mean(ClustA_2,2);
    Prot3 = mean(ClustB_1,2);
    Prot4 = mean(ClustB_2,2);
    
    plot(Prot1(1), Prot1(2), 'g+', 'MarkerSize', 10, 'LineWidth', 2);
    annotation(figure1,'textarrow',[0.3461 0.2869],[0.5594 0.5206],...
    'TextEdgeColor','none',...
    'String',{' A_1'});
    % text(Prot1(1), Prot1(2), 'A', 'Color', 'g', 'LineWidth', 2);
    plot(Prot2(1), Prot2(2), 'g+', 'MarkerSize', 10, 'LineWidth', 2);
    annotation(figure1,'textarrow',[0.5023 0.4408],[0.5594 0.5183],...
    'TextEdgeColor','none',...
    'String',{' B_1'});
    plot(Prot3(1), Prot3(2), 'go', 'MarkerSize', 10, 'LineWidth', 2);
    annotation(figure1,'textarrow',[0.6526 0.5941],[0.5576 0.5133],...
    'TextEdgeColor','none',...
    'String',{' A_2'});
    plot(Prot4(1), Prot4(2), 'go', 'MarkerSize', 10, 'LineWidth', 2);
    annotation(figure1,'textarrow',[0.8013 0.7461],[0.5658 0.5102],...
    'TextEdgeColor','none',...
    'String',{' B_2'});

    FCS_A_1_B_1 = var([mean(ClustA_1,2) mean(ClustB_1,2)]',1) ./ (var(ClustA_1') + var(ClustB_1'));
    FCS_A_2_B_2 = var([mean(ClustA_2,2) mean(ClustB_2,2)]',1) ./ (var(ClustA_2') + var(ClustB_2'));

    TextBoxString1 = ['F(x) over A_1 & B_1 = ' num2str(FCS_A_1_B_1(1), '%3.4f')...
                     ', F(x) over A_2 & B_2 = ' num2str(FCS_A_2_B_2(1), '%3.4f')];
    TextBoxString2 = ['F(y) over A_1 & B_1 = ' num2str(FCS_A_1_B_1(2), '%3.4f')...
                     ',  F(y) over A_2 & B_2 = ' num2str(FCS_A_2_B_2(2), '%3.4f')];

    annotation(figure1,'textbox',[0.1528 0.1333 0.7258 0.1071],...
    'String',{TextBoxString1, TextBoxString2},...
    'FitBoxToText','off');
    
%     plot(LearnerObject.Modalities{1}.SOM.codebook(1:2,1), LearnerObject.Modalities{1}.SOM.codebook(1:2,2), 'g+');
%     plot(LearnerObject.Modalities{1}.SOM.codebook(3:4,1), LearnerObject.Modalities{1}.SOM.codebook(3:4,2), 'go');
                                   
%     for iEpochs = 1:10
%         LearnerObject = LearnerObject.train();
%         plot(LearnerObject.Modalities{1}.SOM.codebook(1:2,1), LearnerObject.Modalities{1}.SOM.codebook(1:2,2), 'g+');
%         plot(LearnerObject.Modalities{1}.SOM.codebook(3:4,1), LearnerObject.Modalities{1}.SOM.codebook(3:4,2), 'go');
%     end    
    
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