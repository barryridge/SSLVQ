classdef Mixture < handle
    % MIXTURE Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        
        %% ------- *** PROPERTIES *** -------------------------------------
        %******************************************************************
        %******************************************************************
        % Learner name...
        Name = [];
        
        % Training & test data...
        TrainingData1Epoch = [];
        TrainingData = [];
        TestData = [];        
        
        % Current timestep...
        t = 0;
        
        % Current sample...
        CurrentSample = [];
        
        % Has the learner been trained?
        is_trained = false;
        
        % Record statistics switch...
        record = false;
        
        % Display command-line output etc...
        verbose = true;
        
        % Display visualization of training...
        visualize = false;
        
        % Use mex functions?
        use_mex = false;
        
        % Feature selection switch...
        feature_selection = 'off';
        
        % Feature selection max features...
        feature_selection_max = [];
        
        % Feature selection feedback in training...
        feature_selection_feedback = false;
        
        % Default classification method...
        classification_method = 'node';
        
        % Default node colouring method...
        node_colouring_method = 'cluster_mean';
        
        % Store training data within object?
        store_training_data = true;
        
        % Store test data within object?
        store_test_data = true;
        
    end

    methods (Abstract)
        
        %% ------- *** SET PROPERTIES *** ---------------------------------
        %******************************************************************
        %******************************************************************
        obj = set(obj, varargin)
        
        %% ------- *** TRAIN *** ------------------------------------------
        %******************************************************************
        %******************************************************************
        obj = train(obj, varargin)
            
        %% ------- *** CLASSIFY *** ---------------------------------------
        %******************************************************************
        %******************************************************************
        obj = classify(obj, varargin)
            
        %% ------- *** EVALUATE *** ---------------------------------------
        %******************************************************************
        %******************************************************************
        obj = evaluate(obj, varargin)

        
    end        
    
    methods (Static = true)
        
        %% ------- *** SETUPDATASTRUCTS *** -------------------------------
        % *****************************************************************
        % *****************************************************************
        function [TrainingData TestData] = setupdatastructs(Data, varargin)
            
            % Defaults...
            TrainingData = [];
            TestData = [];
            TrainingIndices = [];
            TestIndices = [];
            epochs = 1;
            
            % Loop through arguments...
            i = 1;
            iPassedArgs = 1;
            while i <= length(varargin), 
                argok = 1; 
                if ischar(varargin{i}), 
                    switch lower(varargin{i}), 
                        case 'trainingindices', i=i+1; TrainingIndices = varargin{i};
                        case 'testindices', i=i+1; TestIndices = varargin{i};                        
                        case 'epochs', i=i+1; epochs = varargin{i};
                        otherwise
                            PassedArgs{iPassedArgs} = varargin{i};
                            PassedArgs{iPassedArgs+1} = varargin{i+1};
                            i = i + 1;
                            iPassedArgs = iPassedArgs + 2;
                    end
                else
                    argok = 0;
                end

                if ~argok, 
                    disp(['Learner.setupdatastructs(): Ignoring invalid argument #' num2str(i)]);
                    % fprintf(UsageMessage);
                end

                i = i + 1;
            end
            
            % Grab the ground truth labels --------------------------------
            %--------------------------------------------------------------            
            if isfield(Data,'GroundTruthClassIndices')
                nGroundTruths = size(Data.GroundTruthClassIndices,2);
                GroundTruthLabelIndices = Data.GroundTruthClassIndices;
            else
                nGroundTruths = 0;
                GroundTruthLabelIndices = [];
            end
            
            if isempty(TrainingIndices) && isempty(TestIndices)
                TrainingIndices = 1:size(Data.FeatureVectors,2);
            elseif isempty(TrainingIndices) && ~isempty(TestIndices) &&...
                   length(TestIndices) < size(Data.FeatureVectors,2)
                TrainingIndices = find(~ismember(1:size(Data.FeatureVectors,2), TestIndices));
            elseif isempty(TestIndices) && ~isempty(TrainingIndices) &&...
                   length(TrainingIndices) < size(Data.FeatureVectors,2)
                TestIndices = find(~ismember(1:size(Data.FeatureVectors,2), TrainingIndices));
            end

            % Get the field names from the Data struct...
            DataFieldNames = fieldnames(Data);

            % If training data vector indices were specified...
            if ~isempty(TrainingIndices)

                % Copy everything from Data...        
                for i = 1:length(DataFieldNames)
                    if ~strcmp(DataFieldNames{i}, 'Modalities')
                        TrainingData.(DataFieldNames{i}) =...
                            Data.(DataFieldNames{i});
                    end
                end

                % Change the feature vectors and category labels as necessary...
                if isfield(Data,'ClassLabels')
                    TrainingData.ClassLabels = Data.ClassLabels(:,TrainingIndices);
                end
                
                TrainingData.FeatureVectors = Data.FeatureVectors(:,TrainingIndices);
                
                if isfield(Data, 'NormedFeatureVectors')
                    TrainingData.NormedFeatureVectors =...
                        TrainingData.NormedFeatureVectors(:,TrainingIndices);
                end
                
                if isfield(Data, 'FileNames')                    
                    TrainingData.FileNames =...
                        TrainingData.FileNames(TrainingIndices);                    
                end
                
                % nGroundTruths = nGroundTruths;
                % GroundTruthLabelIndices = GroundTruthLabelIndices;
                
                % If modality feature indices were provided, use them
                % to set up modality data structs in the main data
                % struct...
                if isfield(Data, 'Modalities')                    
                    for i = 1:length(Data.Modalities)
                        if isfield(Data.Modalities{i}, 'FeatureIndices')
                            
                            if isfield(TrainingData,'FeatureNames')
                                TrainingData.Modalities{i}.FeatureNames = TrainingData.FeatureNames(:,Data.Modalities{i}.FeatureIndices);
                            end
                            
                            if isfield(TrainingData,'ClassNames')
                                TrainingData.Modalities{i}.ClassNames = TrainingData.ClassNames;
                            end
                            
                            TrainingData.Modalities{i}.FeatureVectors =...
                                TrainingData.FeatureVectors(Data.Modalities{i}.FeatureIndices, :);
                            
                            if isfield(TrainingData, 'NormedFeatureVectors')
                                TrainingData.Modalities{i}.NormedFeatureVectors =...
                                    TrainingData.NormedFeatureVectors(Data.Modalities{i}.FeatureIndices, :);
                            end
                            
                            if isfield(Data.Modalities{i}, 'Norm')
                                TrainingData.Modalities{i}.Norm = Data.Modalities{i}.Norm;
                            end
                            
                            if isfield(TrainingData,'ClassLabels')
                                TrainingData.Modalities{i}.ClassLabels = TrainingData.ClassLabels;
                            end
                            
                            TrainingData.Modalities{i}.nGroundTruths = nGroundTruths;
                            TrainingData.Modalities{i}.GroundTruthLabelIndices = GroundTruthLabelIndices;
                        else
                            TrainingData.Modalities{i} = Data.Modalities{i};
                            
                            if isfield(TrainingData, 'FeatureVectors')
                                TrainingData.Modalities{i}.FeatureVectors = [];
                                TrainingData.Modalities{i}.FeatureVectors =...
                                    Data.Modalities{i}.FeatureVectors(:,TrainingIndices);
                            end
                            
                            if isfield(TrainingData, 'NormedFeatureVectors')
                                TrainingData.Modalities{i}.NormedFeatureVectors = [];
                                TrainingData.Modalities{i}.NormedFeatureVectors =...
                                    Data.Modalities{i}.NormedFeatureVectors(:,TrainingIndices);
                            end
                            
                            if isfield(TrainingData, 'ClassLabels')
                                TrainingData.Modalities{i}.ClassLabels = [];
                                TrainingData.Modalities{i}.ClassLabels =...
                                    Data.Modalities{i}.ClassLabels(:,TrainingIndices);
                            end
                        end
                    end
                end
                
            else
                error('No training indices specified!');
            end

            % If test data vector indices were specified...
            if ~isempty(TestIndices)

                % Copy everything from Data...
                for i = 1:length(DataFieldNames)
                    if ~strcmp(DataFieldNames{i}, 'Modalities')
                        TestData.(DataFieldNames{i}) =...
                            Data.(DataFieldNames{i});
                    end
                end

                % Change the feature vectors and category labels as necessary...
                TestData.ClassLabels = Data.ClassLabels(:,TestIndices);                
                
                TestData.FeatureVectors = Data.FeatureVectors(:,TestIndices);
                
                if isfield(Data, 'NormedFeatureVectors')
                    TestData.NormedFeatureVectors =...
                        TestData.NormedFeatureVectors(:,TestIndices);
                end
                
                if isfield(Data, 'FileNames')                    
                    TestData.FileNames =...
                        TestData.FileNames(TestIndices);
                end
                
                % TestData.nGroundTruths = nGroundTruths;
                % TestData.GroundTruthLabelIndices = GroundTruthLabelIndices;
                
                % If modality feature indices were provided, use them
                % to set up modality data structs in the main data
                % struct...
                if isfield(Data, 'Modalities')
                    for i = 1:length(Data.Modalities)
                        TestData.Modalities{i}.FeatureNames = TestData.FeatureNames(:,Data.Modalities{i}.FeatureIndices);
                        TestData.Modalities{i}.ClassNames = TestData.ClassNames;
                        TestData.Modalities{i}.FeatureVectors =...
                            TestData.FeatureVectors(Data.Modalities{i}.FeatureIndices, :);
                        if isfield(TestData, 'NormedFeatureVectors')
                            TestData.Modalities{i}.NormedFeatureVectors =...
                                TestData.NormedFeatureVectors(Data.Modalities{i}.FeatureIndices, :);
                        end
                        if isfield(Data.Modalities{i}, 'Norm')
                            TestData.Modalities{i}.Norm = Data.Modalities{i}.Norm;
                        end
                        TestData.Modalities{i}.ClassLabels = TestData.ClassLabels;
                        TrainingData.Modalities{i}.nGroundTruths = nGroundTruths;
                        TrainingData.Modalities{i}.GroundTruthLabelIndices = GroundTruthLabelIndices;
                    end
                end
                
            end
            
            %% EXPAND TRAINING DATA STRUCT FOR MULTIPLE EPOCHS ------------
            % If the training data should be used for multiple training
            % epochs, multiply the training data struct appropriately...
            %--------------------------------------------------------------
            if epochs > 1
                TempTrainingData = TrainingData;
                
                for i = 2:epochs
                    
                    TrainingData.FeatureVectors =...
                        [TrainingData.FeatureVectors TempTrainingData.FeatureVectors];
                    TrainingData.ClassLabels =...
                        [TrainingData.ClassLabels TempTrainingData.ClassLabels];
                    
                    % If modality feature indices were provided, use them
                    % to set up modality data structs in the main data
                    % struct...
                    if isfield(Data, 'Modalities')
                        for iMod = 1:length(Data.Modalities)
                            TrainingData.Modalities{iMod}.FeatureVectors =...
                                [TrainingData.Modalities{iMod}.FeatureVectors TempTrainingData.Modalities{iMod}.FeatureVectors];
                            if isfield(TrainingData.Modalities{iMod}, 'NormedFeatureVectors')
                                TrainingData.Modalities{iMod}.NormedFeatureVectors =...
                                [TrainingData.Modalities{iMod}.NormedFeatureVectors TempTrainingData.Modalities{iMod}.NormedFeatureVectors];
                            end
                            if isfield(Data.Modalities{iMod}, 'Norm')
                                TrainingData.Modalities{iMod}.Norm = Data.Modalities{iMod}.Norm;
                            end
                            TrainingData.Modalities{iMod}.ClassLabels =...
                                [TrainingData.Modalities{iMod}.ClassLabels TempTrainingData.Modalities{iMod}.ClassLabels];
                        end
                    end
                                                            
                end
                
                clear TempTrainingData;
            end
            
            
        end
        
        %% ------- *** NORMALIZE *** --------------------------------------
        % *****************************************************************
        % *****************************************************************
        function [NormedData NormStruct] = normalize(Data, varargin)
            
            % Defaults...
            Method = 'range';
            
            % Loop through arguments...
            i = 1;
            while i <= length(varargin), 
                argok = 1; 
                if ischar(varargin{i}), 
                    switch lower(varargin{i}), 
                        case 'method', i=i+1; Method = varargin{i};
                    end
                else
                    argok = 0;
                end

                if ~argok, 
                    disp(['Learner.normalize(): Ignoring invalid argument #' num2str(i)]);
                    % fprintf(UsageMessage);
                end

                i = i + 1;
            end
            
            % Were we passed a data struct, or raw data?
            if isstruct(Data)
                TempData = Data.FeatureVectors;
            else
                TempData = Data;
            end
            
            % Normalize...
            switch Method                
                
                case {'1', '2', '3', '4'}
                    TempData = normalize(TempData, str2num(Method));
                    
                    % NormStruct = cell(size(obj.TrainingData1Epoch.Modalities{1}.FeatureVectors,1),1);                    
                    
                otherwise
                    TempData1 = som_data_struct(TempData');                    
                    TempData1 = som_normalize(TempData1, Method);
                    
                    TempData = TempData1.data';                    
                    NormStruct = TempData1.comp_norm;
            end
            
            % Were we passed a data struct, or raw data?
            if isstruct(Data)
                NormedData = Data;
                NormedData.NormedFeatureVectors = TempData;
                NormedData.Norm = NormStruct;
            else
                NormedData = TempData;
            end
            
            % Are there modalities?
            if isstruct(Data)
                if isfield(Data, 'Modalities')
                    
                    for iMod = 1:length(Data.Modalities)
                        
                        if isfield(Data.Modalities{iMod}, 'FeatureVectors')
                            
                            TempData = Data.Modalities{iMod}.FeatureVectors;

                            % Normalize...
                            switch Method                

                                case {'1', '2', '3', '4'}
                                    TempData = normalize(TempData, str2num(Method));

                                    NormStruct = cell(size(obj.TrainingData1Epoch.Modalities{1}.FeatureVectors,1),1);

                                otherwise
                                    TempData1 = som_data_struct(TempData');                    
                                    TempData1 = som_normalize(TempData1, Method);

                                    TempData = TempData1.data';                    
                                    NormStruct = TempData1.comp_norm;
                            end

                            Data.Modalities{iMod}.NormedFeatureVectors = TempData;
                            Data.Modalities{iMod}.Norm = NormStruct;
                            
                        else
                            
                            for iDim = 1:length(Data.Modalities{iMod}.FeatureIndices)
                                iFeature = Data.Modalities{iMod}.FeatureIndices(iDim);
                                NormedData.Modalities{iMod}.Norm{iDim,:} = NormedData.Norm{iFeature,:};
                            end
                            
                        end
                        
                    end                        
                        
                end
            end                       
            
        end
        
        %% ------- *** RANDOMIZE *** --------------------------------------
        % *****************************************************************
        % *****************************************************************
        function RandomizedData = randomize(Data, varargin)
            
            % Defaults...
            Method = 'samplewise';
            
            % Loop through arguments...
            i = 1;
            while i <= length(varargin), 
                argok = 1; 
                if ischar(varargin{i}), 
                    switch lower(varargin{i}), 
                        case 'method', i=i+1; Method = varargin{i};
                    end
                else
                    argok = 0;
                end

                if ~argok, 
                    disp(['Learner.normalize(): Ignoring invalid argument #' num2str(i)]);
                    % fprintf(UsageMessage);
                end

                i = i + 1;
            end
            
            % rand('twister',sum(size(Data.FeatureVectors,2)*clock));

            switch Method
                
                case 'samplewise',                    
                    RandIndices = randperm(size(Data.FeatureVectors,2));

                    RandomizedData = Data;
                    for i = 1:size(RandIndices,2)

                        RandomizedData.ClassLabels(:,i) = Data.ClassLabels(:,RandIndices(i));
                        RandomizedData.FeatureVectors(:,i) = Data.FeatureVectors(:,RandIndices(i));

                        % If the data struct also contains modality data structs,
                        % randomize them too using the same randomization...
                        if isfield(RandomizedData, 'Modalities')
                            for j = 1:length(RandomizedData.Modalities)

                                if isfield(RandomizedData.Modalities{j}, 'FeatureVectors')
                                    RandomizedData.Modalities{j}.FeatureVectors(:,i) =...
                                        Data.Modalities{j}.FeatureVectors(:,RandIndices(i));
                                end

                                if isfield(RandomizedData.Modalities{j}, 'NormedFeatureVectors')
                                    RandomizedData.Modalities{j}.NormedFeatureVectors(:,i) =...
                                        Data.Modalities{j}.NormedFeatureVectors(:,RandIndices(i));
                                end

                                if isfield(RandomizedData.Modalities{j}, 'ClassLabels')
                                    RandomizedData.Modalities{j}.ClassLabels(:,i) =...
                                        Data.Modalities{j}.ClassLabels(:,RandIndices(i));
                                end
                            end
                        end

                    end
                    
                case 'classwise',
                    
                    RandIndices = randperm(size(Data.AllowedTrainingClassIndices,2));
                    
                    RandClasses = Data.AllowedTrainingClassIndices(RandIndices);
                    
                    RandomizedData = Data;
                    iIn = 1;
                    for iClass = 1:size(RandClasses,2)
                        
                        ClassIndices = find(Data.ClassLabels(RandClasses(iClass),:));
                        
                        if ~isempty(ClassIndices)
                            inc = size(ClassIndices,2);

                            RandomizedData.ClassLabels(:,iIn:iIn+(inc-1)) = Data.ClassLabels(:,ClassIndices);
                            RandomizedData.FeatureVectors(:,iIn:iIn+(inc-1)) = Data.FeatureVectors(:,ClassIndices);

                            % If the data struct also contains modality data structs,
                            % randomize them too using the same randomization...
                            if isfield(RandomizedData, 'Modalities')
                                for j = 1:length(RandomizedData.Modalities)

                                    if isfield(RandomizedData.Modalities{j}, 'FeatureVectors')
                                        RandomizedData.Modalities{j}.FeatureVectors(:,iIn:iIn+(inc-1)) =...
                                            Data.Modalities{j}.FeatureVectors(:,ClassIndices);
                                    end

                                    if isfield(RandomizedData.Modalities{j}, 'NormedFeatureVectors')
                                        RandomizedData.Modalities{j}.NormedFeatureVectors(:,iIn:iIn+(inc-1)) =...
                                            Data.Modalities{j}.NormedFeatureVectors(:,ClassIndices);
                                    end

                                    if isfield(RandomizedData.Modalities{j}, 'ClassLabels')
                                        RandomizedData.Modalities{j}.ClassLabels(:,iIn:iIn+(inc-1)) =...
                                            Data.Modalities{j}.ClassLabels(:,ClassIndices);
                                    end
                                end
                            end

                            iIn = iIn+(inc-1);
                        end
                        
                    end
            end
            
        end
        
    end
    
    methods
        
        %% ------- *** COPY *** -------------------------------------------
        % *****************************************************************
        % Make a copy of a handle object.
        % *****************************************************************
        function new = copy(this)
            % From the file exchange (doesn't fully work):
            
            % % Instantiate new object of the same class.
            % new = feval(class(this));
            % 
            % % Copy all non-hidden properties.
            % p = fieldnames(struct(this));
            % for i = 1:length(p)
            %     new.(p{i}) = this.(p{i});
            % end
            
            % My way...
            Filename = 'temp';
            iFilenum = 1;
            while exist([Filename iFilenum '.mat']) >= 2
                iFilenum = iFilenum + 1;
            end
            save([Filename iFilenum '.mat'], 'this');
            Foo = load([Filename iFilenum '.mat']);
            new = Foo.this;
            delete([Filename iFilenum '.mat']);
        end
        
    end
    
end