classdef BiModalLearner < Learner
    % BIMODALLEARNER Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        
        %% ------- *** OBJECTS *** ----------------------------------------
        %******************************************************************
        %******************************************************************
        % Cell array of modality objects...
        Modalities = [];
        
        % Mapping between modalities...
        crossMapping = [];
        
        %% ------- *** FUNCTION HANDLES *** -------------------------------
        %******************************************************************
        %******************************************************************
        % Handle to a method that computes an auxiliary distance...
        computeauxdist = [];
        
        output_feature_selection = [];       
        output_feature_selection_params = [];
        
        InputClassificationMask = [];
        OutputClassificationMask = [];
                
    end
    
    properties (SetAccess = private, Hidden = true)
        
        % Default settings...
        
        % Usage message...
        UsageMessage = ['\nBiModalLearner Usage: '...
                        'Please see the function comments for more detail.\n\n'];
                
    end
    
    methods

        %% ------- *** CONSTRUCTOR *** ------------------------------------
        %******************************************************************
        %******************************************************************
        function obj = BiModalLearner(varargin)
                          
            % Default settings...
            Data = [];
            InitData = [];
            TrainingData1Epoch = [];
            TrainingData = [];
            TestData = [];
            ModalityTypes = {'codebook', 'codebook'};
            CodebookSizes = {[10 10], [10 10]};
            CodebookNeighs = {'bubble', 'bubble'};
            CodebookLattices = {'hexa', 'hexa'};
            CodebookShapes = {'sheet', 'sheet'};
            CodebookInitMethods = {'rand', 'rand'};
            TrainingIndices = [];
            TestIndices =[];
            epochs = 1;
            normalization = 'all';
            normalization_method = 'range';           
            Updaters = {{'SOM', 'HeurORLVQ'}, {'SOM'}};
            randomize_training_data = false;
            randomize_test_data = false;
            
            
            % Loop through arguments...
            i = 1;
            iPassedArgs = 1;
            while i <= length(varargin), 
                argok = 1; 
                if ischar(varargin{i}), 
                    switch lower(varargin{i}),
                        case 'name', i=i+1; obj.Name = varargin{i};
                        case 'data', i=i+1; Data = varargin{i};
                        case 'initdata', i=i+1; InitData = varargin{i};
                        case 'trainingdata1epoch', i=i+1; TrainingData1Epoch = varargin{i};
                        case 'trainingdata', i=i+1; TrainingData = varargin{i};
                        case 'testdata', i=i+1; TestData = varargin{i};
                        case 'modality_types', i=i+1; ModalityTypes = varargin{i};
                        case 'mapping_type', i=i+1; MappingType = varargin{i};
                        case 'codebook_sizes', i=i+1; CodebookSizes = varargin{i};
                        case 'codebook_neighs', i=i+1; CodebookNeighs = varargin{i};
                        case 'codebook_lattices', i=i+1; CodebookLattices = varargin{i};
                        case 'codebook_shapes', i=i+1; CodebookShapes = varargin{i};
                        case 'codebook_init_method', i=i+1; CodebookInitMethods = varargin{i};
                        case 'trainingindices', i=i+1; TrainingIndices = varargin{i};
                        case 'testindices', i=i+1; TestIndices = varargin{i};
                        case {'store_training_data', 'store_data'},...
                                i=i+1; obj.store_training_data = varargin{i};
                        case {'store_test_data', 'store_data'},...
                                i=i+1; obj.store_test_data = varargin{i};
                        case {'randomize_train', 'randomize_training_data',...
                          'randomizetrain', 'randomizetrainingdata'},...
                            i=i+1; obj.randomize_training_data = varargin{i};
                        case {'randomize_test', 'randomize_test_data',...
                              'randomizetest', 'randomizetestdata'},...
                            i=i+1; obj.randomize_test_data = varargin{i};
                        case 'epochs', i=i+1; epochs = varargin{i};
                        case 'normalization', i=i+1; normalization = varargin{i};
                        case 'normalization_method', i=i+1; normalization_method = varargin{i};
                        case 'updaters',
                            Updaters = varargin{i+1};
                            PassedArgs{iPassedArgs} = varargin{i};
                            PassedArgs{iPassedArgs+1} = varargin{i+1};
                            i = i + 1;
                            iPassedArgs = iPassedArgs + 2;
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
                    disp(['BiModalLearner.BiModalLearner(): Ignoring invalid argument #' num2str(i)]);
                    % fprintf(obj.UsageMessage);
                end

                i = i + 1;
            end
            
            %% SET UP TRAINING AND/OR TEST DATA STRUCTS FOR 1 EPOCH -------
            %--------------------------------------------------------------
            if isempty(Data) && isempty(InitData) && (isempty(TrainingData1Epoch) || isempty(TrainingData))
                % error(obj.UsageMessage);
                    return;
                    
            elseif ~isempty(Data)
                
                if isempty(TrainingIndices)
                    TrainingIndices = 1:size(Data.FeatureVectors,2);
                    TestIndices = [];
                end
                
                [TrainingData1Epoch TestData] =...
                    obj.setupdatastructs(Data,...
                                         'TrainingIndices', TrainingIndices,...
                                         'TestIndices', TestIndices,...
                                         'epochs', 1);
                                     
            elseif ~isempty(InitData)
                
                TrainingData1Epoch = InitData;
                TestData = InitData;
                
            end                        
            
            %% NORMALIZE DATA FOR 1 EPOCH ---------------------------------
            %--------------------------------------------------------------
            if ~isfield(TrainingData1Epoch, 'NormedFeatureVectors') || ~isfield(TestData, 'NormedFeatureVectors')
                switch normalization                
                    case 'all'
                        [TempData InModalityNorm] = obj.normalize([TrainingData1Epoch.Modalities{1}.FeatureVectors...
                                                                   TestData.Modalities{1}.FeatureVectors],...
                                                                  normalization_method);

                        % Record the important stuff...
                        TrainingData1Epoch.Modalities{1}.NormedFeatureVectors =...
                            TempData(:,1:size(TrainingData1Epoch.Modalities{1}.FeatureVectors,2));
                        TrainingData1Epoch.Modalities{1}.Norm = InModalityNorm;
                        TestData.Modalities{1}.NormedFeatureVectors =...
                            TempData(:,size(TrainingData1Epoch.Modalities{1}.FeatureVectors,2)+1:end);
                        TestData.Modalities{1}.Norm = InModalityNorm;

                        [TempData OutModalityNorm] = obj.normalize([TrainingData1Epoch.Modalities{2}.FeatureVectors...
                                                                   TestData.Modalities{2}.FeatureVectors],...
                                                                  normalization_method);

                        % Record the important stuff...
                        TrainingData1Epoch.Modalities{2}.NormedFeatureVectors =...
                            TempData(:,1:size(TrainingData1Epoch.Modalities{2}.FeatureVectors,2));
                        TrainingData1Epoch.Modalities{2}.Norm = OutModalityNorm;
                        TestData.Modalities{2}.NormedFeatureVectors =...
                            TempData(:,size(TrainingData1Epoch.Modalities{2}.FeatureVectors,2)+1:end);
                        TestData.Modalities{2}.Norm = OutModalityNorm;

                    case {'train', 'training', 'training_data'}
                        [TempData InModalityNorm] = obj.normalize(TrainingData1Epoch.Modalities{1}.FeatureVectors,...
                                                                  normalization_method);

                        % Record the important stuff...
                        TrainingData1Epoch.Modalities{1}.NormedFeatureVectors =...
                            TempData(:,1:size(TrainingData1Epoch.Modalities{1}.FeatureVectors,2));
                        TrainingData1Epoch.Modalities{1}.Norm = InModalityNorm;

                        [TempData OutModalityNorm] = obj.normalize(TrainingData1Epoch.Modalities{2}.FeatureVectors,...
                                                                  normalization_method);

                        % Record the important stuff...
                        TrainingData1Epoch.Modalities{2}.NormedFeatureVectors =...
                            TempData(:,1:size(TrainingData1Epoch.Modalities{2}.FeatureVectors,2));
                        TrainingData1Epoch.Modalities{2}.Norm = OutModalityNorm;

                    otherwise
                        TrainingData1Epoch.Modalities{1}.NormedFeatureVectors =...
                            TrainingData1Epoch.Modalities{1}.FeatureVectors;

                        if ~isempty(TestData)
                            TestData.Modalities{1}.NormedFeatureVectors =...
                                TestData.Modalities{1}.FeatureVectors;
                        end

                        % InModalityNorm = cell(size(TrainingData1Epoch.Modalities{1}.FeatureVectors,1),1);
                        TrainingData1Epoch.Modalities{1}.Norm = cell(size(TrainingData1Epoch.Modalities{1}.FeatureVectors,1),1);

                        TrainingData1Epoch.Modalities{2}.NormedFeatureVectors =...
                            TrainingData1Epoch.Modalities{2}.FeatureVectors;


                        if ~isempty(TestData)
                            TestData.Modalities{2}.NormedFeatureVectors =...
                                TestData.Modalities{2}.FeatureVectors;
                        end

                        % OutModalityNorm = cell(size(TrainingData1Epoch.Modalities{2}.FeatureVectors,1),1);
                        TrainingData1Epoch.Modalities{2}.Norm = cell(size(TrainingData1Epoch.Modalities{2}.FeatureVectors,1),1);
                end
            end
            
%             %% SET UP TRAINING AND/OR TEST DATA STRUCTS -------------------
%             %--------------------------------------------------------------
%             if isempty(Data) && (isempty(obj.TrainingData) || isempty(obj.TestData))
%                 error(obj.UsageMessage);
%                     return;
%                     
%             elseif isempty(obj.TrainingData) || isempty(obj.TestData)
%                 [obj.TrainingData obj.TestData] =...
%                     obj.setupdatastructs(Data,...
%                                          'TrainingIndices', TrainingIndices,...
%                                          'TestIndices', TestIndices,...
%                                          'epochs', epochs);
%             end
%             
%             %% NORMALIZE TRAINING DATA ------------------------------------
%             %--------------------------------------------------------------
%             % Temporary data structs...
%             SOMDataTempIn = som_data_struct(obj.TrainingData.Modalities{1}.FeatureVectors',...
%                                           'comp_names', obj.TrainingData.Modalities{1}.FeatureNames');
%             SOMDataTempOut = som_data_struct(obj.TrainingData.Modalities{2}.FeatureVectors',...
%                                           'comp_names', obj.TrainingData.Modalities{2}.FeatureNames');
% 
%             % Normalize...
%             SOMDataTempIn = som_normalize(SOMDataTempIn, 'range');
%             SOMDataTempOut = som_normalize(SOMDataTempOut, 'range');
%             
%             % Record the important stuff...
%             obj.TrainingData.Modalities{1}.NormedFeatureVectors = SOMDataTempIn.data';
%             obj.TrainingData.Modalities{2}.NormedFeatureVectors = SOMDataTempOut.data';
%             InModalityNorm = SOMDataTempIn.comp_norm;
%             OutModalityNorm = SOMDataTempOut.comp_norm;
            
            %% INSTANTIATE MODALITY OBJECTS -------------------------------
            % Instantiate this BiModalLearner object's 2 modality objects...
            %--------------------------------------------------------------
            
            Mod1Updaters = Updaters{1};
            switch lower(Mod1Updaters{1})
                case 'somn'
                    covariance_init_flag = true;
                otherwise
                    covariance_init_flag = false;
            end
            
            obj.Modalities{1} =...
                ModalityFactory.createModality(TrainingData1Epoch.Modalities{1},...
                                               'modality_type', ModalityTypes{1},...
                                               'codebook_size', CodebookSizes{1},...
                                               'codebook_shape', CodebookShapes{1},...
                                               'codebook_lattice', CodebookLattices{1},...
                                               'codebook_neigh', CodebookNeighs{1},...
                                               'codebook_init_method', CodebookInitMethods{1},...
                                               'normalization_struct', TrainingData1Epoch.Modalities{1}.Norm,...
                                               'normalization_method', normalization_method,...
                                               'covariance_init', covariance_init_flag);

            Mod2Updaters = Updaters{2};
            switch lower(Mod2Updaters{1})
                case 'somn'
                    covariance_init_flag = true;
                otherwise
                    covariance_init_flag = false;
            end                                           
                                           
            obj.Modalities{2} =...
                ModalityFactory.createModality(TrainingData1Epoch.Modalities{2},...
                                               'modality_type', ModalityTypes{2},...
                                               'codebook_size', CodebookSizes{2},...
                                               'codebook_shape', CodebookShapes{2},...
                                               'codebook_lattice', CodebookLattices{2},...
                                               'codebook_neigh', CodebookNeighs{2},...
                                               'codebook_init_method', CodebookInitMethods{2},...
                                               'normalization_struct', TrainingData1Epoch.Modalities{2}.Norm,...
                                               'normalization_method', normalization_method,...
                                               'covariance_init', covariance_init_flag);                                                       
                                           
            %% SETUP TRAINING DATA FOR MULTIPLE EPOCHS --------------------
            %
            % NOTE: This is a really stupid way of doing things, but I
            % don't have time to fix it right now.
            %--------------------------------------------------------------
            if obj.store_training_data || obj.store_test_data
                
                if isempty(Data) && isempty(InitData) && isempty(TrainingData)
                    % error(obj.UsageMessage);
                        return;

                elseif ~isempty(Data)
                    [TrainingData TestData] =...
                        obj.setupdatastructs(Data,...
                                             'TrainingIndices', TrainingIndices,...
                                             'TestIndices', TestIndices,...
                                             'epochs', epochs);

                    if ~isempty(TrainingIndices)
                        [TrainingData TestData] =...
                            obj.setupdatastructs(Data,...
                                                 'TrainingIndices', TrainingIndices,...
                                                 'epochs', epochs);
                    elseif ~isempty(TestIndices)
                        [TrainingData TestData] =...
                            obj.setupdatastructs(Data,...
                                                 'TestIndices', TestIndices,...
                                                 'epochs', epochs);
                    end
                    
                elseif ~isempty(InitData)
                    TrainingData = InitData;
                    TestData = InitData;
                end            
            
                %% NORMALIZE DATA ---------------------------------------------
                %--------------------------------------------------------------
                if ~isfield(TrainingData, 'NormedFeatureVectors')
                    switch normalization                
                        case 'all'
                            [TempData InModalityNorm] = obj.normalize([TrainingData.Modalities{1}.FeatureVectors...
                                                                       TestData.Modalities{1}.FeatureVectors],...
                                                                      normalization_method);

                            % Record the important stuff...
                            TrainingData.Modalities{1}.NormedFeatureVectors =...
                                TempData(:,1:size(TrainingData.Modalities{1}.FeatureVectors,2));
                            TrainingData.Modalities{1}.Norm = InModalityNorm;
                            TestData.Modalities{1}.NormedFeatureVectors =...
                                TempData(:,size(TrainingData.Modalities{1}.FeatureVectors,2)+1:end);
                            TestData.Modalities{1}.Norm = InModalityNorm;

                            [TempData OutModalityNorm] = obj.normalize([TrainingData.Modalities{2}.FeatureVectors...
                                                                       TestData.Modalities{2}.FeatureVectors],...
                                                                      normalization_method);

                            % Record the important stuff...
                            TrainingData.Modalities{2}.NormedFeatureVectors =...
                                TempData(:,1:size(TrainingData.Modalities{2}.FeatureVectors,2));
                            TrainingData.Modalities{2}.Norm = OutModalityNorm;
                            TestData.Modalities{2}.NormedFeatureVectors =...
                                TempData(:,size(TrainingData.Modalities{2}.FeatureVectors,2)+1:end);
                            TestData.Modalities{2}.Norm = OutModalityNorm;

                        case {'train', 'training', 'training_data'}
                            [TempData InModalityNorm] = obj.normalize(TrainingData.Modalities{1}.FeatureVectors,...
                                                                      normalization_method);

                            % Record the important stuff...
                            TrainingData.Modalities{1}.NormedFeatureVectors =...
                                TempData(:,1:size(TrainingData.Modalities{1}.FeatureVectors,2));
                            TrainingData.Modalities{1}.Norm = InModalityNorm;

                            [TempData OutModalityNorm] = obj.normalize(TrainingData.Modalities{2}.FeatureVectors,...
                                                                      normalization_method);

                            % Record the important stuff...
                            TrainingData.Modalities{2}.NormedFeatureVectors =...
                                TempData(:,1:size(TrainingData.Modalities{2}.FeatureVectors,2));
                            TrainingData.Modalities{2}.Norm = OutModalityNorm;

                        otherwise
                            TrainingData.Modalities{1}.NormedFeatureVectors =...
                                TrainingData.Modalities{1}.FeatureVectors;

                            if ~isempty(TestData)
                                TestData.Modalities{1}.NormedFeatureVectors =...
                                    TestData.Modalities{1}.FeatureVectors;
                            end

                            % InModalityNorm = cell(size(TrainingData.Modalities{1}.FeatureVectors,1),1);
                            TrainingData.Modalities{1}.Norm = cell(size(TrainingData1Epoch.Modalities{1}.FeatureVectors,1),1);

                            TrainingData.Modalities{2}.NormedFeatureVectors =...
                                TrainingData.Modalities{2}.FeatureVectors;

                            if ~isempty(TestData)
                                TestData.Modalities{2}.NormedFeatureVectors =...
                                    TestData.Modalities{2}.FeatureVectors;
                            end

                            % OutModalityNorm = cell(size(TrainingData.Modalities{2}.FeatureVectors,1),1);
                            TrainingData.Modalities{2}.Norm = cell(size(TrainingData1Epoch.Modalities{2}.FeatureVectors,1),1);
                    end
                end
            end
            
            %% Set up CurrentSample data struct ------------------
            %--------------------------------------------------------------
            if isfield(TrainingData.Modalities{1},'FeatureNames')
                obj.CurrentSample.Modalities{1}.FeatureNames = TrainingData.Modalities{1}.FeatureNames;
            end
            
            if isfield(TrainingData.Modalities{1},'ClassNames')
                obj.CurrentSample.Modalities{1}.ClassNames = TrainingData.Modalities{1}.ClassNames;
            end
            
            if isfield(TrainingData.Modalities{1},'nGroundTruths')
                obj.CurrentSample.Modalities{1}.nGroundTruths = TrainingData.Modalities{1}.nGroundTruths;
            end
            
            if isfield(TrainingData.Modalities{1},'GroundTruthLabelIndices')
                obj.CurrentSample.Modalities{1}.GroundTruthLabelIndices = TrainingData.Modalities{1}.GroundTruthLabelIndices;
            end
            
            if isfield(TrainingData.Modalities{2},'FeatureNames')
                obj.CurrentSample.Modalities{2}.FeatureNames = TrainingData.Modalities{2}.FeatureNames;
            end
            
            if isfield(TrainingData.Modalities{2},'ClassNames')
                obj.CurrentSample.Modalities{2}.ClassNames = TrainingData.Modalities{2}.ClassNames;
            end
            
            if isfield(TrainingData.Modalities{2},'nGroundTruths')
                obj.CurrentSample.Modalities{2}.nGroundTruths = TrainingData.Modalities{2}.nGroundTruths;
            end
            
            if isfield(TrainingData.Modalities{2},'nGroundTruths')
                obj.CurrentSample.Modalities{2}.GroundTruthLabelIndices = TrainingData.Modalities{2}.GroundTruthLabelIndices;                        
            end
            
            %% RANDOMIZE DATA SETS ----------------------------------------
            % Randomize the data if required...
            %--------------------------------------------------------------            
            switch obj.randomize_training_data
                case 'samplewise',
                    TrainingData = Learner.randomize(TrainingData, 'method', 'samplewise');
                    TrainingData.Randomized = true;
                case 'classwise',
                    TrainingData = Learner.randomize(TrainingData, 'method', 'classwise');
                    TrainingData.Randomized = true;
            end
            
            switch obj.randomize_test_data
                case 'samplewise',
                    TestData = Learner.randomize(TestData, 'method', 'samplewise');
                    TestData.Randomized = true;
                case 'classwise',
                    TestData = Learner.randomize(TestData, 'method', 'classwise');
                    TestData.Randomized = true;
            end                        

            %% STORE TRAINING AND TEST DATA? ---------------------
            %--------------------------------------------------------------
            if obj.store_training_data
                obj.TrainingData1Epoch = TrainingData1Epoch;
                obj.TrainingData = TrainingData;
            end
            
            if obj.store_test_data
                obj.TestData = TestData;
            end
            
            %% INSTANTIATE MAPPING OBJECT ---------------------------------
            % Instantiate this CrossMod co-occurence mapping object...
            %--------------------------------------------------------------
            obj.crossMapping = MappingFactory.createMapping('mapping_type', MappingType,...
                                                            obj.Modalities{1},...
                                                            obj.Modalities{2});
                                                                    
            %% CREATE DEFAULT computeauxdist HANDLE -----------------------
            %--------------------------------------------------------------
            obj.computeauxdist = @Utils.hellinger;
            
            %% PASS REMAINING ARGUMENTS TO SET METHOD ---------------------
            %--------------------------------------------------------------
            obj = obj.set(PassedArgs{1:end},...
                          'trainlen', size(TrainingData.FeatureVectors,2),...
                          'nClasses', TrainingData.Modalities{1}.nGroundTruths);
            
        end
        
        %% ------- *** SET PROPERTIES *** ---------------------------------
        %******************************************************************
        %******************************************************************
        function obj = set(obj, varargin)
        
            %% CHECK ARGUMENTS --------------------------------------------
            %--------------------------------------------------------------
            % default values
            epochs_value = 1;            
            Updaters = {{'SOM', 'HeurORLVQ'}, {'SOM'}};
            PhaseShifts = {{0.5}, {NaN}};   
            AlphaTypes = {{'linear', 'constant'}, {'inv'}};
            AlphaInits = {{1, 0.3}, {1}};
            RadiusTypes = {{'linear', 'linear'}, {'linear'}};
            RadiusInits = {{1, 1}, {5}};
            RadiusFins = {{1, 1}, {1}};
            WindowSizes = {NaN 0.3};
            AlphaFeatureTypes = {{NaN, 'constant'}, {NaN}};
            AlphaFeatureInits = {{NaN, 0.1}, {NaN}};
            ActivationTypes = {{'response', 'response'}, {'response'}};
            Metrics = {{'euclidean'}, {'euclidean'}};
            auxdist_type_value = 'hellinger';
                       
            
            % varargin
            i=1; 
            while i<=length(varargin), 
              argok = 1; 
              if ischar(varargin{i}), 
                switch varargin{i}, 
                    % argument IDs                    
                    case {'updaters'},  i=i+1; Updaters = varargin{i};
                    case {'phaseshifts', 'phase_shifts'},  i=i+1; PhaseShifts = varargin{i};   
                    case {'alphatypes', 'alpha_types'},  i=i+1; AlphaTypes = varargin{i};
                    case {'alphainits', 'alpha_inits'},  i=i+1; AlphaInits = varargin{i};
                    case {'radiustypes', 'radius_types'},  i=i+1; RadiusTypes = varargin{i};
                    case {'radiusinits', 'radius_inits'},  i=i+1; RadiusInits = varargin{i};
                    case {'radiusfins', 'radius_fins'},  i=i+1; RadiusFins = varargin{i};
                    case {'windowsizes', 'window_sizes'}, i=i+1; WindowSizes = varargin{i};
                    case {'alphafeaturetypes', 'alpha_feature_types'}, i=i+1; AlphaFeatureTypes = varargin{i};
                    case {'alphafeatureinits', 'alpha_feature_inits'},  i=i+1; AlphaFeatureInits = varargin{i};
                    case {'activationtypes', 'activation_types'}, i=i+1; ActivationTypes = varargin{i};
                    case {'trainlen', 'train_len'}, i=i+1; trainlen = varargin{i};
                    case {'nClasses', 'numclasses'}, i=i+1; nClasses = varargin{i};
                    case {'auxdist', 'auxdist_type'}, i=i+1; auxdist_type_value = varargin{i};
                    case {'featureselection', 'feature_selection'}, i=i+1; obj.feature_selection = varargin{i};
                    case {'featureselectionmax', 'feature_selection_max'}, i=i+1; obj.feature_selection_max = varargin{i};
                    case {'featureselectionfeedback', 'feature_selection_feedback',...
                          'featureselectionintraining', 'feature_selection_in_training'},...
                            i=i+1; obj.feature_selection_feedback = varargin{i};
                    case {'outputfeatureselection', 'output_feature_selection'},...
                        i=i+1; obj.output_feature_selection = varargin{i};
                    case {'outputfeatureselectionparam', 'output_feature_selection_param',...
                          'outputfeatureselectionparams', 'output_feature_selection_params'},...
                        i=i+1; obj.output_feature_selection_params = varargin{i};
                    case {'classificationmethod', 'classification_method'},...
                            i=i+1; obj.classification_method = varargin{i};
                    case {'nodecolouringmethod', 'node_colouring_method'},...
                            i=i+1; obj.node_colouring_method = varargin{i};
                    case {'metric'}, i=i+1; Metrics = varargin{i};
                    case {'record'}, i=i+1; obj.record = varargin{i};
                    case {'verbose'}, i=i+1; obj.verbose = varargin{i};
                    case {'visualize', 'visualization'}, i=i+1; obj.visualize = varargin{i};
                    case {'usemex', 'use_mex'}, i=i+1; obj.use_mex = varargin{i};
                    % Ignore codebook related arguments.  They shouldn't be set
                    % independently of the constructor.
                    case {'codebook_sizes', 'codebook_lattices',...
                          'codebook_shapes', 'codebook_neighs'}, i=i+1;
                      
                 	otherwise, argok=0;
                end
              else
                argok = 0;
              end
              if ~argok, 
                disp(['BiModalLearner.set(): Ignoring invalid argument #' num2str(i+1)]); 
              end
              i = i+1;
            end
            
            %% CREATE computeauxdist HANDLE -------------------------------
            %--------------------------------------------------------------
            switch auxdist_type_value,
                case {'hellinger'},
                    obj.computeauxdist = @Utils.hellinger;
                case {'normed_hellinger', 'hellinger_normed'},
                    obj.computeauxdist = @Utils.normed_hellinger;
                case {'chi squared', 'chisquared', 'chi_squared', 'chi-squared'},
                    obj.computeauxdist = @Utils.chisquared;
                case {'kullback leibler', 'kullbackleibler', 'kullback_leibler', 'kullback-leibler',...
                      'cross entropy', 'crossentropy', 'cross-entropy', 'cross_entropy'},
                    obj.computeauxdist = @Utils.kullbackleibler;
                case {'totalvariation', 'totalvariation', 'total_variation', 'total-variation'},
                    obj.computeauxdist = @Utils.totalvariation;
                case {'cross correlation', 'crosscorrelation', 'cross_correlation', 'cross-correlation'},
                    obj.computeauxdist = @Utils.crosscorrelation;
                case {'ridge'},
                    obj.computeauxdist = @Utils.ridge;
                case {'earth movers', 'earthmovers', 'earth_movers', 'earth-movers'},
                    obj.computeauxdist = @Utils.earthmovers;
                    
                otherwise
                    obj.computeauxdist = @Utils.hellinger;
            end                        

            %% SETUP ------------------------------------------------------
            %--------------------------------------------------------------
            % Set IN modality SOM properties...
            obj.Modalities{1} = obj.Modalities{1}.set('metric', Metrics{1},...
                                                      'updaters', Updaters{1},...
                                                      'phase_shifts', PhaseShifts{1},...
                                                      'alpha_types', AlphaTypes{1},...
                                                      'alpha_inits', AlphaInits{1},...
                                                      'radius_types', RadiusTypes{1},...
                                                      'radius_inits', RadiusInits{1},...
                                                      'radius_fins', RadiusFins{1},...
                                                      'window_sizes', WindowSizes{1},...
                                                      'alpha_feature_types', AlphaFeatureTypes{1},...
                                                      'alpha_feature_inits', AlphaFeatureInits{1},...
                                                      'activation_types', ActivationTypes{1},...
                                                      'trainlen', trainlen,...
                                                      'nClasses', nClasses,...
                                                      'feature_selection_feedback', obj.feature_selection_feedback,...
                                                      'record', obj.record,...
                                                      'verbose', obj.verbose,...
                                                      'use_mex', obj.use_mex);
                                            
            
            % Decide how many auxiliary distances are to be calculated
            % depending on the algorithm...
            for i = 1:length(Updaters{1})
                switch lower(Updaters{1}{i})
                        case {'heurlvq3', 'heurmamrlvq', 'heurfmamrlvq'}
                            obj.Modalities{1}.nAuxDists = 2;
                end
            end
            
            % Set OUT modality SOM properties...
            obj.Modalities{2} = obj.Modalities{2}.set('metric', Metrics{2},...
                                                      'updaters', Updaters{2},...
                                                      'phase_shifts', PhaseShifts{2},...
                                                      'alpha_types', AlphaTypes{2},...
                                                      'alpha_inits', AlphaInits{2},...
                                                      'radius_types', RadiusTypes{2},...
                                                      'radius_inits', RadiusInits{2},...
                                                      'radius_fins', RadiusFins{2},...
                                                      'window_sizes', WindowSizes{2},...
                                                      'alpha_feature_types', AlphaFeatureTypes{2},...
                                                      'alpha_feature_inits', AlphaFeatureInits{2},...
                                                      'activation_types', ActivationTypes{2},...
                                                      'trainlen', trainlen,...
                                                      'record', obj.record,...
                                                      'verbose', obj.verbose,...
                                                      'use_mex', obj.use_mex);
                                                                              
            % Set cross-modal mapping properties...
            obj.crossMapping = obj.crossMapping.set('trainlen', trainlen);           
            
        end
        
        %% ------- *** CLEAR CLASS LABELS *** -----------------------------
        %******************************************************************
        %******************************************************************
        function obj = clearclasslabels(obj, varargin)
            
            for iMod = 1:length(obj.Modalities)
                obj.Modalities{iMod}.ClassLabels = [];
            end
            
        end
        
        %% ------- *** TRAIN *** ------------------------------------------
        %******************************************************************
        %******************************************************************
        function obj = train(obj, varargin)
            
            % Set defaults...
            TrainingData = [];
            sample_size = [];
            
            % Loop through arguments...
            i = 1;
            while i <= length(varargin), 
                argok = 1;
                if isnumeric(varargin{i})
                    sample_size = varargin{i};
                    
                elseif isstruct(varargin{i})
                    TrainingData = varargin{i};
                    sample_size = size(TrainingData.FeatureVectors,2);
                    
                elseif ischar(varargin{i}), 
                    % switch lower(varargin{i}), 
                    %     case 'som_size', i=i+1; obj.SOM_size = varargin{i};
                    % 
                    %     otherwise
                    %         argok = 0;
                    % end
                else
                    argok = 0;
                end

                if ~argok, 
                    disp(['BiModalLearner.train(): Ignoring invalid argument #' num2str(i)]);
                    fprintf(obj.UsageMessage);
                end

                i = i + 1;
            end
            
            if isempty(TrainingData)
                TrainingData = obj.TrainingData;                
            end
            
            if isempty(sample_size)
                sample_size = size(TrainingData.FeatureVectors,2);
            end
            
            %% SET UP FIGURE FOR VISUALIZATION ------------------------
            %----------------------------------------------------------
            if obj.visualize
                CurrentFig = figure;
                hold on;
            end
            
            %% TRAINING LOOP ----------------------------------------------
            %--------------------------------------------------------------
            if obj.verbose                
                fprintf(1, '\nTotal training samples:   %d', size(TrainingData.FeatureVectors,2));
                fprintf(1, '\nCurrent training sample:     ');
            end
            
            for t = obj.t+1 : min(obj.t + sample_size, size(TrainingData.FeatureVectors,2))
            % for t = 1:size(obj.TrainingData.FeatureVectors,2)
                
                obj.t = t;
                
                % Print progress...
                if obj.verbose
                    Backspaces = [];
                    for iBack = 1:ceil(log10(t+1))
                        Backspaces = [Backspaces '\b'];
                    end
                    fprintf(1, [Backspaces '%d'], obj.t);
                end
                
                
                %% GRAB CURRENT SAMPLE ------------------------------------
                %----------------------------------------------------------
                obj.CurrentSample.Modalities{1}.FeatureVectors(:,1) =...
                    TrainingData.Modalities{1}.FeatureVectors(:,t);
                if isfield(TrainingData.Modalities{1}, 'NormedFeatureVectors')
                    obj.CurrentSample.Modalities{1}.NormedFeatureVectors(:,1) =...
                        TrainingData.Modalities{1}.NormedFeatureVectors(:,t);
                else
                    obj.CurrentSample.Modalities{1}.NormedFeatureVectors(:,1) =...
                        TrainingData.Modalities{1}.FeatureVectors(:,t);
                end
                
                if isfield(TrainingData.Modalities{1}, 'ClassLabels')
                    obj.CurrentSample.Modalities{1}.ClassLabels(:,1) =...
                        TrainingData.Modalities{1}.ClassLabels(:,t);
                end
                
                if isfield(TrainingData.Modalities{1}, 'GroundTruthLabelIndices')
                    obj.CurrentSample.Modalities{1}.GroundTruthLabelIndices =...
                        TrainingData.Modalities{1}.GroundTruthLabelIndices;
                end
                
                obj.CurrentSample.Modalities{2}.FeatureVectors(:,1) =...
                    TrainingData.Modalities{2}.FeatureVectors(:,t);
                if isfield(TrainingData.Modalities{2}, 'NormedFeatureVectors')
                    obj.CurrentSample.Modalities{2}.NormedFeatureVectors(:,1) =...
                        TrainingData.Modalities{2}.NormedFeatureVectors(:,t);
                else
                    obj.CurrentSample.Modalities{2}.NormedFeatureVectors(:,1) =...
                        TrainingData.Modalities{2}.FeatureVectors(:,t);
                end
                
                if isfield(TrainingData.Modalities{2}, 'ClassLabels')
                    obj.CurrentSample.Modalities{2}.ClassLabels(:,1) =...
                        TrainingData.Modalities{2}.ClassLabels(:,t);
                end
                
                if isfield(TrainingData.Modalities{2}, 'GroundTruthLabelIndices')
                    obj.CurrentSample.Modalities{2}.GroundTruthLabelIndices =...
                        TrainingData.Modalities{2}.GroundTruthLabelIndices;
                end
                
                
                %% TRAIN OUTPUT MODALITY ----------------------------------
                %----------------------------------------------------------
                obj.Modalities{2} =...
                    obj.Modalities{2}.train(obj.CurrentSample.Modalities{2});
                                
                %% FIND INPUT MODALITY BMU --------------------------------
                %----------------------------------------------------------               
                obj.Modalities{1}.BMUs =...
                    obj.Modalities{1}.findbmus(obj.CurrentSample.Modalities{1}.NormedFeatureVectors');
                
                
                %% CALCULATE AUXILIARY DISTANCES --------------------------
                %----------------------------------------------------------
%                 obj.Modalities{1}.AuxDists(1) =...
%                     obj.computeauxdist(obj.crossMapping.Weights(obj.Modalities{1}.BMUs(1),:)',...
%                                        obj.Modalities{2}.Activations,...
%                                        obj.Modalities{2}.CostMatrix);
                
%                 for i = 1:size(obj.crossMapping.Weights,1)
%                     obj.Modalities{1}.AuxDists(i) =...
%                            obj.computeauxdist(softmax(obj.crossMapping.Weights(i,:)'),...
%                                               softmax(obj.Modalities{2}.Activations),...
%                                               obj.Modalities{2}.CostMatrix);
%                 end

                for iBMU = 1:obj.Modalities{1}.nAuxDists
                    obj.Modalities{1}.AuxDists(iBMU) =...
                           obj.computeauxdist(obj.crossMapping.Weights(obj.Modalities{1}.BMUs(iBMU),:)',...
                                              obj.Modalities{2}.Activations,...
                                              obj.Modalities{2}.CostMatrix);
                end
                
                % Calculate it for the second BMU if required...
%                 if obj.Modalities{1}.nAuxDists > 1
%                         obj.Modalities{1}.AuxDists(2) =...
%                            obj.computeauxdist(obj.crossMapping.Weights(obj.Modalities{1}.BMUs(2),:)',...
%                                               obj.Modalities{2}.Activations,...
%                                               obj.Modalities{2}.CostMatrix);
%                 end

                %% CALCULATE AUXILIARY DISTANCE RUNNING STATISTICS --------
                %----------------------------------------------------------
                obj.Modalities{1}.AuxDistStats.push(obj.Modalities{1}.AuxDists(1));
                
                %% CALCULATE INPUT MODALITY SAMPLE-NODE DISTANCES ---------
                %----------------------------------------------------------
                % obj = obj.mixeddistances(CurrentSample);
                
                %% TRAIN INPUT MODALITY -----------------------------------
                %---------------------------------------------------------- 
                obj.Modalities{1} =...
                    obj.Modalities{1}.train(obj.CurrentSample.Modalities{1});
                
                %% VISUALIZATION ------------------------------------------
                %----------------------------------------------------------
                % if ~mod(t,5)
                %     subplot(1,2,1);
                %     som_grid(obj.Modalities{1}.SOM, 'Coord', obj.Modalities{1}.SOM.codebook);
                %     subplot(1,2,2);
                %     som_grid(obj.Modalities{2}.SOM, 'Coord', obj.Modalities{2}.SOM.codebook);
                %     drawnow
                % end
                
                %% TRAIN CROSS-MODAL MAPPING ----------------------------------
                %--------------------------------------------------------------
                % obj.crossMapping = obj.crossMapping.train(inMatches', outMatches');
                % fprintf('\nTraining cross-modal mappings...');
                % obj.Modalities{1}.Activations = zeros(size(obj.Modalities{1}.SOM.codebook,1),1);
                % obj.Modalities{1}.Activations(obj.Modalities{1}.BMUs(1)) = 1;
                obj.crossMapping = obj.crossMapping.train(obj.Modalities{1}.BMUs(1), obj.Modalities{2}.BMUs(1),...
                                                          obj.Modalities{1}.Activations, obj.Modalities{2}.Activations);
                
%                 subplot(1,2,1);
%                 hold off;
%                 plot(TrainingData1Epoch.Modalities{1}.NormedFeatureVectors(1,:),...
%                      TrainingData1Epoch.Modalities{1}.NormedFeatureVectors(2,:),'b.');
%                 axis([0 1 0 1]);
%                 hold on;
%                 plot(obj.Modalities{1}.SOM.codebook(:,1), obj.Modalities{1}.SOM.codebook(:,2), 'rx');
%                 drawnow;
% 
%                 subplot(1,2,2);
%                 hold off;
%                 plot(TrainingData1Epoch.Modalities{2}.NormedFeatureVectors(1,:),...
%                      TrainingData1Epoch.Modalities{2}.NormedFeatureVectors(2,:),'b.');
%                 axis([0 1 0 1]);
%                 hold on;
%                 plot(obj.Modalities{2}.SOM.codebook(:,1), obj.Modalities{2}.SOM.codebook(:,2), 'rx');
%                 drawnow;
                                                      
                %% RECORD MODALITY TRAINING INFORMATION OVER TIME ---------
                %----------------------------------------------------------
                if obj.record                    
                    obj.Modalities{1}.AuxDistsRecord(t,:) = obj.Modalities{1}.AuxDists;
                    obj.Modalities{1}.AuxDistMeanRecord(t,:) = obj.Modalities{1}.AuxDistStats.mean();
                    obj.Modalities{1}.AuxDistStDRecord(t,:) = obj.Modalities{1}.AuxDistStats.std();
                    obj.Modalities{1}.GroundTruthRecord(t,:) =...
                        find(obj.CurrentSample.Modalities{1}.ClassLabels(...
                                obj.CurrentSample.Modalities{1}.GroundTruthLabelIndices,:));
                    obj.Modalities{1}.ActivationsRecord(t,:) = obj.Modalities{1}.Activations;
                    obj.Modalities{1}.BMURecord(t, :) = obj.Modalities{1}.BMUs;
                end
                                
                %% DISPLAY VISUALIZATION OF TRAINING ----------------------
                %----------------------------------------------------------
                if obj.visualize
                    clf;
                    hold on;
                    obj.display('figure', CurrentFig,...
                                'voronoi',...
                                'trainingdata',...
                                'DrawOutActivation', obj.CurrentSample.Modalities{2}.NormedFeatureVectors',...
                                'DrawInNodeMaps', obj.Modalities{1}.BMUs(1),...
                                'MapsColour', 'r',...
                                'MapsWeights', true);                    
                    drawnow;
                end
                                                      
                %% CLEAR MODALITY BMUs ------------------------------------
                %----------------------------------------------------------
                obj.Modalities{1} = obj.Modalities{1}.clearbmus();
                obj.Modalities{2} = obj.Modalities{2}.clearbmus();
                
            end            
            
            obj.is_trained = true;
            
        end
        
        %% ------- *** ACTIVETRAIN *** ------------------------------------
        % Train using active learning, i.e. active selection of training
        % data at each timestep.
        %******************************************************************
        %******************************************************************
        function obj = activetrain(obj, varargin)
            
            % Set defaults...
            TrainingData = [];
            
            % Loop through arguments...
            i = 1;
            while i <= length(varargin), 
                argok = 1;
                if isnumeric(varargin{i})
                    sample_size = varargin{i};
                    
                elseif isstruct(varargin{i})
                    TrainingData = varargin{i};
                    sample_size = size(TrainingData.FeatureVectors,2);
                    
                elseif ischar(varargin{i}), 
                    % switch lower(varargin{i}), 
                    %     case 'som_size', i=i+1; obj.SOM_size = varargin{i};
                    % 
                    %     otherwise
                    %         argok = 0;
                    % end
                else
                    argok = 0;
                end

                if ~argok, 
                    disp(['BiModalLearner.activetrain(): Ignoring invalid argument #' num2str(i)]);
                    fprintf(obj.UsageMessage);
                end

                i = i + 1;
            end
            
            if isempty(TrainingData)
                TrainingData = obj.TrainingData;
                sample_size = size(TrainingData.FeatureVectors,2);
            end
            
            %% SET UP FIGURE FOR VISUALIZATION ------------------------
            %----------------------------------------------------------
            if obj.visualize
                CurrentFig = figure;
                hold on;
            end
            
            %% TRAINING LOOP ----------------------------------------------
            %--------------------------------------------------------------
            if obj.verbose                
                fprintf(1, '\nTotal training samples:   %d', size(obj.TrainingData.FeatureVectors,2));
                fprintf(1, '\nCurrent training sample:     ');
            end
            
            for t = obj.t+1 : min(obj.t + sample_size, size(obj.TrainingData.FeatureVectors,2))
            % for t = 1:size(obj.TrainingData.FeatureVectors,2)
                
                obj.t = t;
                
                % Print progress...
                if obj.verbose
                    Backspaces = [];
                    for iBack = 1:ceil(log10(t+1))
                        Backspaces = [Backspaces '\b'];
                    end
                    fprintf(1, [Backspaces '%d'], obj.t);
                end
                
                if ~isfield(obj.TrainingData1Epoch,'AvailableSamples') ||...
                   all(obj.TrainingData1Epoch.AvailableSamples == 0)
                    obj.TrainingData1Epoch.AvailableSamples = ones(1,size(obj.TrainingData1Epoch.FeatureVectors,2));
                end
                
                %% SELECT SAMPLE  -----------------------------------------
                %----------------------------------------------------------                    
                sample_selected = false;
                
                if obj.is_trained
                
                    %% CLASSIFY TRAINING DATA -----------------------------
                    %------------------------------------------------------
                    [obj.TrainingData1Epoch obj.TrainingData1Epoch.Results.ClassificationMask] =...
                        obj.classify(obj.TrainingData1Epoch);

                    %% SELECT SAMPLE WITH LOWEST MAX CLASS PROBABILITY ----
                    %------------------------------------------------------------                    
                    % ClassProbVars = var(obj.TrainingData1Epoch.Results.ClassProbs');
                    [ClassProbMaxes ClassProbMaxLabels] = max(obj.TrainingData1Epoch.Results.ClassProbs');
                    
                    if ~any(isnan(ClassProbMaxes))
                        while ~sample_selected                        
                            [~, iSample] = min(ClassProbMaxes);

                            if obj.TrainingData1Epoch.AvailableSamples(iSample) == 1
                                obj.TrainingData1Epoch.AvailableSamples(iSample) = 0;
                                sample_selected = true;
                                break;
                            else
                                ClassProbMaxes(iSample) = Inf;
                            end
                        end
                    end                                        
                end
                
                %% OTHERWISE RANDOMLY SELECT SAMPLE -----------------------
                %----------------------------------------------------------
                if ~sample_selected
                    iSample = ceil(rand*size(obj.TrainingData1Epoch.FeatureVectors,2));
                    obj.TrainingData1Epoch.AvailableSamples(iSample) = 0;
                end
                    
                %% GRAB SAMPLE ------------------------------------
                %----------------------------------------------------------
                obj.CurrentSample.Modalities{1}.FeatureVectors(:,1) =...
                    obj.TrainingData.Modalities{1}.FeatureVectors(:,iSample);
                obj.CurrentSample.Modalities{1}.NormedFeatureVectors(:,1) =...
                    obj.TrainingData.Modalities{1}.NormedFeatureVectors(:,iSample);
                obj.CurrentSample.Modalities{1}.ClassLabels(:,1) =...
                    obj.TrainingData.Modalities{1}.ClassLabels(:,iSample);
                obj.CurrentSample.Modalities{1}.GroundTruthLabelIndices =...
                    obj.TrainingData.Modalities{1}.GroundTruthLabelIndices;
                obj.CurrentSample.Modalities{2}.FeatureVectors(:,1) =...
                    obj.TrainingData.Modalities{2}.FeatureVectors(:,iSample);
                obj.CurrentSample.Modalities{2}.NormedFeatureVectors(:,1) =...
                    obj.TrainingData.Modalities{2}.NormedFeatureVectors(:,iSample);
                obj.CurrentSample.Modalities{2}.ClassLabels(:,1) =...
                    obj.TrainingData.Modalities{2}.ClassLabels(:,iSample);
                obj.CurrentSample.Modalities{2}.GroundTruthLabelIndices =...
                    obj.TrainingData.Modalities{2}.GroundTruthLabelIndices;
                
                
                %% TRAIN OUTPUT MODALITY ----------------------------------
                %----------------------------------------------------------
                obj.Modalities{2} =...
                    obj.Modalities{2}.train(obj.CurrentSample.Modalities{2});
                                
                %% FIND INPUT MODALITY BMU --------------------------------
                %----------------------------------------------------------               
                obj.Modalities{1}.BMUs =...
                    obj.Modalities{1}.findbmus(obj.CurrentSample.Modalities{1}.NormedFeatureVectors');
                
                
                %% CALCULATE AUXILIARY DISTANCES --------------------------
                %----------------------------------------------------------
%                 obj.Modalities{1}.AuxDists(1) =...
%                     obj.computeauxdist(obj.crossMapping.Weights(obj.Modalities{1}.BMUs(1),:)',...
%                                        obj.Modalities{2}.Activations,...
%                                        obj.Modalities{2}.CostMatrix);
                
%                 for i = 1:size(obj.crossMapping.Weights,1)
%                     obj.Modalities{1}.AuxDists(i) =...
%                            obj.computeauxdist(softmax(obj.crossMapping.Weights(i,:)'),...
%                                               softmax(obj.Modalities{2}.Activations),...
%                                               obj.Modalities{2}.CostMatrix);
%                 end

                for iBMU = 1:obj.Modalities{1}.nAuxDists
                    obj.Modalities{1}.AuxDists(iBMU) =...
                           obj.computeauxdist(obj.crossMapping.Weights(obj.Modalities{1}.BMUs(iBMU),:)',...
                                              obj.Modalities{2}.Activations,...
                                              obj.Modalities{2}.CostMatrix);
                end
                
                % Calculate it for the second BMU if required...
%                 if obj.Modalities{1}.nAuxDists > 1
%                         obj.Modalities{1}.AuxDists(2) =...
%                            obj.computeauxdist(obj.crossMapping.Weights(obj.Modalities{1}.BMUs(2),:)',...
%                                               obj.Modalities{2}.Activations,...
%                                               obj.Modalities{2}.CostMatrix);
%                 end

                %% CALCULATE AUXILIARY DISTANCE RUNNING STATISTICS --------
                %----------------------------------------------------------
                obj.Modalities{1}.AuxDistStats.push(obj.Modalities{1}.AuxDists(1));
                
                %% CALCULATE INPUT MODALITY SAMPLE-NODE DISTANCES ---------
                %----------------------------------------------------------
                % obj = obj.mixeddistances(CurrentSample);
                
                %% TRAIN INPUT MODALITY -----------------------------------
                %---------------------------------------------------------- 
                obj.Modalities{1} =...
                    obj.Modalities{1}.train(obj.CurrentSample.Modalities{1});
                
                %% VISUALIZATION ------------------------------------------
                %----------------------------------------------------------
                % if ~mod(t,5)
                %     subplot(1,2,1);
                %     som_grid(obj.Modalities{1}.SOM, 'Coord', obj.Modalities{1}.SOM.codebook);
                %     subplot(1,2,2);
                %     som_grid(obj.Modalities{2}.SOM, 'Coord', obj.Modalities{2}.SOM.codebook);
                %     drawnow
                % end
                
                %% TRAIN CROSS-MODAL MAPPING ----------------------------------
                %--------------------------------------------------------------
                % obj.crossMapping = obj.crossMapping.train(inMatches', outMatches');
                % fprintf('\nTraining cross-modal mappings...');
                % obj.Modalities{1}.Activations = zeros(size(obj.Modalities{1}.SOM.codebook,1),1);
                % obj.Modalities{1}.Activations(obj.Modalities{1}.BMUs(1)) = 1;
                obj.crossMapping = obj.crossMapping.train(obj.Modalities{1}.BMUs(1), obj.Modalities{2}.BMUs(1),...
                                                          obj.Modalities{1}.Activations, obj.Modalities{2}.Activations);
                                                      
                %% RECORD MODALITY TRAINING INFORMATION OVER TIME ---------
                %----------------------------------------------------------
                if obj.record                    
                    obj.Modalities{1}.AuxDistsRecord(t,:) = obj.Modalities{1}.AuxDists;
                    obj.Modalities{1}.AuxDistMeanRecord(t,:) = obj.Modalities{1}.AuxDistStats.mean();
                    obj.Modalities{1}.AuxDistStDRecord(t,:) = obj.Modalities{1}.AuxDistStats.std();
                    obj.Modalities{1}.GroundTruthRecord(t,:) =...
                        find(obj.CurrentSample.Modalities{1}.ClassLabels(...
                                obj.CurrentSample.Modalities{1}.GroundTruthLabelIndices,:));
                    obj.Modalities{1}.ActivationsRecord(t,:) = obj.Modalities{1}.Activations;
                    obj.Modalities{1}.BMURecord(t, :) = obj.Modalities{1}.BMUs;
                end
                                
                %% DISPLAY VISUALIZATION OF TRAINING ----------------------
                %----------------------------------------------------------
                if obj.visualize
                    clf;
                    hold on;
                    obj.display('figure', CurrentFig,...
                                'voronoi',...
                                'trainingdata',...
                                'DrawOutActivation', obj.CurrentSample.Modalities{2}.NormedFeatureVectors',...
                                'DrawInNodeMaps', obj.Modalities{1}.BMUs(1),...
                                'MapsColour', 'r',...
                                'MapsWeights', true);                    
                    drawnow;
                end
                                                      
                %% CLEAR MODALITY BMUs ------------------------------------
                %----------------------------------------------------------
                obj.Modalities{1} = obj.Modalities{1}.clearbmus();
                obj.Modalities{2} = obj.Modalities{2}.clearbmus();
                
                obj.is_trained = true;
                
            end                                    
            
        end        
        
        %% ------- *** CLASSIFY *** ---------------------------------------
        %******************************************************************
        %******************************************************************
        function [TestData] = classify(obj, varargin)
            
            % Defaults...
            switch obj.classification_method
                case {'svm'},...
                    Method = 'svm';
                case {'lda'},...
                    Method = 'lda';
                case {'naivebayes', 'naivebayes_linear', 'naivebayes_diaglinear', 'diaglinear'},
                    Method = 'diaglinear';
                case {'naivebayes_quadratic', 'naivebayes_diagquadratic', 'diagquadratic'},...
                    Method = 'diagquadratic';
                    
                otherwise, Method = 'node';
            end
            
            switch obj.node_colouring_method
                case {'cluster_mean'},...
                    NodeColouringMethod = 'cluster_mean';
                case {'cluster_max'},...
                    NodeColouringMethod = 'cluster_max';
                case{'cluster_mean_node_cull'},...
                    NodeColouringMethod = 'cluster_mean_node_cull';
                
                otherwise, NodeColouringMethod = 'cluster_mean';
            end
            
            % The TestData struct...
            TestData = [];
            TestDataIn = [];
            
            % A flag that lets us know if we're using the TestData struct
            % in the object, or test data that was passed as an argument...
            using_internal_testdata = false;
                        
            Metric = 'euclidean';
            hebbian = true;
            
            % Should the current output modality clustering be retained for
            % class labels?
            RetainClustering = false;
            
            % Should we use a decision threshold?
            DecisionThreshold = [];
            
            % FOR THE MOMENT, THE CLASSIFY METHOD WILL JUST CLASSIFY
            % INTERNAL TEST SAMPLE INPUT MODALITY VECTORS...
            % 
            % ToDo: Extend this!
            
            %% CHECK THAT WE HAVE A TRAINED CLASSIFIER --------------------
            %--------------------------------------------------------------
            if ~obj.is_trained
                fprintf(['\n\nThe classifier has not been trained yet!\n\n'...
                       'Please use CrossMod.train first!\n\n\n']);
                return;
            end
            
            %% CHECK ARGUMENTS --------------------------------------------
            %--------------------------------------------------------------
            % varargin
            i=1; 
            while i<=length(varargin), 
              argok = 1; 
              if ischar(varargin{i}), 
                switch varargin{i}, 
                    % argument IDs
                    case {'data', 'testdata'}, i=i+1; TestDataIn = varargin{i};
                    case {'display'}, obj.display = true;
                    case {'svm'}, Method = 'svm';
                    case {'lda'}, Method = 'lda';
                    case {'lvq'}, Method = 'lvq';
                    case {'nodewise', 'node'}, Method = 'node';
                    case {'euclidean', 'euclid', 'euc'}, Metric = 'euclidean';
                    case {'hellinger', 'hell'}, Metric = 'hellinger';
                    case {'clusterwise', 'cluster'}, Method = 'cluster';
                    case {'nonhebbian'}, hebbian = false;
                    case {'node_colouring_method'}, i=i+1; NodeColouringMethod = varargin{i};
                    case {'keep_clustering', 'keep_clusters',...
                          'retain_clustering', 'retain_clusters'},...
                          RetainClustering = true;
                    case {'decision_threshold', 'threshold'},...
                        i=i+1; DecisionThreshold = varargin{i};
                      
                    otherwise, argok=0; 
                end
              elseif isstruct(varargin{i})
                  TestDataIn = varargin{i};  
              else
                argok = 0; 
              end
              if ~argok, 
                disp(['BiModalLearner.classify(): Ignoring invalid argument #' num2str(i+1)]); 
              end
              i = i+1; 
            end
            
            %% SET UP TEST DATA -------------------------------------------
            %--------------------------------------------------------------
            if isempty(TestDataIn)
                
                TestData = obj.TestData;
                using_internal_testdata = true;
               
            elseif isnumeric(TestDataIn)
               
                NumericData = TestDataIn;        
                TestData.FeatureVectors = NumericData;
                
                if size(TestData.FeatureVectors,1) == size(obj.Modalities{1}.SOM.codebook,2)
                    
                    TestData.Modalities{1}.FeatureVectors = TestData.FeatureVectors;
                    
                elseif size(TestData.FeatureVectors,1) > size(obj.Modalities{1}.SOM.codebook,2)
                    
                    fprintf(['Warning: Dimensionality of numeric input too large'...
                             ', using first %d dimensions.\n'], size(obj.Modalities{1}.SOM.codebook,2));
                    
                    TestData.Modalities{1}.FeatureVectors =...
                        TestData.FeatureVectors(1:size(obj.Modalities{1}.SOM.codebook,2),:);
                    
                else
                    
                    error('Error: Numeric input should be %d-dimensional!',...
                          size(obj.Modalities{1}.SOM.codebook,2));
                    
                end
                
            else
                
                TestData = TestDataIn;
                
            end                                    
            
            %
            % WARNING: THIS IF CLAUSE IS A VERY BAD IDEA FOR ONLINE
            % TESTING!!!
            %
            % if isempty(obj.Modalities{1}.ClassLabels)                            

                %% CLUSTER OUTPUT MODALITY --------------------------------
                %----------------------------------------------------------
                if ~RetainClustering                    
                    fprintf('\nClustering OUTPUT modality...');
                    
                    [obj.Modalities{2} Centroids Indices Errors OptimalK KValidityInfo TestData.Results.OutputMask] =...
                        obj.Modalities{2}.cluster('feature_selection', obj.output_feature_selection,...
                                                  'feature_selection_params', obj.output_feature_selection_params);
                    TestData.Results.OutputMask = TestData.Results.OutputMask';
                    obj.OutputClassificationMask = TestData.Results.OutputMask;
                else
                    TestData.Results.OutputMask = obj.OutputClassificationMask;
                end
                
                %% CALCULATE CLUSTER WEIGHTS FOR EACH INPUT MODALITY NODE--
                %  ...as well as the average difference between cluster
                %     for each node.
                %----------------------------------------------------------                
                % Calculate in-modality node weights for each out-modality
                % cluster...
                for iOutCluster = 1:obj.Modalities{2}.Clustering.OptimalK
                    InNodeToOutClusterWeights(:,iOutCluster) =...
                        sum(obj.crossMapping.Weights(:,obj.Modalities{2}.Clustering.Labels{obj.Modalities{2}.Clustering.OptimalK} == iOutCluster),2) ./...
                            sum(obj.Modalities{2}.Clustering.Labels{obj.Modalities{2}.Clustering.OptimalK} == iOutCluster);
                end

                % Cull ambiguous nodes by giving them negative class labels...
                Foo = mean(repmat(max(InNodeToOutClusterWeights,[],2),1,size(InNodeToOutClusterWeights,2)) - InNodeToOutClusterWeights,2);
                FooMean = mean(Foo);
                FooSTD = std(Foo);
                
                AccuracyHist = Foo ./ norm(Foo,1);

                % Foo is the average difference between cluster
                % weights for each node.
                
                %% USE OUT-MODALITY CLUSTERS AND CROSS-MODAL MAPPINGS TO LABEL
                %  IN-MODALITY NODES...
                %--------------------------------------------------------------
                switch NodeColouringMethod
                    
                    case 'cluster_mean'                        

                        % Clear this...
                        obj.Modalities{1}.ClassProbs = [];
                        
                        % Use these weights to label the in-modality nodes...
                        for iInNode = 1:size(obj.Modalities{1}.SOM.codebook,1)

                            % Old way using SOM neighbours...
                            % Neighs = find(Ne1(iInNode,:));

                            for iOutCluster = 1:size(InNodeToOutClusterWeights,2)
                                % NodeClusterScores(iInCluster) = sum(Weights(iOutCluster,Neighs));
                                NodeClusterScores(iOutCluster) = InNodeToOutClusterWeights(iInNode,iOutCluster);
                            end

                            % Record class probabilities for each node...                            
                            obj.Modalities{1}.ClassProbs(iInNode,:) = NodeClusterScores ./ norm(NodeClusterScores,1);
                            
                            % Maximise over the class probabilities to
                            % label the nodes...
                            [foo obj.Modalities{1}.ClassLabels(iInNode,:)] = max(NodeClusterScores);
                        end                                            
                        
                    case 'cluster_max'
                        for iInNode = 1:size(obj.Modalities{1}.SOM.codebook,1)
                            [foo bar] = max(obj.crossMapping.Weights(iInNode,:));
                            obj.Modalities{1}.ClassLabels(iInNode,:) = obj.Modalities{2}.Clustering.Labels{obj.Modalities{2}.Clustering.OptimalK}(bar);
                        end
                        
                    case 'cluster_mean_node_cull'                        

                        % Clear this...
                        obj.Modalities{1}.ClassProbs = [];
                        
                        % Use these weights to label the in-modality nodes...
                        for iInNode = 1:size(obj.Modalities{1}.SOM.codebook,1)

                            % Old way using SOM neighbours...
                            % Neighs = find(Ne1(iInNode,:));

                            for iOutCluster = 1:size(InNodeToOutClusterWeights,2)
                                % NodeClusterScores(iInCluster) = sum(Weights(iOutCluster,Neighs));
                                NodeClusterScores(iOutCluster) = InNodeToOutClusterWeights(iInNode,iOutCluster);
                            end
                            
                            % Record class probabilities for each node...
                            obj.Modalities{1}.ClassProbs(iInNode,:) = NodeClusterScores ./ norm(NodeClusterScores,1);
                            
                            % Use this heuristic to cull nodes...
                            % if Foo(iInNode) > (FooMean - FooSTD)
                            if Foo(iInNode) >= FooMean
                                % If the node passes the heuristic, set the
                                % class label to the cluster with the
                                % maximum score...
                                [foo obj.Modalities{1}.ClassLabels(iInNode,:)] = max(NodeClusterScores);
                            else
                                % If the node fails the heuristic, set the
                                % class label to minus above value.
                                [foo obj.Modalities{1}.ClassLabels(iInNode,:)] = max(NodeClusterScores);
                                obj.Modalities{1}.ClassLabels(iInNode,:) = -obj.Modalities{1}.ClassLabels(iInNode,:);
                            end
                        end
                        
                        % We should check to make sure each class has at
                        % least one un-culled node.  If not, we set the negative class labels for that
                        % class to positive.
                        for iOutCluster = 1:size(InNodeToOutClusterWeights,2)
                            if ~any(obj.Modalities{1}.ClassLabels == iOutCluster)
                                obj.Modalities{1}.ClassLabels(obj.Modalities{1}.ClassLabels == -iOutCluster) = iOutCluster;
                            end
                        end
                end
            % end
            
            %% USE INDIVIDUAL NODE ALPHA VALUES (FROM OLVQ) TO ESTIMATE ---
            % THE RELEVANCE OF INDIVIDUAL NODES...
            %--------------------------------------------------------------
            % if ~isempty(obj.Modalities{1}.Alphas)
            %     NodeRelevances = 1  - obj.Modalities{1}.Alphas;
            %     Codebook = obj.Modalities{1}.SOM.codebook(NodeRelevances >= mean(NodeRelevances(:)), :);
            %     ClassLabels = obj.Modalities{1}.ClassLabels(NodeRelevances >= mean(NodeRelevances(:)));
            % else
                % NodeRelevances = ones(size(obj.Modalities{1}.ClassLabels));
                Codebook = obj.Modalities{1}.SOM.codebook(obj.Modalities{1}.ClassLabels > 0,:);
                ClassLabels = obj.Modalities{1}.ClassLabels(obj.Modalities{1}.ClassLabels > 0);
                ClassProbs = obj.Modalities{1}.ClassProbs(obj.Modalities{1}.ClassLabels > 0,:);
                CrossMappingWeights = obj.crossMapping.Weights(obj.Modalities{1}.ClassLabels > 0,:);
                AccuracyHist = AccuracyHist(obj.Modalities{1}.ClassLabels > 0);
            % end
            
            %% CLASSIFY TEST VECTORS IN INPUT MODALITY --------------------
            %  Classify the in modality test vectors in terms of out
            %  modality clusters.
            %--------------------------------------------------------------            
            switch Method
                
                case 'node'
                    %% SET UP FEATURE SELECTION -------------------------------------
                    %----------------------------------------------------------------
                    % If LDA-based feature selection is requested, we do
                    % the calculations here a-posteriori (after
                    % training)...
                    if ~isempty(findstr(obj.feature_selection, 'lda'))
                        
                        if ~isempty(AccuracyHist) && sum(AccuracyHist) ~= 0
                            
                            for iClass = 1:max(ClassLabels(:))

                                ClassData = Codebook(ClassLabels == iClass, :);
                                ClassNodeAccuracies = AccuracyHist(ClassLabels == iClass);
                                if sum(ClassNodeAccuracies) == 0
                                    ClassNodeAccuracies = ones(size(ClassNodeAccuracies));
                                end
                                ClassNodeWeights = ClassNodeAccuracies + abs(min(ClassNodeAccuracies));
                                ClassNodeWeights = ClassNodeWeights ./ norm(ClassNodeWeights,1);
                                ClassNodeWeights(isnan(ClassNodeWeights)) = 0;

                                % Just in case the class only contains
                                % one node...
                                if size(ClassData,1) == 1
                                    ClassMeans(iClass,:) = ClassData;
                                    ClassVars(iClass,:) = zeros(size(ClassData));
                                else
                                    % Calculate the weighted mean...
                                    ClassMeans(iClass,:) = sum(repmat(ClassNodeWeights,1,size(ClassData,2)) .* ClassData,1);
                                    % Calculate the weighted variance...
                                    ClassVars(iClass,:) = sum(repmat(ClassNodeWeights,1,size(ClassData,2)) .* (ClassData - repmat(ClassMeans(iClass,:), size(ClassData,1), 1)).^2);
                                end

                            end
                            
                        else
                            
                            for iClass = 1:max(ClassLabels(:))

                                ClassData = Codebook(ClassLabels == iClass, :);

                                % Just in case the cluster only contains
                                % one node...
                                if size(ClassData,1) == 1
                                    ClassMeans(iClass,:) = ClassData;
                                    ClassVars(iClass,:) = zeros(size(ClassData));
                                else
                                    ClassMeans(iClass,:) = mean(ClassData);
                                    ClassVars(iClass,:) = var(ClassData);
                                end
                            end
                        end

                        % Fisher Criterion =
                        %  (Between Class Variance)
                        % --------------------------
                        %  (Within Class Variance)
                        if ~exist('ClassVars', 'var') || ~exist('ClassMeans', 'var') ||...
                           size(ClassMeans,1) <= 1 || size(ClassVars,1) <= 1
                            FisherCriterion = ones(size(obj.Modalities{1}.SOM.mask'));
                        elseif any(sum(ClassVars) == 0)
                            FisherCriterion = var(ClassMeans);
                        else
                            FisherCriterion = var(ClassMeans) ./ sum(ClassVars);
                        end

                        % Watch out for nasty NaNs...
                        FisherCriterion(isnan(FisherCriterion)) = 0;

                        % Make the mask a weight distribution (unit norm)...
                        % Mask = (FisherCriterion ./ norm(FisherCriterion,1))';
                        TestData.Results.InputMask = (FisherCriterion ./ max(FisherCriterion))';
                        obj.InputClassificationMask = TestData.Results.InputMask;

                        % Save the mask for later...
                        % obj.Modalities{1}.SOM.mask = Mask;
                    
                    % LDA-based feature selection based on node statistics
                    % gathered during training, as opposed to the node
                    % weight vectors (as above)...
                    elseif ~isempty(findstr(obj.feature_selection, 'nodestats'))
                        
                        % Some of the notation in the following is derived                        
                        % from equations (18) in "Multivariate Online Kernel
                        % Density Estimation" by Kristan et al.
                                                                        
                        for iClass = 1:max(ClassLabels(:))
                            
                            ClassNodeIndices = find(ClassLabels==iClass);
                            
                            % Sum accuracy weights...
                            w_j = sum(obj.Modalities{1}.AccuracyHist(ClassNodeIndices));                                                        
                            
                            % Calculate the class mean...
                            mu_j = 0;
                            for iNode = 1:size(ClassNodeIndices,1)
                                
                                mu_i = obj.Modalities{1}.NodeStats{ClassNodeIndices(iNode)}.mean();
                                
                                if isnan(mu_i)
                                    mu_i = 0;
                                end
                                
                                mu_j = mu_j + (obj.Modalities{1}.AccuracyHist(ClassNodeIndices(iNode)) *...
                                               mu_i);
                            end                                    
                            mu_j = w_j^(-1) * mu_j;
                            
                            % Calculate the class variance...
                            sig_j = 0;
                            for iNode = 1:size(ClassNodeIndices,1)
                                
                                mu_i = obj.Modalities{1}.NodeStats{ClassNodeIndices(iNode)}.mean();
                                sig_i = obj.Modalities{1}.NodeStats{ClassNodeIndices(iNode)}.var();
                                
                                if isnan(mu_i)
                                    mu_i = 0;
                                end
                                
                                if isnan(sig_i)
                                    sig_i = 0;
                                end
                                
                                sig_j = sig_j + (obj.Modalities{1}.AccuracyHist(ClassNodeIndices(iNode)) *...
                                                 (sig_i + mu_i.^2));
                            end
                            sig_j = (w_j.^(-1) * sig_j) - mu_j.^2;
                            
                            % Save...
                            if isnan(mu_j)
                                ClassMeans(iClass,:) = zeros(1,size(obj.Modalities{1}.SOM.codebook,2));
                            else                                
                                ClassMeans(iClass,:) = mu_j;
                            end
                            
                            if isnan(mu_j)
                                ClassVars(iClass,:) = zeros(1,size(obj.Modalities{1}.SOM.codebook,2));
                            else
                                ClassVars(iClass,:) = sig_j;
                            end
                                                        
                        end
                        
                        % Fisher Criterion =
                        %  (Between Class Variance)
                        % --------------------------
                        %  (Within Class Variance)
                        if ~exist('ClassVars', 'var') || ~exist('ClassMeans', 'var') ||...
                           size(ClassMeans,1) <= 1 || size(ClassVars,1) <= 1
                            FisherCriterion = ones(size(obj.Modalities{1}.SOM.mask'));
                        elseif any(sum(ClassVars) == 0)
                            FisherCriterion = var(ClassMeans);
                        else
                            FisherCriterion = var(ClassMeans) ./ sum(ClassVars);
                        end

                        % Watch out for nasty NaNs...
                        FisherCriterion(isnan(FisherCriterion)) = 0;

                        % Make the mask a weight distribution (unit norm)...
                        % Mask = (FisherCriterion ./ norm(FisherCriterion,1))';
                        TestData.Results.InputMask = (FisherCriterion ./ max(FisherCriterion))';                        
                        obj.InputClassificationMask = TestData.Results.InputMask;

                        % Save the mask for later...
                        % obj.Modalities{1}.SOM.mask = Mask;
                        
                    % Otherwise, just make sure the feature mask is
                    % normalized...
                    else
                        TestData.Results.InputMask = obj.Modalities{1}.SOM.mask ./ norm(obj.Modalities{1}.SOM.mask,1);
                        obj.InputClassificationMask = TestData.Results.InputMask;
                    end
                    
                    %% CLASSIFY TEST VECTORS NODE-WISE IN INPUT MODALITY ------------
                    %----------------------------------------------------------------
                    % For hard feature selection, we pick out the most
                    % relevant features based on the mean feature weight
                    % and re-normalize...
                    if ~isempty(findstr(obj.feature_selection, 'hard'))
                        
                        if isempty(obj.feature_selection_max) || ischar(obj.feature_selection_max)
                            TestData.Results.InputMask(TestData.Results.InputMask < mean(TestData.Results.InputMask)) = 0;
                        else                            
                            if obj.feature_selection_max > 1
                                [Foo Bar] = sort(TestData.Results.InputMask,'descend');
                                TestData.Results.InputMask(Bar(obj.feature_selection_max+1:end)) = 0;
                            else
                                TestData.Results.InputMask(TestData.Results.InputMask < obj.feature_selection_max) = 0;
                            end
                        end                        
                        % Mask = Mask ./ norm(Mask,1);
                        
                        
                        [TestDataInMatches TestData.Modalities{1}.NormedFeatureVectors] =...
                                obj.Modalities{1}.classify(TestData.Modalities{1},...
                                                           'codebook', Codebook(ClassLabels > 0, :),...
                                                           'mask', TestData.Results.InputMask);                        
                    
                    % Query-based exponential feature weighting...
                    elseif ~isempty(findstr(obj.feature_selection, 'exp'))
                        
                        % First off, we have to find the BMUs for the test
                        % data...
                        [TestDataBMUs TestData.Modalities{1}.NormedFeatureVectors] =...
                                    obj.Modalities{1}.classify(TestData.Modalities{1},...
                                                               'codebook', Codebook(ClassLabels > 0, :),...
                                                               'mask', TestData.Results.InputMask,...
                                                               'whichbmus', 'all');
                                                           
                        % Next, based on the class of the first BMU of each
                        % test query, we must determine the closest BMU of
                        % an opposing class, so that we can calculate their
                        % seperating hyperplane.  Then the distance between
                        % the test query and the seperating hyperplane can
                        % be used to locally re-calculate the feature
                        % weights.
                        for iTestData = 1:size(TestDataBMUs,1)
                            
                            BMUClass1 = TestDataBMUs(iTestData,1);
                            
                            % Find the closest BMU of a different class...
                            for iBMU = 2:size(TestDataBMUs,2)                                
                                BMUClass2 = TestDataBMUs(iTestData, iBMU);
                                if obj.Modalities{1}.ClassLabels(BMUClass2) ~=...
                                        obj.Modalities{1}.ClassLabels(BMUClass1)
                                    break;
                                end                                
                            end
                            
                            % Let's calculate the distance between the test
                            % query point and the hyperplane seperating
                            % the BMUs of the different classes...
                            
                            % BMU vectors...
                            a = obj.Modalities{1}.SOM.codebook(BMUClass1,:)';
                            b = obj.Modalities{1}.SOM.codebook(BMUClass2,:)';
                            
                            % Point on the hyperplane...
                            q = a + ((b - a)/2);
                            
                            % Query point...
                            p = TestData.Modalities{1}.NormedFeatureVectors(:,iTestData);
                            
                            % Norm to the hyperplane...
                            n = (b - a);
                            
                            % Distance from point p to the hyperplane...
                            dist = abs(dot((p - q), n)) / norm(n,1);
                            
                            % Use this to calculate the exponential
                            % weighting factor...
                            C = 1 / dist;
                            
                            % Calculate the feature weights for this
                            % query...
                            QueryMask = exp(C * TestData.Results.InputMask) ./ sum(exp(C * TestData.Results.InputMask));
                                                        
                            % Classify
                            QueryMask(QueryMask < mean(QueryMask)) = 0;
                            QueryMask = QueryMask ./ norm(QueryMask,1);
                            TestDataInMatches(iTestData) =...
                                obj.Modalities{1}.classify(TestData.Modalities{1}.FeatureVectors(:,iTestData),...
                                                           'codebook', Codebook(ClassLabels > 0, :),...
                                                           'mask', QueryMask,...
                                                           'whichbmus', 'best');
                                                       
                            TestData.Results.InputMask = QueryMask;
                            
                        end
                        
                    % Fuzzy feature weighting...
                    elseif ~isempty(findstr(obj.feature_selection, 'fuzzy'))
                        
                        [TestDataInMatches TestData.Modalities{1}.NormedFeatureVectors] =...
                                    obj.Modalities{1}.classify(TestData.Modalities{1},...
                                                               'codebook', Codebook(ClassLabels > 0, :),...
                                                               'mask', TestData.Results.InputMask);
                                                           
                        
                    
                    % Otherwise, ignore feature weights...
                    else
                        [TestDataInMatches TestData.Modalities{1}.NormedFeatureVectors] =...
                                    obj.Modalities{1}.classify(TestData.Modalities{1},...
                                                               'codebook', Codebook(ClassLabels > 0, :));
                                                           
                        TestData.Results.InputMask = nan(size(obj.Modalities{1}.SOM.mask));
                        
                    end
                    
                    TestData.Results.InToOutClassification = ClassLabels(TestDataInMatches')';
                    
                    % Let's also add regression to the results...
                    [~,TestDataInToOutNodePredictions] = max(CrossMappingWeights(TestDataInMatches,:),[],2);
                    NormedInToOutRegression = obj.Modalities{2}.SOM.codebook(TestDataInToOutNodePredictions,:);
                    TestData.Results.InToOutRegression = som_denormalize(NormedInToOutRegression, obj.Modalities{2}.NormStruct);

                case 'cluster'
                    %% CLASSIFY TEST VECTORS CLUSTER-WISE IN INPUT MODALITY ---------
                    %----------------------------------------------------------------
                    [TestDataInMatches TestData.Modalities{1}.NormedFeatureVectors] =...
                        obj.Modalities{1}.classify(TestData.Modalities{1}, 'clusterwise');

                    for i = 1:length(TestDataInMatches)

                        for j = 1:obj.Modalities{2}.Clustering.OptimalK

                            % Find the in modality winning cluster nodes for
                            % this test sample...
                            InWinningClusterNodes =...
                                find(TestDataInMatches(i,obj.Modalities{1}.KNN_max_k) ==...
                                    obj.Modalities{1}.Clustering.Labels{obj.Modalities{1}.Clustering.OptimalK});

                            if hebbian
                                % Sum the weights over all the nodes in the in
                                % modality cluster...
                                Weights = sum(obj.crossMapping.Weights(InWinningClusterNodes,:),1);

                                % Calculate the total weight score between the
                                % in modality cluster and the current out modality
                                % cluster (j)
                                TestData.Results.InToOutScores(i,j) =...
                                        sum(Weights(obj.Modalities{2}.Clustering.Labels{obj.Modalities{2}.Clustering.OptimalK} == j)) /...
                                           sum(obj.Modalities{2}.Clustering.Labels{obj.Modalities{2}.Clustering.OptimalK} == j);
                            else

                                % Find most recent co-occurences...
                                RecentCoOccurrences = obj.crossMapping.CoOccurrences(end-100-1:end,:);

                                % Find the ones that match the in-winning
                                % cluster...                                
                                [RecentCoOccurenceMatches InWinningClusterMatches] = find(repmat(RecentCoOccurrences(:,1),1,size(InWinningClusterNodes,1)) ==...
                                    repmat(InWinningClusterNodes', size(RecentCoOccurrences,1),1));

                                % Find the corresponding cluster
                                TestData.Results.InToOutScores(i,j) =...
                                    sum(obj.Modalities{2}.Clustering.Labels{obj.Modalities{2}.Clustering.OptimalK}(RecentCoOccurrences(RecentCoOccurenceMatches,2)) == j);
                            end
                        end

                        [foo TestData.Results.InToOutClassification(i)] =...
                                max(TestData.Results.InToOutScores(i,:));

                    end
                    
                case 'svm'
                    %% TRAIN AN SVM CLASSIFIER IN THE IN-MODALITY -----------------
                    %--------------------------------------------------------------
                    Classifier =...
                        svmtrain(Codebook,...
                                 ClassLabels,...
                                 'Kernel_Function', 'rbf');

                    %% CONVERT & NORMALIZE TEST DATA ------------------------------
                    %--------------------------------------------------------------
                    SOMTestData = som_data_struct(TestData.Modalities{1}.FeatureVectors');

                    % Normalize...
                    SOMTestData = som_normalize(SOMTestData, obj.Modalities{1}.Norm);
                    
                    % Record normalized feature vectors...
                    TestData.Modalities{1}.NormedFeatureVectors = SOMTestData.data';

                    %% CLASSIFICATION USING THE SVM CLASSIFIER --------------------
                    %--------------------------------------------------------------
                    TestData.Results.InToOutClassification =...
                        svmclassify(Classifier, SOMTestData.data);

                    TestData.Results.InToOutClassification = TestData.Results.InToOutClassification';
                    
                case 'lda'
                    %% CONVERT & NORMALIZE TEST DATA ------------------------------
                    %--------------------------------------------------------------
                    SOMTestData = som_data_struct(TestData.Modalities{1}.FeatureVectors');

                    % Normalize...
                    SOMTestData = som_normalize(SOMTestData, obj.Modalities{1}.Norm);
                    
                    % Record normalized feature vectors...
                    TestData.Modalities{1}.NormedFeatureVectors = SOMTestData.data';

                    %% CLASSIFICATION USING THE LDA CLASSIFIER --------------------
                    %--------------------------------------------------------------
                    TestData.Results.InToOutClassification =...
                        classify(SOMTestData.data, Codebook, ClassLabels);

                    TestData.Results.InToOutClassification = TestData.Results.InToOutClassification';
                    
                case 'lvq'
                    [TestDataInMatches TestData.Modalities{1}.NormedFeatureVectors] =...
                        obj.Modalities{1}.classify(TestData.Modalities{1});
                    
                    TestData.Results.InToOutClassification = obj.Modalities{1}.LVQLabels(TestDataInMatches)';
                    
            end
            
            obj.InputClassificationMask = TestData.Results.InputMask;
            
            % Save the results in the TestData struct...
            TestData.Results.InToOutClassification = ClassLabels(TestDataInMatches')';            
            TestData.Results.ClassProbs = ClassProbs(TestDataInMatches',:);
            
            % Do decision thresholding if required...
            if ~isempty(DecisionThreshold)
                Probs = zeros(size(TestData.Results.InToOutClassification));
                
                for iClass = 1:size(TestData.Results.ClassProbs,2)
                    Probs(TestData.Results.InToOutClassification==iClass) =...
                        TestData.Results.ClassProbs(TestData.Results.InToOutClassification==iClass,iClass);
                    
                    TestData.Results.InToOutClassification(Probs(TestData.Results.InToOutClassification==iClass) < DecisionThreshold) =...
                        -iClass;
                end                                
            end
            
            % DEBUG PLOTTING:
%             figure;
%             hold on;
%             Train = obj.TrainingData1Epoch.Modalities{1}.NormedFeatureVectors';
%             TrainLabels = obj.TrainingData1Epoch.Modalities{1}.ClassLabels(15:16,:);
%             plot(Train(TrainLabels(2,:)==1,1),Train(TrainLabels(2,:)==1,2),'bo');
%             plot(Train(TrainLabels(1,:)==1,1),Train(TrainLabels(1,:)==1,2),'rs');            
%             Test = TestData.Modalities{1}.NormedFeatureVectors';
%             TestLabels = TestData.Modalities{1}.ClassLabels(15:16,:);
%             plot(Test(TestLabels(1,:)==1,1),Test(TestLabels(1,:)==1,2),'rx');            
%             plot(Test(TestLabels(2,:)==1,1),Test(TestLabels(2,:)==1,2),'bx');
%             legend('Training Data: Rolling Objects',...
%                    'Training Data: Non-Rolling Objects',...
%                    ['Test Data: ' TestData.ClassNames{find(TestData.Modalities{1}.ClassLabels(1:14,1))}(9:end)],...
%                    'location', 'northwest');
%             axis([0 1 0 1]);
%             xlabel('Curvature Feature 1');
%             ylabel('Curvature Feature 2');
%             title('Object Property Modality Dims 1 & 2 of Training Data & Test Data');
% 
%             figure;
%             hold on;
%             plot(Codebook(ClassLabels==1,1),Codebook(ClassLabels==1,2),'bo')
%             plot(Codebook(ClassLabels==2,1),Codebook(ClassLabels==2,2),'rs');
%             Test = TestData.Modalities{1}.NormedFeatureVectors';
%             TestLabels = TestData.Modalities{1}.ClassLabels(15:16,:);
%             plot(Test(TestLabels(1,:)==1,1),Test(TestLabels(1,:)==1,2),'rx');
%             plot(Test(TestLabels(2,:)==1,1),Test(TestLabels(2,:)==1,2),'bx');
%             legend('Codebook: Rolling Objects',...
%                    'Codebook: Non-Rolling Objects',...
%                    ['Test Data: ' TestData.ClassNames{find(TestData.Modalities{1}.ClassLabels(1:14,1))}(9:end)],...
%                    'location', 'northwest');
%             axis([0 1 0 1]);
%             xlabel('Curvature Feature 1');
%             ylabel('Curvature Feature 2');
%             title({obj.Name; 'Object Property Modality Dims 1 & 2 of Codebook & Test Data'});

%             subplot(1,2,1);
%             hold off;
%             plot(obj.TrainingData1Epoch.Modalities{1}.NormedFeatureVectors(1,:),...
%                  obj.TrainingData1Epoch.Modalities{1}.NormedFeatureVectors(2,:),'b.');
%             axis([0 1 0 1]);
%             hold on;
%             for iLabel = 1:max(obj.Modalities{1}.ClassLabels)
%                 Colours{iLabel} = [rand rand rand];
%                 plot(obj.Modalities{1}.SOM.codebook(obj.Modalities{1}.ClassLabels==iLabel,1),...
%                      obj.Modalities{1}.SOM.codebook(obj.Modalities{1}.ClassLabels==iLabel,2),...
%                      'x',...
%                      'color', Colours{iLabel});
%             end
%             drawnow;
% 
%             subplot(1,2,2);
%             hold off;
%             plot(obj.TrainingData1Epoch.Modalities{2}.NormedFeatureVectors(1,:),...
%                  obj.TrainingData1Epoch.Modalities{2}.NormedFeatureVectors(2,:),'b.');
%             axis([0 1 0 1]);
%             hold on;
%             plot(obj.Modalities{2}.SOM.codebook(:,1), obj.Modalities{2}.SOM.codebook(:,2), 'rx');
%             ClusterLabels = obj.Modalities{2}.Clustering.Labels{obj.Modalities{2}.Clustering.OptimalK};
%             for iLabel = 1:max(ClusterLabels)                
%                 plot(obj.Modalities{2}.SOM.codebook(ClusterLabels==iLabel,1),...
%                      obj.Modalities{2}.SOM.codebook(ClusterLabels==iLabel,2),...
%                      'x',...
%                      'color', Colours{iLabel});
%             end
%             drawnow;

            % If we were using the internal test data in the
            % class, we should overwrite it with the new results...
            if using_internal_testdata
                obj.TestData = TestData;
            end
                
        end
        
        
        %% ------- *** GROUND TRUTH CLASSIFY *** --------------------------
        %******************************************************************
        % Classify test samples in terms of output modality clusters, then
        % match that classification to ground truth labels.
        %******************************************************************
        function [TestData Mask] = gtclassify(obj, varargin)
            
            % The TestData struct...
            TestData = [];
            
            % A flag that lets us know if we're using the TestData struct
            % in the object, or test data that was passed as an argument...
            using_internal_testdata = false;
            
            % Defaults...
            display = false;
            
            %% CHECK ARGUMENTS --------------------------------------------
            %--------------------------------------------------------------
            % varargin
            i=1; 
            while i<=length(varargin), 
              argok = 1; 
              if ischar(varargin{i}), 
                switch varargin{i}, 
                  % argument IDs
                  case {'data', 'testdata'}, i=i+1; TestDataIn = varargin{i};
                      
                  otherwise, argok=0; 
                end
              elseif isstruct(varargin{i})
                  TestDataIn = varargin{i};
              else
                argok = 0; 
              end
              if ~argok, 
                disp(['BiModalLearner.gtclassify(): Ignoring invalid argument #' num2str(i+1)]); 
              end
              i = i+1; 
            end
            
             %% SET UP TEST DATA ------------------------------------------
            %--------------------------------------------------------------
            if isempty(TestDataIn)
               TestDataIn = obj.TestData;
               using_internal_testdata = true;
            end
            
            %% CLASSIFY IN TERMS OF OUTPUT MOD CLUSTERS -------------------
            %--------------------------------------------------------------
            % if isnumeric(TestData) || ~isfield(TestData, 'Results') || ~isfield(TestData.Results, 'InToOutClassification')
            [TestData Mask] = obj.classify('data', TestDataIn);
            
            %% LABEL THE TRAINED SOMS -------------------------------------
            %--------------------------------------------------------------
            % if ~obj.Modalities{1}.SOM_is_labeled || ~obj.Modalities{2}.SOM_is_labeled
                obj.gtlabelsoms();
            % end
            
            %% MATCH Mod2 CLUSTERS TO GROUND TRUTH ----------------
            % Count the number of ground-truth class labels
            % from the training data that appear in
            % each of the clusters in the Mod2 SOM...
            %--------------------------------------------------------------
            nClusters = obj.Modalities{2}.Clustering.OptimalK;
    
            nGroundTruths = length(obj.TrainingData.Modalities{2}.GroundTruthLabelIndices);
            GroundTruthLabels =...
                obj.TrainingData.Modalities{2}.ClassNames(obj.TrainingData.Modalities{2}.GroundTruthLabelIndices);

            % Row = Cluster
            % Col = Ground truth
            ClusterGroundTruthLabelCounts = zeros(nClusters,nGroundTruths);

            for i = 1:nClusters

                LabelsInThisCluster = obj.Modalities{2}.SOM.labels(obj.Modalities{2}.Clustering.Labels{nClusters} == i, :);

                for j = 1:nGroundTruths                
                    for k = 1:numel(LabelsInThisCluster)

                        if (findstr(LabelsInThisCluster{k}, GroundTruthLabels{j}) == 1)
                        ClusterGroundTruthLabelCounts(i,j) = ClusterGroundTruthLabelCounts(i,j) +...
                            str2double(regexp(LabelsInThisCluster{k}, '(\d*)', 'match'));
                        end
                    end
                end
            end

            % Create SOM cluster to ground truth mapping...
            for i = 1:nClusters
                [LabelCount Cluster_To_GT_Mapping(i,:)] = max(ClusterGroundTruthLabelCounts(i,:));
            end
            
            %% MATCH Mod1-CLASSIFIED TEST DATA TO Mod2-CLUSTER-GROUND-TRUTH --
            % Get the class labels for the
            % Mod1-classified test data
            % from the Mod2 clustering...
            %-----------------------------------------------------------------------
            for i = 1:length(TestData.Results.InToOutClassification)
                ClassifiedTestData_To_ClusterGT_Matches(i,1) =...
                    Cluster_To_GT_Mapping(TestData.Results.InToOutClassification(i));
            end
            
            %% SAVE THE RESULTS IN THE TESTDATA STRUCT ---------------------
            %--------------------------------------------------------------
            TestData.Results.GroundTruthClassification =...
                GroundTruthLabels(ClassifiedTestData_To_ClusterGT_Matches);
            
        end
        
  
        %% ------- *** EVALUATE *** ---------------------------------------
        %******************************************************************
        %******************************************************************
        function TestData = evaluate(obj, varargin)
            
            % Set defaults...
            TrainingData = [];
            TrainingData1Epoch = [];
            
            % The TestData struct...
            TestData = [];
            
            % A flag that lets us know if we're using the TestData struct
            % in the object, or test data that was passed as an argument...
            using_internal_testdata = false;
            
            % Defaults...
            display = false;
            
            %% CHECK ARGUMENTS --------------------------------------------
            %--------------------------------------------------------------
            % varargin
            i=1; 
            while i<=length(varargin), 
              argok = 1; 
              if ischar(varargin{i}), 
                switch varargin{i}, 
                  % argument IDs
                    case {'trainingdata'}, i=i+1; TrainingData = varargin{i};
                    case {'trainingdata1epoch'}, i=i+1; TrainingData1Epoch = varargin{i};
                    case {'testdata'}, i=i+1; TestData = varargin{i};
                    case 'display', i=i+1; DisplaySample = varargin{i}; display = true;

                    otherwise, argok=0; 
                end
              elseif isstruct(varargin{i})
                  TestData = varargin{i};
              else
                argok = 0; 
              end
              if ~argok, 
                disp(['BiModalLearner.evaluate(): Ignoring invalid argument #' num2str(i+1)]); 
              end
              i = i+1; 
            end
            
            %% SET UP TRAINING DATA -------------------------------------------
            %--------------------------------------------------------------
            if isempty(TrainingData)
               TrainingData = obj.TrainingData;
            end
            
            if isempty(TrainingData1Epoch)               
               TrainingData1Epoch = obj.TrainingData1Epoch;
            end
            
            %% SET UP TEST DATA -------------------------------------------
            %--------------------------------------------------------------
            if isempty(TestData)
               TestData = obj.TestData;
               using_internal_testdata = true;
            end
            
            InToOutClassification = TestData.Results.InToOutClassification;
            ClassLabels = TestData.ClassLabels;
            
            %% LABEL THE TRAINED SOMS -------------------------------------
            %--------------------------------------------------------------
            % WARNING: HERE BE DRAGONS!  THIS IF CLAUSE COULD CAUSE
            % MAJOR PROBLEMS FOR ONLINE EVALUATION.  SHOULD ALWAYS RE-LABEL
            % THE SOM AT EACH EVALUATION INTERVAL!!!
            % if ~obj.Modalities{1}.SOM_is_labeled || ~obj.Modalities{2}.SOM_is_labeled
                obj.gtlabelsoms('trainingdata', TrainingData1Epoch);
            % end
            
            %% REMOVE REJECTIONS FROM MOD1 TEST VECTOR CLASSIFICATION ---           
            %--------------------------------------------------------------
            
            TestData.Results.Rejections = sum(InToOutClassification < 0);
            TestData.Results.RejectionPercent =...
                TestData.Results.Rejections / length(InToOutClassification);
            InToOutClassification = InToOutClassification(InToOutClassification >= 0);
            ClassLabels = ClassLabels(:,InToOutClassification >= 0);
            
            %% MATCH Mod2 TEST VECTORS TO Mod2 CLUSTERS ---
            % Use KNN to determine what cluster the Mod2 vector
            % parts of the test samples belong to...
            %--------------------------------------------------------------
            TestDataMod2_To_Cluster_Matches = obj.Modalities{2}.classify(TestData.Modalities{2}, 'clusterwise');
            TestDataMod2_To_Cluster_Matches = TestDataMod2_To_Cluster_Matches(InToOutClassification >= 0, :);                        
            
            %% MATCH Mod2 CLUSTERS TO GROUND TRUTH ----------------
            % Count the number of ground-truth class labels
            % from the training data that appear in
            % each of the clusters in the Mod2 SOM...
            %--------------------------------------------------------------
            nClusters = obj.Modalities{2}.Clustering.OptimalK;
    
            nGroundTruths = length(TrainingData.Modalities{2}.GroundTruthLabelIndices);
            GroundTruthLabels =...
                TrainingData.Modalities{2}.ClassNames(TrainingData.Modalities{2}.GroundTruthLabelIndices);

            % Row = Cluster
            % Col = Ground truth
            ClusterGroundTruthLabelCounts = zeros(nClusters,nGroundTruths);

            for i = 1:nClusters

                LabelsInThisCluster = obj.Modalities{2}.SOM.labels(obj.Modalities{2}.Clustering.Labels{nClusters} == i, :);

                for j = 1:nGroundTruths                
                    for k = 1:numel(LabelsInThisCluster)

                        if (findstr(LabelsInThisCluster{k}, GroundTruthLabels{j}) == 1)
                            TempExpression = regexp(regexp(LabelsInThisCluster{k}, '\(\d*\)', 'match'), '\d*', 'match');
                            ClusterGroundTruthLabelCounts(i,j) = ClusterGroundTruthLabelCounts(i,j) +...
                                str2double(TempExpression{1});
                        end
                    end
                end
            end
            
            % Time to do some Bayesian reasoning...
            % ClusterGroundTruthFactor = struct('var', [1 2], 'card', [nClusters nGroundTruths], 'val', ones(1, nClusters * nGroundTruths));
            % 
            % for iCluster = 1:nClusters
            %     for iGroundTruth = 1:nGroundTruths
            %         ClusterGroundTruthFactor = SetValueOfAssignment(ClusterGroundTruthFactor,...
            %                                                         [iCluster iGroundTruth],...
            %                                                         ClusterGroundTruthLabelCounts(iCluster, iGroundTruth));
            %     end
            % end
            
            % for iCluster = 1:nClusters
            %     Marginal = ComputeMarginal([2], ClusterGroundTruthFactor, [1 iCluster]);
            %     [ConditionalProb Cluster_To_GT_Mapping(iCluster,:)] = max(Marginal.val);
            % end

            % Create SOM cluster to ground truth mapping...
            for i = 1:nClusters
                [LabelCount Cluster_To_GT_Mapping(i,:)] = max(ClusterGroundTruthLabelCounts(i,:));
            end
            
            %% MATCH Mod1-CLASSIFIED TEST DATA TO Mod2-CLUSTER-GROUND-TRUTH --
            % Get the class labels for the
            % Mod1-classified test data
            % from the Mod2 clustering...
            %-----------------------------------------------------------------------
            ClassifiedTestData_To_ClusterGT_Matches = [];
            for i = 1:length(InToOutClassification)
                ClassifiedTestData_To_ClusterGT_Matches(i,1) =...
                    Cluster_To_GT_Mapping(InToOutClassification(i));
            end

            %% FIND THE TEST SAMPLE GROUND TRUTH ---------------------------
            %---------------------------------------------------------------
            [Test_Data_Ground_Truth foo] =...
                find(ClassLabels(TrainingData.Modalities{1}.GroundTruthLabelIndices,:));

            % Test data ground truth repmat'd for all KNN k's up to KNN_max_k...
            Test_Data_Ground_Truth_KNN =...
                repmat(Test_Data_Ground_Truth, 1, obj.Modalities{2}.KNN_max_k);
            
            if display
                    TestSampleInBMUs = obj.Modalities{1}.classify(TestData.Modalities{1});
                    TestSampleOutBMUs = obj.Modalities{2}.classify(TestData.Modalities{2});
                    
                    WinningClusterNodes = find((obj.Modalities{2}.Clustering.Labels{obj.Modalities{2}.Clustering.OptimalK} ==...
                                            InToOutClassification(DisplaySample)));
                                        
                    
                    % 'colourinclustermaps',
                    % TestSampleInBMUs(DisplaySample), 100,...
                
                    obj.display('cluster', 'out',...
                                'classlabels',...
                                'colourinnodeweights', TestSampleInBMUs(DisplaySample),...
                                'markoutnode', TestSampleOutBMUs(DisplaySample),...
                                'markoutclusternodes', WinningClusterNodes,...
                                'groundtruthlabels');
            end
            
            
            %% CALCULATE EVALUATION SCORES ---------------------------------
            %---------------------------------------------------------------
            %---------------------------------------------------------------

            % Tools to work with from the above section...
            %
            %   TestDataMod2_To_Cluster_Matches
            %   Cluster_To_GT_Mapping
            %   ClassifiedTestData_To_ClusterGT_Matches
            %   Test_Data_Ground_Truth
            %   Test_Data_Ground_Truth_KNN
            %

            %% DOES THE Mod2 VECTOR CLUSTER-MATCH REFLECT THE GROUND TRUTH? -----
            % i.e. ignoring the Mod1 vector and finding the best
            % matching cluster for the Mod2 vector (using KNN), does the
            % test sample ground truth correspond to the cluster ground truth?
            %----------------------------------------------------------------------

            % Find the cluster ground truth for the Mod2-to-cluster KNN
            % matches...
            if isempty(TestDataMod2_To_Cluster_Matches)
                Mod2_To_Cluster_GT = [];
                TestData.Results.Mod2_To_Cluster_GT_Matches = [];
            else
                Mod2_To_Cluster_GT = Cluster_To_GT_Mapping(TestDataMod2_To_Cluster_Matches);
                if size(TestDataMod2_To_Cluster_Matches,1) <= 1
                    Mod2_To_Cluster_GT = Mod2_To_Cluster_GT';
                end            

                % Test sample GT to Cluster GT matches for all KNN k's up to KNN_max_k......
                TestData.Results.Mod2_To_Cluster_GT_Matches =...
                    (Test_Data_Ground_Truth_KNN == Mod2_To_Cluster_GT);
            end
            
            % Scores for all KNN k's up to KNN_max_k......
            if isempty(TestData.Results.Mod2_To_Cluster_GT_Matches)
                TestData.Results.Mod2_To_Cluster_GT_Scores = 0;
            else
                TestData.Results.Mod2_To_Cluster_GT_Scores =...
                    sum(TestData.Results.Mod2_To_Cluster_GT_Matches, 1);
            end

            % Percentages for all KNN k's up to KNN_max_k......
%             TestData.Results.Mod2_To_Cluster_GT_Percent =...
%                 TestData.Results.Mod2_To_Cluster_GT_Scores /...
%                     size(TestData.Modalities{2}.FeatureVectors,2);
            TestData.Results.Mod2_To_Cluster_GT_Percent =...
                TestData.Results.Mod2_To_Cluster_GT_Scores /...
                    size(InToOutClassification,2);

            % Best KNN k...
            [TestData.Results.Mod2_To_Cluster_GT_Best_Percent...
                TestData.Results.Mod2_To_Cluster_GT_Best_KNN_k] =...
                max(TestData.Results.Mod2_To_Cluster_GT_Percent);

            %% DOES THE Mod1 VECTOR CLASSIFICATION REFLECT THE GROUND TRUTH? ---
            % i.e. ignoring the Mod2 vector and classifying the Mod1
            % vector, does the test sample ground truth correspond to the cluster
            % ground truth?
            %----------------------------------------------------------------------

            % Test sample GT to Cluster GT matches...
            if isempty(Test_Data_Ground_Truth) || isempty(ClassifiedTestData_To_ClusterGT_Matches)
                TestData.Results.Mod1_To_Cluster_GT_Matches = [];
            else
                TestData.Results.Mod1_To_Cluster_GT_Matches =...
                    (Test_Data_Ground_Truth == ClassifiedTestData_To_ClusterGT_Matches);
            end

            % Score...
            if isempty(TestData.Results.Mod1_To_Cluster_GT_Matches)
                TestData.Results.Mod1_To_Cluster_GT_Score = 0;
            else
                TestData.Results.Mod1_To_Cluster_GT_Score =...
                    sum(TestData.Results.Mod1_To_Cluster_GT_Matches, 1);
            end

            % Percentage...
            % TestData.Results.Mod1_To_Cluster_GT_Percent =...
            %     TestData.Results.Mod1_To_Cluster_GT_Score /...
            %         size(TestData.Modalities{1}.FeatureVectors,2);
            TestData.Results.Mod1_To_Cluster_GT_Percent =...
                TestData.Results.Mod1_To_Cluster_GT_Score /...
                    size(InToOutClassification,2);

            %% DOES CLASSIFIER SELECT CORRECT CLUSTER BASED ON ITS TRAINING? ------
            % i.e. does it, by classifying using the Mod1 test vector, select
            % the Mod2 cluster that best matches the Mod2 test vector?
            %----------------------------------------------------------------------
            % Matches
            if isempty(InToOutClassification) || isempty(TestDataMod2_To_Cluster_Matches)
                TestData.Results.Mod1_To_Cluster_Matches = [];
            else
                TestData.Results.Mod1_To_Cluster_Matches =...
                    repmat(InToOutClassification', 1, obj.Modalities{2}.KNN_max_k) ==...
                        TestDataMod2_To_Cluster_Matches;
            end

            % Scores...
            if isempty(TestData.Results.Mod1_To_Cluster_Matches)
                TestData.Results.Mod1_To_Cluster_Scores = 0;
            else
                TestData.Results.Mod1_To_Cluster_Scores =...
                    sum(TestData.Results.Mod1_To_Cluster_Matches, 1);
            end

            % Percentages...
%             TestData.Results.Mod1_To_Cluster_Percent =...
%                 TestData.Results.Mod1_To_Cluster_Scores /...
%                     size(TestData.Modalities{1}.FeatureVectors,2);
            TestData.Results.Mod1_To_Cluster_Percent =...
                TestData.Results.Mod1_To_Cluster_Scores /...
                    size(InToOutClassification,2);

            % Best KNN k...
            [TestData.Results.Mod1_To_Cluster_Best_Percent...
                TestData.Results.Mod1_To_Cluster_Best_KNN_k] =...
                max(TestData.Results.Mod1_To_Cluster_Percent);

            %% IF CORRECT, DOES THAT CLUSTER CORRESPOND TO THE GROUND TRUTH? ------
            % i.e. ground truth based on no. of ground truth labels that gather
            % in the SOM clusters during training (see Cluster_To_GT_Mapping
            % calculation above)...
            %----------------------------------------------------------------------

            if isempty(TestData.Results.Mod1_To_Cluster_Matches)
                TestData.Results.Mod1_To_Cluster_Matches_Corresponding_To_GT = [];
            else
                Foo = zeros(size(TestData.Results.Mod1_To_Cluster_Matches));
                Bar = repmat(InToOutClassification, 1, obj.Modalities{2}.KNN_max_k);

                Foo(TestData.Results.Mod1_To_Cluster_Matches) =...
                    Cluster_To_GT_Mapping(Bar(TestData.Results.Mod1_To_Cluster_Matches));

                % Matches for all KNN k's up to KNN_max_k...
                TestData.Results.Mod1_To_Cluster_Matches_Corresponding_To_GT =...
                    (Test_Data_Ground_Truth_KNN == Foo);
            end

            % Scores...
            if isempty(TestData.Results.Mod1_To_Cluster_Matches_Corresponding_To_GT)
                TestData.Results.Mod1_To_Cluster_Scores_Corresponding_To_GT = 0;
            else
                TestData.Results.Mod1_To_Cluster_Scores_Corresponding_To_GT =...
                    sum(TestData.Results.Mod1_To_Cluster_Matches_Corresponding_To_GT, 1);
            end

            % Percentages...
%             TestData.Results.Mod1_To_Cluster_Percent_Corresponding_To_GT =...
%                 TestData.Results.Mod1_To_Cluster_Scores_Corresponding_To_GT /...
%                     size(TestData.Modalities{1}.FeatureVectors,2);
            TestData.Results.Mod1_To_Cluster_Percent_Corresponding_To_GT =...
                TestData.Results.Mod1_To_Cluster_Scores_Corresponding_To_GT /...
                    size(InToOutClassification,2);

            % Best KNN k...
            [TestData.Results.Mod1_To_Cluster_Corresponding_To_GT_Best_Percent...
                TestData.Results.Mod1_To_Cluster_Corresponding_To_GT_Best_KNN_k] =...
                max(TestData.Results.Mod1_To_Cluster_Percent_Corresponding_To_GT);
            
            %% COLLATE MAIN EVALUATION SCORES -----------------------------
            %--------------------------------------------------------------
            %--------------------------------------------------------------
            % We use a KNN K of 1 for each of these...
            
            if isempty(TestData.Results.Mod1_To_Cluster_Matches_Corresponding_To_GT)
                TestData.Results.Matches = [];
            else
                TestData.Results.Matches =...
                    TestData.Results.Mod1_To_Cluster_Matches_Corresponding_To_GT(:,1);
            end
            
            TestData.Results.Score =...
                TestData.Results.Mod1_To_Cluster_Scores_Corresponding_To_GT(1);
            
            TestData.Results.Percent =...
                TestData.Results.Mod1_To_Cluster_Percent_Corresponding_To_GT(1);
            
            
            %% GENERATE ROC CURVE DATA ------------------------------------
            %--------------------------------------------------------------
            %--------------------------------------------------------------
            
            %% COLLATE CLUSTERING DATA ------------------------------------
            %--------------------------------------------------------------
            %--------------------------------------------------------------           
            TestData.Results.Mod2_Clustering_OptimalK = obj.Modalities{2}.Clustering.OptimalK;
            TestData.Results.Mod2_Clustering_Centroids = obj.Modalities{2}.Clustering.Centroids{obj.Modalities{2}.Clustering.OptimalK};
            TestData.Results.Mod2_Clustering_Labels = obj.Modalities{2}.Clustering.Labels{obj.Modalities{2}.Clustering.OptimalK};
            
            % If we were using the internal test data in the
            % class, we should overwrite it with the new results...
            if using_internal_testdata
                obj.TestData = TestData;
            end                        
    
        end                
        
        %% ------- *** DISPLAY *** ----------------------------------------
        %******************************************************************
        %******************************************************************
        function display(obj, varargin)
            
            % Default settings...
            Cluster = 'no';
            ClusterInModality = false;            
            DrawInModalityClassLabels = false;
            InClassLabelsToDraw = [];
            ClusterOutModality = false;
            DrawInNodeMaps = false;
            DrawOutNodeMaps = false;
            MapsColour = 'r';
            MapsWeights = false;
            ColourInClusterMaps = false;
            ColourGroundTruthMaps = false;
            ColourGroundTruthMapHistory = false;                        
            GroundTruthLabels = false;
            NodeByNode = false;
            DrawMappings = true;
            DrawVoronoi = false;
            DrawTrainingData = false;
            DrawHebbianProjection = false;
            DrawReverseHebbianProjection = false;
            DrawInActivation = false;
            DrawOutActivation = false;
            DrawCovariances = false;
            CurrentFig = [];
            
            % Loop through arguments...
            i = 1;
            while i <= length(varargin), 
                argok = 1; 
                if ischar(varargin{i}), 
                    switch lower(varargin{i}),
                        case {'figure'}, i=i+1; CurrentFig = varargin{i};
                        case {'cluster', 'clusters'}, i=i+1; Cluster = varargin{i};
                        case {'classlabels'},...
                            i=i+1; InClassLabelsToDraw = varargin{i};
                            DrawInModalityClassLabels = true;
                        case 'k', i=i+1; InKMeansK = varargin{i};
                                         OutKMeansK = varargin{i};
                        case 'ink', i=i+1; InKMeansK = varargin{i};
                        case 'outk', i=i+1; OutKMeansK = varargin{i};
                        case 'colouroutclustermaps', ColourOutClusterMaps = true;
                        case 'drawinnodemaps',...
                            i=i+1; InNode = varargin{i};
                            DrawInNodeMaps = true;
                        case 'drawoutnodemaps',...
                            i=i+1; OutNode = varargin{i};
                            DrawOutNodeMaps = true;
                        case {'mapscolor', 'mapscolour'},...
                            i=i+1; MapsColour = varargin{i};
                        case {'mapsweight', 'mapsweights'},...
                            i=i+1; MapsWeights = varargin{i};
                        case 'colourinclustermaps',
                            i=i+1; InNode = varargin{i};
                            i=i+1; LastNCoOccurrences = varargin{i};
                            ColourInClusterMaps = true;
                        case {'colourgroundtruthmaps' 'colourgtmaps'},...
                            ColourGroundTruthMaps = true;
                        case {'colourgtmapshistory' 'colourgtmaphistory'},...
                            ColourGroundTruthMapHistory = true;
                        case {'markinnode', 'markinnodes'},
                            i=i+1; InNodesToMark = varargin{i};
                        case {'markoutnode', 'markoutnodes'},
                            i=i+1; OutNodesToMark = varargin{i};
                        case 'markoutclusternodes', i=i+1; WinningClusterNodes = varargin{i};
                        case 'groundtruthlabels', GroundTruthLabels = true;
                        case 'nodebynode', NodeByNode = true;
                        case {'nomappings', 'nomaps'}, DrawMappings = false;
                        case {'voronoi', 'drawvoronoi'}, DrawVoronoi = true;
                        case {'drawtrainingdata', 'trainingdata', 'drawdata',...
                              'showtrainingdata', 'showdata'}, DrawTrainingData = true;
                        case {'drawhebbianprojection', 'hebbianprojection'},...
                            i=i+1; HebbianProjectionInNode = varargin{i};
                            DrawHebbianProjection = true;
                        case {'drawreversehebbianprojection', 'reversehebbianprojection'},...
                            i=i+1; ReverseHebbianProjectionInNode = varargin{i};
                            DrawReverseHebbianProjection = true;
                        case {'drawinactivation', 'inactivation'},...
                            i=i+1; InActivationSample = varargin{i};
                            DrawInActivation = true;
                        case {'drawoutactivation', 'outactivation'},...
                            i=i+1; OutActivationSample = varargin{i};
                            DrawOutActivation = true;
                        case {'drawcovariances', 'covariances',...
                              'drawcov', 'cov'},...
                            DrawCovariances = true;                            
                            
                        
                        otherwise
                            argok = 0;
                    end
                else
                    argok = 0;
                end

                if ~argok, 
                    disp(['BiModalLearner.display(): Ignoring invalid argument #' num2str(i)]);
                    % fprintf(UsageMessage);
                end

                i = i + 1;
            end
            
            % Set up the figure...            
            if isempty(CurrentFig)
                CurrentFig = figure;
                hold on;
            else
                figure(CurrentFig);
            end                        
            
            % Do we need to cluster over the modalities?
            switch lower(Cluster)
                case {'in', 'inmodality'}, ClusterInModality = true;
                case {'out', 'outmodality'}, ClusterOutModality = true;
                case {'both'}, ClusterInModality = true;
                               ClusterOutModality = true;
            end
            
            %% DRAW THE IN MODALITY... ------------------------------------
            %--------------------------------------------------------------
            if ClusterInModality
                % Cluster the map if necessary...
                if isempty(obj.Modalities{1}.Clustering)
                	obj.Modalities{1} = obj.Modalities{1}.cluster();
                end
                
                WhatToDisplay = obj.Modalities{1}.Clustering.Labels{obj.Modalities{1}.Clustering.OptimalK};                                
                                
                % Change colormap...
                % colormap(jet(KMeansK));
                
            elseif DrawInModalityClassLabels
                
                WhatToDisplay = obj.Modalities{1}.ClassLabels';
                                
            else
                % Get U-Matrix...
                U = som_umat(obj.Modalities{1}.SOM);
                
                WhatToDisplay = U(:);
                                
            end
            
            % Display the map...
            if DrawVoronoi && size(obj.Modalities{1}.SOM.codebook,2) == 2

                % Use the MPT toolbox to get a bounded Voronoi
                % diagram...
                Options.pbound = polytope([0 0; 0 1; 1 0; 1 1]);
                V = mpt_voronoi(obj.Modalities{1}.SOM.codebook, Options);

                Options.extreme_solver = 0;
                Colours = ['g' 'b' 'r' 'w' 'y' 'm' 'c'];
                
                if DrawReverseHebbianProjection
                    ReverseHebbianProjOutWeights = obj.crossMapping.Weights(ReverseHebbianProjectionInNode, :);
                    
                    ReverseHebbianProjection = obj.crossMapping.Weights * ReverseHebbianProjOutWeights';
                    % ReverseHebbianProjection = ReverseHebbianProjection ./ norm(ReverseHebbianProjection,1);
                    ReverseHebbianProjection = ReverseHebbianProjection ./ max(ReverseHebbianProjection);
                    
                elseif DrawInActivation                                        
                    InActivations = obj.Modalities{1}.findactivations(InActivationSample);
                    InActivations = InActivations ./ max(InActivations);
                                        
                end
                
                % Draw each cell...
                for iCell = 1:size(V,2)                    
                    VoronoiCells{iCell} = extreme(V(iCell), Options);
                    VoronoiCells{iCell} = [VoronoiCells{iCell} ones(size(VoronoiCells{iCell},1),1)];                                        
                    
                    if DrawReverseHebbianProjection                        
                        CellFaceAlpha = ReverseHebbianProjection(iCell);
                        CellFaceColour = 'r';
                        FaceColour = 'r';
                    elseif DrawInActivation
                        CellFaceAlpha = InActivations(iCell);
                        CellFaceColour = 'r';
                        FaceColour = 'r';
                    elseif ClusterInModality
                        CellFaceAlpha = 0.5;
                        CellFaceColour = Colours(obj.Modalities{1}.Clustering.Labels{obj.Modalities{1}.Clustering.OptimalK}(iCell));
                        FaceColour = Colours(obj.Modalities{1}.Clustering.Labels{obj.Modalities{1}.Clustering.OptimalK}(iCell));
                    elseif DrawInModalityClassLabels && sum(InClassLabelsToDraw == iCell) > 0
                        CellFaceAlpha = 0.5;
                        CellFaceColour = Colours(obj.Modalities{1}.ClassLabels(iCell));
                        FaceColour = Colours(obj.Modalities{1}.ClassLabels(iCell));
                    else
                        CellFaceAlpha = 0;
                        CellFaceColour = 'w';
                        FaceColour = 'none';
                    end
                    
                    patch(VoronoiCells{iCell}(:,1), VoronoiCells{iCell}(:,2), VoronoiCells{iCell}(:,3),...
                          CellFaceColour, 'FaceColor', FaceColour, 'FaceAlpha', CellFaceAlpha);
                                                            
                end
                
                % Draw covariance ellipses...
                if DrawCovariances
                    
                    for iNode = 1:size(obj.Modalities{1}.SOM.codebook,1)
                    
                        NumPoints = 100;
                        Mean = obj.Modalities{1}.SOM.codebook(iNode,:)';
                        Covariances = obj.Modalities{1}.Cov{iNode};

                        theta = (0:1:NumPoints-1)/(NumPoints-1)*2*pi;

                        epoints = sqrtm(Covariances) * [cos(theta); sin(theta)]*1   + Mean*ones(1,NumPoints);
                        epoints(3,:) = 1;
                        
                        plot3(epoints(1,:),epoints(2,:),epoints(3,:),'r','LineWidth',1);
                        
                    end
                    
                end

            elseif ~DrawTrainingData
                
                CrossMod_som_cplane(obj.Modalities{1}.SOM.topol.lattice,...
                                    obj.Modalities{1}.SOM.topol.msize,...
                                    WhatToDisplay, 1, [1 1 1]);
            end
            
            % Draw data points...
            if DrawTrainingData
                Data = obj.TrainingData1Epoch.Modalities{1}.NormedFeatureVectors';
                Data = [Data ones(size(Data,1),1)];
                
                Colours = ['b' 'g' 'r' 'k' 'y' 'm' 'c'];
                for iClass = 1:size(obj.TrainingData1Epoch.Modalities{2}.ClassLabels,1)
                    ClassIndices = logical(obj.TrainingData1Epoch.Modalities{2}.ClassLabels(iClass,:));                    
                    % plot3(Data(ClassIndices,1), Data(ClassIndices,2), Data(ClassIndices,3), ['.' Colours(iClass)]);
                    plot3(Data(ClassIndices,1), Data(ClassIndices,2), ones(size(Data(ClassIndices,1),1),1), ['.' Colours(iClass)]);
                end
            end
            
            %% DRAW THE OUT MODALITY... -----------------------------------
            %--------------------------------------------------------------
            if ClusterOutModality
                % Cluster the map if necessary...
                if isempty(obj.Modalities{2}.Clustering)
                	obj.Modalities{2} = obj.Modalities{2}.cluster();
                end
                
                WhatToDisplay = obj.Modalities{2}.Clustering.Labels{obj.Modalities{2}.Clustering.OptimalK};                                
                                
                % Change colormap...
                % colormap(jet(KMeansK));
                
            else
                % Get U-Matrix...
                U = som_umat(obj.Modalities{2}.SOM);
                
                WhatToDisplay = U(:);
                                
            end
            
            % Display the map...
            if DrawVoronoi && size(obj.Modalities{1}.SOM.codebook,2) == 2

                % Use the MPT toolbox to get a bounded Voronoi
                % diagram...
                Options.pbound = polytope([0 0; 0 1; 1 0; 1 1]);
                V = mpt_voronoi(obj.Modalities{2}.SOM.codebook, Options);

                Options.extreme_solver = 0;
                Colours = ['g' 'b' 'r' 'k' 'y' 'm' 'c'];
                
                if DrawHebbianProjection                    
                    HebbianProjection = obj.crossMapping.Weights(HebbianProjectionInNode, :);
                    HebbianProjection = HebbianProjection ./ max(HebbianProjection);
                    
                elseif DrawOutActivation
                    OutActivations = obj.Modalities{2}.findactivations(OutActivationSample);
                    OutActivations = OutActivations ./ max(OutActivations);
                    
                end
                
                for iCell = 1:size(V,2)                    
                    VoronoiCells{iCell} = extreme(V(iCell), Options);
                    VoronoiCells{iCell} = [VoronoiCells{iCell} 10 * ones(size(VoronoiCells{iCell},1),1)];
                    
                    if DrawHebbianProjection                        
                        CellFaceAlpha = HebbianProjection(iCell);
                        CellFaceColour = 'r';
                        FaceColour = 'r';
                    elseif DrawOutActivation
                        CellFaceAlpha = OutActivations(iCell);
                        CellFaceColour = 'r';
                        FaceColour = 'r';
                    elseif ClusterOutModality
                        CellFaceAlpha = 0.5;
                        CellFaceColour = Colours(obj.Modalities{2}.Clustering.Labels{obj.Modalities{2}.Clustering.OptimalK}(iCell));
                        FaceColour = Colours(obj.Modalities{2}.Clustering.Labels{obj.Modalities{2}.Clustering.OptimalK}(iCell));
                    else
                        CellFaceAlpha = 0;
                        CellFaceColour = 'w';
                        FaceColour = 'none';
                    end
                    
                    patch(VoronoiCells{iCell}(:,1), VoronoiCells{iCell}(:,2), VoronoiCells{iCell}(:,3),...
                          CellFaceColour, 'FaceColor', FaceColour, 'FaceAlpha', CellFaceAlpha);
                end
                
                % Draw covariance ellipses...
                if DrawCovariances
                    
                    for iNode = 1:size(obj.Modalities{2}.SOM.codebook,1)
                    
                        NumPoints = 100;
                        Mean = obj.Modalities{2}.SOM.codebook(iNode,:)';
                        Covariances = obj.Modalities{2}.Cov{iNode};

                        theta = (0:1:NumPoints-1)/(NumPoints-1)*2*pi;

                        epoints = sqrtm(Covariances) * [cos(theta); sin(theta)]*1   + Mean*ones(1,NumPoints);
                        epoints(3,:) = 10;
                        
                        plot3(epoints(1,:),epoints(2,:),epoints(3,:),'r','LineWidth',1);
                        
                    end
                    
                end
                
            elseif ~DrawTrainingData
                
                CrossMod_som_cplane(obj.Modalities{1}.SOM.topol.lattice,...
                                    obj.Modalities{1}.SOM.topol.msize,...
                                    WhatToDisplay, 1, [1 1 10]);
            end
            
            % Draw data points...
            if DrawTrainingData
                Data = obj.TrainingData1Epoch.Modalities{2}.NormedFeatureVectors';
                Data = [Data (10 * ones(size(Data,1),1))];
                
                Colours = ['b' 'g' 'r' 'k' 'y' 'm' 'c'];
                for iClass = 1:size(obj.TrainingData1Epoch.Modalities{2}.ClassLabels,1)
                    ClassIndices = logical(obj.TrainingData1Epoch.Modalities{2}.ClassLabels(iClass,:));
                    % plot3(Data(ClassIndices,1), Data(ClassIndices,2), Data(ClassIndices,3), ['.' Colours(iClass)]);
                    plot3(Data(ClassIndices,1), Data(ClassIndices,2), 10*ones(size(Data(ClassIndices,1),1),1), ['.' Colours(iClass)]);
                end
                                
            end
            
            %    colormap(jet(i)), som_recolorbar % change colormap
                            
                            
            %% DRAW CROSS-MODAL MAPPINGS -------------------------------
            %--------------------------------------------------------------
            if DrawMappings
                
                if DrawVoronoi
                    inCoords = obj.Modalities{1}.SOM.codebook;
                    outCoords = obj.Modalities{2}.SOM.codebook;
                elseif ~DrawTrainingData
                    inCoords = som_vis_coords(obj.Modalities{1}.SOM.topol.lattice, obj.Modalities{1}.SOM.topol.msize);
                    outCoords = som_vis_coords(obj.Modalities{2}.SOM.topol.lattice, obj.Modalities{2}.SOM.topol.msize);
                else
                    inCoords = obj.TrainingData1Epoch.Modalities{1}.NormedFeatureVectors(1:2,:)';
                    outCoords = obj.TrainingData1Epoch.Modalities{2}.NormedFeatureVectors(1:2,:)';
                end

                
                if DrawInNodeMaps && InNode == Inf
                    UniqueInNodes = unique(obj.crossMapping.CoOccurrences(:,1));

                    for i = 1:length(UniqueInNodes)

                        inMatches = obj.crossMapping.CoOccurrences(obj.crossMapping.CoOccurrences(:,1)==UniqueInNodes(i),1);
                        inMatchCoords = inCoords(inMatches,:);
                        outMatches = obj.crossMapping.CoOccurrences(obj.crossMapping.CoOccurrences(:,1)==UniqueInNodes(i),2);
                        outMatchCoords = outCoords(outMatches,:);
                        inZ = ones(1,size(inMatchCoords,1));
                        outZ = 10 * ones(1,size(inMatchCoords,1));

                        X = [inMatchCoords(:,1)'; outMatchCoords(:,1)'; nan(1,size(inMatchCoords,1))];
                        Y = [inMatchCoords(:,2)'; outMatchCoords(:,2)'; nan(1,size(inMatchCoords,1))];
                        Z = [inZ; outZ; nan(1,size(inMatchCoords,1))];

                        UniqueOutMatches = unique(outMatches);
                        RandR = rand;
                        RandG = rand;
                        RandB = rand;


                        if NodeByNode
                            for j = 1:length(UniqueOutMatches)
                                h = plot3( reshape( X, [1 3*size(X,2)]), reshape( Y, [1 3*size(Y,2)]), reshape( Z, [1 3*size(Z,2)]),...
                                           'color', [RandR RandG RandB],...
                                           'LineWidth', sum(outMatches == UniqueOutMatches(j)));
                            end
                            fprintf('Press a key to see the next node mappings...');
                            pause;
                            clf(h, 'reset');
                        else
                            plot3( reshape( X, [1 3*size(X,2)]), reshape( Y, [1 3*size(Y,2)]), reshape( Z, [1 3*size(Z,2)]), 'color', [rand rand rand] );
                        end
                    end

                elseif DrawOutNodeMaps && OutNode == Inf
                    UniqueOutNodes = unique(obj.crossMapping.CoOccurrences(:,2));

                    for i = 1:length(UniqueOutNodes)

                        inMatchCoords = inCoords(obj.crossMapping.CoOccurrences(find(obj.crossMapping.CoOccurrences(:,2)==UniqueOutNodes(i)),1),:);
                        outMatchCoords = outCoords(obj.crossMapping.CoOccurrences(find(obj.crossMapping.CoOccurrences(:,2)==UniqueOutNodes(i)),2),:);
                        inZ = ones(1,size(inMatchCoords,1));
                        outZ = 10 * ones(1,size(inMatchCoords,1));

                        X = [inMatchCoords(:,1)'; outMatchCoords(:,1)'; nan(1,size(inMatchCoords,1))];
                        Y = [inMatchCoords(:,2)'; outMatchCoords(:,2)'; nan(1,size(inMatchCoords,1))];
                        Z = [inZ; outZ; nan(1,size(inMatchCoords,1))];

                        if NodeByNode
                            h = plot3( reshape( X, [1 3*size(X,2)]), reshape( Y, [1 3*size(Y,2)]), reshape( Z, [1 3*size(Z,2)]), 'color', [rand rand rand] );
                            fprintf('Press a key to see the next node mappings...');
                            pause;
                            clf(h, 'reset');
                        else
                            plot3( reshape( X, [1 3*size(X,2)]), reshape( Y, [1 3*size(Y,2)]), reshape( Z, [1 3*size(Z,2)]), 'color', [rand rand rand] );
                        end
                    end
                    
                elseif DrawInNodeMaps
                    
                    inMatchCoords = repmat(inCoords(InNode,:), size(outCoords,1),1);
                    outMatchCoords = outCoords;
                    inZ = ones(1,size(inMatchCoords,1));
                    outZ = 10 * ones(1,size(inMatchCoords,1));
                    
                    X = [inMatchCoords(:,1)'; outMatchCoords(:,1)'; nan(1,size(inMatchCoords,1))];
                    Y = [inMatchCoords(:,2)'; outMatchCoords(:,2)'; nan(1,size(inMatchCoords,1))];
                    Z = [inZ; outZ; nan(1,size(inMatchCoords,1))];
                    
                    LineWeights = 10 * (obj.crossMapping.Weights(InNode,:) / max(obj.crossMapping.Weights(InNode,:)));
                    
                    % plot3( reshape( X, [1 3*size(X,2)]), reshape( Y, [1 3*size(Y,2)]), reshape( Z, [1 3*size(Z,2)]), 'r', 'LineWidth', ones(100));
                    for i = 1:length(X)
                        if MapsWeights
                            if LineWeights(i) > 0
                                
                                set(0,'DefaultFigureRenderer','OpenGL');
                        
                                r = patch(X(1:2,i),Y(1:2,i),Z(1:2,i),'red');
                                set(r,'EdgeColor',MapsColour)
                                set(r,'facealpha',0.2);
                                set(r,'edgealpha',0.2);
                                set(r,'LineWidth',LineWeights(i))

                                axis([0 1 0 1 1 10 0 1]);                                                                
                                
                                % plot3( X(1:2,i), Y(1:2,i), Z(1:2,i), MapsColour, 'LineWidth', LineWeights(i));
                            end
                        else
                            set(0,'DefaultFigureRenderer','OpenGL');
                        
                            r = patch(X(1:2,i),Y(1:2,i),Z(1:2,i),'red');
                            set(r,'EdgeColor',MapsColour)
                            set(r,'facealpha',0.2);
                            set(r,'edgealpha',0.2);                    

                            axis([0 1 0 1 1 10 0 1]);

                            % plot3( X(1:2,i), Y(1:2,i), Z(1:2,i), MapsColour);
                        end
                    end
                    
                elseif ColourInClusterMaps
                    
                    InWinningCluster = obj.Modalities{1}.Clustering.Labels{obj.Modalities{1}.Clustering.OptimalK}(InNode);
                    
                    InWinningClusterNodes = find((obj.Modalities{1}.Clustering.Labels{obj.Modalities{1}.Clustering.OptimalK} == InWinningCluster));
                    
                    % Find most recent co-occurences...
                    RecentCoOccurrences = obj.crossMapping.CoOccurrences(end-LastNCoOccurrences-1:end,:);

                    % Find the ones that match the in-winning
                    % cluster...                                
                    [RecentCoOccurenceMatches InWinningClusterMatches] = find(repmat(RecentCoOccurrences(:,1),1,size(InWinningClusterNodes,1)) ==...
                        repmat(InWinningClusterNodes', size(RecentCoOccurrences,1),1));
                    
                    inMatchCoords = inCoords(RecentCoOccurrences(RecentCoOccurenceMatches,1),:);
                    outMatchCoords = outCoords(RecentCoOccurrences(RecentCoOccurenceMatches,2),:);
                    inZ = ones(1,size(inMatchCoords,1));
                    outZ = 10 * ones(1,size(inMatchCoords,1));

                    X = [inMatchCoords(:,1)'; outMatchCoords(:,1)'; nan(1,size(inMatchCoords,1))];
                    Y = [inMatchCoords(:,2)'; outMatchCoords(:,2)'; nan(1,size(inMatchCoords,1))];
                    Z = [inZ; outZ; nan(1,size(inMatchCoords,1))];

                    set(0,'DefaultFigureRenderer','OpenGL');
                        
                    r = patch(X,Y,Z,'red');
                    set(r,'EdgeColor',Colours(iClass))
                    set(r,'facealpha',0.01);
                    set(r,'edgealpha',0.01);                    
                        
                    axis([0 1 0 1 1 10 0 1]);
                    
                    % plot3( reshape( X, [1 3*size(X,2)]), reshape( Y, [1 3*size(Y,2)]), reshape( Z, [1 3*size(Z,2)]), 'r' );
                
                elseif ColourGroundTruthMaps || ColourGroundTruthMapHistory
                    
                    inMatchCoords = inCoords(obj.crossMapping.CoOccurrences(:,1),:);
                    outMatchCoords = outCoords(obj.crossMapping.CoOccurrences(:,2),:);
                    
                    Colours = ['y' 'm' 'c' 'r' 'g' 'b'];
                    iColour = 1;
                                
                    for i = 1:obj.TrainingData.Modalities{1}.nGroundTruths
                        
                        GroundTruthIndices = find(obj.TrainingData.Modalities{1}.ClassLabels(...
                                                        obj.TrainingData.Modalities{1}.GroundTruthLabelIndices(i),:));
                                                    
                        inZ = ones(1,size(GroundTruthIndices,2));
                        outZ = 10 * ones(1,size(GroundTruthIndices,2));
                        
                        X = [inMatchCoords(GroundTruthIndices,1)'; outMatchCoords(GroundTruthIndices,1)'; nan(1,size(find(GroundTruthIndices),2))];
                        Y = [inMatchCoords(GroundTruthIndices,2)'; outMatchCoords(GroundTruthIndices,2)'; nan(1,size(find(GroundTruthIndices),2))];
                        Z = [inZ; outZ; nan(1,size(find(GroundTruthIndices),2))];

                        iColour = iColour + 1;
                        if iColour > length(Colours)
                            iColour = 1;
                        end
                        
                        if ColourGroundTruthMapHistory
                            for j = 1:length(X)
                                LineWidth = ceil((5 / size(obj.TrainingData.Modalities{1}.ClassLabels,2)) * GroundTruthIndices(j));
                                plot3( X(1:2,j), Y(1:2,j), Z(1:2,j), Colours(iColour), 'LineWidth', LineWidth);
                            end
                        else
                            plot3( reshape( X, [1 3*size(X,2)]), reshape( Y, [1 3*size(Y,2)]), reshape( Z, [1 3*size(Z,2)]), Colours(iColour));
                        end
                    end
                    
                elseif DrawVoronoi
                    
                    inMatchCoords = inCoords(obj.crossMapping.CoOccurrences(:,1),:);
                    outMatchCoords = outCoords(obj.crossMapping.CoOccurrences(:,2),:);                                                           
                    inZ = ones(1,size(inMatchCoords,1));
                    outZ = 10 * ones(1,size(inMatchCoords,1));

                    X = [inMatchCoords(:,1)'; outMatchCoords(:,1)'; nan(1,size(inMatchCoords,1))];
                    Y = [inMatchCoords(:,2)'; outMatchCoords(:,2)'; nan(1,size(inMatchCoords,1))];
                    Z = [inZ; outZ; nan(1,size(inMatchCoords,1))];

                    set(0,'DefaultFigureRenderer','OpenGL');
                        
                    r = patch(X,Y,Z,'red');
                    set(r,'EdgeColor','r')
                    set(r,'facealpha',0.2);
                    set(r,'edgealpha',0.2);                    
                        
                    axis([0 1 0 1 1 10 0 1]);
                    
                    % plot3( reshape( X, [1 3*size(X,2)]), reshape( Y, [1 3*size(Y,2)]), reshape( Z, [1 3*size(Z,2)]), 'r' );
                    
                else
                    
                    Colours = ['b' 'g' 'r' 'k' 'y' 'm' 'c'];
                    for iClass = 1:size(obj.TrainingData1Epoch.Modalities{2}.ClassLabels,1)
                        ClassIndices = logical(obj.TrainingData1Epoch.Modalities{2}.ClassLabels(iClass,:));
                        
                        inMatchCoords = inCoords(ClassIndices,:);
                        outMatchCoords = outCoords(ClassIndices,:);
                        inZ = ones(1,size(inMatchCoords,1));
                        outZ = 10 * ones(1,size(inMatchCoords,1));

                        X = [inMatchCoords(:,1)'; outMatchCoords(:,1)'; nan(1,size(inMatchCoords,1))];
                        Y = [inMatchCoords(:,2)'; outMatchCoords(:,2)'; nan(1,size(inMatchCoords,1))];
                        Z = [inZ; outZ; nan(1,size(inMatchCoords,1))];
                        
                        set(0,'DefaultFigureRenderer','OpenGL');
                        
                        r = patch(X,Y,Z,'red');
                        set(r,'EdgeColor',Colours(iClass))
                        set(r,'facealpha',0.2);
                        set(r,'edgealpha',0.2);
                        
                        % Draw boxes around the unit squares in each
                        % modality...
                        plot3([0; 0], [0; 1], [10; 10], 'k');
                        plot3([0; 1], [0; 0], [10; 10], 'k');
                        plot3([1; 1], [1; 0], [10; 10], 'k');
                        plot3([1; 0], [1; 1], [10; 10], 'k');
                        
                        plot3([0; 0], [0; 1], [1; 1], 'k');
                        plot3([0; 1], [0; 0], [1; 1], 'k');
                        plot3([1; 1], [1; 0], [1; 1], 'k');
                        plot3([1; 0], [1; 1], [1; 1], 'k');
                        
                        axis([0 1 0 1 1 10 0 1]);
                        
                        % plot3( reshape( X, [1 3*size(X,2)]), reshape( Y, [1 3*size(Y,2)]), reshape( Z, [1 3*size(Z,2)]), [Colours(iClass) '--'] );                                                
                    end
                                        
                end
            end
            
            %% MARK NODES -------------------------------------------------
            %--------------------------------------------------------------
            if exist('InNodesToMark', 'var')
                plot3(inCoords(InNodesToMark, 1), inCoords(InNodesToMark, 2), ones(size(inCoords(InNodesToMark,1))), 'r+');
            end
            
            if exist('OutNodesToMark', 'var')
                plot3(outCoords(OutNodesToMark, 1), outCoords(OutNodesToMark, 2), (10 * ones(size(outCoords(OutNodesToMark,1)))) + 0.1, 'r+');
            end
            
            %% MARK WINNING CLUSTER NODES ---------------------------------
            %--------------------------------------------------------------
            if exist('WinningClusterNodes', 'var')
                for i = 1:length(WinningClusterNodes)
                    plot3(outCoords(WinningClusterNodes(i), 1), outCoords(WinningClusterNodes(i), 2), 10 + 0.1, 'g+');
                end
            end
            
            %% DISPLAY GROUND TRUTH LABELS --------------------------------
            %--------------------------------------------------------------
            if GroundTruthLabels
                
                % In modality...
                nGroundTruths = length(obj.TrainingData.Modalities{1}.GroundTruthLabelIndices);
                GroundTruthLabels =...
                    obj.TrainingData.Modalities{1}.ClassNames(obj.TrainingData.Modalities{1}.GroundTruthLabelIndices);
                
                % Layers of text for the labels...
                Layers = ones(1,size(obj.Modalities{1}.SOM.labels,1)) - 0.5;
                
                for iGroundTruth = 1:nGroundTruths
                    for inode = 1:size(obj.Modalities{1}.SOM.labels,1)
                        for j = 1:size(obj.Modalities{1}.SOM.labels(inode,:),2)

                            if (findstr(obj.Modalities{1}.SOM.labels{inode,j}, GroundTruthLabels{iGroundTruth}) == 1)
                                
                                foo = regexp(obj.Modalities{1}.SOM.labels{inode,j}, '\w*:\s|\W\d*\W', 'split');
                                ClassWords = regexp(foo{2}, '\W*', 'split');
                                
                                Text = '';
                                
                                for iWords = 1:length(ClassWords)
                                    Text = strcat(Text, upper(ClassWords{iWords}(1)));
                                end
                                
                                Text = strcat(Text, '(', regexp(obj.Modalities{1}.SOM.labels{inode,j}, '(\d*)', 'match'), ')');
                                
                                % We found a ground truth label in the
                                % input modality SOM...
%                                 if iGroundTruth == 1
%                                     Text = strcat('NR(', regexp(obj.Modalities{1}.SOM.labels{inode,j}, '(\d*)', 'match'), ')');
%                                 else
%                                     Text = strcat('R(', regexp(obj.Modalities{1}.SOM.labels{inode,j}, '(\d*)', 'match'), ')');
%                                 end
                                
                                text(inCoords(inode,1), inCoords(inode,2), Layers(inode), Text, 'color', 'w');
                                
                                % Next time draw it lower...
                                Layers(inode) = Layers(inode) - 0.5;
                                % Height = Height - 0.5;
                            end
                        end
                    end
                end
                
                % Out modality...
                nGroundTruths = length(obj.TrainingData.Modalities{2}.GroundTruthLabelIndices);
                GroundTruthLabels =...
                    obj.TrainingData.Modalities{2}.ClassNames(obj.TrainingData.Modalities{2}.GroundTruthLabelIndices);
                
                % Layers of text for the labels...
                Layers = (10 * ones(1,size(obj.Modalities{2}.SOM.labels,1))) + 0.5;
                
                for iGroundTruth = 1:nGroundTruths
                    for inode = 1:size(obj.Modalities{2}.SOM.labels,1)
                        for j = 1:size(obj.Modalities{2}.SOM.labels(inode,:),2)

                            if (findstr(obj.Modalities{2}.SOM.labels{inode,j}, GroundTruthLabels{iGroundTruth}) == 1)
                                
                                foo = regexp(obj.Modalities{2}.SOM.labels{inode,j}, '\w*:\s|\W\d*\W', 'split');
                                ClassWords = regexp(foo{2}, '\W*', 'split');
                                
                                Text = '';
                                
                                for iWords = 1:length(ClassWords)
                                    Text = strcat(Text, upper(ClassWords{iWords}(1)));
                                end
                                
                                Text = strcat(Text, '(', regexp(obj.Modalities{2}.SOM.labels{inode,j}, '(\d*)', 'match'), ')');
                                
%                                 if iGroundTruth == 1
%                                     Text = strcat('NR(', regexp(obj.Modalities{2}.SOM.labels{inode,j}, '(\d*)', 'match'), ')');
%                                 else
%                                     Text = strcat('R(', regexp(obj.Modalities{2}.SOM.labels{inode,j}, '(\d*)', 'match'), ')');
%                                 end
                                
                                text(outCoords(inode,1), outCoords(inode,2), Layers(inode), Text, 'color', 'r');
                                
                                % Next time draw it height...                               
                                Layers(inode) = Layers(inode) + 0.5;
                                
                            end
                            
                        end
                    end
                end
                
            end
            
            view(3);
        end
        
    end
        
    methods (Access = private)
        
        %% ------- *** LABEL MODALITY SOM MAPS *** ---------------
        %******************************************************************
        %******************************************************************
        
        function gtlabelsoms(obj, varargin)
            
            % Set defaults...
            TrainingData = [];                        
            
            %% CHECK ARGUMENTS --------------------------------------------
            %--------------------------------------------------------------
            % varargin
            i=1; 
            while i<=length(varargin), 
              argok = 1; 
              if ischar(varargin{i}), 
                switch varargin{i}, 
                  % argument IDs
                    case {'trainingdata'}, i=i+1; TrainingData = varargin{i};

                    otherwise, argok=0; 
                end
              elseif isstruct(varargin{i})
                  TestData = varargin{i};
              else
                argok = 0; 
              end
              if ~argok, 
                disp(['BiModalLearner.gtlabelsoms(): Ignoring invalid argument #' num2str(i+1)]); 
              end
              i = i+1; 
            end
            
            %% SET UP TRAINING DATA -------------------------------------------
            %--------------------------------------------------------------
            if isempty(TrainingData)
               TrainingData = obj.TrainingData;
            end
            
            % Mod1 SOM...
            InSOMData = som_data_struct(TrainingData.Modalities{1}.NormedFeatureVectors',...
                                        'comp_names', TrainingData.Modalities{1}.FeatureNames,...
                                        'label_names', TrainingData.Modalities{1}.ClassNames);
            
            % Add labels to the SOM training data struct...
            for i=1:size(InSOMData.labels, 1)
                tagIndex = 1;
                for j=1:length(TrainingData.Modalities{1}.ClassNames)
                    if TrainingData.Modalities{1}.ClassLabels(j,i) == 1
                        InSOMData.labels{i,tagIndex} =...
                            TrainingData.Modalities{1}.ClassNames{j};
                        tagIndex = tagIndex + 1;
                    end
                end
            end
            
            obj.Modalities{1}.SOM = som_autolabel(obj.Modalities{1}.SOM, InSOMData, 'freq');
            obj.Modalities{1}.SOM_is_labeled = true;
            
            % Mod2 SOM...
            OutSOMData = som_data_struct(TrainingData.Modalities{2}.NormedFeatureVectors',...
                                        'comp_names', TrainingData.Modalities{2}.FeatureNames,...
                                        'label_names', TrainingData.Modalities{2}.ClassNames);
            
            for i=1:size(OutSOMData.labels, 1)
                tagIndex = 1;
                for j=1:length(TrainingData.Modalities{2}.ClassNames)
                    if TrainingData.Modalities{2}.ClassLabels(j,i) == 1
                        OutSOMData.labels{i,tagIndex} =...
                            TrainingData.Modalities{2}.ClassNames{j};
                        tagIndex = tagIndex + 1;
                    end
                end
            end
            
            obj.Modalities{2}.SOM = som_autolabel(obj.Modalities{2}.SOM, OutSOMData, 'freq');
            obj.Modalities{2}.SOM_is_labeled = true;
            
        end
        
    end
        
    methods (Static)       
        
    end
    
end
