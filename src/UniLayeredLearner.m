classdef UniLayeredLearner < Learner
    % UNILAYEREDLEARNER Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        
        %% ------- *** OBJECTS *** ----------------------------------------
        %******************************************************************
        %******************************************************************
        % Cell array of modality objects...
        Modalities = [];
        
    end
    
    properties (SetAccess = private, Hidden = true)
        
        % Default settings...
        
        % Usage message...
        UsageMessage = ['\nUniLayeredLearner Usage: '...
                        'Please see the function comments for more detail.\n\n'];
                
    end
    
    methods

        %% ------- *** CONSTRUCTOR *** ------------------------------------
        %******************************************************************
        %******************************************************************
        function obj = UniLayeredLearner(varargin)
                          
            % Default settings...
            Data = [];
            TrainingData1Epoch = [];
            TrainingData = [];
            TestData = [];
            ModalityTypes = {'codebook'};
            CodebookSizes = {[10 10]};
            CodebookNeighs = {'bubble'};
            CodebookLattices = {'hexa'};
            CodebookShapes = {'sheet'};
            CodebookInitMethods = {'rand'};
            TrainingIndices = [];
            TestIndices =[];
            epochs = 1;
            normalization = 'all';
            normalization_method = 'range';
            Updaters = {{'SOM'}};
            
            
            % Loop through arguments...
            i = 1;
            iPassedArgs = 1;
            while i <= length(varargin), 
                argok = 1; 
                if ischar(varargin{i}), 
                    switch lower(varargin{i}),
                        case 'name', i=i+1; obj.Name = varargin{i};
                        case 'data', i=i+1; Data = varargin{i};
                        case 'trainingdata1epoch', i=i+1; TrainingData1Epoch = varargin{i};
                        case 'trainingdata', i=i+1; TrainingData = varargin{i};
                        case 'testdata', i=i+1; TestData = varargin{i};
                        case 'modality_types', i=i+1; ModalityTypes = varargin{i};
                        case 'codebook_sizes', i=i+1; CodebookSizes = varargin{i};
                        case 'codebook_neighs', i=i+1; CodebookNeighs = varargin{i};
                        case 'codebook_lattices', i=i+1; CodebookLattices = varargin{i};
                        case 'codebook_shapes', i=i+1; CodebookShapes = varargin{i};
                        case 'codebook_init_method', i=i+1; CodebookInitMethods = varargin{i};
                        case 'trainingindices', i=i+1; TrainingIndices = varargin{i};
                        case 'testindices', i=i+1; TestIndices = varargin{i};
                        case {'randomize_train', 'randomize_training_data',...
                          'randomizetrain', 'randomizetrainingdata'},...
                            i=i+1; obj.randomize_training_data = varargin{i};
                        case {'randomize_test', 'randomize_test_data',...
                              'randomizetest', 'randomizetestdata'},...
                                i=i+1; obj.randomize_test_data = varargin{i};
                        case {'store_training_data'},...
                                i=i+1; obj.store_training_data = varargin{i};
                        case {'store_test_data'},...
                                i=i+1; obj.store_test_data = varargin{i};
                        case 'epochs', i=i+1; epochs = varargin{i};
                        case 'normalization', i=i+1; normalization = varargin{i};
                        case 'normalization_method', i=i+1; normalization_method = varargin{i};
                        case {'updater', 'updaters'},
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
                    disp(['UniLayeredLearner.UniLayeredLearner(): Ignoring invalid argument #' num2str(i)]);
                    % fprintf(obj.UsageMessage);
                end

                i = i + 1;
            end
            
            %% SET UP TRAINING AND/OR TEST DATA STRUCTS FOR 1 EPOCH -------
            %--------------------------------------------------------------
            if isempty(Data) && (isempty(TrainingData1Epoch) || isempty(TrainingData) || isempty(TestData))
                % error(obj.UsageMessage);
                    return;
                    
            elseif isempty(TrainingData) || isempty(TestData)
                
                if isempty(TrainingIndices)
                    TrainingIndices = 1:size(Data.FeatureVectors,2);
                    TestIndices = [];
                end
                
                [TrainingData1Epoch TestData] =...
                    obj.setupdatastructs(Data,...
                                         'TrainingIndices', TrainingIndices,...
                                         'TestIndices', TestIndices,...
                                         'epochs', 1);
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

                    case {'train', 'training', 'training_data'}
                        [TempData InModalityNorm] = obj.normalize(TrainingData1Epoch.Modalities{1}.FeatureVectors,...
                                                                  normalization_method);                                                          

                        % Record the important stuff...
                        TrainingData1Epoch.Modalities{1}.NormedFeatureVectors =...
                            TempData(:,1:size(TrainingData1Epoch.Modalities{1}.FeatureVectors,2));
                        TrainingData1Epoch.Modalities{1}.Norm = InModalityNorm;

                    otherwise
                        TrainingData1Epoch.Modalities{1}.NormedFeatureVectors =...
                            TrainingData1Epoch.Modalities{1}.FeatureVectors;

                        TestData.Modalities{1}.NormedFeatureVectors =...
                            TestData.Modalities{1}.FeatureVectors;

                        % InModalityNorm = cell(size(TrainingData1Epoch.Modalities{1}.FeatureVectors,1),1);
                        TrainingData1Epoch.Modalities{1}.Norm = cell(size(TrainingData1Epoch.Modalities{1}.FeatureVectors,1),1);
                end
            end
            
            
            %% INSTANTIATE MODALITY OBJECTS -------------------------------
            % Instantiate this UniLayeredLearner object's 2 modality objects...
            %--------------------------------------------------------------
            switch Updaters{1}{1}
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
            
            %% SETUP TRAINING DATA FOR MULTIPLE EPOCHS --------------------            
            %
            % NOTE: This is a really stupid way of doing things, but I
            % don't have time to fix it right now.
            %--------------------------------------------------------------
            if obj.store_training_data || obj.store_test_data
                
                if isempty(Data) && (isempty(TrainingData) || isempty(TestData))
                    % error(obj.UsageMessage);
                        return;

                elseif isempty(TrainingData) || isempty(TestData)
                    [TrainingData TestData] =...
                        obj.setupdatastructs(Data,...
                                             'TrainingIndices', TrainingIndices,...
                                             'TestIndices', TestIndices,...
                                             'epochs', epochs);
                end

                %% NORMALIZE DATA -----------------------------------------
                %----------------------------------------------------------
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

                        case {'train', 'training', 'training_data'}
                            [TempData InModalityNorm] = obj.normalize(TrainingData.Modalities{1}.FeatureVectors,...
                                                                      normalization_method);                                                          

                            % Record the important stuff...
                            TrainingData.Modalities{1}.NormedFeatureVectors =...
                                TempData(:,1:size(TrainingData.Modalities{1}.FeatureVectors,2));
                            TrainingData.Modalities{1}.Norm = InModalityNorm;

                        otherwise
                            TrainingData.Modalities{1}.NormedFeatureVectors =...
                                TrainingData.Modalities{1}.FeatureVectors;

                            TestData.Modalities{1}.NormedFeatureVectors =...
                                TestData.Modalities{1}.FeatureVectors;

                            % InModalityNorm = cell(size(TrainingData.Modalities{1}.FeatureVectors,1),1);
                            TrainingData.Modalities{1}.Norm = cell(size(TrainingData1Epoch.Modalities{1}.FeatureVectors,1),1);
                    end
                end
                
            end
            
            %% Set up CurrentSample data struct ---------------------------
            %--------------------------------------------------------------
            obj.CurrentSample.Modalities{1}.FeatureNames = TrainingData.Modalities{1}.FeatureNames;
            obj.CurrentSample.Modalities{1}.ClassNames = TrainingData.Modalities{1}.ClassNames;
            obj.CurrentSample.Modalities{1}.nGroundTruths = TrainingData.Modalities{1}.nGroundTruths;
            obj.CurrentSample.Modalities{1}.GroundTruthLabelIndices = TrainingData.Modalities{1}.GroundTruthLabelIndices;
            
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
            
            %% STORE TRAINING AND TEST DATA? ------------------------------
            %--------------------------------------------------------------
            if obj.store_training_data
                obj.TrainingData1Epoch = TrainingData1Epoch;
                obj.TrainingData = TrainingData;
            end
            
            if obj.store_test_data
                obj.TestData = TestData;
            end
            
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
            Updaters = {{'SOM'}};
            PhaseShifts = {{NaN}};   
            AlphaTypes = {{'linear'}};
            AlphaInits = {{1}};
            RadiusTypes = {{'linear'}};
            RadiusInits = {{5}};
            RadiusFins = {{1}};
            WindowSizes = {NaN};
            AlphaFeatureTypes = {{NaN}};
            AlphaFeatureInits = {{NaN}};
            Metrics = {{'euclidean'}};
                       
            
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
                    case {'trainlen', 'train_len'}, i=i+1; trainlen = varargin{i};
                    case {'nClasses', 'numclasses'}, i=i+1; nClasses = varargin{i};
                    case {'auxdist', 'auxdist_type'}, i=i+1; auxdist_type_value = varargin{i};
                    case {'featureselection', 'feature_selection'}, i=i+1; obj.feature_selection = varargin{i};
                    case {'featureselectionmax', 'feature_selection_max'}, i=i+1; obj.feature_selection_max = varargin{i};
                    case {'featureselectionfeedback', 'feature_selection_feedback',...
                          'featureselectionintraining', 'feature_selection_in_training'},...
                            i=i+1; obj.feature_selection_feedback = varargin{i};
                    case {'classificationmethod', 'classification_method'},...
                            i=i+1; obj.classification_method = varargin{i};
                    case {'nodecolouringmethod', 'node_colouring_method'},...
                            i=i+1; obj.node_colouring_method = varargin{i};
                    case {'metric'}, i=i+1; Metrics = varargin{i};
                    case {'record'}, i=i+1; obj.record = varargin{i};
                    case {'verbose'}, i=i+1; obj.verbose = varargin{i};
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
                disp(['UniLayeredLearner.set(): Ignoring invalid argument #' num2str(i+1)]); 
              end
              i = i+1;
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
                                                      'trainlen', trainlen,...
                                                      'nClasses', nClasses,...
                                                      'feature_selection_feedback', obj.feature_selection_feedback,...
                                                      'record', obj.record,...
                                                      'verbose', obj.verbose);                                                                                
            
        end
        
        
        %% ------- *** TRAIN *** ------------------------------------------
        %******************************************************************
        %******************************************************************
        function obj = train(obj, varargin)
            
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
                    disp(['UniModalLearner.train(): Ignoring invalid argument #' num2str(i)]);
                    fprintf(obj.UsageMessage);
                end

                i = i + 1;
            end
            
            if isempty(TrainingData)
                TrainingData = obj.TrainingData;
                sample_size = size(TrainingData.FeatureVectors,2);
            end
            
            %% TRAINING LOOP ----------------------------------------------
            %--------------------------------------------------------------                        
            % for t = obj.t+1 : min(obj.t + sample_size, size(TrainingData.FeatureVectors,2))
            % for t = 1:size(TrainingData.FeatureVectors,2)                                
                
                
                %% GRAB CURRENT SAMPLE ------------------------------------
                %----------------------------------------------------------
                lower_bound = obj.t+1;
                upper_bound = min(obj.t + sample_size, size(TrainingData.FeatureVectors,2));
                
                obj.CurrentSample.Modalities{1}.FeatureVectors(:,1:sample_size) =...
                    TrainingData.Modalities{1}.FeatureVectors(:,lower_bound:upper_bound);
                obj.CurrentSample.Modalities{1}.NormedFeatureVectors(:,1:sample_size) =...
                    TrainingData.Modalities{1}.NormedFeatureVectors(:,lower_bound:upper_bound);
                obj.CurrentSample.Modalities{1}.ClassLabels(:,1:sample_size) =...
                    TrainingData.Modalities{1}.ClassLabels(:,lower_bound:upper_bound);
                obj.CurrentSample.Modalities{1}.GroundTruthLabelIndices =...
                    TrainingData.Modalities{1}.GroundTruthLabelIndices;
                
                %% TRAIN MODALITY -----------------------------------
                %---------------------------------------------------------- 
                obj.Modalities{1} =...
                    obj.Modalities{1}.train(obj.CurrentSample.Modalities{1});
                
%                 obj.Modalities{1} =...
%                     obj.Modalities{1}.train(TrainingData.Modalities{1});
                                                      
                %% RECORD MODALITY TRAINING INFORMATION OVER TIME ---------
                %----------------------------------------------------------
%                 if obj.record                    
%                     obj.Modalities{1}.GroundTruthRecord(t,:) =...
%                         find(obj.CurrentSample.Modalities{1}.ClassLabels(...
%                                 obj.CurrentSample.Modalities{1}.GroundTruthLabelIndices,:));
%                     obj.Modalities{1}.ActivationsRecord(t,:) = obj.Modalities{1}.Activations;
%                     obj.Modalities{1}.BMURecord(t, :) = obj.Modalities{1}.BMUs;
%                 end
                
                                                      
                %% CLEAR MODALITY BMUs ------------------------------------
                %----------------------------------------------------------
                % obj.Modalities{1} = obj.Modalities{1}.clearbmus();
                
            % end            
            
            obj.is_trained = true;
            
        end
        
        
        %% ------- *** CLASSIFY *** ---------------------------------------
        %------------------------------------------------------------------
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
                case {'cull_nodes', 'cull_inaccurate_nodes'},...
                    NodeColouringMethod = obj.node_colouring_method;
                case {'cull_nodes_hard', 'cull_inaccurate_nodes_hard'},...
                    NodeColouringMethod = obj.node_colouring_method;
                
                otherwise, NodeColouringMethod = 'none';
            end
            
            % The TestData struct...
            TestData = [];
            
            % A flag that lets us know if we're using the TestData struct
            % in the object, or test data that was passed as an argument...
            using_internal_testdata = false;
            
            Metric = 'euclidean';
            hebbian = true;                        
            
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
                    case {'data', 'testdata'}, i=i+1; TestData = varargin{i};
                    case {'verbose'}, verbose = true;
                    case {'svm'}, Method = 'svm';
                    case {'lda'}, Method = 'lda';
                    case {'naivebayes', 'naivebayes_linear', 'naivebayes_diaglinear', 'diaglinear'},...
                         Method = 'diaglinear';
                    case {'naivebayes_quadratic', 'naivebayes_diagquadratic', 'diagquadratic'},...
                         Method = 'diagquadratic';
                    case {'lvq'}, Method = 'lvq';
                    case {'nodewise', 'node'}, Method = 'node';
                    case {'euclidean', 'euclid', 'euc'}, Metric = 'euclidean';
                    case {'hellinger', 'hell'}, Metric = 'hellinger';
                    case {'clusterwise', 'cluster'}, Method = 'cluster';
                    case {'nonhebbian'}, hebbian = false;
                    case {'nodecolouringmethod', 'node_colouring_method', 'nodecullingmethod', 'node_culling_method'},...
                            i=i+1; NodeColouringMethod = varargin{i};
                    case {'keep_clustering', 'keep_clusters',...
                          'retain_clustering', 'retain_clusters'},...
                          % BiModalLearner takes this argument sometimes,
                          % so just putting a dummy in here to stop
                          % UniLayeredLearner from throwing invalid
                          % argument warnings.
                          RetainClustering = true;
                      
                    otherwise, argok=0; 
                end
              elseif isstruct(varargin{i})
                  TestData = varargin{i};
              else
                argok = 0; 
              end
              if ~argok, 
                disp(['Ignoring invalid argument #' num2str(i+1)]); 
              end
              i = i+1; 
            end
            
            %% SET UP TEST DATA -------------------------------------------
            %--------------------------------------------------------------
            if isempty(TestData)
               TestData = obj.TestData;
               using_internal_testdata = true;
            end                
            
            %% USE INDIVIDUAL NODE ALPHA VALUES (FROM OLVQ) TO ESTIMATE ---
            % THE RELEVANCE OF INDIVIDUAL NODES...
            %--------------------------------------------------------------
            % if ~isempty(obj.Modalities{1}.Alphas)
            %     NodeRelevances = 1  - obj.Modalities{1}.Alphas;
            % else
            %     NodeRelevances = 0;
            % end
            % 
            % if sum(NodeRelevances) > 0
            %     NodeRelevances = 1  - obj.Modalities{1}.Alphas;
            %     Codebook = obj.Modalities{1}.SOM.codebook(NodeRelevances >= mean(NodeRelevances(:)), :);
            %     ClassLabels = obj.Modalities{1}.ClassLabels(NodeRelevances >= mean(NodeRelevances(:)));
            % else
            %     NodeRelevances = ones(size(obj.Modalities{1}.ClassLabels));
            %     Codebook = obj.Modalities{1}.SOM.codebook;
            %     ClassLabels = obj.Modalities{1}.ClassLabels;
            % end
            
            %% CULL INACCURATE NODES --------------------------------------
            %--------------------------------------------------------------
            switch NodeColouringMethod
                
                case {'cull_nodes', 'cull_inaccurate_nodes'}                    
                    
                    Classes = unique(obj.Modalities{1}.ClassLabels);
                    Codebook = [];
                    ClassLabels = [];
                    AccuracyHist = [];
                    
                    for iClass = 1:length(Classes)                        
                        ThisClassCodebook = obj.Modalities{1}.SOM.codebook(obj.Modalities{1}.ClassLabels==Classes(iClass),:);
                        ThisClassClassLabels = obj.Modalities{1}.ClassLabels(obj.Modalities{1}.ClassLabels==Classes(iClass),:);
                        ThisClassAccuracyHist = obj.Modalities{1}.AccuracyHist(obj.Modalities{1}.ClassLabels==Classes(iClass),:);
                        
                        if sum(ClassNodeAccuracies >= (mean(ThisClassAccuracyHist) + std(ThisClassAccuracyHist))) > 0
                            Codebook = [Codebook' ThisClassCodebook(ThisClassAccuracyHist >= (mean(ThisClassAccuracyHist) + std(ThisClassAccuracyHist)),:)']';
                            ClassLabels = [ClassLabels' ThisClassClassLabels(ThisClassAccuracyHist >= (mean(ThisClassAccuracyHist) + std(ThisClassAccuracyHist)),:)']';
                            AccuracyHist = [AccuracyHist' ThisClassAccuracyHist(ThisClassAccuracyHist >= (mean(ThisClassAccuracyHist) + std(ThisClassAccuracyHist)),:)']';
                        else
                            Codebook = [Codebook' ThisClassCodebook']';
                            ClassLabels = [ClassLabels' ThisClassClassLabels']';
                            AccuracyHist = [AccuracyHist' ThisClassAccuracyHist']';
                        end
                    end
                    
                case {'cull_nodes_hard', 'cull_inaccurate_nodes_hard'}                    
                    
                    Classes = unique(obj.Modalities{1}.ClassLabels);
                    Codebook = [];
                    ClassLabels = [];
                    AccuracyHist = [];
                    
                    for iClass = 1:length(Classes)                        
                        ThisClassCodebook = obj.Modalities{1}.SOM.codebook(obj.Modalities{1}.ClassLabels==Classes(iClass),:);
                        ThisClassClassLabels = obj.Modalities{1}.ClassLabels(obj.Modalities{1}.ClassLabels==Classes(iClass),:);
                        ThisClassAccuracyHist = obj.Modalities{1}.AccuracyHist(obj.Modalities{1}.ClassLabels==Classes(iClass),:);
                        
                        if sum(ThisClassAccuracyHist >= mean(ThisClassAccuracyHist)) > 0
                            Codebook = [Codebook' ThisClassCodebook(ThisClassAccuracyHist >= mean(ThisClassAccuracyHist),:)']';
                            ClassLabels = [ClassLabels' ThisClassClassLabels(ThisClassAccuracyHist >= mean(ThisClassAccuracyHist),:)']';
                            AccuracyHist = [AccuracyHist' ThisClassAccuracyHist(ThisClassAccuracyHist >= mean(ThisClassAccuracyHist),:)']';
                        else
                            Codebook = [Codebook' ThisClassCodebook']';
                            ClassLabels = [ClassLabels' ThisClassClassLabels']';
                            AccuracyHist = [AccuracyHist' ThisClassAccuracyHist']';
                        end
                    end
                    
                otherwise
                    
                    Codebook = obj.Modalities{1}.SOM.codebook;
                    ClassLabels = obj.Modalities{1}.ClassLabels;
                    AccuracyHist = obj.Modalities{1}.AccuracyHist;
                    
            end
            
            %% CLASSIFY TEST VECTORS IN MODALITY -------------------
            %  Classify the in modality test vectors in terms of out
            %  modality clusters.
            %--------------------------------------------------------------
            
            switch Method

                case 'svm',
                    %% TRAIN AN SVM CLASSIFIER IN THE IN-MODALITY -----------------
                    %--------------------------------------------------------------
                    if obj.SOM_based
                        Classifier =...
                            svmtrain(obj.Modalities{1}.SOM.codebook,...
                                     obj.Modalities{1}.ClassLabels,...
                                     'Kernel_Function', 'rbf')';
                    else
                        [ClassLabels bar] = find(obj.TrainingData.Modalities{1}.CategoryLabels(...
                                                 obj.TrainingData.Modalities{1}.AffordanceLabelIndices,1:obj.t));
                        
                        Classifier =...
                            svmtrain(obj.TrainingData.Modalities{1}.NormedFeatureVectors(:,1:obj.t)',...
                                     ClassLabels,...
                                     'Kernel_Function', 'rbf');
                    end

                    %% CONVERT & NORMALIZE TEST DATA ------------------------------
                    %--------------------------------------------------------------
                    SOMTestData = som_data_struct(TestData.Modalities{1}.FeatureVectors');

                    % Normalize...
                    SOMTestData = som_normalize(SOMTestData, obj.Norm);

                    %% CLASSIFICATION USING THE SVM CLASSIFIER --------------------
                    %--------------------------------------------------------------
                    TestData.Results.InToOutClassification =...
                        svmclassify(Classifier, SOMTestData.data);
                    
                    
                case {'diaglinear', 'diagquadratic'},
                    %% CONVERT & NORMALIZE TEST DATA ------------------------------
                    %--------------------------------------------------------------
                    SOMTestData = som_data_struct(TestData.Modalities{1}.FeatureVectors');

                    % Normalize...
                    SOMTestData = som_normalize(SOMTestData, obj.Norm);

                    %% CLASSIFICATION USING THE LDA CLASSIFIER --------------------
                    %--------------------------------------------------------------
                    if obj.SOM_based
                        TestData.Results.InToOutClassification =...
                            classify(SOMTestData.data, obj.Modalities{1}.SOM.codebook, obj.Modalities{1}.ClassLabels)';
                    else
                        [ClassLabels bar] = find(obj.TrainingData.Modalities{1}.CategoryLabels(...
                                                 obj.TrainingData.Modalities{1}.AffordanceLabelIndices,1:obj.t));
                        
                        TestData.Results.InToOutClassification =...
                            classify(SOMTestData.data,...
                                     obj.TrainingData.Modalities{1}.NormedFeatureVectors(:,1:obj.t)',...
                                     ClassLabels,...
                                     Method);
                    end
                    
                
                case 'lda',
                    %% CONVERT & NORMALIZE TEST DATA ------------------------------
                    %--------------------------------------------------------------
                    SOMTestData = som_data_struct(TestData.Modalities{1}.FeatureVectors');

                    % Normalize...
                    SOMTestData = som_normalize(SOMTestData, obj.Norm);

                    %% CLASSIFICATION USING THE LDA CLASSIFIER --------------------
                    %--------------------------------------------------------------
                    if obj.SOM_based
                        TestData.Results.InToOutClassification =...
                            classify(SOMTestData.data, obj.Modalities{1}.SOM.codebook, obj.Modalities{1}.ClassLabels)';
                    else
                        [ClassLabels bar] = find(obj.TrainingData.Modalities{1}.CategoryLabels(...
                                                 obj.TrainingData.Modalities{1}.AffordanceLabelIndices,1:obj.t));
                        
                        TestData.Results.InToOutClassification =...
                            classify(SOMTestData.data, obj.TrainingData.Modalities{1}.NormedFeatureVectors(:,1:obj.t)',...
                                                       ClassLabels);
                    end
                    
                case {'node', 'lvq'},
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
                        if any(sum(ClassVars) == 0)
                            FisherCriterion = var(ClassMeans);
                        else
                            FisherCriterion = var(ClassMeans) ./ sum(ClassVars);
                        end

                        % Watch out for nasty NaNs...
                        FisherCriterion(isnan(FisherCriterion)) = 0;

                        % Make the mask a weight distribution (unit norm)...
                        % Mask = (FisherCriterion ./ norm(FisherCriterion,1))';
                        TestData.Results.InputMask = (FisherCriterion ./ max(FisherCriterion))';                        

                        % Save the mask for later...
                        % obj.Modalities{1}.SOM.mask = TestData.Results.InputMask;
                    
                    % Otherwise, just make sure the feature mask is
                    % normalized...
                    else
                        TestData.Results.InputMask = obj.Modalities{1}.SOM.mask ./ norm(obj.Modalities{1}.SOM.mask,1);
                    end
                    
                    %% CLASSIFY TEST VECTORS NODE-WISE IN INPUT MODALITY ------------
                    %----------------------------------------------------------------
                    % For hard feature selection, we pick out the most
                    % relevant features based on the mean feature weight
                    % and re-normalize...
                    if ~isempty(findstr(obj.feature_selection, 'hard'))
                        
                        if ~isempty(obj.feature_selection_max) && ~isnan(obj.feature_selection_max)                         
                            
                            [foo bar] = sort(TestData.Results.InputMask,'descend');
                            
                            QueryMask = zeros(size(TestData.Results.InputMask));
                            QueryMask(bar(1:obj.feature_selection_max)) = 1;
                            
                        else
                            
                            TestData.Results.InputMask(TestData.Results.InputMask < mean(TestData.Results.InputMask)) = 0;
                            QueryMask = TestData.Results.InputMask ./ norm(TestData.Results.InputMask,1);
                            
                        end
                        
                        [TestDataInMatches TestData.Modalities{1}.NormedFeatureVectors] =...
                                        obj.Modalities{1}.classify(TestData.Modalities{1},...
                                                                   'codebook', Codebook,...
                                                                   'mask', QueryMask);
                    
                    % Query-based exponential feature weighting...
                    elseif ~isempty(findstr(obj.feature_selection, 'exp'))
                        
                        % First off, we have to find the BMUs for the test
                        % data...
                        [TestDataBMUs TestData.Modalities{1}.NormedFeatureVectors] =...
                                    obj.Modalities{1}.classify(TestData.Modalities{1},...
                                                               'codebook', Codebook,...
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
                            
                            TestSample = [];
                            TestSample.NormedFeatureVectors = TestData.Modalities{1}.NormedFeatureVectors(:,iTestData);
                                                        
                            % Classify
                            % QueryMask(QueryMask < mean(QueryMask)) = 0;
                            % QueryMask = QueryMask ./ norm(QueryMask,1);
                            TestDataInMatches(iTestData) =...
                                obj.Modalities{1}.classify(TestSample,...
                                                           'codebook', Codebook,...
                                                           'mask', QueryMask,...
                                                           'whichbmus', 'best');
                            
                        end
                        
                    % Classwise query-based LDA feature weighting...
                    elseif ~isempty(findstr(obj.feature_selection, 'classwise'))
                        
                        % First off, we have to find the BMUs for the test
                        % data...
                        [TestDataBMUs TestData.Modalities{1}.NormedFeatureVectors] =...
                                    obj.Modalities{1}.classify(TestData.Modalities{1},...
                                                               'codebook', Codebook,...
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
                            
                            Class1 = obj.Modalities{1}.ClassLabels(BMUClass1);
                            Class2 = obj.Modalities{1}.ClassLabels(BMUClass2);
                            
                            ClassMeans(1,:) = obj.Modalities{1}.ClassStats{Class1}.mean();
                            ClassMeans(2,:) = obj.Modalities{1}.ClassStats{Class2}.mean();
                            ClassVars(1,:) = obj.Modalities{1}.ClassStats{Class1}.var();
                            ClassVars(2,:) = obj.Modalities{1}.ClassStats{Class2}.var();
                            
                            % Fisher Criterion =
                            %  (Between Class Variance)
                            % --------------------------
                            %  (Within Class Variance)
                            if any(sum(ClassVars) == 0)
                                FisherCriterion = var(ClassMeans);
                            else
                                FisherCriterion = var(ClassMeans) ./ sum(ClassVars);
                            end

                            % Watch out for nasty NaNs...
                            FisherCriterion(isnan(FisherCriterion)) = 0;

                            % Make the mask a weight distribution (unit norm)...
                            % QueryMask = (FisherCriterion ./ norm(FisherCriterion,1))';                            
                            QueryMask = (FisherCriterion ./ max(FisherCriterion))';

                                                        
                            % Classify
                            % QueryMask(QueryMask < mean(QueryMask)) = 0;
                            % QueryMask = QueryMask ./ norm(QueryMask,1);
                            TestDataInMatches(iTestData) =...
                                obj.Modalities{1}.classify(TestData.Modalities{1}.FeatureVectors(:,iTestData)',...
                                                           'codebook', Codebook,...
                                                           'mask', QueryMask,...
                                                           'whichbmus', 'best');
                        end
                        
                    % Nodewise query-based LDA feature weighting...
                    elseif ~isempty(findstr(obj.feature_selection, 'nodewise'))
                        
                        % First off, we have to find the BMUs for the test
                        % data...
                        [TestDataBMUs TestData.Modalities{1}.NormedFeatureVectors] =...
                                    obj.Modalities{1}.classify(TestData.Modalities{1},...
                                                               'codebook', Codebook,...
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
                            
                            %% Relevance determination...
                            ClassMeans(1,:) = obj.Modalities{1}.NodeStats{TestDataBMUs(iTestData,1)}.mean();

                            if isnan(ClassMeans(1,:))
                                return;
                            else
                                OtherVar = obj.Modalities{1}.NodeStats{TestDataBMUs(iTestData,1)}.var();

                                if ~any(isnan(OtherVar))
                                    ClassVars(1,:) = OtherVar;
                                else
                                    ClassVars(1,:) = zeros(size(ClassMeans(1,:)));
                                end
                            end

                            OtherClassNodes = TestDataBMUs(iTestData,obj.Modalities{1}.ClassLabels(TestDataBMUs(iTestData,:)) ~= obj.Modalities{1}.ClassLabels(TestDataBMUs(iTestData,1)));

                            for iOther = 1:size(OtherClassNodes,2)

                                OtherMean = obj.Modalities{1}.NodeStats{OtherClassNodes(iOther)}.mean();

                                if ~any(isnan(OtherMean))

                                    ClassMeans(2,:) = OtherMean;

                                    OtherVar = obj.Modalities{1}.NodeStats{OtherClassNodes(iOther)}.var();

                                    if ~any(isnan(OtherVar))
                                        ClassVars(2,:) = OtherVar;
                                    else
                                        ClassVars(2,:) = zeros(size(ClassMeans(2,:)));
                                    end

                                    break;

                                elseif iOther >= size(OtherClassNodes,2)
                                    return;
                                end

                            end                                                        
                            
                            % Fisher Criterion =
                            %  (Between Class Variance)
                            % --------------------------
                            %  (Within Class Variance)
                            if any(sum(ClassVars) == 0)
                                FisherCriterion = var(ClassMeans);
                            else
                                FisherCriterion = var(ClassMeans) ./ sum(ClassVars);
                            end

                            % Watch out for nasty NaNs...
                            FisherCriterion(isnan(FisherCriterion)) = 0;

                            % Make the mask a weight distribution (unit norm)...
                            % QueryMask = (FisherCriterion ./ norm(FisherCriterion,1))';                            
                            QueryMask = (FisherCriterion ./ max(FisherCriterion))';

                                                        
                            % Classify
                            % QueryMask(QueryMask < mean(QueryMask)) = 0;
                            % QueryMask = QueryMask ./ norm(QueryMask,1);
                            TestDataInMatches(iTestData) =...
                                obj.Modalities{1}.classify(TestData.Modalities{1}.FeatureVectors(:,iTestData)',...
                                                           'codebook', Codebook,...
                                                           'mask', QueryMask,...
                                                           'whichbmus', 'best');
                        end
                        
                    % Fuzzy feature weighting...
                    elseif ~isempty(findstr(obj.feature_selection, 'fuzzy'))
                        
                        if ~isempty(obj.feature_selection_max) && ~isnan(obj.feature_selection_max)                        
                            
                            [foo bar] = sort(TestData.Results.InputMask,'descend');
                            
                            QueryMask = zeros(size(TestData.Results.InputMask));
                            QueryMask(bar(1:obj.feature_selection_max)) = 1;
                            QueryMask(find(QueryMask)) = TestData.Results.InputMask(find(QueryMask));
                            QueryMask = QueryMask ./ norm(QueryMask,1);
                            
                        else
                            
                            QueryMask = TestData.Results.InputMask ./ norm(TestData.Results.InputMask,1);
                            
                        end
                        
                        [TestDataInMatches TestData.Modalities{1}.NormedFeatureVectors] =...
                                    obj.Modalities{1}.classify(TestData.Modalities{1},...
                                                               'codebook', Codebook,...
                                                               'mask', QueryMask);
                    
                    % Otherwise, ignore feature weights...
                    else
                        [TestDataInMatches TestData.Modalities{1}.NormedFeatureVectors] =...
                                    obj.Modalities{1}.classify(TestData.Modalities{1},...
                                                               'codebook', Codebook);
                    end
                    
                    % Save the results in the TestData struct...
                    TestData.Results.InToOutClassification = ClassLabels(TestDataInMatches')';
                    
                    % DEBUG PLOTTING:
%                     figure;
%                     hold on;
%                     Train = TrainingData1Epoch.Modalities{1}.NormedFeatureVectors';
%                     TrainLabels = TrainingData1Epoch.Modalities{1}.ClassLabels(15:16,:);
%                     plot(Train(TrainLabels(2,:)==1,1),Train(TrainLabels(2,:)==1,2),'bo');
%                     plot(Train(TrainLabels(1,:)==1,1),Train(TrainLabels(1,:)==1,2),'rs');            
%                     Test = TestData.Modalities{1}.NormedFeatureVectors';
%                     TestLabels = TestData.Modalities{1}.ClassLabels(15:16,:);
%                     plot(Test(TestLabels(1,:)==1,1),Test(TestLabels(1,:)==1,2),'rx');            
%                     plot(Test(TestLabels(2,:)==1,1),Test(TestLabels(2,:)==1,2),'bx');
%                     legend('Training Data: Rolling Objects',...
%                            'Training Data: Non-Rolling Objects',...
%                            ['Test Data: ' TestData.ClassNames{find(TestData.Modalities{1}.ClassLabels(1:14,1))}(9:end)],...
%                            'location', 'northwest');
%                     axis([0 1 0 1]);
%                     xlabel('Curvature Feature 1');
%                     ylabel('Curvature Feature 2');
%                     title('Object Property Modality Dims 1 & 2 of Training Data & Test Data');
% 
%                     figure;
%                     hold on;
%                     plot(Codebook(ClassLabels==2,1),Codebook(ClassLabels==2,2),'bo')
%                     plot(Codebook(ClassLabels==1,1),Codebook(ClassLabels==1,2),'rs');
%                     Test = TestData.Modalities{1}.NormedFeatureVectors';
%                     TestLabels = TestData.Modalities{1}.ClassLabels(15:16,:);
%                     plot(Test(TestLabels(1,:)==1,1),Test(TestLabels(1,:)==1,2),'rx');
%                     plot(Test(TestLabels(2,:)==1,1),Test(TestLabels(2,:)==1,2),'bx');
%                     legend('Codebook: Rolling Objects',...
%                            'Codebook: Non-Rolling Objects',...
%                            ['Test Data: ' TestData.ClassNames{find(TestData.Modalities{1}.ClassLabels(1:14,1))}(9:end)],...
%                            'location', 'northwest');
%                     axis([0 1 0 1]);
%                     xlabel('Curvature Feature 1');
%                     ylabel('Curvature Feature 2');
%                     title({obj.Name; 'Object Property Modality Dims 1 & 2 of Codebook & Test Data'});                                                            
                    
                    
                    % If we were using the internal test data in the
                    % class, we should overwrite it with the new results...
                    if using_internal_testdata
                        obj.TestData = TestData;
                    end
                
            end
                
        end        
  
        
        %% ------- *** EVALUATE *** ---------------------------------------
        %------------------------------------------------------------------
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
                  case {'data', 'testdata'},  i=i+1; TestData = varargin{i};
                  % case 'display', i=i+1; DisplaySample = varargin{i}; display = true;
                      
                  otherwise, argok=0; 
                end
              elseif isstruct(varargin{i})
                  TestData = varargin{i};
              else
                argok = 0; 
              end
              if ~argok, 
                disp(['Ignoring invalid argument #' num2str(i+1)]); 
              end
              i = i+1; 
            end
            
            %% SET UP TRAINING DATA ---------------------------------------
            %--------------------------------------------------------------
            if isempty(TrainingData)
               TrainingData = obj.TrainingData;
            end
            
            if isempty(TrainingData1Epoch)               
               TrainingData = obj.TrainingData1Epoch;
            end
            
            %% SET UP TEST DATA -------------------------------------------
            %--------------------------------------------------------------
            if isempty(TestData)
               TestData = obj.TestData;
               using_internal_testdata = true;
            end            
            
            %% CALCULATE EVALUATION SCORES --------------------------------
            %--------------------------------------------------------------
            %--------------------------------------------------------------

            [foo bar] = find(TestData.Modalities{1}.ClassLabels(...
                                TrainingData.Modalities{1}.GroundTruthLabelIndices, :));
                            
            TestData.Results.Matches =...
                (foo' == TestData.Results.InToOutClassification);
            
            TestData.Results.Score =...
                sum(TestData.Results.Matches, 2);
            
            TestData.Results.Percent =...
                TestData.Results.Score /...
                    size(TestData.Modalities{1}.FeatureVectors,2);
                
            % If we were using the internal test data in the
            % class, we should overwrite it with the new results...
            if using_internal_testdata
                obj.TestData = TestData;
            end
    
        end

        
    end
        
        
    
    
end
