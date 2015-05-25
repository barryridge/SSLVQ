classdef LRLearner < Learner
    % LRLEARNER Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        
        %% ------- *** OBJECTS *** ----------------------------------------
        %******************************************************************
        %******************************************************************
        % Cell array of modality objects...
        Modalities = [];
        
        % LR model...
        Model = [];
        
        % Normalisation struct...
        NormStruct = [];
        
        % Normalization method...
        NormMethod = 'range';
        
    end
    
    properties (SetAccess = private, Hidden = true)
        
        % Default settings...
        
        % Usage message...
        UsageMessage = ['\nLRLearner Usage: '...
                        'Please see the function comments for more detail.\n\n'];
                
    end
    
    methods

        %% ------- *** CONSTRUCTOR *** ------------------------------------
        %******************************************************************
        %******************************************************************
        function obj = LRLearner(varargin)
                          
            % Default settings...
            Data = [];
            InitData = [];
            TrainingData1Epoch = [];
            TrainingData = [];
            TestData = [];            
            TrainingIndices = [];
            TestIndices =[];
            epochs = 1;
            normalization = 'all';
            normalization_method = 'range';            
            
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
                        case 'trainingindices', i=i+1; TrainingIndices = varargin{i};
                        case 'testindices', i=i+1; TestIndices = varargin{i};
                        case 'epochs', i=i+1; epochs = varargin{i};
                        case 'normalization', i=i+1; normalization = varargin{i};
                        case 'normalization_method', i=i+1; normalization_method = varargin{i};
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
            
            %% SETUP TRAINING DATA FOR MULTIPLE EPOCHS --------------------            
            %--------------------------------------------------------------
%             if isempty(Data) && (isempty(obj.TrainingData) || isempty(obj.TestData))
%                 % error(obj.UsageMessage);
%                     return;
%                     
%             elseif isempty(obj.TrainingData) || isempty(obj.TestData)
%                 [obj.TrainingData obj.TestData] =...
%                     obj.setupdatastructs(Data,...
%                                          'TrainingIndices', TrainingIndices,...
%                                          'TestIndices', TestIndices,...
%                                          'epochs', epochs);
%             end
            
            %% NORMALIZE DATA ---------------------------------------------
            %--------------------------------------------------------------
%             switch normalization                
%                 case 'all'
%                     [TempData InModalityNorm] = obj.normalize([obj.TrainingData.Modalities{1}.FeatureVectors...
%                                                                obj.TestData.Modalities{1}.FeatureVectors]',...
%                                                               normalization_method);
%                                                           
%                     % Record the important stuff...
%                     obj.TrainingData.Modalities{1}.NormedFeatureVectors =...
%                         TempData(1:size(obj.TrainingData.Modalities{1}.FeatureVectors,2),:)';
%                     obj.TestData.Modalities{1}.NormedFeatureVectors =...
%                         TempData(size(obj.TrainingData.Modalities{1}.FeatureVectors,2)+1:end,:)';
%                     
%                 case {'train', 'training', 'training_data'}
%                     [TempData InModalityNorm] = obj.normalize(obj.TrainingData.Modalities{1}.FeatureVectors',...
%                                                               normalization_method);                                                          
%                                                           
%                     % Record the important stuff...
%                     obj.TrainingData.Modalities{1}.NormedFeatureVectors =...
%                         TempData(1:size(obj.TrainingData.Modalities{1}.FeatureVectors,2),:)';
%                
%                 otherwise
%                     obj.TrainingData.Modalities{1}.NormedFeatureVectors =...
%                         obj.TrainingData.Modalities{1}.FeatureVectors;
%                
%                     obj.TestData.Modalities{1}.NormedFeatureVectors =...
%                         obj.TestData.Modalities{1}.FeatureVectors;
%                     
%                     InModalityNorm = cell(size(obj.TrainingData.Modalities{1}.FeatureVectors,1),1);
%             end

            %% STORE TRAINING AND TEST DATA? ---------------------
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
            obj = obj.set(PassedArgs{1:end});
            
        end
        
        
        %% ------- *** SET PROPERTIES *** ---------------------------------
        %******************************************************************
        %******************************************************************
        function obj = set(obj, varargin)                                
            
            return;
            
        end
        
        
        %% ------- *** TRAIN *** ------------------------------------------
        %******************************************************************
        %******************************************************************
        function obj = train(obj, varargin)
            
            % Set defaults...
            % sample_size = size(obj.TrainingData.FeatureVectors,2);
            
            % Loop through arguments...
            i = 1;
            while i <= length(varargin), 
                argok = 1;
                if isnumeric(varargin{i})
                    sample_size = varargin{i};
                    
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
                    disp(['LRLearner.train(): Ignoring invalid argument #' num2str(i)]);
                    fprintf(obj.UsageMessage);
                end

                i = i + 1;
            end
            
            Y_trn = ones(size(obj.TrainingData1Epoch.Modalities{1}.ClassLabels,2),1);            
            for iClass = 2:obj.TrainingData1Epoch.Modalities{1}.nGroundTruths
                Y_trn(logical(obj.TrainingData1Epoch.Modalities{1}.ClassLabels(...
                              obj.TrainingData1Epoch.Modalities{1}.GroundTruthLabelIndices(iClass), :)), :) = iClass;
            end
                  
            X_trn = obj.TrainingData1Epoch.Modalities{1}.NormedFeatureVectors';            
            
            obj.Model = mnrfit(X_trn, Y_trn, 'model', 'nominal');
            
            obj.is_trained = true;
            
        end
        
        
        %% ------- *** CLASSIFY *** ---------------------------------------
        %------------------------------------------------------------------
        %******************************************************************
        %******************************************************************
        function [TestData Mask] = classify(obj, varargin)        
            
            % Defaults...
            
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
            
            %% CONVERT & NORMALIZE TEST DATA ------------------------------
            %--------------------------------------------------------------
            if isstruct(TestData)
                if isfield(TestData, 'NormedFeatureVectors')
                    SOMTestData = som_data_struct(TestData.NormedFeatureVectors');
                else                    
                    SOMTestData = som_data_struct(TestData.FeatureVectors');
                    SOMTestData = som_normalize(SOMTestData, obj.NormStruct);
                end
            else
                SOMTestData = som_data_struct(TestData');
                SOMTestData = som_normalize(SOMTestData, obj.NormStruct);
            end
            
            % Return normalized feature vectors...
            TestData.NormedFeatureVectors = SOMTestData.data';
            
            %% CLASSIFICATION USING THE RF CLASSIFIER --------------------
            %--------------------------------------------------------------
            testing_label_vector = ones(size(TestData.Modalities{1}.ClassLabels,2),1);            
            for iClass = 2:size(TestData.GroundTruthClassIndices,2)
                testing_label_vector(logical(TestData.Modalities{1}.ClassLabels(...
                                        TestData.GroundTruthClassIndices(iClass), :)), :) = iClass;
            end
            
            X_tst = TestData.Modalities{1}.NormedFeatureVectors';
            
            Probs = mnrval(obj.Model, X_tst,...
                           'model', 'nominal',...
                           'confidence', 0.95);
            [maxp, Y_hat] = max(Probs, [], 2);

            TestData.Results.InToOutClassification = Y_hat';
            
            Mask = nan(size(TestData.Modalities{1}.NormedFeatureVectors,1),1);

            % If we were using the internal test data in the
            % class, we should overwrite it with the new results...
            if using_internal_testdata
                obj.TestData = TestData;
            end                        
                
        end        
  
        
        %% ------- *** EVALUATE *** ---------------------------------------
        %------------------------------------------------------------------
        %******************************************************************
        %******************************************************************
        function TestData = evaluate(obj, varargin)
            
            % Defaults...
            % display = false;
            
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
                  case {'data', 'testdata'}, i=i+1; TestData = varargin{i};
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
            
            %% SET UP TEST DATA -------------------------------------------
            %--------------------------------------------------------------
            if isempty(TestData)
               TestData = obj.TestData;
               using_internal_testdata = true;
            end            
            
            %% CALCULATE EVALUATION SCORES ---------------------------------
            %---------------------------------------------------------------
            %---------------------------------------------------------------

            [TestData.Results.GroundTruth bar] = find(TestData.Modalities{1}.ClassLabels(...
                                obj.TrainingData1Epoch.Modalities{1}.GroundTruthLabelIndices, :));
                            
            TestData.Results.GroundTruth = TestData.Results.GroundTruth';
                            
            TestData.Results.ConfusionMatrix = zeros(size(obj.TrainingData1Epoch.Modalities{1}.GroundTruthLabelIndices,2));
            
            for iGroundTruth = 1:size(TestData.Results.ConfusionMatrix,1)
                
                GroundTruthClassIndices = find(TestData.Results.GroundTruth == iGroundTruth);
                
                for iPrediction = 1:size(TestData.Results.ConfusionMatrix,2)
                    
                    TestData.Results.ConfusionMatrix(iGroundTruth, iPrediction) =...
                        sum(TestData.Results.InToOutClassification(GroundTruthClassIndices) == iPrediction);
                end
            end
                            
            TestData.Results.Matches =...
                (TestData.Results.GroundTruth == TestData.Results.InToOutClassification);
            
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
