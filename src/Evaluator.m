classdef Evaluator < handle
    % EVALUATOR Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        
        %% ------- *** PROPERTIES *** -------------------------------------
        %******************************************************************
        %******************************************************************
        % Cell array of parameters for different types of learners...
        Learners = [];
        LearnerNames = [];
        
        % Data...
        Data = [];
        
        % Results...
        Results = [];
        
        % Save filename...
        doSave = false;
        SaveFileName = 'results.mat';
        
        % Number of evaluation trials to run...
        nTrials = 1;
        
        % Number of training epochs, i.e., how many times the training set
        % should be multiplied...
        nEpochs = 1;                
        
        % How big should the interval be between training steps where we
        % evaluate the learners?
        interval_size = Inf;
        test_interval_timesteps = Inf;
        test_interval_epochs = [];
        test_interval_percentage = [];
        
        % Parallel processing...
        doParallel = false;
        nCores = Inf;
        
        % Debug depth level.
        % 0 = Don't save anything except results.
        % 1 = Save the learner states at the final evaluation timestep
        %     without internal learner states.
        % 2 = Save the learner states at each interval evaluation timestep
        %     without internal learner states.
        % 3 = Save the learner states at the final evaluation timestep
        %     with internal learner states.
        % 4 = Save the learner states at each interval evaluation timestep
        %     with internal learner states.        
        debug_level = 0;
        
        % Should the training and test data be replicated in each learner
        % object?
        % If not, we store it here in the Evaluator object.
        replicate_training_data = false;
        replicate_test_data = false;
        TrainingData = [];
        TrainingData1Epoch = [];
        TestData = [];
        
        % Should the initial conditions, i.e. codebooks, in each learner be replicated?
        % E.g. Initial codebook vectors?
        replicate_initial_conditions = false;
        
        % Should the ordering of the training and test data be randomized?
        randomize_training_data = 'samplewise';
        randomize_test_data = 'samplewise';
        randomizer_seed = sum(100*clock);
    end
    
    methods (Abstract)
                
        %% ------- *** RUN *** --------------------------------------------
        %******************************************************************
        %******************************************************************
        obj = run(obj, varargin)
            
    end
    
    methods
        
        %% ------- *** SET PROPERTIES *** ---------------------------------
        %******************************************************************
        %******************************************************************
        function obj = set(obj, varargin)
            
            % Loop through arguments...
            i = 1;
            iPassedArgs = 1;
            while i <= length(varargin), 
                argok = 1; 
                if ischar(varargin{i}), 
                    switch lower(varargin{i}), 
                        case 'data', i=i+1; obj.Data = varargin{i};
                        case 'save', i=i+1;...
                                        obj.doSave = true;
                                        obj.SaveFileName = varargin{i};
                        case 'trials', i=i+1; obj.nTrials = varargin{i};
                        case 'epochs', i=i+1; obj.nEpochs = varargin{i};
                        case {'test_interval_timesteps', 'interval_timesteps'},...
                                i=i+1; obj.test_interval_timesteps = varargin{i};
                        case {'test_interval_epochs', 'interval_epochs'},...
                                i=i+1; obj.test_interval_epochs = varargin{i};
                        case {'test_interval_percentage', 'interval_percentage'},...
                                i=i+1; obj.test_interval_percentage = varargin{i};
                        case {'par', 'parallel'}, i=i+1;...
                                        obj.doParallel = true;
                                        obj.nCores = varargin{i};
                        case {'debug', 'debug_level'}, i=i+1; obj.debug_level = varargin{i};
                        case {'replicate_training_data'},...
                                i=i+1; obj.replicate_training_data = varargin{i};
                        case {'replicate_test_data'},...
                                i=i+1; obj.replicate_test_data = varargin{i};
                        case {'replicate_data'}
                                i=i+1;
                                obj.replicate_training_data = varargin{i};
                                obj.replicate_test_data = varargin{i};
                        case {'replicate_init_conditions', 'replicate_initial_conditions'},...
                                i=i+1; obj.replicate_initial_conditions = varargin{i};
                        case {'randomize_training_data', 'randomize_train_data', 'rand_train_data'},...
                                i=i+1; obj.randomize_training_data = varargin{i};
                        case {'randomize_test_data', 'randomize_test_data', 'rand_test_data'},...
                                i=i+1; obj.randomize_test_data = varargin{i};
                        case {'randomizer_seed', 'randomize_seed'},...
                                i=i+1; obj.randomizer_seed = varargin{i};
                        
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
                    disp(['Evaluator.set(): Ignoring invalid argument #' num2str(i)]);
                    % fprintf(obj.UsageMessage);
                end

                i = i + 1;
            end                        
            
        end
        
        %% ------- *** ADD LEARNER *** ------------------------------------
        %******************************************************************
        %******************************************************************
        function obj = addLearner(obj, varargin)
            
            % Loop through arguments...
            i = 1;
            iPassedArgs = 1;
            while i <= length(varargin), 
                argok = 1; 
                if ischar(varargin{i}), 
                    switch lower(varargin{i}), 
                        case 'name', i=i+1;
                            if isempty(obj.LearnerNames)
                                obj.LearnerNames{1} = varargin{i};
                            else
                                obj.LearnerNames{end+1} = varargin{i};
                            end
                        
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
                    disp(['LOOCrossValidator.set(): Ignoring invalid argument #' num2str(i)]);
                    % fprintf(obj.UsageMessage);
                end

                i = i + 1;
            end            
            
            if isempty(obj.Learners)
                obj.Learners{1} = PassedArgs;
            else
                obj.Learners{end+1} = PassedArgs;
            end
        end
        
        %% ------- *** SAVE *** -------------------------------------------
        %******************************************************************
        % Save current state of experiment to disk...
        %******************************************************************
        function obj = save(obj, varargin)
            
            %% Save the data if required...
            if obj.doSave       
                fprintf('\n\nSaving data...\n\n');

                try
                    % If a directory is specified, save the file in there...
                    if strcmp(obj.SaveFileName(end), '/')            
                        % If the directory doesn't exist, create it...
                        if ~isdir(obj.SaveFileName)
                            mkdir(obj.SaveFileName)
                        end

                        % Save the file in the directory with a timestamp...
                        save([obj.SaveFileName 'Experiment-' datestr(clock, 'dd-mmm-yyyy-HH-MM-SS') '.mat'], 'obj');

                    % Otherwise, just save to the filename...
                    else
                        save(obj.SaveFileName, 'obj');
                    end
                    
                catch MyError
                    
                    fprintf('\nERROR: Unable to save data!\n');
                    
                end

            end
            
        end
        
    end
    
    methods (Static = true)
        
        %% ------- *** CREATE LEARNER *** ---------------------------------
        %******************************************************************
        %******************************************************************
        function LearnerObject = createLearner(varargin)
            
            % Defaults...
            Type = 'bimodal';
            
            % Loop through arguments...
            i = 1;
            iPassedArgs = 1;
            while i <= length(varargin), 
                argok = 1; 
                if ischar(varargin{i}), 
                    switch lower(varargin{i}), 
                        case {'type', 'learner_type'}, i=i+1; Type = varargin{i};
                        
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
                    disp(['Evaluator.createlearner(): Ignoring invalid argument #' num2str(i)]);
                    % fprintf(obj.UsageMessage);
                end

                i = i + 1;
            end
            
            switch lower(Type)
                case {'svm'}
                    LearnerObject = SVMLearner(PassedArgs{1:end});                   
                case {'rf'}
                    LearnerObject = RFLearner(PassedArgs{1:end});
                case {'lr'}
                    LearnerObject = LRLearner(PassedArgs{1:end});
                case {'grlvq'}
                    LearnerObject = GRLVQLearner(PassedArgs{1:end});
                case {'srng'}
                    LearnerObject = SRNGLearner(PassedArgs{1:end});
                case {'unilayer', 'unilayered'}
                    LearnerObject = UniLayeredLearner(PassedArgs{1:end});
                case 'bimodal',
                    LearnerObject = BiModalLearner(PassedArgs{1:end});
                otherwise,
                    LearnerObject = BiModalLearner(PassedArgs{1:end});
            end
            
        end
        
        %% ------- *** PROCESSRESULTS *** ---------------------------------
        %******************************************************************
        %******************************************************************
        function Results = processresults(Results,iCurrentTime)

            %% Collect evaluation data...

            % Trials...
            for iTrial = 1:length(Results.Trials)

                % Classes...
                for iClass = 1:length(Results.Trials{iTrial}.Classes)
                    
                    % Time...
                    for iTime = 1:length(Results.Trials{iTrial}.Classes{iClass}.Time)
                        
                        % Results for training data...
                        if isfield(Results.Trials{iTrial}.Classes{iClass}.Time{iTime}.TrainingData1Epoch, 'Results')
                            FieldNames = fieldnames(Results.Trials{iTrial}.Classes{iClass}.Time{iTime}.TrainingData1Epoch.Results);

                            for iField = 1:length(FieldNames)
                                if ~isempty(findstr(lower(FieldNames{iField}), 'percent')) ||...
                                   ~isempty(findstr(lower(FieldNames{iField}), 'mask')) ||...
                                   ~isempty(findstr(lower(FieldNames{iField}), 'optimalk'))

                                    if isfield(Results, ['Train_' FieldNames{iField}]) && iTime == 1
                                        Results.(['Train_' FieldNames{iField}])(iTrial, iClass, iTime:end) = NaN;
                                        Results.(['Train_' FieldNames{iField}])(iTrial:end, iClass:end, iTime:end) = NaN;
                                    end                                

                                    Results.(['Train_' FieldNames{iField}])(iTrial, iClass, iTime,:) =...
                                        Results.Trials{iTrial}.Classes{iClass}.Time{iTime}.TrainingData1Epoch.Results.(FieldNames{iField});                                

                                end
                                
                                % if ~isempty(findstr(lower(FieldNames{iField}), 'mask'))
                                % 
                                %     if isfield(Results, ['Train_' FieldNames{iField}]) && iTime == 1
                                %         Results.(['Train_' FieldNames{iField} 's'])(iTrial, iClass, iTime:end) = NaN;
                                %         Results.(['Train_' FieldNames{iField} 's'])(iTrial:end, iClass:end, iTime:end) = NaN;
                                %     end                                
                                % 
                                %     Results.(['Train_' FieldNames{iField} 's'])(iTrial, iClass, iTime,:) =...
                                %         Results.Trials{iTrial}.Classes{iClass}.Time{iTime}.TrainingData1Epoch.Results.(FieldNames{iField});                                
                                % 
                                % end
                            end
                        end
                        
                        % Results for test data...
                        if isfield(Results.Trials{iTrial}.Classes{iClass}.Time{iTime}.TestData, 'Results')
                            FieldNames = fieldnames(Results.Trials{iTrial}.Classes{iClass}.Time{iTime}.TestData.Results);

                            for iField = 1:length(FieldNames)
                                if ~isempty(findstr(lower(FieldNames{iField}), 'percent')) ||...
                                   ~isempty(findstr(lower(FieldNames{iField}), 'mask')) ||...
                                   ~isempty(findstr(lower(FieldNames{iField}), 'optimalk'))

                                    if isfield(Results, ['Test_' FieldNames{iField}]) && iTime == 1
                                        Results.(['Test_' FieldNames{iField}])(iTrial, iClass, iTime:end) = NaN;
                                        Results.(['Test_' FieldNames{iField}])(iTrial:end, iClass:end, iTime:end) = NaN;
                                    end                                

                                    Results.(['Test_' FieldNames{iField}])(iTrial, iClass, iTime,:) =...
                                        Results.Trials{iTrial}.Classes{iClass}.Time{iTime}.TestData.Results.(FieldNames{iField});

                                end
                                
                                % if ~isempty(findstr(lower(FieldNames{iField}), 'mask'))
                                % 
                                %     if isfield(Results, ['Test_' FieldNames{iField}]) && iTime == 1
                                %         Results.(['Test_' FieldNames{iField} 's'])(iTrial, iClass, iTime:end) = NaN;
                                %         Results.(['Test_' FieldNames{iField} 's'])(iTrial:end, iClass:end, iTime:end) = NaN;
                                %     end                                
                                % 
                                %     Results.(['Test_' FieldNames{iField} 's'])(iTrial, iClass, iTime,:) =...
                                %         Results.Trials{iTrial}.Classes{iClass}.Time{iTime}.TrainingData1Epoch.Results.(FieldNames{iField});                                
                                % 
                                % end
                            end
                        end                                                                                                
                        
                        % Results for relevance determination (feature
                        % selection)...
                        it_has_modalities = false;
                        ResultsFieldNames = fieldnames(Results.Trials{iTrial}.Classes{iClass}.Time{iTime});
                        for iResultsField = 1:size(ResultsFieldNames,1)
                            if strcmp(ResultsFieldNames{iResultsField},'Modalities') == 1
                                it_has_modalities = true;
                                break;
                            end
                        end
                        
                        if it_has_modalities
                            for iMod = 1:length(Results.Trials{iTrial}.Classes{iClass}.Time{iTime}.Modalities)                            
                                if ~isempty(Results.Trials{iTrial}.Classes{iClass}.Time{iTime}.Modalities{iMod}.SOM.mask)

    %                                 if isfield(Results, ['Mod' num2str(iMod) '_Masks']) && iTime == 1
    %                                     Results.(['Mod' num2str(iMod) '_Masks'])(iTrial, iClass, iTime:end,:) = NaN;
    %                                     Results.(['Mod' num2str(iMod) '_Masks'])(iTrial:end, iClass:end, iTime:end,:) = NaN;
    %                                 end

                                    Results.(['Mod' num2str(iMod) '_Masks'])(iTrial, iClass, iTime, :) =...
                                        Results.Trials{iTrial}.Classes{iClass}.Time{iTime}.Modalities{iMod}.SOM.mask;

                                end
                            end
                        end
                        
                    end
                end

            end
            
            %% Get means and standard deviations for data and print...
            FieldNames = fieldnames(Results);

            for iField = 1:length(FieldNames)
                iStr = findstr(lower(FieldNames{iField}), 'percent');
                if ~isempty(iStr)
                    
                    for iTime = 1:size(Results.(FieldNames{iField}),3)                                                               
                        Foo = Results.(FieldNames{iField})(:,:,iTime);
                        Results.([FieldNames{iField}(1:iStr(end)-1) 'Mean'])(iTime) = nanmean(Foo(:));
                        Results.([FieldNames{iField}(1:iStr(end)-1) 'STD'])(iTime) = nanstd(Foo(:));
                    end

                    fprintf([(FieldNames{iField}) '_Mean: %f +/- %f\n'],...
                            Results.([FieldNames{iField}(1:iStr(end)-1) 'Mean'])(iCurrentTime),...
                            Results.([FieldNames{iField}(1:iStr(end)-1) 'STD'])(iCurrentTime));
                    
                end
                
                iStr = findstr(lower(FieldNames{iField}), 'optimalk');
                if ~isempty(iStr)
                    
                    for iTime = 1:size(Results.(FieldNames{iField}),3)                                                               
                        Foo = Results.(FieldNames{iField})(:,:,iTime);
                        Results.([FieldNames{iField}(1:iStr(end)-1) 'Mean'])(iTime) = nanmean(Foo(:));
                        Results.([FieldNames{iField}(1:iStr(end)-1) 'STD'])(iTime) = nanstd(Foo(:));
                    end

                    fprintf([(FieldNames{iField}) '_Mean: %f +/- %f\n'],...
                            Results.([FieldNames{iField}(1:iStr(end)-1) 'Mean'])(iCurrentTime),...
                            Results.([FieldNames{iField}(1:iStr(end)-1) 'STD'])(iCurrentTime));
                    
                end
                
                % Results for relevance determination (feature
                % selection)...
                iStr = findstr(lower(FieldNames{iField}), 'masks');                
                
                if ~isempty(iStr)
                    
                    for iTime = 1:size(Results.(FieldNames{iField}),3)
                        
                        Foo = Results.(FieldNames{iField})(:,:,iTime,:);
                        SizeFoo = size(Foo);
                        FooData = reshape(Foo, prod(SizeFoo(1:3)), SizeFoo(4));
                        
                        if size(FooData,1) <= 1
                            Results.([FieldNames{iField}(1:iStr(end)+3) '_Mean'])(iTime,:) = FooData;
                            Results.([FieldNames{iField}(1:iStr(end)+3) '_STD'])(iTime,:) = FooData;
                        else
                            Results.([FieldNames{iField}(1:iStr(end)+3) '_Mean'])(iTime,:) = nanmean(FooData);
                            Results.([FieldNames{iField}(1:iStr(end)+3) '_STD'])(iTime,:) = nanstd(FooData);
                        end
                        
                    end
                    
                    FormatString = ' ';
                    for iDim = 1:size(Results.([FieldNames{iField}(1:iStr(end)+3) '_Mean']), 2)
                        FormatString = [FormatString '%0.2f '];
                    end

                    fprintf([FieldNames{iField}(1:iStr(end)+3) '_Mean:' FormatString '\n'],...
                            Results.([FieldNames{iField}(1:iStr(end)+3) '_Mean'])(iCurrentTime,:));
                    fprintf([FieldNames{iField}(1:iStr(end)+3) '_STD:' FormatString '\n'],...
                            Results.([FieldNames{iField}(1:iStr(end)+3) '_STD'])(iCurrentTime,:));
                    
                end
            end                                    

        end
        
    end
    
    methods (Access = protected)
        
        %% ------- *** COMPUTELEARNERJOB *** ------------------------------
        %******************************************************************
        %******************************************************************
        function LearnerObject = computelearnerjob(obj, LearnerObject, iTime)
            
            % Train...
            if ~obj.doParallel
                fprintf('\nTRAINING...');
                tic;
            end
            
            if size(obj.test_interval_timesteps,2) > 1
                if iTime > 1
                    if obj.test_interval_timesteps(iTime) == Inf
                        
                        if obj.replicate_training_data
                            LearnerObject = LearnerObject.train();
                        else
                            LearnerObject = LearnerObject.train(obj.TrainingData);
                        end
                        
                    else
                        if obj.replicate_training_data                            
                            LearnerObject = LearnerObject.train(obj.test_interval_timesteps(iTime) -...
                                                                obj.test_interval_timesteps(iTime-1));
                        else
                            LearnerObject = LearnerObject.train(obj.TrainingData,...
                                                                obj.test_interval_timesteps(iTime) -...
                                                                obj.test_interval_timesteps(iTime-1));
                        end
                    end
                else
                    if obj.replicate_training_data
                        LearnerObject = LearnerObject.train(obj.test_interval_timesteps(iTime));
                    else
                        LearnerObject = LearnerObject.train(obj.TrainingData,...
                                                            obj.test_interval_timesteps(iTime));
                    end
                end
                
            elseif obj.test_interval_timesteps == Inf
                
                if obj.replicate_training_data
                    LearnerObject = LearnerObject.train();
                else
                    LearnerObject = LearnerObject.train(obj.TrainingData);
                end
                
            else
                
                if obj.replicate_training_data
                    LearnerObject = LearnerObject.train(obj.test_interval_timesteps);
                else
                    LearnerObject = LearnerObject.train(obj.TrainingData,...
                                                        obj.test_interval_timesteps);
                end
                
            end
            
            if ~obj.doParallel
                fprintf('\n');
                toc;
            end

            % Classify...
            if ~obj.doParallel
                fprintf('CLASSIFYING...');
            end            
            % LearnerObject.classify();
            if ~strcmp(class(LearnerObject), 'UniLayeredLearner') &&...
               ~strcmp(class(LearnerObject), 'SVMLearner') &&...
               ~strcmp(class(LearnerObject), 'RFLearner') &&...
               ~strcmp(class(LearnerObject), 'LRLearner') &&...
               ~strcmp(class(LearnerObject), 'GRLVQLearner') &&...
               ~strcmp(class(LearnerObject), 'SRNGLearner')
                LearnerObject = LearnerObject.clearclasslabels();
            end
            
            if obj.replicate_training_data
                                
                [LearnerObject.TrainingData1Epoch LearnerObject.TrainingData1Epoch.Results.ClassificationMask] =...
                    LearnerObject.classify(LearnerObject.TrainingData1Epoch);
                [LearnerObject.TestData LearnerObject.TestData.Results.ClassificationMask] =...
                    LearnerObject.classify(LearnerObject.TestData, 'retain_clustering');
                
                % Evaluate...
                if ~obj.doParallel
                    fprintf('\nEVALUATING...');
                end
                
                LearnerObject.TrainingData1Epoch = LearnerObject.evaluate(LearnerObject.TrainingData1Epoch);
                LearnerObject.TestData = LearnerObject.evaluate(LearnerObject.TestData);
                
            else
                
                TempTrainingData1Epoch = LearnerObject.classify(obj.TrainingData1Epoch);                                
                
                TempTestData = LearnerObject.classify(obj.TestData, 'retain_clustering');                                
                
                % Evaluate...
                if ~obj.doParallel
                    fprintf('\nEVALUATING...');
                end
                
                TempTrainingData1Epoch =...
                    LearnerObject.evaluate('trainingdata', obj.TrainingData,...
                                           'trainingdata1epoch', obj.TrainingData1Epoch,...
                                           'testdata', TempTrainingData1Epoch);
                LearnerObject.TrainingData1Epoch.Results = TempTrainingData1Epoch.Results;                
                                       
                TempTestData =...
                    LearnerObject.evaluate('trainingdata', obj.TrainingData,...
                                           'trainingdata1epoch', obj.TrainingData1Epoch,...
                                           'testdata', TempTestData);
                LearnerObject.TestData.Results = TempTestData.Results;
                
            end           
            
        end
        
        %% ------- *** RECORDLEARNERSTATE *** -------------------------------
        %******************************************************************
        %******************************************************************
        function Results = recordlearnerstate(obj, LearnerObject, Results, iTrial, iClass, iTime, nIntervals)
            
            % Save the timestep...
            if isfield(LearnerObject, 'Modalities')
                Results.Trials{iTrial}.Classes{iClass}.Time{iTime}.t =...
                    LearnerObject.Modalities{1}.t;
            else
                Results.Trials{iTrial}.Classes{iClass}.Time{iTime}.t = 1;
            end

            % Save the learner objects for debugging if
            % required...
            if obj.debug_level == 2 && iTime == nIntervals
                
                % Save the learner object...
                Results.Trials{iTrial}.Classes{iClass}.Time{iTime} =...
                    LearnerObject.copy();
                
                % Save the results, throw away the training and test
                % data unless we're at the end of training...
                if isfield(Results.Trials{iTrial}.Classes{iClass}.Time{iTime}.TrainingData1Epoch, 'Results')
                    TrainingData1Epoch.Results = Results.Trials{iTrial}.Classes{iClass}.Time{iTime}.TrainingData1Epoch.Results;
                    Results.Trials{iTrial}.Classes{iClass}.Time{iTime}.TrainingData1Epoch = TrainingData1Epoch;
                end
                
                if isfield(Results.Trials{iTrial}.Classes{iClass}.Time{iTime}.TestData, 'Results')
                    TestData.Results = Results.Trials{iTrial}.Classes{iClass}.Time{iTime}.TestData.Results;
                    Results.Trials{iTrial}.Classes{iClass}.Time{iTime}.TestData = TestData;
                end                                
                
            elseif obj.debug_level == 3
                
                % Save the learner object...
                Results.Trials{iTrial}.Classes{iClass}.Time{iTime} =...
                    LearnerObject.copy();
                
                % Save the results, but throw away the training and test
                % data...
                if isfield(Results.Trials{iTrial}.Classes{iClass}.Time{iTime}.TrainingData1Epoch, 'Results')
                    TrainingData1Epoch.Results = Results.Trials{iTrial}.Classes{iClass}.Time{iTime}.TrainingData1Epoch.Results;
                    Results.Trials{iTrial}.Classes{iClass}.Time{iTime}.TrainingData1Epoch = TrainingData1Epoch;
                end
                
                if isfield(Results.Trials{iTrial}.Classes{iClass}.Time{iTime}.TestData, 'Results')
                    TestData.Results = Results.Trials{iTrial}.Classes{iClass}.Time{iTime}.TestData.Results;
                    Results.Trials{iTrial}.Classes{iClass}.Time{iTime}.TestData = TestData;                
                end

            elseif obj.debug_level == 4 && iTime == nIntervals
                
                % Save the entire learner object...
                Results.Trials{iTrial}.Classes{iClass}.Time{iTime} =...
                    LearnerObject.copy();

            elseif obj.debug_level == 5
                
                % WARNING: This debug level is truly pathalogical and will
                % lead to large memory consumption!
                
                % Save the entire learner object...
                Results.Trials{iTrial}.Classes{iClass}.Time{iTime} =...
                    LearnerObject.copy();
                            
            elseif obj.debug_level == 1
                
                % Record any modality feature masks...
                for iMod = 1:length(LearnerObject.Modalities)
                    Results.Trials{iTrial}.Classes{iClass}.Time{iTime}.Modalities{iMod}.SOM.mask =...
                        LearnerObject.Modalities{iMod}.SOM.mask;
                end
                
                if isfield(LearnerObject.TrainingData1Epoch, 'Results')
                    Results.Trials{iTrial}.Classes{iClass}.Time{iTime}.TrainingData1Epoch.Results =...
                        LearnerObject.TrainingData1Epoch.Results;                                                            
                end
                
                if isfield(LearnerObject.TestData, 'Results')
                    Results.Trials{iTrial}.Classes{iClass}.Time{iTime}.TestData.Results =...
                        LearnerObject.TestData.Results;                    
                end            
                
            % ...otherwise, just save the statistical results and throw away match
            % information.
            else
                % Record any modality feature masks...
                for iMod = 1:length(LearnerObject.Modalities)
                    Results.Trials{iTrial}.Classes{iClass}.Time{iTime}.Modalities{iMod}.SOM.mask =...
                        LearnerObject.Modalities{iMod}.SOM.mask;
                end
                
                if isfield(LearnerObject.TrainingData1Epoch, 'Results')
                    Results.Trials{iTrial}.Classes{iClass}.Time{iTime}.TrainingData1Epoch.Results =...
                        LearnerObject.TrainingData1Epoch.Results;
                    
                    % Throw away the sample-by-sample match results,
                    % because they take up too much memory in online
                    % evaluations...
                    Results.Trials{iTrial}.Classes{iClass}.Time{iTime}.TrainingData1Epoch.Results.InToOutClassification = [];
                    Results.Trials{iTrial}.Classes{iClass}.Time{iTime}.TrainingData1Epoch.Results.ClassProbs = [];
                    Results.Trials{iTrial}.Classes{iClass}.Time{iTime}.TrainingData1Epoch.Results.Mod2_To_Cluster_GT_Matches = [];
                    Results.Trials{iTrial}.Classes{iClass}.Time{iTime}.TrainingData1Epoch.Results.Mod1_To_Cluster_GT_Matches = [];
                    Results.Trials{iTrial}.Classes{iClass}.Time{iTime}.TrainingData1Epoch.Results.Mod1_To_Cluster_Matches = [];
                    Results.Trials{iTrial}.Classes{iClass}.Time{iTime}.TrainingData1Epoch.Results.Mod1_To_Cluster_Matches_Corresponding_To_GT = [];
                    Results.Trials{iTrial}.Classes{iClass}.Time{iTime}.TrainingData1Epoch.Results.Matches = [];
                    
                end
                
                if isfield(LearnerObject.TestData, 'Results')
                    Results.Trials{iTrial}.Classes{iClass}.Time{iTime}.TestData.Results =...
                        LearnerObject.TestData.Results;
                    
                    % Throw away the sample-by-sample match results,
                    % because they take up too much memory in online
                    % evaluations...
                    Results.Trials{iTrial}.Classes{iClass}.Time{iTime}.TestData.Results.InToOutClassification = [];
                    Results.Trials{iTrial}.Classes{iClass}.Time{iTime}.TestData.Results.ClassProbs = [];
                    Results.Trials{iTrial}.Classes{iClass}.Time{iTime}.TestData.Results.Mod2_To_Cluster_GT_Matches = [];
                    Results.Trials{iTrial}.Classes{iClass}.Time{iTime}.TestData.Results.Mod1_To_Cluster_GT_Matches = [];
                    Results.Trials{iTrial}.Classes{iClass}.Time{iTime}.TestData.Results.Mod1_To_Cluster_Matches = [];
                    Results.Trials{iTrial}.Classes{iClass}.Time{iTime}.TestData.Results.Mod1_To_Cluster_Matches_Corresponding_To_GT = [];
                    Results.Trials{iTrial}.Classes{iClass}.Time{iTime}.TestData.Results.Matches = [];
                    
                end
            end
            
            % Results...
            fprintf('\nCURRENT RESULTS for %s:\n', LearnerObject.Name);
            Results = obj.processresults(Results,iTime);
            
        end
        
    end
end