classdef CodebookAlgorithm < handle & Utils
    
    properties
        
        %% ------- *** OBJECTS *** ----------------------------------------
        %******************************************************************
        %******************************************************************
        Updaters = [];
        
        %% ------- *** FUNCTION HANDLES *** -------------------------------
        %******************************************************************
        %******************************************************************
        gettime = [];
        
        %% ------- *** PROPERTIES *** -------------------------------------
        %******************************************************************
        %******************************************************************
        
        %% Current algorithm values...
        %-----------------------------
        % Training   struct for different updaters...
        UpdaterPhases = [];
        iUpdaterPhase = 1;
        
        % Training phases struct for alpha learning rate...
        AlphaPhases = [];
        iAlphaPhase = 1;
        
        % Training phases struct for feature relevance alpha learning rate...
        AlphaFeaturePhases = [];
        iAlphaFeaturePhase = 1;
        
        % Training phases struct for activation radius...
        RadiusPhases = [];
        iRadiusPhase = 1;
        
        % Training phases struct for activation types...
        ActivationTypePhases = [];
        iActivationTypePhase = 1;
        
        % Window size for LVQ2.1-style algorithms...
        %
        % NOTE: I should change this to be like the others later...
        %
        WindowSizes = {NaN};
                
        % Current training phase...
        currentphase = 0;
        
        % Flag for neighbourhood radius calculation...
        calc_radius = true;
        
        % Flag for neighbourhood calculation...
        calc_neighbourhood = true;
        
        % Flag for learning rate calculation...
        calc_learning_rate = true;
        
        % Feature selection learning rate...
        calc_feature_selection_learning_rate = true;
        
        % Should BMUs be cleared internally?
        clearbmu_flag = false;
        
        % Should we calculate node activations?
        % (we do this elsewhere for the SOMN algorithm)
        calc_activations = true;
        
    end
    
    methods
        
        %% --------- *** CONSTRUCTOR *** ----------------------------------
        % *****************************************************************
        % *****************************************************************
        function obj = CodebookAlgorithm(Mod, varargin)
            obj.set(Mod, varargin{1:end});
        end

        %% --------- *** SET *** ------------------------------------------
        % *****************************************************************
        % Apply settings.
        % *****************************************************************
        function set(obj, Mod, varargin)
            
            %% Default algorithm settings...
            %-------------------------
            % Update rules...
            UpdaterTypes = {'SOM'};
            % Learning phase switchover point...
            PhaseShifts = {NaN};
            % Algorithm learning rates...
            AlphaTypes = {'inv'};
            AlphaInits = {1};
            % Algorithm learning rates...
            RadiusTypes = {'linear'};
            RadiusInits = {5};
            RadiusFins = {1};
            % Feature selection learning rates...
            AlphaFeatureTypes = {NaN};
            AlphaFeatureInits = {NaN};
            % Activation type...
            ActivationTypes = {'response'};
            
            % Training phases for different updaters...
            obj.UpdaterPhases{1}.shift_t = 1;
            obj.UpdaterPhases{1}.duration = NaN;
            obj.UpdaterPhases{1}.t = 0;
            
            % Training phases for different updaters...
            obj.AlphaPhases{1}.shift_t = 1;
            obj.AlphaPhases{1}.duration = NaN;
            obj.AlphaPhases{1}.t = 0;
            
            % Training phases for different updaters...
            obj.AlphaFeaturePhases{1}.shift_t = 1;
            obj.AlphaFeaturePhases{1}.duration = NaN;
            obj.AlphaFeaturePhases{1}.t = 0;
            
            % Training phases for different updaters...
            obj.RadiusPhases{1}.shift_t = 1;
            obj.RadiusPhases{1}.duration = NaN;
            obj.RadiusPhases{1}.t = 0;
            
            % Training phases for different updaters...
            obj.ActivationTypePhases{1}.shift_t = 1;
            obj.ActivationTypePhases{1}.duration = NaN;
            obj.ActivationTypePhases{1}.t = 0;
            
            % Loop through arguments...
            i = 1;
            iPassedArgs = 1;
            while i <= length(varargin), 
                argok = 1; 
                if ischar(varargin{i}), 
                    switch lower(varargin{i}), 
                        case {'updater_type', 'updater_types',...
                              'updatertype', 'updatertypess', 'updater', 'updaters'},
                                i=i+1; UpdaterTypes = lower(varargin{i});
                        case {'phase_shift', 'phase_shifts',...
                              'phaseshift', 'phaseshifts', 'shift', 'shifts'},...
                                i=i+1; PhaseShifts = varargin{i};
                        case {'alpha_type', 'alpha_types',...
                              'alphatype', 'alphatypes'},...
                                i=i+1; AlphaTypes = varargin{i};
                        case {'alpha_init', 'alpha_inits', 'alphainit', 'alphainits',...
                              'alpha_ini', 'alpha_inis', 'alphaini', 'alphainis'},...
                                i=i+1; AlphaInits = varargin{i};
                        case {'radius_type', 'radius_types',...
                              'radiustype', 'radiustypes'},...
                                i=i+1; RadiusTypes = varargin{i};
                        case {'radius_init', 'radius_inits', 'radiusinit', 'radiusinits',...
                              'radius_ini', 'radius_inis', 'radiusini', 'radiusinis'},...
                                i=i+1; RadiusInits = varargin{i};
                        case {'radius_fin', 'radius_fins', 'radiusfin', 'radiusfins'},...
                                i=i+1; RadiusFins = varargin{i};
                        case {'window_size', 'window_sizes',...
                              'windowsize', 'windowsizes', 'window', 'windows'},...
                                i=i+1; obj.WindowSizes = varargin{i};
                        case {'alpha_feature_type', 'alpha_feature_types',...
                              'alphafeaturetype', 'alphafeaturetypes'},...
                                i=i+1; AlphaFeatureTypes = varargin{i};
                        case {'alpha_feature_init', 'alpha_feature_inits',...
                              'alpha_feature_ini', 'alpha_feature_inis',...
                              'alphafeatureinit', 'alphafeaturreinits',...
                              'alphafeatureini', 'alphafeaturreinis'},...
                                i=i+1; AlphaFeatureInits = varargin{i};
                        case {'activation_type', 'activation_types',...
                              'activationtype', 'activationtypes'},...
                                i=i+1; ActivationTypes = varargin{i};
                        
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
                    disp(['CodebookAlgorithm.set(): Ignoring invalid argument #' num2str(i)]);
                    fprintf(obj.UsageMessage);
                end

                i = i + 1;
            end
            
            % Create update rule objects...
            for i = 1:length(UpdaterTypes)
                
                obj.Updaters{i} = obj.createUpdateRule(UpdaterTypes{i});
                obj.UpdaterPhases{i}.type = UpdaterTypes{i};
                
            end
            
            % Should we calculate node activations?
            % (we do this elsewhere for the SOMN algorithm)
%             switch lower(obj.UpdaterPhases{1}.type)
%                 case {'somn'}
%                     obj.calc_activations = false;
%             end
            
            % For algorithms with an optimized learning rate,
            % we set individual alpha learning rates for each
            % codebook vector...
            switch obj.UpdaterPhases{1}.type
                case {'olvq1', 'olvq', 'orlvq1', 'orlvq', 'laorlvq',...
                      'heurolvq', 'heurfolvq', 'heurorlvq', 'heurforlvq', 'heurfolrlvq',...
                      'heurolvq1', 'heurorlvq1', 'heurforlvq1', 'heurfolrlvq1',...
                      'ldaolvq', 'laldaolvq', 'lcaldaolvq', 'ldaolvq2', 'ldaolvq3', 'ldaoglvq', 'ldaoglvq_3',...
                      'heurfldaolvq', 'heurfldaolvq3'},
                    Mod.Alphas = AlphaInits{1} * ones(Mod.Size);
            end
                        
            % Phase training shifts in timesteps and phase durations in timesteps...
            if any(~isnan([PhaseShifts{1:end}]))
                for i = 1:(length(PhaseShifts) + 1)                    
                    if i > 1
                        obj.UpdaterPhases{i}.shift_t = (Mod.trainlen / (1/PhaseShifts{i-1})) + 1;
                        obj.UpdaterPhases{i-1}.duration = obj.UpdaterPhases{i}.shift_t - obj.UpdaterPhases{i-1}.shift_t;
                        obj.UpdaterPhases{i-1}.t = 0;
                    end                    
                end
                
                obj.UpdaterPhases{i+1}.shift_t = Inf;
                obj.UpdaterPhases{i}.duration = (Mod.trainlen + 1) - obj.UpdaterPhases{i}.shift_t;
                obj.UpdaterPhases{i+1}.duration = 0;
                obj.UpdaterPhases{i}.t = 0;
                obj.UpdaterPhases{i+1}.t = 0;
                
            else
                obj.UpdaterPhases{2}.shift_t = Inf;
                obj.UpdaterPhases{1}.duration = Mod.trainlen;
                obj.UpdaterPhases{2}.duration = 0;
                obj.UpdaterPhases{1}.t = 0;
                obj.UpdaterPhases{2}.t = 0;
            end
            
            % obj.UpdaterPhases{i}.shift_t and obj.UpdaterPhases{i}.duration
            % will now be canon for the following assignments.
            %
            % Alpha Phases...
            for i = 1:length(AlphaTypes)
                
                obj.AlphaPhases{i}.type = AlphaTypes{i};
                obj.AlphaPhases{i}.initial_value = AlphaInits{i};
                obj.AlphaPhases{i}.t = 0;
                
                if i < length(obj.AlphaPhases{i}.type)
                    obj.AlphaPhases{i}.shift_t = obj.UpdaterPhases{i}.shift_t;
                    obj.AlphaPhases{i}.duration = obj.UpdaterPhases{i}.duration;
                else
                    obj.AlphaPhases{i}.shift_t = obj.UpdaterPhases{i}.shift_t;
                    
                    % Sum up the remaining durations...
                    obj.RadiusPhases{i}.duration = 0;
                    for iPhase = i:length(obj.UpdaterPhases)
                        obj.AlphaPhases{i}.duration = obj.AlphaPhases{i}.duration + obj.UpdaterPhases{iPhase}.duration;
                    end
                    
                end
            end
            % Radius Phases...
            for i = 1:length(RadiusTypes)
                
                obj.RadiusPhases{i}.type = RadiusTypes{i};
                obj.RadiusPhases{i}.initial_value = RadiusInits{i};
                obj.RadiusPhases{i}.final_value = RadiusFins{i};
                obj.RadiusPhases{i}.t = 0;
                
                if i < length(obj.RadiusPhases{i}.type)
                    obj.RadiusPhases{i}.shift_t = obj.UpdaterPhases{i}.shift_t;
                    obj.RadiusPhases{i}.duration = obj.UpdaterPhases{i}.duration;
                else
                    obj.RadiusPhases{i}.shift_t = obj.UpdaterPhases{i}.shift_t;
                    
                    % Sum up the remaining durations...
                    obj.RadiusPhases{i}.duration = 0;
                    for iPhase = i:length(obj.UpdaterPhases)
                        obj.RadiusPhases{i}.duration = obj.RadiusPhases{i}.duration + obj.UpdaterPhases{iPhase}.duration;
                    end
                    
                end
            end
            % AlphaFeature Phases...
            for i = 1:length(AlphaFeatureTypes)
                
                obj.AlphaFeaturePhases{i}.type = AlphaFeatureTypes{i};
                obj.AlphaFeaturePhases{i}.initial_value = AlphaFeatureInits{i};
                obj.AlphaFeaturePhases{i}.t = 0;
                
                if i < length(obj.AlphaFeaturePhases{i}.type)
                    obj.AlphaFeaturePhases{i}.shift_t = obj.UpdaterPhases{i}.shift_t;
                    obj.AlphaFeaturePhases{i}.duration = obj.UpdaterPhases{i}.duration;
                else
                    obj.AlphaFeaturePhases{i}.shift_t = obj.UpdaterPhases{i}.shift_t;
                    
                    % Sum up the remaining durations...
                    obj.AlphaFeaturePhases{i}.duration = 0;
                    for iPhase = i:length(obj.UpdaterPhases)
                        obj.AlphaFeaturePhases{i}.duration = obj.AlphaFeaturePhases{i}.duration + obj.UpdaterPhases{iPhase}.duration;
                    end
                end
            end
            
            % ActivationType Phases...
            for i = 1:length(ActivationTypes)
                
                obj.ActivationTypePhases{i}.type = ActivationTypes{i};
                obj.ActivationTypePhases{i}.t = 0;
                
                if i < length(obj.AlphaFeaturePhases{i}.type)
                    obj.AlphaFeaturePhases{i}.shift_t = obj.UpdaterPhases{i}.shift_t;
                    obj.AlphaFeaturePhases{i}.duration = obj.UpdaterPhases{i}.duration;
                else
                    obj.AlphaFeaturePhases{i}.shift_t = obj.UpdaterPhases{i}.shift_t;
                    
                    % Sum up the remaining durations...
                    obj.AlphaFeaturePhases{i}.duration = 0;
                    for iPhase = i:length(obj.UpdaterPhases)
                        obj.AlphaFeaturePhases{i}.duration = obj.AlphaFeaturePhases{i}.duration + obj.UpdaterPhases{iPhase}.duration;
                    end
                end
            end
                        
%             % Set window size for LVQ2.1-style algorithms...
%             Mod.epsilon = obj.WindowSizes{obj.currentphase};
%             
%             % Point the gettime function handle somewhere...
%             switch obj.AlphaPhases{obj.currentphase}.type
%                 case 'inv_nonnull',
%                     obj.gettime = @obj.getnonnulltime;
%                 otherwise
%                     obj.gettime = @obj.getrealtime;
%             end
            
        end
        
        %% --------- *** RUN *** ------------------------------------------
        % *****************************************************************
        % Template method for running algorithms on
        % CodebookModality objects.
        % *****************************************************************
        function Mod = run(obj, Mod, Data, varargin)
            
            %% CREATE SOM DATA STRUCT & NORMALIZE DATA --------------------
            %--------------------------------------------------------------
            if ~isfield(Data, 'NormedFeatureVectors')
                % Temporary data struct...    
                SOMData = som_data_struct(Data.FeatureVectors',...
                                          'comp_names', Data.FeatureNames,...
                                          'label_names', Data.ClassNames);
                % Normalize...
                switch Mod.NormMethod
                    case 'none',
                        SOMData = som_normalize(SOMData, Mod.NormStruct);
                    otherwise
                        SOMData = SOMData;
                end
                
                % Data...
                % data_name = SOMData.name;
                SOMData = SOMData.data;
            else
                SOMData = Data.NormedFeatureVectors';
            end
            
            % Remove empty vectors from the data...
            % SOMData = SOMData(find(sum(isnan(SOMData),2) < Mod.dim),:);
            % SOMData = SOMData(sum(isnan(SOMData),2) < Mod.dim,:);
            
            % Check input dimension...
            [dlen ddim] = size(SOMData);
            if Mod.dim ~= ddim
                error('Map and data input space dimensions disagree.');
            end
            
            % If this method gets passed more than one data sample, we'll
            % need to clear the BMUs after each iteration of the training
            % loop herein.
            if size(SOMData,1) > 1
                obj.clearbmu_flag = true;
            else
                obj.clearbmu_flag = false;
            end
            
            %% TRAINING LOOP ----------------------------------------------
            %--------------------------------------------------------------
            for iData = 1:size(SOMData,1)
                
                % Increment global timestep...
                Mod.t = Mod.t + 1;
                
                % Move to the next phase if necessary...
                if Mod.t >= round(obj.UpdaterPhases{obj.currentphase + 1}.shift_t)

                    % Advance to the next training phase...
                    obj.currentphase = obj.currentphase + 1;                    
                    
                    obj.iUpdaterPhase = min(length(obj.UpdaterPhases), obj.currentphase);
                    obj.iAlphaPhase = min(length(obj.AlphaPhases), obj.currentphase);
                    obj.iAlphaFeaturePhase = min(length(obj.AlphaFeaturePhases), obj.currentphase);
                    obj.iRadiusPhase = min(length(obj.RadiusPhases), obj.currentphase);
                    obj.iActivationTypePhase = min(length(obj.ActivationTypePhases), obj.currentphase);
                    
                    % Should we calculate radius?
                    if ~isnan(obj.RadiusPhases{min(end,obj.currentphase)}.type)
                        switch obj.RadiusPhases{min(end,obj.currentphase)}.type
                            case 'constant'
                                Mod.r = obj.RadiusPhases{min(end,obj.currentphase)}.initial_value;
                                obj.calc_radius = false;
                            otherwise
                                obj.calc_radius = true;
                        end
                    else
                        obj.calc_radius = false;
                    end
                    
                    % Should we calculate neighbourhood?
                    if ~isnan(Mod.Neigh)                        
                        obj.calc_neighbourhood = true;
                    else
                        obj.calc_neighbourhood = false;
                    end
                    
                    % Should we calculate learning rate?
                    if ~isnan(obj.AlphaPhases{min(end,obj.currentphase)}.type)
                        switch obj.AlphaPhases{min(end,obj.currentphase)}.type
                            case 'constant'
                                Mod.a = obj.AlphaPhases{min(end,obj.currentphase)}.initial_value;
                                obj.calc_learning_rate = false;
                            otherwise
                                obj.calc_learning_rate = true;
                        end
                    else
                        obj.calc_learning_rate = false;
                    end
                    
                    % Should we calculate the feature selection learning rate?
                    if ~isnan(obj.AlphaFeaturePhases{min(end,obj.currentphase)}.type)
                        switch obj.AlphaFeaturePhases{min(end,obj.currentphase)}.type
                            case 'constant'
                                Mod.a_f = obj.AlphaFeaturePhases{min(end,obj.currentphase)}.initial_value;
                                obj.calc_feature_selection_learning_rate = false;
                            otherwise
                                obj.calc_feature_selection_learning_rate = true;
                        end
                    else
                        obj.calc_feature_selection_learning_rate = false;
                    end
                    
                    % Should we calculate node activations?
                    % (we do this elsewhere for the SOMN algorithm)
%                     switch lower(obj.UpdaterPhases{min(end,obj.currentphase)}.type)
%                         case {'somn'}
%                             obj.calc_activations = false;
%                     end
                    
                    % For algorithms with an optimized learning rate,
                    % we set individual alpha learning rates for each
                    % codebook vector...
                    switch obj.UpdaterPhases{min(end,obj.currentphase)}.type
                        case {'olvq1', 'olvq', 'orlvq1', 'orlvq', 'laorlvq',...
                              'heurolvq', 'heurfolvq', 'heurorlvq', 'heurforlvq', 'heurfolrlvq',...
                              'heurolvq1', 'heurorlvq1', 'heurforlvq1', 'heurfolrlvq1',...
                              'ldaolvq', 'laldaolvq', 'lcaldaolvq', 'ldaolvq2', 'ldaolvq3', 'ldaoglvq', 'ldaoglvq_3',...
                              'heurfldaolvq', 'heurfldaolvq3'}
                            Mod.Alphas = obj.AlphaPhases{min(end,obj.currentphase)}.initial_value * ones(Mod.Size);
                    end
                    
                    % Change window size for LVQ2.1-style algorithms...
                    switch obj.UpdaterPhases{min(end,obj.currentphase)}.type
                        case {'heurfmamr', 'heurmamr', 'heurlvq3', 'lvq3'},
                            Mod.epsilon = obj.WindowSizes{obj.currentphase};
                    end
                    
                    % Point the gettime function handle somewhere...
                    switch obj.AlphaPhases{min(end,obj.currentphase)}.type
                        case 'inv_nonnull',
                            obj.gettime = @obj.getnonnulltime;
                        otherwise
                            obj.gettime = @obj.getrealtime;
                    end
                    
                end                                                                                                

                % Get data sample......
                Mod.x = SOMData(iData,:);           % pick it up
                Mod.known = ~isnan(Mod.x);          % its known components

                % Find BMU...
                if isempty(Mod.BMUs)
                    % Find & save BMU in Modality object...
                    Mod.BMUs = Mod.findbmus(Mod.x);
                end
                
                %% UPDATE LEARNING RATES ETC. -----------------------------
                %----------------------------------------------------------
                % Radius...
                if obj.calc_radius
                    
                    obj.RadiusPhases{obj.iRadiusPhase}.t = obj.RadiusPhases{obj.iRadiusPhase}.t + 1;
                    
                    Mod.r = Utils.radius(obj.RadiusPhases{obj.iRadiusPhase}.type,...              % type
                                         obj.RadiusPhases{obj.iRadiusPhase}.initial_value,...     % initial radius
                                         obj.RadiusPhases{obj.iRadiusPhase}.final_value,...       % final radius
                                         obj.RadiusPhases{obj.iRadiusPhase}.duration,...          % length of training phase
                                         obj.RadiusPhases{obj.iRadiusPhase}.shift_t - 1,...       % shift timestep
                                         Mod.t);                                                  % current global timestep
                end

                % Neighbourhood...
                if obj.calc_neighbourhood
                    
                    Mod.h = Utils.neighbourhood(Mod.Neigh,...    % type
                                                Mod.Ud,...       % universal distance matrix
                                                Mod.BMUs(1),...  % best matching unit
                                                Mod.r);          % radius                                                                
                end

                % Learning rate...
                if obj.calc_learning_rate
                    
                    obj.AlphaPhases{obj.iAlphaPhase}.t = obj.AlphaPhases{obj.iAlphaPhase}.t + 1;
                    
                    Mod.a = Utils.learningrate(obj.AlphaPhases{obj.iAlphaPhase}.type,...             % type
                                               obj.AlphaPhases{obj.iAlphaPhase}.initial_value,...    % seed
                                               obj.AlphaPhases{obj.iAlphaPhase}.duration,...         % training length
                                               obj.AlphaPhases{obj.iAlphaPhase}.shift_t - 1,...      % shift timestep
                                               obj.gettime(Mod));                                    % current global timestep
                end
                    
                                       
                % Feature selection learning rate...
                if obj.calc_feature_selection_learning_rate
                    
                    obj.AlphaFeaturePhases{obj.iAlphaFeaturePhase}.t = obj.AlphaFeaturePhases{obj.iAlphaFeaturePhase}.t + 1;
                    
                    Mod.a_f = Utils.learningrate(obj.AlphaFeaturePhases{obj.iAlphaFeaturePhase}.type,...              % type
                                                 obj.AlphaFeaturePhases{obj.iAlphaFeaturePhase}.initial_value,...     % seed
                                                 obj.AlphaFeaturePhases{obj.iAlphaFeaturePhase}.duration,...          % training length
                                                 obj.AlphaFeaturePhases{obj.iAlphaFeaturePhase}.shift_t - 1,...       % shift timestep
                                                 Mod.t);                                                              % current global timestep
                end

                
                %% UPDATE THE MODALITY OBJECT -----------------------------
                %----------------------------------------------------------
                obj.UpdaterPhases{obj.iUpdaterPhase}.t = obj.UpdaterPhases{obj.iUpdaterPhase}.t + 1;
                Mod = obj.Updaters{obj.iUpdaterPhase}.update(obj, Mod, Data, iData);
                
                
                if any(any(isnan(Mod.SOM.codebook)))
                    error('\nCodebook contains NaNs!\n');
                end
                
                
                %% UPDATE CODEBOOK ACTIVATIONS ---------------------------- 
                %----------------------------------------------------------
                % Calculate unit activations for BMUs...
                if obj.calc_activations
                    
                    obj.ActivationTypePhases{obj.iActivationTypePhase}.t = obj.ActivationTypePhases{obj.iActivationTypePhase}.t + 1;
                    
                    Mod.Activations = Utils.activations(obj.ActivationTypePhases{obj.iActivationTypePhase}.type,...                                                        
                                                        Mod.Ud,...
                                                        Mod.Dx,...
                                                        Mod.Cov,...
                                                        Mod.CovInv,...
                                                        Mod.Distances,...
                                                        Mod.BMUs(1),...
                                                        Mod.rini/2);
                end
                
                %% CLEAR BMUS ---------------------------------------------
                %----------------------------------------------------------
                if obj.clearbmu_flag
                    Mod.clearbmus();
                end
                
            end
            
        end
                        
    end
    
    methods (Static = true)
        
        %% --------- *** CREATEUPDATERULE *** -----------------------------
        % *****************************************************************
        % Factory method for creating UpdateRule objects
        % *****************************************************************
        function Updater = createUpdateRule(type)
            
            switch type,

                    % SOM...
                    case {'som', 'lsom'},
                        Updater = SOMUpdateRule();
                        
                    % SOM...
                    case {'heursom'},
                        Updater = HeurSOMUpdateRule();
                        
                    % SOMN...
                    case {'somn'},
                        Updater = SOMNUpdateRule();

                    % LVQ1...
                    case {'lvq1', 'lvq'},
                        Updater = LVQ1UpdateRule();

                    % LVQ3...
                    case {'lvq3'},
                        Updater = LVQ3UpdateRule();

                    % OLVQ1...
                    case {'olvq1', 'olvq'},
                        Updater = OLVQ1UpdateRule();
                        
                    % RLVQ1...
                    case {'rlvq1', 'rlvq'},
                        Updater = RLVQ1UpdateRule();
                        
                    % ORLVQ1...
                    case {'orlvq1', 'orlvq'},
                        Updater = ORLVQ1UpdateRule();
                        
                    % LAORLVQ1...
                    case {'laorlvq'},
                        Updater = LAORLVQUpdateRule();
                        
                    % GLVQ...
                    case {'glvq'}
                        Updater = GLVQUpdateRule();
                        
                    % GRLVQ...
                    case {'grlvq'}
                        Updater = GRLVQUpdateRule();
                        
                    % SRNG...
                    case {'srng'}
                        Updater = SRNGUpdateRule();
                        
                    % LDALVQ1...
                    case {'ldalvq1', 'fc1lvq1'}
                        Updater = LDALVQ1UpdateRule();
                        
                    % LDALVQ1_3...
                    case {'ldalvq1_3', 'fc2lvq1'}
                        Updater = LDALVQ1_3UpdateRule();                        
                        
                    % LDALVQ1_4...
                    case {'ldalvq1_4'}
                        Updater = LDALVQ1_4UpdateRule();
                        
                    % LDALVQ1_5...
                    case {'ldalvq1_5'}
                        Updater = LDALVQ1_5UpdateRule();
                        
                    % LDAGLVQ...
                    case {'ldaglvq', 'fc1glvq'}
                        Updater = LDAGLVQUpdateRule();
                        
                    % LDAGLVQ_3...
                    case {'ldaglvq_3', 'fc2glvq'}
                        Updater = LDAGLVQ_3UpdateRule();
                        
                    % FC2GLVQ...
                    case {'fc2glvq'}
                        Updater = FC2GLVQUpdateRule();
                        
                    % LDAGLVQ_5...
                    case {'ldaglvq_5'}
                        Updater = LDAGLVQ_5UpdateRule();
                        
                    % LDAOLVQ...
                    case {'ldaolvq'}
                        Updater = LDAOLVQUpdateRule();
                        
                    % LDAOLVQ2...
                    case {'ldaolvq2'}
                        Updater = LDAOLVQ2UpdateRule();
                        
                    % LDAOLVQ3...
                    case {'ldaolvq3'}
                        Updater = LDAOLVQ3UpdateRule();
                        
                    % LDAOGLVQ...
                    case {'ldaoglvq'}
                        Updater = LDAOGLVQUpdateRule();
                        
                    % LDAOGLVQ_3...
                    case {'ldaoglvq_3'}
                        Updater = LDAOGLVQ_3UpdateRule();
                        
                    % LALDAGLVQ...
                    case {'laldaglvq'}
                        Updater = LALDAGLVQUpdateRule();
                        
                    % LALDAOLVQ...
                    case {'laldaolvq'}
                        Updater = LALDAOLVQUpdateRule();
                        
                    % LCALDAOLVQ...
                    case {'lcaldaolvq'}
                        Updater = LCALDAOLVQUpdateRule();                        
                        
                    % Heuristic LVQ1 training...
                    case {'heurlvq1', 'heurlvq'},
                        Updater = HeurLVQ1UpdateRule();
                        
                    % Heuristic LVQ1 training with a random signal
                    % instead of the real error signal...
                    case {'heurrandlvq1', 'heurrandlvq'},
                        Updater = HeurRandLVQ1UpdateRule();
                        
                    % Heuristic Fisher Criterion Score-based relevance
                    % determination (Method 2) LVQ1 training...
                    case {'heurfc2lvq1', 'heurfc2lvq'},
                        Updater = HeurFC2LVQ1UpdateRule();
                        
                    % Heuristic Fuzzy-LVQ1 training...
                    case {'heurflvq1', 'heurflvq'},
                        Updater = HeurFLVQ1UpdateRule();
                        
                    % Heuristic Fuzzy-LVQ1 training with a random signal
                    % instead of the real error signal...
                    case {'heurrandflvq1', 'heurrandflvq'},
                        Updater = HeurRandFLVQ1UpdateRule();
                        
                    % Heuristic LVQ3 training...
                    case {'heurlvq3'},
                        Updater = HeurLVQ3UpdateRule();

                    % Heuristic OLVQ1 (Optimized-Learning-Rate LVQ1) training...
                    case {'heurolvq1', 'heurolvq'},
                        Updater = HeurOLVQ1UpdateRule();
                        
                    % Heuristic FOLVQ1 (Fuzzy-Optimized-Learning-Rate LVQ1) training...
                    case {'heurfolvq1', 'heurfolvq'},
                        Updater = HeurFOLVQ1UpdateRule();
                        
                    % Heuristic RLVQ (Relevance-determination LVQ1) training...
                    case {'heurrlvq', 'heurrlvq1'},
                        Updater = HeurRLVQ1UpdateRule();
                        
                    % Heuristic ORLVQ1 (Optimized-Learning-Rate
                    % Relevance-determination LVQ1) training...
                    case {'heurorlvq', 'heurorlvq1'},
                        Updater = HeurORLVQ1UpdateRule();
                        
                    % Heuristic FORLVQ (Fuzzy-Optimized-Learning-Rate
                    % Relevance-determination LVQ) training...
                    case {'heurforlvq', 'heurforlvq1'},
                        Updater = HeurFORLVQ1UpdateRule();
                        
                    % Heuristic FOLRLVQ (Fuzzy-Optimized-Learning-Rate
                    % Local Relevance-determination LVQ) training...
                    case {'heurfolrlvq', 'heurfolrlvq1'},
                        Updater = HeurFOLRLVQ1UpdateRule();
                        
                    % Heuristic MAMRLVQ
                    % (Minimum-Average-Misclassification-Risk LVQ) training...
                    case {'heurmamrlvq'},
                        Updater = HeurMAMRLVQUpdateRule();
                        
                    % Heuristic FMAMRLVQ
                    % (Fuzzy Minimum-Average-Misclassification-Risk LVQ) training...
                    case {'heurfmamrlvq'},
                        Updater = HeurFMAMRLVQUpdateRule();
                        
                    % Heuristic FLDAOLVQ (Fuzzy-Optimized-Learning-Rate
                    % LVQ with LDA-based feature selection) training...
                    case {'heurfldaolvq'},
                        Updater = HeurFLDAOLVQUpdateRule();
                        
                    % Heuristic FLDAOLVQ3 (Fuzzy-Optimized-Learning-Rate
                    % LVQ with LDA-based feature selection) training...
                    case {'heurfldaolvq3'},
                        Updater = HeurFLDAOLVQ3UpdateRule();

                    % Heuristic GLVQ training...
                    case {'heurglvq'},
                        Updater = HeurGLVQUpdateRule();
                        
                    % Heuristic GRLVQ training...
                    case {'heurgrlvq'},
                        Updater = HeurGRLVQUpdateRule();
                        
                    % Heuristic FC2GLVQ training...
                    case {'heurfc2glvq'},
                        Updater = HeurFC2GLVQUpdateRule();
                        
            end
        end
        
    end
    
    methods (Access = private, Hidden = true)
        
        function time = getrealtime(obj, Mod)
            time = Mod.t;
        end
        
        function time = getnonnulltime(obj, Mod)
            time = Mod.t_nonnull;
        end
        
    end
    
end
