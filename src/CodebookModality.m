classdef CodebookModality < Modality
    % CODEBOOKMODALITY Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        
        %% ------- *** OBJECTS *** ----------------------------------------
        %******************************************************************
        %******************************************************************
        % SOM Codebook
        %------------------
        % Ideally this would be an interface to a 'Codebook' object,
        % but it would be tricky to decouple all the somtoolbox stuff,
        % so for now, it will remain a somtoolbox struct...
        SOM = [];
        
        
        %% ------- *** FUNCTION HANDLES *** -------------------------------
        %******************************************************************
        %******************************************************************
        % Handle to a method that finds the best-matching unit(s) (BMUs)
        % in the SOM codebook.
        findbmus = [];
        findbmus_static = [];
        metric = [];
        metric_withmask = [];
        
        mex_metric_code = 1;
        
        
        %% ------- *** PROPERTIES *** -------------------------------------
        %******************************************************************
        %******************************************************************
        % Current training step
        %-----------------------
        % Overall...
        t = 0;
        % A counter for the no. of timesteps we updated at...
        t_nonnull = 1;
        
        % Codebook initialization method...
        %-----------
        InitMethod = [];
        
        % Codebook node covariances initialization?...
        %-----------
        CovarianceInit = false;
        
        % Distances
        %-----------
        % Sample-to-SOM-node distances at current training step,
        % i.e. sum-squared distances...
        Distances = [];
        
        % Auxiliary distances at current training step.
        % This is used to store distances calculated
        % outside of the modality object, i.e. distances related to other
        % modalities (e.g. cross-modal Hellinger distances) which are
        % subsequently used to make calculations inside the object.
        AuxDists = [];
        nAuxDists = 1;
        
        % A place to record running statistics about the auxiliary
        % distance...
        AuxDistStats = [];
        
        % Class labels
        %--------------
        ClassLabels = [];
        ClassProbs = [];
        nClasses = [];
        
        % Best matching units (BMUs)...
        %--------------------------------------
        % Number of BMUs we should look for...
        nBMUs = 1;
        % BMUs for current timestep...
        BMUs = [];
        
        % Map activations over time
        %---------------------------
        Activations = [];
        
        % Covariance & Inverse covariance matrices for nodes...
        % (Required by the SOMN algorithm)
        %---------------------------
        Cov = [];
        CovInv = [];
        
        % Probabilities for each node...
        % (Required by the SOMN algorithm)
        %---------------------------
        Probabilities = [];
        
        % Map activations over time
        %---------------------------
        AccuracyHist = [];
        
        % Mask statistics (RunningStat object)...
        %---------------------------
        MaskStats = [];
        
        % Class statistics (RunningStat object)...
        %---------------------------
        ClassStats = [];
        
        % Node statistics (Cell array of RunningStat objects)...
        NodeStats = [];
        
        % Clustering
        %------------
        Clustering = [];
        
        % Recorded information over training time...
        % (if the record parameter is passed)
        %-------------------------------------------
        AuxDistsRecord = [];
        AuxDistMeanRecord = [];
        AuxDistStDRecord = [];
        GroundTruthRecord = [];
        BMURecord = [];
        ActivationsRecord = [];
        
        %% SOM settings & state...
        %-------------------------
        % Normalization struct...
        NormStruct = [];
        
        % Normalization method...
        NormMethod = 'range';
        
        % Has the SOM been labeled?
        SOM_is_labeled = false;
        
        % Structure...
        Size = [10 10];
        Lattice = 'hexa';
        Shape = 'sheet';
        KNN_max_k = 20;
        
        % Feature dimensions...
        dim = [];
        
        % Neighborhood type...
        Neigh = 'bubble';
        
        % Neighborhood radius...
        radius_type = 'linear';
        rini = 5;
        rfin = 1;

        % Training length...
        trainlen = 0;
        
        % Metric...
        metric_type = 'euclidean';
        
        % Feature selection feedback in training...
        feature_selection_feedback = false;
        
        % Useful stuff...
        Ud = []; % Distance between map units on the grid...
        CostMatrix = []; % Cost matrix for grid units...
        mu_x_1 = [];     % This is used pretty often...
        ZeroMask = [];
        OnesMask = [];
        
        Dx = [];
        
        % Current data sample...
        x = [];
        % Its known components...
        known = [];
        
        % Current alpha learning rate value...
        a = 0.2;
        % Current feature selection alpha learning rate value...
        a_f = 0.2;
        % Current window size for LVQ2.1-style algorithms...
        epsilon = 0.3;
        % Matrix of individual alphas for each codebook vector, used for
        % optimized-learning rate algorithms...
        Alphas = [];
        % Current radius...
        r = 5;        
        % Current neighbourhood...
        h = [];

        % Display updates...
        verbose = false;
        
        % Use mex functions?
        use_mex = false;
        
    end
    
    methods
        
        %% ------- *** CONSTRUCTOR *** ------------------------------------
        %******************************************************************
        %******************************************************************
        function obj = CodebookModality(Data, varargin)
            
            % Pass arguments to set() method...
            obj = obj.set(varargin{:});
            
            %% GET NORMALIZATION STRUCT FROM ARGUMENT OR INITIAL DATA -----
            %--------------------------------------------------------------
            if ~isfield(Data, 'NormedFeatureVectors')
                
                % Temporary data struct...
                SOMDataTemp = som_data_struct(Data.FeatureVectors',...
                                              'comp_names', Data.FeatureNames');

                % Normalize...
                % SOMDataTemp = som_normalize(SOMDataTemp, obj.NormMethod);
                % obj.NormStruct = SOMDataTemp.comp_norm;
                
                switch obj.NormMethod
                    case {'1', '2', '3', '4'}                                
                        SOMDataTemp.data = normalize(SOMDataTemp.data, str2num(obj.NormMethod));
                    otherwise
                        SOMDataTemp = som_normalize(SOMDataTemp, obj.NormMethod);
                end
                
                % Create randomized SOM struct...
                obj.SOM = som_randinit(SOMDataTemp,...
                                       'msize', obj.Size,...
                                       'lattice', obj.Lattice,...
                                       'shape', obj.Shape);
                                   
            else
                
                % Create randomized SOM struct...
                obj.SOM = som_randinit(Data.NormedFeatureVectors',...
                                       'msize', obj.Size,...
                                       'lattice', obj.Lattice,...
                                       'shape', obj.Shape);
                
            end
            
            
            

            %% INITIALIZE CODEBOOK ----------------------------------------
            % The codebook may be initialized in different ways depending
            % on what argument was passed.  Certain LVQ algorithms (e.g.
            % GLVQ) are sensitive to codebook initialization; randomized
            % codebooks can cause such algorithms to fail catastrophically.
            %
            % Options:
            % {'rand', 'random'}:
            %    Random initial codebook vector values.
            % {'sample', 'rand_sample'}:
            %    Random samples from the training data.
            % 'dist_sample':
            %    Each class in the training data is clustered,
            %    then codebook vectors for each class are sampled from
            %    the clusters.
            % 'dist_mean':
            %    Each class in the training data is clustered,
            %    then codebook vectors for each class are assigned to the
            %    mean values of the clusters.
            %--------------------------------------------------------------
            switch lower(obj.InitMethod)
                
                % Randomly initialize each dimension in the codebook...
                case {'rand', 'random'}
                    % The codebook vectors have already been randomly
                    % initialized from above, so we do nothing here.
                                        
                    if obj.CovarianceInit
                            for iNode = 1:size(obj.SOM.codebook,1)                            
                                obj.Cov{iNode} = rand(size(obj.SOM.codebook,2),size(obj.SOM.codebook,2));
                                obj.CovInv{iNode} = rand(size(obj.SOM.codebook,2),size(obj.SOM.codebook,2));
                            end
                    end
                
                % Randomly sample from the training data...
                case {'sample', 'rand_sample'}
                    
                    % If we need to initialize the covariance matrices for
                    % each node (i.e. we're using the SOMN algorithm or
                    % similar), then we subsample from the training data
                    % and take means and covariances from the subsamples...
                    if obj.CovarianceInit                        
                        
                        NumSamples = size(Data.NormedFeatureVectors,2);
                        NumPatIni = 4*max([size(obj.SOM.codebook, 2) + 1,...
                                             ceil(NumSamples / size(obj.SOM.codebook,1))]);
                                         
                        for iNode = 1:size(obj.SOM.codebook,1)                            
                            
                            MySamples = Data.NormedFeatureVectors(:,ceil(NumSamples*rand(1,NumPatIni)));
                            
                            obj.SOM.codebook(iNode,:) = mean(MySamples');
                            obj.Cov{iNode} = cov(MySamples');
                            obj.CovInv{iNode} = inv(obj.Cov{iNode});
                        end
                    
                    % Otherwise, we just draw random samples from the data
                    % for the codebook...
                    else                        
                        
                        NumSamples = size(Data.NormedFeatureVectors,2);
                        
                        for iNode = 1:size(obj.SOM.codebook,1)
                            obj.SOM.codebook(iNode,:) = Data.NormedFeatureVectors(:,ceil(NumSamples*rand));
                        end
                        
                    end
                
                % Randomly sample from the ground truth classes in the
                % training data...
                case {'class_sample'}                    
                    
                    % Set up class labels for the codebook nodes...
                    obj.ClassLabels = zeros(size(obj.SOM.codebook,1),1);
                    increment = floor(size(obj.SOM.codebook,1) / Data.nGroundTruths);
                    for j = 1:Data.nGroundTruths
                        obj.ClassLabels( ((j-1) * increment) + 1 : max(j * increment, size(obj.SOM.codebook,1))) = j;                                       
                    end
                    
                    % For each codebook node of each class, randomly
                    % sample a data vector...
                    for iNode = 1:size(obj.ClassLabels,1)
                        ClassDataIndices = find(Data.ClassLabels(Data.GroundTruthLabelIndices(:,obj.ClassLabels(iNode)),:));
                        iRandomClassSample = ceil(rand * size(find(Data.ClassLabels(Data.GroundTruthLabelIndices(:,obj.ClassLabels(iNode)),:)),2));
                        iSample = ClassDataIndices(iRandomClassSample);
                        obj.SOM.codebook(iNode,:) = Data.NormedFeatureVectors(:,iSample)';
                    end

                % Cluster the data, then randomly sample from the
                % clusters...
                case {'dist_sample', 'dist_mean'}
                    
                    % Set up class labels for the codebook nodes...
                    obj.ClassLabels = zeros(size(obj.SOM.codebook,1),1);
                    increment = floor(size(obj.SOM.codebook,1) / Data.nGroundTruths);
                    for j = 1:Data.nGroundTruths
                        obj.ClassLabels( ((j-1) * increment) + 1 : max(j * increment, size(obj.SOM.codebook,1))) = j;                                       
                    end
                    
                    % For each cluster of each class, use randomly
                    % sampled data vectors to initialize codebook
                    % vectors...
                    iNode = 1;
                    
                    for iClass = 1:size(Data.GroundTruthLabelIndices,2)
                        
                        % Grab the sample indices for this class...
                        ClassDataIndices = find(Data.ClassLabels(Data.GroundTruthLabelIndices(:,iClass),:));
                        
                        % Sometimes the clustering runs amok, so if an
                        % error is caught here, we redo the clustering...
                        iRetry = 1;
                        reclustering_required = true;
                        
                        while reclustering_required
                            
                            try
                                % Cluster that data...
                                ClassData = Data.NormedFeatureVectors(:,ClassDataIndices);
                                [foo Centroids Indices Errors OptimalK KValidityInfo] = obj.cluster('data', ClassData');

                                % Check how many codebook vectors (nodes) we can
                                % assign to this class...
                                nClassNodes = sum(obj.ClassLabels == iClass);                        

                                iClassNode = 1;
                                iCluster = 1;                       

                                % Loop assigning nodes to clusters in
                                % this class until we run out of class nodes...
                                while iClassNode <= nClassNodes && iNode <= size(obj.ClassLabels,1)

                                    ClusterData = ClassData(:,Indices{OptimalK} == iCluster);                                    
                                    
                                    if size(unique(Indices{OptimalK}),1) ~= OptimalK
                                        error('Clustering failed.');
                                    end
                                    
                                    switch lower(obj.InitMethod)
                                        case 'dist_sample'
                                            % Assign class codebook vectors based on the
                                            % cluster distribution...
                                            obj.SOM.codebook(iNode,:) = ClusterData(:,ceil(rand * size(ClusterData,2)))';
                                            
                                        case 'dist_mean'
                                            obj.SOM.codebook(iNode,:) = mean(ClusterData,2);
                                    end

                                    iNode = iNode + 1;
                                    iClassNode = iClassNode + 1;
                                    iCluster = iCluster + 1;
                                    if iCluster > OptimalK
                                        iCluster = 1;
                                    end
                                end
                                
                                reclustering_required = false;
                                
                            catch MyError
                                
                                if iRetry <= 10                                                                                
                                    iRetry = iRetry + 1;                                    
                                    reclustering_required = true;
                                else
                                    reclustering_required = false;
                                    rethrow(MyError);
                                end
                                
                            end
                        end
                    end
            end
             

            %% SET UP -----------------------------------------------------
            %--------------------------------------------------------------
            % neighborhood radius
            obj.rini = max(obj.SOM.topol.msize)/2;

            % Useful stuff...
            obj.Ud = som_unit_dists(obj.SOM.topol); % distance between map units on the grid
            obj.CostMatrix = ((obj.Ud+1).^2) / max(max((obj.Ud+1).^2));
            [munits obj.dim] = size(obj.SOM.codebook);
            obj.mu_x_1 = ones(munits,1);     % this is used pretty often
            obj.ZeroMask = zeros(size(obj.SOM.mask));
            obj.OnesMask = ones(size(obj.SOM.mask));
            
            obj.KNN_max_k = min(obj.KNN_max_k, numel(obj.Size));
            
            % initialize random number generator
            % rand('twister',sum(100*clock));
            
            % Accuracy histogram initialization...
            obj.AccuracyHist = zeros(size(obj.SOM.codebook,1),1);
            
            %% CREATE AuxDistStats OBJECT from RunningStat CLASS ----------
            %--------------------------------------------------------------
            obj.AuxDistStats = RunningStat();
            
        end
        
        %% ------- *** SET PROPERTIES *** ---------------------------------
        %******************************************************************
        %******************************************************************
        function obj = set(obj, varargin)
            
            % Defaults...
            Updaters{1} = '';
            
            % Loop through arguments...
            i = 1;
            iPassedArgs = 1;
            while i <= length(varargin), 
                argok = 1; 
                if ischar(varargin{i}), 
                    switch lower(varargin{i}), 
                        case {'codebook_size', 'size'},...
                                i=i+1; obj.Size = varargin{i}; obj.rini = max(obj.Size)/2;
                        case {'codebook_lattice', 'lattice'}, i=i+1; obj.Lattice = varargin{i};
                        case {'codebook_shape', 'shape'}, i=i+1; obj.Shape = varargin{i};
                        case {'codebook_neigh', 'neigh'}, i=i+1; obj.Neigh = varargin{i};
                        case {'codebook_mask', 'mask'}, i=i+1; obj.SOM.mask = varargin{i};
                        case {'codebook_init_method'}, i=i+1; obj.InitMethod = varargin{i};
                        case 'radius_type',  i=i+1; obj.radius_type = varargin{i};
                        case {'metric_type', 'metric'}, i=i+1; obj.metric_type = lower(varargin{i});                        
                        case 'trainlen', i=i+1; obj.trainlen = varargin{i};
                        case 'knn_max_k', i=i+1; obj.KNN_max_k = varargin{i};
                        case {'norm', 'comp_norm', 'normalization_struct'}, i=i+1; obj.NormStruct = varargin{i};
                        case {'normalization_method'}, i=i+1; obj.NormMethod = varargin{i};
                        case 'nclasses', i=i+1; obj.nClasses = varargin{i};
                        case {'featureselectionfeedback', 'feature_selection_feedback',...
                              'featureselectionintraining', 'feature_selection_in_training'},...
                                i=i+1; obj.feature_selection_feedback = varargin{i};
                        case 'record', i=i+1; obj.record = varargin{i};
                        case 'verbose', i=i+1; obj.verbose = varargin{i};
                        case {'usemex', 'use_mex'}, i=i+1; obj.use_mex = varargin{i};
                        case {'covariance_init', 'covariance_init_method'}, i=i+1; obj.CovarianceInit = varargin{i};
                        case {'updaters'},
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
                    disp(['CodebookModality.set(): Ignoring invalid argument #' num2str(i)]);
                    fprintf(obj.UsageMessage);
                end

                i = i + 1;
            end
            
            %% CREATE findbmus and metric HANDLES -------------------------
            %--------------------------------------------------------------
            if iscell(obj.metric_type)
                metric_type = obj.metric_type{1};
            else
                metric_type = obj.metric_type;
            end
            
            % Do we want to use mex functions?
            if obj.use_mex
                
                % Set the findbmus function...
                if obj.feature_selection_feedback
                    obj.findbmus = @obj.findbmus_mex_wrapper;
                else
                    obj.findbmus = @obj.findbmus_mex_wrapper_nomask;
                end
                
                obj.findbmus_static = @obj.findbmus_mex_wrapper_static;
                
                % Set the metric type...
                switch metric_type,
                    case {'squared', 'sumsquared'},
                        obj.mex_metric_code = 1;
                        
                    case {'euclidean'},
                        obj.mex_metric_code = 2;
                        
                    otherwise
                        obj.mex_metric_code = 1;
                end
                
            else
                
                % Set the findbmus function...
                switch lower(Updaters{1})
                    case 'somn'
                        obj.findbmus = @obj.findbmus_somn_matlab;
                        obj.findbmus_static = @obj.findbmus_somn_matlab_static;
                    otherwise
                        obj.findbmus = @obj.findbmus_matlab;
                        obj.findbmus_static = @obj.findbmus_matlab_static;
                end
                
            end
            
            % Set the metric type...
            switch metric_type,
                case {'squared', 'sumsquared'},
                    if obj.feature_selection_feedback
                        obj.metric = @obj.sumsquared_metric_withmask;
                    else
                        obj.metric = @obj.sumsquared_metric;
                    end

                    obj.metric_withmask = @obj.sumsquared_metric_withmask;

                case {'euclidean'},
                    if obj.feature_selection_feedback
                        obj.metric = @obj.euclidean_metric_withmask;
                    else
                        obj.metric = @obj.euclidean_metric;
                    end

                    obj.metric_withmask = @obj.euclidean_metric_withmask;

                otherwise
                    if obj.feature_selection_feedback
                        obj.metric = @obj.sumsquared_metric_withmask;
                    else
                        obj.metric = @obj.sumsquared_metric;
                    end

                    obj.metric_withmask = @obj.sumsquared_metric_withmask;
            end
            
            %% CREATE ALGORITHM OBJECT ------------------------------------
            %--------------------------------------------------------------
            if exist('PassedArgs', 'var')
                obj.Algo = obj.createAlgorithm('record', obj.record, PassedArgs{1:end});
            end
            
        end
        
        %% ------- *** TRAIN *** ------------------------------------------
        %******************************************************************
        %******************************************************************
        function obj = train(obj, Data, varargin)
            
            obj.Algo.run(obj, Data, varargin{:});
            
        end
        
        %% ------- *** CLASSIFY *** ---------------------------------------
        %******************************************************************
        %******************************************************************
        function [Matches NormedFeatureVectors] = classify(obj, Data, varargin)
            
            % Defaults...
            wise = 'nodewise';
            Codebook = obj.SOM;
            Mask = [];
            WhichBMUs = 'best';
            
            % Loop through arguments...
            i = 1;
            while i <= length(varargin), 
                argok = 1; 
                if ischar(varargin{i}), 
                    switch lower(varargin{i}), 
                        % argument IDs
                        case {'nodewise', 'node_wise'}, i=i+1; wise = 'nodewise';
                        case {'clusterwise', 'cluster_wise'}, i=i+1; wise = 'clusterwise';
                        case {'codebook'},  i=i+1; Codebook = varargin{i};
                        case {'mask', 'featuremask', 'feature_mask'},  i=i+1; Mask = varargin{i};
                        case {'whichbmus'},  i=i+1; WhichBMUs = varargin{i};
                            
                        otherwise
                            argok = 0;
                    end
                else
                    argok = 0;
                end

                if ~argok, 
                    disp(['CodebookModality.classify(): Ignoring invalid argument #' num2str(i)]);
                    fprintf(obj.UsageMessage);
                end

                i = i + 1;
            end
            
            %% CONVERT & NORMALIZE TEST DATA ------------------------------
            %--------------------------------------------------------------
            if isstruct(Data)
                if isfield(Data, 'NormedFeatureVectors')
                    SOMTestData = som_data_struct(Data.NormedFeatureVectors');
                else                    
                    SOMTestData = som_data_struct(Data.FeatureVectors');
                    SOMTestData = som_normalize(SOMTestData, obj.NormStruct);
                end
            else
                SOMTestData = som_data_struct(Data');
                SOMTestData = som_normalize(SOMTestData, obj.NormStruct);
            end
            
            % Return normalized feature vectors...
            NormedFeatureVectors = SOMTestData.data';
            
            switch wise
                case 'nodewise'
                    
                    %% NORMALIZE THE FEATURE MASK IF NECESSARY ------------
                    %------------------------------------------------------
                    if isempty(Mask)
                        Mask = ones(size(obj.SOM.mask)) ./ size(obj.SOM.mask, 1);
                    end
            
                    %% FIND BEST MATCHING UNITS IN SOM --------------------
                    %------------------------------------------------------
                    switch WhichBMUs
                        case 'best', nBMUs = 1;
                        case 'all', nBMUs = size(Codebook, 1);
                    end

                    for iTestData = 1:size(SOMTestData.data,1)
                        Matches(iTestData,:) = obj.findbmus_static(Codebook, SOMTestData.data(iTestData,:), nBMUs, Mask);
                    end
                    
                case 'clusterwise'
                    
                    %% FIND TEST DATA TO NODE DISTANCES --------------------
                    %------------------------------------------------------
                    % Distances between each of the test samples and the
                    % map nodes...
                    TestDataMapDistances = som_eucdist2(Codebook, SOMTestData);
                    
                    % TestSampleWinningClusters
                    % This KNN method messes up sometimes when there's only
                    % one class, so we bypass it if that is the case.
                    % More efficient that way anyway.
                    Unique = unique(obj.Clustering.Labels{obj.Clustering.OptimalK});
                    if length(Unique) > 1
                        [Matches,P] =...
                            knn(TestDataMapDistances',...
                                obj.Clustering.Labels{obj.Clustering.OptimalK},...
                                obj.KNN_max_k);
                    else
                        Matches = ones(size(TestDataMapDistances',1), obj.KNN_max_k) * Unique(1);
                    end
            end
            
        end
        
        %% ------- *** CLUSTER *** ----------------------------------------
        %******************************************************************
        %******************************************************************
        function [obj Centroids Indices Errors OptimalK KValidityInfo Relevance] = cluster(obj, varargin)
            
            % Defaults...
            Data = obj.SOM.codebook;
            save_cluster_info = true;            
            feature_selection = false;
            FeatureSelectionParams = 0;
            Relevance = [];
            
            % Loop through arguments...
            i = 1;
            while i <= length(varargin), 
                argok = 1; 
                if ischar(varargin{i}), 
                    switch lower(varargin{i}), 
                        % argument IDs
                        case {'data'}, i=i+1;
                            Data = varargin{i};
                            save_cluster_info = false;
                            
                        case {'featureselection', 'feature_selection'}, i=i+1;
                            if islogical(varargin{i})                                
                                feature_selection = varargin{i};
                                FeaturesToSelectFrom = 1:size(Data,2);
                            elseif isempty(varargin{i})
                                feature_selection = false;
                            else
                                feature_selection = true;
                                FeaturesToSelectFrom = varargin{i};
                            end
                            
                        case {'featureselectionparam', 'feature_selection_param',...
                              'featureselectionparams', 'feature_selection_params'}, i=i+1;
                             FeatureSelectionParams = varargin{i};
                            
                        otherwise
                            argok = 0;
                    end
                else
                    argok = 0;
                end

                if ~argok, 
                    disp(['CodebookModality.cluster(): Ignoring invalid argument #' num2str(i)]);
                    fprintf(obj.UsageMessage);
                end

                i = i + 1;
            end                        
            
            if feature_selection                
                
                Relevance = [];
                SelectedFeatureMask = ones(1,size(Data,2));
                
                if iscell(FeaturesToSelectFrom)
                                       
                    SelectedFeatureMask(setdiff(1:length(SelectedFeatureMask), [FeaturesToSelectFrom{:}])) = 0;
                    
                    for iFeatureCell = 1:length(FeaturesToSelectFrom)
                        
                        SubsetIndices = FeaturesToSelectFrom{iFeatureCell};
                        
                        % % Histogram...
                        % [n, xout] = hist(Data(:,SubsetIndices));
                        % 
                        % % Entropy...
                        % SubsetEnt = -nansum(n.*log2(n));                        
                        % 
                        % % Normed entropy...
                        % SubsetNormEnt = SubsetEnt - min(SubsetEnt);
                        % SubsetNormEnt = SubsetNormEnt ./ norm(SubsetNormEnt,1);
                        % 
                        % % Normed variance/std...
                        % SubsetVar = var(Data(:,SubsetIndices));
                        % SubsetNormVar = SubsetVar ./ norm(SubsetVar,1);
                        % SubsetSTD = std(Data(:,SubsetIndices));
                        % SubsetNormSTD = SubsetSTD ./ norm(SubsetSTD,1);
                        % 
                        % % Feature relevance...
                        % % SubsetRelevance = SubsetNormSTD ./ SubsetNormEnt;
                        % % SubsetRelevance = SubsetNormSTD;
                        % % SubsetRelevance = abs(SubsetNormEnt - max(SubsetNormEnt));
                        % % SubsetRelevance = SubsetRelevance ./ norm(SubsetRelevance,1);
                        % SubsetRelevance = SubsetNormEnt;
                        
                        BinSizes = [0.1 0.125 0.1667 0.25 0.5];
                        % BinSizes = [0.1];
                        S_bs = [];
                        S_ws = [];
                        
                        for iBin = 1:length(BinSizes)
                        
                            X = Data(:,SubsetIndices);
                            P = 0.0:BinSizes(iBin):1.0;    
                            % P = 0.0:0.1:1.0;
                            M_p = histc(X, P);
                            M = sum(M_p);
                            X_weights = obj.Activations ./ norm(obj.Activations,1);

                            S_b = zeros(1,size(X,2));
                            S_w = zeros(1,size(X,2));

                            for iDim = 1:size(X,2)

                                % X_mean = mean(X(:,iDim));
                                X_mean = X_weights' * X(:,iDim);

                                for p = 1:length(P)-1        
                                    X_p_i = X(:,iDim) > P(p) & X(:,iDim) <= P(p+1);        
                                    X_p = X(X_p_i,iDim);
                                    % X_p_mean = mean(X_p);
                                    X_p_weights = X_weights(X_p_i) ./ norm(X_weights(X_p_i),1);
                                    X_p_mean = X_p_weights' * X_p;

                                    % S_b = Between-class variance...
                                    S_b(iDim) = nansum([S_b(iDim) ((M_p(p,iDim) ./ M(iDim)) .* (X_p_mean - X_mean).^2)]);

                                    % S_w = Within-class variance...
                                    S_w(iDim) = nansum([S_w(iDim) ((M_p(p,iDim) ./ M(iDim)) .* sum((X_p - X_p_mean).^2))]);

                                end
                            end
                            
                            S_bs(iBin,:) = S_b;
                            S_ws(iBin,:) = S_w;
                            
                        end

                        SubsetRelevance = nanmean(S_bs) ./ nanmean(S_ws);
                        % SubsetRelevance = S_b ./ S_w;
                        
                        % Normalize...           
                        FooSubsetRelevance = SubsetRelevance;
                        FooSubsetRelevance(FooSubsetRelevance==Inf) = 0;                
                        SubsetRelevance(SubsetRelevance==Inf) = max(FooSubsetRelevance);
                        SubsetNormedRelevance = SubsetRelevance ./ norm(SubsetRelevance,1);
                        
                        % Add it to the pile...
                        Relevance = [Relevance SubsetNormedRelevance];
                        
                        % SelectedSubsetSubIndices = find(SubsetNormSTD >= mean(SubsetNormSTD) + FeatureSelectionParams * std(SubsetNormSTD));
                        % SelectedSubsetSubIndices = find(SubsetSTD >= mean(SubsetSTD) + FeatureSelectionParams * std(SubsetSTD));
                        % SelectedSubsetSubIndices = find(SubsetRelevance >= mean(SubsetRelevance) + FeatureSelectionParams * std(SubsetRelevance));
                        SelectedSubsetSubIndices = find(SubsetNormedRelevance >= mean(SubsetNormedRelevance) + FeatureSelectionParams * std(SubsetNormedRelevance));
                        % SelectedSubsetSubIndices = find(SubsetRelevance < mean(SubsetRelevance) - FeatureSelectionParams * std(SubsetRelevance));
                        
                        if isempty(SelectedSubsetSubIndices)
                            % [~, SelectedSubsetSubIndices] = max(SubsetRelevance);
                            SelectedSubsetSubIndices = find(FeaturesToSelectFrom{iFeatureCell});
                        end
                        
                        CulledSubsetIndices = setdiff(SubsetIndices, SubsetIndices(SelectedSubsetSubIndices));
                        
                        SelectedFeatureMask(CulledSubsetIndices) = 0;
                    end
                else
                    
                    SubsetIndices = FeaturesToSelectFrom;
                    
                    % % Histogram...
                    % [n, xout] = hist(Data(:,SubsetIndices));
                    % 
                    % % Entropy...
                    % SubsetEnt = -nansum(n.*log2(n));                        
                    % 
                    % % Normed entropy...
                    % SubsetNormEnt = SubsetEnt - min(SubsetEnt);
                    % SubsetNormEnt = SubsetNormEnt ./ norm(SubsetNormEnt,1);
                    % 
                    % % Normed variance/std...
                    % SubsetVar = var(Data(:,SubsetIndices));
                    % SubsetNormVar = SubsetVar ./ norm(SubsetVar,1);
                    % SubsetSTD = std(Data(:,SubsetIndices));
                    % SubsetNormSTD = SubsetSTD ./ norm(SubsetSTD,1);
                    % 
                    % % Feature relevance...
                    % % SubsetRelevance = SubsetNormSTD ./ SubsetNormEnt;
                    % % SubsetRelevance = SubsetNormSTD;
                    % % SubsetRelevance = abs(SubsetNormEnt - max(SubsetNormEnt));
                    % % SubsetRelevance = SubsetRelevance ./ norm(SubsetRelevance,1);
                    % SubsetRelevance = SubsetNormEnt;

                    BinSizes = [0.1 0.125 0.1667 0.25 0.5];
                    % BinSizes = [0.1];
                    S_bs = [];
                    S_ws = [];

                    for iBin = 1:length(BinSizes)

                        X = Data(:,SubsetIndices);
                        P = 0.0:BinSizes(iBin):1.0;    
                        % P = 0.0:0.1:1.0;
                        M_p = histc(X, P);
                        M = sum(M_p);
                        X_weights = obj.Activations ./ norm(obj.Activations,1);

                        S_b = zeros(1,size(X,2));
                        S_w = zeros(1,size(X,2));

                        for iDim = 1:size(X,2)

                            % X_mean = mean(X(:,iDim));
                            X_mean = X_weights' * X(:,iDim);

                            for p = 1:length(P)-1        
                                X_p_i = X(:,iDim) > P(p) & X(:,iDim) <= P(p+1);        
                                X_p = X(X_p_i,iDim);
                                % X_p_mean = mean(X_p);
                                X_p_weights = X_weights(X_p_i) ./ norm(X_weights(X_p_i),1);
                                X_p_mean = X_p_weights' * X_p;

                                % S_b = Between-class variance...
                                S_b(iDim) = nansum([S_b(iDim) ((M_p(p,iDim) ./ M(iDim)) .* (X_p_mean - X_mean).^2)]);

                                % S_w = Within-class variance...
                                S_w(iDim) = nansum([S_w(iDim) ((M_p(p,iDim) ./ M(iDim)) .* sum((X_p - X_p_mean).^2))]);

                            end
                        end

                        S_bs(iBin,:) = S_b;
                        S_ws(iBin,:) = S_w;

                    end

                    SubsetRelevance = nanmean(S_bs) ./ nanmean(S_ws);
                    % SubsetRelevance = S_b ./ S_w;

                    % Normalize...           
                    FooSubsetRelevance = SubsetRelevance;
                    FooSubsetRelevance(FooSubsetRelevance==Inf) = 0;                
                    SubsetRelevance(SubsetRelevance==Inf) = max(FooSubsetRelevance);
                    SubsetNormedRelevance = SubsetRelevance ./ norm(SubsetRelevance,1);

                    % Add it to the pile...
                    Relevance = [Relevance SubsetNormedRelevance];

                    % SelectedSubsetSubIndices = find(SubsetNormSTD >= mean(SubsetNormSTD) + FeatureSelectionParams * std(SubsetNormSTD));
                    % SelectedSubsetSubIndices = find(SubsetSTD >= mean(SubsetSTD) + FeatureSelectionParams * std(SubsetSTD));
                    % SelectedSubsetSubIndices = find(SubsetRelevance >= mean(SubsetRelevance) + FeatureSelectionParams * std(SubsetRelevance));
                    SelectedSubsetSubIndices = find(SubsetNormedRelevance >= mean(SubsetNormedRelevance) + FeatureSelectionParams * std(SubsetNormedRelevance));
                    % SelectedSubsetSubIndices = find(SubsetRelevance < mean(SubsetRelevance) - FeatureSelectionParams * std(SubsetRelevance));

                    if isempty(SelectedSubsetSubIndices)
                        % [~, SelectedSubsetSubIndices] = max(SubsetRelevance);
                        SelectedSubsetSubIndices = find(FeaturesToSelectFrom{iFeatureCell});
                    end

                    CulledSubsetIndices = setdiff(SubsetIndices, SubsetIndices(SelectedSubsetSubIndices));

                    SelectedFeatureMask(CulledSubsetIndices) = 0;
                    
                end
                
                % Normalize relevance scores...
                % FooRelevance = Relevance;
                % FooRelevance(FooRelevance==Inf) = 0;                
                % Relevance(Relevance==Inf) = max(FooRelevance);
                % Relevance = Relevance ./ norm(Relevance,1);
                
                Data = Data(:,logical(SelectedFeatureMask));
            end
            
            % [Centroids, Indices, Errors, DB_Indices] =...
            %     kmeans_clusters(Data); % find clusterings
            
            fkmeans_options.weight = obj.Activations;
            
            for k = 1:10
                if k == 1
                    [Centroids{k}, clusters, err] = som_kmeans('batch', Data, k, 100, 0);
                    Indices{k} = ones(size(Data,1),1);
                else                    
                    [Indices{k} Centroids{k} dis] = fkmeans(Data, k, fkmeans_options);
                    Indices{k} = Indices{k}';
                    Centroids{k} = Centroids{k}';
                end
            end
% 
%             [dummy,best_DB_k] = min(ind); % find the cluster with smallest Davies-Bouldin index
%             oparcObject.Classifier.KMEANS_DB_best_k = best_DB_k;

            Dmatrix = similarity_euclid(Data);

            for k = 1:length(Indices)

                [DB(k),CH(k),Dunn(k),KL(k),Han(k),st] =...
                    valid_internal_deviation(Data, Indices{k}, 1);
                
%                 S = ind2cluster(Indices{k});                
%                 [Hom(k), Sep(k), Cindex(k), wtertra(k)] = ... %, Dunn(k), DB(k)
%                      valid_internal_intra(Dmatrix, S, 1, false);
                 
                Silly = silhouette(Data, Indices{k}, 'euclidean');
                Sil(k) = mean(Silly);
                
            end
            %% Davies-Bouldin
            % Minimum value determines the optimal number of clusters [Bolshakova et al. 2003; Dimitriadou et al. 2002].
            [foo bar] = min(DB(2:length(DB)));
            Winners(1) = bar + 1;
            KValidityInfo{1}.Test = 'Davies-Bouldin';
            KValidityInfo{1}.Result = bar + 1;
            
            %% Calinski-Harabasz
            % Maximum value indicates optimal NC [Dudoit et al. 2002].
            [foo bar] = max(CH(2:length(CH)));
            Winners(2) = bar + 1;
            KValidityInfo{2}.Test = 'Calinski-Harabasz';
            KValidityInfo{2}.Result = bar + 1;
            
            %% Dunn index
            % Maximum value indicates optimal NC [Bolshakova et al. 2003; Halkidi et al. 2001].
            [foo bar] = max(Dunn(2:length(Dunn)));
            Winners(3) = bar + 1;
            KValidityInfo{3}.Test = 'Dunn';
            KValidityInfo{3}.Result = bar + 1;
            
            %% Krzanowski-Lai index
            % maximum value indicates optimal NC [Dudoit et al. 2002].
            [foo bar] = max(KL(2:length(KL)));
            Winners(4) = bar + 1;
            KValidityInfo{4}.Test = 'Krzanowski-Lai';
            KValidityInfo{4}.Result = bar + 1;
            
            %% C index (Hubert-Levin)
            % Minimal C-index indicates optimal NC [Bolshakova et al. 2003; Bolshakova et al. 2006; Dimitriadou et al. 2002].
%             [foo bar] = min(Cindex(2:length(Cindex)));
%             Winners(5) = bar + 1;
%             KValidityInfo{5}.Test = 'C index (Hubert-Levin)';
%             KValidityInfo{5}.Result = bar + 1;

            %% Silhouette index (overall average silhouette)
            % The largest silhouette indicates the optimal NC [Dudoit et al. 2002; Bolshakova et al. 2003].
            [foo bar] = max(Sil(2:length(Sil)));
            Winners(5) = bar + 1;
            KValidityInfo{5}.Test = 'Silhouette';
            KValidityInfo{5}.Result = bar + 1;
            
            %% Weighted inter-intra index
            % Search forward (k=2,3,4,...) and stop at the first down-tick
            % of the index, which indicates optimal NC [Strehl 2002].
%             [foo bar] = max(wtertra(2:length(wtertra)));
%             Winners(6) = bar + 1;
%             KValidityInfo{6}.Test = 'Weighted inter-intra index';
%             KValidityInfo{6}.Result = bar + 1;
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %% SELECT THE K-VALUE THAT WON MOST FREQUENTLY...
            OptimalK = mode(Winners);
            
            
            %% SAVE THIS STUFF IN THE OBJECT...
            if save_cluster_info
                obj.Clustering.Centroids = Centroids;
                obj.Clustering.Labels = Indices;
                
                if ~exist('Errors', 'var')
                    Errors = NaN;
                end
                obj.Clustering.Errors = Errors;
                                
                obj.Clustering.OptimalK = OptimalK;
                obj.Clustering.ValidityInfo = KValidityInfo;
            end
         
            if isempty(Relevance)
                Relevance = ones(1,size(Data,2));
                Relevance = Relevance ./ norm(Relevance,1);
            end
            
        end
        
        
        %% ------- *** CREATEALGORITHM *** --------------------------------
        % *****************************************************************
        % Factory method for algorithm creation.
        % *****************************************************************
        function Algo = createAlgorithm(obj, varargin)
            
            % Defaults...
            Updaters = {'SOM'};
            
            % Loop through arguments...
            i = 1;
            iPassedArgs = 1;
            while i <= length(varargin), 
                argok = 1; 
                if ischar(varargin{i}), 
                    switch lower(varargin{i}), 
                        case {'updaters', 'updater'},...
                                i=i+1; Updaters = varargin{i};
                        
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
                    disp(['CodebookModality.createAlgorithm(): Ignoring invalid argument #' num2str(i)]);
                    fprintf(obj.UsageMessage);
                end

                i = i + 1;
            end
            
            %% NORMALIZE THE FEATURE MASK ---------------------------------
            %--------------------------------------------------------------
            obj.SOM.mask = obj.SOM.mask ./ norm(obj.SOM.mask, 1);
            
            %% SET UP MODALITY CONDITIONS FOR VARIOUS ALGORITHMS ----------
            %--------------------------------------------------------------
            for i = 1:length(Updaters)
                
                % For supervised algorithms, we need to pre-label
                % the codebook vectors...
                switch lower(Updaters{i})                                        
                    case {'lsom',...
                          'lvq1', 'lvq', 'olvq1', 'olvq',...
                          'rlvq', 'rlvq1', 'orlvq1', 'orlvq',...
                          'laorlvq', 'lvq3',...
                          'glvq', 'grlvq', 'srng',...
                          'ldalvq1', 'fc1lvq1', 'ldalvq1_3', 'ldalvq1_4', 'ldalvq1_5', 'ldaolvq', 'ldaolvq2', 'ldaolvq3',...
                          'ldaglvq', 'fc1glvq', 'ldaglvq_3', 'fc1glvq', 'fc2glvq', 'ldaglvq_5', 'ldaoglvq', 'ldaoglvq_3',...
                          'laldaolvq', 'lcaldaolvq', 'laldaglvq'},
                        obj.ClassLabels = zeros(size(obj.SOM.codebook,1),1);
                        increment = floor(size(obj.SOM.codebook,1) / obj.nClasses);
                        for j = 1:obj.nClasses
                            obj.ClassLabels( ((j-1) * increment) + 1 : max(j * increment, size(obj.SOM.codebook,1))) = j;                                       
                        end
                end
                    
                % For algorithms that require the first 2 BMUs,
                % we need to alter the findbmu method appropriately...
                switch lower(Updaters{i})
                    case {'lvq3', 'heurlvq3', 'heurmamrlvq', 'heurfmamrlvq'},...
                        obj.nBMUs = 2;                        
                end
                    
                % Similarly, for GLVQ-style algorithms and some locally
                % adaptive algorithms, we grab ALL of
                % the BMUs, since such algorithms require both the BMU
                % for the correct class and the BMU for the closest
                % incorrect class...
                switch lower(Updaters{i})
                    case {'glvq', 'grlvq', 'ldaglvq', 'fc1glvq', 'ldaglvq_3', 'fc1glvq', 'fc2glvq', 'ldaglvq_5', 'ldalvq1_3', 'ldaolvq3', 'ldaoglvq', 'ldaoglvq_3',...
                          'laldaglvq', 'laldaolvq', 'lcaldaolvq',...
                          'heurfldaolvq', 'heurfldaolvq3', 'heurglvq', 'heurgrlvq', 'heurfc2lvq1', 'heurfc2glvq', 'heursom'}
                        obj.nBMUs = size(obj.SOM.codebook, 1);
                end

                % Some update rules require setting up
                % a RunningStat object for recording a
                % running mean and variance for each class...
                switch lower(Updaters{i})                        
                    case {'ldaolvq2'}                                                                        
                        for iClass = 1:obj.nClasses
                            obj.ClassStats{iClass} = RunningStat();
                        end
                end
                
                % Some cross-modal heuristic update rules will require
                % auxiliary distances for each node...
                switch lower(Updaters{i})
                    case {'heurfldaolvq', 'heurfldaolvq3', 'heurglvq', 'heurgrlvq', 'heurfc2lvq1', 'heurfc2glvq', 'heursom'}
                        obj.nAuxDists = size(obj.SOM.codebook, 1);
                end
                
                % Some update rules require setting up
                % RunningStat objects for recording a
                % running mean and variance for each node...
                switch lower(Updaters{i})
                    case {'ldalvq1_3', 'ldalvq1_4', 'ldalvq1_5', 'ldaolvq3',...
                          'heurfldaolvq3', 'heurforlvq', 'heurorlvq1', 'heurrlvq1',...
                          'heurfc2lvq1', 'heurfc2glvq', 'heursom',...
                          'ldaglvq_3', 'fc2glvq', 'ldaglvq_5', 'ldaoglvq_3',...
                          'lcaldaolvq'}
                        for iNode = 1:size(obj.SOM.codebook,1)
                            obj.NodeStats{iNode} = RunningStat();
                        end
                end
                
                % Probabilities for each node...
                % (required by the SOMN algorithm)
                switch lower(Updaters{i})
                    case {'somn'}
                        obj.Probabilities = ones(size(obj.SOM.codebook,1),1) / size(obj.SOM.codebook,1);
                end
            
                % Set up a RunningStat object for recording a
                % running average of the feature mask...
                obj.MaskStats = RunningStat();
                
            end
                        
            %% CREATE ALGORITHM OBJECT ------------------------------------
            %--------------------------------------------------------------
            Algo = CodebookAlgorithm(obj, 'updaters', Updaters, PassedArgs{1:end});
            
        end
        
        %% ------- *** FAST METHODS FOR FINDING BMU *** -------------------
        %******************************************************************
        % WARNING: The following methods assume that 'Data' is a single
        % feature vector and is already normalized!
        %******************************************************************                
        
        function BMUs = findbmus_mex_wrapper(obj, Data)
            [BMUs obj.Dx obj.Distances] = findbmus_mex(Data', obj.SOM.codebook', obj.SOM.mask, obj.mex_metric_code);
            obj.Dx = obj.Dx';
            BMUs = BMUs(1:obj.nBMUs);
        end
        
        function BMUs = findbmus_mex_wrapper_nomask(obj, Data)
            [BMUs obj.Dx obj.Distances] = findbmus_mex(Data', obj.SOM.codebook', obj.OnesMask, obj.mex_metric_code);
            obj.Dx = obj.Dx';
            BMUs = BMUs(1:obj.nBMUs);
        end
        
        function BMUs = findbmus_mex_wrapper_static(obj, Codebook, Data, nBMUs, Mask)
            [BMUs Dx Distances] = findbmus_mex(Data', Codebook', Mask, obj.mex_metric_code);
            obj.Dx = obj.Dx';
            BMUs = BMUs(1:obj.nBMUs);
        end
        
        function BMUs = findbmus_matlab(obj, Data)
            
            obj.Dx = obj.SOM.codebook - Data(obj.mu_x_1, :);
            
            % findbmus doesn't know if feature selection feedback will be
            % used, so we just use the general metric function pointer...
            obj.Distances = obj.metric(obj.Dx, obj.SOM.mask);
            
            %% Find & save BMUs (faster than full sort?)...
            TempDists = obj.Distances;
            
            for iBMU = 1:obj.nBMUs               
                [qerr bmu] = min(TempDists);  % Find BMU i
                BMUs(iBMU) = bmu;             % Return BMU i...
                
                TempDists(bmu) = Inf;
            end
            
        end
        
        function [BMUs Dx Distances] = findbmus_matlab_static(obj, Codebook, Data, nBMUs, Mask)
            
            Dx = Codebook - Data(ones(size(Codebook,1),1), :);
            
            % findbmus_static definitely gets passed a mask, so we should
            % use the metric_withmask function pointer...
            Distances = obj.metric_withmask(Dx, Mask);
            
            TempDists = Distances;
            
            %% Find & save BMUs (faster than full sort?)...            
            for iBMU = 1:nBMUs               
                [qerr bmu] = min(TempDists);  % Find BMU i
                BMUs(iBMU) = bmu;             % Return BMU i...
                
                TempDists(bmu) = Inf;
            end
            
        end
        
        function BMUs = findbmus_somn_matlab(obj, Data)
            
            obj.Dx = obj.SOM.codebook - Data(obj.mu_x_1, :);
            
            obj.Distances = obj.metric_withmask(obj.Dx, obj.SOM.mask);
            
%             for i = 1:size(obj.Dx)
%                 % No normalizing constant needed
%                 % Respon(NdxNeuro)=exp(-0.5*log(det(Model.C{NdxNeuro}))-0.5 * VectorDiff'*Model.CInv{NdxNeuro}*VectorDiff);
%                 obj.Activations(i,:) = exp(-0.5*log(det(obj.Cov{i})) - (0.5 * obj.Dx(i,:) * obj.CovInv{i} * obj.Dx(i,:)'));
%                 
%                 if ~isfinite(obj.Activations(i,:))
%                     obj.Activations(i,:)=0;
%                 end
%             end
            
            obj.Activations = Utils.activations(obj.Algo.ActivationTypePhases{obj.Algo.iActivationTypePhase}.type,...
                                                obj.Ud, obj.Dx, obj.Cov, obj.CovInv, obj.Distances);
            
            MySum=sum(obj.Activations);
            if MySum>0
                 obj.Activations = obj.Activations/sum(obj.Activations);
            else
                obj.Activations = zeros(size(obj.SOM.codebook,1),1);
            end
            
            %% Find & save BMUs (faster than full sort?)...
            TempActivations = obj.Activations;
            
            for iBMU = 1:obj.nBMUs               
                [qerr bmu] = max(TempActivations);  % Find BMU i
                BMUs(iBMU) = bmu;                   % Return BMU i...
                
                TempActivations(bmu) = 0;
            end
            
        end
        
        function [BMUs Dx Activations] = findbmus_somn_matlab_static(obj, Codebook, Data, nBMUs, Mask)
            
            Dx = Codebook - Data(ones(size(Codebook,1),1), :);
            
            Distances = obj.metric_withmask(Dx, Mask);
            
%             for i = 1:size(Dx)
%                 % No normalizing constant needed
%                 % Respon(NdxNeuro)=exp(-0.5*log(det(Model.C{NdxNeuro}))-0.5 * VectorDiff'*Model.CInv{NdxNeuro}*VectorDiff);
%                 Activations(i,:) = exp(-0.5*log(det(obj.Cov{i}))-0.5 * Dx(i)'*obj.CovInv{i}*Dx(i));
%                 
%                 if ~isfinite(Activations(i,:))
%                     Activations(i,:)=0;
%                 end
%             end
            
            Activations = Utils.activations(obj.Algo.ActivationTypePhases{obj.Algo.iActivationTypePhase}.type,...
                                            obj.Ud, Dx, obj.Cov, obj.CovInv, Distances);
            
            MySum=sum(Activations);
            if MySum>0
                Activations = Activations/sum(Activations);
            else
                Activations = zeros(size(Codebook,1),1);
            end
            
            %% Find & save BMUs (faster than full sort?)...
            TempActivations = Activations;
            
            for iBMU = 1:nBMUs               
                [qerr bmu] = max(TempActivations);  % Find BMU i
                BMUs(iBMU) = bmu;                   % Return BMU i...
                
                TempActivations(bmu) = 0;
            end
            
        end
        
        function Activations = findactivations(obj, Data)
            
            [BMUs Dx Distances] = findbmus_matlab_static(obj, obj.SOM.codebook, Data, 1, obj.SOM.mask);
            
            Activations = Utils.activations(obj.Algo.ActivationTypePhases{obj.Algo.iActivationTypePhase}.type,...
                                            obj.Ud,...
                                            Dx,...
                                            obj.Cov,...
                                            obj.CovInv,...
                                            Distances,...
                                            BMUs(1),...
                                            obj.r);
            
        end
        
        %% ------- *** CLEARBMUS *** --------------------------------------
        % *****************************************************************
        % *****************************************************************
        function obj = clearbmus(obj)
            
            obj.BMUs = [];
            
        end
    end
    
    methods (Access = private)                
        
        function Dists = sumsquared_metric(obj, Dx, varargin)
            Dists = Dx.^2 * obj.OnesMask;
        end
        
        function Dists = sumsquared_metric_withmask(obj, Dx, Mask)
            Dists = Dx.^2 * Mask;
        end
        
        function Dists = euclidean_metric(obj, Dx, varargin)
            Dists = sqrt(Dx.^2 * obj.OnesMask);
        end
        
        function Dists = euclidean_metric_withmask(obj, Dx, Mask)
            Dists = sqrt(Dx.^2 * Mask);
        end               
        
    end
    
end
