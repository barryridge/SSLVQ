classdef HebbianMapping < Mapping
    %HEBBIANMAPPING Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        
        % Default settings...
        
        % Modality point/node matches...
        CoOccurrences = [];
        
        % Modality activations for the point/node matches...
        InActivations = [];
        OutActivations = [];
        
        % Hebbian link weights...
        Weights = [];
        
        % Training length...
        trainlen = 0;
        
        % Current timestep...
        t = [];
        
        % Current alpha learning rate value...
        a = 0.2;
        
    end
    
    methods
        
        function obj = HebbianMapping(inModality, outModality)
            
            obj.Weights = zeros(size(inModality.SOM.codebook,1),...
                                    size(outModality.SOM.codebook,1));
                                
            obj.t = 0;
        end
        
        function obj = set(obj, varargin)
            
            % Defaults...
            
            % Loop through arguments...
            i = 1;
            iPassedArgs = 1;
            while i <= length(varargin), 
                argok = 1; 
                if ischar(varargin{i}), 
                    switch lower(varargin{i}), 
                        case 'trainlen', i=i+1; obj.trainlen = varargin{i};
                        
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
                    disp(['HebbianMapping.set(): Ignoring invalid argument #' num2str(i)]);
                    fprintf(obj.UsageMessage);
                end

                i = i + 1;
            end
            
        end
        
        function obj = train(obj, inMatches, outMatches, InActivations, OutActivations, varargin)
            
            %% Store co-occurrence matches...
            if isempty(obj.CoOccurrences)
                obj.CoOccurrences = [inMatches' outMatches'];
            else
                obj.CoOccurrences = [[obj.CoOccurrences(:,1)' inMatches]' [obj.CoOccurrences(:,2)' outMatches]'];
            end
            
            %% Store current activations...
            obj.InActivations = InActivations';
            obj.OutActivations = OutActivations';
            
            obj.t = obj.t + 1;
            
            %% Training...
            for i = 1:size(obj.InActivations,1)

                WeightAdjustments = obj.DeltaFunction(obj.InActivations(i,:), obj.OutActivations(i,:), obj.t);
                Divisor = repmat(sqrt( sum((obj.Weights + WeightAdjustments).^2, 2)),1, size(obj.Weights,2));
                Divisor(isnan(Divisor)) = 1;
                Divisor(Divisor==0) = 1;

                % obj.Weights = (obj.Weights + WeightAdjustments) ./ Divisor;
                obj.Weights = (obj.Weights + WeightAdjustments);
            end                        
                
        end
    end
    
    methods (Access = private)
        
        function WeightAdjustments = DeltaFunction(obj, InActivations, OutActivations, t)
            
            WeightAdjustments = repmat(InActivations,size(OutActivations,2),1)' .*...
                repmat(OutActivations,size(InActivations,2),1);
            
%             for i = 1:size(obj.InActivations(t,:),2)
%                 for j = 1:size(obj.OutActivations(t,:),2)
%                     WeightAdjustments(i,j) = obj.InActivations(t,i) * obj.OutActivations(t,j);
%                 end
%             end
            
            obj.a = Utils.learningrate('gaussian',...           % type
                                       1,...                    % seed
                                       obj.trainlen,...         % training length
                                       obj.trainlen/2,...       % shift timestep
                                       t);                      % current global timestep                        
            
            WeightAdjustments = obj.a * WeightAdjustments;
        end
    end
    
end

