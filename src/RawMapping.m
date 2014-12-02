classdef RawMapping < Mapping
    %SOM Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        
        % Modality point/node matches...
        CoOccurences = [];
        
        % Modality activations for the point/node matches...
        InActivations = [];
        OutActivations = [];
        
        Weights = [];
        
        % Current timestep...
        t = [];
    end
    
    properties (SetAccess = private)
        
        % Default settings...                
        
    end
    
    methods
        function obj = RawMapping(inModality, outModality)
            
            obj.Weights = zeros(size(inModality.SOM.codebook,1),...
                                    size(outModality.SOM.codebook,1));
                                
            obj.t = 0;
            
        end
        
        function obj = train(obj, inMatches, outMatches, InActivations, OutActivations, varargin)
            
            %% Store matches...
            if isempty(obj.CoOccurences)
                obj.CoOccurences = [inMatches' outMatches'];
            else
                obj.CoOccurences = [[obj.CoOccurences(:,1)' inMatches]' [obj.CoOccurences(:,2)' outMatches]'];
            end
            
            %% Store activations...
            if isempty(obj.InActivations)
                obj.InActivations = InActivations';
            else
                obj.InActivations = [obj.InActivations' InActivations]';
            end
            
            if isempty(obj.OutActivations)
                obj.OutActivations = OutActivations';
            else
                obj.OutActivations = [obj.OutActivations' OutActivations]';
            end
            
            %% Training...
            for t = (obj.t + 1):size(obj.CoOccurences,1)
                
                obj.Weights(obj.CoOccurences(t,1), obj.CoOccurences(t,2)) =...
                    obj.Weights(obj.CoOccurences(t,1), obj.CoOccurences(t,2)) + 1;                
                
            end
            
            obj.t = t;
        end
    end
end