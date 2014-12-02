classdef Mapping < handle    
    % Mapping Interface
    
    properties
        
    end
    
    methods (Abstract)

        obj = train(obj, inMatches, outMatches, InActivations, OutActivations, varargin)
        
    end
    
end