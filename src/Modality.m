classdef Modality < handle
    
    properties
        
        %% ------- *** OBJECTS *** ----------------------------------------
        %******************************************************************
        %******************************************************************                
        % Training Algorithm
        %--------------------
        Algo = [];
        
        %% ------- *** PROPERTIES *** -------------------------------------
        %******************************************************************
        %******************************************************************
        record = false;
        
    end
    
    methods
        
        obj = set(obj, varargin)
        
        obj = train(obj, Data, varargin)
        
        [Matches NormedFeatureVectors] = classify(obj, Data, varargin)
        
        Algo = createAlgorithm(obj, varargin)
        
    end
    
end