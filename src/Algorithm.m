classdef Algorithm < handle
    % ALGORITHM Summary of this class goes here
    %   Detailed explanation goes here
    
    methods (Abstract)

        set(varargin)
        
        run(Mod, Data, varargin)
        
    end
    
end