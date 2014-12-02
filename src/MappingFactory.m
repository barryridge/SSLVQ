classdef MappingFactory < handle
    % MAPPINGFACTORY Summary of this class goes here
    %   Detailed explanation goes here
    
    methods (Static = true)
    
        %% ------- *** CREATEMAPPING *** ---------------------------------
        % *****************************************************************
        % Factory method for creating mapping objects.
        % *****************************************************************
        function Mapping = createMapping(varargin)
            
            % Defaults...
            type = 'hebbian';
            
            % Loop through arguments...
            i = 1;
            PassedArgs = [];
            iPassedArgs = 1;
            while i <= length(varargin)
                if ischar(varargin{i})                    
                    switch lower(varargin{i})
                        case {'type', 'mapping_type', 'mappingtype'},...
                                i=i+1; type = varargin{i};
                                     
                        otherwise
                            PassedArgs{iPassedArgs} = varargin{i};
                            PassedArgs{iPassedArgs+1} = varargin{i+1};
                            i = i + 1;
                            iPassedArgs = iPassedArgs + 2;
                    end
                else
                    PassedArgs{iPassedArgs} = varargin{i};
                    iPassedArgs = iPassedArgs + 1;
                end
                
                i = i + 1;
            end
            
            % Create modality object...
            switch lower(type)
                case {'hebbian'},
                    Mapping = HebbianMapping(PassedArgs{1:end});
                case {'raw'},
                    Mapping = RawMapping(PassedArgs{1:end});
                otherwise
                    Mapping = HebbianMapping(PassedArgs{1:end});
            end
        end
        
    end
    
end