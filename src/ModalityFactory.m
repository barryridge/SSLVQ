classdef ModalityFactory < handle
    % MODALITYFACTORY Summary of this class goes here
    %   Detailed explanation goes here
    
    methods (Static = true)
    
        %% ------- *** CREATEMODALITY *** ---------------------------------
        % *****************************************************************
        % Factory method for creating modality objects.
        % *****************************************************************
        function Mod = createModality(Data, varargin)
            
            % Defaults...
            type = 'codebook';
            
            % Loop through arguments...
            i = 1;
            PassedArgs = [];
            iPassedArgs = 1;
            while i <= length(varargin), 
                argok = 1; 
                if ischar(varargin{i}), 
                    switch lower(varargin{i}), 
                        case {'type', 'modality_type', 'modalitytype'},...
                                i=i+1; type = varargin{i};
                                     
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
                    disp(['ModalityFactory.createModality(): Ignoring invalid argument #' num2str(i)]);
                    % fprintf(obj.UsageMessage);
                end

                i = i + 1;
            end
            
            % Create modality object...
            switch type
                case {'codebook'},
                    Mod = CodebookModality(Data, PassedArgs{1:end});
                otherwise
                    Mod = CodebookModality(Data, PassedArgs{1:end});
            end
        end
        
    end
    
end