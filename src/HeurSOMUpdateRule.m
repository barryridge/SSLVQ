classdef HeurSOMUpdateRule < UpdateRule
    
    methods (Static = true)
        
        function Mod = update(Algo, Mod, Data, iData, varargin)

            %% Update...
            Mod.SOM.codebook(:,Mod.known) =...
                Mod.SOM.codebook(:,Mod.known) - Mod.a*Mod.h(:,ones(sum(Mod.known),1)).*Mod.Dx;
            
            if ~isempty(Mod.NodeStats) && Mod.AuxDists(1) < Mod.AuxDistStats.mean() - Mod.AuxDistStats.std()                    
                % Record running stats of correctly classified samples for
                % each node...
                Mod.NodeStats{Mod.BMUs(1)}.push(Mod.x);
                
                % If the BMU classified the sample correctly, increment its
                % score...
                Mod.AccuracyHist(Mod.BMUs(1)) = Mod.AccuracyHist(Mod.BMUs(1)) + 1;
            end
            
        end
        
    end
    
end