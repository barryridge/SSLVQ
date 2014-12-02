classdef HeurFLVQ1UpdateRule < UpdateRule
    
    methods (Static = true)
        
        function Mod = update(Algo, Mod, Data, iData, varargin)

            % Use the auxiliary (Hellinger?) distances as a
            % heuristic...
            if isnan(Mod.AuxDists(1)) || isnan(Mod.AuxDistStats.mean())
                
                return;
                
            else
                % Update rule...
                Mod.SOM.codebook(Mod.BMUs(1),Mod.known) =...
                    Mod.SOM.codebook(Mod.BMUs(1),Mod.known) +...
                        ((2 * (0.5 - Mod.AuxDists(1))) * Mod.a) *...
                         (Mod.x(Mod.known) - Mod.SOM.codebook(Mod.BMUs(1),Mod.known));
            end
            
        end
        
    end
    
end