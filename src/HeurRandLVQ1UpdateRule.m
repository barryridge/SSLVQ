classdef HeurRandLVQ1UpdateRule < UpdateRule
    
    methods (Static = true)
        
        function Mod = update(Algo, Mod, Data, iData, varargin)

            % Use the auxiliary (Hellinger?) distances as a
            % heuristic...
%             if isnan(Mod.AuxDists(1)) || isnan(Mod.AuxDistStats.mean()) || isnan(Mod.AuxDistStats.std())
%                 
%                 return;
                
            if all(isnan(Mod.AuxDists))
                
                return;
                
            % elseif rand < Mod.AuxDistStats.mean() - Mod.AuxDistStats.std()
            elseif rand < nanmean(Mod.AuxDists) - nanstd(Mod.AuxDists)

                % Correct 'class' update rule...
                Mod.SOM.codebook(Mod.BMUs(1),Mod.known) =...
                    Mod.SOM.codebook(Mod.BMUs(1),Mod.known) +...
                        Mod.a * (Mod.x(Mod.known) - Mod.SOM.codebook(Mod.BMUs(1),Mod.known));
                    
            % elseif rand >= Mod.AuxDistStats.mean() + Mod.AuxDistStats.std()
            elseif rand >= nanmean(Mod.AuxDists) + nanstd(Mod.AuxDists)

                % Incorrect 'class' update rule...
                Mod.SOM.codebook(Mod.BMUs(1),Mod.known) =...
                    Mod.SOM.codebook(Mod.BMUs(1),Mod.known) -...
                        Mod.a * (Mod.x(Mod.known) - Mod.SOM.codebook(Mod.BMUs(1),Mod.known));
            end
            
        end
        
    end
    
end