classdef HeurOLVQ1UpdateRule < UpdateRule
    
    methods (Static = true)
        
        function Mod = update(Algo, Mod, Data, iData, varargin)

            % Use the auxiliary (Hellinger?) distances as a
            % heuristic...
            if isnan(Mod.AuxDists(1)) || isnan(Mod.AuxDistStats.mean())
                
                return;
                
            elseif Mod.AuxDists(1) < Mod.AuxDistStats.mean()
                
                % Update the learning rate for the winning node...
                Mod.Alphas(Mod.BMUs(1)) = Mod.Alphas(Mod.BMUs(1)) / (1 + Mod.Alphas(Mod.BMUs(1)));

                % Correct 'class' update rule...
                Mod.SOM.codebook(Mod.BMUs(1),Mod.known) =...
                    Mod.SOM.codebook(Mod.BMUs(1),Mod.known) +...
                        Mod.a * (Mod.x(Mod.known) - Mod.SOM.codebook(Mod.BMUs(1),Mod.known));
            else

                % Update the learning rate for the winning node...
                Mod.Alphas(Mod.BMUs(1)) = min(Mod.Alphas(Mod.BMUs(1)) / (1 - Mod.Alphas(Mod.BMUs(1))), 1);
                
                % Incorrect 'class' update rule...
                Mod.SOM.codebook(Mod.BMUs(1),Mod.known) =...
                    Mod.SOM.codebook(Mod.BMUs(1),Mod.known) -...
                        Mod.a * (Mod.x(Mod.known) - Mod.SOM.codebook(Mod.BMUs(1),Mod.known));
            end
            
        end
        
    end
    
end