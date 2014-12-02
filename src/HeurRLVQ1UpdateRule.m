classdef HeurRLVQ1UpdateRule < UpdateRule
    
    methods (Static = true)
        
        function Mod = update(Algo, Mod, Data, iData, varargin)

            % Use the auxiliary (Hellinger?) distances as a
            % heuristic...
            if isnan(Mod.AuxDists(1)) || isnan(Mod.AuxDistStats.mean())
                
                return;
                
            % elseif Mod.AuxDists(1) < Mod.AuxDistStats.mean()
            elseif Mod.AuxDists(1) < Mod.AuxDistStats.mean() - Mod.AuxDistStats.std()

                % Update the BMU codebook vector (LVQ)...
                Mod.SOM.codebook(Mod.BMUs(1),Mod.known) =...
                    Mod.SOM.codebook(Mod.BMUs(1),Mod.known) +...
                        Mod.a * (Mod.x(Mod.known) - Mod.SOM.codebook(Mod.BMUs(1),Mod.known));

                % Update the feature weights (RLVQ)...
                Mod.SOM.mask = max(Mod.SOM.mask - (Mod.a_f * abs(Mod.Dx(Mod.BMUs(1),:)))', Mod.ZeroMask);
                Mod.SOM.mask = Mod.SOM.mask / sum(Mod.SOM.mask);
                
                % Record running stats of correctly classified samples for
                % each node...
                Mod.NodeStats{Mod.BMUs(1)}.push(Mod.x);
                
            % else
            elseif Mod.AuxDists(1) >= Mod.AuxDistStats.mean() + Mod.AuxDistStats.std()
                
                % Update the BMU codebook vector (LVQ)...
                Mod.SOM.codebook(Mod.BMUs(1),Mod.known) =...
                    Mod.SOM.codebook(Mod.BMUs(1),Mod.known) -...
                        Mod.a * (Mod.x(Mod.known) - Mod.SOM.codebook(Mod.BMUs(1),Mod.known));

                % Update the feature weights (RLVQ)...
                Mod.SOM.mask = Mod.SOM.mask + (Mod.a_f * abs(Mod.Dx(Mod.BMUs(1),:)))';
                Mod.SOM.mask = Mod.SOM.mask / sum(Mod.SOM.mask);
                
            end

        end
        
    end
    
end