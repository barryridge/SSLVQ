classdef ORLVQ1UpdateRule < UpdateRule
    
    methods (Static = true)
        
        function Mod = update(Algo, Mod, Data, iData, varargin)

            %% OLVQ1 (Optimized-Learning-Rate LVQ) update...
            if Mod.ClassLabels(Mod.BMUs(1)) == find(Data.ClassLabels(Data.GroundTruthLabelIndices, iData))

                % Update the learning rate for the winning node...
                Mod.Alphas(Mod.BMUs(1)) = Mod.Alphas(Mod.BMUs(1)) / (1 + Mod.Alphas(Mod.BMUs(1)));

                % Update the BMU codebook vector (LVQ)...
                Mod.SOM.codebook(Mod.BMUs(1),Mod.known) =...
                    Mod.SOM.codebook(Mod.BMUs(1),Mod.known) +...
                        Mod.Alphas(Mod.BMUs(1)) * (Mod.x(Mod.known) - Mod.SOM.codebook(Mod.BMUs(1),Mod.known));
                    
                % Update the feature weights (RLVQ)...
                Mod.SOM.mask = max(Mod.SOM.mask - (Mod.a_f * abs(Mod.Dx(Mod.BMUs(1),:)))', Mod.ZeroMask);
                Mod.SOM.mask = Mod.SOM.mask ./ norm(Mod.SOM.mask,1);

            else

                % Update the learning rate for the winning node...
                Mod.Alphas(Mod.BMUs(1)) = min(Mod.Alphas(Mod.BMUs(1)) / (1 - Mod.Alphas(Mod.BMUs(1))), 1);

                % Update the BMU codebook vector (LVQ)...
                Mod.SOM.codebook(Mod.BMUs(1),Mod.known) =...
                    Mod.SOM.codebook(Mod.BMUs(1),Mod.known) -...
                        Mod.Alphas(Mod.BMUs(1)) * (Mod.x(Mod.known) - Mod.SOM.codebook(Mod.BMUs(1),Mod.known));
                    
                % Update the feature weights (RLVQ)...
                Mod.SOM.mask = Mod.SOM.mask + (Mod.a_f * abs(Mod.Dx(Mod.BMUs(1),:)))';
                Mod.SOM.mask = Mod.SOM.mask ./ norm(Mod.SOM.mask,1);

            end
            
        end
        
    end
    
end