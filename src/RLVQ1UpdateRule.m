classdef RLVQ1UpdateRule < UpdateRule
    
    methods (Static = true)
        
        function Mod = update(Algo, Mod, Data, iData, varargin)

            %% RLVQ1 (Relevance Determination LVQ1) update...
            if Mod.ClassLabels(Mod.BMUs(1)) == find(Data.ClassLabels(Data.GroundTruthLabelIndices, iData))

                % Update the BMU codebook vector (LVQ)...
                Mod.SOM.codebook(Mod.BMUs(1),Mod.known) =...
                    Mod.SOM.codebook(Mod.BMUs(1),Mod.known) +...
                        Mod.a * (Mod.x(Mod.known) - Mod.SOM.codebook(Mod.BMUs(1),Mod.known));
                    
                % Update the feature weights (RLVQ)...
                Mod.SOM.mask = max(Mod.SOM.mask - (Mod.a_f * abs(Mod.Dx(Mod.BMUs(1),:)))', Mod.ZeroMask);
                Mod.SOM.mask = Mod.SOM.mask ./ norm(Mod.SOM.mask,1);
                
                % If the BMU classified the sample correctly, increment its
                % score...
                Mod.AccuracyHist(Mod.BMUs(1)) = Mod.AccuracyHist(Mod.BMUs(1)) + 1;

            else

                % Update the BMU codebook vector (LVQ)...
                Mod.SOM.codebook(Mod.BMUs(1),Mod.known) =...
                    Mod.SOM.codebook(Mod.BMUs(1),Mod.known) -...
                        Mod.a * (Mod.x(Mod.known) - Mod.SOM.codebook(Mod.BMUs(1),Mod.known));
                    
                % Update the feature weights (RLVQ)...
                Mod.SOM.mask = Mod.SOM.mask + (Mod.a_f * abs(Mod.Dx(Mod.BMUs(1),:)))';
                Mod.SOM.mask = Mod.SOM.mask ./ norm(Mod.SOM.mask,1);

            end
            
        end
        
    end
    
end