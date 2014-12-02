classdef LVQ3UpdateRule < UpdateRule
    
    methods (Static = true)
        
        function Mod = update(Algo, Mod, Data, iData, varargin)
            
            %% Check BMUs...
            flag = 0;
            mj = 0;
            mi = 0;

            % If the first BMU is the correct class...
            if Mod.ClassLabels(Mod.BMUs(1)) == find(Data.ClassLabels(Data.GroundTruthLabelIndices, iData))
                mj = Mod.BMUs(1);
                mi = Mod.BMUs(2);

                if Mod.ClassLabels(Mod.BMUs(2)) == find(Data.ClassLabels(Data.GroundTruthLabelIndices, iData)) 
                    flag = 1;
                end

            % If the second BMU is the correct class...
            elseif Mod.ClassLabels(Mod.BMUs(2)) == find(Data.ClassLabels(Data.GroundTruthLabelIndices, iData)) 
                mj = Mod.BMUs(2);
                mi = Mod.BMUs(1);

                if Mod.ClassLabels(Mod.BMUs(1)) == find(Data.ClassLabels(Data.GroundTruthLabelIndices, iData)) 
                    flag = 1;
                end

            end

            %% Update...
            if mj & mi

                % If first & second BMUs are of the SAME class...
                if flag
                    Mod.SOM.codebook([mj mi],Mod.known) =...
                        Mod.SOM.codebook([mj mi],Mod.known) +...
                        Mod.a * Mod.epsilon * (x([1 1],Mod.known) - Mod.SOM.codebook([mj mi],Mod.known));

                % If first & second BMUs are of DIFFERENT classes...
                else
                    Mod.SOM.codebook(mj,Mod.known) =...
                        Mod.SOM.codebook(mj,Mod.known) + Mod.a * (x(Mod.known) - Mod.SOM.codebook(mj,Mod.known));

                    Mod.SOM.codebook(mi,Mod.known) =...
                        Mod.SOM.codebook(mi,Mod.known) - Mod.a * (x(Mod.known) - Mod.SOM.codebook(mi,Mod.known));
                end
            end
            
        end
        
    end
    
end