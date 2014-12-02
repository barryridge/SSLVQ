classdef HeurLVQ3UpdateRule < UpdateRule
    
    methods (Static = true)
        
        function Mod = update(Algo, Mod, Data, iData, varargin)

            %% Check that we have everything we need to continue...
            if isnan(Mod.AuxDists(1)) || isnan(Mod.AuxDists(2)) || isnan(Mod.AuxDistStats.mean())                
                return;
            end
            
            %% Update...            
            flag = 0;
            mj = 0;
            mi = 0;

            % If the first BMU is the correct 'class'...
            if Mod.AuxDists(1) < Mod.AuxDistStats.mean()
                mj = Mod.BMUs(1);
                mi = Mod.BMUs(2);

                if Mod.AuxDists(2) < Mod.AuxDistStats.mean()
                    flag = 1;
                end

            % If the second BMU is the correct 'class'...
            elseif Mod.AuxDists(2) < Mod.AuxDistStats.mean()
                mj = Mod.BMUs(2);
                mi = Mod.BMUs(1);

                if Mod.AuxDists(1) < Mod.AuxDistStats.mean()
                    flag = 1;
                end

            end

            if mj && mi

                % If first & second BMUs are of the SAME 'class'...
                if flag
                    Mod.SOM.codebook([mj mi],Mod.known) =...
                        Mod.SOM.codebook([mj mi],Mod.known) +...
                        Mod.a * Mod.epsilon * (Mod.x([1 1],Mod.known) - Mod.SOM.codebook([mj mi],Mod.known));

                % If first & second BMUs are of DIFFERENT 'classes'...
                else
                    Mod.SOM.codebook(mj,Mod.known) =...
                        Mod.SOM.codebook(mj,Mod.known) + Mod.a * (Mod.x(Mod.known) - Mod.SOM.codebook(mj,Mod.known));

                    Mod.SOM.codebook(mi,Mod.known) =...
                        Mod.SOM.codebook(mi,Mod.known) - Mod.a * (Mod.x(Mod.known) - Mod.SOM.codebook(mi,Mod.known));
                end
            end
            
        end
        
    end
    
end