classdef HeurGLVQUpdateRule < UpdateRule
    
    methods (Static = true)
        
        function Mod = update(Algo, Mod, Data, iData, varargin)
            

            % Grab w_J, the best matching node of the correct class...
            CorrectClassBMULogicalIndices = Mod.AuxDists < Mod.AuxDistStats.mean() - Mod.AuxDistStats.std();
            if sum(CorrectClassBMULogicalIndices) == 0
                return;
            end
            CorrectClassBMUIndices = find(CorrectClassBMULogicalIndices);
            w_J = Mod.BMUs(CorrectClassBMUIndices(1));
            
            % Grab w_K, the next closest node of an incorrect class...
            InCorrectClassBMULogicalIndices = Mod.AuxDists > Mod.AuxDistStats.mean() + Mod.AuxDistStats.std();
            if sum(InCorrectClassBMULogicalIndices) == 0
                return;
            end
            IncorrectClassBMUIndices = find(InCorrectClassBMULogicalIndices);
            w_K = Mod.BMUs(IncorrectClassBMUIndices(1));
            
            D_J = (Mod.x(Mod.known) - Mod.SOM.codebook(w_J, Mod.known));
            D_K = (Mod.x(Mod.known) - Mod.SOM.codebook(w_K, Mod.known));
            
            % Calculate the distance d_J of x (the sample) to w_J...
            d_J = Mod.metric_withmask(D_J, Mod.SOM.mask);
            
            % Calculate the distance d_K of x (the sample) to w_K...
            d_K = Mod.metric_withmask(D_K, Mod.SOM.mask);
            
            % Calculate the mu(x) part of the GLVQ update...
            mu_x = (d_J - d_K) / (d_J + d_K);
            
            if ~isnan(mu_x)
                
                % Calculate the f(mu,t) sigmoidal function part of the GLVQ
                % update...
                f_mu_t = 1 / (1 + exp(-mu_x * Algo.UpdaterPhases{Algo.currentphase}.t));
                % f_mu_t = 1 / (1 + exp(-10 * mu_x));

                % Estimate its derivative...
                d_f_d_mu = f_mu_t * (1 - f_mu_t);

                % d_f_d_mu = exp(-mu_x) / ((1 + exp(-mu_x))^2);

                % Probability of w_J being of the correct class?
                p_J = 1 - Mod.AuxDists(CorrectClassBMUIndices(1));

                % Probability of w_K being of the incorrect class?
                p_K = Mod.AuxDists(IncorrectClassBMUIndices(1));

                % Update w_J...            
                Mod.SOM.codebook(w_J, Mod.known) =...
                    Mod.SOM.codebook(w_J, Mod.known) +...
                        (((Mod.a * d_f_d_mu * d_K * (p_J + p_K + 1)) / (d_J + d_K)^2) * (Mod.SOM.mask' .* D_J));

                % Update w_K...
                Mod.SOM.codebook(w_K, Mod.known) =... 
                    Mod.SOM.codebook(w_K, Mod.known) -...
                        (((Mod.a * d_f_d_mu * d_J * (p_J + p_K + 1)) / (d_J + d_K)^2) * (Mod.SOM.mask' .* D_K));                
            end
            
        end
        
    end
    
end
