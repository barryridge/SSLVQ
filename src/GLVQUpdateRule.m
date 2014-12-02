classdef GLVQUpdateRule < UpdateRule
    
    methods (Static = true)
        
        function Mod = update(Algo, Mod, Data, iData, varargin)
            
            %% GLVQ (Generalized Learning Vector Quantization) update...
            
            % Grab w_J, the best matching node of the correct class...
            CorrectClassBMUIndices = find(Mod.ClassLabels(Mod.BMUs) == find(Data.ClassLabels(Data.GroundTruthLabelIndices, iData)));
            w_J = Mod.BMUs(CorrectClassBMUIndices(1));
            
            % Grab w_K, the next closest node of an incorrect class...
            IncorrectClassBMUIndices = find(Mod.ClassLabels(Mod.BMUs) ~= find(Data.ClassLabels(Data.GroundTruthLabelIndices, iData)));
            w_K = Mod.BMUs(IncorrectClassBMUIndices(1));
            
            % Calculate the distance d_J of x (the sample) to w_J...
            d_J = Mod.metric((Mod.x(Mod.known) - Mod.SOM.codebook(w_J, Mod.known)), Mod.OnesMask);
            
            % Calculate the distance d_K of x (the sample) to w_K...
            d_K = Mod.metric((Mod.x(Mod.known) - Mod.SOM.codebook(w_K, Mod.known)), Mod.OnesMask);
            
            % Calculate the mu(x) part of the GLVQ update...
            mu_x = (d_J - d_K) / (d_J + d_K);
            
            if ~isnan(mu_x)
                
                % Calculate the f(mu,t) sigmoidal function part of the GLVQ
                % update...
                f_mu_t = 1 / (1 + exp(-mu_x * Algo.UpdaterPhases{Algo.currentphase}.t));
                % f_mu_t = 1 / (1 + exp(-mu_x));

                % Estimate its derivative...
                d_f_d_mu = f_mu_t * (1 - f_mu_t);

                % d_f_d_mu = exp(-mu_x) / ((1 + exp(-mu_x))^2);

                % Update w_J...            
                Mod.SOM.codebook(w_J, Mod.known) =...
                    Mod.SOM.codebook(w_J, Mod.known) +...
                        (((4.0 * Mod.a * d_f_d_mu * d_K) / (d_J + d_K)^2) * (Mod.x(Mod.known) - Mod.SOM.codebook(w_J, Mod.known)));

                % Update w_K...
                Mod.SOM.codebook(w_K, Mod.known) =... 
                    Mod.SOM.codebook(w_K, Mod.known) -...
                        (((4.0 * Mod.a * d_f_d_mu * d_J) / (d_J + d_K)^2) * (Mod.x(Mod.known) - Mod.SOM.codebook(w_K, Mod.known)));


                % Record accuracy histogram for the nodes based on whether the
                % BMU correctly classifys the sample...
                if Mod.ClassLabels(Mod.BMUs(1)) == find(Data.ClassLabels(Data.GroundTruthLabelIndices, iData))

                    % If the BMU classified the sample correctly, increment its
                    % score...
                    Mod.AccuracyHist(Mod.BMUs(1)) = Mod.AccuracyHist(Mod.BMUs(1)) + 1;
                % else

                    % If the BMU classified the sample incorrectly, decrement its
                    % score...
                    % Mod.AccuracyHist(Mod.BMUs(1)) = Mod.AccuracyHist(Mod.BMUs(1)) - 1;
                end
                
            end
            
        end
        
    end
    
end