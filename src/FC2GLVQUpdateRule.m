classdef FC2GLVQUpdateRule < UpdateRule
    
    methods (Static = true)
        
        function Mod = update(Algo, Mod, Data, iData, varargin)
            
            %% GLVQ (Generalized Relevance Determination LVQ) update...
            
            % Grab w_J, the best matching node of the correct class...
            CorrectClassBMULogicalIndices = Mod.ClassLabels(Mod.BMUs) == find(Data.ClassLabels(Data.GroundTruthLabelIndices, iData));
            CorrectClassBMUIndices = find(CorrectClassBMULogicalIndices);
            w_J = Mod.BMUs(CorrectClassBMUIndices(1));
            
            % Grab w_K, the next closest node of an incorrect class...
            IncorrectClassBMUIndices = find(~CorrectClassBMULogicalIndices);
            w_K = Mod.BMUs(IncorrectClassBMUIndices(1));
            
            if Mod.ClassLabels(Mod.BMUs(1)) == find(Data.ClassLabels(Data.GroundTruthLabelIndices, iData))                    
                % Record running stats of correctly classified samples for
                % each node...
                Mod.NodeStats{Mod.BMUs(1)}.push(Mod.x);
                
                % If the BMU classified the sample correctly, increment its
                % score...
                Mod.AccuracyHist(Mod.BMUs(1)) = Mod.AccuracyHist(Mod.BMUs(1)) + 1;
            end
            
            D_J = (Mod.x(Mod.known) - Mod.SOM.codebook(w_J, Mod.known));
            D_K = (Mod.x(Mod.known) - Mod.SOM.codebook(w_K, Mod.known));
            
            % Calculate the distance d_J of x (the sample) to w_J...
            d_J = Mod.metric_withmask(D_J, Mod.SOM.mask);
            
            % Calculate the distance d_K of x (the sample) to w_K...
            d_K = Mod.metric_withmask(D_K, Mod.SOM.mask);
            
            % Calculate the mu(x) part of the GLVQ update...
            mu_x = (d_J - d_K) / (d_J + d_K);
            
            % Calculate the f(mu,t) sigmoidal function part of the GLVQ
            % update...
            % f_mu_t = 1 / (1 + exp(-mu_x * Algo.UpdaterPhases{Algo.currentphase}.t));
            f_mu_t = 1 / (1 + exp(-10 * mu_x));
            
            % Estimate its derivative...
            d_f_d_mu = 10 * f_mu_t * (1 - f_mu_t);
            
            % d_f_d_mu = exp(-mu_x) / ((1 + exp(-mu_x))^2);
            
            % Update w_J...            
            Mod.SOM.codebook(w_J, Mod.known) =...
                Mod.SOM.codebook(w_J, Mod.known) +...
                    (((Mod.a * d_f_d_mu * d_K) / (d_J + d_K)) * (Mod.SOM.mask' .* D_J));
                
            % Update w_K...
            Mod.SOM.codebook(w_K, Mod.known) =... 
                Mod.SOM.codebook(w_K, Mod.known) -...
                    (((Mod.a * d_f_d_mu * d_J) / (d_J + d_K)) * (Mod.SOM.mask' .* D_K));
                

            %% Relevance determination...
            ClassMeans(1,:) = Mod.NodeStats{Mod.BMUs(1)}.mean();
            
            if isnan(ClassMeans(1,:))
                return;
            else
                ClassVar = Mod.NodeStats{Mod.BMUs(1)}.var();
                
                if ~any(isnan(ClassVar))
                    ClassVars(1,:) = ClassVar;
                else
                    ClassVars(1,:) = zeros(size(ClassMeans(1,:)));
                end
            end
            
            OtherClassNodes = Mod.BMUs(Mod.ClassLabels(Mod.BMUs) ~= Mod.ClassLabels(Mod.BMUs(1)));
            
            if ~isempty(OtherClassNodes)
                
%                 ClassData = Mod.SOM.codebook(OtherClassNodes, :);
%                 ClassNodeAccuracies = Mod.AccuracyHist(OtherClassNodes);
%                 if sum(ClassNodeAccuracies) == 0
%                     ClassNodeAccuracies = ones(size(ClassNodeAccuracies));
%                 end
%                 ClassNodeWeights = ClassNodeAccuracies + abs(min(ClassNodeAccuracies));
%                 ClassNodeWeights = ClassNodeWeights ./ norm(ClassNodeWeights,1);
%                 ClassNodeWeights(isnan(ClassNodeWeights)) = 0;

                
                for iOther = 1:size(OtherClassNodes,2)

                    OtherMean = Mod.NodeStats{OtherClassNodes(iOther)}.mean();

                    if ~any(isnan(OtherMean))

                        ClassMeans(2,:) = OtherMean;

                        OtherVar = Mod.NodeStats{OtherClassNodes(iOther)}.var();

                        if ~any(isnan(OtherVar))
                            ClassVars(2,:) = OtherVar;
                        else
                            ClassVars(2,:) = zeros(size(ClassMeans(2,:)));
                        end

                        break;

                    elseif iOther >= size(OtherClassNodes,2)
                        return;
                    end

                end
            else
                return;
            end

            % Fisher Criterion =
            %  (Between Class Variance)
            % --------------------------
            %  (Within Class Variance)
            if any(sum(ClassVars) == 0)
                FisherCriterion = var(ClassMeans);
            else
                FisherCriterion = var(ClassMeans) ./ sum(ClassVars);
            end

            % Watch out for nasty NaNs...
            FisherCriterion(isnan(FisherCriterion)) = 0;

            % Make the mask a weight distribution (unit norm)...            
            % Mask = (FisherCriterion ./ norm(FisherCriterion,1))';
            Mask = (FisherCriterion ./ max(FisherCriterion))';
            
            % Calculate running average of mask...
            Mod.MaskStats.push(Mask);

            % Save the mask for later...
            Mod.SOM.mask = Mod.MaskStats.mean();
            
        end
        
    end
    
end