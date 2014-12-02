classdef LALDALVQUpdateRule < UpdateRule
    
    methods (Static = true)
        
        function Mod = update(Algo, Mod, Data, iData, varargin)
            
            %% GLVQ (Generalized Relevance Determination LVQ) update...
            
            % Grab w_J, the best matching node of the correct class...
            CorrectClassBMUIndices = find(Mod.ClassLabels(Mod.BMUs) == find(Data.ClassLabels(Data.GroundTruthLabelIndices, iData)));
            w_J = Mod.BMUs(CorrectClassBMUIndices(1));
            
            % Grab w_K, the next closest node of an incorrect class...
            IncorrectClassBMUIndices = find(Mod.ClassLabels(Mod.BMUs) ~= find(Data.ClassLabels(Data.GroundTruthLabelIndices, iData)));
            w_K = Mod.BMUs(IncorrectClassBMUIndices(1));
            
            % Let's calculate the distance between the test
            % query point and the hyperplane seperating
            % the BMUs of the different classes...
            % BMU vectors...
            a = Mod.SOM.codebook(w_J,Mod.known)';
            b = Mod.SOM.codebook(w_K,Mod.known)';

            % Point on the hyperplane...
            q = a + ((b - a)/2);

            % Query point...
            p = Mod.x(Mod.known)';

            % Norm to the hyperplane...
            n = (b - a);

            % Distance from point p to the hyperplane...
            dist = abs(dot((p - q), n)) / norm(n,1);

            % Use this to calculate the exponential
            % weighting factor...
            C = 1 / dist;

            % Calculate the feature weights for this
            % query...
            QueryMask = exp(C * Mod.SOM.mask) ./ sum(exp(C * Mod.SOM.mask));
            QueryMask(isnan(QueryMask)) = 1;
            QueryMask = QueryMask ./ norm(QueryMask,1);
            
            % Calculate the distance d_J of x (the sample) to w_J...
            d_J = Mod.metric((Mod.x(Mod.known) - Mod.SOM.codebook(w_J, Mod.known)), QueryMask);
            
            % Calculate the distance d_K of x (the sample) to w_K...
            d_K = Mod.metric((Mod.x(Mod.known) - Mod.SOM.codebook(w_K, Mod.known)), QueryMask);
            
            % Calculate the mu(x) part of the GLVQ update...
            mu_x = (d_J - d_K) / (d_J + d_K);
            
            % Calculate the f(mu,t) sigmoidal function part of the GLVQ
            % update...
            f_mu_t = 1 / (1 + exp(-mu_x * Algo.UpdaterPhases{Algo.currentphase}.t));
            
            % Estimate its derivative...
            d_f_d_mu = f_mu_t * (1 - f_mu_t);
            
            % Update w_J...            
            Mod.SOM.codebook(w_J, Mod.known) =...
                Mod.SOM.codebook(w_J, Mod.known) +...
                    (((Mod.a * d_f_d_mu * d_K) / (d_J + d_K)^2) * (Mod.x(Mod.known) - Mod.SOM.codebook(w_J, Mod.known)));
                
            % Update w_K...
            Mod.SOM.codebook(w_K, Mod.known) =...
                Mod.SOM.codebook(w_K, Mod.known) -...
                    (((Mod.a * d_f_d_mu * d_J) / (d_J + d_K)^2) * (Mod.x(Mod.known) - Mod.SOM.codebook(w_K, Mod.known)));
                

            % Record accuracy histogram for the nodes based on whether the
            % BMU correctly classifys the sample...
            if Mod.ClassLabels(Mod.BMUs(1)) == find(Data.ClassLabels(Data.GroundTruthLabelIndices, iData))
                
                % If the BMU classified the sample correctly, increment its
                % score...
                Mod.AccuracyHist(Mod.BMUs(1)) = Mod.AccuracyHist(Mod.BMUs(1)) + 1;
            else
                
                % If the BMU classified the sample incorrectly, decrement its
                % score...
                Mod.AccuracyHist(Mod.BMUs(1)) = Mod.AccuracyHist(Mod.BMUs(1)) - 1;
            end
                
            % For each class, calculated a weighted mean and variance based
            % on the accuracy histogram as calculated above...
            for iClass = 1:max(Mod.ClassLabels(:))

                ClassData = Mod.SOM.codebook(Mod.ClassLabels == iClass, :);
                ClassNodeAccuracies = Mod.AccuracyHist(Mod.ClassLabels == iClass);
                if sum(ClassNodeAccuracies) == 0
                    ClassNodeAccuracies = ones(size(ClassNodeAccuracies));
                end
                ClassNodeWeights = ClassNodeAccuracies + abs(min(ClassNodeAccuracies));
                ClassNodeWeights = ClassNodeWeights ./ norm(ClassNodeWeights,1);
                ClassNodeWeights(isnan(ClassNodeWeights)) = 0;

                % Just in case the class only contains
                % one node...
                if size(ClassData,1) == 1
                    ClassMeans(iClass,:) = ClassData;
                    ClassVars(iClass,:) = zeros(size(ClassData));
                else
                    % Calculate the weighted mean...
                    ClassMeans(iClass,:) = sum(repmat(ClassNodeWeights,1,size(ClassData,2)) .* ClassData,1);
                    % Calculate the weighted variance...
                    ClassVars(iClass,:) = sum(repmat(ClassNodeWeights,1,size(ClassData,2)) .* (ClassData - repmat(ClassMeans(iClass,:), size(ClassData,1), 1)).^2);
                end
                
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
            Mask = (FisherCriterion ./ norm(FisherCriterion,1))';

            % Calculate running average of mask...
            Mod.MaskStats.push(Mask);

            % Save the mask for later...
            Mod.SOM.mask = Mod.MaskStats.mean();
            
        end
        
    end
    
end