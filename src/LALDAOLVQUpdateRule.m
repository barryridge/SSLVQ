classdef LALDAOLVQUpdateRule < UpdateRule
    
    methods (Static = true)
        
        function Mod = update(Algo, Mod, Data, iData, varargin)
            
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
            
            % Recalculate the BMU...
            Mod.BMUs(1) = Mod.findbmus_static(Mod.SOM.codebook, Mod.x, 1, QueryMask);

            %% OLVQ1 (Optimized-Learning-Rate LVQ) update...
            if Mod.ClassLabels(Mod.BMUs(1)) == find(Data.ClassLabels(Data.GroundTruthLabelIndices, iData))

                % Update the learning rate for the winning node...
                Mod.Alphas(Mod.BMUs(1)) = Mod.Alphas(Mod.BMUs(1)) / (1 + Mod.Alphas(Mod.BMUs(1)));

                % Update the BMU codebook vector (LVQ)...
                Mod.SOM.codebook(Mod.BMUs(1),Mod.known) =...
                    Mod.SOM.codebook(Mod.BMUs(1),Mod.known) +...
                        Mod.Alphas(Mod.BMUs(1)) * (Mod.x(Mod.known) - Mod.SOM.codebook(Mod.BMUs(1),Mod.known));
                    
                % If the BMU classified the sample correctly, increment its
                % score...
                Mod.AccuracyHist(Mod.BMUs(1)) = Mod.AccuracyHist(Mod.BMUs(1)) + 1;

            else

                % Update the learning rate for the winning node...
                Mod.Alphas(Mod.BMUs(1)) = min(Mod.Alphas(Mod.BMUs(1)) / (1 - Mod.Alphas(Mod.BMUs(1))), 1);

                % Update the BMU codebook vector (LVQ)...
                Mod.SOM.codebook(Mod.BMUs(1),Mod.known) =...
                    Mod.SOM.codebook(Mod.BMUs(1),Mod.known) -...
                        Mod.Alphas(Mod.BMUs(1)) * (Mod.x(Mod.known) - Mod.SOM.codebook(Mod.BMUs(1),Mod.known));
                    
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