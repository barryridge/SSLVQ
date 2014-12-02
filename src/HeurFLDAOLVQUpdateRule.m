classdef HeurFLDAOLVQUpdateRule < UpdateRule
    
    methods (Static = true)
        
        function Mod = update(Algo, Mod, Data, iData, varargin)
            
             %% OLVQ1 (Optimized-Learning-Rate LVQ) update...
            if isnan(Mod.AuxDists(1)) || isnan(Mod.AuxDistStats.mean())
                
                return;
                
            elseif Mod.AuxDists(1) < Mod.AuxDistStats.mean() - Mod.AuxDistStats.std()           

                % Update the learning rate for the winning node...
                Mod.Alphas(Mod.BMUs(1)) = Mod.Alphas(Mod.BMUs(1)) / (1 + Mod.Alphas(Mod.BMUs(1)));

                % Update the BMU codebook vector (LVQ)...
                Mod.SOM.codebook(Mod.BMUs(1),Mod.known) =...
                    Mod.SOM.codebook(Mod.BMUs(1),Mod.known) +...
                        Mod.Alphas(Mod.BMUs(1)) * (Mod.x(Mod.known) - Mod.SOM.codebook(Mod.BMUs(1),Mod.known));
                    
                % If the BMU classified the sample correctly, increment its
                % score...
                Mod.AccuracyHist(Mod.BMUs(1)) = Mod.AccuracyHist(Mod.BMUs(1)) + 1;

            elseif Mod.AuxDists(1) > Mod.AuxDistStats.mean() + Mod.AuxDistStats.std()

                % Update the learning rate for the winning node...
                Mod.Alphas(Mod.BMUs(1)) = min(Mod.Alphas(Mod.BMUs(1)) / (1 - Mod.Alphas(Mod.BMUs(1))), 1);

                % Update the BMU codebook vector (LVQ)...
                Mod.SOM.codebook(Mod.BMUs(1),Mod.known) =...
                    Mod.SOM.codebook(Mod.BMUs(1),Mod.known) -...
                        Mod.Alphas(Mod.BMUs(1)) * (Mod.x(Mod.known) - Mod.SOM.codebook(Mod.BMUs(1),Mod.known));
                    
                % If the BMU classified the sample incorrectly, decrement its
                % score...
                % Mod.AccuracyHist(Mod.BMUs(1)) = Mod.AccuracyHist(Mod.BMUs(1)) - 1;

            end
            
            % Label the nodes in the codebook based on the auxiliary
            % distances.
            % Label 1: Nodes that don't appear to match the sample
            %          (large auxiliary distance).
            % Label 2: Nodes that appear to match the sample (small
            %          auxiliary distances).
            ClassLabels = ones(size(Mod.SOM.codebook, 1),1);
            ClassLabels(Mod.BMUs(Mod.AuxDists < (Mod.AuxDistStats.mean() - Mod.AuxDistStats.std())),:) = 2;

            % If we have nodes accurately predicting both the current class
            % and at least one of the other classes, we can do feature
            % relevance determination...
            if Mod.AuxDists(1) < Mod.AuxDistStats.mean() &&...
               sum(Mod.AccuracyHist(ClassLabels==1,:))>1 && sum(Mod.AccuracyHist(ClassLabels==2,:))>1
                
                % For each class, calculate a weighted mean and variance based
                % on the accuracy histogram as calculated above...
                for iClass = 1:max(ClassLabels(:))

                    ClassData = Mod.SOM.codebook(ClassLabels == iClass, :);
                    ClassNodeAccuracies = Mod.AccuracyHist(ClassLabels == iClass);
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
                % Mask = (FisherCriterion ./ norm(FisherCriterion,1))';
                Mask = (FisherCriterion ./ max(FisherCriterion))';

                % Calculate running average of mask...
                Mod.MaskStats.push(Mask);

                % Save the mask for later...
                Mod.SOM.mask = Mod.MaskStats.mean();

            end
            
        end
        
    end
    
end