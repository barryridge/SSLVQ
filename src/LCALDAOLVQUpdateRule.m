classdef LCALDAOLVQUpdateRule < UpdateRule
    
    methods (Static = true)
        
        function Mod = update(Algo, Mod, Data, iData, varargin)
            
            
            %% Local relevance determination...
            
            ClassMeans(1,:) = Mod.SOM.codebook(Mod.BMUs(1),:);
            
            OtherClassNodes = Mod.BMUs(Mod.ClassLabels(Mod.BMUs) ~= Mod.ClassLabels(Mod.BMUs(1)));
            
            ClassMeans(2,:) = Mod.SOM.codebook(OtherClassNodes(1),:);
            
            FisherCriterion = var(ClassMeans);            

            % Watch out for nasty NaNs...
            FisherCriterion(isnan(FisherCriterion)) = 0;

            % Make the mask a weight distribution (unit norm)...
            Mask = (FisherCriterion ./ norm(FisherCriterion,1))';

            % Recalculate the BMUs based on the current locally
            % class-adaptive mask...
            BMUChoice = Mod.findbmus_static(Mod.SOM.codebook([Mod.BMUs(1) OtherClassNodes(1)],:), Mod.x, 1, Mask);
            
            if BMUChoice ~= 1
                Mod.BMUs = OtherClassNodes(1);
            end
            
                
            %% OLVQ1 (Optimized-Learning-Rate LVQ) update...
            if Mod.ClassLabels(Mod.BMUs(1)) == find(Data.ClassLabels(Data.GroundTruthLabelIndices, iData))

                % Update the learning rate for the winning node...
                Mod.Alphas(Mod.BMUs(1)) = Mod.Alphas(Mod.BMUs(1)) / (1 + Mod.Alphas(Mod.BMUs(1)));

                % Update the BMU codebook vector (LVQ)...
                Mod.SOM.codebook(Mod.BMUs(1),Mod.known) =...
                    Mod.SOM.codebook(Mod.BMUs(1),Mod.known) +...
                        Mod.Alphas(Mod.BMUs(1)) * (Mod.x(Mod.known) - Mod.SOM.codebook(Mod.BMUs(1),Mod.known));                                    

                % Record running stats of correctly classified samples for
                % each node...
                Mod.NodeStats{Mod.BMUs(1)}.push(Mod.x);
                
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

            end
            
            %% Global relevance determination...
            % For each class, calculate a weighted mean and variance based
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