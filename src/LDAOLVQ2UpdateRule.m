classdef LDAOLVQ2UpdateRule < UpdateRule
    
    methods (Static = true)
        
        function Mod = update(Algo, Mod, Data, iData, varargin)
            
            SampleClassLabel = find(Data.ClassLabels(Data.GroundTruthLabelIndices, iData));

            %% OLVQ1 (Optimized-Learning-Rate LVQ) update...
            if Mod.ClassLabels(Mod.BMUs(1)) == SampleClassLabel

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
            
            % Update a moving average mean and variance for the current
            % sample's class...
            Mod.ClassStats{SampleClassLabel}.push(Mod.x(Mod.known));
            
            % Grab the current means and variances for all classes...
            for iClass = 1:max(Mod.ClassLabels(:))                
                ClassMean = Mod.ClassStats{iClass}.mean();
                if ~isnan(ClassMean)
                    ClassMeans(iClass,:) = ClassMean;
                else
                    ClassMeans(iClass,:) = Mod.ZeroMask';
                end
                    
                
                ClassVar = Mod.ClassStats{SampleClassLabel}.var();
                if ~isnan(ClassVar)
                    ClassVars(iClass,:) = ClassVar;
                else
                    ClassVars(iClass,:) = Mod.ZeroMask';
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