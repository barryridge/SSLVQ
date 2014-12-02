classdef LDALVQ1_3UpdateRule < UpdateRule
    
    methods (Static = true)
        
        function Mod = update(Algo, Mod, Data, iData, varargin)

            %% LDALVQ1 (LVQ1 with LDA based relevance determination) update...
            if Mod.ClassLabels(Mod.BMUs(1)) == find(Data.ClassLabels(Data.GroundTruthLabelIndices, iData))

                % Update the BMU codebook vector (LVQ)...
                Mod.SOM.codebook(Mod.BMUs(1),Mod.known) =...
                    Mod.SOM.codebook(Mod.BMUs(1),Mod.known) +...
                        Mod.a * (Mod.x(Mod.known) - Mod.SOM.codebook(Mod.BMUs(1),Mod.known));
                    
                % Record running stats of correctly classified samples for
                % each node...
                Mod.NodeStats{Mod.BMUs(1)}.push(Mod.x);
                
                % If the BMU classified the sample correctly, increment its
                % score...
                Mod.AccuracyHist(Mod.BMUs(1)) = Mod.AccuracyHist(Mod.BMUs(1)) + 1;

            else

                % Update the BMU codebook vector (LVQ)...
                Mod.SOM.codebook(Mod.BMUs(1),Mod.known) =...
                    Mod.SOM.codebook(Mod.BMUs(1),Mod.known) -...
                        Mod.a * (Mod.x(Mod.known) - Mod.SOM.codebook(Mod.BMUs(1),Mod.known));                                    

            end
            
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