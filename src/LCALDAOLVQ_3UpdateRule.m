classdef LCALDAOLVQUpdateRule < UpdateRule
    
    methods (Static = true)
        
        function Mod = update(Algo, Mod, Data, iData, varargin)
            
            skip = false;
            
            %% Relevance determination...
            ClassMeans(1,:) = Mod.NodeStats{Mod.BMUs(1)}.mean();
            
            if isnan(ClassMeans(1,:))
                skip = true;
            else
                ClassVar = Mod.NodeStats{Mod.BMUs(1)}.var();
                
                if ~any(isnan(ClassVar))
                    ClassVars(1,:) = ClassVar;
                else
                    ClassVars(1,:) = zeros(size(ClassMeans(1,:)));
                end
            end
            
            if ~skip
                OtherClassNodes = Mod.BMUs(Mod.ClassLabels(Mod.BMUs) ~= Mod.ClassLabels(Mod.BMUs(1)));
            end
            
            if ~skip && ~isempty(OtherClassNodes)
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
                        skip = true;
                    end

                end
            else
                skip = true;
            end

            if ~skip            
                
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
                
                % Recalculate the BMUs based on the current locally
                % class-adaptive mask...
                Mod.BMUs = Mod.findbmus_static(Mod.SOM.codebook, Mod.x, 1, Mask);
                
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

            else

                % Update the learning rate for the winning node...
                Mod.Alphas(Mod.BMUs(1)) = min(Mod.Alphas(Mod.BMUs(1)) / (1 - Mod.Alphas(Mod.BMUs(1))), 1);

                % Update the BMU codebook vector (LVQ)...
                Mod.SOM.codebook(Mod.BMUs(1),Mod.known) =...
                    Mod.SOM.codebook(Mod.BMUs(1),Mod.known) -...
                        Mod.Alphas(Mod.BMUs(1)) * (Mod.x(Mod.known) - Mod.SOM.codebook(Mod.BMUs(1),Mod.known));

            end                            
            
        end
        
    end
    
end