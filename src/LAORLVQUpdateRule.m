classdef LAORLVQUpdateRule < UpdateRule
    
    methods (Static = true)
        
        function Mod = update(Algo, Mod, Data, iData, varargin)
            
            %% USE INDIVIDUAL NODE ALPHA VALUES (FROM OLVQ) TO ESTIMATE ---
            % THE RELEVANCE OF INDIVIDUAL NODES...
            %--------------------------------------------------------------
            % NodeRelevances = 1  - Mod.Alphas;
            % Codebook = Mod.SOM.codebook(NodeRelevances >= mean(NodeRelevances(:)), :);
            % ClassLabels = Mod.ClassLabels(NodeRelevances >= mean(NodeRelevances(:)));
            Codebook = Mod.SOM.codebook;
            ClassLabels = Mod.ClassLabels;
            
            %% LDA ---
            %--------------------------------------------------------------
            for iClass = 1:max(ClassLabels(:))

                ClassData = Codebook(ClassLabels == iClass, :);

                % Just in case the cluster only contains
                % one node...
                if size(ClassData,1) == 1
                    ClassMeans(iClass,:) = ClassData;
                    ClassVars(iClass,:) = zeros(size(ClassData));
                else
                    ClassMeans(iClass,:) = mean(ClassData);
                    ClassVars(iClass,:) = var(ClassData);
                end
            end

            % Fisher Criterion =
            %  (Between Class Variance)
            % --------------------------
            %  (Within Class Variance)
            FisherCriterion = var(ClassMeans) ./ sum(ClassVars);

            % Watch out for nasty NaNs...
            FisherCriterion(isnan(FisherCriterion)) = 0;

            % Make the mask a weight distribution (unit norm)...
            Mod.SOM.mask = (FisherCriterion ./ norm(FisherCriterion,1))';

            
            %% OLVQ1 (Optimized-Learning-Rate LVQ) update...
            %--------------------------------------------------------------
            if Mod.ClassLabels(Mod.BMUs(1)) == find(Data.ClassLabels(Data.GroundTruthLabelIndices, iData))

                % Update the learning rate for the winning node...
                Mod.Alphas(Mod.BMUs(1)) = Mod.Alphas(Mod.BMUs(1)) / (1 + Mod.Alphas(Mod.BMUs(1)));

                % Update the BMU codebook vector (LVQ)...
                Mod.SOM.codebook(Mod.BMUs(1),Mod.known) =...
                    Mod.SOM.codebook(Mod.BMUs(1),Mod.known) +...
                        Mod.Alphas(Mod.BMUs(1)) * (Mod.x(Mod.known) - Mod.SOM.codebook(Mod.BMUs(1),Mod.known));                                    

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