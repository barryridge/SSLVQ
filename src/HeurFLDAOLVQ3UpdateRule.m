classdef HeurFLDAOLVQ3UpdateRule < UpdateRule
    
    methods (Static = true)
        
        function Mod = update(Algo, Mod, Data, iData, varargin)
            
             %% OLVQ1 (Optimized-Learning-Rate LVQ) update...
            if isnan(Mod.AuxDists(1)) || isnan(Mod.AuxDistStats.mean())
                
                return;
                
            elseif Mod.AuxDists(1) < Mod.AuxDistStats.mean() % - Mod.AuxDistStats.std()           

                % Update the learning rate for the winning node...
                Mod.Alphas(Mod.BMUs(1)) = Mod.Alphas(Mod.BMUs(1)) / (1 + Mod.Alphas(Mod.BMUs(1)));

                % Update the BMU codebook vector (LVQ)...
                Mod.SOM.codebook(Mod.BMUs(1),Mod.known) =...
                    Mod.SOM.codebook(Mod.BMUs(1),Mod.known) +...
                        Mod.Alphas(Mod.BMUs(1)) * (Mod.x(Mod.known) - Mod.SOM.codebook(Mod.BMUs(1),Mod.known));
                    
                % If the BMU classified the sample correctly, increment its
                % score...
                Mod.AccuracyHist(Mod.BMUs(1)) = Mod.AccuracyHist(Mod.BMUs(1)) + 1;
                
                % Record running stats of correctly classified samples for
                % each node...
                Mod.NodeStats{Mod.BMUs(1)}.push(Mod.x);
                                    
            elseif Mod.AuxDists(1) > Mod.AuxDistStats.mean() % + Mod.AuxDistStats.std()

                % Update the learning rate for the winning node...
                Mod.Alphas(Mod.BMUs(1)) = min(Mod.Alphas(Mod.BMUs(1)) / (1 - Mod.Alphas(Mod.BMUs(1))), 1);

                % Update the BMU codebook vector (LVQ)...
                Mod.SOM.codebook(Mod.BMUs(1),Mod.known) =...
                    Mod.SOM.codebook(Mod.BMUs(1),Mod.known) -...
                        Mod.Alphas(Mod.BMUs(1)) * (Mod.x(Mod.known) - Mod.SOM.codebook(Mod.BMUs(1),Mod.known));                                    

            end
            
            % Label the nodes in the codebook based on the auxiliary
            % distances.
            % Label 1: Nodes that don't appear to match the sample
            %          (large auxiliary distance).
            % Label 2: Nodes that appear to match the sample (small
            %          auxiliary distances).
            ClassLabels = ones(size(Mod.SOM.codebook, 1),1);
            % ClassLabels(Mod.BMUs(Mod.AuxDists < (Mod.AuxDistStats.mean() - Mod.AuxDistStats.std())),:) = 2;
            ClassLabels(Mod.BMUs(Mod.AuxDists < Mod.AuxDistStats.mean()),:) = 2;
            
            if all((ClassLabels == 2) == 0)
                return;
            end

            %% Relevance determination...
%             ClassMeans(1,:) = Mod.NodeStats{Mod.BMUs(1)}.mean();
%             
%             if isnan(ClassMeans(1,:))
%                 return;
%             else
%                 ClassVar = Mod.NodeStats{Mod.BMUs(1)}.var();
%                 
%                 if ~any(isnan(ClassVar))
%                     ClassVars(1,:) = ClassVar;
%                 else
%                     ClassVars(1,:) = zeros(size(ClassMeans(1,:)));
%                 end
%             end
%             
%             OtherClassNodes = Mod.BMUs(ClassLabels(Mod.BMUs) ~= ClassLabels(Mod.BMUs(1)));
%             
%             if ~isempty(OtherClassNodes)
%                 for iOther = 1:size(OtherClassNodes,2)
% 
%                     OtherMean = Mod.NodeStats{OtherClassNodes(iOther)}.mean();
% 
%                     if ~any(isnan(OtherMean))
% 
%                         ClassMeans(2,:) = OtherMean;
% 
%                         OtherVar = Mod.NodeStats{OtherClassNodes(iOther)}.var();
% 
%                         if ~any(isnan(OtherVar))
%                             ClassVars(2,:) = OtherVar;
%                         else
%                             ClassVars(2,:) = zeros(size(ClassMeans(2,:)));
%                         end
% 
%                         break;
% 
%                     elseif iOther >= size(OtherClassNodes,2)
%                         return;
%                     end
% 
%                 end
%             else
%                 return;
%             end

            % Some of the notation in the following is derived                        
            % from equations (18) in "Multivariate Online Kernel
            % Density Estimation" by Kristan et al.

            for iClass = 1:max(ClassLabels(:))

                ClassNodeIndices = find(ClassLabels==iClass);

                % Sum accuracy weights...
                w_j = sum(Mod.AccuracyHist(ClassNodeIndices));                                                        

                % Calculate the class mean...
                mu_j = 0;
                for iNode = 1:size(ClassNodeIndices,1)

                    mu_i = Mod.NodeStats{ClassNodeIndices(iNode)}.mean();

                    if isnan(mu_i)
                        mu_i = 0;
                    end

                    mu_j = mu_j + (Mod.AccuracyHist(ClassNodeIndices(iNode)) *...
                                   mu_i);
                end                                    
                mu_j = w_j^(-1) * mu_j;

                % Calculate the class variance...
                sig_j = 0;
                for iNode = 1:size(ClassNodeIndices,1)

                    mu_i = Mod.NodeStats{ClassNodeIndices(iNode)}.mean();
                    sig_i = Mod.NodeStats{ClassNodeIndices(iNode)}.var();

                    if isnan(mu_i)
                        mu_i = 0;
                    end

                    if isnan(sig_i)
                        sig_i = 0;
                    end

                    sig_j = sig_j + (Mod.AccuracyHist(ClassNodeIndices(iNode)) *...
                                     (sig_i + mu_i.^2));
                end
                sig_j = (w_j.^(-1) * sig_j) - mu_j.^2;

                % Save...
                if any(isnan(mu_j))
                    return;
                    % ClassMeans(iClass,:) = zeros(1,size(Mod.SOM.codebook,2));
                else                                
                    ClassMeans(iClass,:) = mu_j;
                end

                if any(isnan(mu_j))
                    return;
                    % ClassVars(iClass,:) = zeros(1,size(Mod.SOM.codebook,2));
                else
                    ClassVars(iClass,:) = sig_j;
                end

            end
            
%             if any(any(isnan(ClassMeans))) || any(any(isnan(ClassVars)))
%                 return;
%             end

            % Fisher Criterion =
            %  (Between Class Variance)
            % --------------------------
            %  (Within Class Variance)
            if any(any(ClassVars <= 0))
                return;
                % FisherCriterion = var(ClassMeans);
            else
                FisherCriterion = var(ClassMeans) ./ sum(ClassVars);
                % FisherCriterion = var(ClassMeans);
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