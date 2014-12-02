classdef HeurFC2GLVQUpdateRule < UpdateRule
    
    methods (Static = true)
        
        function Mod = update(Algo, Mod, Data, iData, varargin)
            
             % Grab w_J, the best matching node of the correct class...
            CorrectClassBMULogicalIndices = Mod.AuxDists < Mod.AuxDistStats.mean() - Mod.AuxDistStats.std();
            if sum(CorrectClassBMULogicalIndices) == 0
                return;
            end
            CorrectClassBMUIndices = find(CorrectClassBMULogicalIndices);
            w_J = Mod.BMUs(CorrectClassBMUIndices(1));
            
            % Grab w_K, the next closest node of an incorrect class...
            InCorrectClassBMULogicalIndices = Mod.AuxDists > Mod.AuxDistStats.mean() + Mod.AuxDistStats.std();
            if sum(InCorrectClassBMULogicalIndices) == 0
                return;
            end
            IncorrectClassBMUIndices = find(InCorrectClassBMULogicalIndices);
            w_K = Mod.BMUs(IncorrectClassBMUIndices(1));
            
            if Mod.AuxDists(1) < Mod.AuxDistStats.mean() - Mod.AuxDistStats.std()                    
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
            
            if ~isnan(mu_x)
                
                % Calculate the f(mu,t) sigmoidal function part of the GLVQ
                % update...
                f_mu_t = 1 / (1 + exp(-mu_x * Algo.UpdaterPhases{Algo.currentphase}.t));
                % f_mu_t = 1 / (1 + exp(-10 * mu_x));

                % Estimate its derivative...
                d_f_d_mu = f_mu_t * (1 - f_mu_t);

                % d_f_d_mu = exp(-mu_x) / ((1 + exp(-mu_x))^2);

                % Probability of w_J being of the correct class?
                p_J = 1 - Mod.AuxDists(CorrectClassBMUIndices(1));

                % Probability of w_K being of the incorrect class?
                p_K = Mod.AuxDists(IncorrectClassBMUIndices(1));

                % Update w_J...            
                Mod.SOM.codebook(w_J, Mod.known) =...
                    Mod.SOM.codebook(w_J, Mod.known) +...
                        (((Mod.a * d_f_d_mu * d_K * (p_J + p_K + 1)) / (d_J + d_K)^2) * (Mod.SOM.mask' .* D_J));

                % Update w_K...
                Mod.SOM.codebook(w_K, Mod.known) =... 
                    Mod.SOM.codebook(w_K, Mod.known) -...
                        (((Mod.a * d_f_d_mu * d_J * (p_J + p_K + 1)) / (d_J + d_K)^2) * (Mod.SOM.mask' .* D_K));                
            end
            
            % Label the nodes in the codebook based on the auxiliary
            % distances.
            % Label 1: Nodes that don't appear to match the sample
            %          (large auxiliary distance).
            % Label 2: Nodes that appear to match the sample (small
            %          auxiliary distances).
            ClassLabels = ones(size(Mod.SOM.codebook, 1),1);
            ClassLabels(Mod.BMUs(Mod.AuxDists < (Mod.AuxDistStats.mean() - Mod.AuxDistStats.std())),:) = 2;
            % ClassLabels(Mod.BMUs(Mod.AuxDists < Mod.AuxDistStats.mean()),:) = 2;
            % ClassLabels(Mod.AuxDists < nanmean(Mod.AuxDists) - nanstd(Mod.AuxDists),:) = 2;
            
            if all((ClassLabels == 2) == 0)
                return;
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
            
            OtherClassNodes = Mod.BMUs(ClassLabels(Mod.BMUs) ~= ClassLabels(Mod.BMUs(1)));
            
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

            % Some of the notation in the following is derived                        
            % from equations (18) in "Multivariate Online Kernel
            % Density Estimation" by Kristan et al.

%             for iClass = 1:max(ClassLabels(:))
% 
%                 ClassNodeIndices = find(ClassLabels==iClass);
% 
%                 % Sum accuracy weights...
%                 w_j = sum(Mod.AccuracyHist(ClassNodeIndices));                                                        
% 
%                 % Calculate the class mean...
%                 mu_j = 0;
%                 for iNode = 1:size(ClassNodeIndices,1)
% 
%                     mu_i = Mod.NodeStats{ClassNodeIndices(iNode)}.mean();
% 
%                     if isnan(mu_i)
%                         mu_i = 0;
%                     end
% 
%                     mu_j = mu_j + (Mod.AccuracyHist(ClassNodeIndices(iNode)) *...
%                                    mu_i);
%                 end                                    
%                 mu_j = w_j^(-1) * mu_j;
% 
%                 % Calculate the class variance...
%                 sig_j = 0;
%                 for iNode = 1:size(ClassNodeIndices,1)
% 
%                     mu_i = Mod.NodeStats{ClassNodeIndices(iNode)}.mean();
%                     sig_i = Mod.NodeStats{ClassNodeIndices(iNode)}.var();
% 
%                     if isnan(mu_i)
%                         mu_i = 0;
%                     end
% 
%                     if isnan(sig_i)
%                         sig_i = 0;
%                     end
% 
%                     sig_j = sig_j + (Mod.AccuracyHist(ClassNodeIndices(iNode)) *...
%                                      (sig_i + mu_i.^2));
%                 end
%                 sig_j = (w_j.^(-1) * sig_j) - mu_j.^2;
% 
%                 % Save...
%                 if any(isnan(mu_j))
%                     return;
%                     % ClassMeans(iClass,:) = zeros(1,size(Mod.SOM.codebook,2));
%                 else                                
%                     ClassMeans(iClass,:) = mu_j;
%                 end
% 
%                 if any(isnan(mu_j))
%                     return;
%                     % ClassVars(iClass,:) = zeros(1,size(Mod.SOM.codebook,2));
%                 else
%                     ClassVars(iClass,:) = sig_j;
%                 end
% 
%             end
            
            if any(any(isnan(ClassMeans))) || any(any(isnan(ClassVars)))
                return;
            end

            % Fisher Criterion =
            %  (Between Class Variance)
            % --------------------------
            %  (Within Class Variance)
            if any(any(ClassVars <= 0))
                % return;
                FisherCriterion = var(ClassMeans);
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