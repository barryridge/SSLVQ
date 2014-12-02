classdef Utils < handle
    
    properties
        
    end
    
    methods (Static = true)
        
        %% --------------------- *** RADIUS *** ---------------------------
        %------------------------------------------------------------------
        function r = radius(type, rini, rfin, train_length, shift, t)
            
            % switch Mod.radius_type, % radius
            %     case 'linear', r = rini + (rfin-rini) * ((t - shift) - 1) / ((train_length / (1/shift))-1);
            % end
            r = rini + (rfin - rini) * ((t - shift) - 1) / (train_length - 1);
            
            if ~r, r=eps; end % zero neighborhood radius may cause div-by-zero error  
            
        end
        
        
        %% ----------------- *** NEIGHBOURHOOD *** ------------------------
        %------------------------------------------------------------------
        function h = neighbourhood(type, Ud, bmu, r)
            
            switch type, % neighborhood function 
                case 'bubble',        h = (Ud(:,bmu) <= r);
                case 'gaussian',      h = exp(-(Ud(:,bmu).^2)/(2*r*r));
                case 'gaussian_somn', h = exp(-(Ud(:,bmu)/r).^2);
                case 'cutgauss',      h = exp(-(Ud(:,bmu).^2)/(2*r*r)) .* (Ud(:,bmu) <= r);
                case 'ep',            h = (1 - (Ud(:,bmu).^2)/(r*r)) .* (Ud(:,bmu) <= r);
            end
            
        end


        %% ----------------- *** LEARNING RATE *** ------------------------
        %------------------------------------------------------------------
        function a = learningrate(type, seed, train_length, shift, t, std)
            
            % Defaults...            
            if nargin < 6, std = train_length / 4; end
                        
            %% Calculate current learning rate...            
            switch type,                        
                case 'constant',...
                    a = seed;
                case 'linear',...
                    a = (1-(t - shift) / train_length) * seed;
                case 'linear_inc',...
                    a = ((t - shift) / train_length) * seed;
                case 'inv',...
                    a = seed / (1 + 99*((t - shift)-1) / (train_length-1));
                case 'power',...
                    a = seed * (0.005/seed)^(((t - shift)-1)/train_length);
                case 'inv_nonnull',...
                    a = t^(-1/2);
                case 'gaussian',...
                    a = seed * exp(-(2*std^2)^(-1) * (t - shift).^2);
                
            end
            
        end
        
        %% ----------------- *** NODE ACTIVATIONS *** ---------------------
        %------------------------------------------------------------------
        function Activations = activations(type, Ud, Dx, CoVar, CovInv, Distances, bmu, r)
            
            switch type, % activation function
                
                % From "Dyslexic and Category-Specific Aphasic Impairments
                % in a Self-Organizing Feature Map Model of the Lexicon" by
                % Miikkulainen.
                case {'bubble', 'miikkulainen'},...
                    hHebb = (Ud(:,bmu) <= r);
                    HebbDists = Distances .* hHebb;
                    HebbDists(HebbDists == 0) = NaN;
                    Activations = 1 - (HebbDists - min(HebbDists)) / (max(HebbDists) - min(HebbDists));
                    Activations(isnan(Activations)) = 0;
                
                % A modification of a function from "Probabilistic measures for responses of
                % Self-Organizing Map units" by Alhoniemi, Himberg, and
                % Vesanto.
                case {'response', 'alhoniemi'},...
                    % Original:
                    Activations = 1 ./ (1 + Distances);
                    % Modification:
                    % Activations = 1 ./ (1 + (Distances .* Ud(:,bmu)));
                    Activations = Activations ./ norm(Activations,1);
                    
                case {'gaussian'},...
                    Activations = exp(-(Ud(:,bmu)).^2/(2*r*r));
                
                % From "Self-Organizing Mixture Networks for Probability
                % Density Estimation" by Yin and Allison, 2001.
                case {'gaussian_somn', 'yin'},...
                    for i = 1:size(Dx,1)
                        % No normalizing constant needed
                        % Respon(NdxNeuro)=exp(-0.5*log(det(Model.C{NdxNeuro}))-0.5 * VectorDiff'*Model.CInv{NdxNeuro}*VectorDiff);
                        % Activations(i,:) = exp(-0.5*log(det(obj.Cov{i}))-0.5 * Dx(i)'*obj.CovInv{i}*Dx(i));
                        Activations(i,:) = exp(-0.5*log(det(CoVar{i})) - 0.5 * Dx(i,:)*CovInv{i}*Dx(i,:)');
                        if ~isfinite(Activations(i,:))
                            Activations(i,:)=0;
                        end
                    end
                                        
                
            end
            
        end
        
        %% ------- *** HELLINGER DISTANCE *** -----------------------------
        %------------------------------------------------------------------
        function dist = hellinger(Distribution1, Distribution2, varargin)
            
            % Normalize distributions...
            NormDistribution1 = Distribution1 / sum(Distribution1);
            NormDistribution2 = Distribution2 / sum(Distribution2);
            
            % Calculate Hellinger distance...
            dist = sqrt(sum((sqrt(NormDistribution1) - sqrt(NormDistribution2)).^2));
            
        end
        
        %% ------- *** NORMED HELLINGER DISTANCE *** ----------------------
        %------------------------------------------------------------------
        function dist = normed_hellinger(Distribution1, Distribution2, varargin)
            
            % Normalize distributions...
            NormDistribution1 = Distribution1 / sum(Distribution1);
            NormDistribution2 = Distribution2 / sum(Distribution2);
            
            % Calculate Hellinger distance...
            dist = sqrt(sum((sqrt(NormDistribution1) - sqrt(NormDistribution2)).^2));
            
            % Normalize Hellinger distance...
            dist = dist / sqrt(2);
        end
        
        %% ------- *** CHI-SQUARED DISTANCE *** ---------------------------
        %------------------------------------------------------------------
        function dist = chisquared(Distribution1, Distribution2, varargin)
            
            % Add 1% noise...
            maxDist1 = max(Distribution1);
            maxDist2 = max(Distribution2);
            NormDistribution1 = Distribution1 +...
                                rand(size(Distribution1)) * 0.01 * max(maxDist1, xor(maxDist1, 1));
            NormDistribution2 = Distribution2 +...
                                rand(size(Distribution2)) * 0.01 * max(maxDist2, xor(maxDist2, 1));
            
            % Normalize distributions...
            NormDistribution1 = NormDistribution1 / sum(NormDistribution1);
            NormDistribution2 = NormDistribution2 / sum(NormDistribution2);
            
            % Calculate Chi-squared distance...
            dist = sum(((NormDistribution1 - NormDistribution2).^2)./NormDistribution2);
            
        end
        
        %% ------- *** KULLBACK-LEIBLER DIVERGENCE *** --------------------
        %------------------------------------------------------------------
        function dist = kullbackleibler(Distribution1, Distribution2, varargin)
            
            % Add 1% noise...
            maxDist1 = max(Distribution1);
            maxDist2 = max(Distribution2);
            NormDistribution1 = Distribution1 +...
                                rand(size(Distribution1)) * 0.01 * max(maxDist1, xor(maxDist1, 1));
            NormDistribution2 = Distribution2 +...
                                rand(size(Distribution2)) * 0.01 * max(maxDist2, xor(maxDist2, 1));
            
            % Normalize distributions...
            NormDistribution1 = NormDistribution1 / sum(NormDistribution1);
            NormDistribution2 = NormDistribution2 / sum(NormDistribution2);
            
            % Calculate Kullback-Leibler divergence...
            dist = sum(NormDistribution1 .* log(NormDistribution1./NormDistribution2));
            
        end
        
        %% ------- *** TOTAL VARIATION DISTANCE *** -----------------------
        %------------------------------------------------------------------
        function dist = totalvariation(Distribution1, Distribution2, varargin)
            
            % Normalize distributions...
            NormDistribution1 = Distribution1 / sum(Distribution1);
            NormDistribution2 = Distribution2 / sum(Distribution2);
            
            % Calculate total variation distance...
            dist = sum(abs(NormDistribution1 - NormDistribution2)) / 2;
            
        end
        
        %% -------------- *** CROSS CORRELATION *** -----------------------
        % WARNING: This is just a sandbox right now, not functional
        %          as a metric/distance.
        %------------------------------------------------------------------
        function dist = crosscorrelation(Distribution1, Distribution2, varargin)
            
            % Normalize distributions...
            NormDistribution1 = Distribution1 / sum(Distribution1);
            NormDistribution2 = Distribution2 / sum(Distribution2);
            
            % Calculate Cross-correlation distance...
            dist = 1 / sum(sum(xcorr2(NormDistribution1, NormDistribution2)));
            
        end
        
        %% ----------------- *** RIDGE DISTANCE *** -----------------------
        %------------------------------------------------------------------
        function dist = ridge(Distribution1, Distribution2, varargin)
            
            % Get the universal distance matrix from the arguments list...
            D = varargin{1};
            
            % Normalize distributions...
            NormDistribution1 = Distribution1 / sum(Distribution1);
            NormDistribution2 = Distribution2 / sum(Distribution2);
            
            % Calculate ridge distance...
            N = length(NormDistribution1);
            foo = repmat(NormDistribution1, 1, N);
            bar = repmat(NormDistribution2', N, 1);
            dist = sum(sum(foo * (bar .* D)));
            
        end
        
        %% ------------ *** EARTH MOVER'S DISTANCE *** --------------------
        %------------------------------------------------------------------
        function dist = earthmovers(Distribution1, Distribution2, varargin)
            
            % Get the universal distance matrix from the arguments list...
            D = varargin{1};
            
            % Normalize distributions...
            NormDistribution1 = Distribution1 / sum(Distribution1);
            NormDistribution2 = Distribution2 / sum(Distribution2);
            
            % Calculate ridge distance...
            if sum(isnan(NormDistribution1)) > 0 || sum(isnan(NormDistribution2)) > 0
                dist = NaN;
            else
                dist = emd_mex(NormDistribution1', NormDistribution2', D);
            end
            
        end
        
    end
    
end
