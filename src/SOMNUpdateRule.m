classdef SOMNUpdateRule < UpdateRule
    
    methods (Static = true)
        
        function Mod = update(Algo, Mod, Data, iData, varargin)
            
%             MySample=Samples(:,ceil(NumSamples*rand(1)));
%             if NdxStep<0.5*NumSteps   
%                 % Ordering phase: linear decay
%                 LearningRate=0.4*(1-NdxStep/NumSteps);
%                 % LearningRate=1*(1-NdxStep/NumSteps);
%                 MyRadius=MaxRadius*(1-(NdxStep-1)/NumSteps);
%             else
%                 % Convergence phase: keep constant
%                 LearningRate=0.01;
%                 MyRadius=0.1;
%             end

%             % Calculate the node responses...
%             for NdxNeuro=1:NumNeuro
%                 VectorDiff=MySample-Model.Means{NdxNeuro};
%                 % No normalizing constant needed
%                 Respon(NdxNeuro)=exp(-0.5*log(det(Model.C{NdxNeuro}))-0.5*...
%                     VectorDiff'*Model.CInv{NdxNeuro}*VectorDiff);
%                 if ~isfinite(Respon(NdxNeuro))
%                     Respon(NdxNeuro)=0;
%                 end
%             end
%             MySum=sum(Respon);
%             if MySum>0
%                 Respon=Respon/sum(Respon);
%             else
%                 Respon=zeros(1,NumNeuro);
%                 disp('Bad responsibilities')
%             end
%             
%             % Find the BMU by maximizing over the responses...
%             [Maximo NdxWin]=max(Respon);
%             [CoordWin(1) CoordWin(2)]=ind2sub([NumRowsMap NumColsMap],NdxWin);
%             
%             % Update the neurons...
%             for NdxNeuro=1:NumNeuro
%                 % Topological distance
%                 [MyCoord(1) MyCoord(2)]=ind2sub([NumRowsMap NumColsMap],NdxNeuro);
%                 TopolDist=norm(CoordWin-MyCoord);
%                 Coef=LearningRate*exp(-(TopolDist/MyRadius)^2);
%                 % Update this neuron
%                 Model.Pi(NdxNeuro)=Coef*Respon(NdxNeuro)+...
%                     (1-Coef)*Model.Pi(NdxNeuro);
%                 VectorDiff=MySample-Model.Means{NdxNeuro};
%                 Model.Means{NdxNeuro}=Coef*MySample+...
%                     (1-Coef)*Model.Means{NdxNeuro};            
%                 MyC=Coef*VectorDiff*VectorDiff'+...
%                     (1-Coef)*Model.C{NdxNeuro};
%                 if rcond(MyC)>1.0e-6
%                     Model.C{NdxNeuro}=MyC;
%                     Model.CInv{NdxNeuro}=inv(Model.C{NdxNeuro});
%                 end
%             end
%             
%             MyRadius=MaxRadius*(1-(NdxStep-1)/NumSteps);
%             r = rini + (rfin - rini) * ((t - shift) - 1) / (train_length - 1);
%             case 'bubble',   h = (Ud(:,bmu) <= r);
%             case 'gaussian', h = exp(-(Ud(:,bmu).^2)/(2*r*r));                

            %% Update...

            % Probabilities...
            % Model.Pi(NdxNeuro)=Coef*Respon(NdxNeuro) + (1-Coef)*Model.Pi(NdxNeuro);
            Mod.Probabilities = Mod.Probabilities + Mod.a * Mod.h .* (Mod.Activations - Mod.Probabilities);

            % Means...
            % Model.Means{NdxNeuro}=Coef*MySample + (1-Coef)*Model.Means{NdxNeuro};
            Mod.SOM.codebook(:,Mod.known) =...
                Mod.SOM.codebook(:,Mod.known) - Mod.a * Mod.h(:,ones(sum(Mod.known),1)) .* Mod.Dx;
            
            % Covariances...
            % MyC = Coef * VectorDiff * VectorDiff' + (1-Coef)*Model.C{NdxNeuro};
            % if rcond(MyC)>1.0e-6
            %     Model.C{NdxNeuro}=MyC;
            %     Model.CInv{NdxNeuro}=inv(Model.C{NdxNeuro});
            % end
            for iNode = 1:size(Mod.SOM.codebook,1)
                MyCov = Mod.Cov{iNode} + Mod.a * Mod.h(iNode) * (Mod.Dx(iNode,:)' * Mod.Dx(iNode,:) - Mod.Cov{iNode});
                
                if rcond(MyCov)>1.0e-6
                    Mod.Cov{iNode}=MyCov;
                    Mod.CovInv{iNode}=inv(Mod.Cov{iNode});
                end
            end
            
            
        end
        
    end
    
end