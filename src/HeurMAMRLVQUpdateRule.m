classdef HeurMAMRLVQUpdateRule < UpdateRule
    
    methods (Static = true)
        
        function Mod = update(Algo, Mod, Data, iData, varargin)
            
            %% Check that we have the things we need to continue...
            if isnan(Mod.AuxDists(1)) || isnan(Mod.AuxDists(2)) || isnan(Mod.AuxDistStats.mean())                
                return;
            end
            
            %% Update...
                    
            % Shorten variable names for readability...
            m1 = Mod.SOM.codebook(Mod.BMUs(1),Mod.known);
            m2 = Mod.SOM.codebook(Mod.BMUs(2),Mod.known);

            % Current sample...
            z = Mod.x(Mod.known);

            % Point on the surface Sij between m_i & m_j...
            p = (m1 + m2) / 2;

            % Vector normal to Sij...
            n = (m1 - m2);

            % The current sample z projected onto the
            % surface Sij... 
            z12 = ((dot((p - z),n) / dot(n,n)) * n) - z;

            % The risk of deciding in favour of the 'class'
            % for bmu...
            b_h_to_l1 = (Mod.AuxDists(1) < Mod.AuxDistStats.mean());

            % The risk of deciding in favour of the 'class'
            % for bmu2...
            b_h_to_l2 = (Mod.AuxDists(2) < Mod.AuxDistStats.mean());

            d = sqrt((m1 - m2).^2);

            % If we fall inside the window, update the
            % codebook vectors using MAMR LVQ rules...
            if sqrt((z - z12).^2) <= (Mod.epsilon / 2)

                % Update the BMU codebook vector (LVQ)...
                Mod.SOM.codebook(Mod.BMUs(1),Mod.known) =...
                    Mod.SOM.codebook(Mod.BMUs(1),Mod.known) -...
                    Mod.a * ( ((b_h_to_l2 - b_h_to_l1) * (m1 - z12)) /...
                          (Mod.epsilon  * d) );

                % Update the BMU2 codebook vector (LVQ)...
                Mod.SOM.codebook(Mod.BMUs(2),Mod.known) =...
                    Mod.SOM.codebook(Mod.BMUs(2),Mod.known) -...
                    Mod.a * ( ((b_h_to_l1 - b_h_to_l2) * (m2 - z12)) /...
                          (Mod.epsilon  * d) );

                % Increment t for non-null updates...
                Mod.t_nonnull = Mod.t_nonnull + 1;

                outside_window = false;

            else

                outside_window = true;

            end

            % If we're surrounded by either two correct or two incorrect nodes,
            % or otherwise if we fall outside the window and are closer to one node
            % than the other, these are good opportunities to do feature selection...
            if ~xor(b_h_to_l1, b_h_to_l2) || outside_window
                % If the closest BMU is of the correct 'class'...
                if b_h_to_l1
                    % Update the feature weights (RLVQ)...
                    Mod.SOM.mask = max(Mod.SOM.mask - (Mod.a_f * abs(Mod.Dx(Mod.BMUs(1),:)))', Mod.ZeroMask);
                    Mod.SOM.mask = Mod.SOM.mask / sum(Mod.SOM.mask);

                % If the closest BMU is of the incorrect 'class'...
                else
                    % Update the feature weights (RLVQ)...
                    Mod.SOM.mask = Mod.SOM.mask + (Mod.a_f * abs(Mod.Dx(Mod.BMUs(1),:)))';
                    Mod.SOM.mask = Mod.SOM.mask / sum(Mod.SOM.mask);
                end
            end
            
        end
        
    end
    
end