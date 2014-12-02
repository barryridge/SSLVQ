function [VX VY] = clipvoronoi(VX, VY, Bounds)

    % Bounding box...
    Bound_X1 = Bounds(1);    
    Bound_Y1 = Bounds(2);
    Bound_X2 = Bounds(3);
    Bound_Y2 = Bounds(4);        

    % This nested loop clips the Voronoi diagram (VX, VY) with the borders
    % of the bounding box.
    for i = 1:size(VX,2)
        
        for j = 1:size(VX,1)
            
            if VX(j,i) < Bound_X1           
                m = (VY(2,i) - VY(1,i)) / (VX(2,i) - VX(1,i));
                
                c = VY(2,i) - (m*VX(2,i));
                
                VY_j_i_new = (m*Bound_X1) + c;                                                                
                
                if VY_j_i_new >= Bound_Y1 && VY_j_i_new <= Bound_Y2
                    VY(j,i) = VY_j_i_new;
                    VX(j,i) = Bound_X1;
                end                
                
            end
            
            if VX(j,i) > Bound_X2           
                m = (VY(2,i) - VY(1,i)) / (VX(2,i) - VX(1,i));
                
                c = VY(2,i) - (m*VX(2,i));                
                
                VY_j_i_new = (m*Bound_X2) + c;
                
                if VY_j_i_new >= Bound_Y1 && VY_j_i_new <= Bound_Y2
                    VY(j,i) = VY_j_i_new;
                    VX(j,i) = Bound_X2;
                end

            end
            
            if VY(j,i) < Bound_Y1           
                m = (VY(2,i) - VY(1,i)) / (VX(2,i) - VX(1,i));
                
                c = VY(2,i) - (m*VX(2,i));
                
                VX_j_i_new = (Bound_Y1 - c) / m;
                
                if VX_j_i_new >= Bound_X1 && VX_j_i_new <= Bound_X2
                    VX(j,i) = VX_j_i_new;
                    VY(j,i) = Bound_Y1;
                end
                
            end
            
            if VY(j,i) > Bound_Y2
                m = (VY(2,i) - VY(1,i)) / (VX(2,i) - VX(1,i));
                
                c = VY(2,i) - (m*VX(2,i));
                
                VX_j_i_new = (Bound_Y2 - c) / m;
                                
                if VX_j_i_new >= Bound_X1 && VX_j_i_new <= Bound_X2
                    VX(j,i) = VX_j_i_new;
                    VY(j,i) = Bound_Y2;
                end
                
            end
            
        end                
        
    end
    
    TempX = [VX(1,:) VX(2,:)]';
    TempY = [VY(1,:) VY(2,:)]';

    TempData = unique([TempX TempY], 'rows');

    TRI = delaunay(TempData(:,1), TempData(:,2));