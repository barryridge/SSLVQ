function [NewV NewC] = clipvoronoin(D)

    %
    % NOTE: THIS FUNCTION DOESN'T WORK!  TRYING A NEW STRATEGY WITH
    % findboundedcells()
    %

    Thres = 0.000001;
    
    [VX VY] = voronoi(D(:,1), D(:,2));
    [NewVX NewVY] = clipvoronoi(VX, VY, [0 0 1 1]);
    [V C] = voronoin(D);
    
    % Get rid of that nasty infinite vertex... *shudder* :-(
    NewV = V(2:end,:);
    
    % Ok, we want to first of all replace those unclipped vertices in V
    % that are not the infinite vertex with their clipped counterparts.
    %
    % First we pass through the Voronoi line start-vertices...
    StartVerts = [VX(1,:)' VY(1,:)'];
    NewStartVerts = [NewVX(1,:)' NewVY(1,:)'];
    for i = 1:size(StartVerts,1)
        if any(StartVerts(i,:) < 0) || any(StartVerts(i,:) > 1)
            for j = 1:size(NewV,1)
                if sum(abs(StartVerts(i,:) - NewV(j,:)) < Thres) == 2
                    NewV(j,:) = NewStartVerts(i,:);
                end
            end
        end
    end
    
    % Next, we pass through the Voronoi line end-vertices...
    EndVerts = [VX(2,:)' VY(2,:)'];
    NewEndVerts = [NewVX(2,:)' NewVY(2,:)'];
    for i = 1:size(EndVerts,1)
        if any(EndVerts(i,:) < 0) || any(EndVerts(i,:) > 1)
            for j = 1:size(NewV,1)
                if sum(abs(EndVerts(i,:) - NewV(j,:)) < Thres) == 2
                    NewV(j,:) = NewEndVerts(i,:);
                end
            end
        end
    end
    
    % Ok, now we loop through the Voronoi cells...
    for i = 1:length(C)                
        
        % This cell is already bounded, so we can just patch it directly...
        if all(C{i}~=1)
            
            NewC{i} = C{i} - 1;
                        
            % patch(V(C{i},1),V(C{i},2),i);
            patch(NewV(NewC{i},1),NewV(NewC{i},2),i); % use color i.                        
            
        % This cell is unbounded, so we'll try to bound it, then patch
        % it...
        else
            
            % Pick out the non-infinite vertices of the cell...
            NewC{i} = C{i}(C{i}~=1) - 1;
            
            NewCLength = length(NewC{i});
            for iNewC = 1:NewCLength
                
                MatchFound = false;
                
                % Look for matches to the non-infinite vertices in
                % NewStartVerts & replace them with corresponding vertices
                % in NewEndVerts...
                for iNewStartVerts = 1:size(NewStartVerts,1)
                    if sum(abs(NewStartVerts(iNewStartVerts,:) - NewV(NewC{i}(iNewC),:)) < Thres) == 2
                        NewV(end+1,:) = NewEndVerts(iNewStartVerts,:);
                        NewC{i}(end+1) = size(NewV,1);
                        MatchFound = true;
                        break;
                    end
                end
                
                % If we can't find any matches in NewStartVerts, look in
                % NewEndVerts...
                if ~MatchFound
                    for iNewEndVerts = 1:size(NewEndVerts,1)
                        if sum(abs(NewEndVerts(iNewEndVerts,:) - NewV(NewC{i}(iNewC),:)) < Thres) == 2
                            NewV(end+1,:) = NewStartVerts(iNewEndVerts,:);
                            NewC{i}(end+1) = size(NewV,1);
                            break;
                        end
                    end
                end
            end
            
            % There will probably be duplicate vertices after the above
            % process
            
            CellV = NewV(NewC{i},:);
            CellVConvHull = convhull(CellV(:,1), CellV(:,2));

            % patch(CellV(CellVConvHull,1),CellV(CellVConvHull,2),i); % use color i.
            
        end
        
    end
    axis equal