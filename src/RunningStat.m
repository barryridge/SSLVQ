% Adapted from http://www.johndcook.com/standard_deviation.html

classdef RunningStat < handle
   
    properties (SetAccess = private, Hidden = true)
        
        m_n;
        m_oldM;
        m_newM;
        m_oldS;
        m_newS;
        
    end
    
    methods
        
        function obj = RunningStat()
            obj.m_n = 0;
        end
        
        
        function obj = clear(obj)
            obj.m_n = 0;
        end
        
        
        function obj = push(obj, x)
            
            if isnan(x)
                x = 0;
            end
            
            obj.m_n = obj.m_n + 1;

            % See Knuth TAOCP vol 2, 3rd edition, page 232
            if (obj.m_n == 1)
                
                obj.m_oldM = x;
                obj.m_newM = x;
                obj.m_oldS = zeros(size(x));
            
            else
                
                obj.m_newM = obj.m_oldM + ((x - obj.m_oldM) ./ obj.m_n);
                obj.m_newS = obj.m_oldS + ((x - obj.m_oldM) .* (x - obj.m_newM));
    
                % set up for next iteration
                obj.m_oldM = obj.m_newM; 
                obj.m_oldS = obj.m_newS;
                
            end
            
        end
        
        
        function values = count(obj)
        
            values = obj.m_n;
            
        end

        function current_mean = mean(obj)
        
            if obj.m_n > 0
                current_mean = obj.m_newM;
            else
                current_mean = NaN;
            end
            
        end
        

        function current_variance = var(obj)
        
            if obj.m_n > 1
                current_variance = obj.m_newS / (obj.m_n - 1);
            else
                current_variance = NaN;
            end
            
        end
        

        function current_std = std(obj)
        
            current_std = sqrt( obj.var() );
        end
        
    end
    
end