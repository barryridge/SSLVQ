function [cnear, n] = find_nearpoint(Dist, cut, numb, chois)
% find n elements nearest to fixed point

b1 = -1; 
b2 = -1; 
ds = 0.1*cut;
dm = 0.01*cut;
if chois == 1
  for je = 1:1000
     cnear = find(Dist<=cut);
     n = length(cnear);
     if n > numb
         cut = cut*(1-ds); 
         b1 = je; 
     elseif n < numb 
         cut = cut*(1+ds); 
         b2 = je; 
     end
     if b2-b1 == 1
       if ds >= dm+dm
           ds = ds-dm;
       else
           ds = ds*0.9;
       end
     end
     if n==numb || (ds<=0 && abs(b1-b2)==1)  
         break; 
     end
  end

else
  for je = 1:1000
     cnear = find(Dist>=cut);
     n = length(cnear);
     if n > numb
         cut = cut*(1+ds); 
         b1 = je; 
     elseif n < numb 
         cut = cut*(1-ds); 
         b2 = je; 
     end
     if b2-b1 == 1
       if ds >= dm+dm
           ds = ds-dm;
       else
           ds = ds*0.9;
       end
     end
     if n==numb || (ds<=0 && abs(b1-b2)==1)  
         break; 
     end
  end

end