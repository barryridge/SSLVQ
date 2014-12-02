function x = normalize(x , methode)

%
% x = normalize(x , [methode=1]);
%
%
%
% Author : Sébastien PARIS : sebastien.paris@lsis.org
% -------


if (nargin < 2)

    methode                           = 1;

end

if(methode == 0)

    return;

end


if (methode == 1)

    mindata                           = min(x , [] , 2);

    maxdata                           = max(x , [] , 2);

    temp                              = maxdata - mindata;

    temp(temp==0)                     = 1;

    x                                 = 2*(x - mindata(: , ones(1 , size(x , 2))))./(temp(: , ones(1 , size(x , 2)))) - 1;


end

if (methode == 2)


    mx                            = mean(x , 2);

    res                           = x - mx(: , ones(1 , size(x , 2))) ;

    stdx                          = sqrt(1/(size(x , 2) - 1)*sum(res.*res , 2));

    stdx(stdx==0)                 = 1;

    x                             = res./stdx(: , ones(1 , size(x , 2)));

end


if (methode == 3)


    mindata                       = min(x(:));

    x                             = log(x - mindata + 1 +eps);

    mx                            = mean(x , 2);

    res                           = x - mx(: , ones(1 , size(x , 2))) ;

    stdx                          = sqrt(1/(size(x , 2) - 1)*sum(res.*res , 2));

    stdx(stdx==0)                 = 1;

    x                             = res./stdx(: , ones(1 , size(x , 2)));


end


if (methode == 4)


    mindata                       = min(x(:));

    x                             = 1./(1 + exp(-(x - mindata)));

    %     mx                            = mean(x , 2);
    %
    %     res                           = x - mx(: , ones(1 , size(x , 2))) ;
    %
    %     stdx                          = sqrt(1/(size(x , 2) - 1)*sum(res.*res , 2));
    %
    %     x                             = res./stdx(: , ones(1 , size(x , 2)));


end