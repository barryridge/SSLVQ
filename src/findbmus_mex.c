
/*


  Predict class label for a Test data with a trained model.

  Usage
  ------

  [ytest_est , dist] = NN_predict(Xtest , Wproto_est , yproto_est , [lambda_est] , [options]);

  
  Inputs
  -------

  Xtest                                 Test data (d x Ntest)
  Wproto_est                            Estimated prototypes weigths (d x Nproto)
  yproto_est                            Estimated prototypes labels  (1 x Nproto)
  lambda_est                            Estimated Weigths factor  (d x 1). Default lambda_est = ones(d , 1);
  options.metric                        = 1 for euclidian distance (default), = 2 for d4

  
  Outputs
  -------
  
  yproto_est                            Estimated labels  (1 x Ntest)
  dist                                  Distance between Xtest and Prototypes (Nproto x Ntest)


  To compile
  ----------


  mex  -g  -output NN_predict.dll NN_predict.c

  mex -f mexopts_intelamd.bat -output NN_predict.dll NN_predict.c

  

  Example 1
  ---------
  

  close all
  load ionosphere
  Nproto_pclass                      = 4*ones(1 , length(unique(y)));
  
  options.epsilonk                   = 0.005;
  options.epsilonl                   = 0.001;
  options.epsilonlambda              = 10e-8;
  options.xi                         = 10;
  options.nb_iterations              = 5000;
  options.metric_method              = 1;
  options.shuffle                    = 1;
  options.updatelambda               = 1;

  options.method                     = 7;
  options.holding.rho                = 0.7;
  options.holding.K                  = 1;


  X                                  = normalize(X);
  [Itrain , Itest]                   = sampling(X , y , options);
  [Xtrain , ytrain , Xtest , ytest]  = samplingset(X , y , Itrain , Itest);

  

  [Wproto , yproto , lambda]         = ini_proto(Xtrain , ytrain , Nproto_pclass);
  [Wproto_est , yproto_est , lambda_est,  E_GRLVQ]    = grlvq_model(Xtrain , ytrain , Wproto , yproto , lambda, options);
  
  [ytest_est , disttest]             = NN_predict(Xtest , Wproto_est , yproto_est , lambda_est , options);
  [ytrain_est , disttrain]           = NN_predict(Xtrain , Wproto_est , yproto_est , lambda_est , options);

  Perftrain                          = perf_classif(ytrain , ytrain_est); 
  Perftest                           = perf_classif(ytest , ytest_est);;

  dktrain                            = min(disttrain(yproto==0 , :));
  dltrain                            = min(disttrain(yproto~=0 , :));
  nutrain                            = (dktrain - dltrain)./(dktrain + dltrain);
  [tptrain , fptrain]                = basicroc(ytrain , nutrain);

   
  dktest                             = min(disttest(yproto==0 , :));
  dltest                             = min(disttest(yproto~=0 , :));
  nutest                             = (dktest - dltest)./(dktest + dltest);
  [tptest , fptest]                  = basicroc(ytest , nutest);


  disp('Performances Train/Test')
  disp([Perftrain , Perftest])
  
  figure(1)
  plot(E_GRLVQ);
  title('E_{GRLVQ}(t)' , 'fontsize' , 12)
  
  figure(2)
  stem(lambda_est);
  title('\lambda' , 'fontsize' , 12)

  figure(3)
  plot(fptrain , tptrain , fptest , tptest , 'r' , 'linewidth'  , 2)
  xlabel('false positive rate');
  ylabel('true positive rate');
  title('ROC curve','fontsize' , 12);
  legend(['Train'] , ['Test'])



 Author : S�bastien PARIS : sebastien.paris@lsis.org
 -------  Date : 04/09/2006

 Reference "A new Generalized LVQ Algorithm via Harmonic to Minimumm Distance Measure Transition", A.K. Qin, P.N. Suganthan and J.J. Liang,
 ---------  IEEE International Conference on System, Man and Cybernetics, 2004



*/


#include <math.h>
#include <mex.h>



typedef struct OPTIONS 
{
 
  int    metric_method;
  
} OPTIONS; 




/* Function prototypes */

void findbmus(double *, double *, double *, int, int, int, int , 
			  int *, double *, double *);

void qsindex (double  *, int *, int, int);

/*-------------------------------------------------------------------------------------------------------------- */


void mexFunction( int nlhs, mxArray *plhs[] , int nrhs, const mxArray *prhs[] )

{
    /* double *Xtest , *Wproto , *yproto , *lambda; */
    double *Xtest , *Wproto , *lambda;
    int metric_method;
	/* OPTIONS options = { 1}; */
	/* double *ytest_est , *dist; */
    double *Dx;
    double *Distances;
    int *BMUs;
	double *tmp;
	int d , Ntest  , Nproto;
	int i , j  , l;
	double  currentlabel , ind , temp;
	mxArray *mxtemp;

    /* [ytest_est , disttest] = NN_predict(Xtest , Wproto_est , yproto_est , lambda_est , options); */
    /* BMUs = findbmus(Xtest, Wproto_est, lambda_est, options); */

    /* Input 1  Xtrain */
	Xtest         = mxGetPr(prhs[0]);
    		
	if( mxGetNumberOfDimensions(prhs[0]) !=2 )
	{
		mexErrMsgTxt("Xtest must be (d x Ntest)");
	}
	
	d         = mxGetM(prhs[0]);
	Ntest     = mxGetN(prhs[0]);
	
	
	/* Input 2  Wproto */
	Wproto    = mxGetPr(prhs[1]);
	
	if(mxGetNumberOfDimensions(prhs[1]) !=2 || mxGetM(prhs[1]) != d)
	{
		mexErrMsgTxt("Wproto must be (d x Nproto)");
	}

	Nproto     = mxGetN(prhs[1]);


	/* Input 3   yproto */
	/* yproto         = mxGetPr(prhs[2]);
	
	if(mxGetNumberOfDimensions(prhs[2]) !=2 || mxGetN(prhs[2]) != Nproto)
	{
		mexErrMsgTxt("yproto must be (1 x Nproto)");
	} */
	

	/* Input 3   lambda */
	if (nrhs >= 3 && !mxIsEmpty(prhs[2]))
	{
		lambda    = mxGetPr(prhs[2]);
	}
	else
	{
		lambda = (double *)mxMalloc(d*sizeof(double));

		for (i = 0 ; i < d ; i++)
		{
			lambda[i] = 1.0;
		}
	}
	
    /* Input 4   metric_method */
	if (nrhs >= 4 && !mxIsEmpty(prhs[3]))
	{
		metric_method    = (int)*(mxGetPr(prhs[3]));
	}
	else
	{
        metric_method    = 1;
	}


    /* mexPrintf("%d", (int)metric_method); */

	/* if ( (nrhs >= 4) && !mxIsEmpty(prhs[3]) )
	{
		mxtemp = mxGetField(prhs[3] , 0 , "metric_method");
		
		if(mxtemp != NULL)
		{
            tmp = mxGetPr(mxtemp);
			options.metric_method = (int) tmp[0];
		}
	} */

	/* plhs[0]               = mxCreateDoubleMatrix(1 , Ntest, mxREAL);
	ytest_est             = mxGetPr(plhs[0]);
	
	plhs[1]               = mxCreateDoubleMatrix(Nproto , Ntest, mxREAL);
	dist                  = mxGetPr(plhs[1]); */
	
    plhs[0]               = mxCreateNumericMatrix(Nproto , Ntest, mxINT32_CLASS, mxREAL);
	BMUs                  = (int *) mxGetData(plhs[0]);
    
    plhs[1]               = mxCreateDoubleMatrix(d , Nproto, mxREAL);
	Dx                    = mxGetPr(plhs[1]);
	
    plhs[2]               = mxCreateDoubleMatrix(Nproto , Ntest, mxREAL);
	Distances             = mxGetPr(plhs[2]);
	

	/* Main Call */
	/* glvq_predict(Xtest , Wproto , yproto , lambda , d , Ntest , Nproto , options , 
		         ytest_est , Distances); */ 
	findbmus(Xtest , Wproto , lambda , d , Ntest , Nproto , metric_method , 
		     BMUs, Dx, Distances);
    
 
	if(nrhs < 3 || mxIsEmpty(prhs[2]))
	{

		mxFree(lambda);

	}
	
}


/*-------------------------------------------------------------------------------------------------------------- */

void findbmus(double *Xtest , double *Wproto , double *lambda , int d , int Ntest , int Nproto , int metric_method ,
			  int *BMUs, double *Dx, double *Distances)
{
	int i , j , l , ld , id , lNproto;
    int k, m;
	
	double  disttmp , temp;
    
    double *TempDistances;
    
    TempDistances = mxMalloc(d * Nproto *sizeof(double));
	
    switch (metric_method) {
        
      /* Sum-squared distance... */  
      case 1:         
            for(l = 0 ; l < Ntest ; l++)
            {
                ld       = l*d;
                lNproto  = l*Nproto;

                for (i = 0 ; i < Nproto ; i++)
                {
                    id      = i*d;
                    disttmp = 0.0;

                    for( j = 0 ; j < d ; j++)
                    {
                        temp     = (Wproto[j + id] - Xtest[j + ld]);
                        Dx[j + id] = temp;
                        disttmp += lambda[j]*temp*temp;
                    }

                    Distances[i + lNproto] = disttmp;
                    TempDistances[i + lNproto] = disttmp;
                    BMUs[i + lNproto] = i + 1;

                }

                /* 
                if (NBMUs)
                {
                    double lowest = 1.79769313486231*10e307; 

                    for (k = 0; k < NBMUs; k++)
                    {
                        for (m = lNproto; m < lNproto + Nproto; m++)
                        {
                            if (Distances[m] < lowest)
                            {
                                BMUs[k] = m;
                            }
                        }
                    }
                }
                else
                {
                */                                

                qsindex(TempDistances, BMUs, lNproto , lNproto + (Nproto - 1) );

            }
            
            break;
            
      /* Euclidean distance... */      
      case 2:                                  
            for(l = 0 ; l < Ntest ; l++)
            {
                ld       = l*d;
                lNproto  = l*Nproto;

                for (i = 0 ; i < Nproto ; i++)
                {
                    id      = i*d;
                    disttmp = 0.0;

                    for( j = 0 ; j < d ; j++)
                    {                        
                        temp     = (Wproto[j + id] - Xtest[j + ld]);
                        Dx[j + id] = temp;
                        disttmp += lambda[j]*temp*temp;
                    }

                    Distances[i + lNproto] = sqrt(disttmp);
                    TempDistances[i + lNproto] = disttmp;
                    BMUs[i + lNproto] = i + 1;

                }

                /* 
                if (NBMUs)
                {
                    double lowest = 1.79769313486231*10e307; 

                    for (k = 0; k < NBMUs; k++)
                    {
                        for (m = lNproto; m < lNproto + Nproto; m++)
                        {
                            if (Distances[m] < lowest)
                            {
                                BMUs[k] = m;
                            }
                        }
                    }
                }
                else
                {
                */

                qsindex(TempDistances, BMUs, lNproto , lNproto + (Nproto - 1) );

            }
          
            break;
      
      default:
            for(l = 0 ; l < Ntest ; l++)
            {
                ld       = l*d;
                lNproto  = l*Nproto;

                for (i = 0 ; i < Nproto ; i++)
                {
                    id      = i*d;
                    disttmp = 0.0;

                    for( j = 0 ; j < d ; j++)
                    {
                        temp     = (Wproto[j + id] - Xtest[j + ld]);
                        Dx[j + id] = temp;
                        disttmp += lambda[j]*temp*temp*temp*temp;
                    }

                    Distances[i + lNproto] = disttmp;
                    TempDistances[i + lNproto] = disttmp;
                    BMUs[i + lNproto] = i + 1;

                }

                qsindex(TempDistances, BMUs, lNproto , lNproto + (Nproto - 1) );

            }
            
            break;
    }
    
    mxFree(TempDistances);
		
}


/*----------------------------------------------------------------------------------------------------------------------------------------- */

void qsindex (double  *a, int *index , int lo, int hi)
{
/*  lo is the lower index, hi is the upper index
    of the region of array a that is to be sorted */
    int i=lo, j=hi , ind;
    double x=a[(lo+hi)/2] , h;

    /*  partition */
    do
    {    
        while (a[i]<x) i++; 
        while (a[j]>x) j--;
        if (i<=j)
        {
            h        = a[i]; 
			a[i]     = a[j]; 
			a[j]     = h;
			ind      = index[i];
			index[i] = index[j];
			index[j] = ind;
            i++; 
			j--;
        }
    }
	while (i<=j);

    /*  recursion */
    if (lo<j) qsindex(a , index , lo , j);
    if (i<hi) qsindex(a , index , i , hi);
}

/*-------------------------------------------------------------------------------------------------------------- */

/*
 *  This Quickselect routine is based on the algorithm described in
 *  "Numerical recipes in C", Second Edition,
 *  Cambridge University Press, 1992, Section 8.5, ISBN 0-521-43108-5
 *  This code by Nicolas Devillard - 1998. Public domain.
 */

/* 
#define ELEM_SWAP(a,b) { register double t=(a);(a)=(b);(b)=t; }

double quick_select(double arr[], int n) 
{
    int low, high ;
    int median;
    int middle, ll, hh;

    low = 0 ; high = n-1 ; median = (low + high) / 2;
    for (;;) {
        if (high <= low) /* One element only 
            return arr[median] ;

        if (high == low + 1) {  /* Two elements only 
            if (arr[low] > arr[high])
                ELEM_SWAP(arr[low], arr[high]) ;
            return arr[median] ;
        }

    /* Find median of low, middle and high items; swap into position low 
    middle = (low + high) / 2;
    if (arr[middle] > arr[high])    ELEM_SWAP(arr[middle], arr[high]) ;
    if (arr[low] > arr[high])       ELEM_SWAP(arr[low], arr[high]) ;
    if (arr[middle] > arr[low])     ELEM_SWAP(arr[middle], arr[low]) ;

    /* Swap low item (now in position middle) into position (low+1) 
    ELEM_SWAP(arr[middle], arr[low+1]) ;

    /* Nibble from each end towards middle, swapping items when stuck 
    ll = low + 1;
    hh = high;
    for (;;) {
        do ll++; while (arr[low] > arr[ll]) ;
        do hh--; while (arr[hh]  > arr[low]) ;

        if (hh < ll)
        break;

        ELEM_SWAP(arr[ll], arr[hh]) ;
    }

    /* Swap middle item (in position low) back into correct position 
    ELEM_SWAP(arr[low], arr[hh]) ;

    /* Re-set active partition 
    if (hh <= median)
        low = ll;
        if (hh >= median)
        high = hh - 1;
    }
}

#undef ELEM_SWAP
*/
