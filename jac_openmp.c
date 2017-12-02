#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#define  max(a,b) ((a)>(b)?(a):(b))

int N = (256+2);
double   maxeps = 0.1e-7;
int itmax = 1000;
int i,j,k;
double eps;
double omp_get_wtime();
void relax();
void resid();
void init();
void verify(); 
double start = 0;


int main(int an, char **as)
{
	int it;
	double start_begin = omp_get_wtime();
	if(an > 1) {
		maxeps = atof(as[1]);
	}
	if(an > 2) {
		itmax = atoi(as[2]);
	}
	if(an > 3) {
		N = atoi(as[3]) + 2;
	}
	double A [N][N],  B [N][N];

	init(A);
	double strat_algorithm = omp_get_wtime();
	for(it=1; it<=itmax; it++)
	{
		eps = 0.;
		relax(A, B);
		resid(A, B);
		if (eps < maxeps) break;
	}
	printf("%lf ",omp_get_wtime() - strat_algorithm);
	verify(A);
	printf("%d %d %lf ", it - 1, N - 2, eps);
	printf("%lf ",omp_get_wtime() - start_begin);
	printf("\n");
	return 0;
}

void init(double (*A)[N])
{ 
	#pragma omp parallel for  private(i,j)
	for(i=0; i<=N-1; i++)
	for(j=0; j<=N-1; j++)
	{
		if(i==0 || i==N-1 || j==0 || j==N-1) A[i][j]= 0.;
		else A[i][j]= ( 1. + i + j ) ;
	}
} 

void relax(double (*A)[N], double (*B)[N])
{
	#pragma omp parallel for   private(i,j) 
	for(i=1; i<=N-2; i++)
	for(j=1; j<=N-2; j++)
	{
		B[i][j]=(A[i-1][j]+A[i+1][j]+A[i][j-1]+A[i][j+1])/4.;
	}
}

void resid(double (*A)[N], double (*B)[N])
{ 
	#pragma omp parallel for   private(i,j)  
	for(i=1; i<=N-2; i++)
	for(j=1; j<=N-2; j++)
	{
		double e;
		e = fabs(A[i][j] - B[i][j]);         
		A[i][j] = B[i][j]; 
		if (eps < e) {
		#pragma omp critical
		{
			eps = e;
		}
		}
	}
}

void verify(double (*A)[N])
{
	double s;
	s=0.; 
	#pragma omp parallel for   private(i,j) reduction(+:s)  
	for(i=0; i<=N-1; i++)
	for(j=0; j<=N-1; j++)
	{
		s=s+A[i][j]*(i+1)*(j+1)/(N*N);
	}
}
