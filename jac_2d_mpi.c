#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>
#define  max(a,b) ((a)>(b)?(a):(b))

int N = (512 + 2);
double   maxeps = 0.1e-7;
int itmax = 4000;
int k;
void relax();
void resid();
void init();
void verify(); 
MPI_Request req[4];
MPI_Status status[4];
double start = 0;
int myrank, myranksize;
int startrow, lastrow, nrow;
double **A, **B;


int main(int an, char **as)
{
	MPI_Init(&an, &as);
  	MPI_Comm_rank(MPI_COMM_WORLD, &myrank);  /* what is my id (rank)? */
  	MPI_Comm_size(MPI_COMM_WORLD, &myranksize);  /* how many processes? */
	if(an > 1) {
		itmax = atoi(as[1]);
	}
	if(an > 2) {
		N = atoi(as[2]) + 2;
	} 
	int it;
	startrow = (N * myrank)/myranksize;
	lastrow = (N * (myrank + 1))/myranksize -1;
	nrow = lastrow - startrow + 1;
	A = (double**) malloc((nrow + 2) * sizeof(double*));
	B = (double**) malloc((nrow) * sizeof(double*));
	int i;
	for(i = 0; i < nrow + 2; i++) {
		A[i] = (double*) malloc(N * sizeof(double));
	}
	for(i = 0; i < nrow; i++) {
		B[i] = (double*) malloc(N * sizeof(double));
	}
	MPI_Barrier(MPI_COMM_WORLD);
	init();
	double t1=MPI_Wtime();
	for(it=1; it<=itmax; it++)
	{
		resid();
		relax();
	}
	double t2 = MPI_Wtime() - t1;
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Reduce(&t2, &t2, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
	if(myrank == 0) {
		printf("%d %d %d %lf\n",myranksize, it - 1, N - 2, t2);
	}
	verify();
	MPI_Finalize();
	return 0;
}

void init()
{ 
	double s = 0;
	double count = 0;
	int i, j;
	for(i=1; i<=nrow; i++) {
		for(j=0; j <= N - 1; j++)
		{
			B[i - 1][j] = 1. + startrow + i + j - 1;
			A[i][j] = 0.;	
		}
	}
} 

void relax()
{
	if(myranksize > 1) {
		if(myrank!=0)
			MPI_Irecv(&A[0][0],N,MPI_DOUBLE, myrank-1, 1235, 
			MPI_COMM_WORLD, &req[0]);
		if(myrank!=myranksize-1)
			MPI_Isend(&A[nrow][0],N,MPI_DOUBLE, myrank+1, 1235, 
			MPI_COMM_WORLD, &req[2]);
		if(myrank!=myranksize-1)
	  		MPI_Irecv(&A[nrow+1][0], N, MPI_DOUBLE, myrank+1, 1236, MPI_COMM_WORLD, &req[3]);
	 	if(myrank!=0)
			MPI_Isend(&A[1][0],N,MPI_DOUBLE, myrank-1, 1236, 
			MPI_COMM_WORLD,&req[1]);
		int  ll=4;
	 	int shift=0;
		if (myrank==0) {ll=2;shift=2;}
		if (myrank==myranksize-1) {ll=2;}
		MPI_Waitall(ll, &req[shift],&status[0]);
	}
	int i, j;
	for(i=1; i<=nrow; i++)
	for(j=1; j<=N-2; j++)
	{
		B[i-1][j]=(A[i-1][j]+A[i+1][j]+A[i][j-1]+A[i][j+1])/4.;
	}
}

void resid()
{ 

	int i, j;
	for(i=1; i<=nrow; i++)
	for(j=1; j<=N-2; j++)
	{
		if (((i==1)&&(myrank==0))||((i==nrow)&&(myrank==myranksize-1))) 
			continue;		        
		A[i][j] = B[i - 1][j]; 
	}
}

void verify()
{
	double s;
	double sum;
	s=0.;
	int i, j;
	for(i=1; i<=nrow; i++)
	for(j=0; j<=N-1; j++)
	{
		s=s+A[i - 1][j]*(i + startrow) * (j +  1)/ (N * N);
	} 
	MPI_Reduce(&s, &sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	if(myrank == 0) {
		printf("s = %lf\n", sum);
	}
}

