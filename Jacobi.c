// Jacobi iteration from
// https://devblogs.nvidia.com/parallelforall/openacc-example-part-1/
// Compile: pgcc -m64 -acc -fast -Minfo=accel openacc_test.c
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void work(float *A, float *Anew, int m, int n)
{
    float err = 1e10;
    float tol = 20;
    int iter = 0, iter_max = 100000;

    #pragma acc data copy(A[0:m*n-1]), create(Anew[0:m*n-1])
    {
        while( err > tol && iter < iter_max )
        {
            err=0.0;
            {
              #pragma acc parallel loop reduction(max:err) collapse(2)
              for( int j = 1; j < n-1; j++ ) {
                for( int i = 1; i < m-1; i++ ) {

                  Anew[j*m+i] = 0.25 * (A[j*m+i+1] + A[j*m+i-1] +
                                        A[(j-1)*m+i] + A[(j+1)*m+i]);

                  err = max(err, abs(Anew[j*m+i] - A[j*m+i]));
                }
              }

              #pragma acc parallel loop collapse(2)
              for( int j = 1; j < n-1; j++ ) {
                for( int i = 1; i < m-1; i++ ) {
                  A[j*m+i] = Anew[j*m+i];
                }
              }
            }
            iter++;
            //printf("Iter %d: err: %f\n", iter, err);
        }
    }
}

int main()
{
    int m = 256;
    int n = 256;
    clock_t start, end;
    double duration;

    start = clock();
    for (int k = 0; k < 100; k ++)
    {
        float * A = (float *)malloc (sizeof(float) * m * n);
        float * Anew = (float *)malloc (sizeof(float) * m * n);

        // Initialize with noise
        for (int j=0; j < n; j++)
            for (int i=0; i < m; i++)
            {
                A[j*m+i] = rand() * 255.0 / RAND_MAX;
            }


        work(A, Anew, m, n);
        free(A);
        free(Anew);
    }
    end = clock();
    duration = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Processing time = %f s\n",duration);


    printf("Success\n");
    return 0;
}

