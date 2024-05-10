#include <iostream>
#include <math.h>

// function to add element of two arrays

void add(int n, float *x, float *y)
{
    for (int i = 0; i < n; i++)
        y[i] = x[i] + y[i];
}

int main(void)
{
    int N = 1<<20;  // 1M elements ? 
    float *x = new float[N];
    float *y = new float[N];

    for (int i = 0; i < N; i++){
        x[i] = 1.0f;
        y[1] = 2.0f;
    }

    add(N, x, y);

    // check for error, and values should be 3.0f
    float maxError = 0.0f;
    for (int i = 0; i < N; i++)
        maxError = fmax(maxError, fabs(y[i] - 3.0f)); 
    std::cout << "Max Error: " << maxError << std::endl;

    // free memory
    delete [] x;
    delete [] y;

    return 0;
}