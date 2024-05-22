#include <iostream>
// #include <time.h>
#include <ctime>

using namespace std;

// the host function for calculating the vector sum
void sumArrayOnHost(float *a, float *b, float *c, const int N){
    for(int idx = 0; idx < N; idx++){
        c[idx] = a[idx] + b[idx]; // just take two elements and add them
    }
}

// Work on getting the initial Data into the code
void initialData(float *ip, int size){
    // each vector/ array sent to this function will get random 
    // data assigned to the memory location.
    // generate different seed for random number
    time_t t;  // t is of time time_t, and its address is sent to srand
    // srand((unsigned int) time(&t)); // the returned time_t value is casted
    // Following is the correct way to initialize in c++.
    srand(static_cast<unsigned int>(time(0)));
    for (int j=0; j < size; j++){
        ip[j] = (float) ( rand() & 0xFF ) / 10.0f;
        // if output is more than 8 bits it will essentially give you the last 8 bits of the value
        // https://stackoverflow.com/questions/14713102/what-does-and-0xff-do
    }
}

// Main function

int main(){
    int nElem = 1024;
    size_t nBytes = nElem * sizeof(float);
    // The size of size_t is implementation-dependent 
    // but is typically 4 bytes on 32-bit systems and 8 bytes
    // on 64-bit systems.

    float *h_a, *h_b, *h_c;

    h_a = (float *)malloc(nBytes);
    h_b = (float *)malloc(nBytes);
    h_c = (float *)malloc(nBytes);

    cout << "Value of h_c before summing" << endl;

    for (int dx = 0; dx < 10; dx++){
        cout << "value at index " << dx << " is " << h_c[dx] << endl;
    }


    initialData(h_a, nElem);
    initialData(h_b, nElem);

    sumArrayOnHost(h_a, h_b, h_c, nElem);

    cout << "Value of h_c after summing" << endl;

    for (int dx = 0; dx < 10; dx++){
        cout << "value at index " << dx << " is " << h_c[dx] << endl;
    }
}