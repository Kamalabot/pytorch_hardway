#include <cuda_runtime.h>
#include <iostream>

using namespace std;

int main(){
    int nElem = 1024;
    dim3 block (1024);
    dim3 grid ((nElem + block.x - 1)/block.x);
    cout << "block.x: " << block.x << "grid.x: " << grid.x << endl;

    block.x = 512;
    grid.x = (nElem + block.x - 1)/block.x;
    cout << "block.x: " << block.x << "grid.x: " << grid.x << endl;

    block.x = 256;
    grid.x = (nElem + block.x - 1)/block.x;
    cout << "block.x: " << block.x << "grid.x: " << grid.x << endl;

    block.x = 128;
    grid.x = (nElem + block.x - 1)/block.x;
    cout << "block.x: " << block.x << "grid.x: " << grid.x << endl;

}