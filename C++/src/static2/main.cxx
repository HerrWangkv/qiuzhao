#include <iostream>
using namespace std;

int main() {
    //cout << a << endl;//报错，因为a是在源文件中定义的
    extern int a;
    cout << a << endl;
    //extern int b; // undefined reference to `b'
    //cout << b << endl;
    return 0;
}