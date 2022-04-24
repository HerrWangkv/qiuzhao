#include <iostream>
using namespace std;

void count_calls() {
    static int count = 0;
    cout << ++count << endl;
}

int main() {
    for (int i = 0; i < 5; ++i)
        count_calls(); // 输出1～5
    //cout << count << endl; // ‘count’ was not declared in this scope
    return 0;
}