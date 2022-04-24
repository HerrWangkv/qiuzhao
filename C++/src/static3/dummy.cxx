#include <iostream>
#include "dummy.h"

using namespace std;

static void f1(int x) {
    cout << x << endl;
}

void f2(int x) {
    f1(x);
}