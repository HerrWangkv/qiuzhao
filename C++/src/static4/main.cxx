#include "dummy.h"

int main() {
    Dummy a;
    a.f1();
    Dummy::f2();//静态函数可以直接通过类作用域访问
    return 0;
}