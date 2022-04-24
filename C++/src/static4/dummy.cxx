#include "dummy.h"
#include <iostream>
using namespace std;

//类外定义静态函数/数据成员时不能重复 static 关键字
int Dummy::x = 42;
void Dummy::f2() {
    cout << y << endl;
}
