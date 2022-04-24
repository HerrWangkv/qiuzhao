#include <iostream>
using namespace std;
#ifndef DUMMY_H
#define DUMMY_H
class Dummy {
public:
    static int x; //不能对非常量静态数据成员进行类内初始化
    const static int y = 43;// 常量静态数据成员可以进行类内初始化
    static void f1() {//静态函数可以在类内定义
        cout << x << endl;
    }
    static void f2();//静态函数也可以在类外定义
};
#endif