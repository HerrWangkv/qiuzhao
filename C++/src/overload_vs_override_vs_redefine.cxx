#include <iostream>
using namespace std;

class Base {
public:
    virtual void f(int x) {
        cout << "Base::f(" << x << ")" << endl;
    }
    // overload
    virtual void f(int x, int y) {
        cout << "Base::f(" << x << ", " << y << ")" << endl;
    }
    void g(int x) {
        cout << "Base::g(" << x << ")" << endl;
    }
    // overload
    void g(int x, int y) {
        cout << "Base::g(" << x << ", " << y << ")" << endl;
    }
    void h(int x) {
        cout << "Base::h(" << x << ")" << endl;
    }
};

class Derived: public Base {
public:
    // override
    void f(int x) {
        cout << "Derived::f(" << x << ")" << endl;
    }
    // redefine
    // 隐藏基类中所有同名函数
    void g(int x) {
        cout << "Derived::g(" << x << ")" << endl;
    }
    // 基类中h函数被直接继承
};

int main() {
    Derived d;
    Base *bp = &d;
    Derived *dp = &d;
    bp->f(1); // 虚函数动态绑定， 输出 Derived::f(1)
    bp->f(1, 2); // 虚函数动态绑定，但派生类中未定义 f(int, int) 输出 Base::f(1, 2)
    bp->g(1); // 非虚函数， 输出 Base::g(1)
    bp->g(1, 2); // 非虚函数， 输出 Base::g(1， 2)
    bp->h(1); // 非虚函数， 输出 Base::h(1)

    dp->f(1); // f函数被重写, 输出 Derived::f(1)
    //dp->f(1, 2); // 报错， 派生类中未定义 f(int, int)
    dp->g(1); // g函数被重定义，输出 Derived::g(1)
    //dp->g(1, 2); // 报错, Derived中未定义g(int, int), 基类中定义的g(int, int)被隐藏
    dp->h(1); // Derived中未定义h(int), 但继承了基类中定义的h(int)，输出 Base::h(1)

    return 0;
}