#include <iostream>
#include <string>
using namespace std;

class A {
public:
    // 默认构造函数
    A() = default;
    // 一般构造函数
    A(int _x, string _str): x(_x), str(_str) {}
    // 类型转换构造函数, 阻止了从int向A的隐式转换
    explicit A(int _x) {
        x = _x;
        str = "hello";
    }
    // 拷贝构造函数
    A(const A& rhs) : x(rhs.x), str(rhs.str) {}
    // 拷贝赋值运算符
    A& operator =(const A &rhs) {
        x = rhs.x;
        str = rhs.str;
    }
    int x;
    string str;
};

int main() {
    A a1;
    cout << a1.x << " " << a1.str << endl;
    A a2(42, "abc");
    cout << a2.x << " " << a2.str << endl;
    //A a3 = 42; //  报错，隐式转换被抑制，必须通过直接初始化实现类型转换
    A a3(42);
    cout << a3.x << " " << a3.str << endl;
    A a4(a3);
    cout << a4.x << " " << a4.str << endl;
    A a5 = a3;
    cout << a5.x << " " << a5.str << endl;
    return 0;
}