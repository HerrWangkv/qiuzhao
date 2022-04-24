#include <iostream>
#include <memory>

int main() {
    std::unique_ptr<int> up(new int(42)); 
    std::shared_ptr<int> sp1 = std::make_shared<int>(42);
    std::shared_ptr<int> sp2 = sp1;
    // 该智能指针所指对象的引用计数， 引用计数为0时释放资源
    std::cout << sp1.use_count() << sp2.use_count() << std::endl; // 2 2
    sp2.reset(); //sp1引用计数减1, sp2置为nullptr
    std::cout << sp1.use_count() << sp2.use_count() << (sp2 == nullptr) << std::endl; // 1 0 1
    std::weak_ptr<int> wp = sp1;
    // weak_ptr没有 * 和 -> 操作符，不能直接操纵资源，wp.lock()会返回被wp引用的shared_ptr，会暂时提升引用计数
    std::cout << *wp.lock() << wp.use_count() << std::endl; // 42 2
    std::cout << wp.use_count() << std::endl; // 1
    sp1.reset();
    std::cout << wp.use_count() << std::endl; // 0
    return 0;

}