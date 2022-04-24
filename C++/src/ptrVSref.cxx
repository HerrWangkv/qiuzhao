#include <iostream>
using namespace std;

struct ListNode {
    int val;
    ListNode *next;
    ListNode(): val(0), next(nullptr) {}
    ListNode(int v): val(v), next(nullptr) {}
};

void ptr(ListNode *p) {
    p = p->next;
}

void ref(ListNode &r) {
    r = *r.next;
}

int main() {
    ListNode* head = new ListNode();
    ListNode* tail = new ListNode(1);
    head->next = tail;
    ListNode *p = head;
    ListNode& r = *head;
    ptr(p);
    cout << p->val << endl; // 0
    ref(r);
    cout << r.val << endl; // 1
    delete head;
    delete tail;
    return 0;
}