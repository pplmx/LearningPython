# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, nt=None):
        self.val = val
        self.next = nt


def get_list_node_size(ln: ListNode) -> int:
    size = 0
    while ln:
        size += 1
        ln = ln.next
    return size


def str_no2list_node(num: str) -> ListNode:
    for c in num[::-1]:
        return ListNode(int(c), str_no2list_node(num[:-1]))


def list_node2str(ln: ListNode) -> str:
    if ln.next:
        return str(ln.val) + list_node2str(ln.next)
    return str(ln.val)


def list_node2str_no(ln: ListNode) -> str:
    return list_node2str(ln)[::-1]


class Solution:
    """
    Input: l1 = [2,4,3], l2 = [5,6,4]
    Output: [7,0,8]
    Explanation: 342 + 465 = 807.

    Input: l1 = [0], l2 = [0]
    Output: [0]

    Input: l1 = [9,9,9,9,9,9,9], l2 = [9,9,9,9]
    Output: [8,9,9,9,0,0,0,1]

    Constraints:
        - The number of nodes in each linked list is in the range [1, 100].
        - 0 <= Node.val <= 9
        - It is guaranteed that
            the list represents a number that does not have leading zeros.
    """

    def add_two_numbers(self, l1: ListNode, l2: ListNode) -> ListNode:
        dummy = cur = ListNode()
        carry = 0
        while l1 or l2 or carry:
            if l1:
                carry += l1.val
                l1 = l1.next
            if l2:
                carry += l2.val
                l2 = l2.next
            cur.next = ListNode(carry % 10)
            cur = cur.next
            carry //= 10
        return dummy.next


if __name__ == '__main__':
    num1 = str_no2list_node('969')
    num2 = str_no2list_node('345')
    s = Solution()
    total = s.add_two_numbers(num1, num2)
    print(list_node2str_no(total))
