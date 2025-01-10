

class Solution:
    """
    0 <= nums.length <= 3000
    -105 <= nums[i] <= 105
    """

    def three_sum(self, nums: list[int]) -> list[list[int]]:
        """
        Given an array nums of n integers, are there elements a, b, c in nums
        such that a + b + c = 0?
        Find all unique triplets in the array which gives the sum of zero.

        Notice that the solution set must not contain duplicate triplets.

        Constraints:
            0 <= nums.length <= 3000
            -10^5 <= nums[i] <= 10^5
        :param nums:
        :type nums:
        :return:
        :rtype:
        """
        length = len(nums)
        if length < 3:
            return []
        ret_set = set()
        for i in range(length):
            for j in range(1, length):
                for k in range(2, length):
                    if (
                        nums[i] + nums[j] + nums[k] == 0
                        and i != j
                        and j != k
                        and i != k
                    ):
                        tmp = tuple(sorted((nums[i], nums[j], nums[k])))
                        ret_set |= {tmp}
        return [[i, j, k] for i, j, k in ret_set]

    def three_sum_ultimate(self, nums: list[int]) -> list[list[int]]:
        length = len(nums)
        if length < 3:
            return []
        nums[:] = sorted(nums)
        ...


if __name__ == "__main__":
    nn = [-1, 0, 1, 2, -1, -4]
    s = Solution()
    print(s.three_sum(nn))
