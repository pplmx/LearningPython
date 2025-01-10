

class Solution:
    def remove_duplicates(self, nums: list[int]) -> int:
        """

        :param nums: It is a sorted array
        :type nums:
        :return:
        :rtype:
        """
        nums[:] = sorted(set(nums))
        return len(nums)

    def normal_rm_duplica(self, nums: list[int]) -> int:
        ret_len = 0
        if len(nums) == ret_len:
            return ret_len
        ret_len = 1
        for i in range(len(nums) - 1):
            if nums[i] != nums[i + 1]:
                nums[ret_len] = nums[i + 1]
                ret_len += 1
        nums[:] = nums[:ret_len]
        return ret_len


if __name__ == "__main__":
    s = Solution()
    arr = [1, 1, 1, 1, 2]
    length = s.normal_rm_duplica(arr)
    print(length)
    print(arr)
