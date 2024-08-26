from random import randrange


def dp_total(nums):
    if len(nums) == 1:
        return nums[0]
    dp = res = nums[0]

    p = 0
    q = 0
    t = []

    for i in range(1, len(nums)):
        if dp + nums[i] < nums[i]:
            p = i
            q = 1
        else:
            q += 1

        dp = max(nums[i], dp + nums[i])
        res = max(dp, res)
        if dp == res:
            t.append((p, (p + q)))
    return res, t[-1]


def make_list(start, stop, lenth):
    nums = []
    while lenth:
        nums.append(randrange(start, stop, 1))
        lenth -= 1
    return nums


nums = make_list(-10, 10, 20)
print(nums)
value, dp = dp_total(nums)
print("最大连续子序列和:", value)
print(nums[dp[0]: dp[1]])


'''
动态规划 时间复杂度O(N) 
分析 
步骤1： 
令dp[i]表示已A[i]作为结尾的连续子序列的最大和 
步骤2： 
因为dp[i]要求必须以A[i]结尾的连续序列，那么只有两种情况： 
1.这个最大连续序列只有一个元素，即以A[i]开始，以A[i]结尾 
2.这个最大和的连续序列有多个元素，即以A[p]开始（p小于i），以A[i]结尾 
对于情况1，最大和就是A[i]本身 
对于情况2，最大和是da[i-1]+A[i] 
于是得到状态转移方程： 
dp[i]=max{A[i],dp[i-1]+A[i]} 
步骤3： 
连续子序列的和为 
maxsub[n]=max{dp[i]} (1<=i<=n)
'''