# 双指针

双指针主要用于遍历数组，两个指针指向不同的元素，从而协同完成任务。



## 167.两数之和 II - 输入有序数组

[两数之和 II - 输入有序数组](https://leetcode-cn.com/problems/two-sum-ii-input-array-is-sorted)
**示例:**

```
输入: numbers = [2, 7, 11, 15], target = 9
输出: [1,2]
解释: 2 与 7 之和等于目标数 9 。因此 index1 = 1, index2 = 2 。
```

**解法：**

使用双指针，一个指针指向值较小的元素，一个指针指向值较大的元素。指向较小元素的指针从头向尾遍历，指向较大元素的指针从尾向头遍历。

- 如果两个指针指向元素的和 sum == target，那么得到要求的结果；
- 如果 sum > target，移动较大的元素，使 sum 变小一些；
- 如果 sum < target，移动较小的元素，使 sum 变大一些。



**我的答案**

```python
class Solution:
    def twoSum(self, numbers, target):
        head = 0
        tail = len(numbers) - 1
        while 1:
            num = numbers[head] + numbers[tail]
            if num == target:
                break
            elif num > target:
                tail -= 1
                continue
            elif num < target:
                head += 1
                continue
        return [head, tail]
```



## 633.两数平方和

[两数平方和](https://leetcode-cn.com/problems/sum-of-square-numbers)

给定一个非负整数 c ，你要判断是否存在两个整数 a 和 b，使得 a2 + b2 = c。

**示例:**

```
输入: 5
输出: True
解释: 1 ** 2 + 2 ** 2 = 5

输入: 3
输出: False
```

**解法**：

```
a^2 + b^2 = c
a = sqrt(c - b^2)
因a>0 所以 b的范围为(0~sqrt(c))
```

**答案**：

```python
class Solution:
    def judgeSquareSum(self, c: int) -> bool:
        head = 0
        tail = int(c ** 0.5)
        while head <= tail:
            num = head ** 2 + tail ** 2
            if c == num:
                return True
            elif num > c:
                tail -= 1
            else:
                head += 1
        return False
```



## 345.反转字符串中的元音字母

[反转字符串中的元音字母](https://leetcode-cn.com/problems/reverse-vowels-of-a-string)

编写一个函数，以字符串作为输入，反转该字符串中的元音字母。

**示例**:

```
输入: "hello"
输出: "holle"

输入: "leetcode"
输出: "leotcede"
```

**解法：**

使用双指针指向待反转的两个元音字符，一个指针从头向尾遍历，一个指针从尾到头遍历。

**答案**：

```python
class Solution:
    def reverseVowels(self, s: str) -> str:
        lis = ["a", "e", "i", "o", "u", "A", "E", "I", "O", "U"]
        list_s = list(s)
        head = 0
        tail = len(list_s) - 1
        while head < tail:
            if list_s[head] not in lis:
                head += 1
                continue
            elif list_s[tail] not in lis:
                tail -= 1
                continue
            else:
                list_s[head], list_s[tail] = list_s[tail], list_s[head]
                head += 1
                tail -= 1
                continue
        return "".join(list_s)
```



## 680. 验证回文字符串 Ⅱ

[680. 验证回文字符串 Ⅱ](https://leetcode-cn.com/problems/valid-palindrome-ii/)

给定一个非空字符串 s，最多删除一个字符。判断是否能成为回文字符串。

**示例:**

```
输入: "aba"
输出: True

输入: "abca"
输出: True
解释: 你可以删除c字符。
```

回文串问题，常用如下 Python 的解法

```
test = 'aba'
# 解一
print(test == test[::-1])  # True

# 解二
print(test == ''.join(reversed(test)))  # True
```

```
class Solution:
    def validPalindrome(self, s: str) -> bool:
        p1, p2 = 0, len(s) - 1
        while p1 < p2:
            if s[p1] != s[p2]:
                # 舍弃左字符
                a = s[p1 + 1: p2 + 1]
                # 舍弃右字符
                b = s[p1:p2]
                return a[::-1] == a or b[::-1] == b
            p1 += 1
            p2 -= 1
        return True

```



## 88. 合并两个有序数组

[88. 合并两个有序数组](https://leetcode-cn.com/problems/merge-sorted-array/)

**示例:**

```
输入:
nums1 = [1,2,3,0,0,0], m = 3
nums2 = [2,5,6],       n = 3

输出: [1,2,2,3,5,6]
```

**合并后排序**

```
class Solution(object):
    def merge(self, nums1, m, nums2, n):
        nums1[:] = sorted(nums1[:m] + nums2)
# 
```



**指针方法**

一般而言，对于有序数组可以通过 双指针法 达到O(n + m)O(n+m)的时间复杂度。

最直接的算法实现是将指针p1 置为 nums1的开头， p2为 nums2的开头，在每一步将最小值放入输出数组中。

由于 nums1 是用于输出的数组，需要将nums1中的前m个元素放在其他地方，也就需要 O(m)O(m) 的空间复杂度。

```
class Solution:
    def merge(self, nums1: [int], m: int, nums2: [int], n: int) -> None:
        """
        Do not return anything, modify nums1 in-place instead.
        """
        # 做个备份
        nums1_copy = nums1[:m]
        nums1[:] = []

        # 循环指针
        p1, p2 = 0, 0
        while p1 < m and p2 < n:
            if nums1_copy[p1] <= nums2[p2]:
                nums1.append(nums1_copy[p1])
                p1 += 1
            else:
                nums1.append(nums2[p2])
                p2 += 1

        # 剩余的添加进去
        if p1 < m:
            nums1.extend(nums1_copy[p1:])
        if p2 < n:
            nums1.extend(nums2[p2:])
```



## 141. 环形链表

[141. 环形链表](https://leetcode-cn.com/problems/linked-list-cycle/)

给定一个链表，判断链表中是否有环。

为了表示给定链表中的环，我们使用整数 pos 来表示链表尾连接到链表中的位置（索引从 0 开始）。 如果 pos 是 -1，则在该链表中没有环。

**示例**

```
输入：head = [3,2,0,-4], pos = 1
输出：true
解释：链表中有一个环，其尾部连接到第二个节点。

输入：head = [1,2], pos = 0
输出：true
解释：链表中有一个环，其尾部连接到第一个节点。

输入：head = [1], pos = -1
输出：false
解释：链表中没有环。
```

**解法**：

使用双指针，一个指针每次移动一个节点，一个指针每次移动两个节点，如果存在环，那么这两个指针一定会相遇。

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def hasCycle(self, head: ListNode) -> bool:
        fast, slow = head, head
        while fast and fast.next:
            fast = fast.next.next
            slow = slow.next
            if fast == slow:
                return True
        return False
```



## ~~524. 通过删除字母匹配到字典里最长单词~~

[524. 通过删除字母匹配到字典里最长单词](https://leetcode-cn.com/problems/longest-word-in-dictionary-through-deleting/)

给定一个字符串和一个字符串字典，找到字典里面最长的字符串，该字符串可以通过删除给定字符串的某些字符来得到。如果答案不止一个，返回长度最长且字典顺序最小的字符串。如果答案不存在，则返回空字符串。

**示例**:

```
输入:s = "abpcplea", d = ["ale","apple","monkey","plea"]
输出: "apple"

输入:s = "abpcplea", d = ["a","b","c"]
输出: "a"
```



# 贪心算法

保证每次操作都是局部最优的，并且最后得到的结果是全局最优的。

## 455. 分发饼干

[455. 分发饼干](https://leetcode-cn.com/problems/assign-cookies/)

题目描述：每个孩子都有一个满足度，每个饼干都有一个大小，只有饼干的大小大于等于一个孩子的满足度，该孩子才会获得满足。求解最多可以获得满足的孩子数量。

**示例**：

```
输入: [1,2,3], [1,1]
输出: 1
解释: 
你有三个孩子和两块小饼干，3个孩子的胃口值分别是：1,2,3。
虽然你有两块小饼干，由于他们的尺寸都是1，你只能让胃口值是1的孩子满足。
所以你应该输出1。

输入: [1,2], [1,2,3]
输出: 2
解释: 
你有两个孩子和三块小饼干，2个孩子的胃口值分别是1,2。
你拥有的饼干数量和尺寸都足以让所有孩子满足。
所以你应该输出2.
```

**解法**：

贪心问题。优先满足胃口小的小朋友的需求。

1. 对 g 和 s 升序排序

2. 初始化两个指针分别指向 g 和 s 初始位置

3. 对比 g[i] 和 s[j]

   g[i] <= s[j]：饼干满足胃口，孩子指针右移

   g[i] > s[j]：无法满足胃口

   无论满不满足胃口，都要右移饼干指针

最后返回的就是小孩的指针移动的次数

```python
class Solution:
    def findContentChildren(self, g: List[int], s: List[int]) -> int:
        g, s = sorted(g), sorted(s)
        p1, p2 = 0, 0
        while p1 < len(g) and p2 < len(s):
            if g[p1] <= s[p2]:
                p1 += 1
            p2 += 1
        return p1
```



## 435. 无重叠区间

[435. 无重叠区间](https://leetcode-cn.com/problems/non-overlapping-intervals/)

给定一个区间的集合，找到需要移除区间的最小数量，使剩余区间互不重叠。

**示例**：

```
输入: [ [1,2], [2,3], [3,4], [1,3] ]
输出: 1
解释: 移除 [1,3] 后，剩下的区间没有重叠。

输入: [ [1,2], [1,2], [1,2] ]
输出: 2
解释: 你需要移除两个 [1,2] 来使剩下的区间没有重叠。

输入: [ [1,2], [2,3] ]
输出: 0
解释: 你不需要移除任何区间，因为它们已经是无重叠的了。
```



**解法**：

按区间的结尾进行排序，每次选择结尾最小，并且和前一个区间不重叠的区间。

```python
class Solution:
    def eraseOverlapIntervals(self, intervals: List[List[int]]) -> int:
        if not intervals:
            return 0
        intervals = sorted(intervals,key=lambda x:x[-1])
        curr  = 0
        count = 1
        for i in range(1, len(intervals)):
            if intervals[curr][1] <= intervals[i][0]:
                count += 1
                curr = i
        return len(intervals)-count
```



## ~~452. 用最少数量的箭引爆气球~~

[452. 用最少数量的箭引爆气球](https://leetcode-cn.com/problems/minimum-number-of-arrows-to-burst-balloons/)





# 二分查找

二分查找也称为折半查找，每次都能将查找区间减半，这种折半特性的算法时间复杂度为 O(logN)。

**中值计算**

有两种计算中值 m 的方式：

- m = (l + h) // 2
- m = l + (h - l) // 2

l + h 可能出现加法溢出，也就是说加法的结果大于整型能够表示的范围。但是 l 和 h 都为正数，因此 h - l 不会出现加法溢出问题。所以，最好使用第二种计算法方法。



## 69. x 的平方根

[69. x 的平方根](https://leetcode-cn.com/problems/sqrtx/)

实现 `int(math.sqrt(x))` 函数。

**示例**：

```
输入: 4
输出: 2

输入: 8
输出: 2
说明: 8 的平方根是 2.82842..., 由于返回类型是整数，小数部分将被舍去。
```



**解法**：

不断缩小区间，在将该区间中位数与x做比较

以30为例，区间缩小为[0,16] -> [0,7] -> [4,7] -> [4,5] -> [5,5]

```python
class Solution:
    def mySqrt(self, x: int) -> int:
        # 为了照顾到 0 把左边界设置为 0
        left = 0
        # 为了照顾到 1 把右边界设置为 x // 2 + 1
        right = x // 2 + 1
        while left < right:
            # 注意：这里一定取右中位数，如果取左中位数，代码可能会进入死循环
            # mid = left + (right - left + 1) // 2
            mid = (left + right + 1) >> 1
            square = mid * mid

            if square > x:
                right = mid - 1
            else:
                left = mid
        # 因为一定存在，因此无需后处理
        return left
```



## 744. 寻找比目标字母大的最小字母

[744. 寻找比目标字母大的最小字母](https://leetcode-cn.com/problems/find-smallest-letter-greater-than-target/)

给定一个有序的字符数组 letters 和一个字符 target，要求找出 letters 中大于 target 的最小字符，如果找不到就返回第 1 个字符。

**示例**:

```
 输入:
letters = ["c", "f", "j"]
target = "a"
输出: "c"

输入:
letters = ["c", "f", "j"]
target = "c"
输出: "f"

输入:
letters = ["c", "f", "j"]
target = "k"
输出: "c"
```

**解法**：

二分法

```
class Solution:
    def nextGreatestLetter(self, letters: List[str], target: str) -> str:
        l, h = 0, len(letters) - 1
        while l < h:
            m = (l + h) // 2
            if letters[m] > target:
                h = m
            else:
                l = m + 1
        return letters[l] if letters[l] > target else letters[0]
```

没二分法：

```
class Solution:
    def nextGreatestLetter(self, letters: [str], target: str) -> str:
        for i, j in enumerate(letters):
            if target < j:
                return j
        else:
            return letters[0]
```



## 540. 有序数组中的单一元素

[540. 有序数组中的单一元素](https://leetcode-cn.com/problems/single-element-in-a-sorted-array/)

给定一个只包含整数的有序数组，每个元素都会出现两次，唯有一个数只会出现一次，找出这个数。

**示例**：

```
输入: [1,1,2,3,3,4,4,8,8]
输出: 2

输入: [3,3,7,7,10,11,11]
输出: 10
```

**解法**：

令 index 为 Single Element 在数组中的位置。在 index 之后，数组中原来存在的成对状态被改变。如果 m 为偶数，并且 m + 1 < index，那么 nums[m] == nums[m + 1]；m + 1 >= index，那么 nums[m] != nums[m + 1]。

从上面的规律可以知道，如果 nums[m] == nums[m + 1]，那么 index 所在的数组位置为 [m + 2, h]，此时令 l = m + 2；如果 nums[m] != nums[m + 1]，那么 index 所在的数组位置为 [l, m]，此时令 h = m。

```python
class Solution:
    def singleNonDuplicate(self, nums: [int]) -> int:
        l = 0
        h = len(nums) - 1
        while l < h:
            m = l + (h - l) // 2
            if m % 2 == 1:
                m -= 1
            if nums[m] == nums[m + 1]:
                l = m + 2
            else:
                h = m
        return nums[l]
```



## 278. 第一个错误的版本

[278. 第一个错误的版本](https://leetcode-cn.com/problems/first-bad-version/)

题目描述：给定一个元素 n 代表有 [1, 2, ..., n] 版本，在第 x 位置开始出现错误版本，导致后面的版本都错误。可以调用 isBadVersion(int x) 知道某个版本是否错误，要求找到第一个错误的版本。

示例：

```
给定 n = 5，并且 version = 4 是第一个错误的版本。

调用 isBadVersion(3) -> false
调用 isBadVersion(5) -> true
调用 isBadVersion(4) -> true

所以，4 是第一个错误的版本
```



**解法**：

如果第 m 个版本出错，则表示第一个错误的版本在 [l, m] 之间，令 h = m；

否则第一个错误的版本在 [m + 1, h] 之间，令 l = m + 1。

```python
class Solution:
    def firstBadVersion(self, n):
        l = 1
        h = n
        while l<h:
            m = l + (h-l) // 2
            if isBadVersion(m):
                h = m
            else:
                l = m + 1 
        return l
```



## ~~153. 寻找旋转排序数组中的最小值~~

[153. 寻找旋转排序数组中的最小值](https://leetcode-cn.com/problems/find-minimum-in-rotated-sorted-array/)



## ~~34. 在排序数组中查找元素的第一个和最后一个位置~~

[34. 在排序数组中查找元素的第一个和最后一个位置](https://leetcode-cn.com/problems/find-first-and-last-position-of-element-in-sorted-array/)

给定一个按照升序排列的整数数组 nums，和一个目标值 target。找出给定目标值在数组中的开始位置和结束位置。

你的算法时间复杂度必须是 O(log n) 级别。

如果数组中不存在目标值，返回 [-1, -1]。

**示例**：