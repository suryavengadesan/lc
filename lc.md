# 1. Two Sum
Easy

Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target.

You may assume that each input would have exactly one solution, and you may not use the same element twice.

You can return the answer in any order.

 

Example 1:

Input: nums = [2,7,11,15], target = 9
Output: [0,1]
Explanation: Because nums[0] + nums[1] == 9, we return [0, 1].

Example 2:

Input: nums = [3,2,4], target = 6
Output: [1,2]

Example 3:

Input: nums = [3,3], target = 6
Output: [0,1]

Follow-up: Can you come up with an algorithm that is less than O(n2) time complexity?

## Iteration Hashmap (2-pass Suboptimal)

Note: target = element1 + element2

Iterate through the list and and add target - current element to a hashmap along with the index. 

Iterate through the list again and if new element is found in the hashmap, return the current new index and the index from the hashmap. 

Make sure the current index doesn't equal the index in hashmap. 

Space: O(n)
Time: O(n)

```
def solution(nums, target):
    pairs = {}
    for index in range(len(nums)):
        num = nums[index]
        pair = target - num
        pairs[pair] = index

    for index in range(len(nums)):
        num = nums[index]
        if num in pairs and index != pairs[num]:
            return [index, pairs[num]]

    return 
```

## Iteration Hashmap (1-pass Optimal)

Same as two pass, but if complement found, return immediately. Make sure to modify hashmap for current index after checking for complement. 

Space: O(n)
Time: O(n)

```
def solution(nums, target):
    pairs = {}
    for index in range(len(nums)):
        num = nums[index]
        pair = target - num
        if num in pairs:
            return [index, pairs[num]]
        pairs[pair] = index
```

# 2. Add Two Numbers
Medium

You are given two non-empty linked lists representing two non-negative integers. The digits are stored in reverse order, and each of their nodes contains a single digit. Add the two numbers and return the sum as a linked list.

You may assume the two numbers do not contain any leading zero, except the number 0 itself.

 

Example 1:

Input: l1 = [2,4,3], l2 = [5,6,4]
Output: [7,0,8]
Explanation: 342 + 465 = 807.

Example 2:

Input: l1 = [0], l2 = [0]
Output: [0]

Example 3:

Input: l1 = [9,9,9,9,9,9,9], l2 = [9,9,9,9]
Output: [8,9,9,9,0,0,0,1]

 

Constraints:

    The number of nodes in each linked list is in the range [1, 100].
    0 <= Node.val <= 9
    It is guaranteed that the list represents a number that does not have leading zeros.



## Iteration 

Iterate through the linked list, keeping a integer carry value when summing each digit. 

Initialize the carry value to 0.

Time: O(n)
Space: O(1)


```
class ListNode:
	def __init__(self, val=0, next=None):
		self.val = val
		self.next = next

def solution(l1, l2):
	head = ListNode()
	output = head
	power = 0
	carry = 0 
	while l1 or l2:
		if l1: 
			val1 = l1.val
		else: 
			val1 = 0
		if l2: 
			val2 = l2.val
		else: 
			val2 = 0
			
		remainder = (val1 + val2 + carry)%10
		output.val = remainder
		carry = (val1 + val2 + carry)//10
		if l1: 
			l1 = l1.next
		if l2: 
			l2 = l2.next
		power += 1
		
		if l1 or l2: 
			output.next = ListNode()
			output = output.next
	if carry != 0: 
		output.next = ListNode(carry)
		
	return head
```

# 3. Longest Substring Without Repeating Characters
Medium

Given a string s, find the length of the longest substring without repeating characters.

 

Example 1:

Input: s = "abcabcbb"
Output: 3
Explanation: The answer is "abc", with the length of 3.

Example 2:

Input: s = "bbbbb"
Output: 1
Explanation: The answer is "b", with the length of 1.

Example 3:

Input: s = "pwwkew"
Output: 3
Explanation: The answer is "wke", with the length of 3.
Notice that the answer must be a substring, "pwke" is a subsequence and not a substring.


## Iteration (Brute Force)

Iterate through each letter of the string. Keep a hashmap that contains characters seen so far and a substring length counter set to 0. Keep a variable to keep track of the max size of the subtring length counter. 
The hashmap the value is the letter and the key is it's index. 

If the current letter being iterated over is found the hashmap, move current index to hashmap's index + 1, and empty the hashmap and set the substring counter back to 0.
Repeat until current index is the last index of the input string. 

Space: O(n)
Time: O(n^2)

## Two Pointer - Sliding Window (Suboptimal)

Keep a hashmap that keeps track of the total count of the characters in the current substring. The moment any character count is greater than 1, then move the left index and decrement left character from hashmap until all counts in the hashmap are <= 1. 

Time: O(2n)
Space: O(min(m, n)) s.t. m is size of alphabet

## Two Pointer - Sliding Window (Optimal)

Keep a hashmap that contains the index of the last occurance of each letter. When right pointer points to a duplicate, set left pointer to the max of the current left index and the hashmap's previous occurence index + 1. Compute the maxLength after each iteration of moving the right pointer. 

This ensures that the distance between the right pointer and left pointer reflects all valid intervals without duplicates. 

Time: O(n)
Space: O(min(n, m)) s.t. m = len(alphabet)

```
def lengthOfLongestSubstring(self, s: str) -> int:
    import math 
    l = 0
    r = 0
    hashmap = {}
    maxLength = 0
    for i in range(len(s)):
        rLetter = s[r]
        if rLetter in hashmap: 
            l = max(l, hashmap[rLetter] + 1)
            hashmap[rLetter] = r
        else: 
            hashmap[rLetter] = r 
        maxLength = max(maxLength, r - l + 1)
        r += 1
    return int(maxLength)
```

# 4. Median of Two Sorted Arrays
Hard

Given two sorted arrays nums1 and nums2 of size m and n respectively, return the median of the two sorted arrays.

The overall run time complexity should be O(log (m+n)).

 

Example 1:

Input: nums1 = [1,3], nums2 = [2]
Output: 2.00000
Explanation: merged array = [1,2,3] and median is 2.

Example 2:

Input: nums1 = [1,2], nums2 = [3,4]
Output: 2.50000
Explanation: merged array = [1,2,3,4] and median is (2 + 3) / 2 = 2.5.

 

Constraints:

    nums1.length == m
    nums2.length == n
    0 <= m <= 1000
    0 <= n <= 1000
    1 <= m + n <= 2000
    -106 <= nums1[i], nums2[i] <= 106



## Binary Search

Set the smaller array to nums1 and the larger array to nums2. Set left to 0, and right to len(nums2) * 2.

```
from math import inf
def solution(nums1, nums2):
    if len(nums1) > len(nums2):
        return self.findMedianSortedArrays(nums2, nums1)

    INT_MIN = -inf
    INT_MAX = inf

    lo, hi = 0, len(nums1)
    left_partition_size = (len(nums1) + len(nums2) + 1) // 2
    n = len(nums1) + len(nums2)

    while lo <= hi:
        p1 = (lo + hi) // 2
        p2 = left_partition_size - p1

        nums1_left = nums1[p1 - 1] if p1 > 0 else INT_MIN
        nums1_right = nums1[p1] if p1 < len(nums1) else INT_MAX
        
        nums2_left = nums2[p2 - 1] if p2 > 0 else INT_MIN
        nums2_right = nums2[p2] if p2 < len(nums2) else INT_MAX
        
        if nums1_left > nums2_right:
            hi = p1 - 1
        elif nums2_left > nums1_right:
            lo = p1 + 1
        
        else:
            if n & 1:
                return max(nums1_left, nums2_left)
            return (max(nums1_left, nums2_left) + min(nums1_right, nums2_right)) / 2
```

# 5. Longest Palindromic Substring
Medium

Given a string s, return the longest palindromic substring in s.

 

Example 1:

Input: s = "babad"
Output: "bab"
Explanation: "aba" is also a valid answer.

Example 2:

Input: s = "cbbd"
Output: "bb"


## Iteration (Brute Force)

Iterate over each substring, and check if that substring is a palindrome

## DFS with Memoization - Top-Down 

Keep a memo table with (leftPointer - int, rightPointer - int): validPalidrome - boolean, to allow to avoid rechecking if palindrome. 

## Dynamic Programming - Bottom-Up (Sub Optimal)

Time O(N^2)
Space O(N^2)

## Intelligent Iteration + DFS (Optimal)

Iterate over each index, then serach outwards from that point, keeping track of maximum value.
For each two adjacently equal values, perform the same operation. 

Time O(N^2)
Space O(1)

```
maxLength = 0
maxLeft = 0
maxRight = 0
def search(left, right):
    if left < 0 or right >= len(s): 
        return 
    if s[left] == s[right]:
        if (right - left + 1) > maxLength: 
            maxLength = right - left + 1
            maxLeft = left
            maxRight = right
        search(left - 1, right + 1)

for index in range(len(s)): 
    search(index, index)

for index in range(len(s) - 1):
    search(index, index + 1)

return s[maxLeft:maxRight + 1]
```

```
def solution(s):
    maxLength = 0
    maxLeft = 0
    maxRight = 0
    def search(left, right):
        nonlocal maxLength
        nonlocal maxLeft
        nonlocal maxRight
        if left < 0 or right >= len(s): 
            return 
        if s[left] == s[right]:
            if (right - left + 1) > maxLength: 
                maxLength = right - left + 1
                maxLeft = left
                maxRight = right
            search(left - 1, right + 1)

    for index in range(len(s)): 
        search(index, index)

    for index in range(len(s) - 1):
        search(index, index + 1)
    
    return s[maxLeft:maxRight + 1]
```
# 6. Zigzag Conversion
Medium

The string "PAYPALISHIRING" is written in a zigzag pattern on a given number of rows like this: (you may want to display this pattern in a fixed font for better legibility)

P   A   H   N
A P L S I I G
Y   I   R

And then read line by line: "PAHNAPLSIIGYIR"

Write the code that will take a string and make this conversion given a number of rows:

string convert(string s, int numRows);

 

Example 1:

Input: s = "PAYPALISHIRING", numRows = 3
Output: "PAHNAPLSIIGYIR"

Example 2:

Input: s = "PAYPALISHIRING", numRows = 4
Output: "PINALSIGYAHRPI"
Explanation:
P     I    N
A   L S  I G
Y A   H R
P     I

Example 3:

Input: s = "A", numRows = 1
Output: "A"

 

Constraints:

    1 <= s.length <= 1000
    s consists of English letters (lower-case and upper-case), ',' and '.'.
    1 <= numRows <= 1000

```
def solution(s, numRows):
    a = 0
    steps = 0
    output = ['']*numRows
    for i in s:
        output[steps] = output[steps] + i
        if numRows == 1:
            steps = 0
        else:
            c = a // (numRows - 1)   
            d = (-1)**c
            if d == 1:
                steps = steps + 1
            else: 
                steps = steps - 1
        a = a + 1
    out = ''
    for i in output:
        out = out + i
        
    return out
```

# 7. Reverse Integer
Medium

Given a signed 32-bit integer x, return x with its digits reversed. If reversing x causes the value to go outside the signed 32-bit integer range [-231, 231 - 1], then return 0.

Assume the environment does not allow you to store 64-bit integers (signed or unsigned).

 

Example 1:

Input: x = 123
Output: 321

Example 2:

Input: x = -123
Output: -321

Example 3:

Input: x = 120
Output: 21

 

Constraints:

    -231 <= x <= 231 - 1

```
```

# 11. Container With Most Water
Medium

You are given an integer array height of length n. There are n vertical lines drawn such that the two endpoints of the ith line are (i, 0) and (i, height[i]).

Find two lines that together with the x-axis form a container, such that the container contains the most water.

Return the maximum amount of water a container can store.

Notice that you may not slant the container.

 

Example 1:

Input: height = [1,8,6,2,5,4,8,3,7]
Output: 49
Explanation: The above vertical lines are represented by array [1,8,6,2,5,4,8,3,7]. In this case, the max area of water (blue section) the container can contain is 49.

Example 2:

Input: height = [1,1]
Output: 1


## Brute Force

Iterate over every pair of indices, and compute the volume those indices can store

volume = abs(index1 - index2) * min(height[index1], height[index2])

Then return the maximum volume computed so far. 

```
maxVolume = -inf
    for index in range(len(height)):
        for subindex in range(index):
            volume = abs(index - subindex) * min(height[index], height[subindex])
            maxVolume = max(maxVolume, volume)
    return maxVolume
```

## Two Pointer
Move the smaller height so far first, since moving the larger height gaurantees a decrease in total volume.
Moving the smaller height can either decrease or increase the total volume.

```
left = 0
right = len(height) - 1
maxVolume = -inf
while left < right:
    volume = (right - left) * min(height[left], height[right])
    maxVolume = max(volume, maxVolume)
    if height[left] < height[right]:
        left += 1
    else:
        right -= 1 
return maxVolume
```


```
from math import inf

def solution(height):
    maxVolume = -inf
    for index in range(len(height)):
        for subindex in range(index):
            volume = abs(index - subindex) * min(height[index], height[subindex])
            maxVolume = max(maxVolume, volume)
    return maxVolume

def solution1(height):
    left = 0
    right = len(height) - 1
    maxVolume = -inf
    while left < right:
        volume = (right - left) * min(height[left], height[right])
        maxVolume = max(volume, maxVolume)
        if height[left] < height[right]:
            left += 1
        else:
            right -= 1 
    return maxVolume

def solution2(height):
    left = 0
    right = len(height) - 1
    maxVolume = (right - left) * min(height[left], height[right])
    while left < right:
        volume = (right - left) * min(height[left], height[right])
        #move right
        right -= 1
        volumeRight = (right - left) * min(height[left], height[right])
        #move left
        left += 1
        right += 1
        volumeLeft = (right - left) * min(height[left], height[right])
        #recenter
        left -= 1

        if volumeRight > volumeLeft:
            maxVolume = max(maxVolume, volumeRight)
            right -= 1
        else:
            maxVolume = max(maxVolume, volumeLeft)
            left += 1 
    return maxVolume

def solution2(height):
    maxSoFar = -inf
    maxIndex = -inf
    maxHeight = []
    maxVolume = []
    for index in reversed(range(len(height))):
        if height[index] > maxSoFar:
            maxSoFar = height[index]
            maxIndex = index
        maxHeight.append((maxSoFar, maxIndex))
    
    for index in range(len(height)):
        volume = (maxHeight[index][1] - index) * min(maxHeight[index][0], height[index])
        maxVolume.append(volume)

    maxSoFar = -inf
    maxIndex = -inf
    maxHeight = []
    for index in range(len(height)):
        if height[index] > maxSoFar:
            maxSoFar = height[index]
            maxIndex = index
        maxHeight.append((maxSoFar, maxIndex))

    for index in range(len(height)):
        volume = abs(maxHeight[index][1] - index) * min(maxHeight[index][0], height[index])
        maxVolume.append(volume)
    return max(maxVolume)

def main():
    test = [1,8,6,2,5,4,8,3,7]
    ans = solution(test)
    print(ans)

if __name__ == '__main__':
    main()
```

# 13. Roman to Integer
Easy

Roman numerals are represented by seven different symbols: I, V, X, L, C, D and M.

Symbol       Value
I             1
V             5
X             10
L             50
C             100
D             500
M             1000

For example, 2 is written as II in Roman numeral, just two ones added together. 12 is written as XII, which is simply X + II. The number 27 is written as XXVII, which is XX + V + II.

Roman numerals are usually written largest to smallest from left to right. However, the numeral for four is not IIII. Instead, the number four is written as IV. Because the one is before the five we subtract it making four. The same principle applies to the number nine, which is written as IX. There are six instances where subtraction is used:

    I can be placed before V (5) and X (10) to make 4 and 9. 
    X can be placed before L (50) and C (100) to make 40 and 90. 
    C can be placed before D (500) and M (1000) to make 400 and 900.

Given a roman numeral, convert it to an integer.

 

Example 1:

Input: s = "III"
Output: 3
Explanation: III = 3.

Example 2:

Input: s = "LVIII"
Output: 58
Explanation: L = 50, V= 5, III = 3.

Example 3:

Input: s = "MCMXCIV"
Output: 1994
Explanation: M = 1000, CM = 900, XC = 90 and IV = 4.

 

Constraints:

    1 <= s.length <= 15
    s contains only the characters ('I', 'V', 'X', 'L', 'C', 'D', 'M').
    It is guaranteed that s is a valid roman numeral in the range [1, 3999].

# Hashmap 

We map the values of the roman numerals to their equivalent decimal form and interate through the string to iteratively keep track of the sum. We can either iterative from left to right or right to left. We can also either keep track of all valid pairs in a hashmap or manually recompute the values based on the values in the hashmap of just single roman numerals.


Time: O(1)
Space: O(1)

```
def romanToInt(self, s: str) -> int:
    val = {"I": 1, "V":5, "X": 10, "L": 50, "C": 100, "D": 500, "M":1000}

    num = val[s[-1]]

    for i in reversed(range(len(s) - 1)): 
        if val[s[i]] < val[s[i + 1]]: 
            num -= val[s[i]]
        else: 
            num += val[s[i]]

    return num 
```

# 15. 3Sum
Medium

Given an integer array nums, return all the triplets [nums[i], nums[j], nums[k]] such that i != j, i != k, and j != k, and nums[i] + nums[j] + nums[k] == 0.

Notice that the solution set must not contain duplicate triplets.

 

Example 1:

Input: nums = [-1,0,1,2,-1,-4]
Output: [[-1,-1,2],[-1,0,1]]
Explanation: 
nums[0] + nums[1] + nums[2] = (-1) + 0 + 1 = 0.
nums[1] + nums[2] + nums[4] = 0 + 1 + (-1) = 0.
nums[0] + nums[3] + nums[4] = (-1) + 2 + (-1) = 0.
The distinct triplets are [-1,0,1] and [-1,-1,2].
Notice that the order of the output and the order of the triplets does not matter.

Example 2:

Input: nums = [0,1,1]
Output: []
Explanation: The only possible triplet does not sum up to 0.

Example 3:

Input: nums = [0,0,0]
Output: [[0,0,0]]
Explanation: The only possible triplet sums up to 0.


## Two Pointer

Time: O(n^2)
Space: O(nlogn)

```
def threeSum(self, nums: List[int]) -> List[List[int]]:
    def twoSum(target, arr, triples):
        lo, hi = 1, len(arr) - 1
        while (lo < hi):
            currSum = arr[0] + arr[lo] + arr[hi]
            if currSum < 0:
                lo += 1
            elif currSum > 0:
                hi -= 1
            else:
                triples.append([arr[0], arr[lo], arr[hi]])
                lo += 1
                hi -= 1
                while lo < hi and arr[lo] == arr[lo - 1]:
                    lo += 1
    
    nums.sort()
    validTriples = []
    for index in range(len(nums)): 
        num = nums[index]
        if num > 0: 
            break 
        #else: 
        if index == 0 or nums[index - 1] != nums[index]:
            twoSum(-num, nums[index:], validTriples)
    
    return validTriples
```    

## Sorting and Hashset

First sort the array then iterate through the values, performing a modified 2 sum for each value. If the current value is greater than 0, skip and don't perform the modified 2sum, because there are no more valid three integers than sum to 0 if the smallest value is positive.

The modified 2sum tries to reach the target value which is the current value. In addition, don't use the current index as a valid index when performing 2sum. 

Modifications: 
    - Set current value to the 2sum target value
    - Skip the current index when iterating over the array to find valid pairs

In addition, in the outer loop, if the current value is the same as the previous value, skip the value. 

Time: O(N^2)
Space: O(NlogN)

```
def threeSum(self, nums: List[int]) -> List[List[int]]:
    nums.sort()
    def twoSum(nums, targetIndex):
        target = -1 * nums[targetIndex]
        pairs = {}
        allPairs = set()

        for index in range(targetIndex, len(nums)):
            if index == targetIndex:
                continue
            n = nums[index]
            val = target - n 
            if val in pairs: 
                pairs[val].append(index)
            else: 
                pairs[val] = [index]

        for index in range(targetIndex, len(nums)):
            if index == targetIndex:
                continue
            n = nums[index]
            if n in pairs and pairs[n] != [index]:
                valid = [-1 * target, n, target-n]
                valid.sort()
                valid = tuple(valid)
                allPairs.add(valid) 

        return allPairs

    allPairs = set()
    prev = None
    for index in range(len(nums)):
        curr = nums[index]
        if curr > 0: 
            break
        if curr == prev:
            continue
        prev = curr
        
        pairs = twoSum(nums, index)
        if pairs:
            for p in pairs:
                allPairs.add(p)
    allPairs = list(allPairs)
    return [list(p) for p in allPairs]
```

## No Sorting and Hashset

Time: O(N^2)
Space: O(n)
     
Examples
```
test = [-1,0,1,2,-1,-4]
test = [0, 0, 0]
```

# 16. 3Sum Closest
Medium

Given an integer array nums of length n and an integer target, find three integers in nums such that the sum is closest to target.

Return the sum of the three integers.

You may assume that each input would have exactly one solution.

 

Example 1:

Input: nums = [-1,2,1,-4], target = 1
Output: 2
Explanation: The sum that is closest to the target is 2. (-1 + 2 + 1 = 2).

Example 2:

Input: nums = [0,0,0], target = 1
Output: 0

## Two Pointer

This approach reuses the two sum solution that uses sorting and two pointers, instead of the two sum solution that uses a hashmap.

```
def threeSumClosest(self, nums: List[int], target: int) -> int:
    import math 
    minDiff = math.inf
    nums.sort()
    for i in range(len(nums)):
        n = nums[i]
        l = i + 1
        r = len(nums) - 1
        while l < r:
            currSum = n + nums[l] + nums[r]
            diff = target - currSum
            if abs(diff) < abs(minDiff): 
                minDiff = diff 
            if currSum < target: 
                l += 1 
            else: 
                r -= 1  
    return target - minDiff  
```

## Binary Search

# 17. Letter Combinations of a Phone Number
Medium

Given a string containing digits from 2-9 inclusive, return all possible letter combinations that the number could represent. Return the answer in any order.

A mapping of digits to letters (just like on the telephone buttons) is given below. Note that 1 does not map to any letters.


 

Example 1:

Input: digits = "23"
Output: ["ad","ae","af","bd","be","bf","cd","ce","cf"]
Example 2:

Input: digits = ""
Output: []
Example 3:

Input: digits = "2"
Output: ["a","b","c"]
 

Constraints:

0 <= digits.length <= 4
digits[i] is a digit in the range ['2', '9'].

## Backtracking

Search all valid mappings from digits to letters through a DFS search. 

```
def letterCombinations(self, digits: str) -> List[str]:
    mapping = {
        '2': "abc",
        '3': "def",
        '4': "ghi",
        '5': "jkl",
        '6': "mno",
        '7': "pqrs", 
        '8': "tuv",
        '9': "wxyz"}
    
    self.words = []
    def search(number, word): 
        if not number:
            if word: 
                self.words.append(word)
            return
        else:
            for letter in mapping[number[0]]: 
                search(number[1:], word + letter)
    search(digits, "")
    return self.words

```

# 19. Remove Nth Node From End of List
Medium

Given the head of a linked list, remove the nth node from the end of the list and return its head.

 

Example 1:

Input: head = [1,2,3,4,5], n = 2
Output: [1,2,3,5]

Example 2:

Input: head = [1], n = 1
Output: []

Example 3:

Input: head = [1,2], n = 1
Output: [1]

## Slow and Fast Pointers (Optimal)

Keep a fast and slow pointer. Initialize a dummy head and set the fast and slow pointers to the values. Move the fast pointer forward n + 1 nodes. Move the slow pointer and fast pointer ahead together, until fast pointer reaches the end. 

This ensures that the slow pointer will always be n-1 steps behind the fast pointer. When the fast pointer reaches the end, the slow pointer is effectively n-1 steps from the end. Change the slow pointer's next value to remove the nth node. 

Return the dummy head's next value, which would be the modified linked list. 

Time: O(n)
Space: O(1)

```
def removeNthFromEnd(self, head: Optional[ListNode], n: int) -> Optional[ListNode]:
    dummy = ListNode(0)
    dummy.next = head
    slow = dummy
    fast = dummy

    for i in range(n + 1):
        fast = fast.next 
    
    while fast: 
        slow = slow.next
        fast = fast.next 
    
    slow.next = slow.next.next 
    return dummy.next
```

# 20. Valid Parentheses
Easy

Given a string s containing just the characters '(', ')', '{', '}', '[' and ']', determine if the input string is valid.

An input string is valid if:

    Open brackets must be closed by the same type of brackets.
    Open brackets must be closed in the correct order.
    Every close bracket has a corresponding open bracket of the same type.

 

Example 1:

Input: s = "()"
Output: true

Example 2:

Input: s = "()[]{}"
Output: true

Example 3:

Input: s = "(]"
Output: false

## Stack + Hashmap (Optimal)

Go through the input string and append and pop letters from stack to keep track of valid parenthesis pairs. 

Create a hashmap as well to show the pairs of closing to open parentheses, such that if a closing parentheses is reached, we should expect the corresponding open parentheses on the stack. If not, there exists an invalid pair, so we can return False.

Keep iterating through the elements of the string, and if the stack is empty at the end that means all closing parentheses had a corresponding open parentheses, so the string is valid. 

Code 
```
def isValid(self, s: str) -> bool:
    stack = []
    mapping = {')': '(', '}': '{', ']': '['}
    for i in range(len(s)):
        currLetter = s[i]
        if currLetter in mapping:
            if stack and stack[-1] == mapping[currLetter]:
                stack.pop()
            else: 
                return False
        else: 
            stack.append(currLetter)
    return stack == []
```

Examples
```
'{{{}}}'
'{{]}}'
```

# 21. Merge Two Sorted Lists
Easy

You are given the heads of two sorted linked lists list1 and list2.

Merge the two lists in a one sorted list. The list should be made by splicing together the nodes of the first two lists.

Return the head of the merged linked list.

 

Example 1:

Input: list1 = [1,2,4], list2 = [1,3,4]
Output: [1,1,2,3,4,4]

Example 2:

Input: list1 = [], list2 = []
Output: []

Example 3:

Input: list1 = [], list2 = [0]
Output: [0]

# Two Pointer

Create a new merged list by incrementally appending the minimum of two list heads. Once at least one list is completely merged, this means the other list will contain only values greater than the completed list. This means the remaining list can be appended to the merged list to preserve the sorted ordering.  

This process can be made easier by intializing a dummy head list node for the new merged list, which can be thought of as a pointer to the newly merged string. We can dereference the dummy head to get the correct merged output.

Time: O(n)
Space: O(1)

```
def mergeTwoLists(self, list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
    head = ListNode(0)
    merged = head
    while list1 and list2:
        if list1.val < list2.val: 
            merged.next = list1
            list1 = list1.next
        else: 
            merged.next = list2
            list2 = list2.next 
        merged = merged.next
    merged.next = list1 or list2
    return head.next 
```

# 22. Generate Parentheses
Medium

Given n pairs of parentheses, write a function to generate all combinations of well-formed parentheses.

 

Example 1:

Input: n = 3
Output: ["((()))","(()())","(())()","()(())","()()()"]

Example 2:

Input: n = 1
Output: ["()"]


## Backtrack (Optimal)

Search all possible combinations by starting with the empty string and appending "(" open parentheses and ")" close paranetheses, until the number of open paranethes equal n and the number of closed parentheses equals n. 

Add the closed parantheses only if a specific condition is met, which is that the number of open parantheses is greater than close parentheses. Also only add an open parantheses if another condition  is met -- the number of open parantheses is less than n. 

Adding the '(' open, then ')' recursively while backtracking by popping off elements successfully searches the space of all possibilites.

```
def solution(n):
    ans = []
    def backtrack(l, r, s):
        if l == n and r == n:
            ans.append("".join(s))
            return 
        if l < n:
            s.append('(')
            backtrack(l + 1, r, s)
            s.pop()
        if l - r > 0:
            s.append(')')
            backtrack(l, r + 1, s)
            s.pop()     
    
    backtrack(0, 0, [])
    return ans
```

# 23. Merge k Sorted Lists
Hard

You are given an array of k linked-lists lists, each linked-list is sorted in ascending order.

Merge all the linked-lists into one sorted linked-list and return it.

 

Example 1:

Input: lists = [[1,4,5],[1,3,4],[2,6]]
Output: [1,1,2,3,4,4,5,6]
Explanation: The linked-lists are:
[
  1->4->5,
  1->3->4,
  2->6
]
merging them into one sorted list:
1->1->2->3->4->4->5->6

Example 2:

Input: lists = []
Output: []

Example 3:

Input: lists = [[]]
Output: []


## Merge Lists One at a Time (Suboptimal)

Perform the merge two lists subproblem K times for every list and the subsequent merged list. 

Time: O(k*N)

## Divide and Conquer (Optimal + Complex)

Ew. 

## Heap (Optimal + Simple)

Add all values of the python list onto a heap. Remove the smallest value of the heap, and append it to the new list. Keep going until the heap has no more values. 

Time: O(NLogK)
Time: O(K)

```
minHeap = []
newList = ListNode()
head = newList
for index in range(len(lists)):
    l = lists[index] 
    if l: 
        heapq.heappush(minHeap, (l.val, index))
        lists[index] = l.next

while minHeap:
    val, index = heapq.heappop(minHeap)
    head.next = ListNode(val)
    head = head.next
    
    l = lists[index]
    if l:
        heapq.heappush(minHeap, (l.val, index))
        lists[index] = l.next

return newList.next
```

```
import heapq 

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def solution(lists):
    minHeap = []
    newList = ListNode()
    head = newList
    for index in range(len(lists)):
        l = lists[index] 
        if l: 
            heapq.heappush(minHeap, (l.val, index))
            lists[index] = l.next

    while minHeap:
        val, index = heapq.heappop(minHeap)
        head.next = ListNode(val)
        head = head.next
        
        l = lists[index]
        if l:
            heapq.heappush(minHeap, (l.val, index))
            lists[index] = l.next

    return newList.next


def solution1(lists):
    def div(lists):
        n = len(lists)
        if n == 0:
            return None
        if n == 1:
            return lists[0]
        n = n // 2
        l = div(lists[:n])
        r = div(lists[n:])
        return merge(l, r)
        
    def merge(l1, l2):
        p = ListNode(0)
        prev = p
        while l1 and l2:
            if l1.val < l2.val:
                prev.next = l1
                l1 = l1.next
            else:
                prev.next = l2
                l2 = l2.next
            prev = prev.next
        prev.next = l1 or l2
        return p.next
        
    return div(lists)
```
# 26. Remove Duplicates from Sorted Array
Easy

Given an integer array nums sorted in non-decreasing order, remove the duplicates in-place such that each unique element appears only once. The relative order of the elements should be kept the same.

Since it is impossible to change the length of the array in some languages, you must instead have the result be placed in the first part of the array nums. More formally, if there are k elements after removing the duplicates, then the first k elements of nums should hold the final result. It does not matter what you leave beyond the first k elements.

Return k after placing the final result in the first k slots of nums.

Do not allocate extra space for another array. You must do this by modifying the input array in-place with O(1) extra memory.

Custom Judge:

The judge will test your solution with the following code:

int[] nums = [...]; // Input array
int[] expectedNums = [...]; // The expected answer with correct length

int k = removeDuplicates(nums); // Calls your implementation

assert k == expectedNums.length;
for (int i = 0; i < k; i++) {
    assert nums[i] == expectedNums[i];
}
If all assertions pass, then your solution will be accepted.

 

Example 1:

Input: nums = [1,1,2]
Output: 2, nums = [1,2,_]
Explanation: Your function should return k = 2, with the first two elements of nums being 1 and 2 respectively.
It does not matter what you leave beyond the returned k (hence they are underscores).
Example 2:

Input: nums = [0,0,1,1,1,2,2,3,3,4]
Output: 5, nums = [0,1,2,3,4,_,_,_,_,_]
Explanation: Your function should return k = 5, with the first five elements of nums being 0, 1, 2, 3, and 4 respectively.
It does not matter what you leave beyond the returned k (hence they are underscores).
 

Constraints:

1 <= nums.length <= 3 * 104
-100 <= nums[i] <= 100
nums is sorted in non-decreasing order.


# Two Pointer

```
```

# 31. Next Permutation
Medium

A permutation of an array of integers is an arrangement of its members into a sequence or linear order.

    For example, for arr = [1,2,3], the following are all the permutations of arr: [1,2,3], [1,3,2], [2, 1, 3], [2, 3, 1], [3,1,2], [3,2,1].

The next permutation of an array of integers is the next lexicographically greater permutation of its integer. More formally, if all the permutations of the array are sorted in one container according to their lexicographical order, then the next permutation of that array is the permutation that follows it in the sorted container. If such arrangement is not possible, the array must be rearranged as the lowest possible order (i.e., sorted in ascending order).

    For example, the next permutation of arr = [1,2,3] is [1,3,2].
    Similarly, the next permutation of arr = [2,3,1] is [3,1,2].
    While the next permutation of arr = [3,2,1] is [1,2,3] because [3,2,1] does not have a lexicographical larger rearrangement.

Given an array of integers nums, find the next permutation of nums.

The replacement must be in place and use only constant extra memory.

 

Example 1:

Input: nums = [1,2,3]
Output: [1,3,2]

Example 2:

Input: nums = [3,2,1]
Output: [1,2,3]

Example 3:

Input: nums = [1,1,5]
Output: [1,5,1]


## One Pass Iteration (Optimal)

Iterate from right to left of the array. For the first value lower than the value to it's right perform a key operation. 

For all the elements to the right of the current element, iterate from right to left and swap the first value greater than the current element. 

Then reverse all the elements to the right of the current element and return the array. 

```
from math import inf
def solution(nums):
    prev = -inf
    for index in reversed(range(len(nums))):
        curr = nums[index]
        if curr < prev:
            for innerIndex in reversed(range(len(nums))):
                num = nums[innerIndex]
                if num > curr: 
                    nums[index] = num
                    nums[innerIndex] = curr
                    nums[index+1:] = nums[index + 1:][::-1]
                    return nums
        prev = curr
        
    nums = nums.reverse()
    return nums

def solution(nums):
    mni = None
    mn = None
    prev = 0
    for i in reversed(range(len(nums))):
        if nums[i] < prev:
            j = i + 1
            mni = j
            mn = nums[j]
            while j < len(nums):
                if nums[j] > nums[i] and nums[j] <= mn:
                    mn = nums[j]
                    mni = j
                j += 1
            nums[i], nums[mni] = nums[mni], nums[i]
            nums[i+1:] = nums[i+1:][::-1]
            return nums
        prev = nums[i]
    return nums.reverse()
```

# 32. Longest Valid Parentheses
Hard

Given a string containing just the characters '(' and ')', return the length of the longest valid (well-formed) parentheses 
substring.

 

Example 1:

Input: s = "(()"
Output: 2
Explanation: The longest valid parentheses substring is "()".
Example 2:

Input: s = ")()())"
Output: 4
Explanation: The longest valid parentheses substring is "()()".
Example 3:

Input: s = ""
Output: 0
 

Constraints:

0 <= s.length <= 3 * 104
s[i] is '(', or ')'.

## Dynamic Programming 

Time: O(n)
Space: O(n)

## Stack 

Time: O(n)
Space: O(n)
 
```
def longestValidParentheses(self, s: str) -> int:        
    maxDist = 0

    stack = []
    stack.append(-1)
    for i in range(len(s)):
        c = s[i] 
        if c == '(':
            stack.append(i)
        elif c == ')':
            stack.pop()
            if not stack: 
                stack.append(i)
            maxDist = max(maxDist, i - stack[-1])
    return maxDist
```

## Counters + Iteration 

Perform valid parenthesis using left and right counters twice. Once in a left to right pass, then from a right to left pass. 

Time: O(n)
Space: O(1)

```
def longestValidParentheses(self, s: str) -> int:
    l = 0 
    r = 0 
    maxLength = 0 

    for i in range(len(s)):
        c = s[i]
        if c == '(': 
            l += 1
        else: 
            r += 1 

        if l == r: 
            maxLength = max(maxLength, l + r)
        elif r > l: 
            l = 0 
            r = 0 
    
    l = 0 
    r = 0 
    for i in reversed(range(len(s))):
        c = s[i]
        if c == ')': 
            l += 1
        else: 
            r += 1 

        if l == r: 
            maxLength = max(maxLength, l + r)
        elif r > l: 
            l = 0 
            r = 0 
        
    return maxLength 
```

# 33. Search in Rotated Sorted Array
Medium

There is an integer array nums sorted in ascending order (with distinct values).

Prior to being passed to your function, nums is possibly rotated at an unknown pivot index k (1 <= k < nums.length) such that the resulting array is [nums[k], nums[k+1], ..., nums[n-1], nums[0], nums[1], ..., nums[k-1]] (0-indexed). For example, [0,1,2,4,5,6,7] might be rotated at pivot index 3 and become [4,5,6,7,0,1,2].

Given the array nums after the possible rotation and an integer target, return the index of target if it is in nums, or -1 if it is not in nums.

You must write an algorithm with O(log n) runtime complexity.

 

Example 1:

Input: nums = [4,5,6,7,0,1,2], target = 0
Output: 4

Example 2:

Input: nums = [4,5,6,7,0,1,2], target = 3
Output: -1

Example 3:

Input: nums = [1], target = 0
Output: -1


## One-Pass Linear Search (Suboptimal)

This question requires a O(logN) solution; however, linear search is O(n).

## Two-Pass Binary Search (Optimal)

Perform two binary searches. 

First Binary Search: 

Determine the offset value that the array is rotated by -- this is index of smallest value in the array. 

Second Binary Search: 

Option 1 (Modular Arithmetic): 
 
Use the offset value computed from the first binary search to reorder the left and right pointers to allow for efficient second binary search of the target value. 

Option 2 (Splice Array):

Depending on the offset, only search a small subset of the remaining subarray, by splicing the array. Therefore, this binary search isn't 

Final Notes: 

Notice that the first binary search has a different stopping condition (left < right) than the second binary search (left <= right). 

Also notice that the first binary search has a different if else condition, than the second binary search. 

Finally notice that after the offset from the first binary search is determined, we only search one subarray of the original array. 

```
left = 0
right = len(nums) - 1

while left < right: 
    center = (left + right) // 2
    if nums[right] < nums[center]:
        left = center + 1
    else:
        right = center

offset = left

if nums[0] < nums[-1]:
    nums = nums
elif target == nums[0]:
    return 0
elif target > nums[0]:
    nums = nums[:offset]
    offset = 0
else:
    nums = nums[offset:]

left = 0
right = len(nums) - 1

while left <= right:    
    center = (left + right) // 2
    if nums[center] == target:
        return center + offset
    elif nums[center] > target:
        right = center - 1
    else:
        left = center + 1
        
return -1 
```

```
def solution(nums, target):
    left = 0
    right = len(nums) - 1
    while left < right: 
        mid = (left + right) // 2
        if nums[mid] <= nums[right]: 
            right = mid
        else: 
            left = mid + 1
    
    offset = left
    
    if target <= nums[-1]:
        arr = nums[offset:]
    else:
        arr = nums[:offset]
        offset = 0
    
    left = 0
    right = len(arr) - 1
    while left <= right: 
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid + offset
        if arr[mid] > target: 
            right = mid - 1
        else: 
            left = mid + 1
            
    return -1 

def main():
    '''
    test1 = [4,5,6,7,0,1,2]
    test2 = 3

    ans = solution(test1, test2)
    print(ans)

    test1 = [1, 3]
    test2 = 3

    ans = solution(test1, test2)
    print(ans)

    test1 = [5, 1, 3]
    test2 = 1

    ans = solution(test1, test2)
    print(ans)
    '''

    test1 = [4,5,6,7,8,1,2,3]
    test2 = 8
    ans = solution(test1, test2)
    print(ans)
if __name__ == '__main__':
    main()
```

# 35. Search Insert Position
Easy

Given a sorted array of distinct integers and a target value, return the index if the target is found. If not, return the index where it would be if it were inserted in order.

You must write an algorithm with O(log n) runtime complexity.

 

Example 1:

Input: nums = [1,3,5,6], target = 5
Output: 2

Example 2:

Input: nums = [1,3,5,6], target = 2
Output: 1

Example 3:

Input: nums = [1,3,5,6], target = 7
Output: 4

 

Constraints:

    1 <= nums.length <= 104
    -104 <= nums[i] <= 104
    nums contains distinct values sorted in ascending order.
    -104 <= target <= 104

## Binary Search 

perform binary search on list

can apply binary search, because list is already sorted

start at mid, then based on if curr value is greater than or less than or equal to target, search remaining values

keep searching while left < right

Time: O(logn)
Space: O(n)


```
def searchInsert(self, nums: List[int], target: int) -> int:
    l = 0
    r = len(nums) - 1
    while l < r: 
        mid = ((l + r + 1) // 2)
        midValue = nums[mid]
        
        if midValue == target: 
            return mid
        elif midValue > target: 
            r = mid - 1
                
        else:
            l = mid
            
    if nums[l] < target: 
        return l + 1
    return l
```

# 36. Valid Sudoku
Medium

Determine if a 9 x 9 Sudoku board is valid. Only the filled cells need to be validated according to the following rules:

Each row must contain the digits 1-9 without repetition.
Each column must contain the digits 1-9 without repetition.
Each of the nine 3 x 3 sub-boxes of the grid must contain the digits 1-9 without repetition.
Note:

A Sudoku board (partially filled) could be valid but is not necessarily solvable.
Only the filled cells need to be validated according to the mentioned rules.
 

Example 1:


Input: board = 
[["5","3",".",".","7",".",".",".","."]
,["6",".",".","1","9","5",".",".","."]
,[".","9","8",".",".",".",".","6","."]
,["8",".",".",".","6",".",".",".","3"]
,["4",".",".","8",".","3",".",".","1"]
,["7",".",".",".","2",".",".",".","6"]
,[".","6",".",".",".",".","2","8","."]
,[".",".",".","4","1","9",".",".","5"]
,[".",".",".",".","8",".",".","7","9"]]
Output: true
Example 2:

Input: board = 
[["8","3",".",".","7",".",".",".","."]
,["6",".",".","1","9","5",".",".","."]
,[".","9","8",".",".",".",".","6","."]
,["8",".",".",".","6",".",".",".","3"]
,["4",".",".","8",".","3",".",".","1"]
,["7",".",".",".","2",".",".",".","6"]
,[".","6",".",".",".",".","2","8","."]
,[".",".",".","4","1","9",".",".","5"]
,[".",".",".",".","8",".",".","7","9"]]
Output: false
Explanation: Same as Example 1, except with the 5 in the top left corner being modified to 8. Since there are two 8's in the top left 3x3 sub-box, it is invalid.
 

Constraints:

board.length == 9
board[i].length == 9
board[i][j] is a digit 1-9 or '.'.

## Hashmap or Set

O(n^2)
O(n^2)

```
def isValidSudoku(self, board: List[List[str]]) -> bool:
    duplicate = set()

    for i in range(9):
        duplicate.clear()
        #print(duplicate)
        for j in range(9):
            n = board[i][j]
            if n != '.': 
                if n in duplicate: 
                    return False 
                else: 
                    duplicate.add(n)

    
    for i in range(9):
        duplicate.clear()
        for j in range(9):
            n = board[j][i]
            if n != '.': 
                if n in duplicate: 
                    return False 
                else: 
                    duplicate.add(n)
    
    for i in range(9):
        duplicate.clear()
        for j in range(9):
            di = ((i % 3) * 3) + (j % 3)
            dj = ((i // 3) * 3) + (j//3)
            n = board[di][dj]
            if n != '.': 
                if n in duplicate: 
                    return False 
                else: 
                    duplicate.add(n)
    
    return True 
```

## Bit Manipulation

O(n^2)
O(n)

# 41. First Missing Positive
Hard

Given an unsorted integer array nums, return the smallest missing positive integer.

You must implement an algorithm that runs in O(n) time and uses constant extra space.

 

Example 1:

Input: nums = [1,2,0]
Output: 3
Explanation: The numbers in the range [1,2] are all in the array.
Example 2:

Input: nums = [3,4,-1,1]
Output: 2
Explanation: 1 is in the array but 2 is missing.
Example 3:

Input: nums = [7,8,9,11,12]
Output: 1
Explanation: The smallest positive integer 1 is missing.
 

Constraints:

1 <= nums.length <= 105
-231 <= nums[i] <= 231 - 1

## Hash Map Key 

The first missing positive missing number will be in the range [1, n + 1] where n is the length of the input array nums. This property holds true, because best case all the numbers are sorted from 1 to n in order, or worst case, some of the numbers are not in this range, which means that the first missing positive number would fill in this spot. 

Therefore, remove all numbers out side of the range [1, n], setting the values to 1. Use the array as a hashmap that keeps track of if some number i within the range [1, n] exists in the array. This is done by setting the number stored at the corresponding index for a value in the ranage [1, n] to a negative value. Finally, the index of the first non-negative value corresponds with the first missing positive number. 

Time: O(n)
Space: O(1)

```
def firstMissingPositive(self, nums: List[int]) -> int:
    n = len(nums)

    if 1 not in nums:
        return 1

    for i in range(n):
        if nums[i] <= 0 or nums[i] > n:
            nums[i] = 1

    for i in range(n): 
        a = abs(nums[i])
        if a == n:
            nums[0] = - abs(nums[0])
        else:
            nums[a] = - abs(nums[a])
        
    for i in range(1, n):
        if nums[i] > 0:
            return i

    if nums[0] > 0:
        return n
        
    return n + 1
```

# 42. Trapping Rain Water
Hard

Given n non-negative integers representing an elevation map where the width of each bar is 1, compute how much water it can trap after raining.

 

Example 1:

Input: height = [0,1,0,2,1,0,1,3,2,1,2,1]
Output: 6
Explanation: The above elevation map (black section) is represented by array [0,1,0,2,1,0,1,3,2,1,2,1]. In this case, 6 units of rain water (blue section) are being trapped.

Example 2:

Input: height = [4,2,0,3,2,5]
Output: 9


## Two Pointer (Optimal)

At any given point, water will be filled up to the rightMax - current height or the leftMax - current height. 

We start left and right pointers at left = 0 and right = len(heights) - 1. Then, we move the side with the lower height closer to the other side. For example, if left's height is less than right's height, we move left up by one. 

Return the total water accumulated so far.

```
left = 0
right = len(height) - 1
leftMax, rightMax, totalWater = 0, 0, 0

while left < right:
    leftMax = max(leftMax, height[left])
    rightMax = max(rightMax, height[right])
    
    if  height[left] < height[right]:
        totalWater += max(0, leftMax - height[left])
        left += 1
    else:
        totalWater += max(0, rightMax - height[right])
        right -= 1
return totalWater
```

```
def solution(height):
    left = 0
    right = len(height) - 1
    leftMax, rightMax, totalWater = 0, 0, 0
    
    while left < right:
        leftMax = max(leftMax, height[left])
        rightMax = max(rightMax, height[right])
        
        if  height[left] < height[right]:
            totalWater += max(0, leftMax - height[left])
            left += 1
        else:
            totalWater += max(0, rightMax - height[right])
            right -= 1
    return totalWater
```

# 46. Permutations
Medium
14.6K
250
company
Amazon
company
Microsoft
company
Google
Given an array nums of distinct integers, return all the possible permutations. You can return the answer in any order.

 

Example 1:

Input: nums = [1,2,3]
Output: [[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]
Example 2:

Input: nums = [0,1]
Output: [[0,1],[1,0]]
Example 3:

Input: nums = [1]
Output: [[1]]
 

Constraints:

1 <= nums.length <= 6
-10 <= nums[i] <= 10
All the integers of nums are unique.

## Backtracking 

Check all permuations that start with specific number at index i. Recursively permuate the remaining letters and backtrack, by swapping the elements in the input array nums. 

Time: O(Sum_k=1 to N (P(N,k))) ~> O(N!) < O(x) < O(N * N!)
Space: O(1)

```
def permute(self, nums: List[int]) -> List[List[int]]:
    n = len(nums)
    self.output = []
    def search(start): 
        if start == n:
            self.output.append(list(nums))
            return 
        else: 
            for i in range(start, n): 
                nums[i], nums[start] = nums[start], nums[i]
                search(start + 1)
                nums[i], nums[start] = nums[start], nums[i]
    
    search(0)
    return self.output
```

# 48. Rotate Image
Medium

You are given an n x n 2D matrix representing an image, rotate the image by 90 degrees (clockwise).

You have to rotate the image in-place, which means you have to modify the input 2D matrix directly. DO NOT allocate another 2D matrix and do the rotation.

 

Example 1:

Input: matrix = [[1,2,3],[4,5,6],[7,8,9]]
Output: [[7,4,1],[8,5,2],[9,6,3]]

Example 2:

Input: matrix = [[5,1,9,11],[2,4,8,10],[13,3,6,7],[15,14,12,16]]
Output: [[15,13,2,5],[14,3,4,1],[12,6,8,9],[16,7,10,11]]

## Transpose then Reflect (Optimal)

First transpose the image across it's top left to bottom right diagonal. This essentially is a reflection across the diagonal. 

Then reflect the matrix across the center top to center bottom. 

Together these transformations is equivalent to rotating a square matrix by 90 degrees. In particular, by reflecting values, this is achieved my in-place modifications, requiring no additional memory. 

Time: O(N) s.t. N is the number of elements in the matrix
Space: O(1)
```
def rotate(self, matrix: List[List[int]]) -> None:
    for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                if i < j: 
                    matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
        
    n = len(matrix[0]) - 1
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            if j < len(matrix[0])//2: 
                matrix[i][j], matrix[i][n - j] = matrix[i][n-j], matrix[i][j]
```
```
def solution(matrix):
    def transpose(matrix):
        for row in range(len(matrix)):
            for col in range(row, len(matrix[0])):
                tmp = matrix[row][col]
                matrix[row][col] = matrix[col][row]
                matrix[col][row] = tmp

    def reflect(matrix):
        for row in range(len(matrix)):
            for col in range(0, len(matrix[0])//2): 
                tmp = matrix[row][col]
                matrix[row][col] = matrix[row][len(matrix[0]) - col - 1]
                matrix[row][len(matrix[0]) - col - 1] = tmp

    transpose(matrix)
    reflect(matrix)
    return matrix
```

# 49. Group Anagrams
Medium

Given an array of strings strs, group the anagrams together. You can return the answer in any order.

An Anagram is a word or phrase formed by rearranging the letters of a different word or phrase, typically using all the original letters exactly once.

 

Example 1:

Input: strs = ["eat","tea","tan","ate","nat","bat"]
Output: [["bat"],["nat","tan"],["ate","eat","tea"]]

Example 2:

Input: strs = [""]
Output: [[""]]

Example 3:

Input: strs = ["a"]
Output: [["a"]]


## Sorting and Hashmap (Suboptimal)

Keep a hashmap that is used to store words with the same letters. 

Sort each word by the letters. If the new word isn't in the hashmap add it the key as the word and the value as the list of the word. If the new word exists in the hashmap, append the original word to the list. 

Return the values of the hashmap, which will be a list of lists.

Space: O(N) s.t. N is the number of words
Time: O(N*MLogM) s.t. M is the length of the longest word

```
anagrams = {}
for string in strs:
    sortedString = "".join(sorted(string))
    if sortedString not in anagrams:
        anagrams[sortedString] = []
    anagrams[sortedString].append(string)
        
return anagrams.values()
```

```
def solution(strs):
    anagrams = {}
    for string in strs:
        sortedString = "".join(sorted(string))
        if sortedString not in anagrams:
            anagrams[sortedString] = []
        anagrams[sortedString].append(string)
            
    return anagrams.values()
```

## Count 

Time: O(MN)

# 50. Pow(x, n)
Medium

Implement pow(x, n), which calculates x raised to the power n (i.e., xn).

 

Example 1:

Input: x = 2.00000, n = 10
Output: 1024.00000

Example 2:

Input: x = 2.10000, n = 3
Output: 9.26100

Example 3:

Input: x = 2.00000, n = -2
Output: 0.25000
Explanation: 2-2 = 1/22 = 1/4 = 0.25


## Iteration (Brute Force)

Time: O(n)
Space: O(1)

## Binary Search (Optimal)

Note multiplication rule: n^2x * n^2x = n^4x

Time: O(log n)
Space: O(1)

```
def solution(x, n):
    def recursive(n, x):
        if n == 0:
            return 1
        elif n % 2 == 0:
            return recursive(int(n/2), x) ** 2
        else:
            return recursive(n-1, x) * x

    def iterative(n, x):
        total = 1
        base = x
        while n > 0:
            if n % 2 == 1:
                total = total * base
            base = base * base
            n = int(n/2)
        return total

    if n < 0:
        x = 1/x
        n = -n

    return iterative(n, x)
```

# 53. Maximum Subarray
Medium

Given an integer array nums, find the contiguous subarray (containing at least one number) which has the largest sum and return its sum.

A subarray is a contiguous part of an array.

 

Example 1:

Input: nums = [-2,1,-3,4,-1,2,1,-5,4]
Output: 6
Explanation: [4,-1,2,1] has the largest sum = 6.

Example 2:

Input: nums = [1]
Output: 1

Example 3:

Input: nums = [5,4,-1,7,8]
Output: 23


## Iteration (Brute Force)

Compute prefix sums, then check difference between each pair.

Space: O(1)
Time: O(N^2)

## Kadane's Algo - Dynamic Programming (Optimal)

Intialize the DP problem, with the first value. The state variable is the cumalitive sum of the array, which gets updated as the max of the sum so far or the current value. Also keep track of the max cumalitive sum which is returned. 

Space: O(1)
Time: O(N)

```
def maxSubArray(self, nums: List[int]) -> int:
    import math
    currSum = nums[0] 
    maxSum = nums[0]
    for i in range(1, len(nums)):
        currSum = max(nums[i], currSum + nums[i])
        maxSum = max(maxSum, currSum)        
        
    return maxSum
```

# 54. Spiral Matrix
Medium

Given an m x n matrix, return all elements of the matrix in spiral order.

 

Example 1:

Input: matrix = [[1,2,3],[4,5,6],[7,8,9]]
Output: [1,2,3,6,9,8,7,4,5]

Example 2:

Input: matrix = [[1,2,3,4],[5,6,7,8],[9,10,11,12]]
Output: [1,2,3,4,8,12,11,10,9,5,6,7]



## Use coordinates and direction delta (optimal)
(1,0) -> (0, -1) -> (-1, 0) -> (0, 1) -> repeat
Notice the pattern => dx, dy = dy, -dx

Set starting coordinates to 0 -- s.t. x, y = 0, 0
Keep deltas for the x direction and y direction (dx, dy) that are swapped when meeting a stopping condition. Initialize the values to dx, dy = 1, 0.

Set previously visited elment to a non-integer character e.g. *
Swap dx, dy = -dy, dx, which changes the odirection of the search
Iterate through a total of M * N elements, while changing direction and storing visited elements to the output answer list. 

dx, dy = 1, 0
switch 
dx, dy = dy, -dx
dx, dy = 0, -1
switch
dx, dy = dy, -dx
dx, dy = -1, 0
switch
dx, dy = dy, -dx
dx, dy = 0, 1
switch
dx = dy, dy = -dx
dx = 1, dy = 0
repeat

```
colLength = len(matrix)
rowLength = len(matrix[0])

x = 0
y = 0
dx = 1
dy = 0

ordering = []
for cell in range(rowLength * colLength):
    if not 0 <= x + dx < rowLength or not 0 <= y + dy < colLength or matrix[y + dy][x + dx] == '*':
        tmp = dx
        dx = -dy
        dy = tmp

    ordering.append(matrix[y][x])
    matrix[y][x] = '*'
    x += dx
    y += dy 

return ordering
```

## Recursively search (Doen't work) since can't slice columns of python arrays
If top left, return top row
If top right, return right column
If bottom right, return bottom row
If bottom left, return return left column. 


```
def solution(matrix):
    def validStep(x, y):
        if 0 <= x < len(matrix) and 0 <= y < len(matrix[0]):
            if matrix[x][y] != '#':
                return True
        return False
        
    rows = len(matrix)
    cols = len(matrix[0])
    steps = rows * cols
    
    order = []
    x, y = 0, 0
    dx, dy = 0, 1
    for step in range(steps): 
        if not validStep(x + dx, y + dy):
            tmp = dx
            dx = dy
            dy = -tmp
        order.append(matrix[x][y])
        matrix[x][y] = '#'
        x += dx
        y += dy
    return order
```

# 55. Jump Game
Medium

You are given an integer array nums. You are initially positioned at the array's first index, and each element in the array represents your maximum jump length at that position.

Return true if you can reach the last index, or false otherwise.

 

Example 1:

Input: nums = [2,3,1,1,4]
Output: true
Explanation: Jump 1 step from index 0 to 1, then 3 steps to the last index.

Example 2:

Input: nums = [3,2,1,0,4]
Output: false
Explanation: You will always arrive at index 3 no matter what. Its maximum jump length is 0, which makes it impossible to reach the last index.

## Dynamic Programming (Suboptimal)

Keep a DP table of boolean values, setting all the values to False except for the last value (-1'th element) to True. Iterate from the right to left of the array. 

For the current index, first find the jump length nums[index]. If any of the elements to the right of length jump length equals True, set the current value to True. 

Repeat this operation till you reach the first value. If the DP table of the first value is true, then return True. 

Space: O(n)
Time: O(n)

## Intelligent Iteration (Optimal)

Keep track of the max index reachable so far, by computing maxReach = max(maxReach, currIndex + jumpLength) for each element from left to right. 

If at anypoint the previous maxReach is less than the current index return False. If array is successfully iterated without returnig False, then return True.

Space: O(1)
Time: O(n)

```
maxJump = 0
for index in range(len(nums)):
    if maxJump < index: 
        return False
    maxJump = max(maxJump, index + nums[index])
return True
```

```
def solution(nums):
    maxJump = 0
    for index in range(len(nums)):
        if maxJump < index: 
            return False
        maxJump = max(maxJump, index + nums[index])
    return True

def solutionOld(nums):
    canReach = list(nums)
    canReach[-1] = True

    for i in reversed(range(len(nums) - 1)):
        canJump = False
        canReach[i] = canJump
        reach = min(len(nums) - 1, i + nums[i])
        for j in range(i + 1, reach + 1, 1):
            if canReach[j] == True: 
                canJump = True
                break
        canReach[i] = canJump
    return canReach[0]

def main():
    test = [2, 3, 0, 1, 4]
    print(solution(test))
   
if __name__=="__main__":
    main()
```

# 56. Merge Intervals
Medium

Given an array of intervals where intervals[i] = [starti, endi], merge all overlapping intervals, and return an array of the non-overlapping intervals that cover all the intervals in the input.

 

Example 1:

Input: intervals = [[1,3],[2,6],[8,10],[15,18]]
Output: [[1,6],[8,10],[15,18]]
Explanation: Since intervals [1,3] and [2,6] overlap, merge them into [1,6].

Example 2:

Input: intervals = [[1,4],[4,5]]
Output: [[1,5]]
Explanation: Intervals [1,4] and [4,5] are considered overlapping.


## Sorting + Stack (Optimal)

Determing which intervals is easier to determin after sorting all intervals by start time. This allows us to tell if a new interval, which is gauranteed to start after the previous interval starts, may however start before the previous ends. This will encompass all merge situations.  

To implement this, first sort all elements in the list. Also, initialize a stack, which will store the final output and all intermediate states. 

If the current interval's start time is less than the last merged intervals end time, merge the two intervals. Else append the current interval directly to the merged stack.

Merge intervals by setting the end time of the last merged interval to the max of that value and the current intervals end time. 

Space: O(n)
Time: O(nlogn)

```
def merge(self, intervals: List[List[int]]) -> List[List[int]]:
    intervals.sort()
    stack = []
    for i in range(len(intervals)):
        if stack and intervals[i][0] <= stack[-1][1]:
            stack[-1][1] = max(stack[-1][1], intervals[i][1])
        else: 
            stack.append(intervals[i])
    return stack
```

## Follow-up Question

How do you add intervals and merge them for a large stream of intervals?

If you get an unsorted stream of intervals, you save it into a binary search tree, and modify the interval ends in O(logN) time during each update step. 

# 57. Insert Interval
Medium

You are given an array of non-overlapping intervals intervals where intervals[i] = [starti, endi] represent the start and the end of the ith interval and intervals is sorted in ascending order by starti. You are also given an interval newInterval = [start, end] that represents the start and end of another interval.

Insert newInterval into intervals such that intervals is still sorted in ascending order by starti and intervals still does not have any overlapping intervals (merge overlapping intervals if necessary).

Return intervals after the insertion.

 

Example 1:

Input: intervals = [[1,3],[6,9]], newInterval = [2,5]
Output: [[1,5],[6,9]]

Example 2:

Input: intervals = [[1,2],[3,5],[6,7],[8,10],[12,16]], newInterval = [4,8]
Output: [[1,2],[3,10],[12,16]]
Explanation: Because the new interval [4,8] overlaps with [3,5],[6,7],[8,10].

## Iteration + Stack (Optimal)

Iterate through each interval, if it doesn't intersect with the new interval add it to the output array. If the current interval does intersect with the new interval, update the new interval boundaries, and don't add either intervals to the output array. If the current interval starts after the new Interval ends, then add the new interval to the output array first, then the remaining intervals to the output array right afterwards.  

```
def insert(self, intervals: List[List[int]], newInterval: List[int]) -> List[List[int]]:
    stack = []

    for index in range(len(intervals)): 
        interval = intervals[index]
        if interval[1] < newInterval[0]:
            stack.append(interval)
        elif interval[0] > newInterval[1]:
            return stack + [newInterval] + intervals[index:]
        else:
            newInterval[0] = min(newInterval[0], interval[0])
            newInterval[1] = max(newInterval[1], interval[1])

    return stack + [newInterval]
```

# 62. Unique Paths
Medium

There is a robot on an m x n grid. The robot is initially located at the top-left corner (i.e., grid[0][0]). The robot tries to move to the bottom-right corner (i.e., grid[m - 1][n - 1]). The robot can only move either down or right at any point in time.

Given the two integers m and n, return the number of possible unique paths that the robot can take to reach the bottom-right corner.

The test cases are generated so that the answer will be less than or equal to 2 * 109.

 

Example 1:

Input: m = 3, n = 7
Output: 28

Example 2:

Input: m = 3, n = 2
Output: 3
Explanation: From the top-left corner, there are a total of 3 ways to reach the bottom-right corner:
1. Right -> Down -> Down
2. Down -> Down -> Right
3. Down -> Right -> Down

 

## Dynamic Programming

Initialize a DP table to all 0 values. Set the bottom row and the right most column to values of 1. Then slowly iterate from bottom left to top right, setting current element to the sum of the bottom cell and right cell values. 

Space: O(N) s.t. N is the number of elements in the list
Time: O(N)

```
def solution(m, n):
    rows = m
    cols = n
    matrix = [[0 for c in range(cols)] for r in range(rows)]
    
    for r in range(rows):
        matrix[r][cols - 1] = 1
    
    for c in range(cols):
        matrix[rows - 1][c] = 1
    
    for r in range(rows):
        r = rows - 1 - r
        for c in range(cols):
            c = cols - 1 - c 
            if matrix[r][c] == 0:
                matrix[r][c] = matrix[r+1][c] + matrix[r][c+1]
    
    return matrix[0][0]
        

```

# 63. Unique Paths II
Medium

You are given an m x n integer array grid. There is a robot initially located at the top-left corner (i.e., grid[0][0]). The robot tries to move to the bottom-right corner (i.e., grid[m-1][n-1]). The robot can only move either down or right at any point in time.

An obstacle and space are marked as 1 or 0 respectively in grid. A path that the robot takes cannot include any square that is an obstacle.

Return the number of possible unique paths that the robot can take to reach the bottom-right corner.

The testcases are generated so that the answer will be less than or equal to 2 * 109.

 

Example 1:

Input: obstacleGrid = [[0,0,0],[0,1,0],[0,0,0]]
Output: 2
Explanation: There is one obstacle in the middle of the 3x3 grid above.
There are two ways to reach the bottom-right corner:
1. Right -> Right -> Down -> Down
2. Down -> Down -> Right -> Right

Example 2:

Input: obstacleGrid = [[0,1],[0,0]]
Output: 1


## Dynamic Programming

Iterate through the grid and replace all the obstacles form values of 1 to -1. 

Initialize the bottom row and the right column to 1's. For all values to the right or top of an obstacle, don't initialize the values to 1 and leave them as 0.

Iterate from the bottom right to the top left, setting the current cell to the max of the bottom cell and the right cell. 


Space: O(N) s.t. N is the number of cells
Time: O(N)

```
def solution(obstacleGrid):
    rows = len(obstacleGrid)
    cols = len(obstacleGrid[0])

    for r in range(rows):
        for c in range(cols):
            if obstacleGrid[r][c] == 1:
                obstacleGrid[r][c] = -1
    

    for c in range(cols): 
        c = cols - 1 - c
        curr = obstacleGrid[rows - 1][c]
        if curr == -1:
            break
        else:
            obstacleGrid[rows - 1][c] = 1

    for r in range(rows): 
        r = rows - 1 - r
        curr = obstacleGrid[r][cols - 1]
        if curr == -1:
            break
        else:
            obstacleGrid[r][cols - 1] = 1
        

    for r in range(rows - 1):
        r = rows - 2 - r
        for c in range(cols - 1):
            c = cols - 2 - c
            if obstacleGrid[r][c] != -1:
                if obstacleGrid[r + 1][c] == -1 or obstacleGrid[r][c+1] == -1:
                    obstacleGrid[r][c] = max(obstacleGrid[r + 1][c], obstacleGrid[r][c+1])
                else:
                    obstacleGrid[r][c] = obstacleGrid[r + 1][c] + obstacleGrid[r][c+1]
                    
    return max(obstacleGrid[0][0], 0)
                
```

# 65. Valid Number
Hard

A valid number can be split up into these components (in order):

    A decimal number or an integer.
    (Optional) An 'e' or 'E', followed by an integer.

A decimal number can be split up into these components (in order):

    (Optional) A sign character (either '+' or '-').
    One of the following formats:
        One or more digits, followed by a dot '.'.
        One or more digits, followed by a dot '.', followed by one or more digits.
        A dot '.', followed by one or more digits.

An integer can be split up into these components (in order):

    (Optional) A sign character (either '+' or '-').
    One or more digits.

For example, all the following are valid numbers: ["2", "0089", "-0.1", "+3.14", "4.", "-.9", "2e10", "-90E3", "3e+7", "+6e-1", "53.5e93", "-123.456e789"], while the following are not valid numbers: ["abc", "1a", "1e", "e3", "99e2.5", "--6", "-+3", "95a54e53"].

Given a string s, return true if s is a valid number.

 

Example 1:

Input: s = "0"
Output: true

Example 2:

Input: s = "e"
Output: false

Example 3:

Input: s = "."
Output: false


## 

```
def solution(s):
    n = 0; e = 0; d = 0; ng = 0; i = 0
    while i < len(s):
        curr = s[i]
        if curr in '0123456789':
            n += 1
            i += 1
            continue
        if curr == "-" or curr == '+':
            if n > 0 or ng > 0 or d > 0:
                return False
            else:
                ng += 1; i += 1
                continue
        if curr == '.':
            if d > 0 or e > 0:
                return False
            else:
                d += 1; i += 1
                continue
        if curr == 'e' or curr == 'E':
            if e > 0:
                return False
            elif n == 0:
                return False
            else:
                n = 0; d = 0; ng = 0; e += 1; i += 1
                continue
        return False
        i += 1
    if n == 0:
        return False
    return True
```

# 70. Climbing Stairs
Easy

You are climbing a staircase. It takes n steps to reach the top.

Each time you can either climb 1 or 2 steps. In how many distinct ways can you climb to the top?

 

Example 1:

Input: n = 2
Output: 2
Explanation: There are two ways to climb to the top.
1. 1 step + 1 step
2. 2 steps

Example 2:

Input: n = 3
Output: 3
Explanation: There are three ways to climb to the top.
1. 1 step + 1 step + 1 step
2. 1 step + 2 steps
3. 2 steps + 1 step


## Dynamic Programming

Keep a DP table of size n. Bases cases dp[0] = 1 and dp[1] = 2, corresponding to 1 step stairs and 2 step stairs. 

Add up subproblems dp[i] = dp[i - 1] + dp[i - 2]

In essence, this question is a masked version of the Fibonacci sequence. 

Time: O(N)
Space: O(N)


```
def climbStairs(self, n: int) -> int:
    dp = [0 for i in range(n)]
            
    if n >= 1: 
        dp[0] = 1
    if n >= 2: 
        dp[1] = 2

    for i in range(2, n): 
        dp[i] = dp[i-1] + dp[i-2]

    return dp[-1]
```

# 71. Simplify Path
Medium

Given a string path, which is an absolute path (starting with a slash '/') to a file or directory in a Unix-style file system, convert it to the simplified canonical path.

In a Unix-style file system, a period '.' refers to the current directory, a double period '..' refers to the directory up a level, and any multiple consecutive slashes (i.e. '//') are treated as a single slash '/'. For this problem, any other format of periods such as '...' are treated as file/directory names.

The canonical path should have the following format:

    The path starts with a single slash '/'.
    Any two directories are separated by a single slash '/'.
    The path does not end with a trailing '/'.
    The path only contains the directories on the path from the root directory to the target file or directory (i.e., no period '.' or double period '..')

Return the simplified canonical path.

 

Example 1:

Input: path = "/home/"
Output: "/home"
Explanation: Note that there is no trailing slash after the last directory name.

Example 2:

Input: path = "/../"
Output: "/"
Explanation: Going one level up from the root directory is a no-op, as the root level is the highest level you can go.

Example 3:

Input: path = "/home//foo/"
Output: "/home/foo"
Explanation: In the canonical path, multiple consecutive slashes are replaced by a single one.


## Stack

-Scan through path and append directory names onto a stack, if reach .. then pop last element from stack
-Iterate through stack appending /stack for each element
-Check if / occurs, if so check if next letter is / or not, if it is check again
-Start by removing beginning of the path, if /'s seen remove /
-Split the list according to delimiter / , ignore empty directories
-Go in order and append list values, if . don't add, if .. pop from stack
-Output the stack elements as /e

Space: O(n)
Time: O(n)

```
def solution(path):
    e = path.split('/') 
    s = []
    for i in e:
        if i == '.' or i == '':
            continue 
        elif i == '..':
            if s:
                s.pop()
        else:
            s.append(i)
    return '/' + '/'.join(s)
```

# 73. Set Matrix Zeroes
Medium

Given an m x n integer matrix matrix, if an element is 0, set its entire row and column to 0's.

You must do it in place.

 

Example 1:

Input: matrix = [[1,1,1],[1,0,1],[1,1,1]]
Output: [[1,0,1],[0,0,0],[1,0,1]]

Example 2:

Input: matrix = [[0,1,2,0],[3,4,5,2],[1,3,1,5]]
Output: [[0,0,0,0],[0,4,5,0],[0,3,1,0]]

Follow up:

A straightforward solution using O(mn) space is probably a bad idea.
A simple improvement uses O(m + n) space, but still not the best solution.
Could you devise a constant space solution?


## Hashmap/Set

Iterate through the matrix and for each cell that equal to 0, save the row value into a rowSet and save the column value in a columnSet.

For each row in the rowSet, iterate through the row and set all the values to 0. For each column in the column set, iterate through the column and set all the values to 0. 

## Intelligent In-place Modification

Same as hashmap, but use the first row and first column as in-place replacement of the rowSet and colSet. For each cell that equals 0, set the first Row with the same column value to 0, and set the first column with the same row value to 0. 

Iterate through the first row, and for each 0, set the entire column to 0. Iterate through the first column, and for each 0, set the entire row to 0. 

```
def solution(matrix):
    rows = len(matrix)
    cols = len(matrix[0])

    zeroFirstRow = False
    zeroFirstCol = False

    for row in range(rows):
        if matrix[row][0] == 0:
            zeroFirstCol = True

    for col in range(cols):
        if matrix[0][col] == 0:
            zeroFirstRow = True

    
    for row in range(rows): 
        for col in range(cols): 
            if matrix[row][col] == 0:
                matrix[0][col] = 0
                matrix[row][0] = 0
    
    for row in range(1, rows):
        if matrix[row][0] == 0: 
            for col in range(cols): 
                matrix[row][col] = 0
        
    for col in range(1, cols): 
        if matrix[0][col] == 0:
            for row in range(rows): 
                matrix[row][col] = 0
    
    if zeroFirstRow: 
        for col in range(cols): 
            matrix[0][col] = 0

    if zeroFirstCol: 
        for row in range(rows): 
            matrix[row][0] = 0
    
    return matrix
```

# 74. Search a 2D Matrix
Medium

You are given an m x n integer matrix matrix with the following two properties:

    Each row is sorted in non-decreasing order.
    The first integer of each row is greater than the last integer of the previous row.

Given an integer target, return true if target is in matrix or false otherwise.

You must write a solution in O(log(m * n)) time complexity.

 

Example 1:

Input: matrix = [[1,3,5,7],[10,11,16,20],[23,30,34,60]], target = 3
Output: true

Example 2:

Input: matrix = [[1,3,5,7],[10,11,16,20],[23,30,34,60]], target = 13
Output: false

 

Constraints:

    m == matrix.length
    n == matrix[i].length
    1 <= m, n <= 100
    -104 <= matrix[i][j], target <= 104

## Binary Search

The matrix is essentially multiple consequtive rows stacked on top of each other. Therefore we can directly map the integers from 0 to N, where N is the total number of cells in the matrix, to a specific row column value. 

Therefore, we can compute the index within the matrix from the cell number k using the following formulas: 

- i = k // cols
- j = k % cols 

Time: O(log(M * N))
Space: O(1)

```
def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
    m = len(matrix)
    n = len(matrix[0])

    l = 0
    r = (m * n) - 1

    while l <= r: 
        mid = (l + r) // 2
        i = mid // n
        j = mid % n 
        midVal = matrix[i][j]
        if midVal == target:
            return True 
        elif midVal > target: 
            r = mid - 1
        else: 
            l = mid + 1
    return False 
```

# 75. Sort Colors
Medium

Given an array nums with n objects colored red, white, or blue, sort them in-place so that objects of the same color are adjacent, with the colors in the order red, white, and blue.

We will use the integers 0, 1, and 2 to represent the color red, white, and blue, respectively.

You must solve this problem without using the library's sort function.

 

Example 1:

Input: nums = [2,0,2,1,1,0]
Output: [0,0,1,1,2,2]
Example 2:

Input: nums = [2,0,1]
Output: [0,1,2]
 

Constraints:

n == nums.length
1 <= n <= 300
nums[i] is either 0, 1, or 2.
 

Follow up: Could you come up with a one-pass algorithm using only constant extra space?

## Two Pointer

This problem actually uses three pointers in theory. 

The crux of the problem is to move all 0's to the left end, and all the 2's to the right end. This can be possibly by keeping two pointers that signify the left boundary, where all values before that index are 0, and a right boundary where all value after that index are 1. Also keep track of a current index pointer, which is used to track which value is currently being considered. 

Swap the current value being considered, with the value at the left and right boundaries, depending on the current value. Increment and decrement all three pointers, until the current index pointer hits the right boundary. When this happens, in theory all the 0's should be at the left, the 1's in the center, and the 2's to the right. 

Time: O(n)
Space: O(1)

```
def sortColors(self, nums: List[int]) -> None:
    l = 0 
    r = len(nums) - 1
    curr = 0 
    while curr <= r: 
        if nums[curr] == 0: 
            nums[curr], nums[l] = nums[l], nums[curr]
            l += 1 
            curr += 1
        elif nums[curr] == 2: 
            nums[curr], nums[r] = nums[r], nums[curr]
            r -= 1 
        else: 
            curr += 1
    return nums 
```

# 76. Minimum Window Substring
Hard

Given two strings s and t of lengths m and n respectively, return the minimum window substring of s such that every character in t (including duplicates) is included in the window. If there is no such substring, return the empty string "".

The testcases will be generated such that the answer is unique.

A substring is a contiguous sequence of characters within the string.

 

Example 1:

Input: s = "ADOBECODEBANC", t = "ABC"
Output: "BANC"
Explanation: The minimum window substring "BANC" includes 'A', 'B', and 'C' from string t.

Example 2:

Input: s = "a", t = "a"
Output: "a"
Explanation: The entire string s is the minimum window.

Example 3:

Input: s = "a", t = "aa"
Output: ""
Explanation: Both 'a's from t must be included in the window.
Since the largest window of s only has one 'a', return empty string.

## Sliding Window (Brute Force)

Keep track of every letter within the reference string and its number of occurences, all within a hashmap.

Set a left and right pointer to 0, to allow us iterate through the search string. Create a new hashmap for the character within the window of the search string. Increment the right pointer, adding letters of the right pointer to the hashmap. If at any point, the search hashmap and token hashmap are equal, save the value of the window distance. 

If the search hashmasp has any characters that have a greater frequency than the token hashmap, increase the left pointer. Also remove the characters from the hashmap currently at the left pointer. Keep going until the search hashmap's counter values are all less than or equal to the token hashmap. 

Space: O(1) = O(m) s.t. m = len(alphabet)
Time: O(n) = O(m * n) s.t. m = len(alphabet)

```
def condition(stringFrequency, windowFrequency):
    for letter in stringFrequency:
        if stringFrequency[letter] > windowFrequency[letter]:
            return False
    return True

stringFrequency = Counter(t)

windowFrequency = Counter(s)
for letter in windowFrequency:
    windowFrequency[letter] = 0

minLeft = minRight = 0
left = right = 0
minWindow = window = inf

for index in range(len(s)):
    rightLetter = s[right]
    windowFrequency[rightLetter] += 1
    right += 1
    
    while condition(stringFrequency, windowFrequency):
        window = right - left
        if window < minWindow:
            minLeft = left
            minRight = right
            minWindow = window

        leftLetter = s[left]
        windowFrequency[leftLetter] -= 1
        left += 1

return s[minLeft: minRight]
```

## Sliding Window (Optimal)

Perform the same operations, but use integer counters instead of a dictionary for window. 

```
stringFrequency = Counter(t)
validCount = len(stringFrequency)
windowFrequency = {}
    
count = 0
minLeft = minRight = 0
left = right = 0
minWindow = window = inf

for index in range(len(s)):
    rightLetter = s[right]
    windowFrequency[rightLetter] = windowFrequency.get(rightLetter, 0) + 1
    if windowFrequency[rightLetter] == stringFrequency[rightLetter]:
        count += 1
    right += 1
    
    while count == validCount:
        window = right - left
        if window < minWindow:
            minLeft = left
            minRight = right
            minWindow = window

        leftLetter = s[left]
        windowFrequency[leftLetter] = windowFrequency.get(leftLetter, 0) - 1
        if windowFrequency[leftLetter] < stringFrequency[leftLetter]:
            count -= 1
        left += 1

return s[minLeft: minRight]
```


```
from collections import Counter
from math import inf

def solution3(s, t):
    stringFrequency = Counter(t)
    validCount = len(stringFrequency)
    windowFrequency = {}
     
    count = 0
    minLeft = minRight = 0
    left = right = 0
    minWindow = window = inf

    for index in range(len(s)):
        rightLetter = s[right]
        windowFrequency[rightLetter] = windowFrequency.get(rightLetter, 0) + 1
        if windowFrequency[rightLetter] == stringFrequency[rightLetter]:
            count += 1
        right += 1
        
        while count == validCount:
            window = right - left
            if window < minWindow:
                minLeft = left
                minRight = right
                minWindow = window

            leftLetter = s[left]
            windowFrequency[leftLetter] = windowFrequency.get(leftLetter, 0) - 1
            if windowFrequency[leftLetter] < stringFrequency[leftLetter]:
                count -= 1
            left += 1

    return s[minLeft: minRight]


def solution2(s, t):
    def condition(stringFrequency, windowFrequency):
        for letter in stringFrequency:
            if stringFrequency[letter] > windowFrequency[letter]:
                return False
        return True

    stringFrequency = Counter(t)

    windowFrequency = Counter(s)
    for letter in windowFrequency:
        windowFrequency[letter] = 0

    minLeft = minRight = 0
    left = right = 0
    minWindow = window = inf

    for index in range(len(s)):
        rightLetter = s[right]
        windowFrequency[rightLetter] += 1
        right += 1
        
        while condition(stringFrequency, windowFrequency):
            window = right - left
            if window < minWindow:
                minLeft = left
                minRight = right
                minWindow = window

            leftLetter = s[left]
            windowFrequency[leftLetter] -= 1
            left += 1

    return s[minLeft: minRight]


def solution1(s, t):
    def condition(stringFrequency, windowFrequency):
        for letter in stringFrequency:
            if stringFrequency[letter] > windowFrequency[letter]:
                return False
        return True
    
    def hashEqual(stringFrequency, windowFrequency):
        for letter in stringFrequency:
            if windowFrequency[letter] != stringFrequency[letter]:
                return False
        return True
    
    def hashGreaterThan(stringFrequency, windowFrequency):
        for letter in stringFrequency:
            if windowFrequency[letter] > stringFrequency[letter]:
                #print(letter)
                return True
        return False

    stringFrequency = Counter(t)

    windowFrequency = Counter(s)
    for letter in windowFrequency:
        windowFrequency[letter] = 0

    minLeft = 0
    minRight = 0
    left = right = 0
    minWindow = window = inf
    for index in range(len(s)):
        print(left, right)
        rightLetter = s[right]
        windowFrequency[rightLetter] += 1
        right += 1
        
        '''
        while hashGreaterThan(stringFrequency, windowFrequency) and left < right:
            #print(left, right)
            #print(stringFrequency, windowFrequency)
            leftLetter = s[left]
            windowFrequency[leftLetter] -= 1
            left += 1
        '''

        #while hashEqual(stringFrequency, windowFrequency):
        while condition(stringFrequency, windowFrequency):
            window = right - left
            #minWindow = min(minWindow, window)
            if window < minWindow:
                minLeft = left
                minRight = right
                minWindow = window

            leftLetter = s[left]
            windowFrequency[leftLetter] -= 1
            left += 1

    #print(minLeft, minRight)
    #print(s[minLeft: minRight])
    return s[minLeft: minRight]

def main():
    test1 = "ADOBECODEBANC"
    test2 = "ABC"
    ans = solution3(test1, test2)
    print(ans)

    test1 = "aaaaaaaaaaaabbbbbcdd"
    test2 = "abcdd"
    ans = solution3(test1, test2)
    print(ans)
    


if __name__ == '__main__':
    main()
```

# 78. Subsets
Medium

Given an integer array nums of unique elements, return all possible subsets (the power set).

The solution set must not contain duplicate subsets. Return the solution in any order.

 

Example 1:

Input: nums = [1,2,3]
Output: [[],[1],[2],[1,2],[3],[1,3],[2,3],[1,2,3]]

Example 2:

Input: nums = [0]
Output: [[],[0]]


## Backtracking

- Recursion add curent element or don't add currrent element/backtrack
- Add to stack and search over remainder
- Search until end of length, then add stack
- Account for duplicates

Time: O(2^n)
Space: O(logN)

```
def solution(nums):
    ans = []
    def backtrack(l, s):
        nonlocal ans
        if l == len(nums):
            ans.append(s.copy())
            return
        s.append(nums[l])
        backtrack(l + 1, s)
        s.pop()
        backtrack(l + 1, s)
    backtrack(0, [])
    return ans
```

# 79. Word Search
Medium

Given an m x n grid of characters board and a string word, return true if word exists in the grid.

The word can be constructed from letters of sequentially adjacent cells, where adjacent cells are horizontally or vertically neighboring. The same letter cell may not be used more than once.

 

Example 1:

Input: board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]], word = "ABCCED"
Output: true

Example 2:

Input: board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]], word = "SEE"
Output: true

Example 3:

Input: board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]], word = "ABCB"
Output: false


Follow up: Could you use search pruning to make your solution faster with a larger board?

## Backtrack + DFS 

Iterate through each index of the input board. Perform DFS at each index, splicing the input word through every level of the recursive search. Modify board in place with '#', instead of using a visited data structure. 


Time: O(N * 3^L) s.t. N is cells in board
Space: O(L) s.t. L is length of word

```
def exist(self, board: List[List[str]], word: str) -> bool:
    def search(i, j, word):
        if word[0] == board[i][j]:
            if len(word) == 1: 
                return True 
            board[i][j] = "#"
            neighbors = {(-1, 0), (1, 0), (0, 1), (0, -1)}
            for n in neighbors:
                di = n[0]
                dj = n[1]
                if i + di >= 0 and i + di < len(board): 
                    if j + dj >= 0 and j + dj < len(board[0]):
                        if board[i + di][j + dj] != "#": 
                            if search(i + di, j + dj, word[1:]) == True: 
                                return True 
            board[i][j] = word[0]
        return False 
        

    for i in range(len(board)):
        for j in range(len(board[0])): 
            if search(i, j, word) == True: 
                return True 
    
    return False 
```

# 84. Largest Rectangle in Histogram
Hard

Given an array of integers heights representing the histogram's bar height where the width of each bar is 1, return the area of the largest rectangle in the histogram.

 

Example 1:


Input: heights = [2,1,5,6,2,3]
Output: 10
Explanation: The above is a histogram where width of each bar is 1.
The largest rectangle is shown in the red area, which has an area = 10 units.
Example 2:


Input: heights = [2,4]
Output: 4
 

Constraints:

1 <= heights.length <= 105
0 <= heights[i] <= 104

## Monostack

Keep a monostack that corresponds to increasing height values. Specifically, store the indices of the heights, since the index information is used to compute the width of the maximum rectangle, while the corresponding height value from the index is used to compute the height. Finally compute the maximum area, of all width * height products. 

Time: O(n)
Space: O(n)

```
def largestRectangleArea(self, heights: List[int]) -> int:
        maxHeight = 0 
        heights.append(0)
        stack = []
        stack.append(-1)
        
        for i in range(len(heights)): 
            h = heights[i]

            while len(stack) > 1 and heights[stack[-1]] >= h: 
                    e = stack.pop()
                    currHeight = (i - stack[-1] - 1) * heights[e]
                    maxHeight = max(maxHeight, currHeight)

            stack.append(i)
        return maxHeight 
```

# 88. Merge Sorted Array
Easy

You are given two integer arrays nums1 and nums2, sorted in non-decreasing order, and two integers m and n, representing the number of elements in nums1 and nums2 respectively.

Merge nums1 and nums2 into a single array sorted in non-decreasing order.

The final sorted array should not be returned by the function, but instead be stored inside the array nums1. To accommodate this, nums1 has a length of m + n, where the first m elements denote the elements that should be merged, and the last n elements are set to 0 and should be ignored. nums2 has a length of n.

 

Example 1:

Input: nums1 = [1,2,3,0,0,0], m = 3, nums2 = [2,5,6], n = 3
Output: [1,2,2,3,5,6]
Explanation: The arrays we are merging are [1,2,3] and [2,5,6].
The result of the merge is [1,2,2,3,5,6] with the underlined elements coming from nums1.

Example 2:

Input: nums1 = [1], m = 1, nums2 = [], n = 0
Output: [1]
Explanation: The arrays we are merging are [1] and [].
The result of the merge is [1].

Example 3:

Input: nums1 = [0], m = 0, nums2 = [1], n = 1
Output: [1]
Explanation: The arrays we are merging are [] and [1].
The result of the merge is [1].
Note that because m = 0, there are no elements in nums1. The 0 is only there to ensure the merge result can fit in nums1.

 

Constraints:

    nums1.length == m + n
    nums2.length == n
    0 <= m, n <= 200
    1 <= m + n <= 200
    -109 <= nums1[i], nums2[j] <= 109

 

Follow up: Can you come up with an algorithm that runs in O(m + n) time?


## Iteration
- Modify in place using 2 points
- p1 = m - 1 of nums1
- p2 = n - 1 of nums2
- p = m + n - 1 in nums 1
- If p1 higher than p2, set p to p1
    - decrement p1 and p
- Else, set p to p2
    - decrement p2 and p

Time: O(N) 
Space: O(1)

```
def solution(nums1, m, nums2, n):
    p = m + n - 1
    p1 = m - 1
    p2 = n - 1
    for i in range(p + 1):
        if p2 < 0:
            break
        if p1 >= 0 and nums1[p1] > nums2[p2]:
            nums1[p] = nums1[p1]
            p -= 1
            p1 -= 1
        else:
            nums1[p] = nums2[p2]
            p -= 1
            p2 -= 1
```

# 91. Decode Ways
Medium

A message containing letters from A-Z can be encoded into numbers using the following mapping:

'A' -> "1"
'B' -> "2"  
...
'Z' -> "26"

To decode an encoded message, all the digits must be grouped then mapped back into letters using the reverse of the mapping above (there may be multiple ways). For example, "11106" can be mapped into:

    "AAJF" with the grouping (1 1 10 6)
    "KJF" with the grouping (11 10 6)

Note that the grouping (1 11 06) is invalid because "06" cannot be mapped into 'F' since "6" is different from "06".

Given a string s containing only digits, return the number of ways to decode it.

The test cases are generated so that the answer fits in a 32-bit integer.

 

Example 1:

Input: s = "12"
Output: 2
Explanation: "12" could be decoded as "AB" (1 2) or "L" (12).

Example 2:

Input: s = "226"
Output: 3
Explanation: "226" could be decoded as "BZ" (2 26), "VF" (22 6), or "BBF" (2 2 6).

Example 3:

Input: s = "06"
Output: 0
Explanation: "06" cannot be mapped to "F" because of the leading zero ("6" is different from "06").


## Recursion With Memo (Suboptimal)

## Iteration with linear space (Suboptimal)

```
dp = [0 for _ in range(len(s) + 1)]
dp[0] = 1
dp[1] = 0 if s[0] == '0' else 1

for i in range(2, len(dp)):
    if s[i - 1] != '0':
        dp[i] = dp[i - 1]
    two_digit = int(s[i - 2 : i])
    if two_digit >= 10 and two_digit <= 26:
        dp[i] += dp[i - 2]
        
return dp[-1]
```

## Dynamic Programming with constant space (Optimal)

```
if s[0] == "0":
    return 0

two_back = 1
one_back = 1
for i in range(1, len(s)):
    current = 0
    if s[i] != "0":
        current = one_back
    two_digit = int(s[i - 1: i + 1])
    if two_digit >= 10 and two_digit <= 26:
        current += two_back
    two_back = one_back
    one_back = current

return one_back
```
## OLD COMPLEX SOLUTION - DP with Linear Space

### Base case:
If an empty string is inserted, return 0.
If 0 is preceded by any number other than 1 or 2, then return 0. 
For all other conditions, the number of encodings of the first number is 1.

### Recursive Relationship:
If the previous number equals 2, the current number equals {1, ..., 6}, and the next number doesn't equal 0, 
Then the number of current decodings is the number of decodings of two previous numbers * 2.
However, if value of the two previous numbers is 2 or 1, the current number of decodings is the number of decodings of two previous numbers + the number of decoding of the previous number

If the previous number equals 1, and the next number doesn't equal 0, 
Then the number of current decodings is the number of decodings of two previous numbers * 2.
However, if value of the two previous numbers is 2 or 1, the current number of decodings is the number of decodings of two previous numbers + the number of decoding of the previous number

For all other conditions, the current number of encodings is equal to the number of decodings of the previous number

```
def solution(s):
    if not s: 
        return 0

    if s[0] == '0':
        return 0
    prev = ""
    for letter in s: 
        if letter == '0' and prev not in {'1', '2'}:
            return 0
        prev = letter

    ways = [None for i in range(len(s) + 1)]

    ways[0] = 1
    prev = '-1'
    for index in range(len(s)):
        nextLetter = ''
        curr = s[index]
        if index + 1 < len(s):
            nextLetter = s[index + 1]
        index = index + 1
        if prev == '2' and (curr in {'1', '2', '3', '4', '5', '6'}) and nextLetter != '0':
            if index > 2 and s[index - 2] in {'1', '2'}: 
                ways[index] = ways[index - 2] + ways[index - 1]
            else:
                ways[index] = ways[index - 2] * 2
        elif prev == '1' and curr != '0' and nextLetter != '0':
            if index > 2  and s[index - 2] in {'1', '2'}: 
                ways[index] = ways[index - 2] + ways[index - 1]
            else: 
                ways[index] = ways[index - 2] * 2
        else: 
            ways[index] = ways[index - 1]
        prev = curr

    return ways[-1]
```

```
'''
P: Get number of ways to decode an integer into a string

Dynamic Programming 

Save the number of decodings within an integer array, with each index corresponding to the i'th letter of the original string

a b c d
1 2 3 4 

Base Case (option a): 

If first number in the input string equals 0, then the total number decodings equals 0
If first number in the input string is any number other than zero, the total number of decodings so far equals 1

Recursive relationship (option a): 
Keep track of previous number at all times
If previous number equals 2 or 1, then current decodings equals to the number of decodings from two previous numbers * 2
If the current number is any non-zero number, the current decodings equals the number of decoding of the previous number

Recursive Relationship (option b):
If the previous number equals 2 and the current number equals {0, ..., 6}, then the number of current decodings is the number of decodings of two previous numbers * 2
If the previous number equals 1, then the current number of decodings is the number of decodings of two previous numbers * 2
For all other conditions, the current number of encodings is equal to the number of decodings of the previous number

Base case (option b):
If an empty string is inserted, return 0.
If the first number is 0, or any two numbers in sequence equal 0, then return 0. 
For all other conditions, the number of encodings of the first number is 1. 

Recursive Relationship (obtion c): 
Number of numEncodings(s[0, index]) = numEncodings(s[0, index - 1]) + x, where x depends on the value of the previous number
If the previous number does not equal to 1 or 2, the number of current encodings is equal to the previous number of encodings

Final Answer:
Base Case Option B
Recursive Relationship Option B

122
- 1 2 
    - 1 22
    - 1 2 2 
- 12 
    - 12 2
    - 1 22

121
1 2 1 

226
[x, 2, 2, 6]
[1, 1, 2, 1 * 2 + 2] 
-22

[x, 2, 1, 0, 1]
[1, 1, 1, 1, 1]

[x, 1, 2, 3, 1, 2, 3]
[1, 1, 2, 3, 3, 6, 9]

[x, 1, 1, 2, 3]
[1, 1, 2, 3, 6]

[x, 2, 3, 0]
[1, 1, 2, 2]

'''

def solution(s):
    if not s: 
        return 0

    if s[0] == '0':
        return 0
    prev = ""
    for letter in s: 
        if letter == '0' and prev not in {'1', '2'}:
            return 0
        prev = letter

    ways = [None for i in range(len(s) + 1)]

    ways[0] = 1
    prev = '-1'
    for index in range(len(s)):
        nextLetter = ''
        curr = s[index]
        if index + 1 < len(s):
            nextLetter = s[index + 1]
        index = index + 1
        if prev == '2' and (curr in {'1', '2', '3', '4', '5', '6'}) and nextLetter != '0':
            if index > 2 and s[index - 2] in {'1', '2'}: 
                ways[index] = ways[index - 2] + ways[index - 1]
            else:
                ways[index] = ways[index - 2] * 2
        elif prev == '1' and curr != '0' and nextLetter != '0':
            if index > 2  and s[index - 2] in {'1', '2'}: 
                ways[index] = ways[index - 2] + ways[index - 1]
            else: 
                ways[index] = ways[index - 2] * 2
        else: 
            ways[index] = ways[index - 1]
        prev = curr

    return ways[-1]

"""
PROBLEM: Count number of ways to decode integer back into an equivalent string format (e.g. 1 -> a, ..., 26 -> z)

[FAILED SOLVE in 20 MINUTES - 23/269 test cases ~ The previous solve got 203/269 so I got worse ~ Although my current solution is much more readable, allowing for easy fixes in the future]

Map integer back into starting
1 -> a
2 -> b ...

Count total possible valid pairings

12343123
1.2.3.4.3.1.2.3
1.23.4.3.12.3
1.23.4.3.1.23

Find all valid numbers from start to end, then search again on remaining numbers

123 -> 1, 12

Look at first and second number, then search the remaining integer 

If no valid number can be made from the remaining integer, return 0 for the whole search 

If end reached, add 1 to the count (a.k.a remaining string is empty)

Runtime: O(something) Space: O(something)

Input: s = '123123'
"""

def solutionOld(s):
    if not s: 
        return 0

    if s[0] == '0':
        return 0
    prev = ""
    for letter in s: 
        if letter == '0' and prev == '0':
            return 0
        prev = letter
    
    ways = [None for i in range(len(s) + 1)]
    ways[0] = 1
    prev = ''
    for index in range(len(s)):
        curr = s[index]
        currIndex = index + 1
        if prev == '2' and curr in {'1', '2', '3', '4', '5', '6'}:
            ways[currIndex] = ways[currIndex - 1] + 1
        elif prev == '1' and curr != '0':
            ways[currIndex] = ways[currIndex - 1] + 1
        #elif curr == '0':
        #    ways[index] = ways[index - 1]
        else: 
            ways[currIndex] = ways[currIndex - 1] 
        prev = curr  
    return ways[-1]

def solutionOlder(s):
    total = 0

    if s is None:
        return 0

    def search(remainingIntString):
        nonlocal total 
        if remainingIntString is None: 
            total += 1
            return 
        else: 
            if remainingIntString[0] != '0':
                search(remainingIntString[1:])
            if remainingIntString[0] == '1' and len(remainingIntString) > 1: 
                search(remainingIntString[2:])
            if remainingIntString[0] == '2' and len(remainingIntString) > 1 and remainingIntString[0] in {'0', '1', '2', '3', '4', '5', '6'}:
                search(remainingIntString[2:])

    search(s)

    return total

def main():
    test = "06"
    ans = solution(test)
    print(ans)

if __name__ == '__main__':
    main()

```

# 98. Validate Binary Search Tree
Medium

Given the root of a binary tree, determine if it is a valid binary search tree (BST).

A valid BST is defined as follows:

    The left subtree of a node contains only nodes with keys less than the node's key.
    The right subtree of a node contains only nodes with keys greater than the node's key.
    Both the left and right subtrees must also be binary search trees.

 

Example 1:

Input: root = [2,1,3]
Output: true

Example 2:

Input: root = [5,1,4,null,null,3,6]
Output: false
Explanation: The root node's value is 5 but its right child's value is 4.


## DFS Traversal (Preorder) (Optimal)

Check the range of the node, and modify the high and low values of the range during the recursive search. 

Space: O(N)
Time: O(N)

```
def isValidBST(self, root: Optional[TreeNode]) -> bool:
    def traverse(root, low, high):
        if not root:
            return True
        if not low < root.val < high: 
            return False
        return traverse(root.left, low, root.val) and traverse(root.right, root.val, high)
    return traverse(root, -inf, inf)
```

## DFS Traversal Recursive (Inorder) (Optimal)

For a valid BST, the each next element that is traversed must be greater than the current element of inorder traversal. 

In order: Left -> Parent -> Right

Space: O(N)
Time: O(N)

```
def isValidBST(self, root: Optional[TreeNode]) -> bool:
    import math 
    self.prev = -math.inf
    self.valid = True 

    def search(node):
        if not node: 
            return 
        else: 
            search(node.left)
            if node.val <= self.prev: 
                self.valid = False 
            self.prev = node.val
            search(node.right)

    search(root)
    return self.valid 
```

## DFS Traversal Iterative (Inorder) (Optimal)

Space: O(N)
Time: O(N)

# 100. Same Tree
Easy

Given the roots of two binary trees p and q, write a function to check if they are the same or not.

Two binary trees are considered the same if they are structurally identical, and the nodes have the same value.

 

Example 1:

Input: p = [1,2,3], q = [1,2,3]
Output: true

Example 2:

Input: p = [1,2], q = [1,null,2]
Output: false

Example 3:

Input: p = [1,2,1], q = [1,1,2]
Output: false

 

Constraints:

    The number of nodes in both trees is in the range [0, 100].
    -104 <= Node.val <= 104


## DFS Traversal Iterative 

Space: O(1)
Time: O(n)

## DFS Traversal Recursion

Checks if the two root values are empty. Check if only one of the two root values is empty. Checks if the roots aren't equal. Solve the same problem on the left and right children. If any of the children aren't equal, then the entire tree isn't equal. 

Space: O(n)
Time: O(n)

```
def solution(p, q):
    def traverse(p, q):
        if not p and not q: 
            return True
        if not p or not q: 
            return False
        if p.val != q.val: 
            return False
        return traverse(p.left, q.left) and traverse(p.right, q.right)
    
    return traverse(p, q)
            
```

# 101. Symmetric Tree
Easy

Given the root of a binary tree, check whether it is a mirror of itself (i.e., symmetric around its center).

 

Example 1:

Input: root = [1,2,2,3,4,4,3]
Output: true

Example 2:

Input: root = [1,2,2,null,3,null,3]
Output: false

 

Constraints:

    The number of nodes in the tree is in the range [1, 1000].
    -100 <= Node.val <= 100

 
Follow up: Could you solve it both recursively and iteratively?

## DFS

Check the left and right children, but instead of children,  

Time: O(N)
Space: O(N)

```
def solution(root):
    def mirror(leftRoot, rightRoot):
        if not leftRoot and not rightRoot: 
            return True
        if not leftRoot or not rightRoot: 
            return False
        return leftRoot.val == rightRoot.val and mirror(leftRoot.right, rightRoot.left) and mirror(leftRoot.left, rightRoot.right)
    return mirror(root, root)
```

## DFS with Stack


# 102. Binary Tree Level Order Traversal
Medium

Given the root of a binary tree, return the level order traversal of its nodes' values. (i.e., from left to right, level by level).

 

Example 1:

Input: root = [3,9,20,null,null,15,7]
Output: [[3],[9,20],[15,7]]

Example 2:

Input: root = [1]
Output: [[1]]

Example 3:

Input: root = []
Output: []

 

Constraints:

    The number of nodes in the tree is in the range [0, 2000].
    -1000 <= Node.val <= 1000

## BFS Traversal (Optimal)

Use a queue to perform BFS traversal. Include the level of the each node when adding them to the queue. If the current level is greater that the previous level, then start a new list within the level order list. Then append the new node to the new embedded list. 

This ensures all the nodes within the same level are within their own separate sublists. 

Space: O(n)
Time: O(n)

```
from collections import deque

def levelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
    if not root: 
        return []
    order = []
    queue = []
    
    prevLevel = -1
    currLevel = 0
    queue.insert(0, (root, currLevel))
    while queue: 
        node, currLevel = queue.pop()
        if prevLevel != currLevel: 
            order.append([])
        order[currLevel].append(node.val)
        if node.left: 
            queue.insert(0, (node.left, currLevel+1))
        if node.right: 
            queue.insert(0, (node.right, currLevel+1))
        prevLevel = currLevel
    return order
```

## DFS Traversal (Optimal)

Compute the depth of each element in the tree, by performing recursive tree traversal. Keep track of a hashmap that stores all the node values at a given level. 

Iterate through the depth values in the hashmap and append the list of nodes per level to a larger list. This list of lists will now contain the level ordering of the tree nodes.  

Time: O(n)
Space: O(n)

```
def levelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
    self.levels = {}
    def search(node, depth):
        if not node: 
            return
        else:
            if depth not in self.levels: 
                self.levels[depth] = [node.val]
            else: 
                self.levels[depth].append(node.val)
            search(node.left, depth+1)
            search(node.right, depth+1)
    search(root, 0)

    output = []
    i = 0 
    while i in self.levels: 
        output.append(self.levels[i])
        i += 1

    return output
```

# 104. Maximum Depth of Binary Tree
Easy

Given the root of a binary tree, return its maximum depth.

A binary tree's maximum depth is the number of nodes along the longest path from the root node down to the farthest leaf node.

 

Example 1:

Input: root = [3,9,20,null,null,15,7]
Output: 3

Example 2:

Input: root = [1,null,2]
Output: 2

 

Constraints:

    The number of nodes in the tree is in the range [0, 104].
    -100 <= Node.val <= 100



## DFS Traversal

Recursive check the depth of the left and right children, and setting the current depth to be the max of the values plus 1.

Time: O(n)
Space: O(n)

```
def maxDepth(self, root: Optional[TreeNode]) -> int:
    def search(node, depth):
        if not node: 
            return 0
        else: 
            leftMax = search(node.left, depth)
            rightMax = search(node.right, depth)
            return max(leftMax, rightMax) + 1
    return search(root, 0)
```


## DFS Traveral with Stack

Keep a stack to mimic recursion call stack. For each element popped of stack, add both it's children back on the stack. Keep performing this until stack is empty. 

While performs same number of steps as recursion, it uses less memory. 

Time: O(n)
Space: O(logN) amortized


# 105. Construct Binary Tree from Preorder and Inorder Traversal
Medium

Given two integer arrays preorder and inorder where preorder is the preorder traversal of a binary tree and inorder is the inorder traversal of the same tree, construct and return the binary tree.

 

Example 1:

Input: preorder = [3,9,20,15,7], inorder = [9,3,15,20,7]
Output: [3,9,20,null,null,15,7]

Example 2:

Input: preorder = [-1], inorder = [-1]
Output: [-1]


## Recursion - Build Tree

For every element in the inorder array, everything to the left of an element corresponds to everything left on the tree for the root node. Likewise, everything right of an element is everything to the right of the tree for the root node. 

Build the tree by recursing over subarrays preorder and inorder. Remove the first element of preorder on at a time, to set as new root and construct left and right child nodes based on left and right half of inorder subarries.

Popping the values off of preorder allows preorder to keep shrinking during recursive calls, and since all left children are popped first, right recursive calls have the accurate values.

Stop recursing when inorder subarries and preorder array are empty. 

Time: O(n)
Space: O(n)

```
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def traverse(preorder, inorder):
    if not inorder or not preorder: 
        return

    root = preorder.pop(0)#preorder[0]

    index = inorder.index(root)
    node = inorder[index]

    tree = TreeNode()
    tree.val = node

    leftSubtree = inorder[:index]
    rightSubtree = inorder[index + 1:]

    tree.left = traverse(preorder, leftSubtree)  
    tree.right = traverse(preorder, rightSubtree)

    return tree
        
tree = traverse(preorder, inorder)
return tree
```

# 108. Convert Sorted Array to Binary Search Tree
Easy

Given an integer array nums where the elements are sorted in ascending order, convert it to a
height-balanced
binary search tree.

 

Example 1:

Input: nums = [-10,-3,0,5,9]
Output: [0,-3,9,-10,null,5]
Explanation: [0,-10,5,null,-3,null,9] is also accepted:

Example 2:

Input: nums = [1,3]
Output: [3,1]
Explanation: [1,null,3] and [3,1] are both height-balanced BSTs.

 

Constraints:

    1 <= nums.length <= 104
    -104 <= nums[i] <= 104
    nums is sorted in a strictly increasing order.

## DFS 

Use two pointer instead of using splices of the array

Time: O(n)
Space: O(logN)

```
def sortedArrayToBST(self, nums: List[int]) -> Optional[TreeNode]:
    def build(left, right):
        if left > right: 
            return None 
        mid = (left + right) // 2
        tree = TreeNode(nums[mid])
        tree.left = build(left, mid - 1)
        tree.right = build(mid + 1, right)

        return tree 
    return build(0, len(nums) - 1)
```

# 110. Balanced Binary Tree
Easy

Given a binary tree, determine if it is
height-balanced.

 

Example 1:

Input: root = [3,9,20,null,null,15,7]
Output: true

Example 2:

Input: root = [1,2,2,3,3,null,null,4,4]
Output: false

Example 3:

Input: root = []
Output: true

 

Constraints:

    The number of nodes in the tree is in the range [0, 5000].
    -104 <= Node.val <= 104

## DFS w/ Global Tracker
recursively count up the max depths of right and left children
if left and right child depths differ by more than 1, return false 

```
def isBalanced(self, root: Optional[TreeNode]) -> bool:
    self.balanced = True
    def traverse(node):
        if not node: 
            return 0
        else: 
            leftDepth = traverse(node.left) + 1
            rightDepth = traverse(node.right) + 1
            if not -1 <= leftDepth - rightDepth <= 1: 
                self.balanced = False 
            return max(leftDepth, rightDepth)
    traverse(root)
    return self.balanced
```

## DFS w/o Global Tracker

```
def isBalanced(self, root: Optional[TreeNode]) -> bool:
    def traverse(node):
        if not node: 
            return 0
        else: 
            leftDepth = traverse(node.left)
            rightDepth = traverse(node.right)
            if leftDepth == -1 or rightDepth == -1 or not -1 <= leftDepth - rightDepth <= 1: 
                return -1
            return max(leftDepth, rightDepth) + 1
    return traverse(root) != -1  
```

## DFS with Stack

TBD

# 111. Minimum Depth of Binary Tree
Easy

Given a binary tree, find its minimum depth.

The minimum depth is the number of nodes along the shortest path from the root node down to the nearest leaf node.

Note: A leaf is a node with no children.

 

Example 1:

Input: root = [3,9,20,null,null,15,7]
Output: 2

Example 2:

Input: root = [2,null,3,null,4,null,5,null,6]
Output: 5


## DFS Traversal (Optimal)

Iterate through the tree while passing the depth. When eaching a left node, reset the minimum depth value, which is initialized to infinity. 

Time: O(N)
Space: O(1)

```
from math import inf 
def solution(root):
    if not root:
        return 0
    
    minDepth = inf
    def traverse(root, depth):
        nonlocal minDepth
        if not root: 
            return 
        else:
            if not root.left and not root.right: 
                minDepth = min(depth, minDepth)
            traverse(root.left, depth + 1)
            traverse(root.right, depth + 1)
    
    traverse(root, 1)
    return minDepth

```

# 113. Path Sum II
Medium

Given the root of a binary tree and an integer targetSum, return all root-to-leaf paths where the sum of the node values in the path equals targetSum. Each path should be returned as a list of the node values, not node references.

A root-to-leaf path is a path starting from the root and ending at any leaf node. A leaf is a node with no children.

 

Example 1:


Input: root = [5,4,8,11,null,13,4,7,2,null,null,5,1], targetSum = 22
Output: [[5,4,11,2],[5,8,4,5]]
Explanation: There are two paths whose sum equals targetSum:
5 + 4 + 11 + 2 = 22
5 + 8 + 4 + 5 = 22
Example 2:


Input: root = [1,2,3], targetSum = 5
Output: []
Example 3:

Input: root = [1,2], targetSum = 0
Output: []
 

Constraints:

The number of nodes in the tree is in the range [0, 5000].
-1000 <= Node.val <= 1000
-1000 <= targetSum <= 1000

# DFS

```
```

# 116. Populating Next Right Pointers in Each Node
Medium

You are given a perfect binary tree where all leaves are on the same level, and every parent has two children. The binary tree has the following definition:

struct Node {
  int val;
  Node *left;
  Node *right;
  Node *next;
}

Populate each next pointer to point to its next right node. If there is no next right node, the next pointer should be set to NULL.

Initially, all next pointers are set to NULL.

 

Example 1:

Input: root = [1,2,3,4,5,6,7]
Output: [1,#,2,3,#,4,5,6,7,#]
Explanation: Given the above perfect binary tree (Figure A), your function should populate each next pointer to point to its next right node, just like in Figure B. The serialized output is in level order as connected by the next pointers, with '#' signifying the end of each level.

Example 2:

Input: root = []
Output: []

 

Constraints:

    The number of nodes in the tree is in the range [0, 212 - 1].
    -1000 <= Node.val <= 1000

 

Follow-up:

    You may only use constant extra space.
    The recursive approach is fine. You may assume implicit stack space does not count as extra space for this problem.



## BFS 

Perform BFS and set all the next nodes, than go back and clear the right most left nodes

Perform BFS and when deuqueing, set root.next to the next value off the queue -- afterwards iterate down the right most side of the tree

TIME: O(N + logN)
SPACE: O(logN)

```
def solution(root):
    if root is None:
        return root

    queue = []
    queue.append(root)
    prev = None
    while queue:
        currNode = queue.pop(0)
        if prev:
            prev.next = currNode
        prev = currNode
        if currNode: 
            queue.append(currNode.left)
            queue.append(currNode.right)
    rootcopy = root
    while root: 
        root.next = None
        root = root.right

    return rootcopy


```

# 121. Best Time to Buy and Sell Stock
Easy

You are given an array prices where prices[i] is the price of a given stock on the ith day.

You want to maximize your profit by choosing a single day to buy one stock and choosing a different day in the future to sell that stock.

Return the maximum profit you can achieve from this transaction. If you cannot achieve any profit, return 0.

 

Example 1:

Input: prices = [7,1,5,3,6,4]
Output: 5
Explanation: Buy on day 2 (price = 1) and sell on day 5 (price = 6), profit = 6-1 = 5.
Note that buying on day 2 and selling on day 1 is not allowed because you must buy before you sell.

Example 2:

Input: prices = [7,6,4,3,1]
Output: 0
Explanation: In this case, no transactions are done and the max profit = 0.

 

Constraints:

    1 <= prices.length <= 105
    0 <= prices[i] <= 104

## Kadane's Algorthm (Suboptimal)

Compute an array of stock deltas. Apply kadane's algorithm to compute the maximum contiguous subarray sum from the day to day deltas. 

Time: O(n)
Space: O(n)

```
def maxProfit(self, prices: List[int]) -> int:
    deltas = []
    for i in range(len(prices) - 1):
        deltas.append(prices[i + 1] - prices[i])

    if len(deltas) == 0: 
        return 0 

    currSum = maxSum = deltas[0]
    for i in range(1, len(deltas)):
        currSum = max(deltas[i], currSum + deltas[i])
        maxSum = max(currSum, maxSum)

    return max(0, maxSum)
```


## Iteration (Optimal)

This can be seen as a primitive version of two sum. Keep track of the minimum stock value so far, and compute the maximum value by subtracting the current value by minium value so far. This ensures that all min-then-max pairs are checked, since the minimum value so far in the array is gauranteed to be checked with each index. 
        
Keep track of max difference (profit) computed so far, using the minimum value that has been tracked, and then return this value. 

Time: O(n)
Space: O(1)

Code 
```
from math import inf
def solution(prices):
    maxdiffsofar = -inf
    minsofar = inf
    for p in prices:
        minsofar = min(minsofar, p)
        maxdiffsofar = max(maxdiffsofar, p - minsofar)
    return maxdiffsofar
```

Example

```
[7, 1, 5, 3, 6, 4]
min so far
    7  1  1  1  1  1
max difference so far = max(prevmax, (curr - minsofar))
    0. 0  4  4. 5. 5

7 6 4 3 1
minsofar
7 6 4 3 1
maxdiffsofar
0 0 0 0 0
```

# 122. Best Time to Buy and Sell Stock II
Medium

You are given an integer array prices where prices[i] is the price of a given stock on the ith day.

On each day, you may decide to buy and/or sell the stock. You can only hold at most one share of the stock at any time. However, you can buy it then immediately sell it on the same day.

Find and return the maximum profit you can achieve.

 

Example 1:

Input: prices = [7,1,5,3,6,4]
Output: 7
Explanation: Buy on day 2 (price = 1) and sell on day 3 (price = 5), profit = 5-1 = 4.
Then buy on day 4 (price = 3) and sell on day 5 (price = 6), profit = 6-3 = 3.
Total profit is 4 + 3 = 7.

Example 2:

Input: prices = [1,2,3,4,5]
Output: 4
Explanation: Buy on day 1 (price = 1) and sell on day 5 (price = 5), profit = 5-1 = 4.
Total profit is 4.

Example 3:

Input: prices = [7,6,4,3,1]
Output: 0
Explanation: There is no way to make a positive profit, so we never buy the stock to achieve the maximum profit of 0.

 

Constraints:

    1 <= prices.length <= 3 * 104
    0 <= prices[i] <= 104



## Iteration

Time: O(n)
Space: O(1)

```
def solution(prices):
    if not prices:
        return 0
    tp = 0
    prev = prices[0]
    for day in prices:
        profits = day - prev
        if profits > 0:
            tp += profits
        prev = day
    return tp 
```

# 124. Binary Tree Maximum Path Sum
Hard

A path in a binary tree is a sequence of nodes where each pair of adjacent nodes in the sequence has an edge connecting them. A node can only appear in the sequence at most once. Note that the path does not need to pass through the root.

The path sum of a path is the sum of the node's values in the path.

Given the root of a binary tree, return the maximum path sum of any non-empty path.

 

Example 1:

Input: root = [1,2,3]
Output: 6
Explanation: The optimal path is 2 -> 1 -> 3 with a path sum of 2 + 1 + 3 = 6.

Example 2:

Input: root = [-10,9,20,null,null,15,7]
Output: 42
Explanation: The optimal path is 15 -> 20 -> 7 with a path sum of 15 + 20 + 7 = 42.

 

Constraints:

    The number of nodes in the tree is in the range [1, 3 * 104].
    -1000 <= Node.val <= 1000



## DFS Traversal 

Keep track of a global maximum sum. 

Perform post order traversal throughout the tree. Keep track of the maximum left path length and maximum right path length. 

Keep track of maximum total sum in each recursive call by setting the maximum sum to be equal to the maximum of the left path + root node value, right path  + root node value, left path + right + root node value, root node value, and the current maximum sum so far.  

If the current root node doesn't exist return 0, and if it does exist return the maximum of the left sum path, right path, or the sum of both left and right paths with the root node. 

Return the global maximum sum computed so far. 

Time: O(n) 
Space: O(1)

```
from math import inf
def solution(root):
    maxSum = -inf
    def traverse(root):
        nonlocal maxSum
        if not root:
            return 0
        leftPath = traverse(root.left)
        rightPath = traverse(root.right)
        maxSum = max(maxSum, leftPath + rightPath + root.val, leftPath + root.val, rightPath + root.val, root.val)
        return max(leftPath + root.val, rightPath +root.val, root.val)
    traverse(root)
    return maxSum
```

# 125. Valid Palindrome
Easy

A phrase is a palindrome if, after converting all uppercase letters into lowercase letters and removing all non-alphanumeric characters, it reads the same forward and backward. Alphanumeric characters include letters and numbers.

Given a string s, return true if it is a palindrome, or false otherwise.

 

Example 1:

Input: s = "A man, a plan, a canal: Panama"
Output: true
Explanation: "amanaplanacanalpanama" is a palindrome.

Example 2:

Input: s = "race a car"
Output: false
Explanation: "raceacar" is not a palindrome.

Example 3:

Input: s = " "
Output: true
Explanation: s is an empty string "" after removing non-alphanumeric characters.
Since an empty string reads the same forward and backward, it is a palindrome.

 

Constraints:

    1 <= s.length <= 2 * 105
    s consists only of printable ASCII characters.

## Brute Force (Suboptimal)

Create new string with only alphanumer letters, then comare it with it's reverse. 

Time: O(n)
Space: O(n)

## Two Pointer (Optimal)

Initialized a left and right pointer starting at either end of the string. If the letter at the left pointer isn't alphanumeric increment the left pointer in a while loop. If the letter at the right pointer isn't alphanumeric decrement the right pointer in a while loop. 

If the current letter at the left and right pointers are the same value when lowercases, move to the next step by decrementing the right pointer and incrementing the left pointer. If the current letter isn't the same, then return False.

Keep performing the above steps while the left pointer is less than the the right pointer. If loop successfully terminates, then all the strings have been processed successfully and it is a palindrome. 

This approach saves memory, but not needing to create a separate new string. 

Time: O(n) 
Space: O(1)

```
def solution(s):
    left = 0 
    right = len(s) - 1
    while left < right:
        while left < right and not s[left].isalnum():
            left += 1
        while left < right and not s[right].isalnum():
            right -= 1
        if s[left].lower() == s[right].lower():
            left += 1
            right -= 1
        else:
            return False
    return True
```

# 128. Longest Consecutive Sequence
Medium

Given an unsorted array of integers nums, return the length of the longest consecutive elements sequence.

You must write an algorithm that runs in O(n) time.

 

Example 1:

Input: nums = [100,4,200,1,3,2]
Output: 4
Explanation: The longest consecutive elements sequence is [1, 2, 3, 4]. Therefore its length is 4.

Example 2:

Input: nums = [0,3,7,2,5,8,4,6,0,1]
Output: 9


## Brute Force (Suboptimal)

For each element, check if the element + 1 is found in the list. Repeatedly perform this search until element + 1, not found in the list. Then move onto the next element in the list, until all elements have been exhaustively searched. 

Return the longest search length, which corresponds to the longest consequitive sequence

Time: O(n^3)
Space: O(1)

## Iteration with Set (Optimal)

Convert the input nums array into a set. Then iterate through the elements in the set. The set allows out to perform existence checks in O(1) time. 

If n-1 is in the set, skip the number -- this ensures that only elements that starts a sequence will be checked. 

If n-1 isn't in the set, then count upwards by setting n = n + 1. If the incremented n is in the set, add to length of consequtive sequence. 

Return the maximum such subseuquence length. 

Time: O(n)
Space: O(1)

```
def longestConsecutive(self, nums: List[int]) -> int:
    maxCount = 0
    nums = set(nums)
    for n in nums: 
        if n - 1 not in nums: 
            count = 0
            while n in nums: 
                count += 1
                n += 1 
            maxCount = max(maxCount, count)
    return maxCount
```

```
def solution(nums):
    nums = set(nums)
    maxlength = 0
    for num in nums: 
        length = 1
        if num - 1 not in nums: 
            while num + 1 in nums: 
                length += 1
                num = num + 1
        maxlength = max(maxlength, length)
    return maxlength
```

# 129. Sum Root to Leaf Numbers
Medium

You are given the root of a binary tree containing digits from 0 to 9 only.

Each root-to-leaf path in the tree represents a number.

    For example, the root-to-leaf path 1 -> 2 -> 3 represents the number 123.

Return the total sum of all root-to-leaf numbers. Test cases are generated so that the answer will fit in a 32-bit integer.

A leaf node is a node with no children.

 

Example 1:

Input: root = [1,2,3]
Output: 25
Explanation:
The root-to-leaf path 1->2 represents the number 12.
The root-to-leaf path 1->3 represents the number 13.
Therefore, sum = 12 + 13 = 25.

Example 2:

Input: root = [4,9,0,5,1]
Output: 1026
Explanation:
The root-to-leaf path 4->9->5 represents the number 495.
The root-to-leaf path 4->9->1 represents the number 491.
The root-to-leaf path 4->0 represents the number 40.
Therefore, sum = 495 + 491 + 40 = 1026.

 

Constraints:

    The number of nodes in the tree is in the range [1, 1000].
    0 <= Node.val <= 9
    The depth of the tree will not exceed 10.



## DFS Traversal

Keep a stack of integers as traverse, if reach leaf, add to tracked sum

Time: O(N)
Space: O(1)

```
def solution(root):
    sumTotal = 0
    n = ""
    def traverse(root, n):
        nonlocal sumTotal
        if root is None: 
            return
        else:
            n += str(root.val)
            traverse(root.left, n)
            traverse(root.right, n)
            if root.left is None and root.right is None: 
                sumTotal += int(n)
    traverse(root, n)
    return sumTotal

def solution1(root):
    s = 0
    def add(root, curr):
        nonlocal s
        if not root:
            return
        add(root.left, curr*10 + root.val)
        add(root.right, curr*10 + root.val)
        if not root.left and not root.right:
            s += curr*10 + root.val
    add(root, 0)
    return s


def main():
    test = None
    solution(test)

if __name__ == '__main__':
    main()
```

# 133. Clone Graph
Medium

Given a reference of a node in a connected undirected graph.

Return a deep copy (clone) of the graph.

Each node in the graph contains a value (int) and a list (List[Node]) of its neighbors.

class Node {
    public int val;
    public List<Node> neighbors;
}

 

Test case format:

For simplicity, each node's value is the same as the node's index (1-indexed). For example, the first node with val == 1, the second node with val == 2, and so on. The graph is represented in the test case using an adjacency list.

An adjacency list is a collection of unordered lists used to represent a finite graph. Each list describes the set of neighbors of a node in the graph.

The given node will always be the first node with val = 1. You must return the copy of the given node as a reference to the cloned graph.

 

Example 1:

Input: adjList = [[2,4],[1,3],[2,4],[1,3]]
Output: [[2,4],[1,3],[2,4],[1,3]]
Explanation: There are 4 nodes in the graph.
1st node (val = 1)'s neighbors are 2nd node (val = 2) and 4th node (val = 4).
2nd node (val = 2)'s neighbors are 1st node (val = 1) and 3rd node (val = 3).
3rd node (val = 3)'s neighbors are 2nd node (val = 2) and 4th node (val = 4).
4th node (val = 4)'s neighbors are 1st node (val = 1) and 3rd node (val = 3).

Example 2:

Input: adjList = [[]]
Output: [[]]
Explanation: Note that the input contains one empty list. The graph consists of only one node with val = 1 and it does not have any neighbors.

Example 3:

Input: adjList = []
Output: []
Explanation: This an empty graph, it does not have any nodes.

 

Constraints:

    The number of nodes in the graph is in the range [0, 100].
    1 <= Node.val <= 100
    Node.val is unique for each node.
    There are no repeated edges and no self-loops in the graph.
    The Graph is connected and all nodes can be visited starting from the given node.

## Iteration BFS

Time: O(M + N)
Space: O(n)

```
def solution(node):
    visited = []
    unvisited = []

    def isVisited(node): 
        visits = False
        for i in range(len(visited)):
            if(node.val == visited[i].val):
                visits = True
        return visits

    if(node is not None):
        visited.append(node)
        clone = copy.deepcopy(node)
        for i in range(len(node.neighbors)):
            unvisited.append(node.neighbors[i])
    else:
        return None

    while(unvisited):
        if(not isVisited(unvisited[0])):
            visited.append(unvisited[0])
            copy.deepcopy(unvisited[0])
            for i in range(len(unvisited[0].neighbors)):
                if(not isVisited(unvisited[0].neighbors[i])):
                    unvisited.append(unvisited[0].neighbors[i])
        unvisited.pop(0)
    
    return clone
```

## DFS 

Perform dfs on the given node, and start building the graph while searching. Don't use any adjaceny lists in this queustion, the problem statement can throw you off. 

Make sure to check if the input node isn't an empty graph. In addition, when performing dfs, handle reaching visisted nodes by returning the copy of the visited node and stoping the recursion from getting caught in a cycle, hence an infinite loop. 

When building the graph, create a copy of the node and store it in visited, and also modify the node's copy's neighbors by saving the recursive searches outputs within the copy's list of neighbors. 

Time: O(M + N)
Space: O(N)

```
def cloneGraph(self, node: 'Node') -> 'Node':
    visited = {}
    def search(node, visited):
        if not node: 
            return node 
        if node in visited: 
            return visited[node]
        if node not in visited:
            currNode = Node(node.val, [])
            visited[node] = currNode
            currNode.neighbors = [search(n, visited) for n in node.neighbors]
        return currNode
    return search(node, visited)
```

# 134. Gas Station
Medium

There are n gas stations along a circular route, where the amount of gas at the ith station is gas[i].

You have a car with an unlimited gas tank and it costs cost[i] of gas to travel from the ith station to its next (i + 1)th station. You begin the journey with an empty tank at one of the gas stations.

Given two integer arrays gas and cost, return the starting gas station's index if you can travel around the circuit once in the clockwise direction, otherwise return -1. If there exists a solution, it is guaranteed to be unique

 

Example 1:

Input: gas = [1,2,3,4,5], cost = [3,4,5,1,2]
Output: 3
Explanation:
Start at station 3 (index 3) and fill up with 4 unit of gas. Your tank = 0 + 4 = 4
Travel to station 4. Your tank = 4 - 1 + 5 = 8
Travel to station 0. Your tank = 8 - 2 + 1 = 7
Travel to station 1. Your tank = 7 - 3 + 2 = 6
Travel to station 2. Your tank = 6 - 4 + 3 = 5
Travel to station 3. The cost is 5. Your gas is just enough to travel back to station 3.
Therefore, return 3 as the starting index.

Example 2:

Input: gas = [2,3,4], cost = [3,4,3]
Output: -1
Explanation:
You can't start at station 0 or 1, as there is not enough gas to travel to the next station.
Let's start at station 2 and fill up with 4 unit of gas. Your tank = 0 + 4 = 4
Travel to station 0. Your tank = 4 - 3 + 2 = 3
Travel to station 1. Your tank = 3 - 3 + 3 = 3
You cannot travel back to station 2, as it requires 4 unit of gas but you only have 3.
Therefore, you can't travel around the circuit once no matter where you start.

 

Constraints:

    n == gas.length == cost.length
    1 <= n <= 105
    0 <= gas[i], cost[i] <= 104

## Iteration 

There are two generalization in this question: 

If the total gas is less than the total travel cost, there can be no valid solutions.

If the gas to get to the next location is less the travel cost, then it can't be a valid start point.

Iterate over all such starting points, and move the starting point index accordingly. If no such points exist by the end of iterating over all starting points, there there can't be a starting point either. 


Time: O(N)
Space: O(1)

```
def canCompleteCircuit(self, gas: List[int], cost: List[int]) -> int:
    currDist = 0
    totalDist = 0
    startLocation = 0 

    for i in range(len(gas)):
        currDist += gas[i] - cost[i]
        totalDist += gas[i] - cost[i]
        if currDist < 0:
            startLocation = i + 1 
            currDist = 0 
    
    if totalDist < 0: 
        return -1
    else: 
        return startLocation
```

# 136. Single Number
Easy
12.8K
490
company
Amazon
company
Apple
company
Google
Given a non-empty array of integers nums, every element appears twice except for one. Find that single one.

You must implement a solution with a linear runtime complexity and use only constant extra space.

 

Example 1:

Input: nums = [2,2,1]
Output: 1
Example 2:

Input: nums = [4,1,2,1,2]
Output: 4
Example 3:

Input: nums = [1]
Output: 1
 

Constraints:

1 <= nums.length <= 3 * 104
-3 * 104 <= nums[i] <= 3 * 104
Each element in the array appears twice except for one element which appears only once.

## Bit Manipulation 

Time: O(n)
Space: O(1)

```
def singleNumber(self, nums: List[int]) -> int:
    single = 0 
    for n in nums: 
        single ^= n 
    return single 
```

# 138. Copy List with Random Pointer
Medium

A linked list of length n is given such that each node contains an additional random pointer, which could point to any node in the list, or null.

Construct a deep copy of the list. The deep copy should consist of exactly n brand new nodes, where each new node has its value set to the value of its corresponding original node. Both the next and random pointer of the new nodes should point to new nodes in the copied list such that the pointers in the original list and copied list represent the same list state. None of the pointers in the new list should point to nodes in the original list.

For example, if there are two nodes X and Y in the original list, where X.random --> Y, then for the corresponding two nodes x and y in the copied list, x.random --> y.

Return the head of the copied linked list.

The linked list is represented in the input/output as a list of n nodes. Each node is represented as a pair of [val, random_index] where:

    val: an integer representing Node.val
    random_index: the index of the node (range from 0 to n-1) that the random pointer points to, or null if it does not point to any node.

Your code will only be given the head of the original linked list.

 

Example 1:

Input: head = [[7,null],[13,0],[11,4],[10,2],[1,0]]
Output: [[7,null],[13,0],[11,4],[10,2],[1,0]]

Example 2:

Input: head = [[1,1],[2,1]]
Output: [[1,1],[2,1]]

Example 3:

Input: head = [[3,null],[3,0],[3,null]]
Output: [[3,null],[3,0],[3,null]]

 

Constraints:

    0 <= n <= 1000
    -104 <= Node.val <= 104
    Node.random is null or is pointing to some node in the linked list.



## Iteration

- Copy head
- Interweave the copies of the old list with the new list
- Save next pointer of original pointer as tmp
- Create copy of current node and set next of copy to tmp
- Come back to the head and curr.next.random = curr.random.next
- Then move head to head.next.next
- Finally unweave the answer, head.next.next = head.next.next.next and head.next = null

Time: O(n)
Space: O(1)

```
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
        
def solution(head):
    dic = dict()
    m = n = head
    while m:
        dic[m] = ListNode(m.val)
        m = m.next
    while n:
        dic[n].next = dic.get(n.next)
        dic[n].random = dic.get(n.random)
        n = n.next
    return dic.get(head)
```

# 139. Word Break
Medium

Given a string s and a dictionary of strings wordDict, return true if s can be segmented into a space-separated sequence of one or more dictionary words.

Note that the same word in the dictionary may be reused multiple times in the segmentation.

 

Example 1:

Input: s = "leetcode", wordDict = ["leet","code"]
Output: true
Explanation: Return true because "leetcode" can be segmented as "leet code".

Example 2:

Input: s = "applepenapple", wordDict = ["apple","pen"]
Output: true
Explanation: Return true because "applepenapple" can be segmented as "apple pen apple".
Note that you are allowed to reuse a dictionary word.

Example 3:

Input: s = "catsandog", wordDict = ["cats","dog","sand","and","cat"]
Output: false


## DFS or Backtrack (Brute Force):
For every word in the dictionary, check if it the current string starts with the word as a substring.
If it is a substring, search the remaining string likewise.
If remaining string is empty, return True.

```
def search(word):
    if word == "":
        memo[word] = True
        return True 
    else: 
        valid = False 
        for w in wordDict: 
            length = len(w)
            if word[:length] == w: 
                valid = valid or search(word[length:])
        return valid 

return search(s)
```

## DFS + Memo (Optimal):
Store intermediate inputs into a hashmap. 

```
def solution(s, wordDict):
    wordDict = set(wordDict)
    memo = {}
    
    def search(start):
        if start in memo: 
            return memo[start]
        
        if start == len(s):
            memo[start] = True
            return True
        for index in range(start + 1, len(s) + 1):
            word = s[start:index]
            if word in wordDict and search(index):
                memo[index] = True
                return True
            
        memo[start] = False
        return False

    return search(0)
```

## Dynamic Programming (Optimal):
Solve substring s[0, i], then solve substring s[0, i + 1].
Keep a dp table of valid word breaks, such that dp[i] == True means s[0, i] has a valid word break.
For each new letter, iterate from 0 to i, using j. If s[0, j] is True and s[j, i] is in dictionary, set dp[i] to True.
After all indices computed, return dp[-1], which will tell if a valid string can be made.

```
def solution(s, wordDict):
    table = [False for i in range(len(s) + 1)]
    table[0] = True
    for index in range(1, len(s) + 1):
        for subindex in range(index):
            validString = table[subindex]
            remainingString = s[subindex: index]
            if validString and remainingString in wordDict:
                table[index] = True
                break
    return table[-1]
```

# 140. Word Break II
Hard

Given a string s and a dictionary of strings wordDict, add spaces in s to construct a sentence where each word is a valid dictionary word. Return all such possible sentences in any order.

Note that the same word in the dictionary may be reused multiple times in the segmentation.

 

Example 1:

Input: s = "catsanddog", wordDict = ["cat","cats","and","sand","dog"]
Output: ["cats and dog","cat sand dog"]

Example 2:

Input: s = "pineapplepenapple", wordDict = ["apple","pen","applepen","pine","pineapple"]
Output: ["pine apple pen apple","pineapple pen apple","pine applepen apple"]
Explanation: Note that you are allowed to reuse a dictionary word.

Example 3:

Input: s = "catsandog", wordDict = ["cats","dog","sand","and","cat"]
Output: []

 

Constraints:

    1 <= s.length <= 20
    1 <= wordDict.length <= 1000
    1 <= wordDict[i].length <= 10
    s and wordDict[i] consist of only lowercase English letters.
    All the strings of wordDict are unique.



## Backtrack

Time: O(2^n)
Space: O(n)

```
def solution(s, wordDict):
    results = set()
    stack = []
    def backtrack(s, stack):
        stack = stack.copy()
        if not s:
            results.add(" ".join(stack))
        for word in wordDict:
            if s.startswith(word):
                stack.append(word)
                backtrack(s[len(word):], stack)
                stack.pop() 
    backtrack(s, [])
    return results
```

# 141. Linked List Cycle
Easy

Given head, the head of a linked list, determine if the linked list has a cycle in it.

There is a cycle in a linked list if there is some node in the list that can be reached again by continuously following the next pointer. Internally, pos is used to denote the index of the node that tail's next pointer is connected to. Note that pos is not passed as a parameter.

Return true if there is a cycle in the linked list. Otherwise, return false.

 

Example 1:

Input: head = [3,2,0,-4], pos = 1
Output: true
Explanation: There is a cycle in the linked list, where the tail connects to the 1st node (0-indexed).

Example 2:

Input: head = [1,2], pos = 0
Output: true
Explanation: There is a cycle in the linked list, where the tail connects to the 0th node.

Example 3:

Input: head = [1], pos = -1
Output: false
Explanation: There is no cycle in the linked list.


## Iteration + Hashmap (Brute Force)

Iterate through the linked list, and add each element visted so far to separate set. If the current element is in the set, return False. If finish iterating the linked list, then return True. 

Space: O(n)
Time: O(n)

## Iteration (Optimal)

Implement Floyd's Cycle finding algorithm. Have two counters, such that one counter moves 2 steps at a time and the other counter moves 1 step at a time. If the two counters meet then, return false. If the counter moving one step at a time reaches the end, then return True. 

Two possible while loops can be used, (1) while both the slow and fast runner are valid pointers (2) whilte the slow and fast runner do not equal each other. Each implementation will require a different break condition. When we check if the runners are valid pointrs, we break and return true if they are equal. If we check that they are equal, we break and return false if the runners are not longer valid pointers. The solutions are one of the same. 

Time: O(N)
Space: O(1)

[3,2,0,-4]
1


```
def solution(head):
    slow = head
    fast = head
    
    while slow and fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast: 
            return True
    
    return False
```

# 143. Reorder List
Medium

You are given the head of a singly linked-list. The list can be represented as:

L0 → L1 → … → Ln - 1 → Ln

Reorder the list to be on the following form:

L0 → Ln → L1 → Ln - 1 → L2 → Ln - 2 → …

You may not modify the values in the list's nodes. Only nodes themselves may be changed.

 

Example 1:

Input: head = [1,2,3,4]
Output: [1,4,2,3]

Example 2:

Input: head = [1,2,3,4,5]
Output: [1,5,2,4,3]


## Iteration (Reverse and Merge) (Optimal)

- 3 Steps: 
    - Find the middle of the linked list by using slow and fast pointers
    - Reverse the last half of the linked list
    - Merge the two linked lists

Space: O(1)
Time: O(n)

```
def findMiddle(head):
    slow = head
    fast = head
    while fast and fast.next: 
        slow = slow.next
        fast = fast.next.next
    return slow

def reverseList(head):
    prev = None
    while head:
        tmp1 = head.next
        head.next = prev
        prev = head
        head = tmp1
    return prev


def mergeLists(headA, headB):
    head = headA
    while headB.next:
        tmp1 = headA.next
        headA.next = headB
        headA = tmp1

        tmp2 = headB.next
        headB.next = headA
        headB = tmp2

    return head

middle = findMiddle(head)
reverse = reverseList(middle)
return mergeLists(head, reverse)
```

# 146. LRU Cache
Medium

Design a data structure that follows the constraints of a Least Recently Used (LRU) cache.

Implement the LRUCache class:

    LRUCache(int capacity) Initialize the LRU cache with positive size capacity.
    int get(int key) Return the value of the key if the key exists, otherwise return -1.
    void put(int key, int value) Update the value of the key if the key exists. Otherwise, add the key-value pair to the cache. If the number of keys exceeds the capacity from this operation, evict the least recently used key.

The functions get and put must each run in O(1) average time complexity.

 

Example 1:

Input
["LRUCache", "put", "put", "get", "put", "get", "put", "get", "get", "get"]
[[2], [1, 1], [2, 2], [1], [3, 3], [2], [4, 4], [1], [3], [4]]
Output
[null, null, null, 1, null, -1, null, -1, 3, 4]

Explanation
LRUCache lRUCache = new LRUCache(2);
lRUCache.put(1, 1); // cache is {1=1}
lRUCache.put(2, 2); // cache is {1=1, 2=2}
lRUCache.get(1);    // return 1
lRUCache.put(3, 3); // LRU key was 2, evicts key 2, cache is {1=1, 3=3}
lRUCache.get(2);    // returns -1 (not found)
lRUCache.put(4, 4); // LRU key was 1, evicts key 1, cache is {4=4, 3=3}
lRUCache.get(1);    // return -1 (not found)
lRUCache.get(3);    // return 3
lRUCache.get(4);    // return 4

 

Constraints:

    1 <= capacity <= 3000
    0 <= key <= 104
    0 <= value <= 105
    At most 2 * 105 calls will be made to get and put.



## Doubly Linked List (Optimal)

- Keep a doubly linked list to keep track of least recently used
- Init
    - Use dict as a cache
- Get 
    - If element not in cache, return -1
    - If element in cache
        - Move element to head
        - Return cache value
- Put
    - If element in cache
        - Change value in cache
        - Move node to head
    - If element not in cache
        - Increment curr capacity
        - If curr cap above max cap
            - Remove tail element from DLL
            - Remove element value from cache
            - Subtract curr cap
        - Add element to cache
        - Add element to head of DLL
- Helpers
    - Add Node
    - Delete Node
    - Move Node

Time: O(1)
Space: O(1)

```
class DLL:
    def __init__(self, val: int, key: int):
        self.prev = None
        self.post = None
        self.val = val
        self.key = key

class LRUCache:
    def addNodeSolution(self, node):
        node.pre = self.head
        node.post = self.head.post

        self.head.post.pre = node
        self.head.post = node

    def removeNode(self, node):
        pre = node.pre
        new = node.post

        pre.post = new
        new.pre = pre

    def moveFront(self, node):
        self.removeNode(node)
        self.addNode(node)

    def deleteTail(self):
        tail = self.tail.pre
        self.removeNode(tail)
        return tail
            

    def __init__(self, capacity: int):
        self.maxcap = capacity
        self.currcap = 0
        self.cache = {}
        
        self.head = DLL(1000, None)
        self.tail = DLL(-1000, None)
        self.head.post = self.tail
        self.head.pre = self.tail
        self.tail.post = self.head
        self.tail.pre = self.head
        

    def get(self, key: int) -> int:
        if key not in self.cache:
            return -1
        self.moveFront(self.cache[key])
        return self.cache[key].val
        

    def put(self, key: int, value: int) -> None:
        
        if key in self.cache:
            self.cache[key].val = value
            self.moveFront(self.cache[key])
        else:
            node = DLL(value, key)
            self.cache[key] = node
            self.addNode(node)
            self.currcap += 1
            if self.currcap > self.maxcap:
                tail = self.deleteTail()
                self.cache.pop(tail.key)
                self.currcap -= 1
```

# 148. Sort List
Medium

Given the head of a linked list, return the list after sorting it in ascending order.

 

Example 1:

Input: head = [4,2,1,3]
Output: [1,2,3,4]

Example 2:

Input: head = [-1,5,3,4,0]
Output: [-1,0,3,4,5]

Example 3:

Input: head = []
Output: []

 

Constraints:

    The number of nodes in the list is in the range [0, 5 * 104].
    -105 <= Node.val <= 105

 

Follow up: Can you sort the linked list in O(n logn) time and O(1) memory (i.e. constant space)?

## Merge Sort 

# 150. Evaluate Reverse Polish Notation
Medium
5.4K
844
company
Amazon
company
Google
company
LinkedIn
You are given an array of strings tokens that represents an arithmetic expression in a Reverse Polish Notation.

Evaluate the expression. Return an integer that represents the value of the expression.

Note that:

The valid operators are '+', '-', '*', and '/'.
Each operand may be an integer or another expression.
The division between two integers always truncates toward zero.
There will not be any division by zero.
The input represents a valid arithmetic expression in a reverse polish notation.
The answer and all the intermediate calculations can be represented in a 32-bit integer.
 

Example 1:

Input: tokens = ["2","1","+","3","*"]
Output: 9
Explanation: ((2 + 1) * 3) = 9
Example 2:

Input: tokens = ["4","13","5","/","+"]
Output: 6
Explanation: (4 + (13 / 5)) = 6
Example 3:

Input: tokens = ["10","6","9","3","+","-11","*","/","*","17","+","5","+"]
Output: 22
Explanation: ((10 * (6 / ((9 + 3) * -11))) + 17) + 5
= ((10 * (6 / (12 * -11))) + 17) + 5
= ((10 * (6 / -132)) + 17) + 5
= ((10 * 0) + 17) + 5
= (0 + 17) + 5
= 17 + 5
= 22
 

Constraints:

1 <= tokens.length <= 104
tokens[i] is either an operator: "+", "-", "*", or "/", or an integer in the range [-200, 200].

## Stack 

The relationhip of the integers and operations array, the position of the 1st and 2nd operand with the associated operation, is key to the solution. Iterating over the elements from the front the front to the back, allows to store the elements in a stack. Iterating over the element form the back to the front, allows one to recursively search subarrays. 

```
def evalRPN(self, tokens: List[str]) -> int:
    stack = []
    for n in tokens: 
        if n in {'+', '-', '*', '/'}:
            operand1 = int(stack.pop())
            operand2 = int(stack.pop())
            if n == '+':
                x = operand1 + operand2
            elif n == '-':
                x = operand2 - operand1
            elif n == '*':
                x = operand1 * operand2
            elif n == '/':
                x = operand2 / operand1
            stack.append(int(x))
        else: 
            stack.append(int(n))
    return stack[0]
```

# 152. Maximum Product Subarray
Medium

Given an integer array nums, find a contiguous non-empty subarray within the array that has the largest product,  

The test cases are generated so that the answer will fit in a 32-bit integer.

A subarray is a contiguous subsequence of the array.

 

Example 1:

Input: nums = [2,3,-2,4]
Output: 6
Explanation: [2,3] has the largest product 6.

Example 2:

Input: nums = [-2,0,-1]
Output: 0
Explanation: The result cannot be 2, because [-2,-1] is not a subarray.

## Dynamic Programming

We compute three products at each step: current minimum, a current maximum, and a historic maximum. By updating these values when iterating through the array, they will allow us to handle negative products and the zero product. 

We initialize all the values to the first element, and update them after reading each new value. We compute the current element value, the product of the current element with the previous mininum, and the product of the current element and the previous maximum at each step. The assign the minimum of the three to the currenent minimum and the maximum of the three to the current maximum. 

To get the largest maximum throughout the array, we update the historic maximum with the current maximum at each time step. 

Space: O(n)
Time: O(n) 

```
def maxProduct(self, nums: List[int]) -> int:
        currMax = currMin = totalMax = nums[0]

        for i in range(1, len(nums)):
            curr = nums[i]
            maxProduct = currMax * nums[i]
            minProduct = currMin * nums[i]
            currMin = min(curr, maxProduct, minProduct)
            currMax = max(curr, maxProduct, minProduct)
            totalMax = max(currMax, totalMax)

        return totalMax
```

# 153. Find Minimum in Rotated Sorted Array
Medium

Suppose an array of length n sorted in ascending order is rotated between 1 and n times. For example, the array nums = [0,1,2,4,5,6,7] might become:

    [4,5,6,7,0,1,2] if it was rotated 4 times.
    [0,1,2,4,5,6,7] if it was rotated 7 times.

Notice that rotating an array [a[0], a[1], a[2], ..., a[n-1]] 1 time results in the array [a[n-1], a[0], a[1], a[2], ..., a[n-2]].

Given the sorted rotated array nums of unique elements, return the minimum element of this array.

You must write an algorithm that runs in O(log n) time.

 

Example 1:

Input: nums = [3,4,5,1,2]
Output: 1
Explanation: The original array was [1,2,3,4,5] rotated 3 times.

Example 2:

Input: nums = [4,5,6,7,0,1,2]
Output: 0
Explanation: The original array was [0,1,2,4,5,6,7] and it was rotated 4 times.

Example 3:

Input: nums = [11,13,15,17]
Output: 11
Explanation: The original array was [11,13,15,17] and it was rotated 4 times. 


## Binary Search

The key to solving the problem is to determine which subarray to recursively search down for the minimum value. By comparing the mid point value with the end pointer of the array, we can tell how far the array has been rotated. 

If the middle value is greater than the right most value, then the right subarray it is not sorted. This means the array is rotated past the midpoint, so the values to the right must have the minimum value since that is the start of the original sorted array. 

The binary search can be avoided as well, if the has been not rotated. This can be determined if the left endpoint is less than the right endpoint, in which case the minimum of a sorted array must be the first value. 

Time: O(logN)
Space: O(1)

Code Solution
```
def findMin(self, nums: List[int]) -> int:
    if nums[0] < nums[-1]:
        return nums[0]
    
    left = 0
    right = len(nums) - 1
    
    while left < right: 
        mid = (left + right) // 2
        if nums[mid] > nums[right]: 
            left = mid + 1
        else: 
            right = mid
    
    return nums[left]
```

Example
```
EX: [3, 4, 5, 1, 2]
l = 0
r = 4
c = 2
l = 2
c = (2 + 4)/2 = 3
r = 3
c = (3 + 2)/2 = 2 
r = 2
```

# 155. Min Stack
Medium

Design a stack that supports push, pop, top, and retrieving the minimum element in constant time.

Implement the MinStack class:

MinStack() initializes the stack object.
void push(int val) pushes the element val onto the stack.
void pop() removes the element on the top of the stack.
int top() gets the top element of the stack.
int getMin() retrieves the minimum element in the stack.
You must implement a solution with O(1) time complexity for each function.

 

Example 1:

Input
["MinStack","push","push","push","getMin","pop","top","getMin"]
[[],[-2],[0],[-3],[],[],[],[]]

Output
[null,null,null,null,-3,null,0,-2]

Explanation
MinStack minStack = new MinStack();
minStack.push(-2);
minStack.push(0);
minStack.push(-3);
minStack.getMin(); // return -3
minStack.pop();
minStack.top();    // return 0
minStack.getMin(); // return -2
 

Constraints:

-231 <= val <= 231 - 1
Methods pop, top and getMin operations will always be called on non-empty stacks.
At most 3 * 104 calls will be made to push, pop, top, and getMin.

## Stack 

Keep a stack that keeps track of the minimum value so far in addition to the inputted values. This allows for one to access the minimum value so far at the end of the stack, in addition to the current value last processed by the stack. 

Time: O(1)
Space: O(n)

```
class MinStack:
    def __init__(self):
        self.stack = []
        
    def push(self, val: int) -> None:
        if self.stack: 
            self.stack.append((val, min(val, self.stack[-1][1])))
        else: 
            self.stack.append((val, val))
        
    def pop(self) -> None:
        self.stack.pop()

    def top(self) -> int:
        return self.stack[-1][0]

    def getMin(self) -> int:
        return self.stack[-1][1]
```

# 162. Find Peak Element
Medium

A peak element is an element that is strictly greater than its neighbors.

Given a 0-indexed integer array nums, find a peak element, and return its index. If the array contains multiple peaks, return the index to any of the peaks.

You may imagine that nums[-1] = nums[n] = -∞. In other words, an element is always considered to be strictly greater than a neighbor that is outside the array.

You must write an algorithm that runs in O(log n) time.

 

Example 1:

Input: nums = [1,2,3,1]
Output: 2
Explanation: 3 is a peak element and your function should return the index number 2.

Example 2:

Input: nums = [1,2,1,3,5,6,4]
Output: 5
Explanation: Your function can return either index number 1 where the peak element is 2, or index number 5 where the peak element is 6.

 

Constraints:

    1 <= nums.length <= 1000
    -231 <= nums[i] <= 231 - 1
    nums[i] != nums[i + 1] for all valid i.



## Binary Search

- Behavior:
    - If l - 1 < curr < l + 1, then peak on right
    - If l - 1 > curr > l + 1, then peak on left
    - If l - 1 < curr > l + 1, then reached peak
    - If l - 1 > curr < l + 1, then peak on left or right (search right)
        - Simplified to  l > l + 1 search left, else search right
- Set mid to inf
- While not (l - 1 > mid < l + 1)
    - mid = (l + r)//2
    - if l - 1 < mid < l + 1
        - search right - l = mid + 1
    - else:
        - search left - r = mid

Time: O(logN)
Space: O(1)

```
def solution(nums):
    l = 0; r = len(nums) - 1
    while l < r:
        mid = (l + r)//2
        if nums[mid] > nums[mid + 1]:
            r = mid
        else:
            l = mid + 1
    return l
    
        
```

# 169. Majority Element
Easy

Given an array nums of size n, return the majority element.

The majority element is the element that appears more than ⌊n / 2⌋ times. You may assume that the majority element always exists in the array.

 

Example 1:

Input: nums = [3,2,3]
Output: 3
Example 2:

Input: nums = [2,2,1,1,1,2,2]
Output: 2
 

Constraints:

n == nums.length
1 <= n <= 5 * 104
-109 <= nums[i] <= 109
 

Follow-up: Could you solve the problem in linear time and in O(1) space?

## Hashmap 

## Sorting

## Bit Manipulation

## Boyer-Moore Voting Algorithm

Consider array as a stream of data. Find the majority element within chunks, which are terminated as they are processed. 

This only method only works when a majority element exists, but it does not detect if a majority element exists, but just return what should be the majority element if one were to exist. 

Time: O(n)
Space: O(1)

```
def majorityElement(self, nums: List[int]) -> int:
    count = 0 
    majority = None 

    for n in nums: 
        if count == 0: 
            majority = n 
        
        if n == majority: 
            count += 1
        else: 
            count -= 1 
    
    return majority
```

# 173. Binary Search Tree Iterator
Medium

Implement the BSTIterator class that represents an iterator over the in-order traversal of a binary search tree (BST):

    BSTIterator(TreeNode root) Initializes an object of the BSTIterator class. The root of the BST is given as part of the constructor. The pointer should be initialized to a non-existent number smaller than any element in the BST.
    boolean hasNext() Returns true if there exists a number in the traversal to the right of the pointer, otherwise returns false.
    int next() Moves the pointer to the right, then returns the number at the pointer.

Notice that by initializing the pointer to a non-existent smallest number, the first call to next() will return the smallest element in the BST.

You may assume that next() calls will always be valid. That is, there will be at least a next number in the in-order traversal when next() is called.

 

Example 1:

Input
["BSTIterator", "next", "next", "hasNext", "next", "hasNext", "next", "hasNext", "next", "hasNext"]
[[[7, 3, 15, null, null, 9, 20]], [], [], [], [], [], [], [], [], []]
Output
[null, 3, 7, true, 9, true, 15, true, 20, false]

Explanation
BSTIterator bSTIterator = new BSTIterator([7, 3, 15, null, null, 9, 20]);
bSTIterator.next();    // return 3
bSTIterator.next();    // return 7
bSTIterator.hasNext(); // return True
bSTIterator.next();    // return 9
bSTIterator.hasNext(); // return True
bSTIterator.next();    // return 15
bSTIterator.hasNext(); // return True
bSTIterator.next();    // return 20
bSTIterator.hasNext(); // return False

 

Constraints:

    The number of nodes in the tree is in the range [1, 105].
    0 <= Node.val <= 106
    At most 105 calls will be made to hasNext, and next.

 

Follow up:

    Could you implement next() and hasNext() to run in average O(1) time and use O(h) memory, where h is the height of the tree?



## DFS Traversal

- Perform in order traversal and save the values in a list
- Keep searching root.left and return smallest number - 1
- Save an index to 0 and as long has index is less than length of array, allow has next

Space: O(1) 
Time: O(n)

```
class BSTIterator:
    def __init__(self, root):
        self.st = []
        self.pushleft(root)
    
    def pushleft(self, root):
        while root:
            self.st.append(root)
            root = root.left
            
    def traverse(self, root):
        if not root:
            return 
        self.traverse(root.left)
        self.inorder.append(root.val)
        self.traverse(root.right)

    def next(self):
        curr = self.st.pop()
        if curr.right:
            self.pushleft(curr.right)
        return curr.val
        
    def hasNext(self) -> bool:
        return len(self.st) > 0
        
```

# 179. Largest Number
Medium

Given a list of non-negative integers nums, arrange them such that they form the largest number and return it.

Since the result may be very large, so you need to return a string instead of an integer.

 

Example 1:

Input: nums = [10,2]
Output: "210"
Example 2:

Input: nums = [3,30,34,5,9]
Output: "9534330"
 

Constraints:

1 <= nums.length <= 100
0 <= nums[i] <= 109

## Comparator

```
```

# 189. Rotate Array
Medium

Given an integer array nums, rotate the array to the right by k steps, where k is non-negative.

 

Example 1:

Input: nums = [1,2,3,4,5,6,7], k = 3
Output: [5,6,7,1,2,3,4]
Explanation:
rotate 1 steps to the right: [7,1,2,3,4,5,6]
rotate 2 steps to the right: [6,7,1,2,3,4,5]
rotate 3 steps to the right: [5,6,7,1,2,3,4]
Example 2:

Input: nums = [-1,-100,3,99], k = 2
Output: [3,99,-1,-100]
Explanation: 
rotate 1 steps to the right: [99,-1,-100,3]
rotate 2 steps to the right: [3,99,-1,-100]
 

Constraints:

1 <= nums.length <= 105
-231 <= nums[i] <= 231 - 1
0 <= k <= 105
 

Follow up:

Try to come up with as many solutions as you can. There are at least three different ways to solve this problem.
Could you do it in-place with O(1) extra space?

## Brute Force Sinking Values 

Sink the k values at the end of the array for the front of array one at a time. Each sink operation involves swapping adjacent values until a value from the end of the array is set to a value in the beginning of the array without changing the order of the remaining elements.  

Time: O(k*n)
Space: O(1)

```
def rotate(self, nums: List[int], k: int) -> None:
    n = len(nums)
    k = k % n 
    for i in range(k):
        l = n - k - 1 + i
        for j in range(n - k):
            nums[l], nums[l-1] = nums[l-1], nums[l]
            l -= 1
    return nums   
```

## Array Cycles 

## Array Reverse

Perform three reverse operations. First, reverse the entire array. Then, reverse the first k elements, then the last n-k elements. 

Time: O(n)
Space: O(1)

```
def rotate(self, nums: List[int], k: int) -> None:
    k = k % len(nums) 
    def reverse(l, r): 
        while l <= r: 
            nums[l], nums[r] = nums[r], nums[l]
            l += 1 
            r -= 1

    reverse(0, len(nums) - 1)
    reverse(0, k - 1)
    reverse(k, len(nums) - 1)
    return nums  
```

# 190. Reverse Bits
Easy

Reverse bits of a given 32 bits unsigned integer.

Note:

    Note that in some languages, such as Java, there is no unsigned integer type. In this case, both input and output will be given as a signed integer type. They should not affect your implementation, as the integer's internal binary representation is the same, whether it is signed or unsigned.
    In Java, the compiler represents the signed integers using 2's complement notation. Therefore, in Example 2 above, the input represents the signed integer -3 and the output represents the signed integer -1073741825.
    
 

Example 1:

Input: n = 00000010100101000001111010011100
Output:    964176192 (00111001011110000010100101000000)
Explanation: The input binary string 00000010100101000001111010011100 represents the unsigned integer 43261596, so return 964176192 which its binary representation is 00111001011110000010100101000000.

Example 2:

Input: n = 11111111111111111111111111111101
Output:   3221225471 (10111111111111111111111111111111)
Explanation: The input binary string 11111111111111111111111111111101 represents the unsigned integer 4294967293, so return 3221225471 which its binary representation is 10111111111111111111111111111111.


## Iteration - Bit by Bit

Create a new reversed integer by mapping each bit at the end of the original number to the new number. 

My left shifting the new number and right shifting the original number, the values at the end of the original number now are values at the front of the new number. 

Select the last bit of the original number by performing a logical with 1. Append the bit to the new number my performing a logical or. 

Time: O(1)
Space: O(1)

```
def reverseBits(self, n: int) -> int:
    reverse = 0
    for i in range(32):
        reverse = reverse << 1
        reverse = reverse | (n & 1)
        n = n >> 1
    return reverse 
```

## Byte by Byte Optimization

Memoize byte values (8 bits) at a time with the following formula: 

`(byte * 0x0202020202 & 0x010884422010) % 1023`

Time: O(1)
Space: O(1)

# 191. Number of 1 Bits
Easy

Write a function that takes an unsigned integer and returns the number of '1' bits it has (also known as the Hamming weight).

Note:

    Note that in some languages, such as Java, there is no unsigned integer type. In this case, the input will be given as a signed integer type. It should not affect your implementation, as the integer's internal binary representation is the same, whether it is signed or unsigned.
    In Java, the compiler represents the signed integers using 2's complement notation. Therefore, in Example 3, the input represents the signed integer. -3.

 

Example 1:

Input: n = 00000000000000000000000000001011
Output: 3
Explanation: The input binary string 00000000000000000000000000001011 has a total of three '1' bits.

Example 2:

Input: n = 00000000000000000000000010000000
Output: 1
Explanation: The input binary string 00000000000000000000000010000000 has a total of one '1' bit.

Example 3:

Input: n = 11111111111111111111111111111101
Output: 31
Explanation: The input binary string 11111111111111111111111111111101 has a total of thirty one '1' bits.

 

Constraints:

    The input must be a binary string of length 32.

 
Follow up: If this function is called many times, how would you optimize it?

## Iteration

Keep left shifting n until it is 0. Perform logical and with n and 1, which tells us if the right most bit is a 1 or not. If logical and returns true, increment the count of number of 1's. Finally, return the total count. 

Time: O(n)
Space: O(1)

```
def hammingWeight(self, n: int) -> int:
    count = 0
    while n > 0: 
        if n & 1 == 1: 
            count += 1
        n = n >> 1
    return count 
```

# 198. House Robber
Medium

You are a professional robber planning to rob houses along a street. Each house has a certain amount of money stashed, the only constraint stopping you from robbing each of them is that adjacent houses have security systems connected and it will automatically contact the police if two adjacent houses were broken into on the same night.

Given an integer array nums representing the amount of money of each house, return the maximum amount of money you can rob tonight without alerting the police.

 

Example 1:

Input: nums = [1,2,3,1]
Output: 4
Explanation: Rob house 1 (money = 1) and then rob house 3 (money = 3).
Total amount you can rob = 1 + 3 = 4.

Example 2:

Input: nums = [2,7,9,3,1]
Output: 12
Explanation: Rob house 1 (money = 2), rob house 3 (money = 9) and rob house 5 (money = 1).
Total amount you can rob = 2 + 9 + 1 = 12.

 

Constraints:

    1 <= nums.length <= 100
    0 <= nums[i] <= 400



## Dynamic Programming (Optimal)

Figuring out the maximum value to rob at the n'th house can be solved given the maximum values robbed so far in the previous houses. In particular, we can either decide to rob the nth house or not. 

If we rob the nth house we can get a maximum value of the value at the n'th house and the maximum value robbed up to the n-2'th house. This is because we can't rob the house right before the n'th house or the alarms will go off. 

If we decide to not rob the nth house, we can get a maximum value of the previous maximum value robbed up to the n-1th house. 

At any given point, we can the maximum value to rob by choosing between the maximum of robbing or not robbing the nt'th hosue. 

This gives us the following recursive relationship: `dp[n] = max(dp[i-1], dp[i-2] + nums[i])`. Finally, if there is only 1 house, the max we can rob is the value at that house. In addition, if there are two houses, the maximum we can rob is the maximum vaulue of either of the houses.  

Time: O(n)
Space: O(n)

Code 
```
def rob(self, nums: List[int]) -> int:
    if len(nums) == 1: 
        return nums[0]
    dp = [0 for i in range(len(nums))]
    dp[0] = nums[0]
    dp[1] = max(nums[0], nums[1])
    for i in range(2, len(nums)): 
        dp[i] = max(nums[i] + dp[i-2], dp[i-1])
    return dp[-1]
```

Examples
```
COE - Try at least 3 examples
ex1. [2, 1, 1, 2]
    - check each subset and search again
    - check for the largets of each adjacent values
ex2. [2, 1, 2, 1, 1, 2, 1, 2]
```

# 199. Binary Tree Right Side View
Medium

Given the root of a binary tree, imagine yourself standing on the right side of it, return the values of the nodes you can see ordered from top to bottom.

 

Example 1:

Input: root = [1,2,3,null,5,null,4]
Output: [1,3,4]

Example 2:

Input: root = [1,null,3]
Output: [1,3]

Example 3:

Input: root = []
Output: []

 

Constraints:

    The number of nodes in the tree is in the range [0, 100].
    -100 <= Node.val <= 100



## DFS Traversal 

Preorder Traversal
- Search right first
- If depth == length of ans (depth visited so far), then apppend

Time: O(n)
Space: O(1)

```
def solution(root):
    if not root:
        return []
    ans = []
    def preorder(root, d):
        nonlocal ans
        if not root:
            return
        if d == len(ans):
            ans.append(root.val)
        preorder(root.right, d+1)
        preorder(root.left, d+1)
    preorder(root, 0)
    return ans
    
```

# 200. Number of Islands
Medium

Given an m x n 2D binary grid grid which represents a map of '1's (land) and '0's (water), return the number of islands.

An island is surrounded by water and is formed by connecting adjacent lands horizontally or vertically. You may assume all four edges of the grid are all surrounded by water.

 

Example 1:

Input: grid = [
  ["1","1","1","1","0"],
  ["1","1","0","1","0"],
  ["1","1","0","0","0"],
  ["0","0","0","0","0"]
]
Output: 1

Example 2:

Input: grid = [
  ["1","1","0","0","0"],
  ["1","1","0","0","0"],
  ["0","0","1","0","0"],
  ["0","0","0","1","1"]
]
Output: 3


## DFS (optimal)

Iterate through each index of the grid, if the grid value is 1, increment island count by 1. If grid value is 1, also start BFS. While performing BFS, set values grid values visited to 0, to remove double counts.

Time: O(N*M)
Space: O(N*M)

## BFS (Optimal)

Time: O(N*M)
Space: O(min(N, M))

```
def solution(grid):
    directions = {(0, 1), (0, -1), (1, 0), (-1, 0)}
    def validDirection(grid, newRow, newCol): 
        if newRow >= 0 and newRow < len(grid):
            if newCol >= 0 and newCol < len(grid[0]):
                if grid[newRow][newCol] == '1':
                    return True
        return False

    def traverse(grid, row, col):
        if grid[row][col] == '0':
            return
        else:
            if grid[row][col] == '1':
                grid[row][col] = '0'
            for direction in directions: 
                newRow = row + direction[0]
                newCol = col + direction[1]
                if validDirection(grid, newRow, newCol):
                    traverse(grid, newRow, newCol)

    totalIslands = 0
    for row in range(len(grid)):
        for col in range(len(grid[0])):
            value = grid[row][col]
            if value == '1': 
                totalIslands += 1
                traverse(grid, row, col) 

    return totalIslands

```

# 206. Reverse Linked List
Easy

Given the head of a singly linked list, reverse the list, and return the reversed list.

 

Example 1:

Input: head = [1,2,3,4,5]
Output: [5,4,3,2,1]

Example 2:

Input: head = [1,2]
Output: [2,1]

Example 3:

Input: head = []
Output: []


## Iteration (In-place modification)

Initalize the previous Node to None. Iterate through the list, by storing the next pointer into a temp value. Then set next pointer to the previous node. Update the previous node to the current  node. Finally update the date to the tmp value. 

This keeps going until the head is None, which means that the previous node contains the valid head of the new reversed list. 

Time: O(n)
Space: O(1)

```
def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
    prev = None
    while head:
        tmp = head.next
        head.next = prev
        prev = head
        head = tmp
    return prev
```

# 207. Course Schedule
Medium

There are a total of numCourses courses you have to take, labeled from 0 to numCourses - 1. You are given an array prerequisites where prerequisites[i] = [ai, bi] indicates that you must take course bi first if you want to take course ai.

    For example, the pair [0, 1], indicates that to take course 0 you have to first take course 1.

Return true if you can finish all courses. Otherwise, return false.

 

Example 1:

Input: numCourses = 2, prerequisites = [[1,0]]
Output: true
Explanation: There are a total of 2 courses to take. 
To take course 1 you should have finished course 0. So it is possible.

Example 2:

Input: numCourses = 2, prerequisites = [[1,0],[0,1]]
Output: false
Explanation: There are a total of 2 courses to take. 
To take course 1 you should have finished course 0, and to take course 0 you should also have finished course 1. So it is impossible.

## Topologial Sort (Optimal)

Keep track if a node hasn't been visited, is current being visited, or has finished been visited. A node is finished being visited if it checks all its neighbors. 

```
def canFinish(self, numCourses, prerequisites):
    graph = {}
    for course in range(numCourses):
        graph[course] = []
    for prereq in prerequisites: 
        graph[prereq[0]].append(prereq[1])

    invalid = True
    visited = [0 for i in range(numCourses)]

    def search(course):
        nonlocal visited
        nonlocal invalid
        if visited[course] == 1:
            invalid = False
        if visited[course] == 0:
            visited[course] = 1
            neighbors = graph[course]
            for neighbor in neighbors: 
                search(neighbor)
            visited[course] = 2

    for course in range(numCourses):
        if visited[course] == 0:
            search(course)

    return invalid
```

# 208. Implement Trie (Prefix Tree)
Medium

A trie (pronounced as "try") or prefix tree is a tree data structure used to efficiently store and retrieve keys in a dataset of strings. There are various applications of this data structure, such as autocomplete and spellchecker.

Implement the Trie class:

    Trie() Initializes the trie object.
    void insert(String word) Inserts the string word into the trie.
    boolean search(String word) Returns true if the string word is in the trie (i.e., was inserted before), and false otherwise.
    boolean startsWith(String prefix) Returns true if there is a previously inserted string word that has the prefix prefix, and false otherwise.

 

Example 1:

Input
["Trie", "insert", "search", "search", "startsWith", "insert", "search"]
[[], ["apple"], ["apple"], ["app"], ["app"], ["app"], ["app"]]
Output
[null, null, true, false, true, null, true]

Explanation
Trie trie = new Trie();
trie.insert("apple");
trie.search("apple");   // return True
trie.search("app");     // return False
trie.startsWith("app"); // return True
trie.insert("app");
trie.search("app");     // return True

 

Constraints:

    1 <= word.length, prefix.length <= 2000
    word and prefix consist only of lowercase English letters.
    At most 3 * 104 calls in total will be made to insert, search, and startsWith.



## Hashmap
-init
    -Current Node -> Non letter "."
    -26 Child Nodes - Dictionary of letters keys and values is a Boolean and a new Trie Node
    -Keep empty dictionary and add values- if a letter isn't there
-insert
    -Set
    -Checks if n'th letter is True, and if False, set True and Create new Node. Then, check children of that node and repeat
    -Check if dictionary contains a letter, if not add letter and new key of a dictionary node
-search
    -Iteratively search down a trail of dictionaries, and if any dictionary doesn't contains a letter at any depth, then break look and return False -- else return True
-startsWith
    -Get Modified
    -Search but with prefix

Time: O(n)
Space: O(n)

```
class Trie:
    def __init__(self):
        self.trie = {}

    def insert(self, word: str) -> None:
        trie = self.trie
        for letter in word: 
            if letter not in trie:
                trie[letter] = {}
            trie = trie[letter]
        trie['$'] = {}

    def search(self, word: str) -> bool:
        trie = self.trie
        for letter in word: 
            if letter not in trie: 
                return False
            trie = trie[letter]
        if '$' in trie: 
            return True
        else:
            return False
        

    def startsWith(self, prefix: str) -> bool:
        trie = self.trie
        for letter in prefix: 
            if letter not in trie: 
                return False
            trie = trie[letter]
        return True
```

# 209. Minimum Size Subarray Sum
Medium

Given an array of positive integers nums and a positive integer target, return the minimal length of a
subarray
whose sum is greater than or equal to target. If there is no such subarray, return 0 instead.

 

Example 1:

Input: target = 7, nums = [2,3,1,2,4,3]
Output: 2
Explanation: The subarray [4,3] has the minimal length under the problem constraint.

Example 2:

Input: target = 4, nums = [1,4,4]
Output: 1

Example 3:

Input: target = 11, nums = [1,1,1,1,1,1,1,1]
Output: 0

 

Constraints:

    1 <= target <= 109
    1 <= nums.length <= 105
    1 <= nums[i] <= 104

 
Follow up: If you have figured out the O(n) solution, try coding another solution of which the time complexity is O(n log(n)).

```
def minSubArrayLen(self, target: int, nums: List[int]) -> int:
    left = 0
    sumSoFar = 0
    minLength = inf
    
    for index in range(len(nums)):
        curr = nums[index]
        sumSoFar += curr
        
        while sumSoFar >= target:
            minLength = min(minLength, index - left + 1)
            #print(left, index, minLength, sumSoFar)
            sumSoFar -= nums[left]
            left += 1
    
    if minLength == inf: 
        return 0
    
    return minLength
```
# 210. Course Schedule II
Medium
8.6K
287
company
Amazon
company
TikTok
company
Google
There are a total of numCourses courses you have to take, labeled from 0 to numCourses - 1. You are given an array prerequisites where prerequisites[i] = [ai, bi] indicates that you must take course bi first if you want to take course ai.

For example, the pair [0, 1], indicates that to take course 0 you have to first take course 1.
Return the ordering of courses you should take to finish all courses. If there are many valid answers, return any of them. If it is impossible to finish all courses, return an empty array.

 

Example 1:

Input: numCourses = 2, prerequisites = [[1,0]]
Output: [0,1]
Explanation: There are a total of 2 courses to take. To take course 1 you should have finished course 0. So the correct course order is [0,1].
Example 2:

Input: numCourses = 4, prerequisites = [[1,0],[2,0],[3,1],[3,2]]
Output: [0,2,1,3]
Explanation: There are a total of 4 courses to take. To take course 3 you should have finished both courses 1 and 2. Both courses 1 and 2 should be taken after you finished course 0.
So one correct course order is [0,1,2,3]. Another correct ordering is [0,2,1,3].
Example 3:

Input: numCourses = 1, prerequisites = []
Output: [0]
 

Constraints:

1 <= numCourses <= 2000
0 <= prerequisites.length <= numCourses * (numCourses - 1)
prerequisites[i].length == 2
0 <= ai, bi < numCourses
ai != bi
All the pairs [ai, bi] are distinct.

## BFS 

Perform bfs iteratively on the nodes in the graph that have an indegree of 0. Remove outgoing edges for each new node searched. The next node that has an indegree of 0, is gauranteed to at the end of an edge previously removed. 

Reduce the time to compete the indegree of each node, but keeping a hashmap counter, and updating it as you iterate through the algorithm. 

If the final output sequence of the orders of node visited isn't the length of the total number of courses, this means not all nodes could successfuly be visted, hence there doesn't exist a valid topo sort. 

Time: O(V + E)
Space: O(V + E)

```
def findOrder(self, numCourses: int, prerequisites: List[List[int]]) -> List[int]:
    output = []
    queue = []
    graph = {}
    indegree = {}
    for p in prerequisites: 
        start = p[1]
        end = p[0]
        if start not in graph: 
            graph[start] = [end]
        else: 
            graph[start].append(end)

        if end not in indegree: 
            indegree[end] = 1
        else: 
            indegree[end] += 1

    for n in range(numCourses): 
        if n not in graph: 
            graph[n] = []
        if n not in indegree: 
            indegree[n] = 0 
    
    for n in range(numCourses): 
        if indegree[n] == 0: 
            queue.append(n)

    while queue:
        e = queue.pop()
        output.append(e)
        for e2 in graph[e]:
            indegree[e2] -= 1
            if indegree[e2] == 0: 
                queue.insert(0, e2)

    if len(output) != numCourses: 
        return []
    return output
```

# 211. Design Add and Search Words Data Structure
Medium

Design a data structure that supports adding new words and finding if a string matches any previously added string.

Implement the WordDictionary class:

    WordDictionary() Initializes the object.
    void addWord(word) Adds word to the data structure, it can be matched later.
    bool search(word) Returns true if there is any string in the data structure that matches word or false otherwise. word may contain dots '.' where dots can be matched with any letter.

 

Example:

Input
["WordDictionary","addWord","addWord","addWord","search","search","search","search"]
[[],["bad"],["dad"],["mad"],["pad"],["bad"],[".ad"],["b.."]]
Output
[null,null,null,null,false,true,true,true]

Explanation
WordDictionary wordDictionary = new WordDictionary();
wordDictionary.addWord("bad");
wordDictionary.addWord("dad");
wordDictionary.addWord("mad");
wordDictionary.search("pad"); // return False
wordDictionary.search("bad"); // return True
wordDictionary.search(".ad"); // return True
wordDictionary.search("b.."); // return True

## Trie (Optimal)

Add the words to a trie data structure. Store the trie in an embedded hashmap, and signify the end by adding a special character. In this case, set the value of the last's letter's hashmap (e.g. trie[lastLetter] =  {'#'}).

Recursively search through the trie to find if a word exist in the trie. Handle the sepcial character '.' by searching all letters in the current Trie level. 

Performance optimizations include pruning the the search space, by return at the first valid word found through when searching all letters for '.'

Time: O(M) s.t. M is length of longest string
Space: O(m) s.t. M is length of longest string

## '.' character support

Recursively search through all children values if '.' found. 

```
class WordDictionary:

    def __init__(self):
        self.trie = {}

    def addWord(self, word: str) -> None:
        currTrie = self.trie
        for index in range(len(word)):
            letter = word[index]
            if letter not in currTrie: 
                currTrie[letter] = {}
            currTrie = currTrie[letter]
        currTrie["$"] = {}

    def search(self, word: str) -> bool:
        def traverse(word, trie):
            currTrie = trie
            while word:
                currLetter = word[0]
                word = word[1:]
                if currLetter in currTrie: 
                    currTrie = currTrie[currLetter]
                else:
                    if currLetter == '.':
                        for letter in currTrie: 
                            reached = traverse(word, currTrie[letter])
                            if reached: 
                                return True
                    return False
            return "$" in currTrie
        
        return traverse(word, self.trie)
```

# 212. Word Search II
Hard

Given an m x n board of characters and a list of strings words, return all words on the board.

Each word must be constructed from letters of sequentially adjacent cells, where adjacent cells are horizontally or vertically neighboring. The same letter cell may not be used more than once in a word.

 

Example 1:

Input: board = [["o","a","a","n"],["e","t","a","e"],["i","h","k","r"],["i","f","l","v"]], words = ["oath","pea","eat","rain"]
Output: ["eat","oath"]

Example 2:

Input: board = [["a","b"],["c","d"]], words = ["abcb"]
Output: []

 

Constraints:

## DFS Traversal with Dictionary

Iterate through each index of the board. For each index, iterate through all the words in the dictionary. 

For each word in the dictionary perform a DFS search from the current element of the board. If any neighbors equals the first element of the current substring, then perform another recursive search from there with substring[1:]. Keep searching until the substring is empty.

For each index-word search pair, keep a dictionary to track previous indices visted in the search to avoid double counting a letter. 

## DFS Traversal with Trie

Store all words into a trie. 

Iterate through each index of the board, similar to above, and for each child of the trie, check if it is equal to a neighbor. Keep checking neighbors until marked node in trie is reached. 

Space: O(max(N, M)) s.t. N is size of true, M is size of board
Time: O(M * N) s.t. M is size of trie, M is size of board

## DFS Traversal with Trie and Visited Dictionary and Backtracking

Make sure to perform the optimization of removing letters that have already been visited. 

First add all the words to the trie, and if a word is matched, include a special character "$" to mark the end of word, and set the child of the last letter to "$" : word. 

Perform dfs on each cell on the word grid. Search all neighbors and tracked visited grid areas by directly modifying the grid and replacing it back to the value it previously was in. 

Also make sure to pop the "$" key is visited. After all the neighbors of the parent of the "$" is visited, pop the parent letter as well. 

```
def solution(board, words):
    trie = {}

    def addWords(words):
        nonlocal trie
        for word in words:
            currTrie = trie
            for index in range(len(word)):
                letter = word[index]
                if letter not in currTrie:
                    currTrie[letter] = {}            
                currTrie = currTrie[letter]
            currTrie['$'] = word

    addWords(words)

    directions = {(1, 0), (0, 1), (-1, 0), (0, -1)}

    def search(row, col, trie, wordSoFar):
        nonlocal ans
        nonlocal board
        if not (0 <= row < len(board)) or not (0 <= col < len(board[0])):
            return 
        letter = board[row][col]
        if not trie or letter not in trie:
            return 
        elif letter in trie and board[row][col] != '.':
            wordSoFar += letter
            if '$' in trie[letter]:
                if wordSoFar in trie[letter]['$']:
                    ans.add(wordSoFar)
                    trie[letter].pop('$')
            tmp = board[row][col]
            board[row][col] = '.'
            for direction in directions:
                search(row + direction[0], col + direction[1], trie[letter], wordSoFar)
            board[row][col] = tmp  
            if not trie[letter]:
                trie.pop(letter)

    ans = set()

    for row in range(len(board)):
        for col in range(len(board[0])):
            letter = board[row][col]
            if letter in trie: 
                search(row, col, trie, "")

    return list(ans)
```

# 213. House Robber II
Medium

You are a professional robber planning to rob houses along a street. Each house has a certain amount of money stashed. All houses at this place are arranged in a circle. That means the first house is the neighbor of the last one. Meanwhile, adjacent houses have a security system connected, and it will automatically contact the police if two adjacent houses were broken into on the same night.

Given an integer array nums representing the amount of money of each house, return the maximum amount of money you can rob tonight without alerting the police.

 

Example 1:

Input: nums = [2,3,2]
Output: 3
Explanation: You cannot rob house 1 (money = 2) and then rob house 3 (money = 2), because they are adjacent houses.

Example 2:

Input: nums = [1,2,3,1]
Output: 4
Explanation: Rob house 1 (money = 1) and then rob house 3 (money = 3).
Total amount you can rob = 1 + 3 = 4.

Example 3:

Input: nums = [1,2,3]
Output: 3

 

Constraints:

    1 <= nums.length <= 100
    0 <= nums[i] <= 1000



## Dynamic Programming

- Run House robber twice on the following intervals
    - 1 to n-1 and 2 to n
- Return the maximum robbed over both intervals
- This ensures that the first and last houses which are touching aren't robbed, which also ensuring that the max value is still calculated for cases when the first house and second to last house are robbed or when the second and the last house are robbed. 
    - x1, x2, x3, x4, x5 -> (x1 and x4), (x2 and x5), but not (x1 and x5)

Time: O(n)
Space: O(n)

```
def rob(self, nums: List[int]) -> int:
    if len(nums) <= 2: 
            return max(nums)

    dp1 = [0 for i in range(len(nums))]
    dp1[0] = nums[0]
    dp1[1] = max(nums[0], nums[1])
    for i in range(2, len(nums) - 1):
        dp1[i] = max(dp1[i - 1], dp1[i-2] + nums[i])

    dp2 = [0 for i in range(len(nums))]
    dp2[1] = nums[1]
    dp2[2] = max(nums[1], nums[2])
    for i in range(3, len(nums)):
        dp2[i] = max(dp2[i - 1], dp2[i-2] + nums[i])


    return max(max(dp1), max(dp2))
```

# 215. Kth Largest Element in an Array
Medium

Given an integer array nums and an integer k, return the kth largest element in the array.

Note that it is the kth largest element in the sorted order, not the kth distinct element.

You must solve it in O(n) time complexity.

 

Example 1:

Input: nums = [3,2,1,5,6,4], k = 2
Output: 5

Example 2:

Input: nums = [3,2,3,1,2,4,5,5,6], k = 4
Output: 4

 

Constraints:

    1 <= k <= nums.length <= 105
    -104 <= nums[i] <= 104



## Heap

Time: O(logN)
Space: O(n)

## Quick Select

Time: O(logN)
Space: O(n)

```
import random 

def solution(nums, k):
    def partition(l, r):
        p = random.randint(l, r)
        nums[p], nums[r] = nums[r], nums[p]
        pv = nums[r]
        si = l
        for i in range(l, r):
            if nums[i] < pv:
                nums[i], nums[si] = nums[si], nums[i]
                si += 1
        nums[r], nums[si] = nums[si], nums[r]
        return si

    def select(l, r, p, kv):
        while p != kv:
            p = partition(l, r)
            if p < kv:
                l = p + 1
            elif p > kv:
                r = p - 1

    select(0, len(nums)-1, len(nums), len(nums) - k)
    return nums[len(nums) - k]
        
```

# 217. Contains Duplicate
Easy

Given an integer array nums, return true if any value appears at least twice in the array, and return false if every element is distinct.

 

Example 1:

Input: nums = [1,2,3,1]
Output: true

Example 2:

Input: nums = [1,2,3,4]
Output: false

Example 3:

Input: nums = [1,1,1,3,3,4,3,2,4,2]
Output: true

## Sorting (Optimal)
Sort the array. Since the values are in ascending order, if curr value is same as prev, then there must be a duplicate. 

Space: O(1)
Time: O(nlogn)

## Hashmap (Optimal)
Store each value you run into in a visited set. If an element already found in this set, this means a duplicate exists. If no elements are found in the visisted set, then not dupicates are in the array.

Space: O(n)
Time: O(n) 

```
def solution(nums):
    duplicate = set()
    for n in nums:
        if n in duplicate: 
            return True
        duplicate.add(n)
    return False
```

# 221. Maximal Square
Medium

Given an m x n binary matrix filled with 0's and 1's, find the largest square containing only 1's and return its area.

 

Example 1:

Input: matrix = [["1","0","1","0","0"],["1","0","1","1","1"],["1","1","1","1","1"],["1","0","0","1","0"]]
Output: 4

Example 2:

Input: matrix = [["0","1"],["1","0"]]
Output: 1

Example 3:

Input: matrix = [["0"]]
Output: 0

 

Constraints:

    m == matrix.length
    n == matrix[i].length
    1 <= m, n <= 300
    matrix[i][j] is '0' or '1'.

## Dynamic Programming

The key is to understand the relationship from a square from side length 1 to side length 2. 

If the right, bottom, and bottom right values are all 1, and the top left value is 1, then a square of side length 2 exists at the top left. This relationship can be generalized to square of arbitrary size: 

topleft = min(bottom, right, bottom right) + 1
dp[i][j] = min(dp[i][j + 1], dp[i + 1][j], dp[i+1][i+1]) + 1

In general, if the top left element is 1, then it is the top left element of a square with side length equal to the minimum value of it's lower right neighbor's size plus 1. 

```
def maximalSquare(self, matrix: List[List[str]]) -> int:
    output = [[0 for i in range(len(matrix[0]))] for j in range(len(matrix))]

    for i in reversed(range(len(matrix))):
        for j in reversed(range(len(matrix[0]))):
            output[i][j] = int(matrix[i][j])
    print(output)

    for i in reversed(range(len(output) - 1)):
        for j in reversed(range(len(output[0]) - 1)):
            if output[i][j] != 0: 
                output[i][j] = min(output[i+1][j], output[i+1][j+1], output[i][j+1]) + 1
    
    maxSide = 0
    for i in range(len(output)):
        for j in range(len(output[0])):
            maxSide = max(maxSide, output[i][j])
    return maxSide ** 2
```

# 224. Basic Calculator
Hard

Given a string s representing a valid expression, implement a basic calculator to evaluate it, and return the result of the evaluation.

Note: You are not allowed to use any built-in function which evaluates strings as mathematical expressions, such as eval().

Example 1:

Input: s = "1 + 1"
Output: 2
Example 2:

Input: s = " 2-1 + 2 "
Output: 3
Example 3:

Input: s = "(1+(4+5+2)-3)+(6+8)"
Output: 23
 

Constraints:

1 <= s.length <= 3 * 105
s consists of digits, '+', '-', '(', ')', and ' '.
s represents a valid expression.
'+' is not used as a unary operation (i.e., "+1" and "+(2 + 3)" is invalid).
'-' could be used as a unary operation (i.e., "-1" and "-(2 + 3)" is valid).
There will be no two consecutive operators in the input.
Every number and running calculation will fit in a signed 32-bit integer.

# Stack 

To account for addition and subtraction, which occurs in between two operands, keep track of a variable sign and a variable operand, which will add the right operand * 1, or add the right operand * -1, after it has been computed. This allows for us to traverse the input argument from left to right, while also accounting for arithmetic operations. 

Parentheses are handled, by reseting the operand, and holding the result computed thus far into a stack, which will be popped of after reaching the corresponding end parentheses. 

```
def calculate(self, s: str) -> int:
        stack = []
        operand = 0
        sign = 1
        val = 0 

        for c in s:
            if c.isdigit(): 
                operand = operand * 10 + int(c)
            elif c == '-': 
                val += sign * operand 
                sign = -1
                operand = 0 
            elif c == '+':
                val += sign * operand 
                sign = 1
                operand = 0 
            elif c == '(': 
                stack.append(val)
                stack.append(sign)
                sign = 1
                val = 0 
            elif c == ')': 
                val += sign * operand
                currSign = stack.pop()
                val *= currSign 
                currOperand = stack.pop()
                val += currOperand
                operand = 0 
        return val + (sign * operand)
```

# 226. Invert Binary Tree
Easy

Given the root of a binary tree, invert the tree, and return its root.

 

Example 1:

Input: root = [4,2,7,1,3,6,9]
Output: [4,7,2,9,6,3,1]

Example 2:

Input: root = [2,1,3]
Output: [2,3,1]

Example 3:

Input: root = []
Output: []

 

Constraints:

    The number of nodes in the tree is in the range [0, 100].
    -100 <= Node.val <= 100

## DFS Traversal - Preorder (Optimal)

Recursively traverse the tree. During the traversal, perform a preorder swap of the left and right children. This ensures, that all left and right children are flipped for all nodes of the tree. 

Space: O(n)
Time: O(n)

```
def invertTree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
    def search(node):
        if not node: 
            return 
        else: 
            node.left, node.right = node.right, node.left
            search(node.left)
            search(node.right)

    search(root)
    return root
```

# 227. Basic Calculator II
Medium

Given a string s which represents an expression, evaluate this expression and return its value. 

The integer division should truncate toward zero.

You may assume that the given expression is always valid. All intermediate results will be in the range of [-231, 231 - 1].

Note: You are not allowed to use any built-in function which evaluates strings as mathematical expressions, such as eval().

 

Example 1:

Input: s = "3+2*2"
Output: 7

Example 2:

Input: s = " 3/2 "
Output: 1

Example 3:

Input: s = " 3+5 / 2 "
Output: 5

 

Constraints:

    1 <= s.length <= 3 * 105
    s consists of integers and operators ('+', '-', '*', '/') separated by some number of spaces.
    s represents a valid expression.
    All the integers in the expression are non-negative integers in the range [0, 231 - 1].
    The answer is guaranteed to fit in a 32-bit integer.



## Iteration (Optimal)

- Parse one item at a time
    - Add + operator to beginning of string
    - select operator as first element and increment pointer then get next number
        - Get next number
            - If ' ' increment pointer
            - Set number to 0
            - Add next character value to current number * 10
            - Go until reach another operator
    - Perform operation to stack based on the value of the operator
        - if +, add number to stack
        - if -, add negative of number to stack
        - if *, pop from stack and add product of current and popped number to stack
        - if /, pop from stack and add the divided number back to stack
            - 1 - 2 / 2 = 0
            - 1 -1 = 0
    - 1 pointer, while pointer < string length


Space: O(k) 
Time: O(n) k - number of characters, n - number of unique integers

```
def solution(s):
    i = 0
    s = '+' + s
    l = len(s) - 1
    sm = 0
    while i < l:
        op = s[i]
        n = 0
        i += 1
        while i <= l and s[i] not in '+-*/':
            if s[i] != ' ':
                n = n*10 + int(s[i])
            i += 1
        if op == '+':
            ln = n
            sm += ln
        elif op == '-':
            ln = -1 * n
            sm += ln
        elif op == '*':
            sm -= ln
            ln = ln * n
            sm += ln
        elif op == '/':
            sm -= ln
            ln = int(ln/n)
            sm += ln
    return sm
```

# 230. Kth Smallest Element in a BST
Medium

Given the root of a binary search tree, and an integer k, return the kth smallest value (1-indexed) of all the values of the nodes in the tree.

 

Example 1:

Input: root = [3,1,4,null,2], k = 1
Output: 1

Example 2:

Input: root = [5,3,6,2,4,null,null,1], k = 3
Output: 3

 

Constraints:

    The number of nodes in the tree is n.
    1 <= k <= n <= 104
    0 <= Node.val <= 104

 

Follow up: If the BST is modified often (i.e., we can do insert and delete operations) and you need to find the kth smallest frequently, how would you optimize?
230. Kth Smallest Element in a BST
Medium

Given the root of a binary search tree, and an integer k, return the kth smallest value (1-indexed) of all the values of the nodes in the tree.

 

Example 1:

Input: root = [3,1,4,null,2], k = 1
Output: 1

Example 2:

Input: root = [5,3,6,2,4,null,null,1], k = 3
Output: 3

 

Constraints:

    The number of nodes in the tree is n.
    1 <= k <= n <= 104
    0 <= Node.val <= 104

 

Follow up: If the BST is modified often (i.e., we can do insert and delete operations) and you need to find the kth smallest frequently, how would you optimize?


## DFS Traversal

Perform inorder traversal of the tree, while storing each element visited into a list. Then return the kth element of the list found at index k-1. 

Time: O(n)
Space: O(n)

```
def solution(root, k):
    elements = []
    def inOrder(root):
        nonlocal elements
        if not root:
            return
        else:
            inOrder(root.left)
            elements.append(root.val)
            inOrder(root.right)
    inOrder(root)
    return elements[k - 1]
```
# 232. Implement Queue using Stacks
Easy

Implement a first in first out (FIFO) queue using only two stacks. The implemented queue should support all the functions of a normal queue (push, peek, pop, and empty).

Implement the MyQueue class:

void push(int x) Pushes element x to the back of the queue.
int pop() Removes the element from the front of the queue and returns it.
int peek() Returns the element at the front of the queue.
boolean empty() Returns true if the queue is empty, false otherwise.
Notes:

You must use only standard operations of a stack, which means only push to top, peek/pop from top, size, and is empty operations are valid.
Depending on your language, the stack may not be supported natively. You may simulate a stack using a list or deque (double-ended queue) as long as you use only a stack's standard operations.
 

Example 1:

Input
["MyQueue", "push", "push", "peek", "pop", "empty"]
[[], [1], [2], [], [], []]
Output
[null, null, null, 1, 1, false]

Explanation
MyQueue myQueue = new MyQueue();
myQueue.push(1); // queue is: [1]
myQueue.push(2); // queue is: [1, 2] (leftmost is front of the queue)
myQueue.peek(); // return 1
myQueue.pop(); // return 1, queue is [2]
myQueue.empty(); // return false
 

Constraints:

1 <= x <= 9
At most 100 calls will be made to push, pop, peek, and empty.
All the calls to pop and peek are valid.
 

Follow-up: Can you implement the queue such that each operation is amortized O(1) time complexity? In other words, performing n operations will take overall O(n) time even if one of those operations may take longer.

## Lazy Iterator Evaluation

Use second stack to keep hold of queue values in reverse order from last `pop()` operation. The queue pop operation, pops off all the values of the first stack and appends them to the second stack one at a time. Therefore, the top element of the second stack is the first element.

Every push operation adds elements to the first stack, and updates a global front variable if the first stack is empty to allow for easy access of `peek()`.

Time: O(1)
Space: O(1)

```
class MyQueue:
    def __init__(self):
        self.stack1 = []
        self.stack2 = []
        self.front = None 

    def push(self, x: int) -> None:
        if not self.stack1: 
            self.front = x 
        self.stack1.append(x)
        
    def pop(self) -> int:
        if not self.stack2: 
            while self.stack1:
                e = self.stack1.pop()
                self.stack2.append(e)
        return self.stack2.pop()

    def peek(self) -> int:
        if self.stack2: 
            return self.stack2[-1]
        return self.front 
        
    def empty(self) -> bool:
        return not self.stack1 and not self.stack2
```

# 235. Lowest Common Ancestor of a Binary Search Tree
Medium

Given a binary search tree (BST), find the lowest common ancestor (LCA) node of two given nodes in the BST.

According to the definition of LCA on Wikipedia: “The lowest common ancestor is defined between two nodes p and q as the lowest node in T that has both p and q as descendants (where we allow a node to be a descendant of itself).”

 

Example 1:

Input: root = [6,2,8,0,4,7,9,null,null,3,5], p = 2, q = 8
Output: 6
Explanation: The LCA of nodes 2 and 8 is 6.

Example 2:

Input: root = [6,2,8,0,4,7,9,null,null,3,5], p = 2, q = 4
Output: 2
Explanation: The LCA of nodes 2 and 4 is 2, since a node can be a descendant of itself according to the LCA definition.

Example 3:

Input: root = [2,1], p = 2, q = 1
Output: 2

 

Constraints:

    The number of nodes in the tree is in the range [2, 105].
    -109 <= Node.val <= 109
    All Node.val are unique.
    p != q
    p and q will exist in the BST.

## DFS Traversal - Recursive

The binary search tree properties must be used to convert this problem into a range search. Using nodes p and q as limits of a range, we can search through the binary search through efficiently. 

If the current node is greater than the upper bound, the solution must be found within the left subtree. 

Likewise, if the current node is less than the lower bound, the solution must be found within the right subtree. 

If the current node is within the range, it will likely be the lowest height, so we can return the node.

Space: O(n)
Time: O(n)

```
def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
    high = max(p.val, q.val)
    low = min(p.val, q.val)
    def inorder(root):
        if not root:
            return 
        if low <= root.val <= high:
            return root
        elif root.val <= low:
            return inorder(root.right)
        elif root.val >= high:
            return inorder(root.left)
    return inorder(root)
```

## DFS Traversal - Iterative 

- If low <= root <= high return root
- If root <= low return search root.right
- If root >= high search root.right

Space: O(1)
Time: O(n)

```
def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
    inRange = False
    low = min(p.val, q.val)
    high = max(p.val, q.val)
    while not inRange:
        if low <= root.val <= high:
            return root
        elif root.val <= low:
            root = root.right
        elif root.val >= high:
            root = root.left
```

# 236. Lowest Common Ancestor of a Binary Tree
Medium

Given a binary tree, find the lowest common ancestor (LCA) of two given nodes in the tree.

According to the definition of LCA on Wikipedia: “The lowest common ancestor is defined between two nodes p and q as the lowest node in T that has both p and q as descendants (where we allow a node to be a descendant of itself).”

 

Example 1:

Input: root = [3,5,1,6,2,0,8,null,null,7,4], p = 5, q = 1
Output: 3
Explanation: The LCA of nodes 5 and 1 is 3.

Example 2:

Input: root = [3,5,1,6,2,0,8,null,null,7,4], p = 5, q = 4
Output: 5
Explanation: The LCA of nodes 5 and 4 is 5, since a node can be a descendant of itself according to the LCA definition.

Example 3:

Input: root = [1,2], p = 1, q = 2
Output: 1

 

Constraints:

    The number of nodes in the tree is in the range [2, 105].
    -109 <= Node.val <= 109
    All Node.val are unique.
    p != q
    p and q will exist in the tree.



## DFS Traversal

- Post order
- If l and r, return root
- Else return l or r

Space: O(n)
Time: O(n)

```
def solution(root, p, q):
    ans = None
    def postorder(root):
        if not root:
            return
        l = postorder(root.left)
        r = postorder(root.right)
        if root == p or root == q:
            return root
        if l and r:
            return root
        return l or r
    return postorder(root)
```

# 238. Product of Array Except Self
Medium

Given an integer array nums, return an array answer such that answer[i] is equal to the product of all the elements of nums except nums[i].

The product of any prefix or suffix of nums is guaranteed to fit in a 32-bit integer.

You must write an algorithm that runs in O(n) time and without using the division operation.

 

Example 1:

Input: nums = [1,2,3,4]
Output: [24,12,8,6]

Example 2:

Input: nums = [-1,1,0,-3,3]
Output: [0,0,9,0,0]

## Prefix Sum Iteration (Optimal)

Keep track of products in an array. Compute the prefix sum, intialized at 1, moving left to right on the original array and appending the values to the array of products. Compute the suffix sum, initialized at 1, moving from right to left on the original array. Multiply the suffix to the array of products, and return it as the output.

Space: O(n)
Time: O(n)

```
def productExceptSelf(self, nums: List[int]) -> List[int]:
    product = []
    prefix = 1 
    for i in range(len(nums)):
        product.append(prefix)
        prefix *= nums[i]
        
    suffix = 1
    for i in reversed(range(len(nums))):
        product[i] *= suffix
        suffix *= nums[i]
        
    return product
```

# 242. Valid Anagram
Easy

Given two strings s and t, return true if t is an anagram of s, and false otherwise.

An Anagram is a word or phrase formed by rearranging the letters of a different word or phrase, typically using all the original letters exactly once.

 

Example 1:

Input: s = "anagram", t = "nagaram"
Output: true

Example 2:

Input: s = "rat", t = "car"
Output: false

 

Constraints:

    1 <= s.length, t.length <= 5 * 104
    s and t consist of lowercase English letters.

 

Follow up: What if the inputs contain Unicode characters? How would you adapt your solution to such a case?

## Sorting (suboptimal)

Sort the strings and check if they are equal strings afterwards. 

Space: O(n)
Time: O(nlogn) 


## Hashmap (optimal)

Use a single hashmap to count all the character counts. 

Space: O(n)
Time: O(n)

```
def isAnagram(self, s: str, t: str) -> bool:
    if len(s) != len(t):
        return False 

    counter = {}
    for i in s: 
        if i not in counter: 
            counter[i] = 1
        else: 
            counter[i] += 1
    
    for i in t: 
        if i not in counter:
            return False 
        else:
            counter[i] -= 1
            if counter[i] < 0: 
                return False 
    
    return True 
```

# 249. Group Shifted Strings
Medium

We can shift a string by shifting each of its letters to its successive letter.

    For example, "abc" can be shifted to be "bcd".

We can keep shifting the string to form a sequence.

    For example, we can keep shifting "abc" to form the sequence: "abc" -> "bcd" -> ... -> "xyz".

Given an array of strings strings, group all strings[i] that belong to the same shifting sequence. You may return the answer in any order.

 

Example 1:

Input: strings = ["abc","bcd","acef","xyz","az","ba","a","z"]
Output: [["acef"],["a","z"],["abc","bcd","xyz"],["az","ba"]]

Example 2:

Input: strings = ["a"]
Output: [["a"]]

 

Constraints:

    1 <= strings.length <= 200
    1 <= strings[i].length <= 50
    strings[i] consists of lowercase English letters.



## Hashmap

- subtract the entire string by the first letter's value and store the value as string
- add the resulting integer to a dictionary, and add the current string to that list
- return keys of the dictionary
- azy, zab -> 0,25,24 , 0,-25,-24
- az, ba
    - 0 25, 0 -1 mod 26 -> 

Time: O(n)
Space: O(1)

```
def solution(strings):
    def val(s):
        if not s:
            return ""
        v = ord(s[0])
        o = []
        for i in s:
            o.append((ord(i)-v) % 26)
        return str(o)

    d = {}
    for s in strings:
        v = val(s)
        if v in d:
            d[v].append(s)
        else:
            d[v] = [s]

    return d.values()
```

# 252. Meeting Rooms
Easy

Given an array of meeting time intervals where intervals[i] = [starti, endi], determine if a person could attend all meetings.

 

Example 1:

Input: intervals = [[0,30],[5,10],[15,20]]
Output: false

Example 2:

Input: intervals = [[7,10],[2,4]]
Output: true


## Sorting

Sort the given list of intervals based on the starting time. Iterate through the list, and if the current start time is less than the previous end time, return false. 

Time: O(nlogn)
Space: O(1) because no additional space allocated with sorting



```
from math import inf
def solution(intervals):
    intervals.sort()
    prevEndTime = -inf
    for interval in intervals:
        startTime = interval[0]
        if prevEndTime > startTime:
            return False
        prevEndTime = interval[1]
    return True
```

# 253. Meeting Rooms II
Medium

Given an array of meeting time intervals intervals where intervals[i] = [starti, endi], return the minimum number of conference rooms required.

 

Example 1:

Input: intervals = [[0,30],[5,10],[15,20]]
Output: 2

Example 2:

Input: intervals = [[7,10],[2,4]]
Output: 1

## Heap (optimal)

Keep a Min Heap of endTimes, such that the top element of the heap is the first room that will be empty at any given moment in time. 

Iterate through the meeting times in sorted order of start time. 

If the next meeting time starts after the min endTime so far, perform heap replace -- imagine the current person is taking the existing room. 

If the next meeting time starts before the min endTime so far, perform heap push -- imagine the current person is taking a new room, since no rooms are free. If the minimum endTime is greater than the current startTime, all the other rooms must have largers endTimes, so they must not be free either.   

Time: O(nlogn)
Space: O(n)

```
from collections import heapq

def minMeetingRooms(self, intervals: List[List[int]]) -> int:
    intervals = sorted(intervals, key = lambda x: x[0])
    endTimes = []
    for interval in intervals: 
        startTime = interval[0]
        endTime = interval[1]
        if endTimes and startTime >= endTimes[0]:
            heapq.heapreplace(endTimes, endTime)
        else: 
            heapq.heappush(endTimes, endTime)
    return len(endTimes)
```

# 261. Graph Valid Tree
Medium

You have a graph of n nodes labeled from 0 to n - 1. You are given an integer n and a list of edges where edges[i] = [ai, bi] indicates that there is an undirected edge between nodes ai and bi in the graph.

Return true if the edges of the given graph make up a valid tree, and false otherwise.

 

Example 1:

Input: n = 5, edges = [[0,1],[0,2],[0,3],[1,4]]
Output: true

Example 2:

Input: n = 5, edges = [[0,1],[1,2],[2,3],[1,3],[1,4]]
Output: false

 

Constraints:

    1 <= n <= 2000
    0 <= edges.length <= 5000
    edges[i].length == 2
    0 <= ai, bi < n
    ai != bi
    There are no self-loops or repeated edges.



## BFS Traversal

Construct graph
        
Check if tree
Perform BFS but keep a queue called visited
If edge is in visted return False
Keep track of prev node

Time: O(n)
Space: O(n)

```
def solution(n, edges):
    unvisited = []
    for i in range(n):
        unvisited.append(i)

    visited = []
    bfsqueue = []
    prev = None
    bfsqueue.append((0, prev))
    unvisited.remove(0)
    while bfsqueue:
        node = bfsqueue.pop(0)
        prev = node[1]
        visited.append(node[0])
        if node[0] in unvisited:
            unvisited.remove(node[0])
        for edge in edges:
            if (edge[0] == node[0]) or (edge[1] == node[0]):
                if (edge[0] == prev) or (edge[1] == prev):
                    pass
                elif (edge[1] not in visited):
                    bfsqueue.append((edge[1], node[0]))
                    visited.append(edge[1])
                elif (edge[0] not in visited): 
                    bfsqueue.append((edge[0], node[0]))
                    visited.append(edge[0])
                else: 
                    return False
    
    if unvisited:
        print("BROKEN BC UNVISITED" + str(unvisited))
        return False
    else:
        return True
```

## DFS Traversal (Simple)


If the number of edges does not equal to number of vertices - 1, return false. 

If number of edges condition satisfied, check if connected component. 

Construct graph, then run DFS on any node to check if it is a connected component.

If connected component, return true. Else, return false

Time: O(n)
Space: O(n)

# 266. Palindrome Permutation
Easy

Given a string s, return true if a permutation of the string could form a palindrome.

 

Example 1:

Input: s = "code"
Output: false

Example 2:

Input: s = "aab"
Output: true

Example 3:

Input: s = "carerac"
Output: true

 

Constraints:

    1 <= s.length <= 5000
    s consists of only lowercase English letters.



## Hashmap

- Pop element if already exists in set
- Iterate over every element
    - If exists in set, pop
    - If doesn't add
- If set size is 1 or less, valid pal

Time: O(n)
Space: O(n)

```
from collections import Counter
def solution(s):
    my_dic = Counter(s)
    odd_count = 0 
    for _,value in my_dic.items():
        odd_count += value%2
        if odd_count>=2:
            return False
    return True
```

# 268. Missing Number
Easy

Given an array nums containing n distinct numbers in the range [0, n], return the only number in the range that is missing from the array.

 

Example 1:

Input: nums = [3,0,1]
Output: 2
Explanation: n = 3 since there are 3 numbers, so all numbers are in the range [0,3]. 2 is the missing number in the range since it does not appear in nums.

Example 2:

Input: nums = [0,1]
Output: 2
Explanation: n = 2 since there are 2 numbers, so all numbers are in the range [0,2]. 2 is the missing number in the range since it does not appear in nums.

Example 3:

Input: nums = [9,6,4,2,3,5,7,0,1]
Output: 8
Explanation: n = 9 since there are 9 numbers, so all numbers are in the range [0,9]. 8 is the missing number in the range since it does not appear in nums.

 

## Sorting 
Sort the array, and iterate from from from to last. If any prev value != curr value, return curr value - 1. 

Space O(N)
Time O(NlogN)

## Iteration with Array
Keep a boolen array set to all false. Iterate through the current array, and set the corresponding index in the boolean array to true. 

Return the index of the boolean array equal to False. 

Space O(N)
Time O(N)

## Iteration with Set

Keep a set of all the elements previous seen in a set. Iterate through the indices from [0, n], and return the value of the index that is not in the set. 

Space O(N)
Time O(N)

## Discrete Math 

Apply Gauss's formula to find the num of values from 1 to n. Then subtract the sum of the array from the total sum to find the missing number. 

Space: O(1)
Time: O(N)

## XOR (Optimal)

XOR all the elements in the list with another set of values indices from [0, n]. The remaning number is the expected number. This is since a xor a = 0.

A B XOR(A, B)
1 1 0
1 0 1
0 1 1
0 0 0 

note: 1 xor 1 = 0 and 0 xor 0 = 0. 

Space O(1)
Time O(N)

```
def solution(nums):
    val = 0
    for n in nums:
        val = val ^ n
    for index in range(len(nums) + 1):
        val = val ^ index
    return val
```

# 269. Alien Dictionary
Hard

There is a new alien language that uses the English alphabet. However, the order among the letters is unknown to you.

You are given a list of strings words from the alien language's dictionary, where the strings in words are sorted lexicographically by the rules of this new language.

Return a string of the unique letters in the new alien language sorted in lexicographically increasing order by the new language's rules. If there is no solution, return "". If there are multiple solutions, return any of them.

A string s is lexicographically smaller than a string t if at the first letter where they differ, the letter in s comes before the letter in t in the alien language. If the first min(s.length, t.length) letters are the same, then s is smaller if and only if s.length < t.length.

 

Example 1:

Input: words = ["wrt","wrf","er","ett","rftt"]
Output: "wertf"

Example 2:

Input: words = ["z","x"]
Output: "zx"

Example 3:

Input: words = ["z","x","z"]
Output: ""
Explanation: The order is invalid, so return "".


## Topological Sort

Create a list of letter dependenices. 

Iterate through the list of any two neighboring words. If the two values are not the same, then add the following depenency [1st word's mismatched letter, 2nd word's mismatched letter]. Then proceed to the next to adjacent words and repeat the process, until all words have been searched. A maxmimum of 1 dependency can be added per dependency. 

If a previous word has all the same letters as the current word, and some extra letters afterwards, no valid orderings can exist. 

Append all words into string, then perform counter to get all the unique letters. 

Perform DFS on each letter. Keep track of visited nodes using an array initialized to zeros. If an array is visited, set it to 1. If all the neighbors of an array has been visited, set visited to 2. This ensures after a DFS iteration is complete, only 0's and 2's remain in the visited array, so the next DFS won't fail due to visited nodes from a previous DFS. Instead, the DFS will skip over the 2's and change the existing 0's to 1's -- when this DFS is done the visited array will only have 0's and 2's. 

Time: O(E + V)
Space: O(E + V)

```
from collections import Counter
def solution(words):
    def compare(word1, word2):
        nonlocal letters
        nonlocal invalid
        for index in range(min(len(word1), len(word2))):
            letter1 = word1[index]
            letter2 = word2[index]

            if letter1 != letter2: 
                letters[letter1].append(letter2)
                return 
        
        if len(word1) > len(word2):
            invalid = True

    invalid = False

    letters = ""
    for word in words:
        letters += word
    
    letters = Counter(letters)

    for letter in letters:
        letters[letter] = []

    prev = ""
    for word in words: 
        compare(prev, word)
        prev = word
        if invalid: 
            return ""

    noTopo = False
    ordering = ""

    def search(letter):
        nonlocal ordering
        nonlocal noTopo
        if visited[letter] == 1:
            noTopo = True
            return 
        if visited[letter] == 0: 
            visited[letter] = 1
            neighbors = letters[letter]
            for neighbor in neighbors:
                search(neighbor)
            visited[letter] = 2
            ordering = letter + ordering
        pass

    visited = {}
    for letter in letters:
        visited[letter] = 0

    for letter in letters: 
        search(letter)
        if noTopo:
            return ""
    
    return ordering
```

# 270. Closest Binary Search Tree Value
Easy

Given the root of a binary search tree and a target value, return the value in the BST that is closest to the target.

 

Example 1:

Input: root = [4,2,5,1,3], target = 3.714286
Output: 4

Example 2:

Input: root = [1], target = 4.428571
Output: 1

 

Constraints:

    The number of nodes in the tree is in the range [1, 104].
    0 <= Node.val <= 109
    -109 <= target <= 109



## DFS Traversal

Compute the score of the current value in the tree, with the target, if the  distance is less than minSoFar, set minSoFar and minSoFar value

Time: O(n)
Space: O(n)

```
from math import inf
target = 0 
def solution(root):
	global currTarget 
	global minSoFarDist
	global minSoFarVal 
	currTarget = target
	minSoFarDist = inf
	minSoFarVal = inf 


	def traverse(root):
		global currTarget 
		global minSoFarDist
		global minSoFarVal 

		if root is None:
			return
		else: 
			distance = abs(currTarget - root.val)
			if distance < minSoFarDist: 
				minSoFarDist = distance 
				minSoFarVal = root.val
				
			traverse(root.left)
			traverse(root.right)

	traverse(root)
	return minSoFarVal 



```

# 271. Encode and Decode Strings
Medium

Design an algorithm to encode a list of strings to a string. The encoded string is then sent over the network and is decoded back to the original list of strings.

Machine 1 (sender) has the function:

string encode(vector<string> strs) {
  // ... your code
  return encoded_string;
}

Machine 2 (receiver) has the function:

vector<string> decode(string s) {
  //... your code
  return strs;
}

So Machine 1 does:

string encoded_string = encode(strs);

and Machine 2 does:

vector<string> strs2 = decode(encoded_string);

strs2 in Machine 2 should be the same as strs in Machine 1.

Implement the encode and decode methods.

You are not allowed to solve the problem using any serialize methods (such as eval).

## Brute Force

Append each string with a space delimiter (or any non-ASCII character). Break the string down using the spaces afterwards. 

Space: O(N*S) s.t. S is max string length
Encode Space: O(1)

Time: O(N)

## String Length Encoding (optimal)

Store string length into a 4 character string. Prepend the length to the string, and add both to the serialization. 

Decode Space: O(N*S) s.t. S is max string length
Encode Space: O(1)

Time: O(N*S)

```
def solutionEncode(strs):
    encoding = ""
    for string in strs: 
        length = str(len(string))
        encoding += length + " " + string
    return encoding
    
def soutionDecode(s): 
    decoding = []
    while s: 
        index = s.find(' ')
        stringLength = int(s[:index])
        string = s[index + 1: index + 1 + stringLength]
        decoding.append(string)
        s = s[index + 1 + stringLength:]
    return decoding
```

# 278. First Bad Version
Easy

You are a product manager and currently leading a team to develop a new product. Unfortunately, the latest version of your product fails the quality check. Since each version is developed based on the previous version, all the versions after a bad version are also bad.

Suppose you have n versions [1, 2, ..., n] and you want to find out the first bad one, which causes all the following ones to be bad.

You are given an API bool isBadVersion(version) which returns whether version is bad. Implement a function to find the first bad version. You should minimize the number of calls to the API.

 

Example 1:

Input: n = 5, bad = 4
Output: 4
Explanation:
call isBadVersion(3) -> false
call isBadVersion(5) -> true
call isBadVersion(4) -> true
Then 4 is the first bad version.
Example 2:

Input: n = 1, bad = 1
Output: 1
 

Constraints:

1 <= bad <= n <= 231 - 1

## Binary Search 

Since the array of products are in sorted order, with all good products before the bad products, this problem can be though of performing binary search on an array of 0's followed by 1's, then finding the first 1. 

If the midpoint of the search is a 1, this means the first '1' is to the left of the midpoint. If the midpoint of the search is 0, this means the first '1' is to the right of the midpoint. 

Time: O(logN)
Space: O(1)

```
def firstBadVersion(self, n: int) -> int:
    l = 1
    r = n 

    while l < r: 
        mid = (l + r)//2
        midVal = isBadVersion(mid)
        if midVal == True:
            r = mid 
        else: 
            l = mid + 1
    
    return l 
```


# 282. Expression Add Operators
Hard

Given a string num that contains only digits and an integer target, return all possibilities to insert the binary operators '+', '-', and/or '*' between the digits of num so that the resultant expression evaluates to the target value.

Note that operands in the returned expressions should not contain leading zeros.

 

Example 1:

Input: num = "123", target = 6
Output: ["1*2*3","1+2+3"]
Explanation: Both "1*2*3" and "1+2+3" evaluate to 6.

Example 2:

Input: num = "232", target = 8
Output: ["2*3+2","2+3*2"]
Explanation: Both "2*3+2" and "2+3*2" evaluate to 8.

Example 3:

Input: num = "3456237490", target = 9191
Output: []
Explanation: There are no expressions that can be created from "3456237490" to evaluate to 9191.

 

Constraints:

    1 <= num.length <= 10
    num consists of only digits.
    -231 <= target <= 231 - 1



## Backtrack

Keep track of index, prev_operand, curr_operant, valuesoFar, stringsofar and keep answers list outside
- Base case: Index == Len of string
    - If valueSoFar == targent and curr_operand == 0
        - NOTE: curr is set to zero in each recursive call
        - add to answer list the joined string values
- Reset curr_operand to include next digit
- store curr_opreand in str_operand
- Check if curr_operand is > 0
    - Recurse with next index, keeping prev operand, curr operand, value and string the same
- Addition:
    - Add + string and str operand
    - recurse with next index, prev operand is curr operand, curr operand is 0, value is curr operand + valuesofar, string
    - Remove elements from string stack
- If string stack is not empty
    - Subtraction
        - Add op and str operand to string stack
        - Recurse with next index, prev open is negative of curr operand, curr operand is 0, substract from value the current operand, and use same string stack
        - pop string stack
    - Multiplication
        - Append op and str operand 
        - Recurse using next index, prev_operand is product of curr and prev, curr is 0, value is value - prev + curr * prev, string
        - pop string stack

Time: (4^n)
Space: O(n)

```
def solution(num, target):
    n = len(num)
    ans = []
    def backtrack(index, prev, curr, valsofar, stack):
        if index == n:
            if valsofar == target and curr == 0:
                ans.append("".join(stack[1:]))
            return 
        
        curr = curr*10 + int(num[index])
        strop = str(curr)
        
        if curr > 0:
            backtrack(index+1, prev, curr, valsofar, stack)
        stack.append('+'); stack.append(strop)
        backtrack(index+1, curr, 0, curr + valsofar, stack)
        stack.pop(); stack.pop()
        if stack:
            stack.append('-'); stack.append(strop)
            backtrack(index+1, -curr, 0, valsofar - curr, stack)
            stack.pop(); stack.pop()
            
            stack.append("*"); stack.append(strop)
            backtrack(index+1, curr * prev, 0, valsofar - prev + (curr * prev), stack)
            stack.pop(); stack.pop()
    backtrack(0, 0, 0, 0, [])
    return ans
                
```

# 283. Move Zeroes
Easy
12.5K
317
company
Facebook
company
Yandex
company
Bloomberg
Given an integer array nums, move all 0's to the end of it while maintaining the relative order of the non-zero elements.

Note that you must do this in-place without making a copy of the array.

 

Example 1:

Input: nums = [0,1,0,3,12]
Output: [1,3,12,0,0]
Example 2:

Input: nums = [0]
Output: [0]
 

Constraints:

1 <= nums.length <= 104
-231 <= nums[i] <= 231 - 1
 

Follow up: Could you minimize the total number of operations done?

## Two Pointer 

Sinking the non-zero values to the front requires O(1) swap per non-zero value, when taking a two pointer approach. Keep a left pointer, that keeps track of the index of the last non-zero value swapped to the left of the array. The right index is used to iterate through the array, and is used to detect a non-zero value to swap with the left pointer. By ensuring all non-zero values are swapped in order to the left, the remaining values will only be 0's to the right. 

Sinking the zeros to the back, requires O(n) swap per 0 to push them to the back, to ensure the order of the non-zero values are preserved.

Time: O(n)
Space: O(1)

```
def moveZeroes(self, nums: List[int]) -> None:
    l = 0 
    r = 0 
    while r < len(nums): 
        if nums[r] != 0: 
            nums[r], nums[l] = nums[l], nums[r]
            l += 1
        r += 1
    return nums
```

# 285. Inorder Successor in BST
Medium

Given the root of a binary search tree and a node p in it, return the in-order successor of that node in the BST. If the given node has no in-order successor in the tree, return null.

The successor of a node p is the node with the smallest key greater than p.val.

 

Example 1:


Input: root = [2,1,3], p = 1
Output: 2
Explanation: 1's in-order successor node is 2. Note that both p and the return value is of TreeNode type.
Example 2:


Input: root = [5,3,6,2,4,null,null,1], p = 6
Output: null
Explanation: There is no in-order successor of the current node, so the answer is null.
 

Constraints:

The number of nodes in the tree is in the range [1, 104].
-105 <= Node.val <= 105
All Nodes will have unique values.

## Recursion 

Utilize that an in-order search on an BST, iterates over the values of the tree in sorted order. Therefore, in each step of iteration, the previous node is a predcessors (next smallest element) and the next node is the success (next greatest element). When searching the tree, return the current node, when the previous tracked node is the target node, which implied the current node is the success of the target node. 

Time: O(n)
Space: O(n)

```
def inorderSuccessor(self, root: TreeNode, p: TreeNode) -> Optional[TreeNode]:
    self.prev = None
    self.ans = None 

    def search(root):
        if not root: 
            return 
        else:
            search(root.left)
            if self.prev == p: 
                self.ans = root
            self.prev = root
            search(root.right)

    search(root)
    return self.ans
```

## Iteration 

If the node is less or equal than the target node, both that node and all the nodes to it's left are also less than of equal to the target node, so we can prune those values out of the solution. 

If the current node is greater than the target node, it could be a possible succesor node. 

Time: O(n)
Space: O(1)

```
def inorderSuccessor(self, root: TreeNode, p: TreeNode) -> Optional[TreeNode]:
    curr = root
    succ = None 
    while curr:  
        if curr.val <= p.val: 
            curr = curr.right 
        elif curr.val > p.val: 
            succ = curr
            curr = curr.left 
    
    return succ
```

# 286. Walls and Gates
Medium

You are given an m x n grid rooms initialized with these three possible values.

    -1 A wall or an obstacle.
    0 A gate.
    INF Infinity means an empty room. We use the value 231 - 1 = 2147483647 to represent INF as you may assume that the distance to a gate is less than 2147483647.

Fill each empty room with the distance to its nearest gate. If it is impossible to reach a gate, it should be filled with INF.

 

Example 1:

Input: rooms = [[2147483647,-1,0,2147483647],[2147483647,2147483647,2147483647,-1],[2147483647,-1,2147483647,-1],[0,-1,2147483647,2147483647]]
Output: [[3,-1,0,1],[2,2,1,-1],[1,-1,2,-1],[0,-1,3,4]]

Example 2:

Input: rooms = [[-1]]
Output: [[-1]]

 

Constraints:

    m == rooms.length
    n == rooms[i].length
    1 <= m, n <= 250
    rooms[i][j] is -1, 0, or 231 - 1.



## BFS Traversal 

- Perform bfs from each gate, and store distance inside the graph
    - Check all neighbors, keep track of visited
- If distance is less than existing distance replace

Space: O(n)
Time: O(n)

```
from collections import deque

def solution(rooms):
    neighbors = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    rows = len(rooms)
    cols = len(rooms[0])
    def bfs(node):
        q = deque()
        q.append((node, 0))
        while q:
            e, d = q.popleft()
            for n in neighbors:
                x = e[0] + n[0]; y = e[1] + n[1]
                if 0 <= x < rows and 0 <= y < cols and rooms[x][y] != 1 and d + 1 < rooms[x][y]:
                    rooms[x][y] = d + 1
                    q.append(((x, y), d + 1))
                    
    
    for i in range(len(rooms)):
        for j in range(len(rooms[0])):
            if rooms[i][j] == 0:
                bfs((i, j))
```

# 287. Find the Duplicate Number
Medium

Given an array of integers nums containing n + 1 integers where each integer is in the range [1, n] inclusive.

There is only one repeated number in nums, return this repeated number.

You must solve the problem without modifying the array nums and uses only constant extra space.

 

Example 1:

Input: nums = [1,3,4,2,2]
Output: 2
Example 2:

Input: nums = [3,1,3,4,2]
Output: 3
 

Constraints:

1 <= n <= 105
nums.length == n + 1
1 <= nums[i] <= n
All the integers in nums appear only once except for precisely one integer which appears two or more times.
 

Follow up:

How can we prove that at least one duplicate number must exist in nums?
Can you solve the problem in linear runtime complexity?



## Hashmap 

Time: O(n)
Space: O(n)

## Sorting 

Time: O(nlogn)
Space: O(1)

## Fast and Slow Pointer

Understanding the reduction from multiple duplicates to the linked list cycle start point detction problem is key. 

Understanding why the two phase process of determining the linked list cycle start point is also important. 

Time: O(n)
Space: O(1)

```
def findDuplicate(self, nums: List[int]) -> int:
    fast = nums[0]
    slow = nums[0] 

    while True: #fast != slow: 
        fast = nums[fast]
        fast = nums[fast]
        slow = nums[slow]
        if slow == fast: 
            break 

    intersection = slow 
    
    slow = intersection 
    fast = nums[0]

    while True: 
        if fast == slow: 
            break 
        fast = nums[fast]
        slow = nums[slow]
    return slow
```

# 295. Find Median from Data Stream
Hard

The median is the middle value in an ordered integer list. If the size of the list is even, there is no middle value and the median is the mean of the two middle values.

    For example, for arr = [2,3,4], the median is 3.
    For example, for arr = [2,3], the median is (2 + 3) / 2 = 2.5.

Implement the MedianFinder class:

    MedianFinder() initializes the MedianFinder object.
    void addNum(int num) adds the integer num from the data stream to the data structure.
    double findMedian() returns the median of all elements so far. Answers within 10-5 of the actual answer will be accepted.

 

Example 1:

Input
["MedianFinder", "addNum", "addNum", "findMedian", "addNum", "findMedian"]
[[], [1], [2], [], [3], []]
Output
[null, null, null, 1.5, null, 2.0]

Explanation
MedianFinder medianFinder = new MedianFinder();
medianFinder.addNum(1);    // arr = [1]
medianFinder.addNum(2);    // arr = [1, 2]
medianFinder.findMedian(); // return 1.5 (i.e., (1 + 2) / 2)
medianFinder.addNum(3);    // arr[1, 2, 3]
medianFinder.findMedian(); // return 2.0

Follow up:

    If all integer numbers from the stream are in the range [0, 100], how would you optimize your solution?
    If 99% of all integer numbers from the stream are in the range [0, 100], how would you optimize your solution?

## Insertion Sort

Add the numbers in the proper order in a list. This will allow the list to remain sorted as more numbers are added.

Return the value at the index of the middle value or the average of the two middle values. 

Time: O(n)
Space: O(n) because new array is being created

## Two Heaps (Optimal)

Keep a min heap and a max heap. The max heap store all the smaller values. The min heap stores all the larger values. At any moment, the max heap, can have 1 more element than the min heap. 

For each new element first add it to the max heap. Then remove the largest max heap element and add it to the min heap. This is to ensure that all the elements in the max heap are less than all the elements in the min heap. 

If the min heap becomes larger than the max heap, swap the smallest element of the min heap back to the max heap. This preserves the condition that the max heap is equal to or has one more element than the min heap. 

If the max heap has more elements, return the first max heap element. If both heaps have the same number of values, return the average of both first elements.

Heapq in python is a minheap. Therefore, always negate the elements for the maxHeap, which converts the minheap effectively into a maxheap. 

Time: O(logn)
Space: O(n)

```
import heapq

def solutionInit(self):
    self.minHeap = []
    self.maxHeap = []

def solutionAddNum(self, num):
    heapq.heappush(self.maxHeap, -num)
    
    swap = heapq.heappop(self.maxHeap)
    heapq.heappush(self.minHeap, -swap)


    if len(self.maxHeap) < len(self.minHeap):
        swap = heapq.heappop(self.minHeap)
        heapq.heappush(self.maxHeap, -swap)


def findMedian(self):
    if len(self.maxHeap) > len(self.minHeap): 
            return -self.maxHeap[0]
    else:
        return (-self.maxHeap[0] + self.minHeap[0])/2
```

# 297. Serialize and Deserialize Binary Tree
Hard

Serialization is the process of converting a data structure or object into a sequence of bits so that it can be stored in a file or memory buffer, or transmitted across a network connection link to be reconstructed later in the same or another computer environment.

Design an algorithm to serialize and deserialize a binary tree. There is no restriction on how your serialization/deserialization algorithm should work. You just need to ensure that a binary tree can be serialized to a string and this string can be deserialized to the original tree structure.

Clarification: The input/output format is the same as how LeetCode serializes a binary tree. You do not necessarily need to follow this format, so please be creative and come up with different approaches yourself.

 

Example 1:

Input: root = [1,2,3,null,null,4,5]
Output: [1,2,3,null,null,4,5]

Example 2:

Input: root = []
Output: []

## DFS Traversal (Optimal)

Perform preorder traversal, and store the node values including null children as a comma separated list. Reconstruct the tree from this artifact. 

Store the values in the format root -> left child -> right chil recursively, where missing children are saved as None.

Store the values as string in a comma separated list for serialization. 

Deserialize by splitting list by commas. Then rebuild list by popping off first element one at a time recursively. 

Space: O(N)
Time: O(N)

```
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def solutionSerialize(root):
    tree = []
    def traverse(root):
        if not root: 
            tree.append(None)
        else:
            tree.append(root.val)
            traverse(root.left)
            traverse(root.right)
    traverse(root)
    serialization = ",".join([str(i) for i in tree])
    return serialization
        
def solutionDeserialize(data):
    deserialized = data.split(",")
    def build(data):
        if data[0] == 'None':
            data.pop(0)
            return 
        val = data.pop(0)
        tree = TreeNode(val)
        tree.left = build(data)
        tree.right = build(data)
        return tree
    tree = build(deserialized)
    return tree
```

# 300. Longest Increasing Subsequence
Medium

Given an integer array nums, return the length of the longest strictly increasing subsequence.

A subsequence is a sequence that can be derived from an array by deleting some or no elements without changing the order of the remaining elements. For example, [3,6,2,7] is a subsequence of the array [0,3,1,6,2,2,7].

 

Example 1:

Input: nums = [10,9,2,5,3,7,101,18]
Output: 4
Explanation: The longest increasing subsequence is [2,3,7,101], therefore the length is 4.

Example 2:

Input: nums = [0,1,0,3,2,3]
Output: 4

Example 3:

Input: nums = [7,7,7,7,7,7,7]
Output: 1


## Brute Force
1. Iterate through from each start index
2. Iterate from start index to end of array
3. Keep track of longest increasing sequence using a counter
4. Return max of all counters

## Dynamic Programming

State variable = longest subsequence up until index.

The DP Table contains the length of the longest increasing subsequence up until each index.

Keep a DP table inialized to 1's, since each element, is by default a sequence of length 1.  

Step 1:  Iterate through the given array, setting the corresponding dp[i] = dp[i-1] + 1 if the current value is greater than the previous value. 

 
Step 2: For each element before the current index, if the current value is greater than that value, check if table[j] + 1> table[i], where i is the current index. 

If so, set table[i] to table[j] + 1. 

[1, 2, 3, 1, 2, 1, 1, 5]

Time: O(N)
Space: O(N)

```
def solution(nums):
    dp = [1 for i in range(len(nums))]
    for index in range(len(nums)):
        n = nums[index]
        for subindex in range(index):
            if n > nums[subindex]:
                dp[index] = max(dp[index], dp[subindex] + 1)
    return max(dp)

def main():
    #test = [3, 4, 1, 2, 3]
    test = [0,1,0,3,2,3]
    ans = solution(test)
    print(ans)
```

# 301. Remove Invalid Parentheses
Hard

Given a string s that contains parentheses and letters, remove the minimum number of invalid parentheses to make the input string valid.

Return all the possible results. You may return the answer in any order.

 

Example 1:

Input: s = "()())()"
Output: ["(())()","()()()"]

Example 2:

Input: s = "(a)())()"
Output: ["(a())()","(a)()()"]

Example 3:

Input: s = ")("
Output: [""]

 

Constraints:

    1 <= s.length <= 25
    s consists of lowercase English letters and parentheses '(' and ')'.
    There will be at most 20 parentheses in s.



## Backtrack

- Keep a set of feasible strings
- Recurse
    - If index reaches end and number of ( == ) or counter is 0, then add to answer set
    - If current is not ( or ), add to stack
    - If ) decrement counter, if ( increment counter
        - If counter zero and curr is ), then increment index
    - Keep stack of valid sequences so far
    - Pass stack, curr index, and counter into recursive call

Time: O(2^n)
Space: O(n)

```
from math import inf

def solution(s):
    results = set()
    minRemove = inf
    def backtrack(stack, i, c, r):
        stack = stack.copy()
        nonlocal minRemove
        if i == len(s):
            if c == 0:
                if r <= minRemove:
                    minRemove = r
                    results.add("".join(stack))
            return
        if s[i] not in '()':
            stack.append(s[i])
            i += 1
            backtrack(stack, i, c, r)
        elif s[i] == ')':
            if c == 0:
                i += 1
                backtrack(stack, i, c, r + 1)
            else:
                stack.append(s[i])
                c -= 1
                i += 1
                backtrack(stack, i, c, r)
                stack.pop()
                c += 1
                backtrack(stack, i, c, r + 1)
        elif s[i] == '(':
            stack.append(s[i])
            c += 1
            i += 1
            backtrack(stack, i, c, r)
            stack.pop()
            c -= 1
            backtrack(stack, i, c, r + 1)
    backtrack([], 0, 0, 0)
    return results
    
```

# 310. Minimum Height Trees
Medium

A tree is an undirected graph in which any two vertices are connected by exactly one path. In other words, any connected graph without simple cycles is a tree.

Given a tree of n nodes labelled from 0 to n - 1, and an array of n - 1 edges where edges[i] = [ai, bi] indicates that there is an undirected edge between the two nodes ai and bi in the tree, you can choose any node of the tree as the root. When you select a node x as the root, the result tree has height h. Among all possible rooted trees, those with minimum height (i.e. min(h))  are called minimum height trees (MHTs).

Return a list of all MHTs' root labels. You can return the answer in any order.

The height of a rooted tree is the number of edges on the longest downward path between the root and a leaf.

 

Example 1:


Input: n = 4, edges = [[1,0],[1,2],[1,3]]
Output: [1]
Explanation: As shown, the height of the tree is 1 when the root is the node with label 1 which is the only MHT.
Example 2:


Input: n = 6, edges = [[3,0],[3,1],[3,2],[3,4],[5,4]]
Output: [3,4]
 

Constraints:

1 <= n <= 2 * 104
edges.length == n - 1
0 <= ai, bi < n
ai != bi
All the pairs (ai, bi) are distinct.
The given input is guaranteed to be a tree and there will be no repeated edges.

## Topological Sort ~ BFS

Remove all the remaining leaf nodes, until the number of nodes remianing is <= 2.

```
```

## DFS Twice 

Perform DFS from any two leaves similar to two pointers on a graph. When the two pointer are on the same node or are two nodes away from each other, then return the node(s).

```
```

# 314. Binary Tree Vertical Order Traversal
Medium

Given the root of a binary tree, return the vertical order traversal of its nodes' values. (i.e., from top to bottom, column by column).

If two nodes are in the same row and column, the order should be from left to right.

 

Example 1:

Input: root = [3,9,20,null,null,15,7]
Output: [[9],[3,15],[20],[7]]

Example 2:

Input: root = [3,9,8,4,0,1,7]
Output: [[4],[9],[3,0,1],[8],[7]]

Example 3:

Input: root = [3,9,8,4,0,1,7,null,null,null,2,5]
Output: [[4],[9,5],[3,0,1],[8,2],[7]]

 

Constraints:

    The number of nodes in the tree is in the range [0, 100].
    -100 <= Node.val <= 100



## BFS Traversal

- BFS + Keep track of width
- Add elements of same width to a dictionary
- Return elements of dictionary into a list in order

Space: O(N)
Time: O(N)

```
from collections import deque
from collections import defaultdict

def solution(root):
    if not root:
        return []
    d = defaultdict(list)
    q = deque()
    q.append((0, root))
    m = 0
    while q:
        w, v = q.popleft()
        d[w].append(v.val)
        if v.left: 
            q.append((w-1, v.left)) 
            m = min(w-1, m)
        if v.right: q.append((w+1, v.right))
    ans = []

    for i in range(m, m + len(d)):
        ans.append(d[i])
    return ans
```

# 317. Shortest Distance from All Buildings
Hard

You are given an m x n grid grid of values 0, 1, or 2, where:

    each 0 marks an empty land that you can pass by freely,
    each 1 marks a building that you cannot pass through, and
    each 2 marks an obstacle that you cannot pass through.

You want to build a house on an empty land that reaches all buildings in the shortest total travel distance. You can only move up, down, left, and right.

Return the shortest travel distance for such a house. If it is not possible to build such a house according to the above rules, return -1.

The total travel distance is the sum of the distances between the houses of the friends and the meeting point.

The distance is calculated using Manhattan Distance, where distance(p1, p2) = |p2.x - p1.x| + |p2.y - p1.y|.

 

Example 1:

Input: grid = [[1,0,2,0,1],[0,0,0,0,0],[0,0,1,0,0]]
Output: 7
Explanation: Given three buildings at (0,0), (0,4), (2,2), and an obstacle at (0,2).
The point (1,2) is an ideal empty land to build a house, as the total travel distance of 3+3+1=7 is minimal.
So return 7.

Example 2:

Input: grid = [[1,0]]
Output: 1

Example 3:

Input: grid = [[1]]
Output: -1

 

Constraints:

    m == grid.length
    n == grid[i].length
    1 <= m, n <= 50
    grid[i][j] is either 0, 1, or 2.
    There will be at least one building in the grid.



## BFS Traversal

- For each grid with 1 -- bfs
    - If neighbors is <= 0, decrement value by 1, increment dictionary by distance travelled to reach, add to visited, and add neighbors to queue
    - Continue until queue nonempty
- Reset visited
- For each grid value that is equal to -1 * number of buildings - return largest size

Time: O(N^4)
Space: O(N^2)

```
from collections import defaultdict
from collections import deque

def solution(grid):
    visited = set()
    dist = defaultdict(int)
    neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    def bfs(node):
        nonlocal neighbors, minDistance
        visited.clear()
        q = deque()
        q.append((node, 0))

        p = q.popleft()
        e = p[0]; d = p[1]

        for n in neighbors:
                newx = e[0] + n[0]; newy = e[1] + n[1]
                if 0 <= newx < len(grid) and 0 <= newy < len(grid[0]):
                    if grid[newx][newy] == -buildings and (newx, newy) not in visited:
                        visited.add((newx, newy))
                        q.append(((newx, newy), d + 1))
        while q:
            p = q.popleft()
            e = p[0]; d = p[1]
            grid[e[0]][e[1]] -= 1
            dist[e] += d
            minDistance = min(minDistance, dist[e])
            for n in neighbors:
                newx = e[0] + n[0]; newy = e[1] + n[1]
                if 0 <= newx < len(grid) and 0 <= newy < len(grid[0]):
                    if grid[newx][newy] == -buildings and (newx, newy) not in visited:
                        visited.add((newx, newy))
                        q.append(((newx, newy), d + 1))
        return minDistance
                
    buildings = 0          
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j] == 1:
                minDistance = inf
                minDistance = bfs((i, j))
                buildings += 1

    if minDistance == inf:
        return -1
    return minDistance
```

# 322. Coin Change
Medium

You are given an integer array coins representing coins of different denominations and an integer amount representing a total amount of money.

Return the fewest number of coins that you need to make up that amount. If that amount of money cannot be made up by any combination of the coins, return -1.

You may assume that you have an infinite number of each kind of coin.



Example 1:

Input: coins = [1,2,5], amount = 11
Output: 3
Explanation: 11 = 5 + 5 + 1

Example 2:

Input: coins = [2], amount = 3
Output: -1

Example 3:

Input: coins = [1], amount = 0
Output: 0


## Dynamic Programming

Finding the minimum number of coins to add up to target requires us to solve the supproblems for all integers below target as well.

In particular, we find minimum coins for a target value but finding the minimum value of coins it took to reach all values reachable by subtracting a coin value. 

This gives us the dp relationship: dp[i] = min(dp[i], dp[i - coin] + 1) for all coin values. This is than mapped to a dp table and iteratively filled up solving the smallest to largest subproblems until we reach the target value. 

Time: O(n)
Space: O(n)

```
def coinChange(self, coins: List[int], amount: int) -> int:
    import math 
    dp = [math.inf for i in range(amount + 1)]
    dp[0] = 0
    for i in range(amount + 1):
        for coin in coins: 
            if i - coin >= 0:
                dp[i] = min(dp[i-coin] + 1, dp[i])

    if dp[-1] == math.inf: 
        return -1
    else: 
        return dp[-1]
```

Example 
```
coins = [1,2,5], amount = 11
```

# 323. Number of Connected Components in an Undirected Graph
Medium

You have a graph of n nodes. You are given an integer n and an array edges where edges[i] = [ai, bi] indicates that there is an edge between ai and bi in the graph.

Return the number of connected components in the graph.

 

Example 1:

Input: n = 5, edges = [[0,1],[1,2],[3,4]]
Output: 2

Example 2:

Input: n = 5, edges = [[0,1],[1,2],[2,3],[3,4]]
Output: 1

 

Constraints:

    1 <= n <= 2000
    1 <= edges.length <= 5000
    edges[i].length == 2
    0 <= ai <= bi < n
    ai != bi
    There are no repeated edges.

## DFS 

Repeat DFS at each unvisited node that is kept track in an array. 

```
def countComponents(self, n: int, edges: List[List[int]]) -> int:
    def search(node, graph):
        if visited[node] == 0: 
            visited[node] = 1
            for n in graph[node]:
                if visited[n] == 0: 
                    search(n, graph)

    graph = {}
    for i in range(n):
        graph[i] = []
    for e in edges: 
        a = e[0]
        b = e[1]
        graph[a].append(b)
        graph[b].append(a)

    visited = [0 for i in range(n)]
    
    count = 0
    for i in range(n):
        if visited[i] == 0: 
            search(i, graph)
            count += 1

    return count 
```

## Topological Sort

Repeated DFS with visited graph that stores [0, 1, 2]

Space: O(n)
Time: O(n)

```
def solution(n, edge):
    visited = [0 for _ in range(n)]
    graph = {}
    
    for node in range(n):
        graph[node] = []
        
    for edge in edges:
        graph[edge[0]].append(edge[1])
        graph[edge[1]].append(edge[0])        
    
    def traverse(node):
        if visited[node] == 1:
            return 
        if visited[node] == 0:
            visited[node] = 1
            for neighbor in graph[node]:
                traverse(neighbor)
                visited[node] = 2
            
    count = 0
    for index in range(len(visited)): 
        if visited[index] == 0:
            count += 1
            traverse(index)

    return count
```

# 325. Maximum Size Subarray Sum Equals k
Medium

Given an integer array nums and an integer k, return the maximum length of a subarray that sums to k. If there is not one, return 0 instead.

 

Example 1:

Input: nums = [1,-1,5,-2,3], k = 3
Output: 4
Explanation: The subarray [1, -1, 5, -2] sums to 3 and is the longest.

Example 2:

Input: nums = [-2,-1,2,1], k = 1
Output: 2
Explanation: The subarray [-1, 2] sums to 1 and is the longest.

 

Constraints:

    1 <= nums.length <= 2 * 105
    -104 <= nums[i] <= 104
    -109 <= k <= 109



## Iteration + Hashmap (Optimal)

- Set max to -inf
- Prefix sum
- Keep hashmap to store sum's index
    - keep track of duplicates, only keep first occurence of sum
- If sum - k in hashmap, compute diff current index - sum - k's index
    - if diff > max, increment max

Space: O(n)
Time: O(1)

```
from math import inf 

def solution(nums, k):
    prefixes = {0:-1}
    maxsize = -inf
    total = 0
    for i in range(len(nums)):
        total += nums[i]
        if total not in prefixes:
            prefixes[total] = i
        if total - k in prefixes:
            maxsize = max(maxsize, i - prefixes[total-k])
    if maxsize == -inf:
        return 0
    return maxsize
```

# 328. Odd Even Linked List
Medium

Given the head of a singly linked list, group all the nodes with odd indices together followed by the nodes with even indices, and return the reordered list.

The first node is considered odd, and the second node is even, and so on.

Note that the relative order inside both the even and odd groups should remain as it was in the input.

You must solve the problem in O(1) extra space complexity and O(n) time complexity.

 

Example 1:


Input: head = [1,2,3,4,5]
Output: [1,3,5,2,4]
Example 2:


Input: head = [2,1,3,5,6,4,7]
Output: [2,3,6,7,1,5,4]
 

Constraints:

The number of nodes in the linked list is in the range [0, 104].
-106 <= Node.val <= 106

## Two Pointer 

Update odd next first, then update even next afterwards. 

Odd next is the next of the first even next. The even next is the next of the updated odd next. 

Time: O(n)
Space: O(1)

```
def oddEvenList(self, head: Optional[ListNode]) -> Optional[ListNode]:
    if not head: 
        return head 

    oddHead = head 
    odd = head 
    even = odd.next 
    evenHead = even 

    while even and even.next: 
        odd.next = even.next 
        odd = odd.next 
        even.next = odd.next 
        even = even.next 
    
    odd.next = evenHead 
    return oddHead 
```

# 377. Combination Sum IV
Medium

Given an array of distinct integers nums and a target integer target, return the number of possible combinations that add up to target.

The test cases are generated so that the answer can fit in a 32-bit integer.

Example 1:

Input: nums = [1,2,3], target = 4
Output: 7
Explanation:
The possible combination ways are:
(1, 1, 1, 1)
(1, 1, 2)
(1, 2, 1)
(1, 3)
(2, 1, 1)
(2, 2)
(3, 1)
Note that different sequences are counted as different combinations.

Example 2:

Input: nums = [9], target = 3
Output: 0

## Dynamic Programming

This is an unbounded knapsack problem, since we can assume we have an unlimited amount of each item. In this case the knapsack is the target amount and the items in the knapsack are the number in the array nums. 

Since we are counting the total possible ways to reach the target amount, it is equal to the amount of value use one of the numbers in nums as the last step to reach the target amount. 

Therefore the following recursive relationship holds: `dp[i] = dp[i] + dp[i - n]` for every n in nums. Finally, the base case is that there is only 1 way to make a total sum of value 0. 

Together, we can use a dp table of size target + 1 to store the values of the solutions to the intermediate subproblems. 

Time: O(n)
Space: O(n)

Code 
```
def combinationSum4(self, nums: List[int], target: int) -> int:
    dp = [0 for i in range(target + 1)]
    dp[0] = 1
        
    for i in range(target + 1):
        for n in nums: 
            if i - n >= 0: 
                dp[i] += dp[i - n]

    return dp[-1]
```

Examples
```
nums = [1,2,3], target = 4
```


# 338. Counting Bits
Easy

Given an integer n, return an array ans of length n + 1 such that for each i (0 <= i <= n), ans[i] is the number of 1's in the binary representation of i.

 

Example 1:

Input: n = 2
Output: [0,1,1]
Explanation:
0 --> 0
1 --> 1
2 --> 10

Example 2:

Input: n = 5
Output: [0,1,1,2,1,2]
Explanation:
0 --> 0
1 --> 1
2 --> 10
3 --> 11
4 --> 100
5 --> 101

 

Constraints:

    0 <= n <= 105

 

Follow up:

    It is very easy to come up with a solution with a runtime of O(n log n). Can you do it in linear time O(n) and possibly in a single pass?
    Can you do it without using any built-in function (i.e., like __builtin_popcount in C++)?

## Dynamic Programming

Note the relationship between the binary values of n and n//2. Dividing a number by 2 and flooring the values essentially removes the last bit of the binary value. Therefore, a recursive relationship can be build using this relationship.

If n is a odd number, it essentially is the binary number of n//2 left shifted by 1 and adding a 1 bit. Likewise, if n is a even number, it is essentially the binary number of n//2 left shifted by 1 without addint the 1 bit. We can figure out if the current number if odd by performing a logical and with a 1 bit, which will return a 1 if odd and 0 if even. 

Together, we can create a dp table to build the values: dp[i] = dp[i//2] + dp[i & 1]

Time: O(n)
Space: O(n)

Code
```
def countBits(self, n: int) -> List[int]:
    dp = [0 for i in range(n + 1)]
    for i in range(n + 1):
        dp[i] = dp[int(i//2)] + (i & 1)
    return dp
```

Example
```
- n = 8 
    - [0, 0, 0, 0, 0, 0, 0, 0] Start
    - [0, 1, 0, 0, 0, 0, 0, 0] Power of Two = 1
    - [0, 1, 1, 2, 0, 0, 0, 0] Power of Two = 2
    - Power of Two = 4 
```

# 339. Nested List Weight Sum
Medium

You are given a nested list of integers nestedList. Each element is either an integer or a list whose elements may also be integers or other lists.

The depth of an integer is the number of lists that it is inside of. For example, the nested list [1,[2,2],[[3],2],1] has each integer's value set to its depth.

Return the sum of each integer in nestedList multiplied by its depth.

 

Example 1:

Input: nestedList = [[1,1],2,[1,1]]
Output: 10
Explanation: Four 1's at depth 2, one 2 at depth 1. 1*2 + 1*2 + 2*1 + 1*2 + 1*2 = 10.

Example 2:

Input: nestedList = [1,[4,[6]]]
Output: 27
Explanation: One 1 at depth 1, one 4 at depth 2, and one 6 at depth 3. 1*1 + 4*2 + 6*3 = 27.

Example 3:

Input: nestedList = [0]
Output: 0

 

Constraints:

    1 <= nestedList.length <= 50
    The values of the integers in the nested list is in the range [-100, 100].
    The maximum depth of any integer is less than or equal to 50.



## DFS Traversal

-Recursive sum
-If list, apply function to the element
-If not list, add to global sum multiplied by current depth
-Start with depth 1
-Use the given functions 

Space: O(N) 
Time: O(N)

```
def solution(nestedList):
    s = 0
    def helper(l, d):
        nonlocal s
        for i in l:
            if i.isInteger():
                s += i.getInteger() * d
            else:
                i = i.getList()
                helper(i, d + 1)

    helper(nestedList, 1)
    return s
```

# 346. Moving Average from Data Stream
Easy

Given a stream of integers and a window size, calculate the moving average of all integers in the sliding window.

Implement the MovingAverage class:

    MovingAverage(int size) Initializes the object with the size of the window size.
    double next(int val) Returns the moving average of the last size values of the stream.

 

Example 1:

Input
["MovingAverage", "next", "next", "next", "next"]
[[3], [1], [10], [3], [5]]
Output
[null, 1.0, 5.5, 4.66667, 6.0]

Explanation
MovingAverage movingAverage = new MovingAverage(3);
movingAverage.next(1); // return 1.0 = 1 / 1
movingAverage.next(10); // return 5.5 = (1 + 10) / 2
movingAverage.next(3); // return 4.66667 = (1 + 10 + 3) / 3
movingAverage.next(5); // return 6.0 = (10 + 3 + 5) / 3

 

Constraints:

    1 <= size <= 1000
    -105 <= val <= 105
    At most 104 calls will be made to next.



## Iteration + Queue

- Curr window size
- Max window size = 0
- Curr sum = 0
- Avg = curr sum / curr window size
- If reached max window size and add, then pop left on queue and add left queue

Space: O(n)
Time: O(n)

```
from collections import deque
class MovingAverage:
    def __init__(self, size):
        self.currwin = 0
        self.maxwin = size
        self.currsum = 0
        self.q = deque()
        
    def next(self, val):
        self.q.append(val)
        self.currsum += val
        self.currwin += 1
        if self.currwin > self.maxwin:
            last = self.q.popleft()
            self.currsum -= last
            self.currwin -= 1
        return self.currsum / self.currwin
```

# 347. Top K Frequent Elements
Medium

Given an integer array nums and an integer k, return the k most frequent elements. You may return the answer in any order.

 

Example 1:

Input: nums = [1,1,1,2,2,3], k = 2
Output: [1,2]

Example 2:

Input: nums = [1], k = 1
Output: [1]

 

Constraints:

    1 <= nums.length <= 105
    -104 <= nums[i] <= 104
    k is in the range [1, the number of unique elements in the array].
    It is guaranteed that the answer is unique.

 

Follow up: Your algorithm's time complexity must be better than O(n log n), where n is the array's size.


## Heap (Optimal)

Count the frequency of all elements using a hashmap. 

Add elements including their frequency inside a heap of size k. After elements overflow, first push the new element then and pop the old element. Finally return the heap as a list. 

Time: O(nlogk)
Space: O(N + k)

## QuickSelect 

Time: O(N^2) worst case, O(N) amortized
Space: O(N)

## Bucket Sort

Time: O(N)
Space: O(N)

```
from collections import Counter
import heapq

def solution(nums, k):
    heap = []
    counts = Counter(nums)
    for count in counts: 
        if len(heap) < k: 
            heapq.heappush(heap, (counts[count], count))
        else:
            heapq.heappush(heap, (counts[count], count))
            heapq.heappop(heap)
    return [h[1] for h in heap]

def solution(nums, k):
    freq = Counter(nums)
    num = list(freq.keys())
    k = len(num) - k
    def partition(l, r):
        p = (l + r)//2
        pv = freq[num[p]]
        num[p], num[r] = num[r], num[p]
        si = l
        for i in range(l, r):
            if freq[num[i]] <= pv:
                num[i], num[si] = num[si], num[i]
                si += 1
        num[r], num[si] = num[si], num[r]
        return si
    def select(l, r):
        p = len(num)
        while k != p:
            p = partition(l, r)
            if p < k:
                l = p + 1
            elif p > k:
                r = p - 1
    select(0, len(num)-1)
    return num[k:]
```

# 348. Design Tic-Tac-Toe
Medium

Assume the following rules are for the tic-tac-toe game on an n x n board between two players:

    A move is guaranteed to be valid and is placed on an empty block.
    Once a winning condition is reached, no more moves are allowed.
    A player who succeeds in placing n of their marks in a horizontal, vertical, or diagonal row wins the game.

Implement the TicTacToe class:

    TicTacToe(int n) Initializes the object the size of the board n.
    int move(int row, int col, int player) Indicates that the player with id player plays at the cell (row, col) of the board. The move is guaranteed to be a valid move.

 

Example 1:

Input
["TicTacToe", "move", "move", "move", "move", "move", "move", "move"]
[[3], [0, 0, 1], [0, 2, 2], [2, 2, 1], [1, 1, 2], [2, 0, 1], [1, 0, 2], [2, 1, 1]]
Output
[null, 0, 0, 0, 0, 0, 0, 1]

Explanation
TicTacToe ticTacToe = new TicTacToe(3);
Assume that player 1 is "X" and player 2 is "O" in the board.
ticTacToe.move(0, 0, 1); // return 0 (no one wins)
|X| | |
| | | |    // Player 1 makes a move at (0, 0).
| | | |

ticTacToe.move(0, 2, 2); // return 0 (no one wins)
|X| |O|
| | | |    // Player 2 makes a move at (0, 2).
| | | |

ticTacToe.move(2, 2, 1); // return 0 (no one wins)
|X| |O|
| | | |    // Player 1 makes a move at (2, 2).
| | |X|

ticTacToe.move(1, 1, 2); // return 0 (no one wins)
|X| |O|
| |O| |    // Player 2 makes a move at (1, 1).
| | |X|

ticTacToe.move(2, 0, 1); // return 0 (no one wins)
|X| |O|
| |O| |    // Player 1 makes a move at (2, 0).
|X| |X|

ticTacToe.move(1, 0, 2); // return 0 (no one wins)
|X| |O|
|O|O| |    // Player 2 makes a move at (1, 0).
|X| |X|

ticTacToe.move(2, 1, 1); // return 1 (player 1 wins)
|X| |O|
|O|O| |    // Player 1 makes a move at (2, 1).
|X|X|X|

 

Constraints:

    2 <= n <= 100
    player is 1 or 2.
    0 <= row, col < n
    (row, col) are unique for each different call to move.
    At most n2 calls will be made to move.

 

Follow-up: Could you do better than O(n2) per move() operation?


## Hashmap

- Keep track of a dictionary of live row and column frequency specific per player
- Moment any counter reaches n, return true, else return false

Time: O(n) 
Space: O(n)

```
class TicTacToe:
    def __init__(self, n: int):
        self.n = n
        self.d = {n+2: 0, n+3: 0}
        

    def move(self, row: int, col: int, player: int) -> int:
        row += 1
        col += 1
        
        move = 1
        if player == 2:
            move = -1
        
        if -1 * row in self.d:
            self.d[-1 * row] += move
        else:
            self.d[-1 * row] = move
        
        if col in self.d:
            self.d[col] += move
        else:
            self.d[col] = move
        
        if row == col:
            self.d[self.n + 2] += move
        if (row - 1) + (col - 1) == self.n - 1:
            self.d[self.n + 3] += move
            
            
        if (abs(self.d[-1 * row]) == self.n) or (abs(self.d[col]) == self.n) or (abs(self.d[self.n+2]) == self.n) or (abs(self.d[self.n+3]) == self.n):
            print(row, col)
            print(self.d[-1 * row])
            print(self.d[col])
            print(self.d[self.n+2])
            print(self.d[self.n+3])
            return player
        return 0
```

# 349. Intersection of Two Arrays
Easy

Given two integer arrays nums1 and nums2, return an array of their intersection. Each element in the result must be unique and you may return the result in any order.

 

Example 1:

Input: nums1 = [1,2,2,1], nums2 = [2,2]
Output: [2]

Example 2:

Input: nums1 = [4,9,5], nums2 = [9,4,9,8,4]
Output: [9,4]
Explanation: [4,9] is also accepted.

 

Constraints:

    1 <= nums1.length, nums2.length <= 1000
    0 <= nums1[i], nums2[i] <= 1000



## Hashset

- Initialize a set
    -Add every element into the set
- Return the set as a list

Time: O(n) 
Time: O(n) n - number of elements in both arrays

```
def solution(nums1, nums2):
    a = set()
    b = set()
    o = []
    for n in nums1:
        a.add(n)
    for n in nums2:
        b.add(n)
    for i in a:    
        if i in b:
            o.append(i)
    return o
```
# 362. Design Hit Counter
Medium

Design a hit counter which counts the number of hits received in the past 5 minutes (i.e., the past 300 seconds).

Your system should accept a timestamp parameter (in seconds granularity), and you may assume that calls are being made to the system in chronological order (i.e., timestamp is monotonically increasing). Several hits may arrive roughly at the same time.

Implement the HitCounter class:

HitCounter() Initializes the object of the hit counter system.
void hit(int timestamp) Records a hit that happened at timestamp (in seconds). Several hits may happen at the same timestamp.
int getHits(int timestamp) Returns the number of hits in the past 5 minutes from timestamp (i.e., the past 300 seconds).
 

Example 1:

Input
["HitCounter", "hit", "hit", "hit", "getHits", "hit", "getHits", "getHits"]
[[], [1], [2], [3], [4], [300], [300], [301]]
Output
[null, null, null, null, 3, null, 4, 3]

Explanation
HitCounter hitCounter = new HitCounter();
hitCounter.hit(1);       // hit at timestamp 1.
hitCounter.hit(2);       // hit at timestamp 2.
hitCounter.hit(3);       // hit at timestamp 3.
hitCounter.getHits(4);   // get hits at timestamp 4, return 3.
hitCounter.hit(300);     // hit at timestamp 300.
hitCounter.getHits(300); // get hits at timestamp 300, return 4.
hitCounter.getHits(301); // get hits at timestamp 301, return 3.
 

Constraints:

1 <= timestamp <= 2 * 109
All the calls are being made to the system in chronological order (i.e., timestamp is monotonically increasing).
At most 300 calls will be made to hit and getHits.
 

Follow up: What if the number of hits per second could be huge? Does your design scale?

## Queue 

It is important to note that all function calls have input timestamps that are strictly increasing. Each getHits(timestamp) call has a timestamp value greater than all previous getHits() and hit() call. The same applies for each hit(timestamp) call.

Therefore, we can prune the saved hits each time, removing any hits that are older than 300 seconds old without causing problems to future function calls. 

```
class HitCounter:
    def __init__(self):
        self.hits = []
        

    def hit(self, timestamp: int) -> None:
        self.hits.append(timestamp)
        

    def getHits(self, timestamp: int) -> int:
        while(self.hits != []):
            if self.hits[0] <= timestamp - 300:
                self.hits.pop(0)
            else: 
                break 
        return len(self.hits)

```

## Deque of Pairs

Duplicate consequtive timestamps can be handled, by storing timestamp values as a pair with its count. 



# 370. Range Addition
Medium

You are given an integer length and an array updates where updates[i] = [startIdxi, endIdxi, inci].

You have an array arr of length length with all zeros, and you have some operation to apply on arr. In the ith operation, you should increment all the elements arr[startIdxi], arr[startIdxi + 1], ..., arr[endIdxi] by inci.

Return arr after applying all the updates.

 

Example 1:

Input: length = 5, updates = [[1,3,2],[2,4,3],[0,2,-2]]
Output: [-2,0,3,5,3]

Example 2:

Input: length = 10, updates = [[2,4,6],[5,6,8],[1,9,-4]]
Output: [0,-4,2,2,2,4,4,-4,-4,-4]

 

Constraints:

    1 <= length <= 105
    0 <= updates.length <= 104
    0 <= startIdxi <= endIdxi < length
    -1000 <= inci <= 1000



## Hashmap

- have start hashmap
    - store start index with increment
- have end hashmap
    - store end index with increment
- set curr to 0
- if curr index in start hash
    - increment curr by all values in list
- if curr index in end hash
    - decrement curr by all values in list
- set array value to increment

Space: O(n)
Time: O(n)

```
from collections import defaultdict 

def solution(length, updates):
    start = defaultdict(list)
    end = defaultdict(list)
    for u in updates:
        start[u[0]].append(u[2])
        end[u[1] + 1].append(u[2])
    arr = []
    curr = 0
    for i in range(length):
        if i in start:
            for j in start[i]:
                curr += j
        if i in end:
            for j in end[i]:
                curr -= j
        arr.append(curr)
    return arr
```

# 371. Sum of Two Integers
Medium

Given two integers a and b, return the sum of the two integers without using the operators + and -.

 

Example 1:

Input: a = 1, b = 2
Output: 3

Example 2:

Input: a = 2, b = 3
Output: 5


## Iterative ()

Perform binary addition if both values are negative or both values are positive. 

Perform binary subtraction if one is negative and the other is positive. Subtract the larger absolute value from the smaller absolute value. If the larger absolute value return the negative result of the difference.

Binary addition is performed with a carry. 

a + b = a XOR b = a ^ b
carry = a AND b LEFT SHIFT 1 = a & b << 1

Repeat the carry and addition steps until the carry is 0. 

Binary subtraction is performed with a borrow. 

a - b = a XOR b = a ^ b
borrow = (NOT a) AND b LEFT SHIFT 1 = ~a & b << 1

Repeat the borrow and substraction steps until the borrow is 0. 

b1 b2 a c 
1  1  0 1
1  0  1 0
0  1  1 0
0  0  0 0

a = b1 XOR b2
c = b1 & b2

-------

[OLD EXPLANATION]
Set the default carry value to 0. Follow the following mapping below for addition: 


b1 b2 c1 a c2
1  1  1  1 1
1  1  0  1 0
1  0  1  1 0
0  1  1  1 0 
0  0  0  0 0




Get all the carries at once, and all the answers at once.

To get carry, perform an and operation between all binary 1 and binary 2 values. 

To get carry, perform or operation between all binary one and binary two values. 

a1 = b1 ^ b2 (XOR)
c1 = b1 & b2

b1 = a2
b2 = c1

loop the above instructions until b2 is not 0

return b1

Check for the sign of the input value as well

If one value is negative, then find the difference

If both values are negative, then find the sum and multiply it

b1 b2 a c 
1  1  0 0
1  0  1 0
0  1  1 1
0  0  0 0

1 0 
0 1
---
0 1

1 0 0 
0 0 1
-----
0 1 1

Keep moving bits until no 0 1's exist in the two numbers to perform difference on. 

Flip b1, then and it with b2 to get values to move. Keep moving bits until this value is 0.

There are two states to remeber

b1 b1
1  0  is the only valid number to borrow from
1  1  isn't a valid number ot borrow from, since you will need to reborrow because 1 1 will become 0 1 after the borrow, so we keep searching leftward. 


Space: O(1)
Time: O(1)

(Uses multiplication, but not addition) (Solution without addition or multiplicaiton requires additional work and understanding of language specific representation of numbers)

```
def solution(a, b):
    if (a <= 0 and b <= 0) or (a >= 0 and b >= 0):
        if (a < 0 and b < 0):
            b1 = -a
            b2 = -b
        else:
            b1 = a
            b2 = b

        while b2 != 0:
            answer = b1 ^ b2
            carry = (b1 & b2) << 1

            b1 = answer
            b2 = carry

        if (a < 0 and b < 0):
            return -b1
        return b1

    else: 
        b1 = max(abs(a), abs(b))
        b2 = min(abs(a), abs(b))
        if a < 0:
            b1 = b
            b2 = -a
        else: 
            b1 = a
            b2 = -b

        while b2 != 0:
            answer = b1 ^ b2
            borrow = (~b1 & b2) << 1

            b1 = answer
            b2 = borrow

        if (abs(a) > abs(b) and a > 0) or (abs(b) > abs(a) and b > 0): 
            return b1 

        return -b1

def solution1(a, b):
    b1 = a
    b2 = b

    while b2 != 0:
        a = b1 ^ b2
        c = (b1 & b2) << 1

        b1 = a
        b2 = c

    return b1
```

# 377. Combination Sum IV
Medium

Given an array of distinct integers nums and a target integer target, return the number of possible combinations that add up to target.

The test cases are generated so that the answer can fit in a 32-bit integer.

 

Example 1:

Input: nums = [1,2,3], target = 4
Output: 7
Explanation:
The possible combination ways are:
(1, 1, 1, 1)
(1, 1, 2)
(1, 2, 1)
(1, 3)
(2, 1, 1)
(2, 2)
(3, 1)
Note that different sequences are counted as different combinations.

Example 2:

Input: nums = [9], target = 3
Output: 0

 

Constraints:

    1 <= nums.length <= 200
    1 <= nums[i] <= 1000
    All the elements of nums are unique.
    1 <= target <= 1000

 

Follow up: What if negative numbers are allowed in the given array? How does it change the problem? What limitation we need to add to the question to allow negative numbers?


## Dynamic Programming

Initialize a DP table of size target + 1. Initialize the first element to 1. 

Then for each element in the dp table, iterate over all coins/num and add the previous dp element num steps behind to the current dp element. 

Return the value stored in the last element of the DP table. 

Time: O(N * M) s.t. M is number of different coins/nums
Space: O(N)

```
def solution(nums, target):
    dp = [0 for _ in range(target + 1)]
    dp[0] = 1
    for index in range(len(dp)):
        for num in nums: 
            if index - num >= 0: 
                dp[index] += dp[index - num]
    
    return dp[-1]
```
# 383. Ransom Note
Easy

Given two strings ransomNote and magazine, return true if ransomNote can be constructed by using the letters from magazine and false otherwise.

Each letter in magazine can only be used once in ransomNote.

 

Example 1:

Input: ransomNote = "a", magazine = "b"
Output: false

Example 2:

Input: ransomNote = "aa", magazine = "ab"
Output: false

Example 3:

Input: ransomNote = "aa", magazine = "aab"
Output: true

 

Constraints:

    1 <= ransomNote.length, magazine.length <= 105
    ransomNote and magazine consist of lowercase English letters.

## Hashtable

Use a single counter to figure out if random note needs any letters not found in magazine. Start by creating a hashtable that stores the counts for each letter in magaize. Decrement the counts of letters in the magazine counter while iterating through letters of ransomNote that match. If any value becomes negative, that means that ransom requires additional letters. If a letter in ransom isn't found in the magazine counter, that means ransom also needs an addition letter. In either cases return false.

```
def canConstruct(self, ransomNote: str, magazine: str) -> bool:
    count = {}

    for letter in magazine: 
        if letter not in count: 
            count[letter] = 1
        else: 
            count[letter] += 1
        
    for letter in ransomNote: 
        if letter in count:
            count[letter] -= 1
            if count[letter] < 0: 
                return False 
        else: 
            return False 
    
    return True 
```

# 387. First Unique Character in a String
Easy

Given a string s, find the first non-repeating character in it and return its index. If it does not exist, return -1.

 

Example 1:

Input: s = "leetcode"
Output: 0

Example 2:

Input: s = "loveleetcode"
Output: 2

Example 3:

Input: s = "aabb"
Output: -1

 

Constraints:

    1 <= s.length <= 105
    s consists of only lowercase English letters.



## Hashmap

Go through and save count
- Go through again, and if count == 1, return index
- Return -1

Time: O(n)
Space: O(1) b/c 26 letters

```
from collections import Counter
def solution(s):
    freq = Counter(s)
    for i in range(len(s)):
        if freq[s[i]] == 1:
            return i
    return -1
```

# 389. Find the Difference

You are given two strings s and t.

String t is generated by random shuffling string s and then add one more letter at a random position.

Return the letter that was added to t.

 

Example 1:

Input: s = "abcd", t = "abcde"
Output: "e"
Explanation: 'e' is the letter that was added.

Example 2:

Input: s = "", t = "y"
Output: "y"

 

Constraints:

    0 <= s.length <= 1000
    t.length == s.length + 1
    s and t consist of lowercase English letters.



## Sorting

```
s = sorted(s)
t = sorted(t)

for i in range(len(s)):
    if t[i] != s[i]:
        return t[i]

return t
```

## Hashmap

```
from collections import Counter
        
s_count = Counter(s)
t_count = Counter(t)
for t in t_count:
    if t not in s_count:
        return t
    else: 
        if s_count[t] != t_count[t]: 
            return t

return ""
```    
    


```

```

# 390. Elimination Game

You have a list arr of all integers in the range [1, n] sorted in a strictly increasing order. Apply the following algorithm on arr:

    Starting from left to right, remove the first number and every other number afterward until you reach the end of the list.
    Repeat the previous step again, but this time from right to left, remove the rightmost number and every other number from the remaining numbers.
    Keep repeating the steps again, alternating left to right and right to left, until a single number remains.

Given the integer n, return the last number that remains in arr.

 

Example 1:

Input: n = 9
Output: 6
Explanation:
arr = [1, 2, 3, 4, 5, 6, 7, 8, 9]
arr = [2, 4, 6, 8]
arr = [2, 6]
arr = [6]

Example 2:

Input: n = 1
Output: 1

 

Constraints:

    1 <= n <= 109





```

```

## Given an integer array nums with possible duplicates, randomly output the index of a given target number. You can assume that the given target number must exist in the array.

Implement the Solution class:

    Solution(int[] nums) Initializes the object with the array nums.
    int pick(int target) Picks a random index i from nums where nums[i] == target. If there are multiple valid i's, then each index should have an equal probability of returning.

 

Example 1:

Input
["Solution", "pick", "pick", "pick"]
[[[1, 2, 3, 3, 3]], [3], [1], [3]]
Output
[null, 4, 0, 2]

Explanation
Solution solution = new Solution([1, 2, 3, 3, 3]);
solution.pick(3); // It should return either index 2, 3, or 4 randomly. Each index should have equal probability of returning.
solution.pick(1); // It should return 0. Since in the array only nums[0] is equal to 1.
solution.pick(3); // It should return either index 2, 3, or 4 randomly. Each index should have equal probability of returning.


## Iteration with Hashmap

Keep a hashmap of of integer keys and list values that stores all the indices of a number in the input list. 

When asked asked to retrieve a index at random, pick a random integer between 0 and the length of the hashmap's list according to the target value. Then return the value at corresponding index of the list, according to the random integer value. 

Space: O(n)
Time: O(1)

## Resovoir Sampling

IDK WHAT TO DO>>>>>

Space: O(1)
Time: O(n)

```

```

# 408. Valid Word Abbreviation
Easy

A string can be abbreviated by replacing any number of non-adjacent, non-empty substrings with their lengths. The lengths should not have leading zeros.

For example, a string such as "substitution" could be abbreviated as (but not limited to):

    "s10n" ("s ubstitutio n")
    "sub4u4" ("sub stit u tion")
    "12" ("substitution")
    "su3i1u2on" ("su bst i t u ti on")
    "substitution" (no substrings replaced)

The following are not valid abbreviations:

    "s55n" ("s ubsti tutio n", the replaced substrings are adjacent)
    "s010n" (has leading zeros)
    "s0ubstitution" (replaces an empty substring)

Given a string word and an abbreviation abbr, return whether the string matches the given abbreviation.

A substring is a contiguous non-empty sequence of characters within a string.

 

Example 1:

Input: word = "internationalization", abbr = "i12iz4n"
Output: true
Explanation: The word "internationalization" can be abbreviated as "i12iz4n" ("i nternational iz atio n").

Example 2:

Input: word = "apple", abbr = "a2e"
Output: false
Explanation: The word "apple" cannot be abbreviated as "a2e".

 

Constraints:

    1 <= word.length <= 20
    word consists of only lowercase English letters.
    1 <= abbr.length <= 10
    abbr consists of lowercase English letters and digits.
    All the integers in abbr will fit in a 32-bit integer.



## Queue

- Place word into queue
- Check if next abbrv value is a letter, if so pop 1 + currnumber from stack and set curr number to zero
- Compare values and return false if different
- Check if next abbrv value is a digit, if so add to curr number at end of loop

Space: O(n)
Time: O(n)

```
from collections import deque

def solution(word, abbr):
    q = deque(word)
    curr = '0'
    for i in abbr:
        if i == '0' and int(curr) == 0:
            return False
        elif i in '0123456789':
            curr += i
        else:
            for j in range(int(curr)):
                try: 
                    q.popleft()
                except:
                    return False
            try:
                c = q.popleft()
            except:
                return False
            curr = '0'
            if c != i:
                return False
    if int(curr) != len(q):
        return False

    return True
```

# 415. Add Strings
Easy

Given two non-negative integers, num1 and num2 represented as string, return the sum of num1 and num2 as a string.

You must solve the problem without using any built-in library for handling large integers (such as BigInteger). You must also not convert the inputs to integers directly.

 

Example 1:

Input: num1 = "11", num2 = "123"
Output: "134"

Example 2:

Input: num1 = "456", num2 = "77"
Output: "533"

Example 3:

Input: num1 = "0", num2 = "0"
Output: "0"

 

Constraints:

    1 <= num1.length, num2.length <= 104
    num1 and num2 consist of only digits.
    num1 and num2 don't have any leading zeros except for the zero itself.



## Stack

- Convert num1 and num2 into list of characters
- Interate over max of len of num1 and num2
- pop elements from each list
- keep carry integer
- append each sum into result stack
- return reverse of stack

Time: O(N)
Space: O(N)

```
def solution(num1, num2):
    carry = '0'
    sm = []
    l1 = len(num1) - 1
    l2 = len(num2) - 1
    while l1 >= 0 or l2 >= 0:
        if l1 >= 0:
            n1 = num1[l1]
        else:
            n1 = '0'
        if l2 >= 0:
            n2 = num2[l2]
        else:
            n2 = '0'
        s = int(n1) + int(n2) + int(carry)
        carry = str(s//10)
        result = str(s%10)
        sm.append(result)
        l1 -= 1
        l2 -= 1
    if carry != '0':
        sm.append(carry)
    return "".join(sm[::-1])
```

# 417. Pacific Atlantic Water Flow
Medium

There is an m x n rectangular island that borders both the Pacific Ocean and Atlantic Ocean. The Pacific Ocean touches the island's left and top edges, and the Atlantic Ocean touches the island's right and bottom edges.

The island is partitioned into a grid of square cells. You are given an m x n integer matrix heights where heights[r][c] represents the height above sea level of the cell at coordinate (r, c).

The island receives a lot of rain, and the rain water can flow to neighboring cells directly north, south, east, and west if the neighboring cell's height is less than or equal to the current cell's height. Water can flow from any cell adjacent to an ocean into the ocean.

Return a 2D list of grid coordinates result where result[i] = [ri, ci] denotes that rain water can flow from cell (ri, ci) to both the Pacific and Atlantic oceans.

 

Example 1:

Input: heights = [[1,2,2,3,5],[3,2,3,4,4],[2,4,5,3,1],[6,7,1,4,5],[5,1,1,2,4]]
Output: [[0,4],[1,3],[1,4],[2,2],[3,0],[3,1],[4,0]]
Explanation: The following cells can flow to the Pacific and Atlantic oceans, as shown below:
[0,4]: [0,4] -> Pacific Ocean 
       [0,4] -> Atlantic Ocean
[1,3]: [1,3] -> [0,3] -> Pacific Ocean 
       [1,3] -> [1,4] -> Atlantic Ocean
[1,4]: [1,4] -> [1,3] -> [0,3] -> Pacific Ocean 
       [1,4] -> Atlantic Ocean
[2,2]: [2,2] -> [1,2] -> [0,2] -> Pacific Ocean 
       [2,2] -> [2,3] -> [2,4] -> Atlantic Ocean
[3,0]: [3,0] -> Pacific Ocean 
       [3,0] -> [4,0] -> Atlantic Ocean
[3,1]: [3,1] -> [3,0] -> Pacific Ocean 
       [3,1] -> [4,1] -> Atlantic Ocean
[4,0]: [4,0] -> Pacific Ocean 
       [4,0] -> Atlantic Ocean
Note that there are other possible paths for these cells to flow to the Pacific and Atlantic oceans.

Example 2:

Input: heights = [[1]]
Output: [[0,0]]
Explanation: The water can flow from the only cell to the Pacific and Atlantic oceans.


## DFS with Backtracking (Optimal)

The key to this problem is to search from each boarder outward. Searching if each cell reaches a border is significantly more complex. 

Perform DFS from all pacific border coordingates. 

Perform DFS for all atlantic border coordinates.

Return the intersection of the coordinates that can reach Pacfifc and can reach Atlantic. 

Pass as input, separate arrays for cells reached from the pacific and the atlantic. Store the cell information in an array of 0's and 1's or as a set of pairs. 

Time: O(M*N)
Space: O(M*N)

```
def solution(heights):
    reachedPacific = set()
    reachedAtlantic = set()
    directions = {(0, 1), (0, -1), (1, 0), (-1, 0)}
    result = []

    def valid(row, col):
        if row >= 0 and row < len(heights):
            if col >= 0 and col < len(heights[0]):
                return True
        return False

    def search(row, col, reached):
        nonlocal reachedAtlantic
        nonlocal reachedPacific
        if (row, col) in reached:
            return 
        else:
            reached.add((row, col))
            for direction in directions: 
                newRow = row + direction[0] 
                newCol = col + direction[1]
                if valid(newRow, newCol) and heights[newRow][newCol] >= heights[row][col]:
                    reached.add((row, col))
                    search(newRow, newCol, reached)

    for row in range(len(heights)):
        search(row, 0, reachedPacific)
    for col in range(len(heights[0])):
        search(0, col, reachedPacific)
    for row in range(len(heights)):
        search(row, len(heights[0]) - 1, reachedAtlantic)
    for col in range(len(heights[0])):
        search(len(heights) - 1, col, reachedAtlantic)

    for row in range(len(heights)):
        for col in range(len(heights[0])):
            if (row, col) in reachedPacific and (row, col) in reachedAtlantic:
                result.append([row, col])

    return result
```

# 424. Longest Repeating Character Replacement
Medium

You are given a string s and an integer k. You can choose any character of the string and change it to any other uppercase English character. You can perform this operation at most k times.

Return the length of the longest substring containing the same letter you can get after performing the above operations.

 

Example 1:

Input: s = "ABAB", k = 2
Output: 4
Explanation: Replace the two 'A's with two 'B's or vice versa.

Example 2:

Input: s = "AABABBA", k = 1
Output: 4
Explanation: Replace the one 'A' in the middle with 'B' and form "AABBBBA".
The substring "BBBB" has the longest repeating letters, which is 4.


## Sliding Window

Keep a set of all the unique elements in the string, and designate this as the alphabet.  

For each letter of the alphabet, perform the sliding window search. 

Set the left and right pointer to 0. 
If right pointer equals the target letter, increment right pointer by 1. If right pointer doesn't equal to target letter and if right pointer - left pointer < k + letterCount, increment right pointer and decrement k. If k is equal to 0, increment left pointer. If k is equal to 0 and if left pointer points to target letter, don't increment k. If k is equal to 0 and if left pointer doesn't point to target letter, incremement k by 1. Return max distance so far, right - left.


```
def slidingWindow(s, letter, k):
    left = 0
    right = 0
    maxWindow = 0
    letterCount  = 0
    windowSize = k
    while right < len(s):
        if s[right] == letter:
            letterCount += 1
        if s[right] != letter and right - left < windowSize + letterCount:
            k -= 1
        right += 1

        if k < 0 or right - left > windowSize + letterCount:
            if s[left] != letter:
                k += 1
            if s[left] == letter:
                letterCount -= 1
            left += 1
        
        window = right - left
        maxWindow = max(window, maxWindow)

    return maxWindow

alphabet = set()
for letter in s:
    alphabet.add(letter)

maxWindow = 0
for letter in alphabet:
    window = slidingWindow(s, letter, k)
    maxWindow = max(window, maxWindow)
return maxWindow
```

Time: O(n) = O(c * n) s.t. c = len(alphabet)
Space: O(1)

# 426. Convert Binary Search Tree to Sorted Doubly Linked List
Medium

Convert a Binary Search Tree to a sorted Circular Doubly-Linked List in place.

You can think of the left and right pointers as synonymous to the predecessor and successor pointers in a doubly-linked list. For a circular doubly linked list, the predecessor of the first element is the last element, and the successor of the last element is the first element.

We want to do the transformation in place. After the transformation, the left pointer of the tree node should point to its predecessor, and the right pointer should point to its successor. You should return the pointer to the smallest element of the linked list.

 

Example 1:

Input: root = [4,2,5,1,3]


Output: [1,2,3,4,5]

Explanation: The figure below shows the transformed BST. The solid line indicates the successor relationship, while the dashed line means the predecessor relationship.

Example 2:

Input: root = [2,1,3]
Output: [1,2,3]

 

Constraints:

    The number of nodes in the tree is in the range [0, 2000].
    -1000 <= Node.val <= 1000
    All the values of the tree are unique.



## DFS Traversal

- Inorder traversal
- Set right pointer to next node visited
- Set left pointer to previous node visited
- Pass along previous node, start previous with None
- Pass along next node, start with root.right (prev)
- Set prev.right to curr and curr.left to prev
- Point last right node to first node, and first left to last node
- Return last node
- Keep track of last node nonlocally

Time: O(N)
Space: O(1)

```
def solution(root):
    if not root:
        return None
    ln = None 
    fn = None
    def inorder(root):
        nonlocal ln, fn
        if not root:
            return
        inorder(root.left)
        if ln:
            ln.right = root
            root.left = ln
        else:
            fn = root
        ln = root
        inorder(root.right)
    
    inorder(root)
    fn.left = ln
    ln.right = fn
    return ln.right
```

# 435. Non-overlapping Intervals
Medium

Given an array of intervals intervals where intervals[i] = [starti, endi], return the minimum number of intervals you need to remove to make the rest of the intervals non-overlapping.

 

Example 1:

Input: intervals = [[1,2],[2,3],[3,4],[1,3]]
Output: 1
Explanation: [1,3] can be removed and the rest of the intervals are non-overlapping.

Example 2:

Input: intervals = [[1,2],[1,2],[1,2]]
Output: 2
Explanation: You need to remove two [1,2] to make the rest of the intervals non-overlapping.

Example 3:

Input: intervals = [[1,2],[2,3]]
Output: 0
Explanation: You don't need to remove any of the intervals since they're already non-overlapping.


## Iteration

Calculate the number of overlaps for each interval. Store the values in a heap. Remove the intervals with most to least over laps until the intervals are not longer overlapping. 

## Greedy - Sort by Start Time (Suboptimal)

sort intervals by value at first index 

remove the value with the longer next index 

final interavals no. minus starting interval no. is min removals

```
def eraseOverlapIntervals(self, intervals: List[List[int]]) -> int:
    n = len(intervals)
    intervals.sort()

    stack = []
    for i in range(len(intervals)):
        curr = intervals[i]
        if not stack:
            stack.append(curr)
        else:
            if curr[0] < stack[-1][1]:
                if curr[1] < stack[-1][1]: 
                    stack.pop()
                    stack.append(curr)
            else: 
                stack.append(curr)
    
    m = len(stack)
    return n - m
```

## Greedy - Sort by End Time (Optimal)

Sort the array by end times. Iterate over each interval, and if the current interval start time overlaps the previous interval end time, the add to total count, else update the previous interval end time to the current end time. 

This works because 

```
from math import inf
def eraseOverlapIntervals(self, intervals: List[List[int]]) -> int:
    end = -inf
    removals = 0
    intervals = sorted(intervals, key = lambda x: x[1])

    for interval in intervals: 
        if interval[0] >= end: 
            end = interval[1]
        else:
            removals += 1
    return removals
```

# 437. Path Sum III
Medium

Given the root of a binary tree and an integer targetSum, return the number of paths where the sum of the values along the path equals targetSum.

The path does not need to start or end at the root or a leaf, but it must go downwards (i.e., traveling only from parent nodes to child nodes).

 

Example 1:


Input: root = [10,5,-3,3,2,null,11,3,-2,null,1], targetSum = 8
Output: 3
Explanation: The paths that sum to 8 are shown.
Example 2:

Input: root = [5,4,8,11,null,13,4,7,2,null,null,5,1], targetSum = 22
Output: 3
 

Constraints:

The number of nodes in the tree is in the range [0, 1000].
-109 <= Node.val <= 109
-1000 <= targetSum <= 1000

## DFS + Prefix Sum

Use hashmap that store the prefix sum's compute sofar, similar to continuous subarray sum. This allows to check if the current sum - target sum exists in the previously computed prefix sums, in O(1) time. If the current sum - target sum exists in the set of prefix sums, this means there exists two nodes where the start node is (curr sum - target sum) and the end node is (curr sum), therefore the difference between the nodes is target sum, which is a valid path. 

Perform dfs, while keeping track of this hashmap and the current prefix sum, by popping and removing elements from the hashmap and summing values to the prefix sum, while recursing through the tree. 

Time: O(n)
Space: O(n)

```
def pathSum(self, root: Optional[TreeNode], targetSum: int) -> int:
    self.count = 0 
    def search(node, hashmap, currSum):
        if not node: 
            return 
        else:
            currSum += node.val
            if currSum - targetSum in hashmap:
                self.count += hashmap[currSum - targetSum]
            
            if currSum not in hashmap: 
                hashmap[currSum] = 1
            else: 
                hashmap[currSum] += 1
            
            search(node.left, hashmap, currSum)
            search(node.right, hashmap, currSum)
            
            if hashmap[currSum] == 1: 
                hashmap.pop(currSum)
            else: 
                hashmap[currSum] -= 1

    search(root, {0:1}, 0)
    return self.count 
```

# 496. Next Greater Element I
Easy

The next greater element of some element x in an array is the first greater element that is to the right of x in the same array.

You are given two distinct 0-indexed integer arrays nums1 and nums2, where nums1 is a subset of nums2.

For each 0 <= i < nums1.length, find the index j such that nums1[i] == nums2[j] and determine the next greater element of nums2[j] in nums2. If there is no next greater element, then the answer for this query is -1.

Return an array ans of length nums1.length such that ans[i] is the next greater element as described above.

 

Example 1:

Input: nums1 = [4,1,2], nums2 = [1,3,4,2]
Output: [-1,3,-1]
Explanation: The next greater element for each value of nums1 is as follows:
- 4 is underlined in nums2 = [1,3,4,2]. There is no next greater element, so the answer is -1.
- 1 is underlined in nums2 = [1,3,4,2]. The next greater element is 3.
- 2 is underlined in nums2 = [1,3,4,2]. There is no next greater element, so the answer is -1.

Example 2:

Input: nums1 = [2,4], nums2 = [1,2,3,4]
Output: [3,-1]
Explanation: The next greater element for each value of nums1 is as follows:
- 2 is underlined in nums2 = [1,2,3,4]. The next greater element is 3.
- 4 is underlined in nums2 = [1,2,3,4]. There is no next greater element, so the answer is -1.

 

Constraints:

    1 <= nums1.length <= nums2.length <= 1000
    0 <= nums1[i], nums2[i] <= 104
    All integers in nums1 and nums2 are unique.
    All the integers of nums1 also appear in nums2.

 
Follow up: Could you find an O(nums1.length + nums2.length) solution?

## Monostack

Keep array for NGE's of size nums2
Move from left to right of nums2
If current element is less than last element on stack add
If current element is greater than last element on stack, pop elements until current element is less than last element on stack, then add
If stack is empty, then set value to -1

Space: O(n)
Time: O(1)

## Monostack
Perform monostack trick on nums2
    - add elements of num2 to stack from right to left 
    - if curr elment is less than right most stack element,
        keep popping elements from stack until less than curr number 
    - set top element of stack to NGE value 
    - keep going until reach left most value

Map index of nums1 to element of num2

Map again to monostack NGE values

return array 

```
#Initialize to none
def solution(nums1, nums2):
    monostack = []
    nge = [None] * len(nums2)
    for i in range(len(nums2)-1, -1, -1):
        while monostack and nums2[i] >= monostack[-1]:
            monostack.pop()
        if not monostack:
            nge[i] = -1
        else:
            nge[i] = monostack[-1]
        monostack.append(nums2[i])
        
    ngeDict = {}
    for i in range(len(nge)):
        ngeDict[nums2[i]] = nge[i]
        
        
    for i in range(len(nums1)):
        nums1[i] = ngeDict[nums1[i]]

    return nums1

# Initialize to -1
def nextGreaterElement(self, nums1: List[int], nums2: List[int]) -> List[int]:
    nge = [-1] * len(nums2)
    stack = []
    for i in reversed(range(len(nums2))):
        curr = nums2[i]
        
        while stack and curr >= stack[-1]: 
            stack.pop()

        if stack and curr < stack[-1]: 
            currNge = stack[-1]
            nge[i] = currNge
        
        stack.append(curr)
        
    numMap = {}
    for i in range(len(nums2)):
        curr = nums2[i]
        if curr not in numMap: 
            numMap[curr] = i

    output = []
    for i in range(len(nums1)):
        output.append(nge[numMap[nums1[i]]])
        
    return output

```

# 498. Diagonal Traverse
Medium

Given an m x n matrix mat, return an array of all the elements of the array in a diagonal order.

 

Example 1:

Input: mat = [[1,2,3],[4,5,6],[7,8,9]]
Output: [1,2,4,7,5,3,6,8,9]

Example 2:

Input: mat = [[1,2],[3,4]]
Output: [1,2,3,4]

 

Constraints:

    m == mat.length
    n == mat[i].length
    1 <= m, n <= 104
    1 <= m * n <= 104
    -105 <= mat[i][j] <= 105



## Simulation

- while UR
- R
- D
- while LL
- D
- R
- Iterate until reach end, and append continously to ans output

Time: O(N)
Space: O(1)

```
def solution(mat):
    if not mat:
        return mat
    i = 0; j = 0
    ans = []
    for k in range(len(mat) * len(mat[0])):
        ans.append(mat[i][j])
        if (i + j)%2 == 0:
            if j == len(mat[0]) - 1:
                i += 1
            elif i == 0:
                j += 1
            else:
                i -= 1
                j += 1
        else:
            if i == len(mat) - 1:
                j += 1
            elif j == 0:
                i += 1
            else:
                i += 1
                j -= 1
    return ans
```

# 523. Continuous Subarray Sum
Medium

Given an integer array nums and an integer k, return true if nums has a continuous subarray of size at least two whose elements sum up to a multiple of k, or false otherwise.

An integer x is a multiple of k if there exists an integer n such that x = n * k. 0 is always a multiple of k.

 

Example 1:

Input: nums = [23,2,4,6,7], k = 6
Output: true
Explanation: [2, 4] is a continuous subarray of size 2 whose elements sum up to 6.

Example 2:

Input: nums = [23,2,6,4,7], k = 6
Output: true
Explanation: [23, 2, 6, 4, 7] is an continuous subarray of size 5 whose elements sum up to 42.
42 is a multiple of 6 because 42 = 7 * 6 and 7 is an integer.

Example 3:

Input: nums = [23,2,6,4,7], k = 13
Output: false

 

Constraints:

    1 <= nums.length <= 105
    0 <= nums[i] <= 109
    0 <= sum(nums[i]) <= 231 - 1
    1 <= k <= 231 - 1



## Prefix Sum

- Prefix sum 
- Compute the prefix sum mod k
- Add the values into a dictionary
- If value repeats, return True
    - If no value repeats in total, return False
- 0, k = 9
- If len < 1, return False
- 0 2, k = 9
- If p % k = 0, return True
    - Skip the first element

Time: O(n)
Space: O(n)

```
def solution(nums, k):
    ps = {0: -1}
    s = 0
    for i,v in enumerate(nums):
        s = (s + v) % k
        if s not in ps:
            ps[s] = i
        else:
            if i - ps[s] > 1:
                return True
    return False

```

# 528. Random Pick with Weight
Medium

You are given a 0-indexed array of positive integers w where w[i] describes the weight of the ith index.

You need to implement the function pickIndex(), which randomly picks an index in the range [0, w.length - 1] (inclusive) and returns it. The probability of picking an index i is w[i] / sum(w).

    For example, if w = [1, 3], the probability of picking index 0 is 1 / (1 + 3) = 0.25 (i.e., 25%), and the probability of picking index 1 is 3 / (1 + 3) = 0.75 (i.e., 75%).

 

Example 1:

Input
["Solution","pickIndex"]
[[[1]],[]]
Output
[null,0]

Explanation
Solution solution = new Solution([1]);
solution.pickIndex(); // return 0. The only option is to return 0 since there is only one element in w.

Example 2:

Input
["Solution","pickIndex","pickIndex","pickIndex","pickIndex","pickIndex"]
[[[1,3]],[],[],[],[],[]]
Output
[null,1,1,1,1,0]

Explanation
Solution solution = new Solution([1, 3]);
solution.pickIndex(); // return 1. It is returning the second element (index = 1) that has a probability of 3/4.
solution.pickIndex(); // return 1
solution.pickIndex(); // return 1
solution.pickIndex(); // return 1
solution.pickIndex(); // return 0. It is returning the first element (index = 0) that has a probability of 1/4.

Since this is a randomization problem, multiple answers are allowed.
All of the following outputs can be considered correct:
[null,1,1,1,1,0]
[null,1,1,1,1,1]
[null,1,1,1,0,0]
[null,1,1,1,0,1]
[null,1,0,1,0,0]
......
and so on.

 

Constraints:

    1 <= w.length <= 104
    1 <= w[i] <= 105
    pickIndex will be called at most 104 times.



## Binary Search

- Create an array that is populated by w's values * weight
- Randomly generate integer between range (0, sum(weights))
- Rename the array with prefix sums
    - Iterate through array, until cross the number requested
    - Return index of that number
- 1, 9, 1 -> 1, 10, 11 
- Create prefix sum, then iterate with binary search
    - l = 0
    - r = len(prefix) - 1
    - mid = (l + r)//2
    - if n > mid:
        - l = mid 
    - if n < mid:
        - r = mid

Time: O(logn)
Space: O(n)

```
import random 

class Solution:
    def __init__(self, w):
        self.prefixsum = []
        currsum = 0
        for i in w:
            currsum += i
            self.prefixsum.append(currsum)
        self.totalsum = currsum    

    def pickIndex(self):
        target = random.random() * self.totalsum
        l = 0; r = len(self.prefixsum) - 1
        while l < r:
            mid = (l + r)//2
            if target > self.prefixsum[mid]:
                l = mid + 1
            else:
                r = mid
        return l
```

# 536. Construct Binary Tree from String
Medium

You need to construct a binary tree from a string consisting of parenthesis and integers.

The whole input represents a binary tree. It contains an integer followed by zero, one or two pairs of parenthesis. The integer represents the root's value and a pair of parenthesis contains a child binary tree with the same structure.

You always start to construct the left child node of the parent first if it exists.

 

Example 1:

Input: s = "4(2(3)(1))(6(5))"
Output: [4,2,6,3,1,5]

Example 2:

Input: s = "4(2(3)(1))(6(5)(7))"
Output: [4,2,6,3,1,5,7]

Example 3:

Input: s = "-4(2(3)(1))(6(5)(7))"
Output: [-4,2,6,3,1,5,7]

 

Constraints:

    0 <= s.length <= 3 * 104
    s consists of digits, '(', ')', and '-' only.



## Stack

- Keep track of 3 states ~ not started, left done, right done
- Loop index until reach end
- Pop last element from stack
    - Check if current index is digit
        - If so add digit to last element on stack
        - Also check if next index is '('
            - Add another node to stack
            - Set prev.left to this node
    - Elif check if current index is '('
        - If so, that means left child has been processed
        - Add new node to stack, and set prev.right to this node
- Increment index by 1
- Return last element of stack if stack is nonempty after while loop
    - Else return None

Time: O(n)
Space: O(n)

```
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def solution(s):
    if not s:
        return None
    i = 0
    root = TreeNode()
    stack = [root]
    while i < len(s):
        node = stack.pop()
        neg = False
        n = 0
        if s[i] == '-' or s[i].isdigit():
            if s[i] == '-':
                i += 1
                neg = True
            while i < len(s) and s[i].isdigit():
                n = n * 10 + int(s[i])
                i += 1
            if neg:
                node.val = -n
            else:
                node.val = n
            if i < len(s) and s[i] == '(':
                stack.append(node)
                node.left = TreeNode()
                stack.append(node.left)
        elif node.left and s[i] == '(':
            stack.append(node)
            node.right = TreeNode()
            stack.append(node.right)
        i += 1
    if stack:
        return stack.pop()
    else:
        return root
```

# 543. Diameter of Binary Tree
Easy

Given the root of a binary tree, return the length of the diameter of the tree.

The diameter of a binary tree is the length of the longest path between any two nodes in a tree. This path may or may not pass through the root.

The length of a path between two nodes is represented by the number of edges between them.

 

Example 1:

Input: root = [1,2,3,4,5]
Output: 3
Explanation: 3 is the length of the path [4,2,1,3] or [5,2,1,3].

Example 2:

Input: root = [1,2]
Output: 1

 

Constraints:

    The number of nodes in the tree is in the range [1, 104].
    -100 <= Node.val <= 100



## DFS Traversal

-keep track of global max
-find max left depth and max right depth
-if sum of max left curr right depth > global max, reset global max
-post order

Space: O(n)
Time: O(n)

```
from math import inf
def solution(root):
    gmax = -inf
    def postorder(root):
        nonlocal gmax
        if not root:
            return 0
        l = postorder(root.left)
        r = postorder(root.right)
        s = l + r
        gmax = max(s, gmax)
        return max(l, r) + 1
    postorder(root)
    return gmax

```

# 560. Subarray Sum Equals K
Medium

Given an array of integers nums and an integer k, return the total number of subarrays whose sum equals to k.

A subarray is a contiguous non-empty sequence of elements within an array.

 

Example 1:

Input: nums = [1,1,1], k = 2
Output: 2

Example 2:

Input: nums = [1,2,3], k = 3
Output: 2

 

Constraints:

    1 <= nums.length <= 2 * 104
    -1000 <= nums[i] <= 1000
    -107 <= k <= 107



## Prefix Sum

- Compute prefix sum
    - Difference between two prefix sum is sum of sub array
    - a b -> b - a == k
    - a % k and b % k, then 
- Two Sum + Prefix Sum
    - Store prefix sum values into array
        - If currprefixsum - k is in array, increment counter by frequency

Time: O(N)
Space: O(N)

```
from collections import defaultdict 

def solution(nums, k):
    prefixes = defaultdict(int)
    prefixes[0] += 1
    total = 0
    count = 0
    for i in nums:
        total += i
        count += prefixes[total - k]
        prefixes[total] += 1
        
    return count
```

# 572. Subtree of Another Tree
Easy

Given the roots of two binary trees root and subRoot, return true if there is a subtree of root with the same structure and node values of subRoot and false otherwise.

A subtree of a binary tree tree is a tree that consists of a node in tree and all of this node's descendants. The tree tree could also be considered as a subtree of itself.

 

Example 1:

Input: root = [3,4,5,1,2], subRoot = [4,1,2]
Output: true

Example 2:

Input: root = [3,4,5,1,2,null,null,null,null,0], subRoot = [4,1,2]
Output: false


## DFS Traversal (Suboptimal)

This problem can be simplied as perform the same tree problem N times. Therefore, we must traverse through the tree, and perform the same tree check on each node.

Time: O(M * N) s.t. M = nodes in original tree, N = nodes in subtree
Space: O(M + N)

```
def isSubtree(self, root: Optional[TreeNode], subRoot: Optional[TreeNode]) -> bool:
    def same(node, subnode):
        if not node and not subnode:
            return True
        if not node or not subnode:
            return False
        else: 
            if node.val == subnode.val: 
                leftSame = same(node.left, subnode.left)
                rightSame = same(node.right, subnode.right)
                return leftSame and rightSame
            else:
                return False 

    def search(node):
        if not node:
            return False 
        else:
            if same(node, subRoot):
                return True
            leftMatch = search(node.left)
            rightMatch = search(node.right)
            return leftMatch or rightMatch

    return search(root)
```

## DFS Traversal + Tree Hash (Optimal)

Map the orignal tree's subtrees to unique hash values. Compare the specified subtree's hash value with the original tree's hash values, which takes O(1) time per comparison. 

Time: O(M + N)
Space: O(M + N)

[TODO]


# 588. Design In-Memory File System
Hard

Design a data structure that simulates an in-memory file system.

Implement the FileSystem class:

    FileSystem() Initializes the object of the system.
    List<String> ls(String path)
        If path is a file path, returns a list that only contains this file's name.
        If path is a directory path, returns the list of file and directory names in this directory.
    The answer should in lexicographic order.
    void mkdir(String path) Makes a new directory according to the given path. The given directory path does not exist. If the middle directories in the path do not exist, you should create them as well.
    void addContentToFile(String filePath, String content)
        If filePath does not exist, creates that file containing given content.
        If filePath already exists, appends the given content to original content.
    String readContentFromFile(String filePath) Returns the content in the file at filePath.

 

Example 1:

Input
["FileSystem", "ls", "mkdir", "addContentToFile", "ls", "readContentFromFile"]
[[], ["/"], ["/a/b/c"], ["/a/b/c/d", "hello"], ["/"], ["/a/b/c/d"]]
Output
[null, [], null, null, ["a"], "hello"]

Explanation
FileSystem fileSystem = new FileSystem();
fileSystem.ls("/");                         // return []
fileSystem.mkdir("/a/b/c");
fileSystem.addContentToFile("/a/b/c/d", "hello");
fileSystem.ls("/");                         // return ["a"]
fileSystem.readContentFromFile("/a/b/c/d"); // return "hello"

 

Constraints:

    1 <= path.length, filePath.length <= 100
    path and filePath are absolute paths which begin with '/' and do not end with '/' except that the path is just "/".
    You can assume that all directory names and file names only contain lowercase letters, and the same names will not exist in the same directory.
    You can assume that all operations will be passed valid parameters, and users will not attempt to retrieve file content or list a directory or file that does not exist.
    1 <= content.length <= 50
    At most 300 calls will be made to ls, mkdir, addContentToFile, and readContentFromFile.

## Trie using Hashmap

A filesystem is a hash map.

A directory is an element within the hashmap, whose value is a hashmap.

A file is an element within the hashmap, whose value is a string (not a hashmap).

Mkdir traverses the tree, creating new directories, children in the tree if they do not already exist 

AddContentToFile traverses tree and adds string to the last node 

ls traverse tree and returns list of the last node 

Space: O(n)
Time: O(logN)

```
class FileSystem:
    def __init__(self):
        self.fs = {}

    def ls(self, path: str) -> List[str]:
        if path == '/': 
            pathList = []
        else: 
            pathList = path.split("/")
            pathList = pathList[1:]

        currNode = self.fs
        for p in pathList: 
            if p not in currNode: 
                currNode[p] = {}
            currNode = currNode[p]

        if type(currNode) == str: 
            return [pathList[-1]]
        return sorted(list(currNode.keys()))
        
    def mkdir(self, path: str) -> None:
        pathList = path.split("/")
        pathList = pathList[1:]
        currNode = self.fs
        for p in pathList: 
            if p not in currNode: 
                currNode[p] = {}
            currNode = currNode[p]

    def addContentToFile(self, filePath: str, content: str) -> None:
        pathList = filePath.split("/")
        pathList = pathList[1:]
        file = pathList[-1]
        pathList = pathList[:-1]
        
        currNode = self.fs
        for p in pathList: 
            if p not in currNode: 
                currNode[p] = {}
            currNode = currNode[p]
        
        if file not in currNode: 
            currNode[file] = content
        else: 
            currNode[file] += content
        

    def readContentFromFile(self, filePath: str) -> str:
        pathList = filePath.split("/")
        pathList = pathList[1:]
        file = pathList[-1]
        pathList = pathList[:-1]
        
        currNode = self.fs
        for p in pathList: 
            if p not in currNode: 
                currNode[p] = {}
            currNode = currNode[p]
        
        if file not in currNode: 
            return ""
        else: 
            return currNode[file]

# Your FileSystem object will be instantiated and called as such:
# obj = FileSystem()
# param_1 = obj.ls(path)
# obj.mkdir(path)
# obj.addContentToFile(filePath,content)
# param_4 = obj.readContentFromFile(filePath)
```

# 621. Task Scheduler
Medium

Given a characters array tasks, representing the tasks a CPU needs to do, where each letter represents a different task. Tasks could be done in any order. Each task is done in one unit of time. For each unit of time, the CPU could complete either one task or just be idle.

However, there is a non-negative integer n that represents the cooldown period between two same tasks (the same letter in the array), that is that there must be at least n units of time between any two same tasks.

Return the least number of units of times that the CPU will take to finish all the given tasks.

 

Example 1:

Input: tasks = ["A","A","A","B","B","B"], n = 2
Output: 8
Explanation: 
A -> B -> idle -> A -> B -> idle -> A -> B
There is at least 2 units of time between any two same tasks.
Example 2:

Input: tasks = ["A","A","A","B","B","B"], n = 0
Output: 6
Explanation: On this case any permutation of size 6 would work since n = 0.
["A","A","A","B","B","B"]
["A","B","A","B","A","B"]
["B","B","B","A","A","A"]
...
And so on.
Example 3:

Input: tasks = ["A","A","A","A","A","A","B","C","D","E","F","G"], n = 2
Output: 16
Explanation: 
One possible solution is
A -> B -> C -> A -> D -> E -> A -> F -> G -> A -> idle -> idle -> A -> idle -> idle -> A
 

Constraints:

1 <= task.length <= 104
tasks[i] is upper-case English letter.
The integer n is in the range [0, 100].

## Heap

Fill out the schedule by adding the most frequent tasks's element until no elements left. Use a maxHeap to sort the values based on frequency. 

Time: O(N) b/c T is constant ~ O(NlogT) s.t. T is types of tasks
Space: O(1)

```
def leastInterval(self, tasks: List[str], n: int) -> int:
    from collections import Counter 
    frequency = Counter(tasks)

    heap = []

    for f in frequency: 
        heapq.heappush(heap, (-1 * frequency[f], f))

    schedule = 0 

    while heap: 
        intervalLen = n + 1
        interval = []
        while intervalLen > 0 and heap: 
            e = heapq.heappop(heap)
            interval.append(e)
            intervalLen -= 1 
            schedule += 1
        
        for i in interval:
            if i[0] + 1 < 0: 
                heapq.heappush(heap, (i[0] + 1, i[1]))
            
        if not heap: 
            break 

        schedule += intervalLen

    return schedule 
```

## Math

Note that the maximum frequency element will determine the total time length. In particular, the most frequent eleement along with the required idle time will space out the tasks. The remaining tasks will greedily fill the remaining time. 


Time: O(n)
Space: O(1)

```
def leastInterval(self, tasks: List[str], n: int) -> int:
    from collections import Counter 
    frequency = Counter(tasks)

    maxFreq = max(frequency.values())

    maxFreqCount = 0 
    for f in frequency:
        if frequency[f] == maxFreq:
            maxFreqCount += 1
    
    return max(((maxFreq - 1) * (n + 1)) + maxFreqCount, len(tasks))
```

# 629. K Inverse Pairs Array
Hard

For an integer array nums, an inverse pair is a pair of integers [i, j] where 0 <= i < j < nums.length and nums[i] > nums[j].

Given two integers n and k, return the number of different arrays consist of numbers from 1 to n such that there are exactly k inverse pairs. Since the answer can be huge, return it modulo 109 + 7.

 

Example 1:

Input: n = 3, k = 0
Output: 1
Explanation: Only the array [1,2,3] which consists of numbers from 1 to 3 has exactly 0 inverse pairs.

Example 2:

Input: n = 3, k = 1
Output: 2
Explanation: The array [1,3,2] and [2,1,3] have exactly 1 inverse pair.

 

Constraints:

    1 <= n <= 1000
    0 <= k <= 1000



## Dynamic Programming

Time: O(nk)
Space: O(nk)

```

```

# 636. Exclusive Time of Functions
Medium

On a single-threaded CPU, we execute a program containing n functions. Each function has a unique ID between 0 and n-1.

Function calls are stored in a call stack: when a function call starts, its ID is pushed onto the stack, and when a function call ends, its ID is popped off the stack. The function whose ID is at the top of the stack is the current function being executed. Each time a function starts or ends, we write a log with the ID, whether it started or ended, and the timestamp.

You are given a list logs, where logs[i] represents the ith log message formatted as a string "{function_id}:{"start" | "end"}:{timestamp}". For example, "0:start:3" means a function call with function ID 0 started at the beginning of timestamp 3, and "1:end:2" means a function call with function ID 1 ended at the end of timestamp 2. Note that a function can be called multiple times, possibly recursively.

A function's exclusive time is the sum of execution times for all function calls in the program. For example, if a function is called twice, one call executing for 2 time units and another call executing for 1 time unit, the exclusive time is 2 + 1 = 3.

Return the exclusive time of each function in an array, where the value at the ith index represents the exclusive time for the function with ID i.

 

Example 1:

Input: n = 2, logs = ["0:start:0","1:start:2","1:end:5","0:end:6"]
Output: [3,4]
Explanation:
Function 0 starts at the beginning of time 0, then it executes 2 for units of time and reaches the end of time 1.
Function 1 starts at the beginning of time 2, executes for 4 units of time, and ends at the end of time 5.
Function 0 resumes execution at the beginning of time 6 and executes for 1 unit of time.
So function 0 spends 2 + 1 = 3 units of total time executing, and function 1 spends 4 units of total time executing.

Example 2:

Input: n = 1, logs = ["0:start:0","0:start:2","0:end:5","0:start:6","0:end:6","0:end:7"]
Output: [8]
Explanation:
Function 0 starts at the beginning of time 0, executes for 2 units of time, and recursively calls itself.
Function 0 (recursive call) starts at the beginning of time 2 and executes for 4 units of time.
Function 0 (initial call) resumes execution then immediately calls itself again.
Function 0 (2nd recursive call) starts at the beginning of time 6 and executes for 1 unit of time.
Function 0 (initial call) resumes execution at the beginning of time 7 and executes for 1 unit of time.
So function 0 spends 2 + 4 + 1 + 1 = 8 units of total time executing.

Example 3:

Input: n = 2, logs = ["0:start:0","0:start:2","0:end:5","1:start:6","1:end:6","0:end:7"]
Output: [7,1]
Explanation:
Function 0 starts at the beginning of time 0, executes for 2 units of time, and recursively calls itself.
Function 0 (recursive call) starts at the beginning of time 2 and executes for 4 units of time.
Function 0 (initial call) resumes execution then immediately calls function 1.
Function 1 starts at the beginning of time 6, executes 1 unit of time, and ends at the end of time 6.
Function 0 resumes execution at the beginning of time 6 and executes for 2 units of time.
So function 0 spends 2 + 4 + 1 = 7 units of total time executing, and function 1 spends 1 unit of total time executing.

 

Constraints:

    1 <= n <= 100
    1 <= logs.length <= 500
    0 <= function_id < n
    0 <= timestamp <= 109
    No two start events will happen at the same timestamp.
    No two end events will happen at the same timestamp.
    Each function has an "end" log for each "start" log.



## Stack

- If there is start, append time to stack
- If there is end, pop from stack
    - Compute difference between last element and and current element
    - Keep track of total time extracted and append to each element
        - Subtract that value from each end statement
        - Set total time extracted to difference + prev total
    - Save extracted time on last element of stack

Space: O(N)
Time: O(N)

```
def solution(n, logs):
    st = []
    d = {}
    for i in logs:
        log = i.split(":")
        f, state, t = int(log[0]), log[1], int(log[2])
        if state == "start":
            if st:
                if st[-1][0] in d:
                    d[st[-1][0]] += t - st[-1][1]
                else:
                    d[st[-1][0]] = t - st[-1][1]
            st.append([f, t])
        else:
            pt = st.pop()
            if f in d:
                d[f] += t - pt[1] + 1
            else:
                d[f] = t - pt[1] + 1
            if st:
                st[-1][1] = t + 1
    
    o = []
    for i in range(len(d)):
        o.append(d[i])
    return o         
```

# 647. Palindromic Substrings
Medium

Given a string s, return the number of palindromic substrings in it.

A string is a palindrome when it reads the same backward as forward.

A substring is a contiguous sequence of characters within the string.

 

Example 1:

Input: s = "abc"
Output: 3
Explanation: Three palindromic strings: "a", "b", "c".

Example 2:

Input: s = "aaa"
Output: 6
Explanation: Six palindromic strings: "a", "a", "a", "aa", "aa", "aaa".

## Dynamic Programming

Time: O(N^2)
Space: O(N^2)

## Intelligent Iteration with Recursive Search

Start at each index and search outward. Start at each pair of indices and search outward. This allows us to consider all possible palindromes, both ones of even and odd length. Using two pointers simplifies the search step, and using a counter variable simplified the counting step. 

Time: O(N^2)
Space: O(1)

```
def countSubstrings(self, s: str) -> int:
        count = 0
        for i in range(len(s)):
            l = r = i 
            while 0 <= l < len(s) and 0 <= r < len(s) and s[l] == s[r]:  
                l -= 1
                r += 1
                count += 1
        
        for i in range(1, len(s)):
            l = i - 1
            r = i 
            while 0 <= l < len(s) and 0 <= r < len(s) and s[l] == s[r]:   
                l -= 1
                r += 1
                count += 1
        
        return count 
```

# 658. Find K Closest Elements
Medium

Given a sorted integer array arr, two integers k and x, return the k closest integers to x in the array. The result should also be sorted in ascending order.

An integer a is closer to x than an integer b if:

    |a - x| < |b - x|, or
    |a - x| == |b - x| and a < b

 

Example 1:

Input: arr = [1,2,3,4,5], k = 4, x = 3
Output: [1,2,3,4]

Example 2:

Input: arr = [1,2,3,4,5], k = 4, x = -1
Output: [1,2,3,4]

 

Constraints:

    1 <= k <= arr.length
    1 <= arr.length <= 104
    arr is sorted in ascending order.
    -104 <= arr[i], x <= 104



## Quicksort

- Find left boundry of splice size k
- l = 0, r = len(arr) - k
- mid = (l + r)//2
- if arr[mid] - x > arr[mid + k] - x
    - r = mid
- else: 
    - l = mid + 1

Time: O(nlogk)
Space: O(1)

```
def solution(arr, k, x):
    if len(arr) == k:
        return arr
    l = 0; r = len(arr) - k
    while l < r:
        mid = (l + r)//2
        if x - arr[mid] > arr[mid+k] - x: 
            l = mid + 1
        else:
            r = mid
    return arr[l: l+k]
```

# 662. Maximum Width of Binary Tree
Medium

Given the root of a binary tree, return the maximum width of the given tree.

The maximum width of a tree is the maximum width among all levels.

The width of one level is defined as the length between the end-nodes (the leftmost and rightmost non-null nodes), where the null nodes between the end-nodes that would be present in a complete binary tree extending down to that level are also counted into the length calculation.

It is guaranteed that the answer will in the range of a 32-bit signed integer.

 

Example 1:


Input: root = [1,3,2,5,3,null,9]
Output: 4
Explanation: The maximum width exists in the third level with length 4 (5,3,null,9).
Example 2:


Input: root = [1,3,2,5,null,null,9,6,null,7]
Output: 7
Explanation: The maximum width exists in the fourth level with length 7 (6,null,null,null,null,null,7).
Example 3:


Input: root = [1,3,2,5]
Output: 2
Explanation: The maximum width exists in the second level with length 2 (3,2).
 

Constraints:

The number of nodes in the tree is in the range [1, 3000].
-100 <= Node.val <= 100

## BFS

```
```


## DFS 

Perform normal dfs while keeping track of the width value of the left most visited node, which is equal to the first node visisted for a given depth when performing dfs. 

When reaching the same depth again, compute the max width difference, and return the maximum width. Update the left child width as 2 * parent width, and the right child width as 2 * parent width + 1. This preserves the exponential growth of children nodes.

Time: O(n)
Space: O(n)

```
def widthOfBinaryTree(self, root: Optional[TreeNode]) -> int:
    self.leftWidth = {}
    self.max = 0

    def search(node, depth, width):
        if not node: 
            return
        else: 
            if depth not in self.leftWidth: 
                self.leftWidth[depth] = width

            self.max = max(self.max, width - self.leftWidth[depth] + 1)
            search(node.left, depth + 1, 2 * width)
            search(node.right, depth + 1, 2 * width + 1)
    
    search(root, 0, 0)
    return self.max
```

# 670. Maximum Swap
Medium

You are given an integer num. You can swap two digits at most once to get the maximum valued number.

Return the maximum valued number you can get.

 

Example 1:

Input: num = 2736
Output: 7236
Explanation: Swap the number 2 and the number 7.

Example 2:

Input: num = 9973
Output: 9973
Explanation: No swap.

 

Constraints:

    0 <= num <= 108



## Buckets

- 52334231 -> 54444331
- Use buckets for last index of digit

Time: O(n)
Space: O(1)

```
def solution(num):
    num = list(str(num))
        
    d = {'0': 0, '1': 0, '2': 0, '3': 0, '4': 0, '5': 0, '6': 0, '7': 0, '8': 0, '9':0}
    c = 0
    for i in num:
        d[i] = c
        c += 1
    
    c = 0
    for i in num:
        k = 9
        while k > int(i):
            if d[str(k)] > c:
                #swap
                tmp = i
                num[c] = num[d[str(k)]]
                num[d[str(k)]] = tmp
                return int("".join(num))
            k -= 1
        c += 1
    return int("".join(num))
```

# 680. Valid Palindrome II
Easy

Given a string s, return true if the s can be palindrome after deleting at most one character from it.

 

Example 1:

Input: s = "aba"
Output: true

Example 2:

Input: s = "abca"
Output: true
Explanation: You could delete the character 'c'.

Example 3:

Input: s = "abc"
Output: false

 

Constraints:

    1 <= s.length <= 105
    s consists of lowercase English letters.

## Iteration

- Check if string is valid palindrome
    - If so, return
- If l == r, return 
- Start and left and right
    - If s[l] == s[r], keep going
        - Search l+1, r-1
    - If s[l] != s[r], either prune right or left
        - Search l+1, r --> prunes left
        - Search l, r-1 --> prunes right
            - return left or right is true
    - Keep track of number of prunes, global prune = False
- Check palinrome helper
    - if s == s[::-1], return True
    - else return False

Time: O(n)
Space: O(n)

```
def solution(s):
    def isPalindrome(p):
        return p == p[::-1]
    l = 0; r = len(s) - 1
    while l < r:
        if s[l] != s[r]:
            return isPalindrome(s[l: r]) or isPalindrome(s[l+1:r+1])
        l += 1
        r -= 1
    return True
```

# 696. Count Binary Substrings
Easy

Given a binary string s, return the number of non-empty substrings that have the same number of 0's and 1's, and all the 0's and all the 1's in these substrings are grouped consecutively.

Substrings that occur multiple times are counted the number of times they occur.

 

Example 1:

Input: s = "00110011"
Output: 6
Explanation: There are 6 substrings that have equal number of consecutive 1's and 0's: "0011", "01", "1100", "10", "0011", and "01".
Notice that some of these substrings repeat and are counted the number of times they occur.
Also, "00110011" is not a valid substring because all the 0's (and 1's) are not grouped together.

Example 2:

Input: s = "10101"
Output: 4
Explanation: There are 4 substrings: "10", "01", "10", "01" that have equal number of consecutive 1's and 0's.

 

Constraints:

    1 <= s.length <= 105
    s[i] is either '0' or '1'.



## Two Pointer

- Two pointer
- If current == 0 and next == 1 or curr == 1 and next == 0 add to counter, then decrease l and r, if l >= 0 and r <= len(s) and s[l] == s[r], increase counter again

Space: O(1)
Time: O(N)

```
def solution(s):
    cnt = 0
    for i in range(len(s) - 1):
        l = i
        r = i + 1
        curr = s[l]
        next = s[r]
        if curr == '1' and next == '0':
            while (l >= 0 and r <= len(s) - 1) and (s[l] == '1' and s[r] == '0'):
                cnt += 1
                l -= 1
                r += 1
        l = i
        r = i + 1
        curr = s[l]
        next = s[r]
        if curr == '0' and next == '1':
            while (l >= 0 and r <= len(s) - 1) and (s[l] == '0' and s[r] == '1'):
                cnt += 1
                l -= 1
                r += 1
    return cnt
```

# 704. Binary Search
Easy

Given an array of integers nums which is sorted in ascending order, and an integer target, write a function to search target in nums. If target exists, then return its index. Otherwise, return -1.

You must write an algorithm with O(log n) runtime complexity.

 

Example 1:

Input: nums = [-1,0,3,5,9,12], target = 9
Output: 4
Explanation: 9 exists in nums and its index is 4
Example 2:

Input: nums = [-1,0,3,5,9,12], target = 2
Output: -1
Explanation: 2 does not exist in nums so return -1
 

Constraints:

1 <= nums.length <= 104
-104 < nums[i], target < 104
All the integers in nums are unique.
nums is sorted in ascending order.

## Binary Search (Find Lower Bound)

Consider all valid lower bound insert index positions. In this case from 0 to len(nums) - 1. If you reach the target value, update the right pointer, a lower bound won't need to consider those value, and duplicate target values to the right of the current target value are not required. 

Since the left pointer represents the lower bound, target exists in the array if the lower bound index + 1 equal to target. 

Time: O(logn)
Space: O(1)

## Binary Search (Find Upper Bound)

Consider all valid index insert positions for an upper bound value, ranging from 1 to len(nums). If the current index value is equal to target, then update left pointer, since previous values will only be lower than the target value, and duplicate target values to the right of the current target value will only be considered. 

The final left index will represent the upper bound value, so the target will exist in the array if the upper bound index - 1 equals to target. 

Time: O(logn)
Space: O(1)

## Binary Search (Find Exact Value)

Determine if the current midpoint is the target value. Else, check indices strictly to the right or left of the midpoint, by moving the left or right pointer respectively. 

Time: O(logn)
Space: O(1)

```
def search(self, nums: List[int], target: int) -> int:
    l = 0
    r = len(nums) - 1 # exact math and lower bound
    #r = len(nums) # upper bound 

    while l <= r: # exact match 
    # while l < r: # upper bound and lower bound 
        mid = (l + r)//2
        midVal = nums[mid]
        if midVal == target:
            return mid # exact
            #r = mid # lower bound
            #l = mid + 1 # upper bound 
            
        elif midVal < target:
            l = mid + 1 # exact
            #l = mid + 1 # lower bound 
            #l = mid + 1 # upper bound 
            
        elif midVal > target:
            r = mid - 1 # exact
            #r = mid # lower bound
            #r = mid # upper bound 

    # exact match 
    return -1  

    # lower bound 
    '''
    if l < len(nums) and nums[l] == target:
        return l 
    else: 
        return -1 
    '''

    
    # upper bound 
    '''
    if l > 0 and nums[l - 1] == target: 
        return l - 1
    else: 
        return -1
    '''
```

# 708. Insert into a Sorted Circular Linked List
Medium

Given a Circular Linked List node, which is sorted in non-descending order, write a function to insert a value insertVal into the list such that it remains a sorted circular list. The given node can be a reference to any single node in the list and may not necessarily be the smallest value in the circular list.

If there are multiple suitable places for insertion, you may choose any place to insert the new value. After the insertion, the circular list should remain sorted.

If the list is empty (i.e., the given node is null), you should create a new single circular list and return the reference to that single node. Otherwise, you should return the originally given node.

 

Example 1:

 

Input: head = [3,4,1], insertVal = 2
Output: [3,4,1,2]
Explanation: In the figure above, there is a sorted circular list of three elements. You are given a reference to the node with value 3, and we need to insert 2 into the list. The new node should be inserted between node 1 and node 3. After the insertion, the list should look like this, and we should still return node 3.



Example 2:

Input: head = [], insertVal = 1
Output: [1]
Explanation: The list is empty (given head is null). We create a new single circular list and return the reference to that single node.

Example 3:

Input: head = [1], insertVal = 0
Output: [1,0]

 

Constraints:

    The number of nodes in the list is in the range [0, 5 * 104].
    -106 <= Node.val, insertVal <= 106



## Iteration

- If null return val with next pointing at itself
- Loop until you find where you started 
- If node > cur and node < next add and return
- If node > curr and node > next do nothing
- If node < cur and node < next do nothing
- Else increment pointer
- If node == head again add and return
- h3 3 3 + 0

Time: O(N)
Space: O(1)

```
class Node:
    def __init__(self, val=None, next=None):
        self.val = val
        self.next = next

def solution(head, insertVal):
    if not head:
        a = Node(insertVal); a.next = a
        return a
    
    p = head
    while True:
        if p.val <= insertVal <= p.next.val:
            p.next = Node(insertVal, p.next)
            return head
        if p.next.val < p.val and (insertVal >= p.val or insertVal <= p.next.val):
            p.next = Node(insertVal, p.next)
            return head
        p = p.next
        if p == head:
            p.next = Node(insertVal, p.next)
            return head
    
```

# 721. Accounts Merge
Medium

Given a list of accounts where each element accounts[i] is a list of strings, where the first element accounts[i][0] is a name, and the rest of the elements are emails representing emails of the account.

Now, we would like to merge these accounts. Two accounts definitely belong to the same person if there is some common email to both accounts. Note that even if two accounts have the same name, they may belong to different people as people could have the same name. A person can have any number of accounts initially, but all of their accounts definitely have the same name.

After merging the accounts, return the accounts in the following format: the first element of each account is the name, and the rest of the elements are emails in sorted order. The accounts themselves can be returned in any order.

 

Example 1:

Input: accounts = [["John","johnsmith@mail.com","john_newyork@mail.com"],["John","johnsmith@mail.com","john00@mail.com"],["Mary","mary@mail.com"],["John","johnnybravo@mail.com"]]
Output: [["John","john00@mail.com","john_newyork@mail.com","johnsmith@mail.com"],["Mary","mary@mail.com"],["John","johnnybravo@mail.com"]]
Explanation:
The first and second John's are the same person as they have the common email "johnsmith@mail.com".
The third John and Mary are different people as none of their email addresses are used by other accounts.
We could return these lists in any order, for example the answer [['Mary', 'mary@mail.com'], ['John', 'johnnybravo@mail.com'], 
['John', 'john00@mail.com', 'john_newyork@mail.com', 'johnsmith@mail.com']] would still be accepted.

Example 2:

Input: accounts = [["Gabe","Gabe0@m.co","Gabe3@m.co","Gabe1@m.co"],["Kevin","Kevin3@m.co","Kevin5@m.co","Kevin0@m.co"],["Ethan","Ethan5@m.co","Ethan4@m.co","Ethan0@m.co"],["Hanzo","Hanzo3@m.co","Hanzo1@m.co","Hanzo0@m.co"],["Fern","Fern5@m.co","Fern1@m.co","Fern0@m.co"]]
Output: [["Ethan","Ethan0@m.co","Ethan4@m.co","Ethan5@m.co"],["Gabe","Gabe0@m.co","Gabe1@m.co","Gabe3@m.co"],["Hanzo","Hanzo0@m.co","Hanzo1@m.co","Hanzo3@m.co"],["Kevin","Kevin0@m.co","Kevin3@m.co","Kevin5@m.co"],["Fern","Fern0@m.co","Fern1@m.co","Fern5@m.co"]]

 

Constraints:

    1 <= accounts.length <= 1000
    2 <= accounts[i].length <= 10
    1 <= accounts[i][j].length <= 30
    accounts[i][0] consists of English letters.
    accounts[i][j] (for j > 0) is a valid email.



## DFS Traversal + Hashmap

- create a dictionary that maps each email to all possible account ids
- store visited dictionary, so same account isn't visited twice
- iterate through each account's emails
    - if account not visisted
        - dfs through each email's neighrbors using the email to neighbors map
    - return list of account name and sorted emails

Space: O(1)
Time: O(N)

```
def solution(accounts):
    graph = {}
    visited = set()
    ans = []
    for i, a in enumerate(accounts):
        n = a[0]
        for email in a[1:]:
            if email not in graph:
                graph[email] = [i]
            else:
                graph[email].append(i)
    
    def dfs(an, emails):
        if an not in visited:
            visited.add(an)
            for e in range(1, len(accounts[an])):
                email = accounts[an][e]
                emails.add(email)
                for neighbor in graph[email]:
                    dfs(neighbor, emails)
    
    for i, a in enumerate(accounts):
        if i not in visited:
            emails = set()
            dfs(i, emails)
            visited.add(i)
            ans.append([a[0]] + sorted(list(emails)))
            
    return ans
```

# 733. Flood Fill
Easy

An image is represented by an m x n integer grid image where image[i][j] represents the pixel value of the image.

You are also given three integers sr, sc, and color. You should perform a flood fill on the image starting from the pixel image[sr][sc].

To perform a flood fill, consider the starting pixel, plus any pixels connected 4-directionally to the starting pixel of the same color as the starting pixel, plus any pixels connected 4-directionally to those pixels (also with the same color), and so on. Replace the color of all of the aforementioned pixels with color.

Return the modified image after performing the flood fill.

 

Example 1:

Input: image = [[1,1,1],[1,1,0],[1,0,1]], sr = 1, sc = 1, color = 2
Output: [[2,2,2],[2,2,0],[2,0,1]]
Explanation: From the center of the image with position (sr, sc) = (1, 1) (i.e., the red pixel), all pixels connected by a path of the same color as the starting pixel (i.e., the blue pixels) are colored with the new color.
Note the bottom corner is not colored 2, because it is not 4-directionally connected to the starting pixel.

Example 2:

Input: image = [[0,0,0],[0,0,0]], sr = 0, sc = 0, color = 0
Output: [[0,0,0],[0,0,0]]
Explanation: The starting pixel is already colored 0, so no changes are made to the image.

 

Constraints:

    m == image.length
    n == image[i].length
    1 <= m, n <= 50
    0 <= image[i][j], color < 216
    0 <= sr < m
    0 <= sc < n

## DFS 

Perform a standard DFS traversal using the target color. 

```
def floodFill(self, image: List[List[int]], sr: int, sc: int, color: int) -> List[List[int]]:
    target = image[sr][sc]
    visited = [[0 for i in range(len(image[0]))] for j in range(len(image))]

    def traverse(i, j):
        currColor = image[i][j]
        if currColor == target and visited[i][j] != -1:
            visited[i][j] = -1 
            image[i][j] = color 
            neighbors = {(0, 1), (0, -1), (1, 0), (-1, 0)}
            for n in neighbors: 
                di = n[0]
                dj = n[1]
                i1 = i + di
                j1 = j + dj 
                if 0 <= i1 < len(image) and 0 <= j1 < len(image[0]):
                    traverse(i1, j1)

    traverse(sr, sc)
    return image 
```

# 735. Asteroid Collision
Medium

We are given an array asteroids of integers representing asteroids in a row.

For each asteroid, the absolute value represents its size, and the sign represents its direction (positive meaning right, negative meaning left). Each asteroid moves at the same speed.

Find out the state of the asteroids after all collisions. If two asteroids meet, the smaller one will explode. If both are the same size, both will explode. Two asteroids moving in the same direction will never meet.

 

Example 1:

Input: asteroids = [5,10,-5]
Output: [5,10]
Explanation: The 10 and -5 collide resulting in 10. The 5 and 10 never collide.

Example 2:

Input: asteroids = [8,-8]
Output: []
Explanation: The 8 and -8 collide exploding each other.

Example 3:

Input: asteroids = [10,2,-5]
Output: [10]
Explanation: The 2 and -5 collide resulting in -5. The 10 and -5 collide resulting in 10.

 

Constraints:

    2 <= asteroids.length <= 104
    -1000 <= asteroids[i] <= 1000
    asteroids[i] != 0



## Stack

- move right pointer only when conditions bet
    - stack empty
    - stack is non empty
        - value is greater than previous stack value

Space: O(n)
Time: O(n)

```
def solution(asteroids):
    stack = []
    r = 0
    while r < len(asteroids):
        curr = asteroids[r]
        if not stack:
            stack.append(curr)
            r += 1
        else:
            last = stack[-1]
            if curr * last > 0:
                stack.append(curr)
                r += 1
            else:
                if curr > 0 and last < 0:
                    stack.append(curr)
                    r += 1
                else:
                    if abs(curr) > abs(last):
                        stack.pop()
                    elif abs(curr) == abs(last):
                        stack.pop()
                        r += 1
                    else:
                        r += 1
    return stack
```

# 739. Daily Temperatures
Medium
Given an array of integers temperatures represents the daily temperatures, return an array answer such that answer[i] is the number of days you have to wait after the ith day to get a warmer temperature. If there is no future day for which this is possible, keep answer[i] == 0 instead.

 

Example 1:

Input: temperatures = [73,74,75,71,69,72,76,73]
Output: [1,1,4,2,1,1,0,0]
Example 2:

Input: temperatures = [30,40,50,60]
Output: [1,1,1,0]
Example 3:

Input: temperatures = [30,60,90]
Output: [1,1,0]
 

Constraints:

1 <= temperatures.length <= 105
30 <= temperatures[i] <= 100

## Monostack

A monostack is stack where the values in the stack follow a strict ordering. 

Keep a stack of tuples with the temperate of each day, along with the day number. While the stack hash values, if the temperature of the current day is greater than the last element on the stack, then set the distance between days of the last element of the stack to be equal to the difference between the current say and that specific day. Keep popping values off the stack while the condition holds. After the stack has been modified, add the current day and temperature to the stack. Keep going to the end of the array. 

```
def dailyTemperatures(self, temperatures: List[int]) -> List[int]:
    output = [0 for i in range(len(temperatures))]
    stack = []

    for i in range(len(temperatures)): 
        t = temperatures[i]
        while stack and stack[-1][0] < t:
            output[stack[-1][1]] = i - stack[-1][1]
            stack.pop()
        stack.append((t, i))

    return output
```

## Iteration
`

# 743. Network Delay Time
Medium

You are given a network of n nodes, labeled from 1 to n. You are also given times, a list of travel times as directed edges times[i] = (ui, vi, wi), where ui is the source node, vi is the target node, and wi is the time it takes for a signal to travel from source to target.

We will send a signal from a given node k. Return the minimum time it takes for all the n nodes to receive the signal. If it is impossible for all the n nodes to receive the signal, return -1.

 

Example 1:

Input: times = [[2,1,1],[2,3,1],[3,4,1]], n = 4, k = 2
Output: 2

Example 2:

Input: times = [[1,2,1]], n = 2, k = 1
Output: 1

Example 3:

Input: times = [[1,2,1]], n = 2, k = 2
Output: -1

 

Constraints:

    1 <= k <= n <= 100
    1 <= times.length <= 6000
    times[i].length == 3
    1 <= ui, vi <= n
    ui != vi
    0 <= wi <= 100
    All the pairs (ui, vi) are unique. (i.e., no multiple edges.)



## DFS

## Dijkstra (Shortest Path Algorithm)

```

```

# 759. Employee Free Time
Hard

We are given a list schedule of employees, which represents the working time for each employee.

Each employee has a list of non-overlapping Intervals, and these intervals are in sorted order.

Return the list of finite intervals representing common, positive-length free time for all employees, also in sorted order.

(Even though we are representing Intervals in the form [x, y], the objects inside are Intervals, not lists or arrays. For example, schedule[0][0].start = 1, schedule[0][0].end = 2, and schedule[0][0][0] is not defined).  Also, we wouldn't include intervals like [5, 5] in our answer, as they have zero length.

 

Example 1:

Input: schedule = [[[1,2],[5,6]],[[1,3]],[[4,10]]]
Output: [[3,4]]
Explanation: There are a total of three employees, and all common
free time intervals would be [-inf, 1], [3, 4], [10, inf].
We discard any intervals that contain inf as they aren't finite.
Example 2:

Input: schedule = [[[1,3],[6,7]],[[2,4]],[[2,5],[9,12]]]
Output: [[5,6],[7,9]]
 

Constraints:

1 <= schedule.length , schedule[i].length <= 50
0 <= schedule[i].start < schedule[i].end <= 10^8

## Stack

The free time for all employees is equal to the compliment of all times for all employees merged together. Therefore, we can flatten all n employee's intervals into one array, and perform merge intervals on that array. Finally, the solution would be the set of intervals between the non-overlapping intervals. 

Merging the intervals requires us to sort the array by start time, then appending values to a new stack of merged intervals while performing a comparision between each new interval and the last element on the stack.  

Time: O(nlogn)
Space: O(n) 

```
def employeeFreeTime(self, schedule: '[[Interval]]') -> '[Interval]':
        schedule = [interval for intervals in schedule for interval in intervals]

        schedule = sorted(schedule, key=lambda x: x.start)

        stack = []

        for i in schedule: 
            if not stack: 
                stack.append(i)
            else: 
                prevEndTime = stack[-1].end
                currStartTime = i.start 
                if currStartTime <= prevEndTime: 
                    stack[-1].end = max(i.end, stack[-1].end)
                else: 
                    stack.append(i)

        interval = []
        prevEnd = None
        for i in stack: 
            if prevEnd: 
                interval.append(Interval(prevEnd, i.start))
            prevEnd = i.end

        return interval 
```

# 766. Toeplitz Matrix
Easy

Given an m x n matrix, return true if the matrix is Toeplitz. Otherwise, return false.

A matrix is Toeplitz if every diagonal from top-left to bottom-right has the same elements.

 

Example 1:

Input: matrix = [[1,2,3,4],[5,1,2,3],[9,5,1,2]]
Output: true
Explanation:
In the above grid, the diagonals are:
"[9]", "[5, 5]", "[1, 1, 1]", "[2, 2, 2]", "[3, 3]", "[4]".
In each diagonal all elements are the same, so the answer is True.

Example 2:

Input: matrix = [[1,2],[2,2]]
Output: false
Explanation:
The diagonal "[1, 2]" has different elements.

 

Constraints:

    m == matrix.length
    n == matrix[i].length
    1 <= m, n <= 20
    0 <= matrix[i][j] <= 99

 

Follow up:

    What if the matrix is stored on disk, and the memory is limited such that you can only load at most one row of the matrix into the memory at once?
    What if the matrix is so large that you can only load up a partial row into the memory at once?



## Iteration

Space: O(1)
Time: O(n)

```
def solution(matrix):
    for i in range(len(matrix) - 1):
        for j in range(len(matrix[0]) -1):
            if matrix[i][j] != matrix[i+1][j+1]:
                return False
    return True

```

# 818. Race Car
Hard

Your car starts at position 0 and speed +1 on an infinite number line. Your car can go into negative positions. Your car drives automatically according to a sequence of instructions 'A' (accelerate) and 'R' (reverse):

    When you get an instruction 'A', your car does the following:
        position += speed
        speed *= 2
    When you get an instruction 'R', your car does the following:
        If your speed is positive then speed = -1
        otherwise speed = 1
    Your position stays the same.

For example, after commands "AAR", your car goes to positions 0 --> 1 --> 3 --> 3, and your speed goes to 1 --> 2 --> 4 --> -1.

Given a target position target, return the length of the shortest sequence of instructions to get there.

 

Example 1:

Input: target = 3
Output: 2
Explanation: 
The shortest instruction sequence is "AA".
Your position goes from 0 --> 1 --> 3.

Example 2:

Input: target = 6
Output: 5
Explanation: 
The shortest instruction sequence is "AAARA".
Your position goes from 0 --> 1 --> 3 --> 7 --> 7 --> 6.

 

Constraints:

    1 <= target <= 104





```

```

# 849. Maximize Distance to Closest Person
Medium

You are given an array representing a row of seats where seats[i] = 1 represents a person sitting in the ith seat, and seats[i] = 0 represents that the ith seat is empty (0-indexed).

There is at least one empty seat, and at least one person sitting.

Alex wants to sit in the seat such that the distance between him and the closest person to him is maximized. 

Return that maximum distance to the closest person.

 

Example 1:

Input: seats = [1,0,0,0,1,0,1]
Output: 2
Explanation: 
If Alex sits in the second open seat (i.e. seats[2]), then the closest person has distance 2.
If Alex sits in any other open seat, the closest person has distance 1.
Thus, the maximum distance to the closest person is 2.

Example 2:

Input: seats = [1,0,0,0]
Output: 3
Explanation: 
If Alex sits in the last seat (i.e. seats[3]), the closest person is 3 seats away.
This is the maximum distance possible, so the answer is 3.

Example 3:

Input: seats = [0,1]
Output: 1

 

Constraints:

    2 <= seats.length <= 2 * 104
    seats[i] is 0 or 1.
    At least one seat is empty.
    At least one seat is occupied.



## Two Pointer (Optimal)

Time: O(n)
Space: O(1)


## Iteration (Suboptimal)

Complete the length of all intervals of 0's

If the first element is a 1, append a 0 to the beginning of list of interval lengths
If the last element is a 1, append a 0 to the end of the list of interval lengths

Compute the max over all elements of the list of intervals 

Time: O(n)
Space: O(1)

```
def solution(seats):
	interval = 0
	prev = -1
	lengths = []
	for i in range(len(seats)):
		curr = seats[i]
		if curr == 0 and prev == 0:
			interval += 1
		elif curr == 1 and prev == 0: 
			lengths.append(interval)
			interval = 0
		elif curr == 0 and (prev == 1 or prev == -1):  
			interval = 1
			
		prev = curr
	if interval != 0: 
		lengths.append(interval)
		
	maxLength = 0
	
	if seats[0] == 1: 
		lengths.insert(0, 0)
	
	if seats[-1] == 1: 
		lengths.append(0)

	print(lengths)
	
	for i in range(len(lengths)):
		curr = lengths[i]
		if i == 0:
			maxLength = max(maxLength, curr)
		elif i == len(lengths) - 1: 
			maxLength = max(maxLength, curr)
		else: 
			maxLength = max(maxLength, (curr + 1)//2)
			
	return maxLength
```

# 876. Middle of the Linked List
Easy

Given the head of a singly linked list, return the middle node of the linked list.

If there are two middle nodes, return the second middle node.

 

Example 1:

Input: head = [1,2,3,4,5]
Output: [3,4,5]
Explanation: The middle node of the list is node 3.

Example 2:

Input: head = [1,2,3,4,5,6]
Output: [4,5,6]
Explanation: Since the list has two middle nodes with values 3 and 4, we return the second one.

 

Constraints:

    The number of nodes in the list is in the range [1, 100].
    1 <= Node.val <= 100

## Fast and Slow Pointer 

```
def middleNode(self, head: Optional[ListNode]) -> Optional[ListNode]:
    slow = fast = head 

    while fast and fast.next: 
        slow = slow.next 
        fast = fast.next.next 
    return slow
```

# 844. Backspace String Compare
Easy

Given two strings s and t, return true if they are equal when both are typed into empty text editors. '#' means a backspace character.

Note that after backspacing an empty text, the text will continue empty.

 

Example 1:

Input: s = "ab#c", t = "ad#c"
Output: true
Explanation: Both s and t become "ac".

Example 2:

Input: s = "ab##", t = "c#d#"
Output: true
Explanation: Both s and t become "".

Example 3:

Input: s = "a#c", t = "b"
Output: false
Explanation: s becomes "c" while t becomes "b".

 

Constraints:

    1 <= s.length, t.length <= 200
    s and t only contain lowercase letters and '#' characters.

## Stack 

Time: O(N)
Space: O(N)

## Two Pointer

Time: O(N)
Space: O(1)

# 895. Maximum Frequency Stack
Hard

Design a stack-like data structure to push elements to the stack and pop the most frequent element from the stack.

Implement the FreqStack class:

FreqStack() constructs an empty frequency stack.
void push(int val) pushes an integer val onto the top of the stack.
int pop() removes and returns the most frequent element in the stack.
If there is a tie for the most frequent element, the element closest to the stack's top is removed and returned.
 

Example 1:

Input
["FreqStack", "push", "push", "push", "push", "push", "push", "pop", "pop", "pop", "pop"]
[[], [5], [7], [5], [7], [4], [5], [], [], [], []]
Output
[null, null, null, null, null, null, null, 5, 7, 5, 4]

Explanation
FreqStack freqStack = new FreqStack();
freqStack.push(5); // The stack is [5]
freqStack.push(7); // The stack is [5,7]
freqStack.push(5); // The stack is [5,7,5]
freqStack.push(7); // The stack is [5,7,5,7]
freqStack.push(4); // The stack is [5,7,5,7,4]
freqStack.push(5); // The stack is [5,7,5,7,4,5]
freqStack.pop();   // return 5, as 5 is the most frequent. The stack becomes [5,7,5,7,4].
freqStack.pop();   // return 7, as 5 and 7 is the most frequent, but 7 is closest to the top. The stack becomes [5,7,5,4].
freqStack.pop();   // return 5, as 5 is the most frequent. The stack becomes [5,7,4].
freqStack.pop();   // return 4, as 4, 5 and 7 is the most frequent, but 4 is closest to the top. The stack becomes [5,7].
 

Constraints:

0 <= val <= 109
At most 2 * 104 calls will be made to push and pop.
It is guaranteed that there will be at least one element in the stack before calling pop.

## Map of Stacks 

```
class FreqStack:
    def __init__(self):
        self.freq = {}
        self.stacks = {}
        self.maxFreq = 1
        

    def push(self, val: int) -> None:
        if val not in self.freq: 
            self.freq[val] = 1
        else: 
            self.freq[val] += 1 

        f = self.freq[val]

        if f > self.maxFreq: 
            self.maxFreq = f 
        
        if f in self.stacks: 
            self.stacks[f].append(val)
        else: 
            self.stacks[f] = [val]
        
    def pop(self) -> int:
        val = self.stacks[self.maxFreq].pop()
        self.freq[val] -= 1 

        if not self.stacks[self.maxFreq]: 
            self.maxFreq -= 1


        return val 
```

# 904. Fruit Into Baskets
Medium

You are visiting a farm that has a single row of fruit trees arranged from left to right. The trees are represented by an integer array fruits where fruits[i] is the type of fruit the ith tree produces.

You want to collect as much fruit as possible. However, the owner has some strict rules that you must follow:

    You only have two baskets, and each basket can only hold a single type of fruit. There is no limit on the amount of fruit each basket can hold.
    Starting from any tree of your choice, you must pick exactly one fruit from every tree (including the start tree) while moving to the right. The picked fruits must fit in one of your baskets.
    Once you reach a tree with fruit that cannot fit in your baskets, you must stop.

Given the integer array fruits, return the maximum number of fruits you can pick.

 

Example 1:

Input: fruits = [1,2,1]
Output: 3
Explanation: We can pick from all 3 trees.

Example 2:

Input: fruits = [0,1,2,2]
Output: 3
Explanation: We can pick from trees [1,2,2].
If we had started at the first tree, we would only pick from trees [0,1].

Example 3:

Input: fruits = [1,2,3,2,2]
Output: 4
Explanation: We can pick from trees [2,3,2,2].
If we had started at the first tree, we would only pick from trees [1,2].

 

Constraints:

    1 <= fruits.length <= 105
    0 <= fruits[i] < fruits.length

## Two Pointer (Sliding Window)

Keep set of unique elements 

Find largest interval with only two numbers 

Move right pointer to the right until you run into a 3nd unique tree

Left pointer to the index right before the 3rd unique tree

Move left pointer back until not same tree

Move right pointer until new tree

Move left pointer to index before new tree and move back until not same tree

Keep track of longest distance between intervals in between

Time: O(n)
Space: O(1)

### Note: Longest Subarray with 2 Elements, can be generalized to Longest Subarray with K Elements

```
def totalFruit(self, fruits: List[int]) -> int:
    l = 0
    r = 0 
    maxInterval = 0
    
    unique = set()
    for i in range(len(fruits)):
        r = i 
        curr = fruits[r]
        unique.add(curr)
        
        if len(unique) > 2:
            l = r - 1
            prev = fruits[l]
            
            while fruits[l] == prev: 
                l -= 1
                if fruits[l] != prev:
                    unique.remove(fruits[l])
                    l += 1 
                    break
        
        maxInterval = max(maxInterval, r - l + 1)
    
    return maxInterval 
```

# 925. Long Pressed Name
Easy

Your friend is typing his name into a keyboard. Sometimes, when typing a character c, the key might get long pressed, and the character will be typed 1 or more times.

You examine the typed characters of the keyboard. Return True if it is possible that it was your friends name, with some characters (possibly none) being long pressed.

Example 1:

Input: name = "alex", typed = "aaleex"
Output: true
Explanation: 'a' and 'e' in 'alex' were long pressed.

Example 2:

Input: name = "saeed", typed = "ssaaedd"
Output: false
Explanation: 'e' must have been pressed twice, but it was not in the typed output.

 

Constraints:

    1 <= name.length, typed.length <= 1000
    name and typed consist of only lowercase English letters.



## Two pointer

Iterate over ever index of the typed name.

If the index on the original name has not reached the index, and the letter found on both indexes are the same, then increment index on the original name. 

Else if, the index on the type name also isn't equal to the previous index's letter or the index on the typed name is still 0, return False

After iterating over all typed index letters, if the index on the orignal name has reached the right most value, return True. 


Left Word (original), Right Word (typed)

Iterate over all the indices of the right word.






```

```

# 929. Unique Email Addresses
Easy

Every valid email consists of a local name and a domain name, separated by the '@' sign. Besides lowercase letters, the email may contain one or more '.' or '+'.

    For example, in "alice@leetcode.com", "alice" is the local name, and "leetcode.com" is the domain name.

If you add periods '.' between some characters in the local name part of an email address, mail sent there will be forwarded to the same address without dots in the local name. Note that this rule does not apply to domain names.

    For example, "alice.z@leetcode.com" and "alicez@leetcode.com" forward to the same email address.

If you add a plus '+' in the local name, everything after the first plus sign will be ignored. This allows certain emails to be filtered. Note that this rule does not apply to domain names.

    For example, "m.y+name@email.com" will be forwarded to "my@email.com".

It is possible to use both of these rules at the same time.

Given an array of strings emails where we send one email to each emails[i], return the number of different addresses that actually receive mails.

 

Example 1:

Input: emails = ["test.email+alex@leetcode.com","test.e.mail+bob.cathy@leetcode.com","testemail+david@lee.tcode.com"]
Output: 2
Explanation: "testemail@leetcode.com" and "testemail@lee.tcode.com" actually receive mails.

Example 2:

Input: emails = ["a@leetcode.com","b@leetcode.com","c@leetcode.com"]
Output: 3

 

Constraints:

    1 <= emails.length <= 100
    1 <= emails[i].length <= 100
    emails[i] consist of lowercase English letters, '+', '.' and '@'.
    Each emails[i] contains exactly one '@' character.
    All local and domain names are non-empty.
    Local names do not start with a '+' character.
    Domain names end with the ".com" suffix.

## String Matching (Suboptimal)

check if email is valid (don't need to check)
    - doesn't contain multiple @
    - doesn't contain multiple . after @

strip each email to bare bones
    - remove string after + and before @

add email string to set
    
return length of the set

pattern: string, regex, set DS

Time: O(n)
Space: O(1)

```
def numUniqueEmails(self, emails: List[str]) -> int:
    validEmails = set()
    for email in emails: 
        names = email.split("@")
        if len(names) > 2: 
            break 
        localName = names[0]
        domainName = names[1]
        
        filters = localName.split(".")
        localName = "".join(filters)
        
        filters = localName.split("+")
        localName = filters[0]
        
        validEmail = localName + "@" + domainName
        validEmails.add(validEmail)
                
    return len(validEmails)
```

# 977. Squares of a Sorted Array
Easy

Given an integer array nums sorted in non-decreasing order, return an array of the squares of each number sorted in non-decreasing order.

 

Example 1:

Input: nums = [-4,-1,0,3,10]
Output: [0,1,9,16,100]
Explanation: After squaring, the array becomes [16,1,0,9,100].
After sorting, it becomes [0,1,9,16,100].
Example 2:

Input: nums = [-7,-3,2,3,11]
Output: [4,9,9,49,121]
 

Constraints:

1 <= nums.length <= 104
-104 <= nums[i] <= 104
nums is sorted in non-decreasing order.
 

Follow up: Squaring each element and sorting the new array is very trivial, could you find an O(n) solution using a different approach?

## Two Pointer 

Since initial arry of positive and negative values is sorted, if we squared the values, we could retreive the largest to smallest numbers in the array by peeling off the larger number from the left or right ends. To avoid reversing the final array which would store the largest to smallest values, we initialize the output array and back fill the value from right to left to skip this step. 

Time: O(n)
Space: O(1)

```
def sortedSquares(self, nums: List[int]) -> List[int]:
    output = [0 for i in range(len(nums))]
    i = len(nums) - 1
    l = 0 
    r = len(nums) - 1
    while l <= r: 
        if abs(nums[l]) < nums[r]:
            output[i] = nums[r] ** 2
            r -=1 
        else:
            output[i] = nums[l] ** 2
            l += 1
        i -= 1
    return output
```

# 981. Time Based Key-Value Store
Medium

Design a time-based key-value data structure that can store multiple values for the same key at different time stamps and retrieve the key's value at a certain timestamp.

Implement the TimeMap class:

TimeMap() Initializes the object of the data structure.
void set(String key, String value, int timestamp) Stores the key key with the value value at the given time timestamp.
String get(String key, int timestamp) Returns a value such that set was called previously, with timestamp_prev <= timestamp. If there are multiple such values, it returns the value associated with the largest timestamp_prev. If there are no values, it returns "".
 

Example 1:

Input
["TimeMap", "set", "get", "get", "set", "get", "get"]
[[], ["foo", "bar", 1], ["foo", 1], ["foo", 3], ["foo", "bar2", 4], ["foo", 4], ["foo", 5]]
Output
[null, null, "bar", "bar", null, "bar2", "bar2"]

Explanation
TimeMap timeMap = new TimeMap();
timeMap.set("foo", "bar", 1);  // store the key "foo" and value "bar" along with timestamp = 1.
timeMap.get("foo", 1);         // return "bar"
timeMap.get("foo", 3);         // return "bar", since there is no value corresponding to foo at timestamp 3 and timestamp 2, then the only value is at timestamp 1 is "bar".
timeMap.set("foo", "bar2", 4); // store the key "foo" and value "bar2" along with timestamp = 4.
timeMap.get("foo", 4);         // return "bar2"
timeMap.get("foo", 5);         // return "bar2"
 

Constraints:

1 <= key.length, value.length <= 100
key and value consist of lowercase English letters and digits.
1 <= timestamp <= 107
All the timestamps timestamp of set are strictly increasing.
At most 2 * 105 calls will be made to set and get.

## Linear Search 

Perform a linear search through the array of timestamp-value pairs for a given key value. Start the the end of the array, with the largest timestamps, and iterate backward. 

```
class TimeMap:
    def __init__(self):
        self.map = {}
        

    def set(self, key: str, value: str, timestamp: int) -> None:
        if key in self.map: 
            self.map[key].append((timestamp, value))
        else: 
            self.map[key] = [(timestamp, value)]
        

    def get(self, key: str, timestamp: int) -> str:
        if key not in self.map: 
            return ""
        values = self.map[key]
        for v in reversed(values):
            if v[0] <= timestamp: 
                return v[1] 
        return ""
```

## Binary Search 

Perform a binary search through the array of timestamp-value pairs for a given key value. Set the proper midpoint, and keep halfing the search space. 

```
class TimeMap:
    def __init__(self):
        self.map = {}
        

    def set(self, key: str, value: str, timestamp: int) -> None:
        if key in self.map: 
            self.map[key].append((timestamp, value))
        else: 
            self.map[key] = [(timestamp, value)]
        

    def get(self, key: str, timestamp: int) -> str:
        if key not in self.map: 
            return ""
        
        if timestamp < self.map[key][0][0]:
            return ""

        values = self.map[key]
        l = 0
        r = len(values)
        while l < r: 
            mid = (l + r)//2
            if values[mid][0] <= timestamp: 
                l = mid + 1
            else: 
                r = mid
        
        if r == 0: return ""
        return values[r-1][1]
```

# 1025. Divisor Game
Easy

Alice and Bob take turns playing a game, with Alice starting first.

Initially, there is a number n on the chalkboard. On each player's turn, that player makes a move consisting of:

    Choosing any x with 0 < x < n and n % x == 0.
    Replacing the number n on the chalkboard with n - x.

Also, if a player cannot make a move, they lose the game.

Return true if and only if Alice wins the game, assuming both players play optimally.

 

Example 1:

Input: n = 2
Output: true
Explanation: Alice chooses 1, and Bob has no more moves.

Example 2:

Input: n = 3
Output: false
Explanation: Alice chooses 1, Bob chooses 1, and Alice has no more moves.

 

Constraints:

    1 <= n <= 1000



## Dynamic Programming

## Number Theory 



```

```

# 1143. Longest Common Subsequence
Medium

Given two strings text1 and text2, return the length of their longest common subsequence. If there is no common subsequence, return 0.

A subsequence of a string is a new string generated from the original string with some characters (can be none) deleted without changing the relative order of the remaining characters.

    For example, "ace" is a subsequence of "abcde".

A common subsequence of two strings is a subsequence that is common to both strings.

 

Example 1:

Input: text1 = "abcde", text2 = "ace" 
Output: 3  
Explanation: The longest common subsequence is "ace" and its length is 3.

Example 2:

Input: text1 = "abc", text2 = "abc"
Output: 3
Explanation: The longest common subsequence is "abc" and its length is 3.

Example 3:

Input: text1 = "abc", text2 = "def"
Output: 0
Explanation: There is no such common subsequence, so the result is 0.

## Dynamic Programming (Optimal): 

To solve the larger common subsequence problem, solve subproblems of substrings. In particular, there exists a recursive relation between smaller substrings:

1. If letters match in the substring, increment the longest subsequence by one value.

dp[i][j] = dp[i + 1][j + 1] + 1


2. If letters don't match, save value of the longest subsequence from removing one letter form either other letter.

dp[i][j] = max(dp[i + 1][j], dp[i][j+1])

```
def longestCommonSubsequence(self, text1: str, text2: str) -> int:
    dp = [[0 for i in range(len(text2) + 1)] for j in range(len(text1) + 1)]

    for i in reversed(range(len(text1))):
        for j in reversed(range(len(text2))):
            if text1[i] == text2[j]:
                dp[i][j] = dp[i + 1][j + 1] + 1
            else: 
                dp[i][j] = max(dp[i + 1][j], dp[i][j + 1])

    return dp[0][0]
```

Examples
```
s1 = 'bad'
s2 = 'cda

+ b a d +
c 1 1 1 0
d 1 1 1 0 
a 1 1 0 0 
+ 0 0 0 0
```

# 1197. Minimum Knight Moves
Medium

In an infinite chess board with coordinates from -infinity to +infinity, you have a knight at square [0, 0].

A knight has 8 possible moves it can make, as illustrated below. Each move is two squares in a cardinal direction, then one square in an orthogonal direction.


Return the minimum number of steps needed to move the knight to the square [x, y]. It is guaranteed the answer exists.

 

Example 1:

Input: x = 2, y = 1
Output: 1
Explanation: [0, 0] → [2, 1]
Example 2:

Input: x = 5, y = 5
Output: 4
Explanation: [0, 0] → [2, 1] → [4, 2] → [3, 4] → [5, 5]
 

Constraints:

-300 <= x, y <= 300
0 <= |x| + |y| <= 300

# DFS 

```
def minKnightMoves(self, x: int, y: int) -> int:
    @lru_cache
    def search(x, y): 
        if x + y == 0: 
            return 0
        elif x + y == 2: 
            return 2
        else:
            return min(search(abs(x-1), abs(y-2)), search(abs(x-2), abs(y-1))) + 1
    
    return search(abs(x), abs(y))
```

# 1213. Intersection of Three Sorted Arrays
Easy

Given three integer arrays arr1, arr2 and arr3 sorted in strictly increasing order, return a sorted array of only the integers that appeared in all three arrays.

 

Example 1:

Input: arr1 = [1,2,3,4,5], arr2 = [1,2,5,7,9], arr3 = [1,3,4,5,8]
Output: [1,5]
Explanation: Only 1 and 5 appeared in the three arrays.

Example 2:

Input: arr1 = [197,418,523,876,1356], arr2 = [501,880,1593,1710,1870], arr3 = [521,682,1337,1395,1764]
Output: []

 

Constraints:

    1 <= arr1.length, arr2.length, arr3.length <= 1000
    1 <= arr1[i], arr2[i], arr3[i] <= 2000

## Three Set (Sub-optimal)
Add elements of first arr1 and second arr2 into separate sets
        
Find the intersection of the sets

Add elements of third arr3

Find intersection with previous intersection

Return set converted into array and sorted

Time: O(n)
Space: O(n)

```
def arraysIntersection(self, arr1: List[int], arr2: List[int], arr3: List[int]) -> List[int]:
    set1 = set(arr1)
    set2 = set(arr2)
    intersection = set1.intersection(set2)
    set3 = set(arr3)
    return sorted(list(intersection.intersection(set3)))
```

## Three Hashmap (Sub-Optimal)
Since array's are strictly increasing, duplicates values within an array can be ruled out

Time: O(n)
Space: O(n)

## Three Pointer (Optimal)
Start all pointers at 0. If all pointers point to same value, increment count. 

If not: 
    if p1 < p2, then p1 +=1
    if p2 < p3, then p2 +=1
    else, p3 += 1

Time: O(n)
Space: O(1)



# 1375. Number of Times Binary String Is Prefix-Aligned
Medium

You have a 1-indexed binary string of length n where all the bits are 0 initially. We will flip all the bits of this binary string (i.e., change them from 0 to 1) one by one. You are given a 1-indexed integer array flips where flips[i] indicates that the bit at index i will be flipped in the ith step.

A binary string is prefix-aligned if, after the ith step, all the bits in the inclusive range [1, i] are ones and all the other bits are zeros.

Return the number of times the binary string is prefix-aligned during the flipping process.

 

Example 1:

Input: flips = [3,2,4,1,5]
Output: 2
Explanation: The binary string is initially "00000".
After applying step 1: The string becomes "00100", which is not prefix-aligned.
After applying step 2: The string becomes "01100", which is not prefix-aligned.
After applying step 3: The string becomes "01110", which is not prefix-aligned.
After applying step 4: The string becomes "11110", which is prefix-aligned.
After applying step 5: The string becomes "11111", which is prefix-aligned.
We can see that the string was prefix-aligned 2 times, so we return 2.

Example 2:

Input: flips = [4,1,2,3]
Output: 1
Explanation: The binary string is initially "0000".
After applying step 1: The string becomes "0001", which is not prefix-aligned.
After applying step 2: The string becomes "1001", which is not prefix-aligned.
After applying step 3: The string becomes "1101", which is not prefix-aligned.
After applying step 4: The string becomes "1111", which is prefix-aligned.
We can see that the string was prefix-aligned 1 time, so we return 1.

 

Constraints:

    n == flips.length
    1 <= n <= 5 * 104
    flips is a permutation of the integers in the range [1, n].



## Intelligent Iteration 

Compute the maximum flip number so far, as iterating through the array flips. If the current max flip number is equal to the index, add to a counter. Return the final count


If the current max flip number equals the current index of the flip array, that means that there are a total of index numbers of which the max is the index. This is equivalent to saying that 1, 2, ... , flip are thre previous numbers. The counter simply counts all the number of times that condition is met throughout flips. 

Space: O(n)
Time: O(1)

```
from math import inf
def solution(flips):
    maxSoFar = -inf
    count = 0
    for index in range(len(flips)):
        flip = flips[index]
        maxSoFar = max(maxSoFar, flip)
        if maxSoFar == index + 1: 
            count += 1
            
    return count
```

# 1466. Reorder Routes to Make All Paths Lead to the City Zero
Medium

There are n cities numbered from 0 to n - 1 and n - 1 roads such that there is only one way to travel between two different cities (this network form a tree). Last year, The ministry of transport decided to orient the roads in one direction because they are too narrow.

Roads are represented by connections where connections[i] = [ai, bi] represents a road from city ai to city bi.

This year, there will be a big event in the capital (city 0), and many people want to travel to this city.

Your task consists of reorienting some roads such that each city can visit the city 0. Return the minimum number of edges changed.

It's guaranteed that each city can reach city 0 after reorder.

 

Example 1:

Input: n = 6, connections = [[0,1],[1,3],[2,3],[4,0],[4,5]]
Output: 3
Explanation: Change the direction of edges show in red such that each node can reach the node 0 (capital).

Example 2:

Input: n = 5, connections = [[1,0],[1,2],[3,2],[3,4]]
Output: 2
Explanation: Change the direction of edges show in red such that each node can reach the node 0 (capital).

Example 3:

Input: n = 3, connections = [[1,0],[2,0]]
Output: 0

 

Constraints:

    2 <= n <= 5 * 104
    connections.length == n - 1
    connections[i].length == 2
    0 <= ai, bi <= n - 1
    ai != bi



## DFS Traversal

Generate matrix as adjacency list 

Time: O(n)
Space: O(n)



```
def solution(n, connections):
    graph = {}
    connectionSet = set()
    for connection in connections:
        start = connection[0]
        end = connection[1]
        if end in graph:
            graph[end].add(start)
        else:
            graph[end] = set()
            graph[end].add(start)
            
        if start in graph: 
            graph[start].add(end)
        else:
            graph[start] = set()
            graph[start].add(end)
            
        connectionSet.add((start, end))

    count = 0 
    visited = set()
    def traverse(root):
        nonlocal count
        nonlocal visited
        nonlocal connectionSet
        visited.add(root)
        for neighbor in graph[root]:
            if neighbor not in visited:
                
                
                traverse(neighbor)
                if (neighbor, root) not in connectionSet:
                    count += 1
                
                
    
    traverse(0)
    return count
```

# 1578. Minimum Time to Make Rope Colorful
Medium

Alice has n balloons arranged on a rope. You are given a 0-indexed string colors where colors[i] is the color of the ith balloon.

Alice wants the rope to be colorful. She does not want two consecutive balloons to be of the same color, so she asks Bob for help. Bob can remove some balloons from the rope to make it colorful. You are given a 0-indexed integer array neededTime where neededTime[i] is the time (in seconds) that Bob needs to remove the ith balloon from the rope.

Return the minimum time Bob needs to make the rope colorful.

 

Example 1:

Input: colors = "abaac", neededTime = [1,2,3,4,5]
Output: 3
Explanation: In the above image, 'a' is blue, 'b' is red, and 'c' is green.
Bob can remove the blue balloon at index 2. This takes 3 seconds.
There are no longer two consecutive balloons of the same color. Total time = 3.

Example 2:

Input: colors = "abc", neededTime = [1,2,3]
Output: 0
Explanation: The rope is already colorful. Bob does not need to remove any balloons from the rope.

Example 3:

Input: colors = "aabaa", neededTime = [1,2,3,4,1]
Output: 2
Explanation: Bob will remove the ballons at indices 0 and 4. Each ballon takes 1 second to remove.
There are no longer two consecutive balloons of the same color. Total time = 1 + 1 = 2.

 

Constraints:

    n == colors.length == neededTime.length
    1 <= n <= 105
    1 <= neededTime[i] <= 104

## Iteration (Sub-optimal)

Iterate through the elements, and find the sum of all adjacent elements that have the same number. Keep track of the max of the adjacent values, and subtract that value from the total tracking sum. 

Use a prev boolean to keep track of the values. 

Time: O(n)
Space: O(1)

## Two Pointer (Optimal)

Time: O(n)
Space: O(1)

```
def solution(colors, neededTime):
	cPrev = -1
	tPrev = -1
	prevMax = -1
	equalToPrev = False
	totalSum = 0 
	for i in range(len(colors)):
		c = colors[i]
		t = neededTime[i]
		if equalToPrev == True and c != cPrev: 
			totalSum -= prevMax
			equalToPrev = False 
		if equalToPrev == True and c == cPrev:
			prevMax = max(t, prevMax)
			totalSum += t
			equalToPrev = True        
		if equalToPrev == False and c == cPrev:
			prevMax = max(t, tPrev)
			totalSum += t
			totalSum += tPrev
			equalToPrev = True       
		cPrev = c
		tPrev = t 
	if equalToPrev == True: 
		totalSum -= prevMax
		
	return totalSum
```

# 1615. Maximal Network Rank
Medium

There is an infrastructure of n cities with some number of roads connecting these cities. Each roads[i] = [ai, bi] indicates that there is a bidirectional road between cities ai and bi.

The network rank of two different cities is defined as the total number of directly connected roads to either city. If a road is directly connected to both cities, it is only counted once.

The maximal network rank of the infrastructure is the maximum network rank of all pairs of different cities.

Given the integer n and the array roads, return the maximal network rank of the entire infrastructure.

 

Example 1:

Input: n = 4, roads = [[0,1],[0,3],[1,2],[1,3]]
Output: 4
Explanation: The network rank of cities 0 and 1 is 4 as there are 4 roads that are connected to either 0 or 1. The road between 0 and 1 is only counted once.

Example 2:

Input: n = 5, roads = [[0,1],[0,3],[1,2],[1,3],[2,3],[2,4]]
Output: 5
Explanation: There are 5 roads that are connected to cities 1 or 2.

Example 3:

Input: n = 8, roads = [[0,1],[1,2],[2,3],[2,4],[5,6],[5,7]]
Output: 5
Explanation: The network rank of 2 and 5 is 5. Notice that all the cities do not have to be connected.

 

Constraints:

    2 <= n <= 100
    0 <= roads.length <= n * (n - 1) / 2
    roads[i].length == 2
    0 <= ai, bi <= n-1
    ai != bi
    Each pair of cities has at most one road connecting them.



## Brute Force (Optimal)

Time: O(n^2)
Space: O(n)

```
def solution(n, roads): 
	count = {}
	for i in range(len(roads)):
		a = roads[i][0]
		b = roads[i][1]
		if a not in count: 
			count[a] = 1
		else: 
			count[a] += 1
		
		if b not in count: 
			count[b] = 1
		else: 
			count[b] += 1

	maxRank = 0 

	adjMatrix = set()
	for i in range(len(roads)): 
		a = roads[i][0]
		b = roads[i][1]
		
		adjMatrix.add((a, b))
		adjMatrix.add((b, a))
	
	for i in count: 
		for j in count:
			if i != j: 
				if ((i, j) in adjMatrix) or ((j, i) in adjMatrix):
					maxRank = max(maxRank, count[i] + count[j] - 1)
				else: 
					maxRank = max(maxRank, count[i] + count[j])

	return maxRank
```

# 1641. Count Sorted Vowel Strings
Medium

Given an integer n, return the number of strings of length n that consist only of vowels (a, e, i, o, u) and are lexicographically sorted.

A string s is lexicographically sorted if for all valid i, s[i] is the same as or comes before s[i+1] in the alphabet.

 

Example 1:

Input: n = 1
Output: 5
Explanation: The 5 sorted strings that consist of vowels only are ["a","e","i","o","u"].

Example 2:

Input: n = 2
Output: 15
Explanation: The 15 sorted strings that consist of vowels only are
["aa","ae","ai","ao","au","ee","ei","eo","eu","ii","io","iu","oo","ou","uu"].
Note that "ea" is not a valid string since 'e' comes after 'a' in the alphabet.

Example 3:

Input: n = 33
Output: 66045




```
'''
Q: Final all lexographically sorted strings equal to n, made solely from vowels

Brute Force: Try to find the answer combinatorially

Dynamic Programming: 
n = 1, totalStrings = 5
n + 1 = n + 
Set base case to
'''

def solution():
    return
```

# 1730. Shortest Path to Get Food
Medium

You are starving and you want to eat food as quickly as possible. You want to find the shortest path to arrive at any food cell.

You are given an m x n character matrix, grid, of these different types of cells:

'*' is your location. There is exactly one '*' cell.
'#' is a food cell. There may be multiple food cells.
'O' is free space, and you can travel through these cells.
'X' is an obstacle, and you cannot travel through these cells.
You can travel to any adjacent cell north, east, south, or west of your current location if there is not an obstacle.

Return the length of the shortest path for you to reach any food cell. If there is no path for you to reach food, return -1.

 

Example 1:


Input: grid = [["X","X","X","X","X","X"],["X","*","O","O","O","X"],["X","O","O","#","O","X"],["X","X","X","X","X","X"]]
Output: 3
Explanation: It takes 3 steps to reach the food.
Example 2:


Input: grid = [["X","X","X","X","X"],["X","*","X","O","X"],["X","O","X","#","X"],["X","X","X","X","X"]]
Output: -1
Explanation: It is not possible to reach the food.
Example 3:


Input: grid = [["X","X","X","X","X","X","X","X"],["X","*","O","X","O","#","O","X"],["X","O","O","X","O","O","X","X"],["X","O","O","O","O","#","O","X"],["X","X","X","X","X","X","X","X"]]
Output: 6
Explanation: There can be multiple food cells. It only takes 6 steps to reach the bottom food.
 

Constraints:

m == grid.length
n == grid[i].length
1 <= m, n <= 200
grid[row][col] is '*', 'X', 'O', or '#'.
The grid contains exactly one '*'.

## BFS

```
def getFood(self, grid: List[List[str]]) -> int:
    queue = []
    for i in range(len(grid)):
        for j in range(len(grid[0])): 
            if grid[i][j] == "*": 
                queue.insert(0, (i, j, 0))
                grid[i][j] = "X"

    
    while queue: 
        i, j, d = queue.pop()
        
        neighbors = {(0, 1), (1, 0), (-1, 0), (0, -1)}
        for n in neighbors: 
            di = i + n[0]
            dj = j + n[1]
            if 0 <= di < len(grid) and 0 <= dj < len(grid[0]) and grid[di][dj] in {"O", "#"}: 
                if grid[di][dj] == "O": 
                    queue.insert(0, (di, dj, d + 1))
                if grid[di][dj] == "#":
                    return d + 1
                grid[di][dj] = "X"
                    
    return -1 
```

# 1881. Maximum Value after Insertion
Medium

You are given a very large integer n, represented as a string,​​​​​​ and an integer digit x. The digits in n and the digit x are in the inclusive range [1, 9], and n may represent a negative number.

You want to maximize n's numerical value by inserting x anywhere in the decimal representation of n​​​​​​. You cannot insert x to the left of the negative sign.

    For example, if n = 73 and x = 6, it would be best to insert it between 7 and 3, making n = 763.
    If n = -55 and x = 2, it would be best to insert it before the first 5, making n = -255.

Return a string representing the maximum value of n​​​​​​ after the insertion.

 

Example 1:

Input: n = "99", x = 9
Output: "999"
Explanation: The result is the same regardless of where you insert 9.

Example 2:

Input: n = "-13", x = 2
Output: "-123"
Explanation: You can make n one of {-213, -123, -132}, and the largest of those three is -123.

 

Constraints:

    1 <= n.length <= 105
    1 <= x <= 9
    The digits in n​​​ are in the range [1, 9].
    n is a valid representation of an integer.
    In the case of a negative n,​​​​​​ it will begin with '-'.



## Brute Force (Suboptimal)

Create a list of all valid numbers, adding a digit in a specific holder

Return the maximum value from the list of all valid numbers

Time: O(n)
Space: O(n)

## Iteration (Optimal)

Iterate through each digit, and add the current digit if it is greater than or less than the the digit

Move from left to right of each digit and compare the current digit with the digit of the current index

If negative, add to left if current digit is greater than digit at current index

If positive, add to left if current digit is less than digit at current index 

Otherwise, add to right most position 

Time: O(n)
Space: O(1)

```
def solution(n, x): 
	if n[0] == "-":
		for i in range(1, len(n)):
			if int(n[i]) > x:
				n = n[:i] + str(x) + n[i:]
				return n
		return n + str(x)  
	else: 
		for i in range(len(n)):
			if int(n[i]) < x: 
				n = n[:i] + str(x) + n[i:]
				return n 
		return n + str(x)
```
# 2361. Minimum Costs Using the Train Line
Hard
66
13
company
Citadel

A train line going through a city has two routes, the regular route and the express route. Both routes go through the same n + 1 stops labeled from 0 to n. Initially, you start on the regular route at stop 0.

You are given two 1-indexed integer arrays regular and express, both of length n. regular[i] describes the cost it takes to go from stop i - 1 to stop i using the regular route, and express[i] describes the cost it takes to go from stop i - 1 to stop i using the express route.

You are also given an integer expressCost which represents the cost to transfer from the regular route to the express route.

Note that:

    There is no cost to transfer from the express route back to the regular route.
    You pay expressCost every time you transfer from the regular route to the express route.
    There is no extra cost to stay on the express route.

Return a 1-indexed array costs of length n, where costs[i] is the minimum cost to reach stop i from stop 0.

Note that a stop can be counted as reached from either route.

 

Example 1:

Input: regular = [1,6,9,5], express = [5,2,3,10], expressCost = 8
Output: [1,7,14,19]
Explanation: The diagram above shows how to reach stop 4 from stop 0 with minimum cost.
- Take the regular route from stop 0 to stop 1, costing 1.
- Take the express route from stop 1 to stop 2, costing 8 + 2 = 10.
- Take the express route from stop 2 to stop 3, costing 3.
- Take the regular route from stop 3 to stop 4, costing 5.
The total cost is 1 + 10 + 3 + 5 = 19.
Note that a different route could be taken to reach the other stops with minimum cost.

Example 2:

Input: regular = [11,5,13], express = [7,10,6], expressCost = 3
Output: [10,15,24]
Explanation: The diagram above shows how to reach stop 3 from stop 0 with minimum cost.
- Take the express route from stop 0 to stop 1, costing 3 + 7 = 10.
- Take the regular route from stop 1 to stop 2, costing 5.
- Take the express route from stop 2 to stop 3, costing 3 + 6 = 9.
The total cost is 10 + 5 + 9 = 24.
Note that the expressCost is paid again to transfer back to the express route.

 

Constraints:

    n == regular.length == express.length
    1 <= n <= 105
    1 <= regular[i], express[i], expressCost <= 105

## BFS (Suboptimal & Incorrect)

Modify the regular indexes to (index+1) * 2 (even)
Modify the express indexes to (index+1) * 2 - 1 (odd)

Apply djikstra's on the graph

For each index, return the min((index + 1) * 2, (index + 1) * 2 - 1)

## Dynamic Programming - 1D variant

3 possible steps at any given moment: regular, express, regular + toll 

- two paths 
    - (1) move forward, then switch 
    - (2)switch, then forward

code for both approaches provided below

Time: O(n)
Space: O(n)


```
def minimumCosts(self, regular: List[int], express: List[int], expressCost: int) -> List[int]:
        dpr = [0] * (len(regular) + 1)
        dpe = [0] * (len(regular) + 1)
        dpe[0] = expressCost
        output = [0] * len(regular)
        for i in range(1, len(regular) + 1):
            #Switch Forward
            #dpr[i] = min(dpr[i - 1] + regular[i-1], dpe[i - 1] + regular[i-1])
            #dpe[i] = min(dpr[i-1] + expressCost + express[i-1], dpe[i-1] + express[i-1])
            #Forward Switch 
            dpr[i] = min(dpr[i-1] + regular[i-1], dpe[i-1] + express[i-1])
            dpe[i] = min(dpr[i-1] + regular[i-1] + expressCost, dpe[i-1] + express[i-1])
            output[i - 1] = min(dpr[i], dpe[i])
        return output
```

# 2416. Sum of Prefix Scores of Strings
Hard

You are given an array words of size n consisting of non-empty strings.

We define the score of a string word as the number of strings words[i] such that word is a prefix of words[i].

    For example, if words = ["a", "ab", "abc", "cab"], then the score of "ab" is 2, since "ab" is a prefix of both "ab" and "abc".

Return an array answer of size n where answer[i] is the sum of scores of every non-empty prefix of words[i].

Note that a string is considered as a prefix of itself.

 

Example 1:

Input: words = ["abc","ab","bc","b"]
Output: [5,4,3,2]
Explanation: The answer for each string is the following:
- "abc" has 3 prefixes: "a", "ab", and "abc".
- There are 2 strings with the prefix "a", 2 strings with the prefix "ab", and 1 string with the prefix "abc".
The total is answer[0] = 2 + 2 + 1 = 5.
- "ab" has 2 prefixes: "a" and "ab".
- There are 2 strings with the prefix "a", and 2 strings with the prefix "ab".
The total is answer[1] = 2 + 2 = 4.
- "bc" has 2 prefixes: "b" and "bc".
- There are 2 strings with the prefix "b", and 1 string with the prefix "bc".
The total is answer[2] = 2 + 1 = 3.
- "b" has 1 prefix: "b".
- There are 2 strings with the prefix "b".
The total is answer[3] = 2.

Example 2:

Input: words = ["abcd"]
Output: [4]
Explanation:
"abcd" has 4 prefixes: "a", "ab", "abc", and "abcd".
Each prefix has a score of one, so the total is answer[0] = 1 + 1 + 1 + 1 = 4.

 

Constraints:

    1 <= words.length <= 1000
    1 <= words[i].length <= 1000
    words[i] consists of lowercase English letters.



## Trie

Time: O(N)
Space: O(N)

```
class Trie: 
    def __init__(self):
        self.trie = {}
        self.count = 0
class Solution:
    def sumPrefixScores(self, words: List[str]) -> List[int]:
        trie = Trie()
        for word in words: 
            currTrie = trie
            for letter in word: 
                if letter not in currTrie.trie: 
                    currTrie.trie[letter] = Trie()
                currTrie.trie[letter].count += 1
                currTrie = currTrie.trie[letter]
            
        prefix = [0 for _ in range(len(words))]
        for index in range(len(words)): 
            word = words[index]
            currTrie = trie
            for letter in word: 
                prefix[index] += currTrie.trie[letter].count
                currTrie = currTrie.trie[letter]
        return prefix
```

# 2455. Average Value of Even Numbers That Are Divisible by Three

Given an integer array nums of positive integers, return the average value of all even integers that are divisible by 3.

Note that the average of n elements is the sum of the n elements divided by n and rounded down to the nearest integer.

 

Example 1:

Input: nums = [1,3,6,10,12,15]
Output: 9
Explanation: 6 and 12 are even numbers that are divisible by 3. (6 + 12) / 2 = 9.

Example 2:

Input: nums = [1,2,4,7,10]
Output: 0
Explanation: There is no single number that satisfies the requirement, so return 0.

 

Constraints:

    1 <= nums.length <= 1000
    1 <= nums[i] <= 1000



## Iteration

Time: O(N)
Space: O(1)

```
def solution(nums):
    currSum = 0 
    count = 0
    for n in nums: 
        if n % 2 == 0 and n % 3 == 0: 
            currSum += n
            count += 1
    return currSum / count
```

# 2456. Most Popular Video Creator

You are given two string arrays creators and ids, and an integer array views, all of length n. The ith video on a platform was created by creator[i], has an id of ids[i], and has views[i] views.

The popularity of a creator is the sum of the number of views on all of the creator's videos. Find the creator with the highest popularity and the id of their most viewed video.

    If multiple creators have the highest popularity, find all of them.
    If multiple videos have the highest view count for a creator, find the lexicographically smallest id.

Return a 2D array of strings answer where answer[i] = [creatori, idi] means that creatori has the highest popularity and idi is the id of their most popular video. The answer can be returned in any order.

 

Example 1:

Input: creators = ["alice","bob","alice","chris"], ids = ["one","two","three","four"], views = [5,10,5,4]
Output: [["alice","one"],["bob","two"]]
Explanation:
The popularity of alice is 5 + 5 = 10.
The popularity of bob is 10.
The popularity of chris is 4.
alice and bob are the most popular creators.
For bob, the video with the highest view count is "two".
For alice, the videos with the highest view count are "one" and "three". Since "one" is lexicographically smaller than "three", it is included in the answer.

Example 2:

Input: creators = ["alice","alice","alice"], ids = ["a","b","c"], views = [1,2,2]
Output: [["alice","b"]]
Explanation:
The videos with id "b" and "c" have the highest view count.
Since "b" is lexicographically smaller than "c", it is included in the answer.

 

Constraints:

    n == creators.length == ids.length == views.length
    1 <= n <= 105
    1 <= creators[i].length, ids[i].length <= 5
    creators[i] and ids[i] consist only of lowercase English letters.
    0 <= views[i] <= 105



## BF

Find sum of all creator videos, then return creator with max sum (hashmap turned to list)

Collect all views associated with creator, then return max

Time: O(N)
Space: O(N) s.t. n is number of creators

## Optimal
Time: O(N)
Space: O(1)
        


```
def solution(creators, ids, views):
    creatorViews = {}
    
    for index in range(len(creators)):
        creator = creators[index]
        if creator not in creatorViews:
            creatorViews[creator] = views[index]
        else:
            creatorViews[creator] += views[index]
    
    keys, values = creatorViews.items()
    orderedViews = []
    for index in range(len(keys)):
        orderedViews.append((keys[index], values[index]))

    orderedViews.sort()
    
    famousCreator = orderedViews[-1][1]
    famousIds = []
    for index in range(len(creators)):
        if creators[index] == famousCreator: 
            famousIds.append((views[index], ids[index]))
        
    famoudIds.sort()
    return famousIds[-1][1]
    
```

# 2461. Maximum Sum of Distinct Subarrays With Length K

You are given an integer array nums and an integer k. Find the maximum subarray sum of all the subarrays of nums that meet the following conditions:

    The length of the subarray is k, and
    All the elements of the subarray are distinct.

Return the maximum subarray sum of all the subarrays that meet the conditions. If no subarray meets the conditions, return 0.

A subarray is a contiguous non-empty sequence of elements within an array.

 

Example 1:

Input: nums = [1,5,4,2,9,9,9], k = 3
Output: 15
Explanation: The subarrays of nums with length 3 are:
- [1,5,4] which meets the requirements and has a sum of 10.
- [5,4,2] which meets the requirements and has a sum of 11.
- [4,2,9] which meets the requirements and has a sum of 15.
- [2,9,9] which does not meet the requirements because the element 9 is repeated.
- [9,9,9] which does not meet the requirements because the element 9 is repeated.
We return 15 because it is the maximum subarray sum of all the subarrays that meet the conditions

Example 2:

Input: nums = [4,4,4], k = 3
Output: 0
Explanation: The subarrays of nums with length 3 are:
- [4,4,4] which does not meet the requirements because the element 4 is repeated.
We return 0 because no subarrays meet the conditions.

 

Constraints:

    1 <= k <= nums.length <= 105
    1 <= nums[i] <= 105



## Prefix Sum Set 
Compute prefix sum of values in the array 


```

```

# 2465. Number of Distinct Averages

You are given a 0-indexed integer array nums of even length.

As long as nums is not empty, you must repetitively:

    Find the minimum number in nums and remove it.
    Find the maximum number in nums and remove it.
    Calculate the average of the two removed numbers.

The average of two numbers a and b is (a + b) / 2.

    For example, the average of 2 and 3 is (2 + 3) / 2 = 2.5.

Return the number of distinct averages calculated using the above process.

Note that when there is a tie for a minimum or maximum number, any can be removed.

 

Example 1:

Input: nums = [4,1,4,0,3,5]
Output: 2
Explanation:
1. Remove 0 and 5, and the average is (0 + 5) / 2 = 2.5. Now, nums = [4,1,4,3].
2. Remove 1 and 4. The average is (1 + 4) / 2 = 2.5, and nums = [4,3].
3. Remove 3 and 4, and the average is (3 + 4) / 2 = 3.5.
Since there are 2 distinct numbers among 2.5, 2.5, and 3.5, we return 2.

Example 2:

Input: nums = [1,100]
Output: 1
Explanation:
There is only one average to be calculated after removing 1 and 100, so we return 1.

 

Constraints:

    2 <= nums.length <= 100
    nums.length is even.
    0 <= nums[i] <= 100



## Sort + Two Pointer + Set

Time: O(nlogn)
Space: O(n)

```
def distinctAverages(nums) -> int:
    nums.sort()
    l = 0
    r = len(nums) - 1
    s = set()
    while (l < r):
        s.add((nums[l] + nums[r])/ 2)
        l += 1
        r -= 1
    return len(s)
```

# My Notes

Below are key definitions, tricks, tips, and code templates for common patterns. 

# DFS (Depth First Search)

Definition: Search deep into a graph before searching all immediate neighbors. 
Application: When you need to traverse a tree or graph, searching for a specific heuristic. 

```
def dfs(root):
    if not root: 
        return
    else:
        for child in root.children: 
            if condition(child): 
                dfs(child)
```

- Variants + Problems
    - Graph Search
        - (417) Pacific Atlantic Water Flow     
            - Heuristic = Height 
	- Connected Components 
		- (547) Number of Provinces
        - (323) Number of Connected Components in an Undirected Graph
    - String 
        - (139) Word Break
    - Topological Sort
        - ()

# BFS (Breadth First Search)

Definition: Search all immediate neighbors before searching deep into a graph. 
Application: When it is optimal to search a graph-like data structure in level order

- Variants + Problems
	- General Search 
		- (200) Number of Islands 
	- Djikstra's Algorithm
		- (743) Network Delay Time 
    - Topological Sort

Code Templates
```
def bfs(root):
    queue = [root]
    while queue: 
        nextNode = queue.popleft()
        queue.append(nextNode.left)
        queue.append(nextNode.right)
```

# Graph Traversal

- Graph Representations: 
    - Adjacency List - O(N)
    - Ajacency Matrix - O(N^2)

- Variants + Problems 
	- Accounts Merge
	- Making a Large Island

Code Templates
```
def buildAdjacencyList(edges):
    graph = {}
    for edge in edges:
        start = edge[0]
        end = edge[1]
        if start not in graph: 
            graph[start] = [end]
        else:
            graph[start].append(end)
```

# Backtrack
	
Keep track of visited data structure to prune search 

- Variants + Problems 
	- Stack + String
		- (79) Word Break 
		- (140) Work Break II
		- (301) Remove Invalid Parentheses 
	- Stack + Array
		- (78) Subsets
    - Grid + Tree + String 
        - (212) Word Search II

Template [TODO]
```
def backtrack(grid):
    return 

def condition(x):
    return True

def main():
    n = 10
    grid = [[0 for i in range(len(n))] for j in range(len(n))]
    backtrack(grid)
```

# Djikstra's Algorithm 

[TODO] - Explain the algorithm steps 

- Goal: Find shortest path to all remaining nodes
- Implementation: BFS + Priority Queue 

- Problems + Variants
	- (743) Network Delay Time 

Code Template [TODO]
```
```

# Topological Sort

- Goal: Find an ordering of nodes, where each node has no incoming edges from the later nodes in the ordering
- Applications: [TODO]
    
- DFS Version 
    - Less Intuitive but Concise Code
    - Similar to cycle detction & finding connected components 
    - Steps 
        - Keep track of visited table with values 0, 1, 2
            - Node Status
                - 0 => not visited
                - 1 => visited (inprogress)
                - 2 => all neighbors visisted (complete)
        - Iterate through the visited table 
        - If node not visited yet (e.g. visited[i] == 0), perform dfs
        - If visited node ever reached when searching, no topological sorts exist

- BFS Version 
    - More Intuitive but Lengthy Code
    - Steps 
        - Add all nodes with 0 indegree to queue
        - Perform BFS using this intialized queue
        - Remove edges from graph that have already been traversed
        - Add new nodes with 0 indegree
        - Repeat until queue empty

- Kahn’s Algo [TODO]
    - DFS (Tri Color)
    - Morris Search [No Recursion Stack]

- Variants + Problems
    - General Topo
        - (207) Course Schedule 
        - (269) Alien Dictionary
    - Cycle Detection Topo
        - () Number of Connected Components in an Undirected Graph 
        - (261) Graph Valid Tree
    - Kahn's Algorithm
        - [TODO]

Code Template
DFS Version
```
def search(course):
    nonlocal visited
    nonlocal invalid
    if visited[course] == 1:
        invalid = False
    if visited[course] == 0:
        visited[course] = 1
        neighbors = graph[course]
        for neighbor in neighbors: 
            search(neighbor)
        visited[course] = 2

graph = {}

for course in range(numCourses):
    graph[course] = []

for prereq in prerequisites: 
    graph[prereq[0]].append(prereq[1])

invalid = True

visited = [0 for i in range(numCourses)]

for course in range(numCourses):
    if visited[course] == 0:
        search(course)

return invalid 
```

# Recursive Tree Traversal

- Types of Trees
    - Binary Tree
    - Binary Search Tree
    - N-ary Tree (Trie)

- Build a search function that traverses a binary tree: 
    - Preorder - Visit Root First, before Left or Right Child 
    - Inorder - Visit Root after Left Child, but before Right Child
    - Postorder - Visit root after right Child  
- Keep track of a variable to update during traversal 
    - Global variable (e.g. `self.max = -math.inf`)
    - Function parameter and function return value (e.g. `f(depth + 1)` or `return depth`)
- Modify the nodes while going through the tree
    - Swap children (invert tree)

- Trie [TODO]
    - Iterate through a string, and add letters one a time while updating position within trie  
    - Mark the end of a word with a special character like `#`
    - Recursively search trie with substrings of the word to find a word

- Build Trees Recursively [TODO]
    - Set the return value of the traverse function to be a new tree

- Problems + Variants  
    - DFS Traversal 
        - General 
            - (105) Construct Binary Tree From Preorder and Inorder Traversal 
            - (235) Lowest Common Ancestor of a Binary Search Tree
            - (236) Lowest Common Ancestor of a Binary Tree
            - (100) Same Tree
            - (102) Binary Tree Level Order Traversal
        - Pre Order
            - (226) Invert Binary Tree
        - In Order
            - (98) Validate Binary Search Tree
        - Post Order
    - Trie
        - (208) Implement Trie (Prefix Tree)
        - (211) Design Add and Search Words Data Structure
        - (212) Word Search II
    - Build Trees
        - (297) Serialize and Deserialize Binary Tree
        - (105) Construct Binary Tree from Preorder and Inorder Traversal
- Memorization Tips
    - [TODO]
    
Code Templates 
```
def dfsBinaryTree(root):
    if not root: 
        return
    else: 
        print('preorder: parent -> leftchild -> rightchild')
        dfsBinaryTree(root.left)
        print('inorder: leftchild -> parent -> rightchild')
        dfsBinaryTree(root.right)
        print('postorder: leftchild -> rightchild -> parent')
```
```
if not root: 
    return 
else: 
    #preorder
    search(root.left)
    #inorder
    search(root.right)
    #postorder
```

# Iterative Tree Traversal 

- Used when recursion stack grows too long) 
- Replace the recursive call stack with a literal stack data structure
- Only used in follow-up question as an optimization to recursive traversals   

[TODO] - Explain Stack + Inorder  

- Problems + Variants [TODO]
    - Morris
	- Parent Traversal (Linked List)
	- Node Stack (e.g. Flatten BT)
- Memorization Tips
    - Memorize iteration using a stack template code

[TODO] - Write Code templates for iterative preorder, inorder, and post order
```
stack = []
def dfsIterative(root):
    stack.append(root)
    while stack:
        newNode = stack.pop()
        stack.append(newNode.left)
        stack.append(newNode.right)
```
```
[Q98 solution]
stack, prev = [], -math.inf
while stack or root:
    while root:
        stack.append(root)
        root = root.left
    root = stack.pop()
    # If next element in inorder traversal
    # is smaller than the previous one
    # that's not BST.
    if root.val <= prev:
        return False
    prev = root.val
    root = root.right
return True
```

# Binary Search

[TODO] - Explain mid point relationship with upper and lower bounds

- General Steps
    - Step 1: Find the left and right pointers
    - Step 2: Set the middle pointer
    - Step 3: Move left or right pointer to middle, based on specific conditions

- Common Issues + Errors with Execution 
	- Infinite loop
		- if mid set incorrectly gets rounded down incorrectly 
			- mid = ((r - l) // 2) + 1
			- mid = ((r - 2) //2) 
		- if left pointer increments incorrectly
		- if right pointer decrements incorrectly 
		- if left pointer == right pointer, but left pointer never < right pointer 

- Choosing mid: 
	- Think about if you are finding upper bound or lower bound
		- If l = mid + 1 (left moved) and r = mid 
            - then use mid = (l + r)//2
		- If r = mid - 1 (right moved) and l = mid
            - then use mid = (l + r + 1)//2

- Variants + Problems:
    - Search of Separate Search Space
        - (1891) Cutting Ribbons
    - Search based on Specific Condition
        - (658) Find K Closest Elements (x - arr[mid] > arr[mid + k] - x)
        - (162) Find Peak Element (arr[mid] > arr[mid + 1])
        - (153) Find Minimum in Rotated Sorted Array
    - Search based on Search Space
        - (528) Random Pick with Weight
    - Perform Two Searches
        - (33) Search in Rotated Subarary
- Memorization Tips
    - Memorize mid pointer selection steps
    - Memorize upper and lower bound defintions

Code Templates:
```	
sortedArray = []
def binarysearch(left, right):
    while left < right:
        mid = (left + right)//2
        if sortedArray[mid] < sortedArray[right]:
            left = mid + 1
        else:
            right = mid - 1
    return sortedArray[left]
```
```
def binarySearch():
	l = 0
	r = len(arr) - 1
	while l < r: 
		mid = (l + r) // 2
		curr = arr[mid]
		if mid == target: 
			return mid 
		elif mid < target: 
			l = mid 
		else:
			r = mid 
	return l 
```

- References:
- https://jonisalonen.com/2016/get-binary-search-right-the-first-time/ 
https://medium.com/swlh/binary-search-find-upper-and-lower-bound-3f07867d81fb 
	- Choosing next range’s L and R

# Two Pointer

When you need to traverse a list in a specified order from both sides, or one side 

Pending Theory Questions
Difference between while r < len(s) and for i in range(0, len(s))? 

- Variants + Problems 
    - Start and End of Array or String 
        - Trapping Rain Water
        - (125) Valid Palindrome
        - Valid Palindrome II
    - Sliding Window
        - (76) Minimum Window Substring
        - (1004) Max Consecutive Ones III
        - (904) Fruit into Basket
        - (424) Longest Repeating Character Replacement
            - 26 different sliding windows for each letter
    - Start of Two Different Arrays
        - (415) Add Strings [todo?]
- Memorization Tips

Code Templates
```
Def twoPointer(l, r):
	While l < r: 
		#Do something to move left and right pointer
		If stopping condition: 
			Return False
	Return True
```

```
def slidingWindow(array):
    value = 0 
    length = 0
    minOrMax = 0
    for right in range(len(array)):  
        value += array[right]
        length += 1
        while condition(value):
            value -= array[left]
            left += 1
            length -=1 
        minOrMax = min(max(length))
    return minOrMax
```

References
https://leetcode.com/problems/fruit-into-baskets/solutions/170740/java-c-python-sliding-window-for-k-elements/?orderBy=most_votes 


# Simulation 

- Matrix Properties
    - Apply general linear algebra properties 
        - 90 degree matrix rotation = transpose + flip accross vertical center 

- Simulation (Matrix)

- Variants + Problems 
    - General 
        - (54) Spiral Matrix
        - Toeplitz Matrix
        - Diagonal Traversal
    - Tranformation
        - (48) Rotate Image

- Memorization Tips
    - Memorize common matrix transformations trick
    - Memorize common simulation tricks such using two variables as a changing unit vector 

# One Pointer

When you need to traverse a list in a specified order

- Variants + Problems 
    - Math Logic
        - Angle Between Hands of Clock
    - Hashmap Trick (Array Problems)
        - Prefix Sum
        - Continuous Subarray Sum (Keep defaultdict(int))
        - (217) Contains Duplicate
    - Stacks 
    - Sets
    - Lists
        - (252) Meeting Rooms

Code Templates
```
def prefixSum(array):
    prefixSum = [0 for _ in range(len(array))]
    currSum = 0
    for index in range(len(array)):
        curr = array[index]
        currSum += curr
        prefixSum[index] = currSum
    return prefixSum
```

# Array 

- Ways to simplify an array problem: 
    - Sort the values, and exploit a pattern
    - Keep track of previously visited values, through a hashmap or set 
    - Explore a range of values using two pointers 

- Variant + Problems 
    - Hashmap Trick
        - (1) Two Sum
        - (15) 3Sum
    - Set Trick 
        - (217) Contains Duplicate
    - Two Pointer 
    - Sorting Values 
        - (252) Meeting Rooms 

- Memorization Tips
    - [TODO]


# Linked List

- Techniques
    - Slow & Fast Pointers
        - find cycle 
        - find middle element
        - find nth last element
    - Create dummy head list node (used when merging lists)
        - The next value from the dummy head will contain the accurate value

- Variants + Problems 
    - (206) Reverse Linked List
    - (141) Linked List Cycle
    - (21) Merge Two Sorted Lists
    - (143) Reorder List
    - (2) Add Two Numbers

- Memorization Tips
    - Memorize commmon linked list pointer tricks

Code Templates [TODO]

List Node Class
```
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
```
    

# Bit Manipulation

- XOR Properties
    - 1 xor 1 = 0 and 0 xor 0 = 0
- AND Properties
    - n & 1 => return the last bit of n
    - n = n & 0 => sets the last bit of n to 0 
- OR Properties
    - n | 1 => sets the last bit of n to 1 

- Variants + Problems
    - XOR
        - (371) Sum of Two Integers 
        - (268) Missing Number
    - Dynamic Programming
        - (338) Counting Bits 
    - AND 
        - (191) Number of 1 Bits
    - OR 
        - (190) Reverse Bits  
- Memorization Tips
    - Memorize common logical operation tricks 

- Code Templates 
```
def binaryAddition(a, b):
    sum = (a ^ b)
    carry = (a & b) << 1

    while b: 
        sum = (a ^ b)
        carry = (a & b) << 1
        a = sum
        b = carry

    return a 
```
```
def binarySubtraction(a, b):
    diff = (a ^ b)
    borrow = (~a & b) << 1

    while b: 
        diff = (a ^ b)
        borrow = (~a & b) << 1
        a = diff
        b = borrow

    return a
```
```
def binaryCounting():
    return 
```

# Stack

- Stack: Data structure with data processed first in first out (FIFO)
    - Think a "stack" of pancakes

- MonoStack = Stack + Staggered and Conditional Removing and Adding 
	- hence the elements within the stack are always increasing? 	
	Types: 
		- Increasing MS
		- Decreasing MS 

- Problems + Variants:
	- Regular Stack [TODO]
		- idk? 
	- Monotonic Stack
		- Next Greater Element I
		- Buildings with Ocean View

- Memorization Tips
    - Memorize steps for using stack to perform monostack functionality

Code Templates [TODO]

```
```

- References:
    - https://leetcode.com/problems/sum-of-subarray-minimums/discuss/178876/stack-solution-with-very-detailed-explanation-step-by-step
    - https://labuladong.gitbook.io/algo-en/ii.-data-structure/monotonicstack 


# Intervals

[TODO]

Defintion: An interval is defined as a pair of numbers => (start time, end time)

- Variants + Problems: 
    - One Pointer
        - (435) Non-overlapping Intervals
    - Stack 
        - (57) Insert Interval 
        - (56) Merge Intervals
    - Heap 
        - (253) Meeting Rooms II 

Code Templates [TODO]
- Template 1
```
Def Iterate():
For n in range(nums):
Do something with n
Modify a tracker
Return tracker 
```
- Template 2
```
Def Iterate()
While not stopping condition:
Modify nums
Modify tracker
Return tracker
```

- Implementation
    - https://leetcode.com/problems/minimum-number-of-arrows-to-burst-balloons/discuss/93735/a-concise-template-for-overlapping-interval-problem (Template)

# Dynamic Programming

- Definition: Solve a larger problem, but solving it's subproblems one at a time
- Applications: [TODO]

- Finding problems' recursive relationship:
    - Step 0: Find the base case 
    - Step 1: Find the state variable
    - Step 2: Find the recurrence relationship

- In other words:
    - Step 1: Identify optimimal substructure
    - Step 2: Find overlapping subproblems

- Advanced techniques
- Types of recurrence relations
	- Type 1: Using just integer counters
	- Type 2: Using lists of integers for counters
- Two Approaches
    - Top down
        - Recursive 
        - More intuitive
    - Bottom up 
        - Tabular
        - More concise
- Space Optimizaitons
    - Store 2d dp array values in less space
        - Map essential state variables to 1d array of specific variables

- Kadane's Algorithm
    - Uses optimal substructures of previous problems to reach new solution
        - Recurrence relation is a conditional that determines which maximum and minimum values to track

- 5 types of recurrent relationships (Reddit)
    - 0/1 knapsack 
        - classic knapsack 
        - (416) Partition Equal Subset Sum  
            - bounded knapsack
    - unbounded knapsack
        - coin change 
        - (337) Combination Sum IV
    - fibonacci
        - house robber 
    - longest common substring
        - longest common subsequence
    - Kadane's algo
        - (121) best time to buy sell stock 

- Variants + Problems: 
	- 2d DP
		- (62) Unique Paths
        - (221) Maximal Square
	- 1d DP 
        - (70) Climbing Stairs (Fibonacci)
        - (300) Longest Increasing Subsequence 
		- (322) Coin Change 
		- (198) House Robber
        - (91) Decode Ways 
        - (2361) Minimum Cost Using Train Line 
        - (213) House Robber II
        - (377) Combination Sum IV
    - Kadane's Algorithm
        - (53) Maximum (Sum) Subarray
        - (152) Maximum Product Subarray
        - (128) Longest Consequtive Sequence
        - (121) Best Time to Buy and Sell Stock 

- Memorization Tips
- Understand and memorize all recursive relationships for each commonly asked question

Code Template
```
def dynamicProgramming(array):
    dp = [0 for _ in range(len(array))]
    for index in range(len(array)):
        curr = array[index]
        if condition(curr):
            dp[index] = dp[index - 1]
    return dp[-1]
```

# String Logic

Defintion: [TODO]
Applications: [TODO]
[TODO]

- Variants + Problems
    - DFA 
        - Valid Number
        - String to Integer (atoi)

- Memorization Tips
    - Memorize all string modification functions 

# Sorting 

Defintion: [TODO]
Application: [TODO]
[TODO]

- Variants + Problems
	- Bucket Sort (Used when working with frequencies)
	- Cycle Sort (While loop until arr[i] != arr[arr[i]])
	- Counting Sort
	- Bubble Sort
	- Quick Sort
    - Quick Sort (Quick Select)
    - Merge Sort
        - (148) Sort List
- Memorization Tips
    - Understand and memorize all common sorting algos
    - Memorize common heap operation in python

References 
https://en.wikipedia.org/wiki/Bucket_sort 
https://en.wikipedia.org/wiki/Cycle_sort 

# Sorting w/ Heaps

[TODO]
- Variants + Problems
    - (347) Top K Frequent Elements

# Quick Select

- Implementation [TODO]
    - Select
    - Moves pivot until reaches k
    - Calls partition to find next pivot value
    - Partition

- Variants + Problems:
    - Lumoto’s Partition
        - K Closest Points to Origin
        - Top K Frequent Elements
    - Hoare’s Partition 
        - Random Pivot or Middle Pivot

- References:
    - Sorting Visualized: https://www.youtube.com/watch?v=MZaf_9IZCrc 
    - Lomuto Implementation: https://en.wikipedia.org/wiki/Quickselect#Algorithm 

# Union Find

[TODO]
- Properties
- Used to find minimum path

- Variants + Problems: 
	- Path Compression

# Math 

[TODO]

- Variants + Problems: 
    - Gauss's Formula
        - (268) Missing Number
- Memorization Tips
    - Memorize common discrete math formulas

# Design, Language, DS 

OOP/OOD

Abstract Classes and Methods
https://medium.com/techtofreedom/abstract-classes-in-python-f49cf4efdb3d 

# Python Tricks

- Mutability and Immutability 
    - Hashmaps can only store immutable types 
        - Strings are immutable types 

- Variable Swapping
    - Trade values saved in variables a and b 
    - Can also apply to values within python arrays
    - Steps 
        - `a, b = b, a`
        - ```
        tmp = a
        a = b
        b = tmp 
        ```

- Hashmap functions
    - `map.items()` -> returns list of key-value tuples 

- Use python counter to count occurences within strings 
    - `from collections import Counter`
    - `count = Counter(s)`

- String Formatting
    - `isalnum()`
    - `lower()`

- Sort List by Element via Lambda Statements
`sorted(intervals, key = lambda x: x[1])`

- Heap's are initialized to minHeaps in Python
    - Input value * -1 to create a maxHeap 

```
from collections import heapq
import heapq

heap = []
heapq.heappush(heap, x)
heapq.heapreplace(heap, y)
z = heapq.heappop(heap)
```

# Java Tricks
Tree Map
https://www.geeksforgeeks.org/treemap-in-java/ 

# C++ Tricks
[TODO]

# Prep, Learning, Strategy, Originization, Schedule, Etc. 
- What is a pattern? 
    - A repeating abstraction found in algorithm problems (e.g. Two Pointer, Backtracking)

- What is a data structure? 
    - Queue, Heap, Linked List, Tree, Graph, Disjoint Union Set, Monotonic Stack, Doubly Linked List, Combination of the above

- Goal  
	- Break up the problems into patterns + variants to increase recall  
	- Learn how to study well  
	- Document your learning so that it’s easy to pick up where you left off  
	- Find the best practices for interview prep by talking to people who succeeded  
	- Improve my study abilities over 5-6 months (part of life long journey)  

- Interview  
	- How to Quickly Run Through Test Case  
	- Write code in a way that you can easily read it  
	- Take time to organize code into ordered chunks before reading it  
	- Identify if solution is complex or simple before hand  
	- How to Pick Right Test Case  

- Mock Interview  
	- Ask if he can optimize it  
	- Ask runtime and space complexity  
	- Follow him solving it  
	- Ask if he can walk through an example  

- Steps  
	- Step 1: Identify the pattern (practice patterns)  
	- Step 2: Apply the pattern (you cannot proceed after this step)  
	- Step 3: Prove the solution in plain english, before attempting it (memorize proofs)  
	- Step 4: Implement the solution (memorize templates)  

- Advanced techniques  

- Analogies  
	- The difference between an okay musician and a good muscian is the fundamentals and theory  

- How to get unstuck  
	- Review previous solved DP problems  

- How to review  
	- If you can't explain the solutions in plain english to a non computer science student, you won't remember the concept yourself.  

- Daily Prep  
    - Hours per day  
	    - 8 hrs per day  

- Schedule  
	- Day A : Do 3 practice onsites  
	- Day B : Reflect on the previous day's practice offsites  

- Questions to self  
	- What do you want to improve on?  

- Meta Learning  
	- Stages of Learning  

- How to Practice LC Properly  
	- Set a 2 minute timer after each failed test case  
	- Go through at least one example before compiling  
	- Insert comments in code in the final pass  

- Learning Strategy  
	- pick a lc question, then create a txt file with the number  
	- start a timer for 20 minutes, and write down as much of an answer as you can 
	- separate scratch work from final code solution

- Learning Philosophy
    - Looking back at old suboptimal solutions, or poorly explained explanations will only slow down learning process, if the goal is to memorize the optimal solutions for each solution to pass OA's and interviews
    - Keep a clean and concise answer for each question, to memorize to improve pattern matching skills. 
    - Memorizing a incorrect or suboptimal solutions will only lead to poor recall 

- Not all problems are made the same  
	- For example both Word Break II and Remove Invalid Parentheses are both classic BackTracking problems
    - One causes a world of pain 
    - the other doesn’t (hint: Remove Invalid Parentheses == RIP)
    - Some problems can be solved with one pattern + variant, and solved optimally with a different pattern + variant
        - Subarray Sum Equals K
		- Solved with 2 Pointer/DP by checking all subarrays
		- Solved with 1 Pointer and Hashmap by considering mod arithmetic

- How to integrate with anki
	- Tag problems as solved and unsolved efficiently. 
	- Copy the question and attempted solution to a google sheet, then download and import them into anki
	- Match problems in anki with excel sheet, to quickly determine is a question is found or not found 

- Timed Practice
    - Approach 1: Use timer
    - Approach 2: Use stopwatch 
	    - lap when finishing each problem

- Types of Tests

- OA 
	- Code Signal
		- measures speed, cleanliness, correctness 
			- https://codesignal.com/resource/general-coding-assessment-framework/ 
	- Coder Pad
		- measures Correctness, Speed, Cleanliness and Brevity 
			- https://coderpad.io/test-cases/ 
	- Hacker Rank [Public Test Cases]
		- tests correctness, and engineering
	- Hacker Rank [Hidden Test Cases]
		- tests unit testing and logical thinking 
	- Codility [Public & Hidden Test Cases]
		- tests correctness, engineering, and logical thinking

- Phone Screen 
	- Coder Pad 
    - Collabedit - https://collabedit.com/ 

- Onsite 
	- Coder Pad 

- Never consume coffee ever again before a test

- Measurement of mastery
    - Scoring
        - Award 1 point if question can be solved in 15 minutes
        - Award 0 points if question can not be solved in 15 minutes
    - Reach 95% in Blind 75 ~ Solve more than 70 questions in less than 15 minutes each

- Failed OAs
    - Quant
       - Akuna Capital [2022]
       - DRW [2022]
       - Peak6 [2022]
       - Optiver [2022]
    - Tech 
        - Tesla [2022]
        - Samsara [2023]
        - Databricks [2021]

- Failed Interviews
    - Quant
        - Citadel [2021]
    - Tech 
        - Meta [2022]

## Mock Interview

- 1/27/23
    - my interview 
        - glossed over space complexity 
            - didn't detail it out when first explaining
            - it may be good or bad to use more variables
        - practice explaining the details of complexity
    - evan interview
        - null case 
        - killed time on floyd proof 