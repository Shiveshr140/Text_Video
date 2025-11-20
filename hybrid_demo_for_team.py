"""
HYBRID DEMO FOR YOUR TEAM
==========================

Combines educational slides (Hindi) + Animated visualizations
Perfect impressive demo!
"""

from simple_app import text_to_video

# Best demo content: Sorting Algorithm with detailed explanation in Hindi

sorting_content = """
Bubble Sort Algorithm

Bubble Sort is one of the simplest sorting algorithms used in computer science. It works by repeatedly comparing adjacent elements in an array and swapping them if they are in the wrong order. This process continues until the entire array is sorted.

How Bubble Sort Works

The algorithm makes multiple passes through the array. In each pass, it compares consecutive pairs of elements. When a larger element appears before a smaller one, they are swapped. After each complete pass, the largest unsorted element bubbles up to its correct position at the end of the array.

Step by Step Process

First Pass: Compare first and second elements. If first is greater, swap them. Then compare second and third. Continue until the end. The largest element is now at the last position. Second Pass: Repeat the process for remaining unsorted elements. The second largest element reaches its position. Continue: Keep making passes until no swaps are needed, meaning the array is sorted.

Time Complexity Analysis

Bubble Sort has a time complexity of O(n²) in both worst and average cases. This means for an array of n elements, it may need to perform up to n² comparisons and swaps. The best case occurs when the array is already sorted, giving O(n) complexity with an optimized version that detects if no swaps occurred.

Space Complexity

The space complexity is O(1) because Bubble Sort is an in-place sorting algorithm. It does not require any additional storage space proportional to the input size. Only a few temporary variables are needed for swapping elements, making it memory efficient.

Advantages and Disadvantages

Advantages: Simple to understand and implement. Works well for small datasets. Can detect if the array is already sorted. Disadvantages: Very inefficient for large datasets. Much slower than advanced algorithms like Quick Sort or Merge Sort. Not suitable for production systems with large data.

Real World Applications

While Bubble Sort is rarely used in real-world production systems due to its inefficiency, it serves excellent educational purposes. It helps beginners understand sorting concepts, algorithm analysis, and the importance of choosing efficient algorithms. It is also used in situations where simplicity matters more than performance.

Code Implementation

The algorithm can be implemented in just a few lines of code. Use nested loops: outer loop for passes, inner loop for comparisons. Include a flag to optimize by detecting early completion. Add swap logic using a temporary variable. The simple structure makes it perfect for teaching programming fundamentals.
"""

if __name__ == "__main__":
    print("\n" + "🌟"*30)
    print("HYBRID DEMO: BUBBLE SORT IN HINDI")
    print("Educational Slides + Concept Explanation")
    print("🌟"*30 + "\n")
    
    print("This demo will create:")
    print("  ✅ Detailed Bubble Sort explanation")
    print("  ✅ Hindi narration (code-mixed)")
    print("  ✅ Professional ElevenLabs voice")
    print("  ✅ Multiple slides with content")
    print("  ✅ Perfect synchronization")
    print("  ✅ Easy to understand")
    print("\n" + "="*70 + "\n")
    
    # Generate the demo
    result = text_to_video(
        sorting_content,
        "bubble_sort_hindi_demo",
        audio_language="hindi"
    )
    
    if result:
        print(f"\n{'🎉'*30}")
        print(f"✅ DEMO READY: {result['final_video']}")
        print(f"{'🎉'*30}\n")
        print("\n🎯 Show your team:")
        print("  1. Complete Bubble Sort algorithm explanation")
        print("  2. Natural Hindi narration (technical terms in English)")
        print("  3. Professional voice quality (ElevenLabs)")
        print("  4. Clear, organized slides")
        print("  5. Perfect for educational demos!")
        print("\n💡 This can be done for ANY topic:")
        print("   - Machine Learning algorithms")
        print("   - Data Structures")
        print("   - System Design concepts")
        print("   - Cloud Computing")
        print("   - Any technical topic!")
    else:
        print("\n❌ Demo failed!")

