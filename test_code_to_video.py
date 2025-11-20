"""
Test Code to Video
==================

Testing the code_to_video function with sample Python code
"""

from simple_app import code_to_video

if __name__ == "__main__":
    print("\n" + "💻"*30)
    print("CODE TO VIDEO TEST")
    print("💻"*30 + "\n")
    
    sample_code = """
 def bubble_sort(arr):
    n = len(arr)
    
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    
    return arr

"""
    
    print("This will create:")
    print("  ✅ Code displayed on screen")
    print("  ✅ Audio explaining the code")
    print("  ✅ Synchronized video")
    print("  ✅ Professional code explanation")
    print("\n" + "="*70 + "\n")
    
    result = code_to_video(
        code_content=sample_code,
        output_name="bubble_sort_code_explanation",
        audio_language="english"
    )
    
    if result:
        print(f"\n{'🎉'*30}")
        print(f"✅ VIDEO READY: {result['final_video']}")
        print(f"{'🎉'*30}\n")
        print("\n🎯 This video has:")
        print("  1. Code displayed clearly on screen")
        print("  2. Audio explaining each part")
        print("  3. Synchronized code and narration")
        print("  4. Professional code explanation!")
    else:
        print("\n❌ Failed to create video")

