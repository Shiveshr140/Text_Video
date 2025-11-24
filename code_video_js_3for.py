#!/usr/bin/env python3
"""
Test Code to Video
==================

Testing the code_to_video function with sample Python code

IMPORTANT: Use Python 3.11 with Whisper installed!
Run: source venv311/bin/activate && python code_video_java.py
Or: ./run_code_video.sh
"""

from simple_app import code_to_video

if __name__ == "__main__":
    print("\n" + "💻"*30)
    print("CODE TO VIDEO TEST")
    print("💻"*30 + "\n")
    
    sample_code = """
   function areAnagrams(str1, str2) {
    if (str1.length !== str2.length) {
        return false; 
    }

    let count1 = {};
    let count2 = {};

    // Count frequency of each character in str1
    for (let i = 0; i < str1.length; i++) {
        let char = str1[i];
        count1[char] = (count1[char] || 0) + 1;
    }

    // Count frequency of each character in str2
    for (let i = 0; i < str2.length; i++) {
        let char = str2[i];
        count2[char] = (count2[char] || 0) + 1;
    }

    // Compare the two frequency objects
    for (let char in count1) {
        if (count1[char] !== count2[char]) {
            return false; 
        }
    }

    return true; 
}
console.log(areAnagrams("listen", "silent"));

"""
    
    print("This will create:")
    print("  ✅ Code displayed on screen")
    print("  ✅ Audio explaining the code")
    print("  ✅ Synchronized video")
    print("  ✅ Professional code explanation")
    print("\n" + "="*70 + "\n")
    
    result = code_to_video(
        code_content=sample_code,
        output_name="pyramid_code_explanation",
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

