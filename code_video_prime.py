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
   function isPrime(num) {
    if (num <= 1) 
        return false;
    for (let i = 2; i < num; i++) 
    {
        if (num % i === 0) 
            return false;
    }
    return true;
}

console.log(isPrime(7)); 


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

