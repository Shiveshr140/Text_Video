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

// Java Program to Print the Pyramid pattern

// Main class
public class GFG {

    // Main driver method
    public static void main(String[] args)
    {
        int num = 5;
        int x = 0;

        // Outer loop for rows
        for (int i = 1; i <= num; i++) {
            x = i - 1;

            // inner loop for "i"th row printing
            for (int j = i; j <= num - 1; j++) {

                // First Number Space
                System.out.print(" ");

                // Space between Numbers
                System.out.print("  ");
            }

            // Pyramid printing
            for (int j = 0; j <= x; j++)
                System.out.print((i + j) < 10
                                     ? (i + j) + "  "
                                     : (i + j) + " ");

            for (int j = 1; j <= x; j++)
                System.out.print((i + x - j) < 10
                                     ? (i + x - j) + "  "
                                     : (i + x - j) + " ");

            // By now we reach end for one row, so
            // new line to switch to next
            System.out.println();
        }
    }
}

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

