#!/usr/bin/env python3
"""
Test with SHORT code (no scrolling) to verify highlight positioning
"""

import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from simple_app import code_to_video

# SHORT Java code - should fit on screen without scrolling
code = """
public class Test {
    public static void main(String[] args) {
        int x = 5;
        for (int i = 0; i < x; i++) {
            System.out.println(i);
        }
    }
}
"""

print("\n" + "="*60)
print("SHORT CODE TEST (No Scrolling)")
print("="*60)
print("\nThis will test:")
print("  ✅ Highlight positioning without scrolling")
print("  ✅ Comment removal")
print("  ✅ Block detection on short code")
print("\n" + "="*60 + "\n")

result = code_to_video(
    code_content=code,
    output_name="short_test",
    audio_language="english"
)

if result:
    print(f"\n✅ Video created: {result['final_video']}")
else:
    print("\n❌ Video creation failed")
