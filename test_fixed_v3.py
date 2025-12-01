#!/usr/bin/env python3
"""
Test FIXED Animated Video V3
"""

from simple_app import query_to_animated_video_v3

if __name__ == "__main__":
    print("\n" + "✨"*30)
    print("TESTING FIXED VERSION V3")
    print("✨"*30 + "\n")
    
    query = "Explain Bayes theorem"
    
    print("Query:", query)
    print("\nFixes:")
    print("  ✅ Text slides with proper line wrapping (vertical)")
    print("  ✅ Animations centered properly")
    print("  ✅ No overlapping text")
    print("  ✅ Proper spacing\n")
    print("="*70 + "\n")
    
    result = query_to_animated_video_v3(
        query,
        "google_fixed_v3",
        audio_language="english"
    )
    
    if result:
        print(f"\n{'🎉'*30}")
        print(f"✅ VIDEO READY: {result['final_video']}")
        print(f"{'🎉'*30}\n")
    else:
        print("\n❌ Failed")
