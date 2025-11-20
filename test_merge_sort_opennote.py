"""
Test Merge Sort Animation - OpenNote.com Quality
=================================================

Creating a comprehensive educational video with:
- Text explanations on the left
- Visual animations on the right
- Step-by-step progression
"""

from simple_app import animation_to_video

if __name__ == "__main__":
    print("\n" + "🎓"*30)
    print("OPENNOTE.COM STYLE: MERGE SORT")
    print("Text Explanations + Visual Animations")
    print("🎓"*30 + "\n")
    
    prompt = "explain merge sort algorithm"
    
    print("This will create:")
    print("  ✅ Comprehensive Merge Sort explanation")
    print("  ✅ Text on left, animation on right")
    print("  ✅ Step-by-step progression")
    print("  ✅ Professional narration")
    print("  ✅ OpenNote.com quality output")
    print("\n" + "="*70 + "\n")
    
    result = animation_to_video(
        prompt=prompt,
        output_name="merge_sort_opennote"
    )
    
    if result:
        print(f"\n{'🎉'*30}")
        print(f"✅ VIDEO READY: {result['final_video']}")
        print(f"{'🎉'*30}\n")
        print("\n🎯 This video has:")
        print("  1. Text explanations synchronized with animations")
        print("  2. Clear step-by-step progression")
        print("  3. Professional visual layout")
        print("  4. Educational narration")
        print("  5. OpenNote.com quality!")
        print("\n💡 Show this to your team - it's production quality!")
    else:
        print("\n❌ Failed to create video")

