"""
Quick test: Hindi code-mixed audio
"""

from simple_app import text_to_video

test_content = """
Neural Networks

Neural networks are computing systems inspired by biological brains. They consist of interconnected nodes called neurons organized in layers.

How They Learn

The network learns through backpropagation. It makes predictions, calculates errors, and adjusts weights to improve accuracy over time.

Applications

Neural networks power facial recognition, language translation, and recommendation systems. They're transforming technology across industries.
"""

if __name__ == "__main__":
    print("\n🇮🇳 Testing Hindi Code-Mixed Audio\n")
    print("Visuals: English")
    print("Audio: Hinglish (natural code-mixing)\n")
    
    result = text_to_video(
        test_content,
        "neural_hindi_test",
        audio_language="hindi"
    )
    
    if result:
        print(f"\n✅ SUCCESS! Video: {result['final_video']}")
        print("\n📝 What to expect:")
        print("  - Slides show English text")
        print("  - Audio says things like:")
        print("    'Neural networks ek computing system hai...'")
        print("    'Backpropagation se network seekhta hai...'")
    else:
        print("\n❌ Failed")

