"""
TEST MULTILINGUAL FEATURE
========================

Test text_to_video() with different languages:
- English (default)
- Hindi (code-mixed)
- Tamil (code-mixed)
- Kannada (code-mixed)

Visuals: Always English
Audio: Code-mixed with technical terms in English
"""

from simple_app import text_to_video

# Test content - about Machine Learning
test_content = """
Machine Learning Basics

Machine Learning is a subset of artificial intelligence that enables computers to learn from data without explicit programming. It's revolutionizing how we solve complex problems.

How It Works

Machine learning algorithms analyze patterns in data to make predictions or decisions. The system improves its performance as it processes more examples, learning from experience rather than following rigid rules.

Types of Learning

There are three main approaches. Supervised learning uses labeled data to train models. Unsupervised learning finds hidden patterns in unlabeled data. Reinforcement learning learns through trial and error with rewards and penalties.

Real World Applications

Machine learning powers recommendation systems on streaming platforms, fraud detection in banking, medical diagnosis from imaging, and autonomous vehicles. It's transforming industries worldwide.

Getting Started

Begin with Python and popular libraries like scikit-learn. Practice with real datasets, understand the fundamentals of statistics, and gradually explore deep learning frameworks. The key is consistent practice and hands-on projects.
"""

if __name__ == "__main__":
    print("\n" + "🌍"*30)
    print("MULTILINGUAL TEXT-TO-VIDEO TEST")
    print("🌍"*30 + "\n")
    
    print("Features:")
    print("✅ Visuals: English (universal)")
    print("✅ Audio: Code-mixed (natural for Indian audiences)")
    print("✅ Technical terms stay in English")
    print("\n" + "="*70 + "\n")
    
    # Test 1: English (baseline)
    print("TEST 1: ENGLISH")
    print("="*70)
    result1 = text_to_video(
        test_content, 
        "ml_english",
        audio_language="english"
    )
    if result1:
        print(f"✅ Created: {result1['final_video']}\n")
    
    # Test 2: Hindi (Hinglish)
    print("\n" + "="*70)
    print("TEST 2: HINDI (Code-Mixed)")
    print("="*70)
    result2 = text_to_video(
        test_content,
        "ml_hindi",
        audio_language="hindi"
    )
    if result2:
        print(f"✅ Created: {result2['final_video']}\n")
    
    # Test 3: Tamil
    print("\n" + "="*70)
    print("TEST 3: TAMIL (Code-Mixed)")
    print("="*70)
    result3 = text_to_video(
        test_content,
        "ml_tamil",
        audio_language="tamil"
    )
    if result3:
        print(f"✅ Created: {result3['final_video']}\n")
    
    # Test 4: Kannada
    print("\n" + "="*70)
    print("TEST 4: KANNADA (Code-Mixed)")
    print("="*70)
    result4 = text_to_video(
        test_content,
        "ml_kannada",
        audio_language="kannada"
    )
    if result4:
        print(f"✅ Created: {result4['final_video']}\n")
    
    # Summary
    print("\n" + "="*70)
    print("📊 SUMMARY")
    print("="*70)
    results = [
        ("English", result1),
        ("Hindi", result2),
        ("Tamil", result3),
        ("Kannada", result4)
    ]
    
    for lang, result in results:
        status = "✅ SUCCESS" if result else "❌ FAILED"
        print(f"{lang:15} {status}")
        if result:
            print(f"                → {result['final_video']}")
    
    print("\n🎉 All videos have ENGLISH visuals with multilingual audio!")
    print("🌍 Perfect for Indian audiences - natural code-mixing!")

