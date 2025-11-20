"""
Test Neural Network content with simple_app.py
NO CODE CHANGES - just different input!
"""

from simple_app import text_to_video

# Same neural network content we tested before
neural_network_text = """
Neural Networks: The Foundation of Modern AI

Introduction to Neural Networks

Neural networks are computing systems inspired by the biological neural networks in animal brains. 
They consist of interconnected nodes, called neurons, organized in layers. These systems learn to 
perform tasks by analyzing examples, without being explicitly programmed with task-specific rules.

Architecture Overview

A typical neural network has three types of layers: an input layer that receives data, hidden 
layers that process information, and an output layer that produces results. Each connection 
between neurons has a weight that adjusts during training, allowing the network to learn patterns 
and relationships in the data.

How Learning Works

Neural networks learn through a process called backpropagation. The network makes predictions, 
compares them to actual outcomes, calculates the error, and adjusts weights to reduce that error. 
This process repeats thousands of times until the network becomes accurate at its task.

Types of Neural Networks

Different architectures serve different purposes. Convolutional Neural Networks excel at image 
processing, Recurrent Neural Networks handle sequential data like text, and Transformer networks 
power modern language models. Each architecture is optimized for specific types of problems.

Real World Applications

Neural networks power facial recognition in smartphones, language translation services, and 
voice assistants. They enable self-driving cars to recognize objects, help doctors diagnose 
diseases from medical images, and allow streaming services to recommend content you might enjoy.

The Future of Neural Networks

Research continues to push boundaries, creating networks that require less data, consume less 
energy, and solve increasingly complex problems. From drug discovery to climate modeling, neural 
networks are becoming essential tools for tackling humanity's biggest challenges.

Ethical Considerations

As neural networks become more powerful, questions about bias, privacy, and accountability grow 
more important. Ensuring these systems are fair, transparent, and used responsibly is crucial 
for building trust and maximizing their benefit to society.

Conclusion

Neural networks represent a paradigm shift in computing, moving from rule-based systems to 
learning-based ones. Understanding their capabilities and limitations helps us harness their 
power while being mindful of their impact on society.
"""

if __name__ == "__main__":
    print("🧠 Testing Neural Network content...")
    print("📝 Using text_to_video() function from simple_app.py")
    print("🔑 NO CODE CHANGES - just different input!\n")
    
    result = text_to_video(neural_network_text, "neural_network_demo")
    
    if result:
        print(f"\n✅ SUCCESS! Video created: {result['final_video']}")
    else:
        print("\n❌ Failed to create video")

