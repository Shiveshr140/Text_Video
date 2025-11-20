"""
Test cloud computing with GPT-4o version (old simple_app)
"""

# Import from simple_app_gpt-4o instead of simple_app
import sys
import importlib.util

# Load simple_app_gpt-4o.py as a module
spec = importlib.util.spec_from_file_location("simple_app_gpt4o", "/Users/apple/Desktop/Manim-Copy/simple_app_gpt-4o.py")
simple_app_gpt4o = importlib.util.module_from_spec(spec)
spec.loader.exec_module(simple_app_gpt4o)

# Use text_to_video from GPT-4o version
text_to_video = simple_app_gpt4o.text_to_video

# Long cloud computing content
long_content = """
Cloud Computing Revolution

Cloud computing has fundamentally transformed how organizations manage their IT infrastructure and deliver services. Instead of maintaining expensive on-premises data centers, businesses can now access computing resources on-demand through the internet, paying only for what they use.

Understanding Cloud Services

There are three primary service models in cloud computing. Infrastructure as a Service provides virtualized computing resources including servers, storage, and networking. Platform as a Service offers a complete development and deployment environment. Software as a Service delivers fully functional applications over the internet, eliminating the need for local installation and maintenance.

Deployment Models Explained

Organizations can choose from different deployment strategies based on their needs. Public clouds are owned by third-party providers and shared among multiple customers, offering maximum scalability and cost efficiency. Private clouds are dedicated to a single organization, providing enhanced security and control. Hybrid clouds combine both approaches, allowing data and applications to move between private and public environments.

Key Benefits and Advantages

The advantages of cloud computing are substantial and multifaceted. Scalability allows resources to expand or contract based on demand, ensuring optimal performance during peak periods. Cost efficiency eliminates large upfront capital investments in hardware and reduces ongoing maintenance expenses. Reliability comes from redundant systems distributed across multiple data centers, ensuring high availability and disaster recovery capabilities.

Security and Compliance

While cloud providers implement robust security measures, organizations must understand the shared responsibility model. Providers secure the underlying infrastructure, but customers remain responsible for protecting their data, applications, and access controls. Implementing encryption, multi-factor authentication, and regular security audits are essential practices for maintaining a secure cloud environment.

Future Trends and Innovation

The cloud computing landscape continues to evolve rapidly. Edge computing brings processing power closer to data sources, reducing latency for real-time applications. Serverless computing abstracts infrastructure management entirely, allowing developers to focus purely on code. Artificial intelligence and machine learning capabilities are becoming increasingly integrated into cloud platforms, democratizing access to advanced technologies.

Getting Started with Cloud

Organizations should begin their cloud journey strategically. Start by migrating non-critical workloads to gain experience and confidence. Invest in training your team on cloud technologies and best practices. Consider obtaining relevant certifications to validate expertise. Gradually expand your cloud footprint as you develop the necessary skills and processes for effective cloud management.
"""

if __name__ == "__main__":
    print("\n" + "🔄"*30)
    print("TESTING GPT-4o VERSION WITH CLOUD CONTENT")
    print("🔄"*30 + "\n")
    
    print("Model: GPT-4o (not GPT-5)")
    print("Voice: alloy")
    print("Content: Cloud Computing (long)")
    print("\n" + "="*70 + "\n")
    
    # Test with Hindi
    result = text_to_video(
        long_content,
        "cloud_gpt4o_hindi",
        audio_language="hindi"
    )
    
    if result:
        print(f"\n✅ Video created: {result['final_video']}")
        print("\nCompare this with GPT-5 version!")
    else:
        print("\n❌ Failed")

