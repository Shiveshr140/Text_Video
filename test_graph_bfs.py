"""
Test Graph BFS Animation
Shows what AI generates for algorithm visualization
"""

from simple_app import animation_to_video

if __name__ == "__main__":
    print("\n" + "🌟"*30)
    print("Testing: Graph BFS Traversal Animation")
    print("🌟"*30 + "\n")
    
    result = animation_to_video(
        prompt="Graph BFS traversal",
        output_name="graph_bfs"
    )
    
    if result:
        print(f"\n🎉 SUCCESS! Video created: {result['final_video']}")
    else:
        print("\n❌ Failed to create video")

