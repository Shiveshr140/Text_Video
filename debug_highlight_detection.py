from simple_app import parse_code_to_blocks

sample_code = """
function findLargest(arr) {
    let largest = arr[0]; 
    for (let i = 1; i < arr.length; i++) {
        if (arr[i] > largest) {
            largest = arr[i]; 
        }
    }
    return largest;
}

console.log(findLargest([99, 5, 3, 100, 1]));
"""

print("="*60)
print("PARSING CODE")
print("="*60)
code_blocks = parse_code_to_blocks(sample_code, language="javascript")

print(f"\n{'='*60}")
print(f"DETECTED BLOCKS: {len(code_blocks)}")
print("="*60)
for i, block in enumerate(code_blocks, 1):
    print(f"\nBlock {i}:")
    print(f"  Type: {block['type']}")
    print(f"  Lines: {block['start_line']}-{block['end_line']}")
    print(f"  Code preview: {block['code'][:80]}...")

print(f"\n{'='*60}")
print("HIGHLIGHTING LOGIC")
print("="*60)

# Step 1: Keep ALL blocks for narration
all_blocks_for_narration = code_blocks.copy()
print(f"\n📝 Blocks for NARRATION: {len(all_blocks_for_narration)}")
for i, block in enumerate(all_blocks_for_narration, 1):
    print(f"   {i}. {block['type']} (lines {block['start_line']}-{block['end_line']})")

# Step 2: Filter blocks for HIGHLIGHTING
blocks_for_highlights = []

for block in code_blocks:
    if block['type'] == 'class':
        print(f"\n❌ Skipping {block['type']} (lines {block['start_line']}-{block['end_line']}): Class blocks are too large")
        continue
    
    # Check if this block contains other blocks
    contains_other_blocks = False
    contained_blocks = []
    for other_block in code_blocks:
        if other_block != block:
            if (block['start_line'] < other_block['start_line'] and 
                block['end_line'] > other_block['end_line']):
                contains_other_blocks = True
                contained_blocks.append(other_block)
                print(f"\n🔍 {block['type']} (lines {block['start_line']}-{block['end_line']}) CONTAINS {other_block['type']} (lines {other_block['start_line']}-{other_block['end_line']})")
    
    if contains_other_blocks:
        # Special rule: If it's a loop and contains only conditionals/statements (no inner loops), highlight it
        if 'loop' in block['type']:
            has_inner_loop = any('loop' in b['type'] for b in contained_blocks)
            if not has_inner_loop:
                blocks_for_highlights.append(block)
                print(f"   ✅ {block['type']} (lines {block['start_line']}-{block['end_line']}): HIGHLIGHT ✅ (loop preference - contains non-loop blocks)")
                continue
        
        print(f"   ❌ {block['type']} (lines {block['start_line']}-{block['end_line']}): HIGHLIGHT ❌ (contains {len(contained_blocks)} other blocks)")
    else:
        blocks_for_highlights.append(block)
        print(f"\n✅ {block['type']} (lines {block['start_line']}-{block['end_line']}): HIGHLIGHT ✅ (doesn't contain other blocks)")

print(f"\n{'='*60}")
print("FINAL RESULT")
print("="*60)
print(f"📝 Blocks for NARRATION: {len(all_blocks_for_narration)}")
print(f"🎯 Blocks for HIGHLIGHTING: {len(blocks_for_highlights)}")
print(f"\nHighlighted blocks:")
for i, block in enumerate(blocks_for_highlights, 1):
    print(f"   {i}. {block['type']} (lines {block['start_line']}-{block['end_line']})")
