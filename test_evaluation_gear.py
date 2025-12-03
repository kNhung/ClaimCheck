"""
Test script for GEAR-based evaluation module improvements.
"""
from factchecker.modules import evaluation

# Test claim and evidence
claim = "Nhà làm việc trên đỉnh Hòn Bà của bác sĩ Alexandre Yersin được xây dựng khoảng năm 1914 đã được Bộ Văn hóa, Thể thao và Du lịch quyết định bổ sung là di tích quốc gia."

evidence_pieces = [
    "Nhà làm việc trên đỉnh Hòn Bà của bác sĩ Alexandre Yersin được xây dựng khoảng năm 1914. Theo bài viết của chính bác sĩ Alexandre Yersin đăng trên Tập san Hội Nghiên cứu Đông Dương, cách đây khoảng 109 năm (vào năm 1914), ông đã quan sát và tìm cách lên một đỉnh núi Hòn Bà ở độ cao khoảng 1.500m.",
    "Ngôi nhà gỗ trên đỉnh Hòn Bà của bác sĩ Alexandre Yersin được xếp hạng di tích quốc gia",
    "Nơi làm việc của bác sĩ Alexandre Yersin, người thành lập viện Pasteur ở Hà Nội, Nha Trang, Đà Lạt, được xây khoảng 100 năm trước, trên núi cao hơn 1.500 m."
]

print("=== Testing GEAR-based Evaluation ===")
print(f"\nClaim: {claim}\n")

# Test 1: Build graph
print("1. Building fully-connected evidence graph...")
G = evaluation.build_evidence_graph(claim, evidence_pieces)
print(f"   Graph has {len(G.nodes())} nodes and {len(G.edges())} edges")
print(f"   Expected: {1 + len(evidence_pieces)} nodes, {len(evidence_pieces) + (len(evidence_pieces) * (len(evidence_pieces) - 1)) // 2} edges (fully-connected)")

# Test 2: Aggregate evidence
print("\n2. Aggregating evidence with GEAR method...")
scores = evaluation.aggregate_evidence_with_graph(G, claim, evidence_pieces)
print(f"   Support score: {scores['support_score']:.3f}")
print(f"   Refute score: {scores['refute_score']:.3f}")
print(f"   Neutral score: {scores['neutral_score']:.3f}")
print(f"   Mean alignment: {scores['mean_alignment']:.3f}")
print(f"   Number of evidence: {scores['num_evidence']}")

# Test 3: Classify verdict
print("\n3. Classifying verdict...")
verdict, justification = evaluation.classify_verdict(scores)
print(f"   Verdict: {verdict}")
print(f"   Justification: {justification}")

print("\n=== Test completed ===")




