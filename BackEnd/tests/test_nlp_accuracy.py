"""
NLP Accuracy Test
===================
Tests the upgraded Hybrid Semantic + Keyword routing pipeline with
real academic questions to ensure target accuracy.

Run:
    python tests/test_nlp_accuracy.py
"""

import sys
from pathlib import Path

# Ensure BackEnd/ is on sys.path
_BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from app.services.nlp.Question_Classification import predict_topic_with_confidence

TEST_CASES = [
    ("What is tcp handshake", "Computer Networks"),
    ("What is semaphore", "Operating System"),
    ("What is linear transformation", "Mathematics"),
    ("What is finite automata", "Theory of Computation"),
    ("What is linked list", "Programming and Data Structure"),
    ("What is truth table", "Digital Logic"),
    ("Explain the difference between a mutex and a semaphore", "Operating System"),
    ("How does a pipelined processor work", "Computer Organization and Architecture"),
    ("What's the derivative of x squared", "Mathematics"),
    ("Can you explain OSPF routing?", "Computer Networks"),
]

def run_tests():
    print("=" * 60)
    print("  NLP Accuracy Upgrade Evaluation")
    print("=" * 60)
    
    passed = 0
    total = len(TEST_CASES)
    
    for i, (question, expected_topic) in enumerate(TEST_CASES, 1):
        print(f"\n[Test {i}/{total}]")
        print(f"Question: \"{question}\"")
        print(f"Expected: {expected_topic}")
        
        # This will internally print the Top-K logic and confidences
        predicted_topic, confidence = predict_topic_with_confidence(question)
        
        if predicted_topic == expected_topic:
            print(f"Result: ✅ PASS (Confidence: {confidence:.2f})")
            passed += 1
        else:
            print(f"Result: ❌ FAIL (Predicted: {predicted_topic}, Confidence: {confidence:.2f})")
            
    print("\n" + "=" * 60)
    print(f"  Summary: {passed}/{total} passed ({passed/total:.0%})")
    print("=" * 60)
    
if __name__ == "__main__":
    run_tests()
