"""
Manual Real-World Integration Test (Speech + NLP)
===================================================
Run this test from the terminal to verify the FULL pipeline using
your REAL microphone and the REAL NLP classifier.

Run::

    python tests/manual_test_speech_nlp.py
"""

import sys
from pathlib import Path

# Ensure BackEnd/ is on sys.path
_BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from app.services.orchestrator.question_pipeline import QuestionPipeline


def main() -> None:
    print("=" * 50)
    print("  Speech + NLP Manual Integration Test")
    print("=" * 50)
    print("Speak your question in English.")
    print()

    # Load the real pipeline
    pipeline = QuestionPipeline(
        language="en-US",
        timeout=5,
        phrase_time_limit=10,
    )

    try:
        # This function handles the microphone prompts and processing
        result = pipeline.process_voice_question()
        
        if result:
            print("\n" + "=" * 50)
            print("  Result Summary:")
            print("=" * 50)
            print(f"Recognized text:\n{result['question']}\n")
            print(f"Predicted topic:\n{result['topic']} ({result.get('topic_confidence', 0):.0%})")
            print("=" * 50)
        else:
            print("\n[ERROR] Process returned None. Check error messages above.")
            
    except KeyboardInterrupt:
        print("\n\nTest cancelled by user.")
    except Exception as exc:
        print(f"\n[ERROR] Unexpected error: {exc}")


if __name__ == "__main__":
    main()
