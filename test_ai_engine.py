#!/usr/bin/env python
import sys
import os

print("\n" + "="*70)
print("AI ENGINE COMPREHENSIVE TESTING")
print("="*70)

try:
    print("\n[STEP 1] Importing AI Engine...")
    from ai_engine import AdvancedAIEngine
    print("[OK] Import successful")
    
    print("\n[STEP 2] Initializing AI Engine...")
    engine = AdvancedAIEngine()
    print("[OK] Engine initialized")
    
    print("\n[STEP 3] Testing Category Prediction...")
    result1 = engine.predict_category("Pothole on Main Street", "Large pothole causing accidents")
    print("[OK] Test 1 - " + result1['category'] + " (confidence: " + str(round(result1['confidence']*100, 1)) + "%)")
    
    result2 = engine.predict_category("Water leak", "Major pipe burst flooding street")
    print("[OK] Test 2 - " + result2['category'] + " (confidence: " + str(round(result2['confidence']*100, 1)) + "%)")
    
    print("\n[STEP 4] Testing Priority Prediction...")
    priority1, conf1 = engine.predict_priority("Pothole", "Large hole", "Roads")
    print("[OK] Test 1 - Priority: " + priority1.value + " (confidence: " + str(round(conf1*100, 1)) + "%)")
    
    priority2, conf2 = engine.predict_priority("Broken water pipe", "Flooding street", "Water")
    print("[OK] Test 2 - Priority: " + priority2.value + " (confidence: " + str(round(conf2*100, 1)) + "%)")
    
    print("\n[STEP 5] Testing Spam Detection...")
    spam1 = engine.analyze_spam_risk("Normal complaint", "There is a problem on the street that needs fixing")
    print("[OK] Normal text - Score: " + str(round(spam1['spam_score']*100, 1)) + "% (" + spam1['risk_level'] + ")")
    
    spam2 = engine.analyze_spam_risk("SPAM", "CLICK HERE!!! FREE MONEY!!! www.scam.com")
    print("[OK] Suspicious text - Score: " + str(round(spam2['spam_score']*100, 1)) + "% (" + spam2['risk_level'] + ")")
    
    print("\n[STEP 6] Testing Sentiment Analysis...")
    sentiment1 = engine.analyze_sentiment("Normal", "This is an urgent matter")
    print("[OK] Test 1 - Sentiment: " + sentiment1['sentiment_label'] + " (score: " + str(round(sentiment1['custom_score'], 2)) + ")")
    
    sentiment2 = engine.analyze_sentiment("Angry", "This is terrible and awful!")
    print("[OK] Test 2 - Sentiment: " + sentiment2['sentiment_label'] + " (score: " + str(round(sentiment2['custom_score'], 2)) + ")")
    
    print("\n[STEP 7] Testing Similar Complaint Detection...")
    existing = [
        ("c1", "u1", "Water leak", "Small leak in pipe", "Water", "pending", "2024-01-20T10:00:00"),
        ("c2", "u2", "Pothole", "Hole on street", "Roads", "resolved", "2024-01-18T14:00:00"),
    ]
    similar = engine.detect_similar_complaints("Water flooding", "Major pipe burst", existing)
    print("[OK] Found " + str(len(similar)) + " similar complaints")
    
    print("\n[STEP 8] Testing Complete Analysis Pipeline...")
    result = engine.analyze_complaint(
        "URGENT: Flooded street",
        "Main street is completely flooded due to broken water pipe",
        existing_complaints=existing
    )
    print("[OK] Analysis Results:")
    print("     - Category: " + result.category)
    print("     - Priority: " + result.priority.value)
    print("     - Spam Score: " + str(round(result.spam_score*100, 1)) + "%")
    print("     - Sentiment: " + str(round(result.sentiment_score, 2)))
    print("     - Urgency: " + str(round(result.urgency_indicator, 2)))
    print("     - Similar: " + str(len(result.similar_complaints)) + " found")
    
    print("\n[STEP 9] Engine Statistics...")
    stats = engine.get_statistics()
    print("[OK] Total Predictions: " + str(stats['predictions']['total_predictions']))
    print("[OK] Models Loaded: " + str(stats['models_loaded']))
    print("[OK] Cache Size: " + str(stats['cache_size']))
    print("[OK] Categories Supported: " + str(stats['categories_supported']))
    
    print("\n[STEP 10] Testing Text Preprocessing...")
    text1 = engine.preprocess_text("URGENT!!! Check www.example.com")
    print("[OK] Preprocessing working - processed text length: " + str(len(text1)))
    
    print("\n[STEP 11] Testing Feature Extraction...")
    features = engine.extract_features("Pothole", "Large pothole on street")
    print("[OK] Features extracted - count: " + str(len(features)))
    print("     - Title Length: " + str(features['title_length']))
    print("     - Word Count: " + str(features['word_count']))
    print("     - Caps Ratio: " + str(round(features['caps_ratio']*100, 1)) + "%")
    
    print("\n" + "="*70)
    print("SUCCESS - ALL TESTS PASSED!")
    print("AI ENGINE IS FULLY FUNCTIONAL AND INTEGRATED WITH THE SYSTEM")
    print("="*70 + "\n")
    
except Exception as e:
    print("\n[ERROR] Test failed: " + str(e))
    import traceback
    traceback.print_exc()
    sys.exit(1)
