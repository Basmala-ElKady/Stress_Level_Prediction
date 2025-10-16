#!/usr/bin/env python3
"""
Test script to verify model loading and basic functionality
"""

import sys
import os
import traceback

def test_imports():
    """Test if all required packages can be imported"""
    print("🔍 Testing imports...")
    try:
        import streamlit as st
        print("✅ Streamlit imported successfully")
        
        import tensorflow as tf
        print("✅ TensorFlow imported successfully")
        
        import pandas as pd
        print("✅ Pandas imported successfully")
        
        import numpy as np
        print("✅ NumPy imported successfully")
        
        import joblib
        print("✅ Joblib imported successfully")
        
        import sklearn
        print("✅ Scikit-learn imported successfully")
        
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_model_loading():
    """Test if model files can be loaded"""
    print("\n🧠 Testing model loading...")
    
    try:
        # Check if files exist
        model_path = "models/stress_nn_model.h5"
        scaler_X_path = "models/scaler_X.pkl"
        scaler_y_path = "models/scaler_y.pkl"
        
        if not os.path.exists(model_path):
            print(f"❌ Model file not found: {model_path}")
            return False
        print(f"✅ Model file found: {model_path}")
        
        if not os.path.exists(scaler_X_path):
            print(f"❌ Scaler X file not found: {scaler_X_path}")
            return False
        print(f"✅ Scaler X file found: {scaler_X_path}")
        
        if not os.path.exists(scaler_y_path):
            print(f"❌ Scaler Y file not found: {scaler_y_path}")
            return False
        print(f"✅ Scaler Y file found: {scaler_y_path}")
        
        # Try to load the model
        from tensorflow import keras
        model = keras.models.load_model(model_path)
        print("✅ Model loaded successfully")
        print(f"   Model summary: {model.input_shape} -> {model.output_shape}")
        
        # Try to load scalers
        import joblib
        scaler_X = joblib.load(scaler_X_path)
        scaler_y = joblib.load(scaler_y_path)
        print("✅ Scalers loaded successfully")
        
        # Test prediction
        import numpy as np
        test_data = np.array([[800.0, 100.0, 1000.0, 50.0]])
        test_scaled = scaler_X.transform(test_data)
        prediction_scaled = model.predict(test_scaled, verbose=0)
        prediction = scaler_y.inverse_transform(prediction_scaled)
        print(f"✅ Test prediction successful: {prediction[0][0]:.2f} BPM")
        
        return True
        
    except Exception as e:
        print(f"❌ Model loading error: {e}")
        traceback.print_exc()
        return False

def test_streamlit_app():
    """Test if the Streamlit app can be imported"""
    print("\n📱 Testing Streamlit app...")
    
    try:
        # Add app directory to path
        sys.path.insert(0, 'app')
        
        # Try to import the main function
        from app import main
        print("✅ Streamlit app imported successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Streamlit app error: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("🚀 Starting Stress Level Prediction App Tests\n")
    
    # Test imports
    imports_ok = test_imports()
    
    # Test model loading
    model_ok = test_model_loading()
    
    # Test streamlit app
    app_ok = test_streamlit_app()
    
    # Summary
    print("\n" + "="*50)
    print("📊 TEST SUMMARY")
    print("="*50)
    print(f"Imports: {'✅ PASS' if imports_ok else '❌ FAIL'}")
    print(f"Model: {'✅ PASS' if model_ok else '❌ FAIL'}")
    print(f"App: {'✅ PASS' if app_ok else '❌ FAIL'}")
    
    if imports_ok and model_ok and app_ok:
        print("\n🎉 All tests passed! The app should work correctly.")
        print("\nTo run the app, use:")
        print("streamlit run app/app.py")
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")
        
        if not imports_ok:
            print("\n💡 To fix import issues, run:")
            print("pip install -r requirements.txt")
        
        if not model_ok:
            print("\n💡 To fix model issues, check if model files exist in models/ directory")

if __name__ == "__main__":
    main()
