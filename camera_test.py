# camera_test.py - Test camera functionality before running the main app

import cv2
import sys
import numpy as np

def test_camera():
    """Test camera functionality"""
    print("🔍 Testing camera functionality...")
    print("=" * 50)
    
    # List available cameras
    print("1. Checking available cameras...")
    available_cameras = []
    
    for i in range(10):  # Check first 10 camera indices
        try:
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    available_cameras.append(i)
                    print(f"   ✅ Camera {i}: Available ({frame.shape[1]}x{frame.shape[0]})")
                else:
                    print(f"   ❌ Camera {i}: Opens but no frame")
                cap.release()
            else:
                print(f"   ❌ Camera {i}: Cannot open")
        except Exception as e:
            print(f"   ❌ Camera {i}: Error - {e}")
    
    if not available_cameras:
        print("\n❌ No cameras found!")
        print("\n🔧 Troubleshooting steps:")
        print("   1. Check if camera is connected")
        print("   2. Close other apps using camera (Zoom, Skype, etc.)")
        print("   3. Check camera permissions")
        print("   4. Try different USB ports")
        print("   5. Restart your computer")
        return False
    
    print(f"\n✅ Found {len(available_cameras)} working camera(s): {available_cameras}")
    
    # Test the first available camera
    camera_index = available_cameras[0]
    print(f"\n2. Testing camera {camera_index} functionality...")
    
    try:
        # Test with different backends
        backends = [
            (cv2.CAP_ANY, "Default"),
            (cv2.CAP_DSHOW, "DirectShow (Windows)"),
            (cv2.CAP_V4L2, "V4L2 (Linux)"),
            (cv2.CAP_AVFOUNDATION, "AVFoundation (macOS)")
        ]
        
        working_backend = None
        
        for backend, name in backends:
            try:
                cap = cv2.VideoCapture(camera_index, backend)
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        print(f"   ✅ {name}: Working")
                        working_backend = backend
                        cap.release()
                        break
                    else:
                        print(f"   ❌ {name}: Opens but no frame")
                cap.release()
            except Exception as e:
                print(f"   ❌ {name}: Error - {e}")
        
        if working_backend is None:
            print("\n❌ No working camera backend found!")
            return False
        
        # Detailed camera test
        print(f"\n3. Detailed test with working backend...")
        cap = cv2.VideoCapture(camera_index, working_backend)
        
        # Set properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Get actual properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"   Resolution: {width}x{height}")
        print(f"   FPS: {fps}")
        
        # Test frame capture
        print(f"\n4. Testing frame capture (5 frames)...")
        for i in range(5):
            ret, frame = cap.read()
            if ret:
                print(f"   Frame {i+1}: ✅ ({frame.shape})")
            else:
                print(f"   Frame {i+1}: ❌ Failed to capture")
                cap.release()
                return False
        
        cap.release()
        print(f"\n🎉 Camera test completed successfully!")
        print(f"🚀 Your camera is ready for the hand gesture app!")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Camera test failed: {e}")
        return False

def show_live_test():
    """Show live camera feed for testing"""
    print(f"\n5. Live camera test (Press 'q' to quit)...")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot open camera for live test")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to capture frame")
            break
        
        frame_count += 1
        
        # Add frame counter
        cv2.putText(frame, f"Frame: {frame_count}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, "Press 'q' to quit", (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, "If you see this, camera works!", (10, 110),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow('Camera Test - Hand Gesture Recognition', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("✅ Live test completed!")

if __name__ == "__main__":
    print("🤚 Hand Gesture Recognition - Camera Test")
    print("=" * 50)
    
    # Run camera test
    if test_camera():
        # Ask for live test
        response = input("\n🎥 Would you like to run a live camera test? (y/n): ").lower().strip()
        if response == 'y' or response == 'yes':
            show_live_test()
    
    print("\n📋 Next steps:")
    print("   1. If camera test passed: Run 'python app.py'")
    print("   2. If camera test failed: Check troubleshooting steps above")
    print("   3. Open http://localhost:5000 in your browser")
    
    input("\nPress Enter to exit...")