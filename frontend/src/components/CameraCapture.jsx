import { useState, useRef, useEffect } from 'react'
import { analyzeImage } from '../services/api'

function CameraCapture({ onAnalysisComplete }) {
  const [stream, setStream] = useState(null)
  const [capturing, setCapturing] = useState(false)
  const [analyzing, setAnalyzing] = useState(false)
  const [error, setError] = useState(null)
  const [faceDetected, setFaceDetected] = useState(false)
  const [faceDistance, setFaceDistance] = useState('') // 'good', 'too-far', 'too-close'

  const videoRef = useRef(null)
  const canvasRef = useRef(null)
  const overlayCanvasRef = useRef(null)
  const faceMeshRef = useRef(null)
  const cameraRef = useRef(null)
  const animationFrameRef = useRef(null)

  // Initialize MediaPipe Face Mesh (from CDN)
  useEffect(() => {
    const initFaceMesh = () => {
      try {
        // Wait for MediaPipe libraries to load from CDN
        const checkMediaPipe = setInterval(() => {
          if (window.FaceMesh && window.Camera) {
            clearInterval(checkMediaPipe)

            console.log('MediaPipe libraries loaded from CDN')

            const faceMesh = new window.FaceMesh({
              locateFile: (file) => {
                return `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}`
              }
            })

            faceMesh.setOptions({
              maxNumFaces: 1,
              refineLandmarks: true,
              minDetectionConfidence: 0.5,
              minTrackingConfidence: 0.5
            })

            faceMesh.onResults((results) => onFaceMeshResults(results))

            faceMeshRef.current = { faceMesh, Camera: window.Camera }
            console.log('MediaPipe Face Mesh initialized successfully')
          }
        }, 100)

        // Timeout after 10 seconds
        setTimeout(() => {
          clearInterval(checkMediaPipe)
          if (!faceMeshRef.current) {
            console.error('MediaPipe libraries failed to load from CDN')
          }
        }, 10000)
      } catch (err) {
        console.error('MediaPipe initialization error:', err)
      }
    }

    initFaceMesh()

    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current)
      }
      stopCamera()
    }
  }, [])

  // Process MediaPipe Face Mesh results
  const onFaceMeshResults = (results) => {
    const canvas = overlayCanvasRef.current
    if (!canvas || !window.drawConnectors || !window.FACEMESH_TESSELATION) return

    const ctx = canvas.getContext('2d')
    ctx.clearRect(0, 0, canvas.width, canvas.height)

    if (results.multiFaceLandmarks && results.multiFaceLandmarks.length > 0) {
      const landmarks = results.multiFaceLandmarks[0]
      setFaceDetected(true)

      // Calculate face size (distance from camera)
      const faceWidth = Math.abs(landmarks[454].x - landmarks[234].x) // Left to right face
      const faceHeight = Math.abs(landmarks[10].y - landmarks[152].y) // Top to bottom face

      // Determine if face distance is good (should fill about 40-60% of frame)
      if (faceHeight < 0.35) {
        setFaceDistance('too-far')
      } else if (faceHeight > 0.65) {
        setFaceDistance('too-close')
      } else {
        setFaceDistance('good')
      }

      // Draw face mesh connections (the landmark lines) - using global window functions
      window.drawConnectors(ctx, landmarks, window.FACEMESH_TESSELATION, {
        color: '#00FF00',
        lineWidth: 0.5
      })

      // Draw face oval guide overlay
      drawFaceGuide(ctx, canvas.width, canvas.height, faceDistance === 'good')
    } else {
      setFaceDetected(false)
      setFaceDistance('')
      // Still draw guide even if no face detected
      drawFaceGuide(ctx, canvas.width, canvas.height, false)
    }
  }

  // Draw face-only oval guide (smaller, passport-photo size)
  const drawFaceGuide = (ctx, width, height, isGood) => {
    const centerX = width / 2
    const centerY = height / 2.2 // Slightly higher to account for face position
    const radiusX = width * 0.22  // SMALLER - face only (was 0.3)
    const radiusY = height * 0.35 // SMALLER - face only (was 0.45)

    // Draw semi-transparent dark overlay (Apple-style: lighter, more subtle)
    ctx.save()
    ctx.fillStyle = 'rgba(0, 0, 0, 0.3)'  // Reduced from 0.6 to 0.3
    ctx.fillRect(0, 0, width, height)

    // Cut out oval (face area only)
    ctx.globalCompositeOperation = 'destination-out'
    ctx.beginPath()
    ctx.ellipse(centerX, centerY, radiusX, radiusY, 0, 0, 2 * Math.PI)
    ctx.fill()
    ctx.restore()

    // Draw oval border with color based on detection (Apple-style: thin and subtle)
    const borderColor = isGood ? '#00FF00' : (faceDetected ? '#FFA500' : '#FFFFFF')
    ctx.strokeStyle = borderColor
    ctx.lineWidth = 1  // Thin border like Apple
    ctx.beginPath()
    ctx.ellipse(centerX, centerY, radiusX, radiusY, 0, 0, 2 * Math.PI)
    ctx.stroke()

    // No guide lines - pure Apple minimalism
  }

  const startCamera = async () => {
    try {
      setError(null)
      setFaceDetected(false)
      console.log('Requesting camera access...')

      const mediaStream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 1280 },
          height: { ideal: 720 },
          facingMode: 'user'
        }
      })

      console.log('Camera access granted')
      setStream(mediaStream)
      setCapturing(true)

      setTimeout(async () => {
        const video = videoRef.current
        const canvas = overlayCanvasRef.current

        if (!video || !faceMeshRef.current) return

        video.srcObject = mediaStream

        video.onloadedmetadata = async () => {
          console.log('Video ready:', video.videoWidth, 'x', video.videoHeight)

          // Set canvas dimensions
          canvas.width = video.videoWidth
          canvas.height = video.videoHeight

          // Play video
          await video.play()
          console.log('Video playing')

          // Start MediaPipe Camera
          const { Camera, faceMesh } = faceMeshRef.current
          const camera = new Camera(video, {
            onFrame: async () => {
              if (faceMesh) {
                await faceMesh.send({ image: video })
              }
            },
            width: video.videoWidth,
            height: video.videoHeight
          })

          camera.start()
          cameraRef.current = camera
          console.log('MediaPipe Face Mesh started')
        }

      }, 100)

    } catch (err) {
      console.error('Camera error:', err)
      setError(`Camera access failed: ${err.message}`)
    }
  }

  const stopCamera = () => {
    if (cameraRef.current) {
      cameraRef.current.stop()
      cameraRef.current = null
    }

    if (stream) {
      stream.getTracks().forEach(track => track.stop())
      setStream(null)
    }

    setCapturing(false)
    setFaceDetected(false)
    setFaceDistance('')
  }

  const capturePhoto = async () => {
    if (!videoRef.current || !canvasRef.current) return

    const video = videoRef.current
    const canvas = canvasRef.current

    // Set canvas to video dimensions
    canvas.width = video.videoWidth
    canvas.height = video.videoHeight

    const ctx = canvas.getContext('2d')
    ctx.drawImage(video, 0, 0)

    // Crop to face area only (passport photo style)
    const centerX = canvas.width / 2
    const centerY = canvas.height / 2.2
    const cropWidth = canvas.width * 0.44  // Double the oval radius
    const cropHeight = canvas.height * 0.70

    // Create cropped canvas
    const croppedCanvas = document.createElement('canvas')
    croppedCanvas.width = cropWidth
    croppedCanvas.height = cropHeight
    const croppedCtx = croppedCanvas.getContext('2d')

    // Draw cropped region
    croppedCtx.drawImage(
      canvas,
      centerX - cropWidth / 2,
      centerY - cropHeight / 2,
      cropWidth,
      cropHeight,
      0,
      0,
      cropWidth,
      cropHeight
    )

    // Convert cropped image to blob
    croppedCanvas.toBlob(async (blob) => {
      if (!blob) {
        setError('Failed to capture image')
        return
      }

      setAnalyzing(true)
      setError(null)

      try {
        // Send cropped face image to API
        const result = await analyzeImage(blob)

        // Get cropped image data URL for display
        const imageData = croppedCanvas.toDataURL('image/jpeg', 0.95)

        // Stop camera
        stopCamera()

        // Pass results to parent
        onAnalysisComplete(result, imageData)
      } catch (err) {
        setError(`Analysis failed: ${err.message}`)
        console.error('Analysis error:', err)
      } finally {
        setAnalyzing(false)
      }
    }, 'image/jpeg', 0.95)
  }

  return (
    <div className="max-w-4xl mx-auto">
      <div className="bg-white rounded-lg shadow-lg p-6">
        <h2 className="text-2xl font-bold mb-4">📸 Capture Your Photo</h2>

        {/* Instructions */}
        {!capturing && !analyzing && (
          <div className="bg-gradient-to-r from-blue-50 to-indigo-50 border-2 border-blue-300 rounded-lg p-5 mb-6">
            <h3 className="font-bold text-blue-900 mb-3 text-lg flex items-center gap-2">
              <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              Position Guide for Best Results:
            </h3>
            <div className="grid grid-cols-2 gap-4 text-sm text-blue-800">
              <div className="flex items-start gap-2">
                <span className="text-green-600 font-bold text-lg">✓</span>
                <div>
                  <strong>Face Detection Lines:</strong> Green mesh lines will appear on your face when detected
                </div>
              </div>
              <div className="flex items-start gap-2">
                <span className="text-green-600 font-bold text-lg">✓</span>
                <div>
                  <strong>Distance:</strong> Position so your face fills the small oval (head and neck only)
                </div>
              </div>
              <div className="flex items-start gap-2">
                <span className="text-green-600 font-bold text-lg">✓</span>
                <div>
                  <strong>Alignment:</strong> Align your eyes with the top guide line
                </div>
              </div>
              <div className="flex items-start gap-2">
                <span className="text-green-600 font-bold text-lg">✓</span>
                <div>
                  <strong>Lighting:</strong> Good natural light, no harsh shadows
                </div>
              </div>
              <div className="flex items-start gap-2">
                <span className="text-green-600 font-bold text-lg">✓</span>
                <div>
                  <strong>Expression:</strong> Neutral, no smiling
                </div>
              </div>
              <div className="flex items-start gap-2">
                <span className="text-green-600 font-bold text-lg">✓</span>
                <div>
                  <strong>Remove:</strong> Glasses, hats, and keep hair away from face
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Status indicators (Apple-style: minimal and subtle) */}
        {capturing && !analyzing && (
          <div className="mb-3 flex gap-2 justify-center">
            <div className={`px-3 py-1.5 rounded-full font-medium text-xs ${
              faceDetected ? 'bg-green-100 text-green-800 border border-green-500' : 'bg-gray-100 text-gray-600 border border-gray-300'
            }`}>
              {faceDetected ? '✓ Face Detected' : '⚠ No Face Detected'}
            </div>

            {faceDetected && (
              <div className={`px-3 py-1.5 rounded-full font-medium text-xs ${
                faceDistance === 'good' ? 'bg-green-100 text-green-800 border border-green-500' :
                faceDistance === 'too-far' ? 'bg-orange-100 text-orange-800 border border-orange-500' :
                faceDistance === 'too-close' ? 'bg-orange-100 text-orange-800 border border-orange-500' :
                'bg-gray-100 text-gray-600 border border-gray-300'
              }`}>
                {faceDistance === 'good' && '✓ Perfect Distance'}
                {faceDistance === 'too-far' && '← Move Closer'}
                {faceDistance === 'too-close' && 'Move Back →'}
                {!faceDistance && 'Checking...'}
              </div>
            )}
          </div>
        )}

        {/* Error message */}
        {error && (
          <div className="bg-red-50 border-2 border-red-300 rounded-lg p-4 mb-4">
            <p className="text-red-800 font-semibold">{error}</p>
          </div>
        )}

        {/* Video container */}
        <div className="relative bg-gray-900 rounded-xl overflow-hidden mb-4 shadow-2xl" style={{ paddingTop: '56.25%' }}>

          {/* Video element */}
          {capturing && (
            <video
              ref={videoRef}
              autoPlay
              playsInline
              muted
              className="absolute top-0 left-0 w-full h-full object-cover"
            />
          )}

          {/* Overlay canvas for face mesh + guide */}
          {capturing && (
            <canvas
              ref={overlayCanvasRef}
              className="absolute top-0 left-0 w-full h-full object-cover pointer-events-none"
              style={{ zIndex: 10 }}
            />
          )}

          {/* Camera not started */}
          {!capturing && !analyzing && (
            <div className="absolute top-0 left-0 w-full h-full flex items-center justify-center bg-gradient-to-br from-gray-800 to-gray-900">
              <div className="text-center">
                <svg className="mx-auto h-32 w-32 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 9a2 2 0 012-2h.93a2 2 0 001.664-.89l.812-1.22A2 2 0 0110.07 4h3.86a2 2 0 011.664.89l.812 1.22A2 2 0 0018.07 7H19a2 2 0 012 2v9a2 2 0 01-2 2H5a2 2 0 01-2-2V9z" />
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 13a3 3 0 11-6 0 3 3 0 016 0z" />
                </svg>
                <p className="text-gray-300 mt-6 text-xl">Click "Start Camera" below</p>
              </div>
            </div>
          )}

          {/* Analyzing */}
          {analyzing && (
            <div className="absolute top-0 left-0 w-full h-full flex items-center justify-center bg-black bg-opacity-90 z-30">
              <div className="text-center">
                <div className="inline-block animate-spin rounded-full h-20 w-20 border-t-4 border-b-4 border-blue-500"></div>
                <p className="text-white mt-6 text-xl font-bold">Analyzing Your Skin...</p>
                <p className="text-gray-300 text-sm mt-2">Processing 184 facial features</p>
              </div>
            </div>
          )}
        </div>

        {/* Hidden canvas for capture */}
        <canvas ref={canvasRef} className="hidden" />

        {/* Action buttons */}
        <div className="flex gap-4 justify-center mb-4">
          {!capturing && (
            <button
              onClick={startCamera}
              disabled={analyzing}
              className="px-10 py-4 bg-gradient-to-r from-blue-600 to-indigo-600 text-white rounded-xl hover:from-blue-700 hover:to-indigo-700 disabled:bg-gray-300 disabled:cursor-not-allowed transition-all font-bold text-lg shadow-lg transform hover:scale-105"
            >
              🎥 Start Camera
            </button>
          )}

          {capturing && !analyzing && (
            <>
              <button
                onClick={capturePhoto}
                disabled={!faceDetected || faceDistance !== 'good'}
                className={`px-10 py-4 rounded-xl font-bold text-lg shadow-lg transform transition-all ${
                  faceDetected && faceDistance === 'good'
                    ? 'bg-gradient-to-r from-green-500 to-emerald-500 text-white hover:from-green-600 hover:to-emerald-600 hover:scale-105'
                    : 'bg-gray-300 text-gray-500 cursor-not-allowed'
                }`}
                title={!faceDetected ? 'Waiting for face detection...' : faceDistance !== 'good' ? 'Adjust your distance first' : 'Capture'}
              >
                {faceDetected && faceDistance === 'good' ? '📸 Capture Face' : '⏳ Waiting...'}
              </button>
              <button
                onClick={stopCamera}
                className="px-10 py-4 bg-gradient-to-r from-gray-500 to-gray-600 text-white rounded-xl hover:from-gray-600 hover:to-gray-700 transition-all font-bold text-lg shadow-lg transform hover:scale-105"
              >
                ✕ Cancel
              </button>
            </>
          )}
        </div>

        {/* Real-time instructions */}
        {capturing && !analyzing && (
          <div className="bg-blue-50 border-2 border-blue-300 rounded-lg p-4">
            <div className="flex items-start gap-3">
              <svg className="w-6 h-6 text-blue-600 flex-shrink-0 mt-0.5" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clipRule="evenodd" />
              </svg>
              <div className="text-sm text-blue-800">
                <strong className="font-bold">Live Feedback:</strong>
                {!faceDetected && " Move into camera view - you'll see green mesh lines appear on your face when detected."}
                {faceDetected && faceDistance === 'too-far' && " Move CLOSER to the camera - the green mesh will help guide you."}
                {faceDetected && faceDistance === 'too-close' && " Move BACK from the camera - adjust until the oval turns green."}
                {faceDetected && faceDistance === 'good' && " Perfect! Your face is at the ideal distance. Look straight at camera and capture when ready."}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

export default CameraCapture
