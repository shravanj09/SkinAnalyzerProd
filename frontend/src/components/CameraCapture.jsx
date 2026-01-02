import { useState, useRef, useEffect } from 'react'
import { analyzeImage } from '../services/api'

function CameraCapture({ onAnalysisComplete }) {
  const [stream, setStream] = useState(null)
  const [capturing, setCapturing] = useState(false)
  const [analyzing, setAnalyzing] = useState(false)
  const [error, setError] = useState(null)
  const videoRef = useRef(null)
  const canvasRef = useRef(null)

  const startCamera = async () => {
    try {
      setError(null)
      console.log('Requesting camera access...')

      const mediaStream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 1280 },
          height: { ideal: 720 },
          facingMode: 'user'
        }
      })

      console.log('Camera access granted, stream:', mediaStream)
      setStream(mediaStream)
      setCapturing(true)

      // Wait for next render cycle to ensure video element exists
      setTimeout(() => {
        if (videoRef.current) {
          console.log('Setting video source...')
          videoRef.current.srcObject = mediaStream
          videoRef.current.play().then(() => {
            console.log('Video playing successfully')
          }).catch(err => {
            console.error('Error playing video:', err)
            setError(`Failed to start video playback: ${err.message}`)
          })
        } else {
          console.error('Video ref not available')
          setError('Video element not ready')
        }
      }, 100)

    } catch (err) {
      console.error('Camera error:', err)
      setError(`Camera access failed: ${err.message}. Please allow camera permissions.`)
    }
  }

  const stopCamera = () => {
    if (stream) {
      stream.getTracks().forEach(track => track.stop())
      setStream(null)
      setCapturing(false)
    }
  }

  const capturePhoto = async () => {
    if (!videoRef.current || !canvasRef.current) return

    const video = videoRef.current
    const canvas = canvasRef.current

    // Set canvas dimensions to match video
    canvas.width = video.videoWidth
    canvas.height = video.videoHeight

    // Draw video frame to canvas
    const ctx = canvas.getContext('2d')
    ctx.drawImage(video, 0, 0)

    // Convert to blob
    canvas.toBlob(async (blob) => {
      if (!blob) {
        setError('Failed to capture image')
        return
      }

      setAnalyzing(true)
      setError(null)

      try {
        // Send to API for analysis
        const result = await analyzeImage(blob)

        // Get image data URL for display
        const imageData = canvas.toDataURL('image/jpeg')

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

  useEffect(() => {
    return () => {
      stopCamera()
    }
  }, [])

  return (
    <div className="max-w-4xl mx-auto">
      <div className="bg-white rounded-lg shadow-lg p-6">
        <h2 className="text-2xl font-bold mb-4">Capture Your Photo</h2>

        {/* Instructions */}
        {!capturing && !analyzing && (
          <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 mb-6">
            <h3 className="font-semibold text-blue-900 mb-2">Tips for best results:</h3>
            <ul className="text-sm text-blue-800 space-y-1">
              <li>• Face the camera directly with good lighting</li>
              <li>• Remove glasses and hair from face</li>
              <li>• Use natural expression (no smiling)</li>
              <li>• Ensure face fills most of the frame</li>
            </ul>
          </div>
        )}

        {/* Error message */}
        {error && (
          <div className="bg-red-50 border border-red-200 rounded-lg p-4 mb-6">
            <p className="text-red-800">{error}</p>
          </div>
        )}

        {/* Video/Canvas container */}
        <div className="relative bg-gray-900 rounded-lg overflow-hidden mb-6" style={{ paddingTop: '56.25%' }}>
          {capturing && (
            <video
              ref={videoRef}
              autoPlay
              playsInline
              muted
              className="absolute top-0 left-0 w-full h-full object-cover"
            />
          )}
          {!capturing && !analyzing && (
            <div className="absolute top-0 left-0 w-full h-full flex items-center justify-center">
              <div className="text-center">
                <svg className="mx-auto h-24 w-24 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 9a2 2 0 012-2h.93a2 2 0 001.664-.89l.812-1.22A2 2 0 0110.07 4h3.86a2 2 0 011.664.89l.812 1.22A2 2 0 0018.07 7H19a2 2 0 012 2v9a2 2 0 01-2 2H5a2 2 0 01-2-2V9z" />
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 13a3 3 0 11-6 0 3 3 0 016 0z" />
                </svg>
                <p className="text-gray-300 mt-4">Camera not started</p>
              </div>
            </div>
          )}
          {analyzing && (
            <div className="absolute top-0 left-0 w-full h-full flex items-center justify-center bg-black bg-opacity-75">
              <div className="text-center">
                <div className="inline-block animate-spin rounded-full h-16 w-16 border-t-2 border-b-2 border-white"></div>
                <p className="text-white mt-4">Analyzing your skin...</p>
                <p className="text-gray-300 text-sm">This may take 1-2 seconds</p>
              </div>
            </div>
          )}
        </div>

        {/* Hidden canvas for capture */}
        <canvas ref={canvasRef} className="hidden" />

        {/* Action buttons */}
        <div className="flex gap-4 justify-center">
          {!capturing && (
            <button
              onClick={startCamera}
              disabled={analyzing}
              className="px-6 py-3 bg-primary text-white rounded-lg hover:bg-blue-600 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors"
            >
              Open Camera
            </button>
          )}

          {capturing && !analyzing && (
            <>
              <button
                onClick={capturePhoto}
                className="px-6 py-3 bg-primary text-white rounded-lg hover:bg-blue-600 transition-colors"
              >
                Capture Photo
              </button>
              <button
                onClick={stopCamera}
                className="px-6 py-3 bg-gray-200 text-gray-700 rounded-lg hover:bg-gray-300 transition-colors"
              >
                Cancel
              </button>
            </>
          )}
        </div>
      </div>
    </div>
  )
}

export default CameraCapture
