import { useState } from 'react'
import CameraCapture from './components/CameraCapture'
import ResultsDashboard from './components/ResultsDashboard'

function App() {
  const [analysisResult, setAnalysisResult] = useState(null)
  const [capturedImage, setCapturedImage] = useState(null)

  const handleAnalysisComplete = (result, imageData) => {
    setAnalysisResult(result)
    setCapturedImage(imageData)
  }

  const handleReset = () => {
    setAnalysisResult(null)
    setCapturedImage(null)
  }

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white shadow-sm">
        <div className="max-w-7xl mx-auto px-4 py-4 sm:px-6 lg:px-8">
          <h1 className="text-2xl font-bold text-gray-900">
            AI Facial Analysis Platform
          </h1>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 py-8 sm:px-6 lg:px-8">
        {!analysisResult ? (
          <CameraCapture onAnalysisComplete={handleAnalysisComplete} />
        ) : (
          <ResultsDashboard
            result={analysisResult}
            image={capturedImage}
            onReset={handleReset}
          />
        )}
      </main>

      {/* Footer */}
      <footer className="bg-white border-t mt-12">
        <div className="max-w-7xl mx-auto px-4 py-6 sm:px-6 lg:px-8">
          <p className="text-center text-sm text-gray-500">
            Powered by FastAPI, React, MediaPipe, and Claude AI
          </p>
        </div>
      </footer>
    </div>
  )
}

export default App
