import { useState, useEffect } from 'react'
import CategoryAccordion from './CategoryAccordion'

function ResultsDashboard({ result, image, onReset }) {
  const [viewMode, setViewMode] = useState('hierarchical') // 'hierarchical' or 'flat'

  // Debug logging
  useEffect(() => {
    console.log('=== ResultsDashboard Mounted ===')
    console.log('Full result object:', result)
    console.log('Hierarchical features:', result?.hierarchical_features)
    console.log('Features count:', result?.feature_count)
  }, [result])

  const features = result?.features || {}
  const featureCount = result?.feature_count || 0
  const overallScore = result?.overall_score || 0
  const processingTime = result?.processing_time_seconds || 0
  const hierarchicalFeatures = result?.hierarchical_features || {}
  const categorizedFeatures = result?.categorized_features || {}

  // Calculate totals from hierarchical structure
  const categoryCount = Object.keys(hierarchicalFeatures).length
  const totalPlanned = Object.values(hierarchicalFeatures).reduce(
    (sum, cat) => sum + (cat.total_planned || 0),
    0
  )
  const totalExtracted = Object.values(hierarchicalFeatures).reduce(
    (sum, cat) => sum + (cat.extracted_count || 0),
    0
  )

  console.log(`Categories: ${categoryCount}, Planned: ${totalPlanned}, Extracted: ${totalExtracted}`)

  return (
    <div className="max-w-7xl mx-auto">
      <div className="bg-white rounded-lg shadow-lg p-6">
        {/* Header */}
        <div className="flex justify-between items-center mb-6">
          <div>
            <h2 className="text-3xl font-bold text-gray-900">Analysis Results</h2>
            <p className="text-sm text-gray-600 mt-1">
              Processing time: {processingTime}s • {featureCount} features extracted
            </p>
          </div>
          <button
            onClick={onReset}
            className="px-6 py-3 bg-gradient-to-r from-gray-600 to-gray-700 text-white rounded-lg hover:from-gray-700 hover:to-gray-800 transition-all shadow-md hover:shadow-lg"
          >
            ← New Analysis
          </button>
        </div>

        {/* Overall Score Card */}
        <div className="bg-gradient-to-br from-blue-500 to-purple-600 rounded-xl p-6 mb-6 text-white shadow-lg">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="text-xl font-bold mb-2">Overall Skin Health Score</h3>
              <p className="text-blue-100 text-sm">
                {totalExtracted > 0 ? `${totalExtracted}/${totalPlanned}` : featureCount} features extracted
                {categoryCount > 0 && ` • ${categoryCount} categories`}
              </p>
            </div>
            <div className="text-center">
              <div className="text-6xl font-bold">
                {overallScore.toFixed(1)}
              </div>
              <div className="text-sm text-blue-100 mt-1">out of 10</div>
            </div>
          </div>

          {/* Progress bar */}
          <div className="mt-4 bg-white bg-opacity-30 rounded-full h-3 overflow-hidden">
            <div
              className="h-full bg-white transition-all duration-500"
              style={{ width: `${(overallScore / 10) * 100}%` }}
            />
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Image Display */}
          <div className="lg:col-span-1">
            <div className="bg-gradient-to-br from-gray-50 to-gray-100 rounded-xl p-4 shadow-md">
              <h3 className="font-bold text-lg mb-3 text-gray-800">Captured Image</h3>
              {image && (
                <img
                  src={image}
                  alt="Captured face"
                  className="w-full rounded-lg shadow-lg border-2 border-white"
                />
              )}
            </div>

            {/* Model Status */}
            {result?.model_results && (
              <div className="mt-4 bg-white rounded-xl p-4 shadow-md border border-gray-200">
                <h3 className="font-bold text-lg mb-3 text-gray-800">Model Performance</h3>
                <div className="space-y-2">
                  {Object.entries(result.model_results).map(([model, status]) => (
                    <div key={model} className="flex justify-between items-center text-sm bg-gray-50 rounded-lg p-2">
                      <span className="text-gray-700 font-medium">{model}</span>
                      <div className="flex items-center gap-2">
                        <span className={status.status === 'success' ? 'text-green-600 font-bold' : 'text-red-600 font-bold'}>
                          {status.status === 'success' ? '✓' : '✗'}
                        </span>
                        {status.latency_ms && (
                          <span className="text-gray-500 text-xs">
                            {Math.round(status.latency_ms)}ms
                          </span>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>

          {/* Features Display */}
          <div className="lg:col-span-2">
            <div className="mb-4 flex items-center justify-between bg-gradient-to-r from-blue-50 to-purple-50 rounded-lg p-4 shadow-sm">
              <div>
                <h3 className="font-bold text-xl text-gray-900">
                  📊 Skin Analysis Features
                </h3>
                <p className="text-sm text-gray-600 mt-1">
                  {categoryCount > 0
                    ? `${categoryCount} categories • ${totalExtracted} features extracted`
                    : `${featureCount} features extracted`
                  }
                </p>
              </div>

              {/* View Mode Toggle */}
              <div className="flex gap-2 bg-white rounded-lg p-1 shadow-sm">
                <button
                  onClick={() => setViewMode('hierarchical')}
                  className={`px-4 py-2 rounded-md text-sm font-medium transition-all ${
                    viewMode === 'hierarchical'
                      ? 'bg-gradient-to-r from-blue-500 to-purple-500 text-white shadow-md'
                      : 'bg-transparent text-gray-700 hover:bg-gray-100'
                  }`}
                >
                  📁 By Category
                </button>
                <button
                  onClick={() => setViewMode('flat')}
                  className={`px-4 py-2 rounded-md text-sm font-medium transition-all ${
                    viewMode === 'flat'
                      ? 'bg-gradient-to-r from-blue-500 to-purple-500 text-white shadow-md'
                      : 'bg-transparent text-gray-700 hover:bg-gray-100'
                  }`}
                >
                  📄 Flat List
                </button>
              </div>
            </div>

            <div className="max-h-[700px] overflow-y-auto pr-2 custom-scrollbar">
              {viewMode === 'hierarchical' ? (
                <>
                  {categoryCount > 0 ? (
                    <CategoryAccordion hierarchicalFeatures={hierarchicalFeatures} />
                  ) : (
                    <div className="bg-yellow-50 border-2 border-yellow-200 rounded-xl p-8 text-center">
                      <div className="text-6xl mb-4">⚠️</div>
                      <h4 className="text-xl font-bold text-yellow-900 mb-2">
                        Hierarchical View Not Available
                      </h4>
                      <p className="text-yellow-700 mb-4">
                        The categorized view is not available. Showing flat list instead.
                      </p>
                      <button
                        onClick={() => setViewMode('flat')}
                        className="px-6 py-3 bg-yellow-500 text-white rounded-lg hover:bg-yellow-600 transition-colors font-medium"
                      >
                        Switch to Flat List
                      </button>
                    </div>
                  )}
                </>
              ) : (
                <div className="bg-white rounded-lg p-4 shadow-md border border-gray-200">
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                    {Object.entries(features).map(([key, value]) => (
                      <div key={key} className="bg-gradient-to-br from-white to-gray-50 rounded-lg p-4 shadow-sm border border-gray-200 hover:shadow-md transition-shadow">
                        <div className="text-xs font-semibold text-gray-600 uppercase tracking-wide mb-1">
                          {key.replace(/_/g, ' ')}
                        </div>
                        <div className="text-2xl font-bold text-gray-900">
                          {typeof value === 'number' ? value.toFixed(2) : String(value)}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Recommendations */}
        {result?.recommendations && (
          <div className="mt-6 bg-gradient-to-r from-green-50 to-blue-50 rounded-xl p-6 border border-green-200 shadow-md">
            <h3 className="font-bold text-xl mb-4 text-gray-900">💡 Personalized Recommendations</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {result.recommendations.products?.length > 0 && (
                <div>
                  <h4 className="font-semibold text-sm text-gray-700 mb-3">Recommended Products:</h4>
                  <ul className="space-y-3">
                    {result.recommendations.products.map((product, idx) => (
                      <li key={idx} className="bg-white rounded-lg p-3 shadow-sm">
                        <strong className="text-gray-900">{product.name}</strong>
                        <p className="text-gray-600 text-xs mt-1">{product.reason}</p>
                      </li>
                    ))}
                  </ul>
                </div>
              )}
              {result.recommendations.routines?.length > 0 && (
                <div>
                  <h4 className="font-semibold text-sm text-gray-700 mb-3">Recommended Routine:</h4>
                  <ul className="space-y-3">
                    {result.recommendations.routines.map((routine, idx) => (
                      <li key={idx} className="bg-white rounded-lg p-3 shadow-sm">
                        <strong className="text-gray-900">{routine.name}</strong>
                        <ul className="text-xs text-gray-600 mt-2 ml-4 space-y-1">
                          {routine.steps?.map((step, stepIdx) => (
                            <li key={stepIdx} className="flex items-start">
                              <span className="mr-2">•</span>
                              <span>{step}</span>
                            </li>
                          ))}
                        </ul>
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          </div>
        )}
      </div>

      {/* Custom Scrollbar Styles */}
      <style>{`
        .custom-scrollbar::-webkit-scrollbar {
          width: 8px;
        }
        .custom-scrollbar::-webkit-scrollbar-track {
          background: #f1f1f1;
          border-radius: 10px;
        }
        .custom-scrollbar::-webkit-scrollbar-thumb {
          background: linear-gradient(to bottom, #3b82f6, #8b5cf6);
          border-radius: 10px;
        }
        .custom-scrollbar::-webkit-scrollbar-thumb:hover {
          background: linear-gradient(to bottom, #2563eb, #7c3aed);
        }
      `}</style>
    </div>
  )
}

export default ResultsDashboard
