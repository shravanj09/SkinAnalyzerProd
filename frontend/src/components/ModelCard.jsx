import { useState } from 'react'

function ModelCard({ modelName, modelData, categoryName }) {
  const [expanded, setExpanded] = useState(false)
  const [copiedFeature, setCopiedFeature] = useState(null)

  const status = modelData.status || 'pending'
  const latency = modelData.latency_ms || 0
  const extractedFeatures = modelData.extracted_features || {}
  const extractedCount = modelData.extracted_count || 0
  const totalFeatures = modelData.total_features || 0
  const pendingFeatures = modelData.pending_features || []
  const serviceName = modelData.service_name || 'pending'

  const getStatusIcon = (status) => {
    switch (status) {
      case 'success':
        return '✅'
      case 'error':
        return '❌'
      case 'pending':
        return '⏳'
      default:
        return '⚪'
    }
  }

  const getStatusColor = (status) => {
    switch (status) {
      case 'success':
        return 'text-green-600 bg-green-50'
      case 'error':
        return 'text-red-600 bg-red-50'
      case 'pending':
        return 'text-gray-500 bg-gray-50'
      default:
        return 'text-gray-500 bg-gray-50'
    }
  }

  const formatFeatureName = (name) => {
    return name
      .replace(/_/g, ' ')
      .split(' ')
      .map(word => word.charAt(0).toUpperCase() + word.slice(1))
      .join(' ')
  }

  const formatValue = (value) => {
    if (typeof value === 'number') {
      return value.toFixed(2)
    }
    if (Array.isArray(value)) {
      return value.join(', ')
    }
    return String(value)
  }

  const handleCopy = (featureName, featureValue) => {
    navigator.clipboard.writeText(String(featureValue))
    setCopiedFeature(featureName)
    setTimeout(() => setCopiedFeature(null), 2000)
  }

  return (
    <div className="bg-white rounded-xl border-2 border-gray-200 overflow-hidden shadow-md hover:shadow-xl transition-all duration-200">
      {/* Model Header */}
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full px-5 py-4 flex items-center justify-between hover:bg-gradient-to-r hover:from-gray-50 hover:to-blue-50 transition-all"
      >
        <div className="flex items-center gap-3 flex-1">
          <span className="text-2xl transition-transform duration-200" style={{
            transform: expanded ? 'rotate(90deg)' : 'rotate(0deg)'
          }}>
            ▶
          </span>
          <div className="text-left flex-1">
            <div className="flex items-center gap-3">
              <span className="text-3xl">🤖</span>
              <div>
                <h4 className="font-bold text-lg text-gray-900">{modelName}</h4>
                <div className="flex items-center gap-3 mt-1 text-xs text-gray-600">
                  <span className="font-medium">Service: {serviceName}</span>
                  {status === 'success' && latency > 0 && (
                    <span className="bg-blue-100 text-blue-700 px-2 py-0.5 rounded font-semibold">
                      ⚡ {latency.toFixed(0)}ms
                    </span>
                  )}
                </div>
              </div>
            </div>
          </div>
        </div>

        <div className="flex items-center gap-3">
          {/* Feature Count */}
          <div className="text-right">
            <div className="text-lg font-bold text-gray-900">
              {extractedCount}/{totalFeatures}
            </div>
            <div className="text-xs text-gray-500">features</div>
          </div>

          {/* Status Icon */}
          <div className={`px-3 py-2 rounded-lg ${getStatusColor(status)}`}>
            <span className="text-2xl">{getStatusIcon(status)}</span>
          </div>
        </div>
      </button>

      {/* Expanded Content - Features */}
      {expanded && (
        <div className="px-5 pb-5 pt-2 border-t-2 border-gray-200 bg-gradient-to-br from-gray-50 to-white">
          {/* Extracted Features */}
          {extractedCount > 0 && (
            <div className="mb-4">
              <h5 className="font-bold text-md text-gray-800 mb-3 flex items-center gap-2">
                <span className="text-green-500">✓</span>
                Extracted Features ({extractedCount})
              </h5>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                {Object.entries(extractedFeatures).map(([featureName, featureValue]) => (
                  <div
                    key={featureName}
                    className="bg-white rounded-lg p-4 border-2 border-gray-200 hover:border-blue-400 hover:shadow-md transition-all group"
                  >
                    <div className="flex justify-between items-start">
                      <div className="flex-1 min-w-0">
                        <div className="text-xs font-bold text-gray-600 uppercase tracking-wide mb-2">
                          {formatFeatureName(featureName)}
                        </div>
                        <div className="text-2xl font-bold text-blue-600">
                          {formatValue(featureValue)}
                        </div>
                      </div>
                      <button
                        onClick={(e) => {
                          e.stopPropagation()
                          handleCopy(featureName, featureValue)
                        }}
                        className={`ml-2 p-2 rounded-lg transition-all ${
                          copiedFeature === featureName
                            ? 'bg-green-500 text-white'
                            : 'bg-gray-100 text-gray-500 hover:bg-blue-500 hover:text-white'
                        }`}
                        title={copiedFeature === featureName ? 'Copied!' : 'Copy value'}
                      >
                        {copiedFeature === featureName ? '✓' : '📋'}
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Pending Features */}
          {pendingFeatures.length > 0 && (
            <div>
              <h5 className="font-bold text-md text-gray-800 mb-3 flex items-center gap-2">
                <span className="text-gray-400">⏳</span>
                Pending Features ({pendingFeatures.length})
              </h5>
              <div className="bg-gradient-to-br from-gray-100 to-gray-50 rounded-lg p-4 border-2 border-gray-200">
                <div className="flex flex-wrap gap-2">
                  {pendingFeatures.map((featureName) => (
                    <span
                      key={featureName}
                      className="px-3 py-1.5 bg-white rounded-lg text-xs font-medium text-gray-700 border border-gray-300 shadow-sm"
                    >
                      {formatFeatureName(featureName)}
                    </span>
                  ))}
                </div>
              </div>
            </div>
          )}

          {/* No Features Message */}
          {extractedCount === 0 && pendingFeatures.length === 0 && (
            <div className="text-center py-8 bg-gray-100 rounded-lg border border-gray-200">
              <div className="text-4xl mb-2">📭</div>
              <p className="text-gray-500 font-medium">No features configured for this model</p>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

export default ModelCard
