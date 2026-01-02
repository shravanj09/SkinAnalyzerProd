import { useState, useEffect } from 'react'
import ModelCard from './ModelCard'

function CategoryAccordion({ hierarchicalFeatures }) {
  const [expandedCategory, setExpandedCategory] = useState(null)
  const [searchTerm, setSearchTerm] = useState('')

  useEffect(() => {
    console.log('=== CategoryAccordion Mounted ===')
    console.log('Hierarchical features received:', hierarchicalFeatures)
    console.log('Number of categories:', Object.keys(hierarchicalFeatures || {}).length)
  }, [hierarchicalFeatures])

  if (!hierarchicalFeatures || Object.keys(hierarchicalFeatures).length === 0) {
    return (
      <div className="bg-red-50 border-2 border-red-200 rounded-xl p-8 text-center">
        <div className="text-6xl mb-4">❌</div>
        <h4 className="text-xl font-bold text-red-900 mb-2">No Categories Found</h4>
        <p className="text-red-700">
          Hierarchical features data is not available or empty.
        </p>
        <pre className="mt-4 text-left bg-white p-4 rounded text-xs overflow-auto max-h-40">
          {JSON.stringify(hierarchicalFeatures, null, 2)}
        </pre>
      </div>
    )
  }

  const toggleCategory = (categoryName) => {
    console.log('Toggling category:', categoryName)
    setExpandedCategory(expandedCategory === categoryName ? null : categoryName)
  }

  const getStatusColor = (status) => {
    switch (status) {
      case 'complete':
        return 'bg-green-100 text-green-800 border-green-400'
      case 'partial':
        return 'bg-yellow-100 text-yellow-800 border-yellow-400'
      case 'pending':
        return 'bg-gray-100 text-gray-700 border-gray-400'
      default:
        return 'bg-gray-100 text-gray-700 border-gray-400'
    }
  }

  const getStatusIcon = (status) => {
    switch (status) {
      case 'complete':
        return '✓'
      case 'partial':
        return '◐'
      case 'pending':
        return '○'
      default:
        return '○'
    }
  }

  const getProgressColor = (status) => {
    switch (status) {
      case 'complete':
        return 'bg-gradient-to-r from-green-500 to-green-600'
      case 'partial':
        return 'bg-gradient-to-r from-yellow-500 to-yellow-600'
      case 'pending':
        return 'bg-gray-400'
      default:
        return 'bg-gray-400'
    }
  }

  // Filter categories based on search
  const filteredCategories = Object.entries(hierarchicalFeatures).filter(([categoryName]) =>
    categoryName.toLowerCase().includes(searchTerm.toLowerCase())
  )

  // Sort categories by status (complete first, then partial, then pending)
  const sortedCategories = filteredCategories.sort(([, dataA], [, dataB]) => {
    const statusOrder = { complete: 0, partial: 1, pending: 2 }
    return statusOrder[dataA.status || 'pending'] - statusOrder[dataB.status || 'pending']
  })

  return (
    <div className="space-y-4">
      {/* Search Bar */}
      <div className="bg-white rounded-lg p-3 shadow-md border border-gray-200">
        <div className="relative">
          <input
            type="text"
            placeholder="🔍 Search categories..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="w-full px-4 py-2 pl-10 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none"
          />
          <div className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400">
            🔍
          </div>
          {searchTerm && (
            <button
              onClick={() => setSearchTerm('')}
              className="absolute right-3 top-1/2 transform -translate-y-1/2 text-gray-400 hover:text-gray-600"
            >
              ✕
            </button>
          )}
        </div>
        <p className="text-xs text-gray-500 mt-2">
          Showing {filteredCategories.length} of {Object.keys(hierarchicalFeatures).length} categories
        </p>
      </div>

      {/* Categories List */}
      {sortedCategories.length === 0 ? (
        <div className="bg-gray-50 border border-gray-200 rounded-xl p-8 text-center">
          <div className="text-4xl mb-2">🔍</div>
          <p className="text-gray-600">No categories match "{searchTerm}"</p>
        </div>
      ) : (
        sortedCategories.map(([categoryName, categoryData]) => {
          const isExpanded = expandedCategory === categoryName
          const status = categoryData.status || 'pending'
          const extractedCount = categoryData.extracted_count || 0
          const totalPlanned = categoryData.total_planned || 0
          const progress = totalPlanned > 0 ? (extractedCount / totalPlanned) * 100 : 0
          const modelCount = Object.keys(categoryData.models || {}).length

          return (
            <div
              key={categoryName}
              className="border-2 border-gray-200 rounded-xl overflow-hidden bg-white shadow-md hover:shadow-lg transition-all duration-200"
            >
              {/* Category Header */}
              <button
                onClick={() => toggleCategory(categoryName)}
                className="w-full px-5 py-4 flex items-center justify-between hover:bg-gradient-to-r hover:from-blue-50 hover:to-purple-50 transition-all"
              >
                <div className="flex items-center gap-4 flex-1">
                  <span className="text-2xl transition-transform duration-200" style={{
                    transform: isExpanded ? 'rotate(90deg)' : 'rotate(0deg)'
                  }}>
                    ▶
                  </span>
                  <div className="text-left flex-1">
                    <h3 className="font-bold text-lg text-gray-900">
                      {categoryName}
                    </h3>
                    <div className="flex items-center gap-3 mt-1">
                      <p className="text-sm text-gray-600">
                        {extractedCount}/{totalPlanned} features
                      </p>
                      <span className="text-xs text-gray-500">
                        • {modelCount} model{modelCount !== 1 ? 's' : ''}
                      </span>
                    </div>
                  </div>
                </div>

                <div className="flex items-center gap-4">
                  {/* Progress Bar */}
                  <div className="w-32 bg-gray-200 rounded-full h-2.5 shadow-inner">
                    <div
                      className={`h-2.5 rounded-full transition-all duration-500 ${getProgressColor(status)}`}
                      style={{ width: `${progress}%` }}
                    />
                  </div>

                  {/* Status Badge */}
                  <span
                    className={`px-4 py-1.5 rounded-full text-xs font-bold border-2 ${getStatusColor(status)} shadow-sm`}
                  >
                    {getStatusIcon(status)} {status.toUpperCase()}
                  </span>
                </div>
              </button>

              {/* Expanded Content - Models */}
              {isExpanded && (
                <div className="px-5 pb-5 pt-2 bg-gradient-to-br from-gray-50 to-gray-100 border-t-2 border-gray-200">
                  <div className="space-y-4 mt-3">
                    {Object.keys(categoryData.models || {}).length === 0 ? (
                      <div className="bg-white rounded-lg p-6 text-center border border-gray-200">
                        <p className="text-gray-500">No models configured for this category</p>
                      </div>
                    ) : (
                      Object.entries(categoryData.models).map(([modelName, modelData]) => (
                        <ModelCard
                          key={modelName}
                          modelName={modelName}
                          modelData={modelData}
                          categoryName={categoryName}
                        />
                      ))
                    )}
                  </div>
                </div>
              )}
            </div>
          )
        })
      )}
    </div>
  )
}

export default CategoryAccordion
