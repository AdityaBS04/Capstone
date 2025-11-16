import React, { useState } from 'react';
import axios from 'axios';

const TextAnalysis = ({ onBack }) => {
  const [inputText, setInputText] = useState('');
  const [analyzing, setAnalyzing] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);

  const sampleTexts = [
    "0.5,0.3,0.8,0.2,0.1,0.6,0.4,0.9,0.7,0.3", // Sample microbiome data
    "0.2,0.7,0.4,0.6,0.3,0.5,0.8,0.1,0.4,0.9",
    "0.9,0.1,0.3,0.5,0.7,0.2,0.6,0.4,0.8,0.3"
  ];

  // Disease info for future use when text models are ready
  // const diseaseInfo = {
  //   esophagitis: {
  //     name: 'Esophagitis',
  //     description: 'Inflammation of the esophagus',
  //     color: 'from-red-500 to-orange-500',
  //     icon: '🔴'
  //   },
  //   polyps: {
  //     name: 'Polyps',
  //     description: 'Abnormal tissue growths',
  //     color: 'from-blue-500 to-cyan-500',
  //     icon: '🔵'
  //   },
  //   ulcerative_colitis: {
  //     name: 'Ulcerative Colitis',
  //     description: 'Inflammatory bowel disease',
  //     color: 'from-purple-500 to-pink-500',
  //     icon: '🟣'
  //   }
  // };

  const analyzeText = async () => {
    if (!inputText.trim()) {
      setError('Please enter microbiome data to analyze');
      return;
    }

    setAnalyzing(true);
    setError(null);

    try {
      // Convert comma-separated values to array
      const dataArray = inputText.split(',').map(val => parseFloat(val.trim())).filter(val => !isNaN(val));

      if (dataArray.length === 0) {
        setError('Please enter valid comma-separated numbers (e.g., 0.5,0.3,0.8,...)');
        setAnalyzing(false);
        return;
      }

      const response = await axios.post('http://localhost:5001/api/analyze-text', {
        data: dataArray
      });
      setResults(response.data);
    } catch (err) {
      setError(err.response?.data?.error || 'Failed to analyze microbiome data. Make sure the backend server is running.');
      console.error('Error:', err);
    } finally {
      setAnalyzing(false);
    }
  };

  const loadSample = (text) => {
    setInputText(text);
    setResults(null);
    setError(null);
  };

  const clearText = () => {
    setInputText('');
    setResults(null);
    setError(null);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-50 via-pink-50 to-blue-50 py-12 px-4">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="flex items-center justify-between mb-8 animate-slide-down">
          <button
            onClick={onBack}
            className="flex items-center gap-2 bg-white px-6 py-3 rounded-xl shadow-lg hover:shadow-xl transition-all hover:scale-105"
          >
            <span className="text-2xl">←</span>
            <span className="font-semibold text-gray-700">Back to Home</span>
          </button>
          <h1 className="text-4xl md:text-5xl font-bold bg-gradient-to-r from-purple-600 to-pink-600 bg-clip-text text-transparent">
            🧬 Microbiome Analysis
          </h1>
        </div>

        <div className="grid md:grid-cols-2 gap-8">
          {/* Input Section */}
          <div className="space-y-6 animate-slide-up">
            <div className="bg-white rounded-3xl shadow-2xl p-8">
              <h2 className="text-2xl font-bold text-gray-800 mb-6">Enter Microbiome Data</h2>

              <div className="space-y-4">
                <textarea
                  value={inputText}
                  onChange={(e) => setInputText(e.target.value)}
                  placeholder="Enter microbiome abundance values (comma-separated numbers)&#10;Example: 0.5,0.3,0.8,0.2,0.1,0.6,0.4,0.9,0.7,0.3"
                  className="w-full h-64 p-4 border-2 border-gray-300 rounded-2xl focus:border-purple-500 focus:outline-none resize-none text-gray-700"
                />

                <div className="flex gap-3">
                  <button
                    onClick={clearText}
                    className="flex-1 bg-gray-200 text-gray-700 font-semibold py-3 px-6 rounded-xl hover:bg-gray-300 transition-all"
                  >
                    🗑️ Clear
                  </button>
                  <button
                    onClick={analyzeText}
                    disabled={analyzing || !inputText.trim()}
                    className={`flex-1 font-semibold py-3 px-6 rounded-xl transition-all shadow-lg hover:shadow-xl ${
                      analyzing || !inputText.trim()
                        ? 'bg-gray-400 cursor-not-allowed'
                        : 'bg-gradient-to-r from-purple-600 to-pink-600 text-white hover:from-purple-700 hover:to-pink-700'
                    }`}
                  >
                    {analyzing ? (
                      <span className="flex items-center justify-center gap-2">
                        <div className="w-5 h-5 border-3 border-white border-t-transparent rounded-full animate-spin"></div>
                        Analyzing...
                      </span>
                    ) : (
                      '🧬 Analyze Microbiome'
                    )}
                  </button>
                </div>
              </div>
            </div>

            {/* Sample Texts */}
            <div className="bg-white rounded-3xl shadow-2xl p-8">
              <h3 className="text-xl font-bold text-gray-800 mb-4">📋 Sample Texts</h3>
              <div className="space-y-3">
                {sampleTexts.map((text, index) => (
                  <button
                    key={index}
                    onClick={() => loadSample(text)}
                    className="w-full text-left p-4 bg-gray-50 hover:bg-purple-50 rounded-xl transition-all border-2 border-transparent hover:border-purple-300"
                  >
                    <p className="text-sm text-gray-600 line-clamp-2">{text}</p>
                  </button>
                ))}
              </div>
            </div>

            {/* Coming Soon Notice */}
            <div className="bg-gradient-to-br from-purple-500 to-pink-600 rounded-3xl shadow-2xl p-8 text-white">
              <h3 className="text-xl font-bold mb-4">🚧 Coming Soon</h3>
              <div className="space-y-3">
                <p>Text analysis models are currently in development.</p>
                <div className="space-y-2">
                  <div className="flex items-center gap-2">
                    <span>✅</span>
                    <span>NLP model architecture designed</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <span>⏳</span>
                    <span>Training in progress</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <span>📊</span>
                    <span>Integration pending</span>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Results Section */}
          <div className="animate-slide-up" style={{ animationDelay: '0.1s' }}>
            {error && (
              <div className="bg-yellow-100 border-l-4 border-yellow-500 rounded-2xl p-6 mb-6 shadow-lg">
                <div className="flex items-center gap-3">
                  <span className="text-3xl">⚠️</span>
                  <div>
                    <h3 className="text-lg font-bold text-yellow-800">Notice</h3>
                    <p className="text-yellow-700">{error}</p>
                  </div>
                </div>
              </div>
            )}

            {results ? (
              <div className="space-y-6">
                {/* Prediction Result */}
                <div className="bg-white rounded-3xl shadow-2xl p-8">
                  <h2 className="text-2xl font-bold text-gray-800 mb-6 flex items-center gap-3">
                    <span>🧬</span>
                    Prediction Results
                  </h2>

                  {/* Main Prediction */}
                  <div className={`p-6 rounded-2xl mb-6 ${
                    results.predicted === 'Healthy'
                      ? 'bg-green-50 border-2 border-green-300'
                      : 'bg-red-50 border-2 border-red-300'
                  }`}>
                    <div className="text-center">
                      <div className="text-5xl mb-3">
                        {results.predicted === 'Healthy' ? '✅' : '⚠️'}
                      </div>
                      <h3 className="text-3xl font-bold mb-2">
                        {results.predicted.replace(/_/g, ' ')}
                      </h3>
                      <p className="text-lg text-gray-600 mb-3">
                        Confidence: <span className="font-bold">{results.confidence}</span>
                      </p>
                      {results.disease_prob !== undefined && (
                        <div className="flex justify-center gap-8 mt-4">
                          <div>
                            <p className="text-sm text-gray-600">Healthy</p>
                            <p className="text-2xl font-bold text-green-600">
                              {(results.healthy_prob * 100).toFixed(1)}%
                            </p>
                          </div>
                          <div>
                            <p className="text-sm text-gray-600">Disease</p>
                            <p className="text-2xl font-bold text-red-600">
                              {(results.disease_prob * 100).toFixed(1)}%
                            </p>
                          </div>
                        </div>
                      )}
                    </div>
                  </div>

                  {/* Risk Scores */}
                  {results.risk_weighted_probabilities && (
                    <div className="space-y-4">
                      <h3 className="font-bold text-lg text-gray-700">📊 Risk-Weighted Probabilities:</h3>
                      {Object.entries(results.risk_weighted_probabilities)
                        .sort((a, b) => b[1] - a[1])
                        .map(([disease, probability]) => (
                          <div key={disease} className="space-y-2">
                            <div className="flex justify-between items-center">
                              <span className="font-semibold text-gray-700 flex items-center gap-2">
                                <span>{disease === 'Healthy' ? '✅' : '🦠'}</span>
                                {disease.replace(/_/g, ' ')}
                              </span>
                              <span className="font-bold text-gray-800">
                                {(probability * 100).toFixed(2)}%
                              </span>
                            </div>
                            <div className="w-full bg-gray-200 rounded-full h-4 overflow-hidden">
                              <div
                                className={`h-full transition-all duration-1000 ease-out rounded-full ${
                                  disease === 'Healthy'
                                    ? 'bg-gradient-to-r from-green-400 to-green-600'
                                    : probability > 0.3
                                    ? 'bg-gradient-to-r from-red-400 to-red-600'
                                    : probability > 0.1
                                    ? 'bg-gradient-to-r from-yellow-400 to-orange-500'
                                    : 'bg-gradient-to-r from-blue-400 to-blue-600'
                                }`}
                                style={{ width: `${Math.max(probability * 100, 2)}%` }}
                              ></div>
                            </div>
                          </div>
                        ))}
                    </div>
                  )}

                  {/* Note if using demo */}
                  {results.note && (
                    <div className="mt-6 p-4 bg-yellow-50 border-l-4 border-yellow-400 rounded">
                      <p className="text-sm text-yellow-800">
                        <span className="font-bold">Note:</span> {results.note}
                      </p>
                    </div>
                  )}
                </div>
              </div>
            ) : (
              <div className="bg-white rounded-3xl shadow-2xl p-12 text-center h-full flex flex-col items-center justify-center">
                <div className="text-6xl mb-4">🧠</div>
                <h3 className="text-2xl font-bold text-gray-700 mb-2">Ready to Analyze</h3>
                <p className="text-gray-500 mb-4">Enter medical text to see AI-powered analysis</p>

                {/* Feature Preview */}
                <div className="mt-8 space-y-4 text-left w-full max-w-md">
                  <div className="flex items-start gap-3 p-4 bg-purple-50 rounded-xl">
                    <span className="text-2xl">🎯</span>
                    <div>
                      <p className="font-semibold text-gray-700">Symptom Detection</p>
                      <p className="text-sm text-gray-600">Identify key symptoms and patterns</p>
                    </div>
                  </div>
                  <div className="flex items-start gap-3 p-4 bg-blue-50 rounded-xl">
                    <span className="text-2xl">📊</span>
                    <div>
                      <p className="font-semibold text-gray-700">Disease Prediction</p>
                      <p className="text-sm text-gray-600">AI-powered classification</p>
                    </div>
                  </div>
                  <div className="flex items-start gap-3 p-4 bg-pink-50 rounded-xl">
                    <span className="text-2xl">💡</span>
                    <div>
                      <p className="font-semibold text-gray-700">Confidence Scores</p>
                      <p className="text-sm text-gray-600">Detailed probability metrics</p>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default TextAnalysis;
