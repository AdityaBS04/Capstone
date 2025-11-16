import React, { useState, useRef } from 'react';
import axios from 'axios';

const ImageAnalysis = ({ onBack }) => {
  const [selectedImage, setSelectedImage] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);
  const fileInputRef = useRef(null);

  const diseaseInfo = {
    esophagitis: {
      name: 'Esophagitis',
      description: 'Inflammation of the esophagus, often caused by acid reflux',
      color: 'from-red-500 to-orange-500',
      icon: '🔴'
    },
    polyps: {
      name: 'Polyps',
      description: 'Abnormal tissue growths in the digestive tract',
      color: 'from-blue-500 to-cyan-500',
      icon: '🔵'
    },
    ulcerative_colitis: {
      name: 'Ulcerative Colitis',
      description: 'Chronic inflammatory bowel disease',
      color: 'from-purple-500 to-pink-500',
      icon: '🟣'
    }
  };

  const handleImageSelect = (e) => {
    const file = e.target.files[0];
    if (file) {
      setSelectedImage(file);
      setPreviewUrl(URL.createObjectURL(file));
      setResults(null);
      setError(null);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
      setSelectedImage(file);
      setPreviewUrl(URL.createObjectURL(file));
      setResults(null);
      setError(null);
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
  };

  const analyzeImage = async () => {
    if (!selectedImage) return;

    setAnalyzing(true);
    setError(null);

    const formData = new FormData();
    formData.append('image', selectedImage);

    try {
      const response = await axios.post('http://localhost:5001/api/analyze-image', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      setResults(response.data);
    } catch (err) {
      setError(err.response?.data?.error || 'Failed to analyze image. Make sure the backend server is running.');
      console.error('Error:', err);
    } finally {
      setAnalyzing(false);
    }
  };

  const resetAnalysis = () => {
    setSelectedImage(null);
    setPreviewUrl(null);
    setResults(null);
    setError(null);
  };

  const getConfidenceColor = (confidence) => {
    if (confidence >= 0.8) return 'text-green-600 bg-green-100';
    if (confidence >= 0.6) return 'text-yellow-600 bg-yellow-100';
    return 'text-orange-600 bg-orange-100';
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-purple-50 to-pink-50 py-12 px-4">
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
          <h1 className="text-4xl md:text-5xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
            🖼️ Image Analysis
          </h1>
        </div>

        <div className="grid md:grid-cols-2 gap-8">
          {/* Upload Section */}
          <div className="space-y-6 animate-slide-up">
            <div className="bg-white rounded-3xl shadow-2xl p-8">
              <h2 className="text-2xl font-bold text-gray-800 mb-6">Upload Medical Image</h2>

              {!previewUrl ? (
                <div
                  onDrop={handleDrop}
                  onDragOver={handleDragOver}
                  onClick={() => fileInputRef.current?.click()}
                  className="border-4 border-dashed border-gray-300 rounded-2xl p-12 text-center cursor-pointer hover:border-blue-500 hover:bg-blue-50 transition-all"
                >
                  <div className="text-6xl mb-4">📤</div>
                  <p className="text-xl font-semibold text-gray-700 mb-2">
                    Drop image here or click to browse
                  </p>
                  <p className="text-gray-500">Supports JPG, PNG</p>
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept="image/*"
                    onChange={handleImageSelect}
                    className="hidden"
                  />
                </div>
              ) : (
                <div className="space-y-4">
                  <div className="relative rounded-2xl overflow-hidden shadow-xl">
                    <img
                      src={previewUrl}
                      alt="Preview"
                      className="w-full h-96 object-contain bg-gray-100"
                    />
                  </div>
                  <div className="flex gap-3">
                    <button
                      onClick={resetAnalysis}
                      className="flex-1 bg-gray-200 text-gray-700 font-semibold py-3 px-6 rounded-xl hover:bg-gray-300 transition-all"
                    >
                      🔄 Change Image
                    </button>
                    <button
                      onClick={analyzeImage}
                      disabled={analyzing}
                      className={`flex-1 font-semibold py-3 px-6 rounded-xl transition-all shadow-lg hover:shadow-xl ${
                        analyzing
                          ? 'bg-gray-400 cursor-not-allowed'
                          : 'bg-gradient-to-r from-blue-600 to-purple-600 text-white hover:from-blue-700 hover:to-purple-700'
                      }`}
                    >
                      {analyzing ? (
                        <span className="flex items-center justify-center gap-2">
                          <div className="w-5 h-5 border-3 border-white border-t-transparent rounded-full animate-spin"></div>
                          Analyzing...
                        </span>
                      ) : (
                        '🔬 Analyze Now'
                      )}
                    </button>
                  </div>
                </div>
              )}
            </div>

            {/* Information Card */}
            <div className="bg-gradient-to-br from-blue-500 to-purple-600 rounded-3xl shadow-2xl p-8 text-white">
              <h3 className="text-xl font-bold mb-4">📋 How It Works</h3>
              <div className="space-y-3">
                <div className="flex items-start gap-3">
                  <span className="text-2xl">1️⃣</span>
                  <p>Upload a medical endoscopy image</p>
                </div>
                <div className="flex items-start gap-3">
                  <span className="text-2xl">2️⃣</span>
                  <p>Our AI analyzes with 3 preprocessing methods</p>
                </div>
                <div className="flex items-start gap-3">
                  <span className="text-2xl">3️⃣</span>
                  <p>Get instant disease detection and classification</p>
                </div>
                <div className="flex items-start gap-3">
                  <span className="text-2xl">4️⃣</span>
                  <p>View detailed probability scores</p>
                </div>
              </div>
            </div>
          </div>

          {/* Results Section */}
          <div className="animate-slide-up" style={{ animationDelay: '0.1s' }}>
            {error && (
              <div className="bg-red-100 border-l-4 border-red-500 rounded-2xl p-6 mb-6 shadow-lg">
                <div className="flex items-center gap-3">
                  <span className="text-3xl">⚠️</span>
                  <div>
                    <h3 className="text-lg font-bold text-red-800">Error</h3>
                    <p className="text-red-700">{error}</p>
                  </div>
                </div>
              </div>
            )}

            {results ? (
              <div className="space-y-6">
                {/* Binary Classification Result */}
                {results.binary && (
                  <div className="bg-white rounded-3xl shadow-2xl p-8">
                    <h2 className="text-2xl font-bold text-gray-800 mb-6 flex items-center gap-3">
                      <span>🎯</span>
                      Disease Detection
                    </h2>
                    <div className={`p-6 rounded-2xl ${results.binary.has_disease ? 'bg-red-50 border-2 border-red-300' : 'bg-green-50 border-2 border-green-300'}`}>
                      <div className="text-center">
                        <div className="text-5xl mb-3">
                          {results.binary.has_disease ? '🚨' : '✅'}
                        </div>
                        <h3 className="text-3xl font-bold mb-2">
                          {results.binary.has_disease ? 'Disease Detected' : 'No Disease Detected'}
                        </h3>
                        <p className="text-lg text-gray-600">
                          Confidence: <span className="font-bold">{(results.binary.confidence * 100).toFixed(1)}%</span>
                        </p>
                      </div>
                    </div>
                  </div>
                )}

                {/* Disease Classification Results */}
                {results.disease && results.binary?.has_disease && (
                  <div className="bg-white rounded-3xl shadow-2xl p-8">
                    <h2 className="text-2xl font-bold text-gray-800 mb-6 flex items-center gap-3">
                      <span>🔬</span>
                      Disease Classification
                    </h2>

                    {/* Predicted Disease */}
                    <div className={`bg-gradient-to-r ${diseaseInfo[results.disease.predicted_class_name]?.color || 'from-gray-500 to-gray-700'} rounded-2xl p-6 mb-6 text-white shadow-xl`}>
                      <div className="text-center">
                        <div className="text-5xl mb-3">
                          {diseaseInfo[results.disease.predicted_class_name]?.icon || '🔍'}
                        </div>
                        <h3 className="text-3xl font-bold mb-2">
                          {diseaseInfo[results.disease.predicted_class_name]?.name || results.disease.predicted_class_name}
                        </h3>
                        <p className="text-white/90 mb-3">
                          {diseaseInfo[results.disease.predicted_class_name]?.description}
                        </p>
                        <div className={`inline-block px-6 py-2 rounded-full ${getConfidenceColor(results.disease.confidence)}`}>
                          <span className="font-bold text-lg">
                            {(results.disease.confidence * 100).toFixed(1)}% Confidence
                          </span>
                        </div>
                      </div>
                    </div>

                    {/* Probability Breakdown */}
                    <div className="space-y-4">
                      <h3 className="font-bold text-lg text-gray-700">📊 Probability Breakdown:</h3>
                      {Object.entries(results.disease.matched_probabilities).map(([disease, probability]) => (
                        <div key={disease} className="space-y-2">
                          <div className="flex justify-between items-center">
                            <span className="font-semibold text-gray-700 flex items-center gap-2">
                              <span>{diseaseInfo[disease]?.icon}</span>
                              {diseaseInfo[disease]?.name || disease}
                            </span>
                            <span className="font-bold text-gray-800">
                              {(probability * 100).toFixed(2)}%
                            </span>
                          </div>
                          <div className="w-full bg-gray-200 rounded-full h-4 overflow-hidden">
                            <div
                              className={`h-full bg-gradient-to-r ${diseaseInfo[disease]?.color || 'from-gray-400 to-gray-600'} transition-all duration-1000 ease-out rounded-full`}
                              style={{ width: `${probability * 100}%` }}
                            ></div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            ) : (
              <div className="bg-white rounded-3xl shadow-2xl p-12 text-center">
                <div className="text-6xl mb-4">🔬</div>
                <h3 className="text-2xl font-bold text-gray-700 mb-2">Ready to Analyze</h3>
                <p className="text-gray-500">Upload an image to see detailed results here</p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default ImageAnalysis;
