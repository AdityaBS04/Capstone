import React from 'react';

const LandingPage = ({ onNavigate }) => {
  return (
    <div className="min-h-screen flex flex-col items-center justify-center px-4 py-12 gradient-animate bg-gradient-to-br from-blue-500 via-purple-500 to-pink-500">
      {/* Floating particles background */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <div className="absolute w-96 h-96 bg-white/10 rounded-full blur-3xl top-20 left-20 floating"></div>
        <div className="absolute w-96 h-96 bg-white/10 rounded-full blur-3xl bottom-20 right-20 floating" style={{ animationDelay: '1s' }}></div>
      </div>

      {/* Main Content */}
      <div className="relative z-10 max-w-6xl w-full">
        {/* Hero Section */}
        <div className="text-center mb-16 animate-fade-in">
          <div className="mb-8">
            <h1 className="text-6xl md:text-8xl font-bold text-white mb-4 drop-shadow-2xl">
              🔬 MediScan AI
            </h1>
            <div className="h-2 w-48 mx-auto bg-gradient-to-r from-yellow-400 via-pink-400 to-blue-400 rounded-full"></div>
          </div>
          <p className="text-2xl md:text-3xl text-white/90 font-light mb-6 drop-shadow-lg">
            Advanced Disease Detection & Classification
          </p>
          <p className="text-lg md:text-xl text-white/80 max-w-3xl mx-auto drop-shadow-md">
            Powered by cutting-edge AI to identify and classify gastrointestinal diseases
            with high accuracy using image and text analysis
          </p>
        </div>

        {/* Feature Cards */}
        <div className="grid md:grid-cols-2 gap-8 mb-16">
          {/* Image Analysis Card */}
          <div
            className="glass rounded-3xl p-8 hover:scale-105 transform transition-all duration-300 cursor-pointer group shadow-2xl"
            onClick={() => onNavigate('image')}
          >
            <div className="flex flex-col items-center text-center">
              <div className="w-32 h-32 bg-gradient-to-br from-blue-400 to-cyan-400 rounded-3xl flex items-center justify-center mb-6 group-hover:rotate-6 transition-transform shadow-xl">
                <span className="text-6xl">🖼️</span>
              </div>
              <h2 className="text-3xl font-bold text-white mb-4">Image Analysis</h2>
              <p className="text-white/90 text-lg mb-6">
                Upload medical images for instant disease detection and classification
              </p>
              <div className="space-y-3 text-left w-full">
                <div className="flex items-start gap-3">
                  <span className="text-2xl">✨</span>
                  <div>
                    <p className="text-white font-semibold">Multi-Stage Detection</p>
                    <p className="text-white/80 text-sm">Binary identification + Classification</p>
                  </div>
                </div>
                <div className="flex items-start gap-3">
                  <span className="text-2xl">🎯</span>
                  <div>
                    <p className="text-white font-semibold">3 Disease Types</p>
                    <p className="text-white/80 text-sm">Esophagitis, Polyps, Ulcerative Colitis</p>
                  </div>
                </div>
                <div className="flex items-start gap-3">
                  <span className="text-2xl">📊</span>
                  <div>
                    <p className="text-white font-semibold">Probability Scores</p>
                    <p className="text-white/80 text-sm">Detailed confidence metrics</p>
                  </div>
                </div>
              </div>
              <button className="mt-8 w-full bg-white text-blue-600 font-bold py-4 px-8 rounded-2xl hover:bg-blue-50 transition-all shadow-lg hover:shadow-xl">
                Analyze Image →
              </button>
            </div>
          </div>

          {/* Text Analysis Card */}
          <div
            className="glass rounded-3xl p-8 hover:scale-105 transform transition-all duration-300 cursor-pointer group shadow-2xl"
            onClick={() => onNavigate('text')}
          >
            <div className="flex flex-col items-center text-center">
              <div className="w-32 h-32 bg-gradient-to-br from-purple-400 to-pink-400 rounded-3xl flex items-center justify-center mb-6 group-hover:rotate-6 transition-transform shadow-xl">
                <span className="text-6xl">📝</span>
              </div>
              <h2 className="text-3xl font-bold text-white mb-4">Text Analysis</h2>
              <p className="text-white/90 text-lg mb-6">
                Analyze medical reports and symptoms for disease prediction
              </p>
              <div className="space-y-3 text-left w-full">
                <div className="flex items-start gap-3">
                  <span className="text-2xl">🧠</span>
                  <div>
                    <p className="text-white font-semibold">NLP-Powered</p>
                    <p className="text-white/80 text-sm">Advanced text understanding</p>
                  </div>
                </div>
                <div className="flex items-start gap-3">
                  <span className="text-2xl">⚡</span>
                  <div>
                    <p className="text-white font-semibold">Fast Analysis</p>
                    <p className="text-white/80 text-sm">Instant results from text input</p>
                  </div>
                </div>
                <div className="flex items-start gap-3">
                  <span className="text-2xl">🔍</span>
                  <div>
                    <p className="text-white font-semibold">Coming Soon</p>
                    <p className="text-white/80 text-sm">Text models in development</p>
                  </div>
                </div>
              </div>
              <button className="mt-8 w-full bg-white text-purple-600 font-bold py-4 px-8 rounded-2xl hover:bg-purple-50 transition-all shadow-lg hover:shadow-xl">
                Analyze Text →
              </button>
            </div>
          </div>
        </div>

        {/* Info Banner */}
        <div className="glass rounded-2xl p-6 text-center shadow-xl">
          <p className="text-white text-sm md:text-base">
            <span className="font-bold">⚠️ Medical Disclaimer:</span> This tool is for educational and research purposes only.
            Always consult healthcare professionals for medical diagnosis and treatment.
          </p>
        </div>

        {/* Stats Section */}
        <div className="mt-12 grid grid-cols-3 gap-6 text-center">
          <div className="glass rounded-2xl p-6 shadow-xl">
            <div className="text-4xl font-bold text-white mb-2">88-93%</div>
            <div className="text-white/80">Accuracy</div>
          </div>
          <div className="glass rounded-2xl p-6 shadow-xl">
            <div className="text-4xl font-bold text-white mb-2">3</div>
            <div className="text-white/80">Disease Types</div>
          </div>
          <div className="glass rounded-2xl p-6 shadow-xl">
            <div className="text-4xl font-bold text-white mb-2">AI</div>
            <div className="text-white/80">Powered</div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default LandingPage;
