import React, { useState } from 'react';
import LandingPage from './components/LandingPage';
import ImageAnalysis from './components/ImageAnalysis';
import TextAnalysis from './components/TextAnalysis';
import './App.css';

function App() {
  const [currentPage, setCurrentPage] = useState('home'); // 'home', 'image', 'text'

  const renderPage = () => {
    switch (currentPage) {
      case 'image':
        return <ImageAnalysis onBack={() => setCurrentPage('home')} />;
      case 'text':
        return <TextAnalysis onBack={() => setCurrentPage('home')} />;
      default:
        return <LandingPage onNavigate={setCurrentPage} />;
    }
  };

  return (
    <div className="App min-h-screen bg-gradient-to-br from-blue-50 via-purple-50 to-pink-50">
      {renderPage()}
    </div>
  );
}

export default App;
