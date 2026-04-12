import React, { useState, useEffect, useCallback } from 'react';
import axios from 'axios';
import VideoPlayer from './components/VideoPlayer';
import PPEDescription from './components/PPEDescription';
import ChatBot from './components/ChatBot';
import LogoBar from './components/LogoBar';
import ConfigModal from './components/ConfigModal';
import SourceSection from './components/SourceSection';
import { API_URL } from './config';
import './App.css';
import './App.custom.css';
import architectureDiagram from './itap-demo.png'; // Make sure this path is correct

function App() {
  const [showDiagram, setShowDiagram] = useState(false);
  const [showConfig, setShowConfig] = useState(false);
  const [configs, setConfigs] = useState([]);
  const [activeConfigId, setActiveConfigId] = useState(null);

  const handleSelectConfig = useCallback(async (configId) => {
    if (configId == null) {
      setActiveConfigId(null);
      return;
    }
    try {
      await axios.post(`${API_URL}/active_config`, { config_id: configId });
      // Drive the video URL only after the backend has started the stream (avoids MJPEG before start_streaming).
      setActiveConfigId(configId);
    } catch (err) {
      console.error('Failed to set active config:', err);
    }
  }, []);

  const fetchConfigs = useCallback(async () => {
    try {
      const res = await axios.get(`${API_URL}/config`);
      setConfigs(res.data);
    } catch (err) {
      setConfigs([]);
    }
  }, []);

  useEffect(() => {
    fetchConfigs();
  }, [fetchConfigs]);

  const handleConfigModalClose = () => {
    setShowConfig(false);
    fetchConfigs();
  };

  const toggleDiagram = () => {
    setShowDiagram(!showDiagram);
  };

  return (
    <div className="App">
      <button className="diagram-toggle" onClick={toggleDiagram}>
        Architecture Diagram
      </button>
      
      {showDiagram && (
        <div className="diagram-overlay">
          <img src={architectureDiagram} alt="Architecture Diagram" />
        </div>
      )}

      <ConfigModal isOpen={showConfig} onClose={handleConfigModalClose} />
      <LogoBar onConfigClick={() => setShowConfig(true)} />
      <h1 className="main-title">
        Multi Modal and Multi Model Monitoring System
      </h1>
      <div className="content-wrapper three-column">
        <aside className="source-column">
          <SourceSection
            configs={configs}
            activeConfigId={activeConfigId}
            onSelectConfig={handleSelectConfig}
          />
        </aside>
        <main className="main-column">
          <VideoPlayer hasSource={activeConfigId != null} activeConfigId={activeConfigId} />
          <PPEDescription />
        </main>
        <aside className="chat-column">
          <ChatBot activeConfigId={activeConfigId} />
        </aside>
      </div>
    </div>
  );
}

export default App;
