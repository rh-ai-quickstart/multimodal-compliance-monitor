import React, { useState, useEffect, useCallback, useRef } from 'react';
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

function App() {
  const [showConfig, setShowConfig] = useState(false);
  const [configs, setConfigs] = useState([]);
  const [selectedConfigId, setSelectedConfigId] = useState(null);
  const [activeConfigId, setActiveConfigId] = useState(null);
  const switchRequestSeq = useRef(0);

  const handleSelectConfig = useCallback(async (configId) => {
    // Update UI selection immediately; stream switch still waits for backend activation.
    setSelectedConfigId(configId);
    // Switch the feed immediately so toggling sources feels responsive.
    setActiveConfigId(configId);
    const currentSeq = ++switchRequestSeq.current;
    if (configId == null) {
      return;
    }
    axios.post(`${API_URL}/active_config`, { config_id: configId }).catch((err) => {
      // Ignore stale responses if a newer selection already happened.
      if (currentSeq !== switchRequestSeq.current) return;
      console.error('Failed to set active config:', err);
    });
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

  return (
    <div className="App">
      <ConfigModal isOpen={showConfig} onClose={handleConfigModalClose} />
      <LogoBar />
      <h1 className="main-title">
        Multi Modal and Multi Model Monitoring System
      </h1>
      <div className="content-wrapper three-column">
        <aside className="source-column">
          <SourceSection
            configs={configs}
            selectedConfigId={selectedConfigId}
            onSelectConfig={handleSelectConfig}
            onAddVideo={() => setShowConfig(true)}
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
