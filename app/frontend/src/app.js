import React, { useState, useEffect, useCallback, useRef } from 'react';
import { Routes, Route, useNavigate } from 'react-router-dom';
import axios from 'axios';
import VideoPlayer from './components/VideoPlayer';
import PPEDescription from './components/PPEDescription';
import ChatBot from './components/ChatBot';
import LogoBar from './components/LogoBar';
import SourceSection from './components/SourceSection';
import ConfigPage from './pages/ConfigPage';
import { API_URL } from './config';
import './App.css';
import './App.custom.css';

function Dashboard() {
  const navigate = useNavigate();
  const [configs, setConfigs] = useState([]);
  const [selectedConfigId, setSelectedConfigId] = useState(null);
  const [activeConfigId, setActiveConfigId] = useState(null);
  const switchRequestSeq = useRef(0);

  const handleSelectConfig = useCallback(async (configId) => {
    setSelectedConfigId(configId);
    const currentSeq = ++switchRequestSeq.current;
    if (configId == null) {
      setActiveConfigId(null);
      return;
    }
    try {
      await axios.post(`${API_URL}/active_config`, { config_id: configId });
    } catch (err) {
      if (currentSeq !== switchRequestSeq.current) return;
      console.error('Failed to set active config:', err);
      return;
    }
    if (currentSeq !== switchRequestSeq.current) return;
    setActiveConfigId(configId);
  }, []);

  const fetchConfigs = useCallback(async () => {
    try {
      const res = await axios.get(`${API_URL}/config`);
      const data = res.data;
      setConfigs(Array.isArray(data) ? data : []);
    } catch (err) {
      setConfigs([]);
    }
  }, []);

  useEffect(() => {
    fetchConfigs();
  }, [fetchConfigs]);

  return (
    <div className="App">
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
            onAddVideo={() => navigate('/configure')}
          />
        </aside>
        <main className="main-column">
          <VideoPlayer hasSource={activeConfigId != null} activeConfigId={activeConfigId} />
          <PPEDescription activeConfigId={activeConfigId} />
        </main>
        <aside className="chat-column">
          <ChatBot activeConfigId={activeConfigId} />
        </aside>
      </div>
    </div>
  );
}

function App() {
  return (
    <Routes>
      <Route path="/configure" element={<ConfigPage />} />
      <Route path="/" element={<Dashboard />} />
    </Routes>
  );
}

export default App;
