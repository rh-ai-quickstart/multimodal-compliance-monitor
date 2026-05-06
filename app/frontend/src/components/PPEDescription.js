import React, { useState, useEffect, useCallback, useRef } from 'react';
import axios from 'axios';
import './PPEDescription.css';
import { API_URL } from '../config';

const describeAlertResult = (alert) => {
  const rule = (alert.rule || '').replace(/^alert\s+(me\s+)?when\s+/i, '').replace(/\.$/, '');

  if (alert.status === 'processing') return `Checking: ${rule}...`;
  if (alert.status === 'error') return alert.error || `Error checking: ${rule}`;
  if (!alert.result && alert.status === 'done') return `No findings for: ${rule}`;

  const result = alert.result;
  let count = 0;
  if (typeof result === 'number') count = result;
  else if (Array.isArray(result)) count = result.length;
  else if (typeof result === 'object' && result !== null) {
    count = typeof result.count === 'number' ? result.count
      : typeof result.total === 'number' ? result.total : 1;
  } else {
    count = 1;
  }

  if (count === 0) return `✓ All clear — ${rule}`;
  return `⚠ ${count} ${count === 1 ? 'finding' : 'findings'} — ${rule}`;
};

const PPEDescription = ({ activeConfigId }) => {
  const [description, setDescription] = useState('Initializing...');
  const [summaries, setSummaries] = useState([]);
  const [alerts, setAlerts] = useState([]);
  const prevConfigId = useRef(activeConfigId);

  useEffect(() => {
    if (prevConfigId.current !== activeConfigId) {
      setSummaries([]);
      setAlerts([]);
      setDescription('Initializing...');
      prevConfigId.current = activeConfigId;
    }
  }, [activeConfigId]);

  useEffect(() => {
    if (!activeConfigId) return undefined;

    const fetchLatestInfo = async () => {
      try {
        const response = await axios.get(`${API_URL}/latest_info`);
        const data = response.data;
        if (data.active_config_id !== activeConfigId) return;
        setDescription(data.description);
        setSummaries((prev) => [
          { text: data.summary, isCurrent: true },
          ...prev.slice(0, 2).map(s => ({ ...s, isCurrent: false })),
        ]);
      } catch (error) {
        console.error('Error fetching latest info:', error);
      }
    };

    const intervalId = setInterval(fetchLatestInfo, 5000);
    return () => clearInterval(intervalId);
  }, [activeConfigId]);

  const fetchAlerts = useCallback(async () => {
    if (!activeConfigId) {
      setAlerts([]);
      return;
    }
    try {
      const res = await axios.get(`${API_URL}/alerts/${activeConfigId}`);
      setAlerts(Array.isArray(res.data) ? res.data : []);
    } catch {
      setAlerts([]);
    }
  }, [activeConfigId]);

  useEffect(() => {
    fetchAlerts();
  }, [fetchAlerts]);

  useEffect(() => {
    if (!activeConfigId) return undefined;
    const intervalId = setInterval(fetchAlerts, 5000);
    return () => clearInterval(intervalId);
  }, [activeConfigId, fetchAlerts]);

  if (!activeConfigId) {
    return (
      <div className="ppe-description">
        <div className="description-section">
          <h3>Latest Detection</h3>
          <p className="detection-info">Select a video source to begin.</p>
        </div>
      </div>
    );
  }

  const activeAlerts = alerts.filter(a => a.status === 'done' || a.status === 'processing');

  return (
    <div className="ppe-description">
      <div className="description-section">
        <h3>Latest Detection</h3>
        <p className="detection-info">{description}</p>
      </div>

      <div className="summary-section">
        <h3>Safety Trends</h3>
        <div className="summary-feed">
          {activeAlerts.map((a) => {
            const text = describeAlertResult(a);
            const hasFindings = text.startsWith('⚠');
            const isProcessing = a.status === 'processing';
            return (
              <div
                key={`alert-${a.id}`}
                className={`safety-trends ${
                  hasFindings ? `safety-trends--alert safety-trends--${a.severity || 'medium'}` :
                  isProcessing ? 'safety-trends--processing' :
                  'safety-trends--clear'
                }`}
              >
                <pre>{text}</pre>
              </div>
            );
          })}
          {summaries.length === 0 && activeAlerts.length === 0 ? (
            <div className="safety-trends">
              <pre>Processing video...</pre>
            </div>
          ) : (
            summaries.map((summary, index) => (
              <div
                key={`trend-${index}`}
                className={`safety-trends ${summary.isCurrent ? 'current-summary' : ''}`}
              >
                <pre>{summary.text}</pre>
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  );
};

export default PPEDescription;
