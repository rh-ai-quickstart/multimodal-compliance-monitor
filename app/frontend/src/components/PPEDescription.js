import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import './PPEDescription.css';
import { API_URL } from '../config';

const describeAlertResult = (alert) => {
  const rule = (alert.rule || '').replace(/^alert\s+(me\s+)?when\s+/i, '').replace(/\.$/, '');

  if (alert.status === 'processing') return `Checking: ${rule}...`;
  if (alert.status === 'error') return alert.error || `Error checking: ${rule}`;
  if (!alert.result && alert.status === 'done') return `✓ 0 violations — ${rule}`;

  const result = alert.result;
  let count = 0;
  if (typeof result === 'number') {
    count = result;
  } else if (Array.isArray(result)) {
    if (result.length === 1 && typeof result[0] === 'object' && result[0] !== null) {
      const vals = Object.values(result[0]);
      if (vals.length === 1 && typeof vals[0] === 'number') {
        count = vals[0];
      } else {
        count = result.length;
      }
    } else {
      count = result.length;
    }
  } else if (typeof result === 'object' && result !== null) {
    count = typeof result.count === 'number' ? result.count
      : typeof result.total === 'number' ? result.total : 0;
  }

  if (count === 0) return `✓ 0 violations — ${rule}`;
  return `⚠ ${count} ${count === 1 ? 'violation' : 'violations'} — ${rule}`;
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
        setAlerts(Array.isArray(data.alerts) ? data.alerts : []);
      } catch (error) {
        console.error('Error fetching latest info:', error);
      }
    };

    fetchLatestInfo();
    const intervalId = setInterval(fetchLatestInfo, 5000);
    return () => clearInterval(intervalId);
  }, [activeConfigId]);

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
