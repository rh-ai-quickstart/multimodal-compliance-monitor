import React, { useState, useEffect, useCallback } from 'react';
import axios from 'axios';
import { API_URL } from '../config';
import './AlertPanel.css';

const renderAlertResult = (result) => {
  if (!result) return 'No data yet.';
  if (typeof result === 'number') return `Matches: ${result}`;
  if (Array.isArray(result)) {
    if (result.length === 1 && typeof result[0] === 'object' && result[0] !== null) {
      const vals = Object.values(result[0]);
      if (vals.length === 1 && typeof vals[0] === 'number') {
        return vals[0] === 0 ? 'No matches returned.' : `Matches: ${vals[0]}`;
      }
    }
    return result.length ? `Matches: ${result.length}` : 'No matches returned.';
  }
  if (typeof result === 'object') {
    if (typeof result.count === 'number') return `Matches: ${result.count}`;
    if (typeof result.total === 'number') return `Matches: ${result.total}`;
    return 'Result available.';
  }
  return String(result);
};

const AlertPanel = ({ configId }) => {
  const [alertRule, setAlertRule] = useState('');
  const [alertSeverity, setAlertSeverity] = useState('medium');
  const [alerts, setAlerts] = useState([]);
  const [alertsLoading, setAlertsLoading] = useState(false);
  const [alertSubmitting, setAlertSubmitting] = useState(false);
  const [editingAlertId, setEditingAlertId] = useState(null);
  const [editingRule, setEditingRule] = useState('');
  const [editingSeverity, setEditingSeverity] = useState('medium');
  const [alertSaving, setAlertSaving] = useState(false);
  const [deletingAlertId, setDeletingAlertId] = useState(null);
  const [alertsError, setAlertsError] = useState('');

  const fetchAlerts = useCallback(async () => {
    if (!configId) return;
    setAlertsLoading(true);
    setAlertsError('');
    try {
      const res = await axios.get(`${API_URL}/alerts/${configId}`);
      setAlerts(Array.isArray(res.data) ? res.data : []);
    } catch (err) {
      setAlertsError(err.response?.data?.error || err.message);
      setAlerts([]);
    } finally {
      setAlertsLoading(false);
    }
  }, [configId]);

  useEffect(() => {
    fetchAlerts();
  }, [fetchAlerts]);

  useEffect(() => {
    if (!configId) return undefined;
    const intervalId = setInterval(fetchAlerts, 5000);
    return () => clearInterval(intervalId);
  }, [configId, fetchAlerts]);

  const handleCreateAlert = async (e) => {
    e.preventDefault();
    const rule = alertRule.trim();
    if (!configId || !rule) return;
    setAlertSubmitting(true);
    setAlertsError('');
    try {
      await axios.post(`${API_URL}/alerts`, {
        app_config_id: configId,
        rule,
        severity: alertSeverity,
      });
      setAlertRule('');
      await fetchAlerts();
    } catch (err) {
      setAlertsError(err.response?.data?.error || err.message);
    } finally {
      setAlertSubmitting(false);
    }
  };

  const startEdit = (alert) => {
    setEditingAlertId(alert.id);
    setEditingRule(alert.rule || '');
    setEditingSeverity(alert.severity || 'medium');
    setAlertsError('');
  };

  const cancelEdit = () => {
    setEditingAlertId(null);
    setEditingRule('');
    setEditingSeverity('medium');
  };

  const handleSaveAlert = async (alertId) => {
    if (!configId || !editingRule.trim()) return;
    setAlertSaving(true);
    setAlertsError('');
    try {
      await axios.patch(`${API_URL}/alerts/${configId}/${alertId}`, {
        rule: editingRule.trim(),
        severity: editingSeverity,
      });
      cancelEdit();
      await fetchAlerts();
    } catch (err) {
      setAlertsError(err.response?.data?.error || err.message);
    } finally {
      setAlertSaving(false);
    }
  };

  const handleDeleteAlert = async (alertId) => {
    if (!configId) return;
    if (!window.confirm('Delete this alert?')) return;
    setDeletingAlertId(alertId);
    setAlertsError('');
    try {
      await axios.delete(`${API_URL}/alerts/${configId}/${alertId}`);
      if (editingAlertId === alertId) cancelEdit();
      await fetchAlerts();
    } catch (err) {
      setAlertsError(err.response?.data?.error || err.message);
    } finally {
      setDeletingAlertId(null);
    }
  };

  return (
    <div className="alert-panel">
      <div className="alert-panel__header">Alerts</div>

      <form className="alert-panel__form" onSubmit={handleCreateAlert}>
        <textarea
          value={alertRule}
          onChange={(e) => setAlertRule(e.target.value)}
          placeholder="Describe an alert rule in plain English..."
          className="alert-panel__textarea"
        />
        <div className="alert-panel__form-row">
          <select
            value={alertSeverity}
            onChange={(e) => setAlertSeverity(e.target.value)}
            className="alert-panel__select"
          >
            <option value="low">Low</option>
            <option value="medium">Medium</option>
            <option value="high">High</option>
          </select>
          <button
            type="submit"
            className="alert-panel__create-btn"
            disabled={alertSubmitting || !alertRule.trim()}
          >
            {alertSubmitting ? 'Creating...' : 'Create Alert'}
          </button>
        </div>
      </form>

      {alertsError ? (
        <div className="alert-panel__error" role="alert">{alertsError}</div>
      ) : null}

      <div className="alert-panel__list">
        {alertsLoading ? (
          <div className="alert-panel__empty">Loading...</div>
        ) : alerts.length === 0 ? (
          <div className="alert-panel__empty">No alerts for this source.</div>
        ) : (
          alerts.map((a) => (
            <article key={a.id} className="alert-card">
              <div className="alert-card__top">
                <span className={`alert-card__badge alert-card__badge--${a.severity || 'medium'}`}>
                  {a.severity || 'medium'}
                </span>
                <span className={`alert-card__status alert-card__status--${a.status || 'done'}`}>
                  {a.status || 'done'}
                </span>
              </div>
              {editingAlertId === a.id ? (
                <div className="alert-card__edit">
                  <textarea
                    value={editingRule}
                    onChange={(e) => setEditingRule(e.target.value)}
                    className="alert-panel__textarea"
                  />
                  <select
                    value={editingSeverity}
                    onChange={(e) => setEditingSeverity(e.target.value)}
                    className="alert-panel__select"
                  >
                    <option value="low">Low</option>
                    <option value="medium">Medium</option>
                    <option value="high">High</option>
                  </select>
                  <div className="alert-card__actions">
                    <button
                      type="button"
                      className="alert-card__btn alert-card__btn--primary"
                      disabled={alertSaving || !editingRule.trim()}
                      onClick={() => handleSaveAlert(a.id)}
                    >
                      {alertSaving ? 'Saving...' : 'Save'}
                    </button>
                    <button
                      type="button"
                      className="alert-card__btn"
                      onClick={cancelEdit}
                      disabled={alertSaving}
                    >
                      Cancel
                    </button>
                  </div>
                </div>
              ) : (
                <>
                  <p className="alert-card__rule">{a.rule}</p>
                  <p className="alert-card__result">{renderAlertResult(a.result)}</p>
                  {a.error ? <p className="alert-card__err">{a.error}</p> : null}
                  <div className="alert-card__actions">
                    <button type="button" className="alert-card__btn" onClick={() => startEdit(a)}>
                      Edit
                    </button>
                    <button
                      type="button"
                      className="alert-card__btn alert-card__btn--danger"
                      onClick={() => handleDeleteAlert(a.id)}
                      disabled={deletingAlertId === a.id}
                    >
                      {deletingAlertId === a.id ? 'Deleting...' : 'Delete'}
                    </button>
                  </div>
                </>
              )}
            </article>
          ))
        )}
      </div>
    </div>
  );
};

export default AlertPanel;
