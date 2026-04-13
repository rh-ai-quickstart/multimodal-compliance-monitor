import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { API_URL } from '../config';
import './ConfigModal.css';

/** Validate classes field (new JSON format with name + trackable). Returns { valid, error }. */
function validateClasses(value) {
  const s = (value || '').trim();
  if (!s) return { valid: false, error: 'Classes cannot be empty' };
  try {
    const obj = JSON.parse(s);
    if (typeof obj !== 'object' || obj === null || Array.isArray(obj)) {
      return { valid: false, error: 'JSON must be an object like {"0":{"name":"Person","trackable":true}}' };
    }
    if (Object.keys(obj).length === 0) return { valid: false, error: 'Classes cannot be empty' };
    for (const [idx, v] of Object.entries(obj)) {
      if (typeof v !== 'object' || v === null) {
        return { valid: false, error: `Class "${idx}" must be an object with "name" and "trackable"` };
      }
      const name = v?.name;
      if (!name || typeof name !== 'string' || !name.trim()) {
        return { valid: false, error: `Class "${idx}" must have a non-empty "name"` };
      }
    }
    return { valid: true };
  } catch (e) {
    return { valid: false, error: `Invalid JSON: ${e.message}` };
  }
}

const ConfigModal = ({ isOpen, onClose }) => {
  const [modelUrl, setModelUrl] = useState('');
  const [ovmsModelName, setOvmsModelName] = useState('');
  const [video, setVideo] = useState('');
  const [classes, setClasses] = useState('');
  const [configs, setConfigs] = useState([]);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');
  const [loading, setLoading] = useState(false);
  const [uploading, setUploading] = useState(false);

  const fetchConfigs = React.useCallback(async () => {
    try {
      const res = await axios.get(`${API_URL}/config`);
      setConfigs(res.data);
    } catch (err) {
      setError(err.response?.data?.error || err.message);
    }
  }, []);

  useEffect(() => {
    if (isOpen) {
      fetchConfigs();
      setError('');
      setSuccess('');
      setModelUrl('');
      setOvmsModelName('');
      setVideo('');
      setClasses('');
    }
  }, [isOpen, fetchConfigs]);

  const classesValidation = validateClasses(classes);
  const modelUrlError = !modelUrl.trim() ? 'Model URL is required' : null;
  const modelNameError = !ovmsModelName.trim()
    ? 'OVMS model name is required'
    : null;
  const videoError = !video.trim() ? 'Video source is required' : null;
  const classesError = !classes.trim()
    ? 'Classes cannot be empty'
    : (classesValidation.error ?? null);

  const allFieldsFilled =
    modelUrl.trim() && ovmsModelName.trim() && video.trim() && classes.trim();
  const allFieldsValid = allFieldsFilled && classesValidation.valid;

  const handleAdd = async () => {
    if (!allFieldsValid) return;
    setError('');
    setSuccess('');
    setLoading(true);
    try {
      const classesObj = JSON.parse(classes.trim());
      await axios.post(`${API_URL}/config`, {
        model_url: modelUrl.trim(),
        model_name: ovmsModelName.trim(),
        video_source: video.trim(),
        classes: classesObj,
      });
      await fetchConfigs();
      setModelUrl('');
      setOvmsModelName('');
      setVideo('');
      setClasses('');
      setSuccess('Configuration added successfully.');
    } catch (err) {
      setError(err.response?.data?.error || err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleDelete = async (id) => {
    if (
      !window.confirm(
        'Delete this configuration? All detection classes, tracks, and observations for this config will be removed. This cannot be undone.',
      )
    ) {
      return;
    }
    setError('');
    setSuccess('');
    try {
      await axios.delete(`${API_URL}/config/${id}`);
      await fetchConfigs();
      setSuccess('Configuration deleted.');
    } catch (err) {
      setError(err.response?.data?.error || err.message);
    }
  };

  const handleUpload = async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setUploading(true);
    setError('');
    try {
      const formData = new FormData();
      formData.append('file', file);
      const res = await axios.post(`${API_URL}/config/upload`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      setVideo(res.data.path || res.data.filename || file.name);
    } catch (err) {
      setError(err.response?.data?.error || err.message);
    } finally {
      setUploading(false);
      e.target.value = '';
    }
  };

  const handleClose = () => {
    setError('');
    onClose();
  };

  if (!isOpen) return null;

  return (
    <div className="config-modal-overlay" onClick={handleClose}>
      <div className="config-modal" onClick={(e) => e.stopPropagation()}>
        <div className="config-modal-header">
          <h2>Add New Video Source</h2>
          <button className="config-modal-close" onClick={handleClose} aria-label="Close">
            &times;
          </button>
        </div>

        <div className="config-form">
          <h3>Source Configuration</h3>

          <div className="config-field">
            <label>Model URL (OVMS endpoint)</label>
            <input
              type="text"
              value={modelUrl}
              onChange={(e) => setModelUrl(e.target.value)}
              placeholder="ovms:8081 or https://ovms.example.com:8080"
              aria-invalid={!!modelUrlError}
              aria-describedby={modelUrlError ? 'model-url-error' : undefined}
            />
            {modelUrlError && (
              <div id="model-url-error" className="config-field-error" role="alert">
                {modelUrlError}
              </div>
            )}
          </div>

          <div className="config-field">
            <label>OVMS model name</label>
            <p className="config-hint" id="model-name-hint">
              Tip: enter the served model id exposed by OVMS—for this project that is typically{' '}
              <code>ppe</code> or <code>bird</code> (same name as the weight file stem).
            </p>
            <input
              type="text"
              value={ovmsModelName}
              onChange={(e) => setOvmsModelName(e.target.value)}
              aria-invalid={!!modelNameError}
              aria-describedby={
                modelNameError ? 'model-name-error' : 'model-name-hint'
              }
            />
            {modelNameError && (
              <div id="model-name-error" className="config-field-error" role="alert">
                {modelNameError}
              </div>
            )}
          </div>

          <div className="config-field">
            <label>Video source (MP4 path or RTSP URL)</label>
            <div className="config-video-row">
              <input
                type="text"
                value={video}
                onChange={(e) => setVideo(e.target.value)}
                placeholder="/path/to/video.mp4 or rtsp://username:pass@camera-ip:554/stream"
                aria-invalid={!!videoError}
                aria-describedby={videoError ? 'video-error' : undefined}
              />
              <label className="config-upload-btn">
                <input type="file" accept=".mp4,.avi,.mov,.mkv" onChange={handleUpload} hidden />
                {uploading ? 'Uploading...' : 'Upload'}
              </label>
            </div>
            {videoError && (
              <div id="video-error" className="config-field-error" role="alert">
                {videoError}
              </div>
            )}
          </div>

          <div className="config-field">
            <label>Classes (JSON with name and trackable)</label>
            <p className="config-hint">
              Hint: {`{"0":{"name":"Person","trackable":true},"1":{"name":"Hardhat","trackable":false}}`}
            </p>
            <textarea
              value={classes}
              onChange={(e) => setClasses(e.target.value)}
              placeholder='{"0":{"name":"Person","trackable":true},"1":{"name":"Hardhat","trackable":false},...}'
              rows={4}
              aria-invalid={!!classesError}
              aria-describedby={classesError ? 'classes-error' : undefined}
            />
            {classesError && (
              <div id="classes-error" className="config-field-error" role="alert">
                {classesError}
              </div>
            )}
          </div>

          {error && <div className="config-error" role="alert">{error}</div>}
          {success && <div className="config-success" role="status">{success}</div>}

          <div className="config-actions">
            <button
              className="config-btn config-btn-primary"
              onClick={handleAdd}
              disabled={!allFieldsValid || loading}
              title={!allFieldsValid ? 'Fill all required fields with valid values' : undefined}
            >
              {loading ? 'Adding...' : 'Add'}
            </button>
          </div>
        </div>

        <div className="config-table-section">
          <h3>Saved Configurations</h3>
          <div className="config-table-wrapper">
            <table className="config-table">
              <thead>
                <tr>
                  <th>ID</th>
                  <th>Model URL</th>
                  <th>OVMS name</th>
                  <th>Video source</th>
                  <th>Actions</th>
                </tr>
              </thead>
              <tbody>
                {configs.length === 0 ? (
                  <tr>
                    <td colSpan={5} className="config-table-empty">
                      No configurations yet
                    </td>
                  </tr>
                ) : (
                  configs.map((c) => (
                    <tr key={c.id}>
                      <td>{c.id}</td>
                      <td className="config-table-cell">{c.model_url}</td>
                      <td className="config-table-cell">
                        {c.model_name?.trim() ? c.model_name : '—'}
                      </td>
                      <td className="config-table-cell">{c.video_source}</td>
                      <td>
                        <button
                          type="button"
                          className="config-table-btn config-table-btn-delete"
                          onClick={() => handleDelete(c.id)}
                        >
                          Delete
                        </button>
                      </td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
        </div>

        <div className="config-modal-footer">
          <button className="config-btn config-btn-close" onClick={handleClose}>
            Close
          </button>
        </div>
      </div>
    </div>
  );
};

export default ConfigModal;
