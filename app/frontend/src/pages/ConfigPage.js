import React, { useState, useEffect, useCallback, useRef } from 'react';
import { Link } from 'react-router-dom';
import axios from 'axios';
import { API_URL } from '../config';
import LogoBar from '../components/LogoBar';
import AlertPanel from '../components/AlertPanel';
import './ConfigPage.css';

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

function videoDisplayLabel(src) {
  if (!src || typeof src !== 'string') return '—';
  const tail = src.split(/[/\\]/).pop() || src;
  return tail.split('?')[0].split('#')[0] || '—';
}

function classCount(classes) {
  if (!classes || typeof classes !== 'object') return 0;
  return Object.keys(classes).length;
}

function formatCreatedAt(iso) {
  if (!iso) return null;
  try {
    const d = new Date(iso);
    if (Number.isNaN(d.getTime())) return null;
    return d.toLocaleString(undefined, {
      dateStyle: 'medium',
      timeStyle: 'short',
    });
  } catch {
    return null;
  }
}

const emptyForm = () => ({
  modelUrl: '',
  ovmsModelName: '',
  video: '',
  classes: '',
});

const ConfigPage = () => {
  const [view, setView] = useState('list');
  const [detailId, setDetailId] = useState(null);
  const [form, setForm] = useState(emptyForm);
  const [configs, setConfigs] = useState([]);
  const [listBanner, setListBanner] = useState(null);
  const [formError, setFormError] = useState('');
  const [loading, setLoading] = useState(false);
  const [uploading, setUploading] = useState(false);
  const bannerTimer = useRef(null);

  const fetchConfigs = useCallback(async () => {
    try {
      const res = await axios.get(`${API_URL}/config`);
      const data = res.data;
      setConfigs(Array.isArray(data) ? data : []);
    } catch (err) {
      setListBanner({ type: 'error', text: err.response?.data?.error || err.message });
    }
  }, []);

  useEffect(() => {
    fetchConfigs();
  }, [fetchConfigs]);

  useEffect(() => {
    if (!listBanner || listBanner.type !== 'success') return undefined;
    if (bannerTimer.current) clearTimeout(bannerTimer.current);
    bannerTimer.current = setTimeout(() => setListBanner(null), 4500);
    return () => {
      if (bannerTimer.current) clearTimeout(bannerTimer.current);
    };
  }, [listBanner]);

  const openCreate = () => {
    setDetailId(null);
    setForm(emptyForm());
    setFormError('');
    setView('form');
  };

  const openView = (c) => {
    setDetailId(c.id);
    setView('detail');
  };

  const backToList = () => {
    setView('list');
    setDetailId(null);
    setForm(emptyForm());
    setFormError('');
  };

  const classesValidation = validateClasses(form.classes);
  const modelUrlError = !form.modelUrl.trim() ? 'Model URL is required' : null;
  const modelNameError = !form.ovmsModelName.trim() ? 'OVMS model name is required' : null;
  const videoError = !form.video.trim() ? 'Video source is required' : null;
  const classesError = !form.classes.trim()
    ? 'Classes cannot be empty'
    : (classesValidation.error ?? null);

  const allFieldsFilled =
    form.modelUrl.trim() &&
    form.ovmsModelName.trim() &&
    form.video.trim() &&
    form.classes.trim();
  const allFieldsValid = allFieldsFilled && classesValidation.valid;

  const handleSave = async () => {
    if (!allFieldsValid) return;
    setFormError('');
    setLoading(true);
    try {
      const classesObj = JSON.parse(form.classes.trim());
      const body = {
        model_url: form.modelUrl.trim(),
        model_name: form.ovmsModelName.trim(),
        video_source: form.video.trim(),
        classes: classesObj,
      };
      await axios.post(`${API_URL}/config`, body);
      setListBanner({ type: 'success', text: 'Video source added.' });
      await fetchConfigs();
      backToList();
    } catch (err) {
      setFormError(err.response?.data?.error || err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleDelete = async (id) => {
    if (
      !window.confirm(
        'Delete this video source? All detection classes, tracks, and observations for it will be removed. This cannot be undone.',
      )
    ) {
      return;
    }
    try {
      await axios.delete(`${API_URL}/config/${id}`);
      await fetchConfigs();
      setListBanner({ type: 'success', text: 'Video source deleted.' });
      if (detailId === id) {
        setDetailId(null);
        setView('list');
      }
    } catch (err) {
      setListBanner({ type: 'error', text: err.response?.data?.error || err.message });
    }
  };

  const handleUpload = async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setUploading(true);
    setFormError('');
    try {
      const formData = new FormData();
      formData.append('file', file);
      const res = await axios.post(`${API_URL}/config/upload`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      const path = res.data.path || res.data.filename || file.name;
      setForm((prev) => ({ ...prev, video: path }));
    } catch (err) {
      setFormError(err.response?.data?.error || err.message);
    } finally {
      setUploading(false);
      e.target.value = '';
    }
  };

  const viewedSource = detailId != null ? configs.find((c) => c.id === detailId) : null;
  const detailCreatedAt = viewedSource ? formatCreatedAt(viewedSource.created_at) : null;
  const classesJson =
    viewedSource?.classes != null
      ? JSON.stringify(viewedSource.classes, null, 2)
      : '';

  return (
    <div className="config-page">
      <LogoBar />
      <div className="config-page-inner">
        <header className="config-page-top">
          <div>
            <h1 className="config-page-title">Video sources</h1>
            <p className="config-page-subtitle">
              Manage OVMS-backed streams and files. Add new sources or remove ones you no longer
              need.
            </p>
          </div>
          <Link to="/" className="config-page-back">
            ← Monitoring
          </Link>
        </header>

        {view === 'list' && listBanner && (
          <div
            className={`config-page-banner config-page-banner--${listBanner.type}`}
            role={listBanner.type === 'error' ? 'alert' : 'status'}
          >
            {listBanner.text}
          </div>
        )}

        {view === 'list' && (
          <>
            <div className="cfg-list-toolbar">
              <p className="cfg-list-meta">
                <strong>{configs.length}</strong>{' '}
                {configs.length === 1 ? 'source' : 'sources'} configured
              </p>
              <button type="button" className="cfg-btn-primary" onClick={openCreate}>
                + Add video source
              </button>
            </div>

            {configs.length === 0 ? (
              <div className="cfg-empty">
                <h2 className="cfg-empty-title">No video sources yet</h2>
                <p className="cfg-empty-text">
                  Create your first source with an OVMS model URL, model name, video path or RTSP
                  URL, and class definitions.
                </p>
                <button type="button" className="cfg-btn-primary" onClick={openCreate}>
                  + Add video source
                </button>
              </div>
            ) : (
              <div className="cfg-card-grid">
                {configs.map((c) => {
                  const created = formatCreatedAt(c.created_at);
                  const nClasses = classCount(c.classes);
                  return (
                    <article key={c.id} className="cfg-source-card">
                      <div className="cfg-source-card__head">
                        <div>
                          <div className="cfg-source-card__id">Source #{c.id}</div>
                          <h2 className="cfg-source-card__title">
                            {(c.model_name || '').trim() || 'Unnamed model'}
                          </h2>
                        </div>
                      </div>
                      <p className="cfg-source-card__video" title={c.video_source}>
                        {videoDisplayLabel(c.video_source)}
                      </p>
                      <p className="cfg-source-card__model" title={c.model_url}>
                        {c.model_url || '—'}
                      </p>
                      <div className="cfg-chip-row">
                        <span className="cfg-chip">{nClasses} classes</span>
                        {created ? (
                          <span className="cfg-chip cfg-chip--muted">{created}</span>
                        ) : null}
                      </div>
                      <div className="cfg-source-card__actions">
                        <button type="button" className="cfg-btn-view" onClick={() => openView(c)}>
                          View
                        </button>
                        <button type="button" className="cfg-btn-danger" onClick={() => handleDelete(c.id)}>
                          Delete
                        </button>
                      </div>
                    </article>
                  );
                })}
              </div>
            )}
          </>
        )}

        {view === 'detail' && (
          <div className="cfg-editor cfg-editor--readonly">
            <div className="cfg-editor__bar">
              <div>
                <button type="button" className="cfg-btn-text" onClick={backToList}>
                  ← Back to list
                </button>
                <h2 className="cfg-editor__title">
                  {viewedSource
                    ? `Source #${viewedSource.id} · ${(viewedSource.model_name || '').trim() || 'Unnamed model'}`
                    : 'Source'}
                </h2>
                <p className="cfg-editor__subtitle">
                  Read-only details. To change this source, delete it and add a new one.
                </p>
              </div>
            </div>

            <div className="cfg-form-body cfg-form-body--readonly">
              {!viewedSource ? (
                <p className="cfg-detail-missing">
                  This source is no longer in the list.{' '}
                  <button type="button" className="cfg-btn-text cfg-btn-text--inline" onClick={backToList}>
                    Back to list
                  </button>
                </p>
              ) : (
                <>
                  <div className="cfg-field cfg-field--readonly">
                    <span className="cfg-readonly-label">Model URL (OVMS endpoint)</span>
                    <div className="cfg-readonly-value" tabIndex={0}>
                      {viewedSource.model_url || '—'}
                    </div>
                  </div>
                  <div className="cfg-field cfg-field--readonly">
                    <span className="cfg-readonly-label">OVMS model name</span>
                    <div className="cfg-readonly-value" tabIndex={0}>
                      {(viewedSource.model_name || '').trim() || '—'}
                    </div>
                  </div>
                  <div className="cfg-field cfg-field--readonly">
                    <span className="cfg-readonly-label">Video source</span>
                    <div className="cfg-readonly-value cfg-readonly-value--multiline" tabIndex={0}>
                      {viewedSource.video_source || '—'}
                    </div>
                  </div>
                  {detailCreatedAt ? (
                    <div className="cfg-field cfg-field--readonly">
                      <span className="cfg-readonly-label">Created</span>
                      <div className="cfg-readonly-value" tabIndex={0}>
                        {detailCreatedAt}
                      </div>
                    </div>
                  ) : null}
                  <div className="cfg-field cfg-field--readonly">
                    <span className="cfg-readonly-label">Classes (JSON)</span>
                    <pre className="cfg-readonly-pre" tabIndex={0}>
                      {classesJson || '{}'}
                    </pre>
                  </div>
                  <AlertPanel configId={viewedSource.id} />
                  <div className="cfg-form-actions">
                    <button type="button" className="cfg-btn-secondary" onClick={backToList}>
                      Close
                    </button>
                  </div>
                </>
              )}
            </div>
          </div>
        )}

        {view === 'form' && (
          <div className="cfg-editor">
            <div className="cfg-editor__bar">
              <div>
                <button type="button" className="cfg-btn-text" onClick={backToList}>
                  ← Back to list
                </button>
                <h2 className="cfg-editor__title">New video source</h2>
                <p className="cfg-editor__subtitle">
                  Connect an OVMS deployment to a file path, S3 object, or RTSP stream.
                </p>
              </div>
            </div>

            <div className="cfg-form-body">
              {formError ? (
                <div className="cfg-inline-error" role="alert">
                  {formError}
                </div>
              ) : null}

              <div className="cfg-field">
                <label htmlFor="cfg-model-url">Model URL (OVMS endpoint)</label>
                <input
                  id="cfg-model-url"
                  type="text"
                  value={form.modelUrl}
                  onChange={(e) => setForm((p) => ({ ...p, modelUrl: e.target.value }))}
                  placeholder="ovms:8081 or https://ovms.example.com:8080"
                  aria-invalid={!!modelUrlError}
                />
                {modelUrlError ? (
                  <div className="cfg-field-error" role="alert">
                    {modelUrlError}
                  </div>
                ) : null}
              </div>

              <div className="cfg-field">
                <label htmlFor="cfg-ovms-name">OVMS model name</label>
                <p className="cfg-hint" id="cfg-model-hint">
                  Served model id from OVMS — often <code>ppe</code>, <code>bird</code>, or{' '}
                  <code>yolov8n</code>.
                </p>
                <input
                  id="cfg-ovms-name"
                  type="text"
                  value={form.ovmsModelName}
                  onChange={(e) => setForm((p) => ({ ...p, ovmsModelName: e.target.value }))}
                  aria-invalid={!!modelNameError}
                  aria-describedby="cfg-model-hint"
                />
                {modelNameError ? (
                  <div className="cfg-field-error" role="alert">
                    {modelNameError}
                  </div>
                ) : null}
              </div>

              <div className="cfg-field">
                <label htmlFor="cfg-video">Video source</label>
                <div className="cfg-video-row">
                  <input
                    id="cfg-video"
                    type="text"
                    value={form.video}
                    onChange={(e) => setForm((p) => ({ ...p, video: e.target.value }))}
                    placeholder="Path, s3://…/file.mp4, or rtsp://…"
                    aria-invalid={!!videoError}
                  />
                  <label className="cfg-upload-btn">
                    <input type="file" accept=".mp4,.avi,.mov,.mkv" onChange={handleUpload} hidden />
                    {uploading ? '…' : 'Upload'}
                  </label>
                </div>
                {videoError ? (
                  <div className="cfg-field-error" role="alert">
                    {videoError}
                  </div>
                ) : null}
              </div>

              <div className="cfg-field">
                <label htmlFor="cfg-classes">Classes (JSON)</label>
                <p className="cfg-hint">
                  Example:{' '}
                  <code>
                    {`{"0":{"name":"Person","trackable":true},"1":{"name":"Hardhat","trackable":false}}`}
                  </code>
                </p>
                <textarea
                  id="cfg-classes"
                  value={form.classes}
                  onChange={(e) => setForm((p) => ({ ...p, classes: e.target.value }))}
                  placeholder='{"0":{"name":"Person","trackable":true}, ...}'
                  aria-invalid={!!classesError}
                />
                {classesError ? (
                  <div className="cfg-field-error" role="alert">
                    {classesError}
                  </div>
                ) : null}
              </div>

              <div className="cfg-form-actions">
                <button
                  type="button"
                  className="cfg-btn-primary"
                  onClick={handleSave}
                  disabled={!allFieldsValid || loading}
                  title={!allFieldsValid ? 'Fill all required fields with valid values' : undefined}
                >
                  {loading ? 'Saving…' : 'Create source'}
                </button>
                <button type="button" className="cfg-btn-secondary" onClick={backToList}>
                  Cancel
                </button>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default ConfigPage;
