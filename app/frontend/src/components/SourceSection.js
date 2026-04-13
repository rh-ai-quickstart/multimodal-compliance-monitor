import React from 'react';
import { API_URL } from '../config';
import './SourceSection.css';

const isRtspUrl = (video) =>
  typeof video === 'string' && video.trim().toLowerCase().startsWith('rtsp://');

const isFileVideo = (video) => {
  if (typeof video !== 'string' || !video.trim()) return false;
  const v = video.trim().toLowerCase();
  return v.endsWith('.mp4') || v.endsWith('.avi') || v.endsWith('.mov') || v.endsWith('.mkv');
};

const getThumbnailPath = (videoPath) => {
  const name = videoPath.split(/[/\\]/).pop() || '';
  const stem = name.replace(/\.[^.]+$/, '');
  if (!stem) return null;
  const base = (API_URL || '').replace(/\/$/, '');
  return base ? `${base}/thumbnails/${stem}.jpg` : `/thumbnails/${stem}.jpg`;
};

const SourceSection = ({ configs, selectedConfigId, onSelectConfig, onAddVideo }) => {
  const rtspConfigs = configs.filter((c) => isRtspUrl(c.video_source));
  const mp4Configs = configs.filter((c) => isFileVideo(c.video_source));

  const handleMp4Click = (config) => {
    onSelectConfig(config.id);
  };

  const handleRtspChange = (e) => {
    const id = e.target.value ? Number(e.target.value) : null;
    onSelectConfig(id);
  };

  return (
    <div className="source-section">
      <h3 className="source-section-title">Source</h3>

      <div className="source-rtsp-box">
        <label htmlFor="rtsp-select" className="source-box-label">
          RTSP URL
        </label>
        <select
          id="rtsp-select"
          value={selectedConfigId && rtspConfigs.some((c) => c.id === selectedConfigId) ? selectedConfigId : ''}
          onChange={handleRtspChange}
          className="source-rtsp-select"
        >
          <option value="">Select RTSP stream...</option>
          {rtspConfigs.map((c) => (
            <option key={c.id} value={c.id}>
              {c.video_source}
            </option>
          ))}
        </select>
      </div>

      <div className="source-or-divider">OR</div>

      <div className="source-mp4-box">
        <span className="source-box-label">MP4 Videos</span>
        <div className="source-thumbnails">
          {mp4Configs.length === 0 ? (
            <div className="source-thumbnails-empty">No MP4 videos</div>
          ) : (
            mp4Configs.map((c) => {
              const thumbPath = getThumbnailPath(c.video_source);
              const isSelected = selectedConfigId === c.id;
              return (
                <button
                  key={c.id}
                  type="button"
                  className={`source-thumbnail-btn ${isSelected ? 'source-thumbnail-selected' : ''}`}
                  onClick={() => handleMp4Click(c)}
                  title={`${c.video_source} (${c.model_url})`}
                >
                  {thumbPath ? (
                    <img
                      src={thumbPath}
                      alt={c.video_source}
                      onError={(e) => {
                        e.target.style.display = 'none';
                        e.target.nextSibling?.classList.remove('source-thumbnail-fallback-hidden');
                      }}
                    />
                  ) : null}
                  <span className={thumbPath ? 'source-thumbnail-fallback source-thumbnail-fallback-hidden' : 'source-thumbnail-fallback'}>
                    {c.video_source.split(/[/\\]/).pop() || 'Video'}
                  </span>
                </button>
              );
            })
          )}
        </div>
      </div>

      <button type="button" className="source-add-video-btn" onClick={onAddVideo}>
        + Add New Video Source
      </button>
    </div>
  );
};

export default SourceSection;
