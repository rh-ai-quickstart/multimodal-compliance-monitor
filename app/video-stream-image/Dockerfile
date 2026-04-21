# Video stream publisher: downloads MP4 from MinIO and streams via RTSP to MediaMTX
FROM docker.io/minio/mc@sha256:a7fe349ef4bd8521fb8497f55c6042871b2ae640607cf99d9bede5e9bdf11727 AS mc

FROM docker.io/library/alpine@sha256:6baf43584bcb78f2e5847d1de515f23499913ac9f12bdf834811a3145eb11ca1

RUN apk add --no-cache ffmpeg netcat-openbsd

COPY --from=mc /usr/bin/mc /usr/bin/mc

WORKDIR /stream

COPY video-stream-image/stream-from-minio.sh /stream/stream-from-minio.sh
RUN chmod +x /stream/stream-from-minio.sh

ENTRYPOINT ["/stream/stream-from-minio.sh"]
