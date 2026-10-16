# Downloads an MP4 from aws-compatible-storage and publishes it to MediaMTX.
FROM docker.io/library/alpine@sha256:6baf43584bcb78f2e5847d1de515f23499913ac9f12bdf834811a3145eb11ca1

RUN apk add --no-cache ffmpeg aws-cli netcat-openbsd && \
    mkdir -p /etc/aws && \
    AWS_CONFIG_FILE=/etc/aws/config aws configure set default.s3.addressing_style path && \
    chmod a+r /etc/aws/config

WORKDIR /stream

COPY video-stream-image/stream-from-storage.sh /stream/stream-from-storage.sh
RUN chmod a+rx /stream/stream-from-storage.sh

ENV HOME=/tmp \
    AWS_CONFIG_FILE=/etc/aws/config \
    AWS_ENDPOINT_URL=http://aws-compatible-storage:7480 \
    AWS_ACCESS_KEY_ID=s4admin \
    AWS_SECRET_ACCESS_KEY=s4secret \
    AWS_DEFAULT_REGION=us-east-1

ENTRYPOINT ["/stream/stream-from-storage.sh"]
