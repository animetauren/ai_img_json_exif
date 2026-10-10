FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# exiftool reads every metadata block (EXIF, GPS, MakerNotes, XMP, IPTC, ICC, HEIC/RAW)
RUN apt-get update \
 && apt-get install -y --no-install-recommends libimage-exiftool-perl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /srv

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

# /data holds saved workflows, /videos the video uploads (volumes in compose.yaml).
# uid 1001 matches the video service, which shares /videos.
RUN useradd --system --uid 1001 app \
 && mkdir -p /data /videos && chown app /data /videos
ENV DATA_DIR=/data VIDEO_DIR=/videos
USER app

EXPOSE 8080
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
