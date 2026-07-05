# Garbage Vision

A small Python camera-checking service for local development on macOS and Docker-based production on a Linux host such as a Proxmox mini PC.

The current detector is a simple frame-to-frame motion/change baseline. It proves the full pipeline: read image/video/camera input, log results, save detections, and optionally send a webhook. You can later replace `MotionBaselineDetector` with a model-based detector without changing the runtime layout.

## Project Layout

```text
src/garbage_vision/     Python app source
samples/images/         Local test images, mounted read-only in Docker
data/detections/        Saved detection frames
.env.example            Safe configuration template
docker-compose.yml      Local Docker test and production service
Dockerfile              Linux container build
pyproject.toml          uv-managed Python project
```

## macOS Development Setup

Install `uv` if needed:

```bash
brew install uv
```

Create your local config:

```bash
cp .env.example .env
uv sync
```

Run test mode on sample images:

```bash
uv run garbage-vision test --source images
```

Add `.jpg`, `.png`, or `.webp` files to `samples/images/`. Test mode forces dry-run notifications, so it will log notification payloads instead of sending real alerts.

## Test Inputs

### Local image folder

Set in `.env`:

```bash
IMAGE_DIR=samples/images
DRY_RUN_NOTIFICATIONS=true
```

Run:

```bash
uv run garbage-vision test --source images
```

### Saved video file

Set in `.env`:

```bash
VIDEO_FILE=samples/video/test.mp4
DRY_RUN_NOTIFICATIONS=true
```

Run:

```bash
uv run garbage-vision test --source video
```

### Real Reolink camera

Snapshot mode is usually simplest:

```bash
CAMERA_SOURCE=snapshot
CAMERA_SNAPSHOT_URL=https://CAMERA_IP/cgi-bin/api.cgi?cmd=Snap&channel=0&rs=garbagevision
CAMERA_USERNAME=your-user
CAMERA_PASSWORD=your-password
CAMERA_VERIFY_TLS=false
```

RTSP mode is also supported:

```bash
CAMERA_SOURCE=rtsp
CAMERA_RTSP_URL=rtsp://CAMERA_IP:554/h264Preview_01_main
CAMERA_USERNAME=your-user
CAMERA_PASSWORD=your-password
```

Run a single camera check without real notifications:

```bash
uv run garbage-vision test --source camera
```

If trash is already visible when the app starts, the first frame cannot be used as a clean baseline. Clear the camera view first, then capture a clean reference:

```bash
uv run garbage-vision --capture-baseline
```

After that, camera tests and production mode compare against `BASELINE_IMAGE` instead of treating the first frame as clean:

```bash
uv run garbage-vision test --source camera
```

Preview the checked area:

```bash
uv run garbage-vision --preview-roi
```

This writes `data/previews/roi_preview.jpg` and `data/previews/roi_crop.jpg`.

## Optional YOLO Verification

The app can run YOLO only after the ROI change detector fires. This helps separate “something changed on the desk” from “a trash-like object is present.”

`OBJECT_MODEL` is the primary model. It decides whether the change is accepted as trash/dishes by matching `OBJECT_CLASSES`.

`OBJECT_MODEL_2` is optional. Use it for a general COCO YOLO model such as `models/yolov8m.pt`; its boxes are used as context for overlap marking, but it does not accept a detection by itself.

Install the optional dependency:

```bash
uv sync --extra yolo
```

Then enable it in `.env`:

```bash
OBJECT_DETECTION_ENABLED=true
OBJECT_VERIFY_REQUIRED=true
OBJECT_MODEL=models/desk-trash-v1.pt
OBJECT_MODEL_2=models/yolov8m.pt
OBJECT_CONFIDENCE=0.25
OBJECT_IMAGE_SIZE=1280
OBJECT_CLASSES=afval,vaat
TRASH_CLASSES=afval,bottle
DISH_CLASSES=vaat,plate,plates,bowl,dish,dishes
DISH_CUP_CLASSES=cup,mug,wine glass
DISH_CUP_THRESHOLD=2
NOTIFICATION_SUPPRESS_CLASSES=person
```

With `OBJECT_VERIFY_REQUIRED=false`, YOLO findings are logged and drawn on saved detection images, but ROI changes can still notify. With `OBJECT_VERIFY_REQUIRED=true`, primary-model matches from `OBJECT_CLASSES` allow a notification. Trash evidence from either model also allows a notification: any class from `TRASH_CLASSES` counts as `afval`, so a COCO `bottle` finding is shown on `latest_roi_marked.jpg`. Dish evidence from either model also allows a notification: any class from `DISH_CLASSES`, or at least `DISH_CUP_THRESHOLD` findings from `DISH_CUP_CLASSES`, counts as `vaat`. Standard COCO YOLO models do not have a literal `trash` or `plate` class, but they can usually detect `bottle`, `cup`, `bowl`, and `wine glass`; keep the custom `afval`/`vaat` model as the primary model. If any class from `NOTIFICATION_SUPPRESS_CLASSES` is seen by either YOLO model, the image is saved but no notification is sent.

For Docker, set `INSTALL_YOLO: "true"` in `docker-compose.yml` before building.

YOLO runs on the ROI crop only, not the full camera frame. For small objects, increase `OBJECT_IMAGE_SIZE`, for example `1280` or `1536`, and test with a lower `OBJECT_CONFIDENCE` such as `0.20` before tightening it again.

Detection evidence overwrites the latest files instead of accumulating a list:

```text
data/detections/latest.jpg
data/detections/latest_roi_marked.jpg
```

## Custom Classes

To detect your own classes like `afval` and `vaat`, train a custom YOLO model. Training and labeling files live in a separate sibling project next to this repo, for example `../Garbage-training`, so they do not get deployed to the Linux machine. The dataset config is:

```text
../Garbage-training/local-training/training/datasets/desk-trash/desk-trash.yaml
```

Dataset layout:

```text
../Garbage-training/local-training/training/datasets/desk-trash/
  images/unlabeled/
  images/train/
  images/val/
  labels/train/
  labels/val/
```

Collect ROI crop images for annotation:

```bash
uv run python ../Garbage-training/local-training/scripts/collect_training_crops.py
```

Useful variants:

```bash
uv run python ../Garbage-training/local-training/scripts/collect_training_crops.py --interval 5
uv run python ../Garbage-training/local-training/scripts/collect_training_crops.py --max-images 20
uv run python ../Garbage-training/local-training/scripts/collect_training_crops.py --only-on-change
```

This saves cropped camera images to:

```text
../Garbage-training/local-training/training/datasets/desk-trash/images/unlabeled/
```

Import those images into Label Studio, draw boxes for `afval` and `vaat`, then export YOLO labels. After annotation, move roughly 80% of images and labels to `images/train` and `labels/train`, and 20% to `images/val` and `labels/val`.

Class IDs:

```text
0 = afval
1 = vaat
```

Each image needs a matching label file in YOLO format:

```text
class x_center y_center width height
```

Coordinates are normalized from `0` to `1`.

Train from a pretrained model:

```bash
uv run yolo detect train model=../Garbage-training/yolov8m.pt data=../Garbage-training/local-training/training/datasets/desk-trash/desk-trash.yaml imgsz=1280 epochs=100 batch=2 workers=0 project=../Garbage-training/runs/desk-trash name=desk-trash-v1
```

After training, copy the best weights into the runtime model folder and point the app there:

```bash
cp ../Garbage-training/runs/desk-trash/desk-trash-v1/weights/best.pt models/desk-trash-v1.pt
```

```bash
OBJECT_MODEL=models/desk-trash-v1.pt
OBJECT_MODEL_2=models/yolov8m.pt
OBJECT_CLASSES=afval,vaat
TRASH_CLASSES=afval,bottle
DISH_CLASSES=vaat,plate,plates,bowl,dish,dishes
DISH_CUP_CLASSES=cup,mug,wine glass
DISH_CUP_THRESHOLD=2
OBJECT_IMAGE_SIZE=1280
OBJECT_CONFIDENCE=0.25
```

For best small-object precision, label tightly, use many examples from your actual camera angle, include negative examples where tools/staplers/mugs changed but are not trash, and keep using ROI crops so the object occupies more pixels.

## Docker Local Testing

Build the image:

```bash
docker compose build
```

Run image-folder test mode:

```bash
docker compose --profile test run --rm garbage-vision-test
```

Run a camera test in Docker:

```bash
docker compose --profile test run --rm garbage-vision-test test --source camera
```

For a saved video file, place it under the project, for example `samples/video/test.mp4`, then set:

```bash
VIDEO_FILE=samples/video/test.mp4
```

## Production Mode

Production mode checks the configured camera periodically:

```bash
APP_MODE=prod
POLL_SECONDS=60
DRY_RUN_NOTIFICATIONS=false
NOTIFY_ENABLED=true
NOTIFY_PROVIDER=ntfy
NTFY_SERVER=https://ntfy.sh
NTFY_TOPIC=your-private-topic-name
```

For ntfy, install the ntfy app on your phone and subscribe to the same private topic. ntfy supports simple HTTP publishing to a topic and image attachments using the `Filename` header.

For Pushover instead, set:

```bash
NOTIFY_PROVIDER=pushover
PUSHOVER_APP_TOKEN=your-app-token
PUSHOVER_USER_KEY=your-user-key
```

For a generic JSON webhook instead, set:

```bash
NOTIFY_PROVIDER=webhook
WEBHOOK_URL=https://example.com/your-webhook
```

Run locally with `uv`:

```bash
uv run garbage-vision prod
```

Run with Docker Compose:

```bash
docker compose up -d garbage-vision
docker compose logs -f garbage-vision
```

Detection images are saved under `data/detections/`.

## Tailscale Image Sharing

The `garbage-vision-images` sidecar serves `data/detections/` with nginx, but
only on localhost:

```yaml
ports:
  - "127.0.0.1:8765:80"
```

That means the image endpoint is not open on the LAN. Expose it over your
Tailnet with Tailscale Serve on `hadrian`:

```bash
cd /home/art/garbage-vision
docker compose up -d garbage-vision-images
tailscale serve --bg --https=8765 http://127.0.0.1:8765
```

Then the latest marked detection image is available inside your Tailnet at:

```text
https://hadrian.tail818628.ts.net:8765/latest_roi_marked.jpg
```

Useful checks:

```bash
curl -I http://127.0.0.1:8765/latest_roi_marked.jpg
tailscale serve status
```

Home Assistant can show this image with a generic camera:

```yaml
camera:
  - platform: generic
    name: Garbage Vision Latest
    still_image_url: https://hadrian.tail818628.ts.net:8765/latest_roi_marked.jpg
    verify_ssl: true
```

Because Tailscale Serve provides the HTTPS Tailnet endpoint, do not publish
port `8765` on `0.0.0.0` or forward it from your router.

## Deploy to the Home-Lab Linux VM

The target host from `ArtNooijen/home-lab` is the Ubuntu `linux-docker-host`
VM on Proxmox. This app is best deployed as its own source checkout on that VM,
next to the `home-lab` checkout, because the container image is built from this
repo and the private `.env`, `data/`, and `models/` files should stay out of Git.

1. Build and test on macOS before shipping:

```bash
uv sync
uv run garbage-vision test --source images
uv run garbage-vision test --source camera
docker compose build
docker compose --profile test run --rm garbage-vision-test
```

2. Commit and push this repo:

```bash
git add .
git commit -m "Add garbage vision service"
git push
```

3. SSH into the Linux Docker host:

```bash
ssh art@hadrian
cd /home/art
```

4. Clone or update the app checkout:

```bash
git clone https://github.com/ArtNooijen/Garbage.git garbage-vision
cd garbage-vision
```

For later updates:

```bash
cd /home/art/garbage-vision
git pull
```

5. Create the VM-local config:

```bash
cp .env.example .env
nano .env
```

Recommended production values:

```bash
APP_MODE=prod
CAMERA_SOURCE=snapshot
CAMERA_SNAPSHOT_URL=https://192.168.2.44/cgi-bin/api.cgi?cmd=Snap&channel=0&rs=garbagevision
CAMERA_USERNAME=admin
CAMERA_PASSWORD=your-camera-password
CAMERA_VERIFY_TLS=false
POLL_SECONDS=60
BASELINE_IMAGE=data/baseline/clean.jpg
DETECTION_THRESHOLD=0.02
MIN_CHANGED_AREA=4000
DETECTION_ROI=0,760,1900,856
OBJECT_DETECTION_ENABLED=true
OBJECT_VERIFY_REQUIRED=false
OBJECT_MODEL=models/desk-trash-v1.pt
OBJECT_MODEL_2=models/yolov8m.pt
OBJECT_CLASSES=afval,vaat
TRASH_CLASSES=afval,bottle
DISH_CLASSES=vaat,plate,plates,bowl,dish,dishes
DISH_CUP_CLASSES=cup,mug,wine glass
DISH_CUP_THRESHOLD=2
NOTIFICATION_SUPPRESS_CLASSES=person
NOTIFY_ENABLED=true
DRY_RUN_NOTIFICATIONS=false
NOTIFY_PROVIDER=ntfy
NTFY_SERVER=https://ntfy.sh
NTFY_TOPIC=your-private-topic-name
```

6. Copy runtime model weights to the VM outside Git:

```bash
ssh art@hadrian 'mkdir -p /home/art/garbage-vision/models'
scp models/desk-trash-v1.pt art@hadrian:/home/art/garbage-vision/models/desk-trash-v1.pt
scp models/yolov8m.pt art@hadrian:/home/art/garbage-vision/models/yolov8m.pt
```

The `models/` directory is mounted into Docker, but `.pt` files are ignored by
Git.

7. Build the container and run a camera sanity check on the VM:

```bash
cd /home/art/garbage-vision
docker compose build garbage-vision
docker compose --profile test run --rm garbage-vision-test test --source camera
```

8. Capture a clean baseline on the VM after the camera view is clear:

```bash
docker compose run --rm garbage-vision --capture-baseline
docker compose run --rm garbage-vision --preview-roi
```

Check the generated preview files:

```text
data/baseline/clean.jpg
data/previews/roi_preview.jpg
data/previews/roi_crop.jpg
```

Then keep production running:

```bash
docker compose up -d garbage-vision
docker compose logs -f garbage-vision
```

No macOS-specific runtime dependencies are used in production. The container
uses Linux, Python 3.12, OpenCV headless, and `uv`.

### Optional Home-Lab Repo Entry

The `home-lab` repo tracks service compose files and its README, not private
runtime data. Add a lightweight note there rather than copying `.env`, model
weights, or detection images into that repo.

Add this to the `linux-docker-host` container table:

```text
| garbage-vision | local build from /home/art/garbage-vision | no published ports |
```

If you want a service folder in `home-lab`, create
`home-lab/garbage-vision/compose.yml` with a pointer to the app checkout:

```yaml
services:
  garbage-vision:
    build:
      context: /home/art/garbage-vision
      args:
        INSTALL_YOLO: "true"
    container_name: garbage-vision
    env_file:
      - path: /home/art/garbage-vision/.env
        required: false
    volumes:
      - /home/art/garbage-vision/data:/app/data
      - /home/art/garbage-vision/models:/app/models:ro
      - /home/art/garbage-vision/samples:/app/samples:ro
    restart: unless-stopped
    command: ["prod"]
```

Run that variant from the home-lab checkout with:

```bash
cd /home/art/home-lab/garbage-vision
docker compose up -d --build
```

Because `home-lab/.gitignore` allow-lists service folders, also add:

```gitignore
!/garbage-vision/
!/garbage-vision/**/
```

The root `docker-compose.yml` in this repo remains the primary deployment file.
The home-lab compose pointer is only for keeping service inventory in one place.

## Configuration

All secrets and environment-specific values belong in `.env`, which is ignored by Git.

Important values:

```bash
APP_MODE=test|prod
CAMERA_SOURCE=snapshot|rtsp
CAMERA_SNAPSHOT_URL=
CAMERA_RTSP_URL=
CAMERA_USERNAME=
CAMERA_PASSWORD=
CAMERA_VERIFY_TLS=false
POLL_SECONDS=60
BASELINE_IMAGE=data/baseline/clean.jpg
DETECTION_THRESHOLD=0.02
MIN_CHANGED_AREA=4000
DETECTION_ROI=0,760,1900,856
OBJECT_DETECTION_ENABLED=false
OBJECT_VERIFY_REQUIRED=false
OBJECT_MODEL=models/desk-trash-v1.pt
OBJECT_MODEL_2=models/yolov8m.pt
OBJECT_CONFIDENCE=0.35
OBJECT_IMAGE_SIZE=1280
OBJECT_CLASSES=afval,vaat
TRASH_CLASSES=afval,bottle
DISH_CLASSES=vaat,plate,plates,bowl,dish,dishes
DISH_CUP_CLASSES=cup,mug,wine glass
DISH_CUP_THRESHOLD=2
NOTIFICATION_SUPPRESS_CLASSES=person
NOTIFY_ENABLED=false
DRY_RUN_NOTIFICATIONS=true
WEBHOOK_URL=
```

## Troubleshooting

`uv sync` fails on macOS:
Update `uv` with `brew upgrade uv`, then run `uv sync` again.

Docker cannot reach the camera:
Check that the Proxmox host/container is on the same network/VLAN as the Reolink camera. Test with `curl` for snapshots or `ffmpeg`/VLC for RTSP from the host.

RTSP works on macOS but not in Docker:
Prefer snapshot mode first. RTSP can be blocked by network isolation, camera stream limits, or credentials with special characters. This app URL-encodes credentials when they are supplied separately as `CAMERA_USERNAME` and `CAMERA_PASSWORD`.

Snapshot URL returns an unreadable image:
Open the snapshot URL in a browser while signed out of Reolink. If the camera requires embedded credentials, keep the URL without credentials and put them in `.env`.

Snapshot request fails with a self-signed certificate:
Set `CAMERA_VERIFY_TLS=false`. Many local Reolink cameras redirect HTTP to HTTPS with a self-signed certificate.

No detections appear:
If a clean reference baseline exists, compare the logged `score` with `DETECTION_THRESHOLD`. For a tight ROI, start with `DETECTION_THRESHOLD=0.02` and `MIN_CHANGED_AREA=4000`, then tune from the logged `score` and `area`.

Movement elsewhere in the room triggers detections:
Set `DETECTION_ROI=x,y,width,height` to only watch the trash area. For the current camera view, `0,760,1900,856` covers the PC desk on the left and the main workbench area.

Notifications are not sent:
Confirm `NOTIFY_ENABLED=true` and `DRY_RUN_NOTIFICATIONS=false`. For ntfy, set `NOTIFY_PROVIDER=ntfy` and `NTFY_TOPIC`. For Pushover, set `NOTIFY_PROVIDER=pushover`, `PUSHOVER_APP_TOKEN`, and `PUSHOVER_USER_KEY`. Test mode always dry-runs notifications; use `prod` mode for real alerts.
