# Home-Lab Compose Pointer

This compose file is for the `ArtNooijen/home-lab` service inventory repo.
It points at the real app checkout on the Linux Docker host:

```text
/home/art/garbage-vision
```

Runtime files stay in the app checkout:

```text
/home/art/garbage-vision/.env
/home/art/garbage-vision/data/
/home/art/garbage-vision/models/
```

If you add this folder to `home-lab`, update `home-lab/.gitignore` because that
repo ignores everything by default:

```gitignore
!/garbage-vision/
!/garbage-vision/**/
```

Then run from the VM:

```bash
cd /home/art/home-lab/garbage-vision
docker compose up -d --build
docker compose logs -f garbage-vision
```

The `garbage-vision-images` service serves the latest detection images on
localhost only:

```text
http://127.0.0.1:8765/latest_roi_marked.jpg
```

Expose it to Home Assistant over Tailscale Serve:

```bash
tailscale serve --bg --https=8765 http://127.0.0.1:8765
```

Tailnet URL:

```text
https://hadrian.tail818628.ts.net:8765/latest_roi_marked.jpg
```
