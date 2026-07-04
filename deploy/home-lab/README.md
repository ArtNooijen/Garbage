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
