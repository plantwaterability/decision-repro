"""Round C (ctl2, CPU): BLINK multi-view + visual-correspondence rows from Hypersim (Apple, CC BY-SA 3.0, synthetic indoor).

download: per scene, remote-zip range reads of only camera keyframes, scene metadata, and every 5th frame's preview JPG +
          world-space position HDF5 (cam_00).
build:    hs_multiview  - BLINK Multi-view_Reasoning text; first/last frames of an orbit-like segment (look-at point held),
                          label = camera lateral motion (left/right) in the first frame's camera coordinates;
          hs_viscorr    - BLINK Visual_Correspondence text; REF circled on frame 1, A-D circled on frame 2; the true point is
                          the frame-2 pixel whose world position is nearest the REF pixel's (accepted only within 3 cm,
                          which also rejects occluded points); distractors are >= 0.5 m away in 3D and >= 60 px apart.
usage: python vbc_hypersim.py download <n_scenes> | build
"""
import hashlib
import io
import json
import random
import sys
import zipfile
from collections import Counter
from pathlib import Path

zipfile.ZipExtFile.MIN_READ_SIZE = 2 ** 20  # as in Hypersim's own partial downloader: fewer range requests
C = Path("/data/shared/visdata_c")
RAW = C / "hypersim"
SEED = 20261111


class WebFile(io.RawIOBase):
    def __init__(self, url, client):
        self.url, self.client, self.offset = url, client, 0
        r = client.head(url, follow_redirects=True)
        self.size = int(r.headers["content-length"])

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.offset

    def seek(self, offset, whence=0):
        self.offset = offset if whence == 0 else (self.offset + offset if whence == 1 else self.size + offset)
        return self.offset

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.offset
        n = min(n, self.size - self.offset)
        if n <= 0:
            return b""
        r = self.client.get(self.url, headers={"Range": f"bytes={self.offset}-{self.offset + n - 1}"})
        self.offset += len(r.content)
        return r.content


def download(n_scenes):
    import httpx

    urls = [u.strip() for u in open(Path(__file__).with_name("hypersim_urls.txt")) if u.strip()]
    pick = urls[:: max(1, len(urls) // n_scenes)][:n_scenes]
    frames = {f"frame.{i:04d}." for i in range(0, 100, 5)}
    client = httpx.Client(timeout=120)
    for u in pick:
        scene = u.rsplit("/", 1)[-1][:-4]
        done = RAW / scene / ".done"
        if done.exists():
            continue
        try:
            z = zipfile.ZipFile(WebFile(u, client))
            n = 0
            for e in z.infolist():
                f = e.filename
                want = (f.endswith(("_detail/metadata_scene.csv", "cam_00/camera_keyframe_positions.hdf5", "cam_00/camera_keyframe_orientations.hdf5",
                                    "cam_00/camera_keyframe_look_at_positions.hdf5"))
                        or ("scene_cam_00_final_preview/" in f and f.endswith(".color.jpg") and any(fr in f for fr in frames))
                        or ("scene_cam_00_geometry_hdf5/" in f and f.endswith(".position.hdf5") and any(fr in f for fr in frames)))
                if want and not e.is_dir():
                    z.extract(e, RAW)
                    n += 1
            done.parent.mkdir(parents=True, exist_ok=True)
            done.write_text(str(n))
            print(json.dumps({"scene": scene, "files": n}), flush=True)
        except Exception as ex:
            print(json.dumps({"scene": scene, "error": str(ex)[:200]}), flush=True)


def build():
    import h5py
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    sys.path.insert(0, "/data/local/visdata/scripts")
    import vb_build as B

    B.IMG = C / "img_c"
    B.ITEMS = C / "items_c"
    B.ITEMS.mkdir(parents=True, exist_ok=True)
    B.LIC.update({
        "hs_multiview": ("BLINK", "clean", "Hypersim (Apple; CC BY-SA 3.0; synthetic indoor renders); camera-motion labels from camera keyframes"),
        "hs_viscorr": ("BLINK", "clean", "Hypersim (Apple; CC BY-SA 3.0); correspondences from per-pixel world positions"),
    })
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
    MV_Q = ("The images are frames from a video. The video is shooting a static scene. The camera is either moving clockwise (left) or "
            "counter-clockwise (right) around the object. The first image is from the beginning of the video and the second image is from "
            "the end. Is the camera moving left or right when shooting the video?")
    VC_Q = ("A point is circled on the first image, labeled with REF. We change the camera position or lighting and shoot the second image. "
            "You are given multiple red-circled points on the second image, choices of \"A, B, C, D\" are drawn beside each circle. Which point "
            "on the second image corresponds to the point in the first image?")
    rng = random.Random(SEED)
    sv_mv, sv_vc = B.Saver("hs_multiview"), B.Saver("hs_viscorr")
    out_mv, out_vc = [], []

    def circle(im, pts, labels):
        im = im.copy()
        d = ImageDraw.Draw(im)
        for (x, y), lab in zip(pts, labels):
            d.ellipse([x - 12, y - 12, x + 12, y + 12], outline=(255, 0, 0), width=4)
            d.text((x + 15, y - 14), lab, fill=(255, 0, 0), font=font, stroke_width=2, stroke_fill=(255, 255, 255))
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=92)
        return buf.getvalue()

    for sd in sorted(p for p in RAW.glob("ai_*") if (p / ".done").exists()):
        det = sd / "_detail" / "cam_00"
        try:
            P = h5py.File(det / "camera_keyframe_positions.hdf5")["dataset"][:]
            R = h5py.File(det / "camera_keyframe_orientations.hdf5")["dataset"][:]
            LA = h5py.File(det / "camera_keyframe_look_at_positions.hdf5")["dataset"][:]
            mpu = 1.0
            for line in open(sd / "_detail" / "metadata_scene.csv"):
                if line.startswith("meters_per_asset_unit"):
                    mpu = float(line.strip().split(",")[1])
        except Exception:
            continue
        fdir = sd / "images" / "scene_cam_00_final_preview"
        gdir = sd / "images" / "scene_cam_00_geometry_hdf5"
        avail = sorted(int(p.name.split(".")[1]) for p in fdir.glob("frame.*.color.jpg"))
        # multi-view: pairs 10-50 frames apart, orbit-like (look-at moves < 35% of the camera's travel), clear lateral motion
        n_mv = 0
        for i in avail:
            for j in avail:
                if not 10 <= j - i <= 50 or n_mv >= 30:
                    continue
                d = R[i].T @ (P[j] - P[i])  # camera-i coordinates: x right, y up, -z forward
                travel = np.linalg.norm(P[j] - P[i])
                if travel * mpu < 0.4 or abs(d[0]) < 0.5 * travel or np.linalg.norm(LA[j] - LA[i]) > 0.35 * travel:
                    continue
                lab = "right" if d[0] > 0 else "left"
                sid = f"{sd.name}:{i}:{j}"
                imgs = [sv_mv.save((fdir / f"frame.{k:04d}.color.jpg").read_bytes(), f"{sd.name}_{k:04d}") for k in (i, j)]
                out_mv.append(B.item("hs_multiview", sid, imgs, MV_Q, ["left", "right"], "A" if lab == "left" else "B", "Multi-view_Reasoning", sv_mv))
                n_mv += 1
        # visual correspondence: frames 5-20 apart
        n_vc = 0
        for i in avail:
            if n_vc >= 20:
                break
            js = [j for j in avail if 5 <= abs(j - i) <= 20]
            if not js:
                continue
            j = rng.choice(js)
            try:
                Pi = h5py.File(gdir / f"frame.{i:04d}.position.hdf5")["dataset"][:].astype(np.float32)
                Pj = h5py.File(gdir / f"frame.{j:04d}.position.hdf5")["dataset"][:].astype(np.float32)
                Ii = Image.open(fdir / f"frame.{i:04d}.color.jpg").convert("RGB")
                Ij = Image.open(fdir / f"frame.{j:04d}.color.jpg").convert("RGB")
            except Exception:
                continue
            H, W = Pi.shape[:2]
            gi = np.asarray(Ii.convert("L"), dtype=np.float32)
            grad = np.abs(np.diff(gi, axis=1, prepend=gi[:, :1])) + np.abs(np.diff(gi, axis=0, prepend=gi[:1, :]))
            flatj = Pj[::2, ::2].reshape(-1, 3)
            okj = np.isfinite(flatj).all(1)
            for _ in range(40):
                y, x = rng.randrange(40, H - 40), rng.randrange(40, W - 40)
                if grad[y, x] < 12 or not np.isfinite(Pi[y, x]).all():
                    continue
                dist = np.linalg.norm(flatj - Pi[y, x], axis=1)
                dist[~okj] = np.inf
                k = int(dist.argmin())
                if dist[k] * mpu > 0.03:
                    continue
                ty, tx = (k // (W // 2)) * 2, (k % (W // 2)) * 2
                if not (30 <= tx < W - 30 and 30 <= ty < H - 30):
                    continue
                far = np.where(np.isfinite(dist) & (dist * mpu > 0.5))[0]
                cands = []
                for kk in rng.sample(list(far), min(400, len(far))):
                    cy, cx = (kk // (W // 2)) * 2, (kk % (W // 2)) * 2
                    if 30 <= cx < W - 30 and 30 <= cy < H - 30 and all((cx - a) ** 2 + (cy - b) ** 2 > 3600 for a, b in cands + [(tx, ty)]):
                        cands.append((cx, cy))
                    if len(cands) == 3:
                        break
                if len(cands) < 3:
                    continue
                pts = cands + [(tx, ty)]
                rng.shuffle(pts)
                gold = "ABCD"[pts.index((tx, ty))]
                sid = f"{sd.name}:{i}:{j}:{y}:{x}"
                r1 = sv_vc.save(circle(Ii, [(x, y)], ["REF"]), f"{sd.name}_{i:04d}_{y}_{x}_ref")
                r2 = sv_vc.save(circle(Ij, pts, list("ABCD")), f"{sd.name}_{j:04d}_{y}_{x}_cand")
                out_vc.append(B.item("hs_viscorr", sid, [r1, r2], VC_Q, ["Point A", "Point B", "Point C", "Point D"], gold, "Visual_Correspondence", sv_vc))
                n_vc += 1
                break
    for fam, items in (("hs_multiview", out_mv), ("hs_viscorr", out_vc)):
        with open(B.ITEMS / f"{fam}.jsonl", "w") as f:
            for it in items:
                f.write(json.dumps(it) + "\n")
        print(json.dumps({"family": fam, "items": len(items), "gold": dict(Counter(it["gold"] for it in items))}), flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "download":
        download(int(sys.argv[2]))
    else:
        build()
