"""SALBP 벤치마크(Scholl 1993)를 원 사이트에서 내려받아 data/salbp/ 에 푼다 — 저장소에는 넣지 않는다."""
import io
import urllib.request
import zipfile
from pathlib import Path

URL = "https://assembly-line-balancing.de/wp-content/uploads/2017/01/SALBP-data-sets.zip"
DEST = Path(__file__).resolve().parent.parent / "data" / "salbp"

if __name__ == "__main__":
    DEST.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(URL) as r:
        zipfile.ZipFile(io.BytesIO(r.read())).extractall(DEST)
    print("ok", sorted(p.name for p in (DEST / "precedence graphs").glob("*.IN2"))[:3], "...")
