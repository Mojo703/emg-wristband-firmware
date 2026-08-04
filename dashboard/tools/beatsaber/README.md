# InfernoSaber, running locally

The Beat Saber automapper the collection game falls back on for a song with no
hand-made map. `../ingest_track.py` drives it: with `--beatsaber-map` it
converts an existing map and never comes here at all, and without one it runs
the model over the audio, caches the result, and converts that.

`InfernoSaber/` (the clone) and `Data/` (the weights and working files, 1.2 GB
including every map the model has produced) are gitignored — only the setup
notes and the runner are tracked.

## Install and run

Arch, system python 3.14. InfernoSaber needs 3.10, so it gets its own venv;
`../.venv-beatsaber` is separate from `../.venv` and does not disturb it.

```bash
cd dashboard/tools/beatsaber
git clone --depth 1 -b main_app \
  https://github.com/fred-brenner/InfernoSaber---BeatSaber-Automapper.git InfernoSaber

cd ..
uv venv --python 3.10 .venv-beatsaber
grep -v '^#' beatsaber/InfernoSaber/requirements.txt | grep -v gradio > /tmp/req.txt
VIRTUAL_ENV=$PWD/.venv-beatsaber uv pip install -r /tmp/req.txt
VIRTUAL_ENV=$PWD/.venv-beatsaber uv pip install pydub
```

Dropping gradio skips the web UI, which we do not use. pydub is imported by
`bs_shift/export_map.py` but commented out of `requirements.txt`.

aubio is the awkward one. It has no wheels, and 0.4.9 will not compile against
either GCC 14 or ffmpeg 7. Two workarounds, both needed:

```bash
# 1. GCC 14 promoted -Wincompatible-pointer-types to an error.
# 2. aubio's avcodec backend uses AVCodecContext->channels, removed in ffmpeg 5.
#    Hide the libav .pc files so aubio builds against libsndfile instead;
#    sndfile reads the ogg/egg files InfernoSaber feeds it.
SHIM=$(mktemp -d)
cd /usr/lib/pkgconfig
for f in *.pc; do case "$f" in libav*|libsw*) ;; *) ln -sf /usr/lib/pkgconfig/$f $SHIM/$f;; esac; done

cd -
PKG_CONFIG_LIBDIR=$SHIM CFLAGS="-Wno-incompatible-pointer-types -Wno-int-conversion" \
  VIRTUAL_ENV=$PWD/.venv-beatsaber uv pip install --no-build-isolation-package aubio aubio
```

With that in place, stage audio and run. The first run pulls 1.2 GB of models from Hugging Face
(`BierHerr/InfernoSaber`, branch `fav_15`) into `Data/model/`, which took
about 16 s here.

```bash
mkdir -p beatsaber/Data/prediction/songs_predict
cp ../config/arcade-fire-afterlife.ogg beatsaber/Data/prediction/songs_predict/

cd beatsaber/InfernoSaber
../../.venv-beatsaber/bin/python ../run_infernosaber.py --difficulty 5
```

`--difficulty` is notes per second, 1 to 10. Output lands in
`Data/prediction/new_map/1234_<difficulty*4>_<song>/` as `info.dat`,
`Expert.dat` and `ExpertPlus.dat`, plus a zip for the online map viewer.

Note that `run_infernosaber.py` rewrites `dir_path` inside the clone's
`tools/config/paths.py`. That is how InfernoSaber stores its working
directory; it is not something we can pass in.

CPU only. tensorflow 2.15 without the CUDA extras runs a 5.5 minute song in
about 20 s on this machine, so the GPU is not worth the CUDA 12 pinning that
`tensorflow[and-cuda]==2.15` would drag in.

A produced map is ordinary Beat Saber v3 JSON, so `--beatsaber-map` converts
it the same way it converts a hand-made one:

```bash
cd ..
.venv/bin/python ingest_track.py --beatsaber-map \
  beatsaber/Data/prediction/new_map/1234_20.0_<song>
```
