# Local CPU detection model

The `opencv_ssd` backend supports the original Caffe MobileNet-SSD model from
[chuanqi305/MobileNet-SSD](https://github.com/chuanqi305/MobileNet-SSD), pinned to
commit `bb17b6c3eef36d80be441ae8e5339be66e8e3b7a`. It runs locally through OpenCV's
DNN CPU backend. OpenCV 4.6 provides the Caffe loader and CPU target used here;
the physical Pi still needs the bench checks in [phase2.md](phase2.md).

The author's [demo](https://github.com/chuanqi305/MobileNet-SSD/blob/bb17b6c3eef36d80be441ae8e5339be66e8e3b7a/demo.py)
defines the 300×300 BGR preprocessing and VOC class ordering. This model has 20
object classes plus background: aeroplane, bicycle, bird, boat, bottle, bus, car,
cat, chair, cow, dining table, dog, horse, motorbike, person, potted plant, sheep,
sofa, train, and TV monitor. It does **not** detect doors or signs, estimate
distance, or establish a safe route. Empty detections do not mean a clear path.

## Explicit download

The project owner approved this model's download and local testing. These
commands contact GitHub only to retrieve the assets; inference uploads no images.
Run them from the repository root on each machine that needs the model. The app
never downloads anything automatically, and `models/` is excluded from Git.

```bash
mkdir -p models/mobilenet-ssd
curl --fail --location --output models/mobilenet-ssd/deploy.prototxt https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/bb17b6c3eef36d80be441ae8e5339be66e8e3b7a/deploy.prototxt
curl --fail --location --output models/mobilenet-ssd/mobilenet_iter_73000.caffemodel https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/bb17b6c3eef36d80be441ae8e5339be66e8e3b7a/mobilenet_iter_73000.caffemodel
curl --fail --location --output models/mobilenet-ssd/LICENSE https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/bb17b6c3eef36d80be441ae8e5339be66e8e3b7a/LICENSE
```

The weights are approximately 23 MB. Preserve the upstream MIT license alongside
the files. Verify the following SHA-256 hashes using `sha256sum` on Pi or
`shasum -a 256` on macOS before loading them:

| File | SHA-256 |
| --- | --- |
| `deploy.prototxt` | `2d180f723b3109e21f8287f6b3c691390d07b60eed998327cd3259ffa0e50608` |
| `mobilenet_iter_73000.caffemodel` | `52eed8be80522c152a17fb56740de705b79881bde1a167e0e747310523685fc7` |

```bash
sha256sum models/mobilenet-ssd/deploy.prototxt models/mobilenet-ssd/mobilenet_iter_73000.caffemodel
```

## Local smoke check

After installing vidi into the selected environment:

```bash
python -m vidi --source sample --detector opencv_ssd --model models/mobilenet-ssd/mobilenet_iter_73000.caffemodel --model-config models/mobilenet-ssd/deploy.prototxt --report artifacts/cpu-smoke.json
```

This runs the real model on a synthetic frame. Success checks model loading and
tensor decoding; it does not measure recognition accuracy. For a real, local
photo, replace `--source sample` with `--image /path/to/photo.jpg` and optionally
add `--output artifacts/detections.png` to inspect the boxes.

On the Pi, `python -m vidi --config config/pi.yaml` selects Picamera2 camera 0
and the same CPU model. `fps: 2.0` is a requested run rate, not a measured speed.
The separate `config/pi-imx500.yaml` is only for a future verified IMX500 sensor.

Replacing this model requires a matching architecture, preprocessing, output
decoder, and labels. Changing only the filename does not make an arbitrary model
compatible. Optional label overrides must preserve all 21 class positions.
