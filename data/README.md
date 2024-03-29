The data is located in the following address: [http://dx.doi.org/10.21227/7zbf-se57](http://dx.doi.org/10.21227/7zbf-se57).

When the data is decompressed, there should be a subfolder for each batch type (Type1, ..., Type36).

An easy way to decompress all the zip files in a Linux environment is:

```
find . -name "*.zip" -exec unzip {} \;
```

From the folder that contains all the zip files.

Abstract
----------------------------
This data set comprises 4223 videos from a laser surface heat treatment process (also called laser heat treatment) applied to cylindrical workpieces made of steel. The purpose of the dataset is to detect anomalies in the laser heat treatment learning a model from a set of non-anomalous videos.

In the laser heat treatment, the laser beam is following a pattern similar to an "eight" with a frequency of 100 Hz:

![](../demo/OrigDiagram.png)
 
 
This pattern is sometimes modified to avoid obstacles in the workpieces:

![](../demo/ModDiagram.png)

The videos are recorded at a frequency of 1000 frames per second with a thermal camera. The thermal camera pixel values range from 0 to 1024, where higher values means higher temperatures. The thermal camera has a resolution of 32x32 pixels, but a region of interest (ROI) is applied to the videos, so only the laser heat treatment is shown in the videos.

Data set structure
-------------------------
The videos are distributed in 36 batches, as the videos are obtained from 4 different workstations and comprise 9 different workpieces types.

Each batch type contains the videos in npz format (to be read by numpy) and a metadata.pkl which is a Python pickle which contains the following metadata about each video in the batch size:

- `Index`: Batch type index (1-36).

- `KeyfIniObstacle1`: Frame index where the first obstacle starts.

- `KeyfEndObstacle1`: Frame index where the first obstacle ends.

- `KeyfIniObstacle2`: Frame index where the second obstacle starts.

- `KeyfEndObstacle2`: Frame index where the second obstacle ends.

- `LeftBordPx`: Pixel index of the left border (after applying the region of interest).

- `RightBordPx`: Pixel index of the right border (after applying the region of interest). (exclusive)

- `SupBordPx`: Pixel index of the upper border (after applying the region of interest).

- `InfBordPx`: Pixel index of the bottom border (after applying the region of interest). (exclusive)

- `SizeInFrames`: Number of frames in the video.

For example, loading a metadata pickle with Python to obtain the number of frames of the video `0001`:

```python
with open('Type1/metadata.pkl', 'rb') as f:
	metadata = pickle.load(f)
	nframes = metadata['0001']['SizeInFrames']
```

To load a npz file, you should use numpy load function.

```python
array = np.load("Type1/0001.npz")['image']
```

which loads a numpy ndarray in `array`. The loaded ndarray has a shape `(nframes, vertical_size, horizontal_size)`, where:

- `nframes` is the number of frames, e.g. `metadata['0001']['SizeInFrames']`
- `vertical_size` is the number of pixels in the vertical axis. It could be calculated as: `metadata['0001']['InfBordPx'] - metadata['0001']['SupBordPx'] + 1`
- `horizontal_size` is the number of pixels in the vertical axis. It could be calculated as: `metadata['0001']['RightBordPx'] - metadata['0001']['LeftBordPx'] + 1`

Anomaly Detection
----------------------------

The objective of the dataset is to detect anomalies by learning a model from non-anomalous videos.

**The only video confirmed to be an anomaly is "1673"**. The remaining videos are considered non-anomalous, and a model of normality can be built from them.

Difference videos
----------------------------

For confidentiality reasons, raw videos captured from the thermal camera are not released. Instead, the released videos were generated by calculating the difference between frames in the raw videos. Then, the negative pixel differences were set to 0. The resulting videos are called _difference_ videos.

The _difference_ videos show the movement of the laser spot, which can be useful to detect anomalies.