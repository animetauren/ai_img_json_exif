# LLM Image Classifier

A self-hosted web app for classifying photos with local models and writing the results into their metadata, and for splitting long videos into scenes, transcribing their speech and tagging each scene.

![The Single model tab: RT-DETRv2 has boxed two people, two bicycles and a boat, with the JSON result on the right and the photo's EXIF below](docs/screenshot.jpg)

*Photo: [Retired electrician](https://commons.wikimedia.org/wiki/File:Moscow_2025-08-10_bicycle_race,_sidewalk_02.jpg), CC0, via Wikimedia Commons.*

- **LLM backend**: any OpenAI-compatible vision model, such as llama.cpp's `llama-server`. You write the prompt, and the app shows the raw output next to the parsed JSON.
- **Vision backend**: dedicated image models in a second container. SigLIP 2 scores your own labels, EfficientNetV2 and ResNet-50 use the ImageNet classes, and RT-DETRv2 detects objects.
- **Workflows** chain several models together. **Batch** runs a whole folder. **Export** writes the results into EXIF, XMP and IPTC tags on a copy of each photo.
- **Video Classifier** (switch in the page title) splits videos of any length into scenes with TransNetV2, PySceneDetect or FFmpeg, transcribes their speech with Whisper, tags every scene with a model or workflow, and exports the scene list with each scene's thumbnail, words and JSON.

Both backends return the same top-level JSON (`category`, `tags`, `confidence`), so their results compare directly. A walkthrough on first load explains each setting; press **?** to replay it. Drag the gaps between panels and columns to resize them, and double-click a gap to reset it.

## Quick start

1. Start an OpenAI-compatible server with a vision model (see [Recommended LLM models](#recommended-llm-models)). With llama.cpp, `-hf` downloads a model and its `mmproj` file, which lets it see images:

   ```sh
   llama-server -hf Qwen/Qwen3-VL-8B-Instruct-GGUF --host 0.0.0.0 --port 8080
   ```

   For a model you already have, pass both files: `llama-server -m model.gguf --mmproj mmproj.gguf --host 0.0.0.0 --port 8080`.

2. Point the app at it and start it:

   ```sh
   echo 'LLM_BASE_URL=http://host.docker.internal:8080/v1' > .env
   docker compose up --build -d
   ```

3. Open <http://localhost:8090>.

On Linux, `host.docker.internal` needs `extra_hosts: ["host.docker.internal:host-gateway"]` in `compose.yaml`. Alternatively, use the host's IP address.

> **No authentication.** Run the app on a trusted network only. Photos are sent only to the model servers you configure.

## Recommended LLM models

Any model that accepts images works. These are good places to start: all are Apache-2.0, run in llama.cpp, and are ready to download with `-hf` (4-bit, with their `mmproj`):

| Model                | Download | Good for                                                                         | `llama-server`                         |
| -------------------- | -------- | -------------------------------------------------------------------------------- | -------------------------------------- |
| Gemma 4 E4B          | 5 GB     | Quick categories, tags and captions on a laptop                                  | `-hf ggml-org/gemma-4-E4B-it-GGUF`     |
| Qwen3-VL 8B Instruct | 6 GB     | Reading text in photos (signs, documents, screenshots); answers without thinking | `-hf Qwen/Qwen3-VL-8B-Instruct-GGUF`   |
| Qwen3.5 9B           | 7 GB     | A newer all-rounder of the same size; thinks first                               | `-hf unsloth/Qwen3.5-9B-GGUF`          |
| Gemma 4 26B A4B      | 15 GB    | Richer descriptions; a mixture of experts, so faster than its size suggests      | `-hf ggml-org/gemma-4-26B-A4B-it-GGUF` |
| Qwen3.6 35B A3B      | 21 GB    | The most capable of these, and fast for its size; thinks first                   | `-hf ggml-org/Qwen3.6-35B-A3B-GGUF`    |

- **Memory:** allow the download size plus 2–3 GB of RAM or VRAM. On Apple Silicon, llama.cpp runs on the GPU in shared memory.
- **Thinking models** reason before they answer, which takes longer and can use up **Max tokens** (1024 by default) before any JSON arrives. Start llama-server with `--reasoning off`, or raise Max tokens to 4096 or more.
- **Context:** a photo at 1536 px can take over 1,500 tokens of context in the Qwen models. If a request fails for lack of context, add `-c 8192`.
- **Ollama and LM Studio** offer the same families. Ollama's endpoint is `http://host.docker.internal:11434/v1`; type the model's name in the Model field.
- More models are listed in [llama.cpp's multimodal docs](https://github.com/ggml-org/llama.cpp/blob/master/docs/multimodal.md).

## Configuration

| Variable               | Default                                   | Purpose                                                                  |
| ---------------------- | ----------------------------------------- | ------------------------------------------------------------------------ |
| `LLM_BASE_URL`         | `http://host.containers.internal:8080/v1` | Default LLM endpoint, editable in the UI; set it as in Quick start       |
| `LLM_API_KEY`          | *(empty)*                                 | Bearer token, if your server needs one                                   |
| `LLM_TIMEOUT`          | `300`                                     | Request timeout in seconds                                               |
| `VISION_MAX_LOADED`    | `2`                                       | Vision models kept in RAM; the least recently used one is unloaded first |
| `VIDEO_DATA`           | `video-data` (a volume)                   | Where uploaded videos are kept; a host path makes it a bind mount        |
| `VIDEO_RETENTION_DAYS` | `7`                                       | Days a video is kept after it was last opened                            |

UI settings are stored in the browser. Saved workflows live in the `app-data` volume, model weights in `vision-cache` and `video-cache`, and videos in `video-data`.

## Using it

**Single model.** Drop a photo and press **Run** (Cmd/Ctrl+Enter).

- Before upload, the browser scales photos to at most 1536 px.
- The EXIF panel reads every tag from the original file, including HEIC and RAW.
- **Force JSON** makes llama.cpp return valid JSON only.
- The JSON panel ignores `<think>` blocks and code fences when it parses the output.
- Thinking models' reasoning appears in the LLM output panel ahead of the answer, whether the server sends it as `reasoning_content` (llama.cpp, LM Studio) or `reasoning` (Ollama, vLLM). If a model spends all of **Max tokens** before it answers, the app says so rather than showing an empty result: raise Max tokens, or use a non-thinking model.

**Workflows.** A workflow is an ordered list of vision and LLM steps that run on the server.

- In an LLM prompt, `{{step_id.field}}` inserts an earlier result, e.g. `{{objects.counts.person}}`, `{{scene.summary}}` or `{{exif.DateTimeOriginal}}`.
- **Run** uses your current edits, even before you save them.
- Three starter workflows are included.

**Batch.** Choose a folder and every photo in it appears in a grid.

- Borders show each photo's state: grey (not run), green (done), red (failed).
- **Run all** skips photos that are already done and resumes where it stopped.
- If a model server goes down, the batch retries, then pauses. Nothing is marked as failed.
- Results are kept in the browser, so picking the same folder again brings them back.
- Search, filter, sort and group by any result field or EXIF tag, e.g. `category:portrait people_count>2 exif.Model:"EOS R5"`.
- **Export all…** downloads a zip of tagged copies, keeping file names and subfolders. If the page closes first, the zip is offered again on the next visit.

## Metadata export

**Export image** downloads a copy of the original, full-size file with the result written into its metadata. Your original is never changed. Known JSON keys, at any depth, map to standard tags:

| JSON key                                            | Tags written                                                                  |
| --------------------------------------------------- | ----------------------------------------------------------------------------- |
| `description`, `caption`, `summary`                 | `IFD0:ImageDescription`, `XMP-dc:Description`, `IPTC:Caption-Abstract`        |
| `title` / `headline`                                | `XMP-dc:Title`, `IPTC:ObjectName` / `XMP-photoshop:Headline`, `IPTC:Headline` |
| `tags`, `keywords`, `labels`, `objects`, `category` | `XMP-dc:Subject`, `IPTC:Keywords`                                             |
| `rating` (0–5)                                      | `XMP-xmp:Rating`                                                              |
| `artist`, `creator`, `author` / `copyright`         | `IFD0:Artist`, `XMP-dc:Creator` / `IFD0:Copyright`, `XMP-dc:Rights`           |
| `city`, `state`, `country`, `location`, `event`     | `XMP-photoshop:*`, `XMP-iptcCore:Location`, `XMP-iptcExt:Event`               |
| `people`, `persons`                                 | `XMP-iptcExt:PersonInImage`                                                   |

- The full result JSON is also stored in `UserComment`.
- Existing values are kept unless you tick **Overwrite existing metadata**.
- Camera tags (make, model, exposure, dates, GPS) are never written.

## Video scenes

![The Video Classifier's Scenes view: a NASA video about asteroid Bennu, split into 19 scenes by TransNetV2, with its score curve and threshold on the timeline, a Speech row showing where the narrator talks, and the Whisper transcript highlighting the line being spoken](docs/screenshot-video.jpg)

*Video: [Why Did NASA Choose Asteroid Bennu?](https://svs.gsfc.nasa.gov/14398), NASA's Scientific Visualization Studio, public domain, via [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Why_Did_NASA_Choose_Asteroid_Bennu-_(SVS14398).webm).*

Choose **Video Classifier** in the page title. Drop a video, pick a detector and press **Detect scenes**. The video side has its own two views: **Scenes**, for detecting scenes and [transcribing speech](#transcribing-speech), and **Tag scenes**, for [running a model or workflow on every scene](#tagging-scenes). Cmd/Ctrl+Enter runs **Detect scenes**, or **Tag all** in Tag scenes.

- **Any size.** The browser uploads the file in 64 MB chunks straight to the server's disk, so neither side holds the whole video in memory. If the upload stops, choose the same file again and it resumes.
- **Preparing.** The server first makes a 360p copy of the video. Every detector reads that copy, so trying another detector or threshold doesn't decode the full-size file again, and the copy plays in the browser even when the original's codec doesn't.
- **Comparing.** Every result is kept. The timeline shows the scenes, the detector's score and its threshold, with one row per result underneath so cuts line up. Click a row to switch to it.
- **Tuning.** Drag the dashed threshold line to get more or fewer scenes. TransNetV2 and the FFmpeg filter save a score for every frame, so their scenes update as you drag and a new threshold never re-reads the video. PySceneDetect's detectors re-run when you let go.
- **Export scenes** downloads a zip of `scenes.json`, `scenes.csv` and a 360p thumbnail of every scene. Each scene has its start, end and duration in seconds, timecodes, frame numbers in the original, and its thumbnail's path; the JSON also describes the video and the detector settings. Scenes tagged with the model or workflow chosen in **Tag scenes** also carry its JSON as `metadata` (a JSON column in the CSV), and with a transcript shown, each scene carries its words as `speech`. The **CSV** and **JSON** links above the scene list download the list without thumbnails. The video itself is never cut.
- Videos are deleted 7 days after you last opened them (`VIDEO_RETENTION_DAYS`), or with **delete** in the Video panel, together with their scene results, tags and transcripts.

| Detector                 | Finds                                    | Notes                                                    |
| ------------------------ | ---------------------------------------- | -------------------------------------------------------- |
| TransNetV2               | hard cuts, dissolves, fades              | Neural network; the most accurate                        |
| PySceneDetect · content  | hard cuts                                | Fast; lower the threshold to find more                   |
| PySceneDetect · adaptive | hard cuts                                | Fewer false cuts in fast motion and pans                 |
| PySceneDetect · fades    | fades to and from black                  | Cuts where the picture goes dark                         |
| FFmpeg scene filter      | hard cuts                                | Rough split; some cuts score low, so try 0.2             |

AutoShot is listed but not available yet: its weights are only published on Baidu Netdisk.

### Transcribing speech

The **Transcript** panel beside the scene list turns the video's speech into timed lines with Whisper ([faster-whisper](https://github.com/SYSTRAN/faster-whisper), on the server's CPU). Choose a model and, if you know it, the language, then press **Transcribe**; otherwise the language is detected from the first speech.

- **Models:** *tiny* and *base* are quick but rough, *small* is a good start, *medium* and *large-v3 turbo* are more accurate and slower, and *large-v3* is the most accurate and slowest. Each downloads once (78 MB to 3 GB) into the `video-cache` volume.
- **Long videos** are transcribed ten minutes at a time, each piece ending at a quiet moment so no word is cut, and silence is skipped. Every word keeps its own time.
- **Reading along:** the line being spoken is highlighted as the video plays, and the list follows it unless you have just scrolled. Click a line to jump there. **Search** shows only the lines with those words. A **Speech** row under the timeline shows where people talk; hover the timeline to see the line at that moment.
- **Downloads:** **SRT** and **VTT** are subtitles (at most two lines and six seconds each, split between words), **TXT** has a time on each line, and **JSON** has every line and word with its times and confidence.
- **With scenes:** while a transcript is shown, **Export scenes** gives each scene the words spoken in it (`speech`, also a CSV column) and adds `transcript.srt` to the zip, and Tag scenes' workflow prompts get them as `{{exif.Speech}}`.
- A video with several audio tracks gets a **Track** menu. Each model, language and track keeps its own transcript.

### Tagging scenes

Once the scenes look right, open **Tag scenes** (next to **Scenes** in the header). It is the photo **Batch** tab for the result shown on the timeline, with the same controls:

- **Run with** a single model or a workflow, with the same model, prompt and workflow settings as the image side (they are shared, so a change on one side applies to the other). The prompt or workflow editor and the output panels sit beside the scene grid, as on the Batch tab.
- Each scene's frame at its thumbnail time is taken from the original video at up to 1536 px, the same size photos are sent at, and run through the model or workflow.
- **Tag all** runs every scene that has no result yet for these settings; with a search or status filter it becomes **Tag shown**, and **Re-run done** runs the done ones again. Stop at any time and press it again to carry on. If the model server stops answering, it retries, then pauses with the remaining scenes kept.
- Search, **Sort** and **Group** work as on the photo grid, on the scenes' JSON plus `scene`, `start`, `end`, `duration` and `status`, e.g. `category:landscape duration>5`.
- Click a scene to see its frame and result, and **Tag scene** to run it again; ← and → move between scenes, Esc goes back to the grid. The strip above the grid shows every scene's status along the video; click it to jump to a scene.
- Workflow prompts get the scene in place of EXIF: `{{exif.FileName}}`, `{{exif.Scene}}`, `{{exif.SceneStart}}`, `{{exif.SceneEnd}}`, `{{exif.SceneDuration}}`, when the video records it, `{{exif.DateTimeOriginal}}` at the frame's time, and with a transcript shown, `{{exif.Speech}}`: the words spoken in the scene.

Results are saved on the server next to the scenes, per scene and per model or workflow, so each keeps its own, as Batch results do. A green strip on the main timeline marks tagged scenes, the Scenes list shows each scene's description (or category and tags), and **Export scenes** includes the JSON from the model or workflow chosen in Tag scenes. Tags are deleted with their scene result or video.

**Disk space.** A video needs its own size in free space, plus room for the preview copy: usually a few percent more, but up to 15% for noisy footage. Uploads are refused when space is short. Docker Desktop and Podman keep volumes inside their VM, whose disk is often only 60–100 GB. For large videos, put them on the host instead: `echo 'VIDEO_DATA=/Volumes/Media/video-data' >> .env`.

**Speed.** Everything runs on the CPU. As a guide, on an 8-core Apple Silicon Mac under Podman, a 29 GB, 4-hour 1080p video uploaded in about 2 minutes and was prepared in about 6. TransNetV2 processes about 90 frames a second, so it needs about an hour for 4 hours at 25 fps, and more for higher frame rates. The PySceneDetect and FFmpeg detectors read the 360p copy, so they are much quicker. Whisper *small* transcribes about 5 times faster than real time, and *large-v3 turbo* about 3 times, so 4 hours of talk take roughly 50 or 80 minutes; silence is skipped. Jobs run one at a time, in the order they were started.

## Command line

`python -m app.cli` runs the settings you exported from the web UI (**Export settings**) over any number of photos, without the browser:

```sh
python3 -m venv .venv-cli && .venv-cli/bin/pip install -r requirements-cli.txt
docker compose up -d     # the vision service listens on 127.0.0.1:8000 for the CLI

.venv-cli/bin/python -m app.cli catalog.settings.json ~/Pictures/Trip --tag --zip
```

| Flag          | Output                                                        |
| ------------- | ------------------------------------------------------------- |
| *(default)*   | `results.jsonl`, one line per photo; used to resume           |
| `--tag`       | tagged copies, keeping names and subfolders, plus `_run.json` |
| `--zip`       | a zip of the tagged copies, like the Batch tab's Export all   |
| `--sidecar`   | `<photo>.json` with each photo's result                       |
| `--overwrite` | replace metadata the photos already have                      |

- `--tag` and `--zip` need [exiftool](https://exiftool.org/).
- Output goes to `photo-organizer-runs/<name>`; `-o` chooses another folder.
- Running the same command again resumes, and `--rerun` starts over.
- The API key is never saved in the settings file. Pass `--api-key` or set `LLM_API_KEY`.
- See `--help` for all options.

## Vision models

| Model              | Kind           | Labels        | Source                                     |
| ------------------ | -------------- | ------------- | ------------------------------------------ |
| `siglip2-base`     | zero-shot      | your own list | Google SigLIP 2                            |
| `efficientnetv2-s` | classification | ImageNet-1k   | `tf_efficientnetv2_s.in21k_ft_in1k` (timm) |
| `resnet50`         | classification | ImageNet-1k   | `resnet50.a1_in1k` (timm)                  |
| `rtdetr-v2-s`      | detection      | COCO-80       | `PekingU/rtdetr_v2_r18vd`                  |

- **Weights:** each model downloads its weights the first time it's used.
- **Speed:** models run on CPU, at roughly 40–800 ms per image on an Apple Silicon Mac.
- **Memory:** SigLIP 2 needs about 1.5 GB of RAM. If the container runs out of memory, give Docker more or set `VISION_MAX_LOADED=1`.
- **Adding a model:** add a `Spec` to `SPECS` in [vision/server.py](vision/server.py).
- **Licences:** all model code and weights are Apache-2.0 or BSD. The ImageNet classifiers were trained on ImageNet, whose terms are for non-commercial research.
- **Video detectors:** TransNetV2 ([soCzech/TransNetV2](https://github.com/soCzech/TransNetV2), MIT) is vendored in [video/transnetv2.py](video/transnetv2.py), and its PyTorch weights download on first use. PySceneDetect is BSD-3-Clause. FFmpeg comes from Debian's packages.
- **Speech:** faster-whisper and its Silero VAD are MIT, as are the Whisper weights; they download on first use, pinned to a revision ([video/transcribe.py](video/transcribe.py)).

## Development

```sh
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
LLM_BASE_URL=http://localhost:8080/v1 uvicorn app.main:app --reload --port 8090
```

To work on the video side outside the containers, run the video service as well. It needs ffmpeg and the packages in [video/requirements.txt](video/requirements.txt) plus torch. Both processes must share one videos folder:

```sh
(cd video && VIDEO_DIR=../data/videos VIDEO_CACHE=../data/cache uvicorn server:app --port 8001) &
VIDEO_DIR=data/videos VIDEO_BASE_URL=http://localhost:8001 uvicorn app.main:app --reload --port 8090
```

## License

[MIT](LICENSE)
