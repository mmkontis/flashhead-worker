"""LiveKit streaming avatar session for the FlashHead RunPod worker.

The worker joins a LiveKit room as the "flashhead-avatar" participant,
receives the agent's TTS audio over DataStream, generates video in
FlashHead's native 28-frame (1.12 s) KV-cached chunks, and publishes
synchronized video+audio tracks via AvatarRunner.

Idle behaviour: when no TTS audio is buffered, zero-audio chunks are
generated continuously so the avatar keeps subtle idle motion.
"""

import asyncio
import base64
import os
import tempfile
import time
from collections import deque

import numpy as np
from loguru import logger

from livekit import rtc
from livekit.agents.voice.avatar import (
    AudioSegmentEnd,
    AvatarOptions,
    AvatarRunner,
    DataStreamAudioReceiver,
    VideoGenerator,
)

from flash_head.inference import (
    get_audio_embedding,
    get_base_data,
    get_infer_params,
    run_pipeline,
)

WIDTH = 512
HEIGHT = 512
FPS = 25
SAMPLE_RATE = 16000
AVATAR_IDENTITY = "flashhead-avatar"


class FlashHeadChunkGenerator(VideoGenerator):
    """Drives FlashHead chunk-by-chunk from a live audio buffer."""

    def __init__(self, pipeline, out_w: int = WIDTH, out_h: int = HEIGHT) -> None:
        self._pipeline = pipeline
        self._out_w = out_w
        self._out_h = out_h
        params = get_infer_params()
        self._frame_num = params["frame_num"]  # 33
        self._motion_frames = params["motion_frames_num"]  # 5
        self._slice_len = self._frame_num - self._motion_frames  # 28 new frames/chunk
        self._chunk_samples = self._slice_len * SAMPLE_RATE // FPS  # 17920
        cached_len = SAMPLE_RATE * params["cached_audio_duration"]  # 8 s context
        self._audio_ctx = deque([0.0] * cached_len, maxlen=cached_len)
        self._audio_end_idx = params["cached_audio_duration"] * FPS
        self._audio_start_idx = self._audio_end_idx - self._frame_num

        self._buffer = bytearray()  # pending int16 pcm from the agent
        self._segment_end_pending = False
        self._interrupted = False
        self._closed = False
        self._chunk_times: deque[float] = deque(maxlen=50)

    # -- AvatarRunner interface --------------------------------------------

    async def push_audio(self, frame: rtc.AudioFrame | AudioSegmentEnd) -> None:
        if self._closed:
            return
        if isinstance(frame, AudioSegmentEnd):
            self._segment_end_pending = True
            return
        data = bytes(frame.data)
        if frame.sample_rate != SAMPLE_RATE:
            # defensive: agent is configured to send 16 kHz; naive resample if not
            src = np.frombuffer(data, dtype=np.int16).astype(np.float32)
            n_dst = int(len(src) * SAMPLE_RATE / frame.sample_rate)
            idx = np.linspace(0, len(src) - 1, n_dst)
            data = np.interp(idx, np.arange(len(src)), src).astype(np.int16).tobytes()
        self._buffer.extend(data)

    async def clear_buffer(self) -> None:
        self._buffer.clear()
        self._segment_end_pending = False
        self._interrupted = True

    def __aiter__(self):
        return self._stream()

    # -- generation loop ----------------------------------------------------

    def _generate_chunk_sync(self, chunk_f32: np.ndarray) -> np.ndarray:
        """Blocking CUDA work: rolling context -> embedding -> frames [28,out_h,out_w,3] u8."""
        self._audio_ctx.extend(chunk_f32.tolist())
        audio_array = np.array(self._audio_ctx)
        emb = get_audio_embedding(
            self._pipeline, audio_array, self._audio_start_idx, self._audio_end_idx
        )
        video = run_pipeline(self._pipeline, emb)
        video = video[self._motion_frames :]
        if self._out_w != WIDTH or self._out_h != HEIGHT:
            # un-stretch on GPU: the model works on a 512² (stretched) face,
            # published frames go back to the source aspect (e.g. 9:16)
            import torch.nn.functional as F

            v = video.permute(0, 3, 1, 2)  # [T,C,H,W]
            v = F.interpolate(v, size=(self._out_h, self._out_w), mode="bilinear", align_corners=False)
            video = v.permute(0, 2, 3, 1).contiguous()
        return video.cpu().numpy().astype(np.uint8)

    async def _stream(self):
        frame_interval = 1.0 / FPS
        chunk_bytes = self._chunk_samples * 2
        next_tick = time.monotonic()
        while not self._closed:
            if self._interrupted:
                self._interrupted = False
                yield AudioSegmentEnd()

            # segment end that arrived after the buffer already drained
            if self._segment_end_pending and not self._buffer:
                self._segment_end_pending = False
                yield AudioSegmentEnd()

            # take up to one chunk of pending speech, pad the tail with silence
            speaking = len(self._buffer) > 0
            take = min(len(self._buffer), chunk_bytes)
            pcm = bytes(self._buffer[:take])
            del self._buffer[:take]
            if take < chunk_bytes:
                pcm = pcm + b"\x00" * (chunk_bytes - take)
            end_after = self._segment_end_pending and len(self._buffer) == 0 and speaking
            if end_after:
                self._segment_end_pending = False

            chunk_f32 = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
            t0 = time.monotonic()
            try:
                frames = await asyncio.to_thread(self._generate_chunk_sync, chunk_f32)
            except Exception:
                logger.exception("chunk generation failed; emitting silence tick")
                await asyncio.sleep(0.2)
                continue
            gen_s = time.monotonic() - t0
            self._chunk_times.append(gen_s)
            if speaking:
                logger.info(f"chunk: {gen_s:.2f}s for {self._slice_len / FPS:.2f}s video (speech)")

            n_frames = min(self._slice_len, frames.shape[0])
            for i in range(n_frames):
                if self._closed or self._interrupted:
                    break
                yield rtc.VideoFrame(
                    width=self._out_w, height=self._out_h, type=rtc.VideoBufferType.RGB24,
                    data=frames[i].tobytes(),
                )
                if speaking:
                    audio_slice = pcm[i * 1280 : (i + 1) * 1280]  # 640 samples/frame
                    yield rtc.AudioFrame(
                        data=audio_slice,
                        sample_rate=SAMPLE_RATE,
                        num_channels=1,
                        samples_per_channel=len(audio_slice) // 2,
                    )
                next_tick += frame_interval
                delay = next_tick - time.monotonic()
                if delay > 0:
                    await asyncio.sleep(delay)
            # if generation is slower than realtime, don't accumulate schedule debt
            if next_tick < time.monotonic():
                next_tick = time.monotonic()

            if end_after and not self._interrupted:
                yield AudioSegmentEnd()

    async def aclose(self) -> None:
        self._closed = True

    @property
    def stats(self) -> dict:
        times = list(self._chunk_times)
        return {
            "chunks": len(times),
            "avg_chunk_s": round(sum(times) / len(times), 3) if times else None,
            "max_chunk_s": round(max(times), 3) if times else None,
        }


async def run_avatar_session(job_input: dict, pipeline) -> dict:
    """Join the room, run the avatar until the agent leaves or timeout."""
    url = job_input["livekit_url"]
    token = job_input["livekit_token"]
    agent_identity = job_input.get("agent_identity")
    max_session_s = float(job_input.get("max_session_s", 570))
    seed = int(job_input.get("seed", 42))

    # per-session identity conditioning
    tmp = tempfile.mkdtemp()
    img_path = os.path.join(tmp, "cond.png")
    with open(img_path, "wb") as f:
        f.write(base64.b64decode(job_input["image_b64"]))
    t0 = time.time()
    get_base_data(pipeline, cond_image_path_or_dir=img_path, base_seed=seed,
                  use_face_crop=bool(job_input.get("use_face_crop", False)))
    base_s = round(time.time() - t0, 2)
    logger.info(f"get_base_data done in {base_s}s")

    room = rtc.Room()
    await room.connect(url, token)
    logger.info(f"connected to room {room.name} as {room.local_participant.identity}")

    audio_recv = DataStreamAudioReceiver(room, sender_identity=agent_identity)
    out_w = max(256, min(int(job_input.get("output_width", WIDTH)) // 2 * 2, 1024))
    out_h = max(256, min(int(job_input.get("output_height", HEIGHT)) // 2 * 2, 1024))
    generator = FlashHeadChunkGenerator(pipeline, out_w=out_w, out_h=out_h)
    runner = AvatarRunner(
        room=room,
        audio_recv=audio_recv,
        video_gen=generator,
        options=AvatarOptions(
            video_width=out_w, video_height=out_h, video_fps=FPS,
            audio_sample_rate=SAMPLE_RATE, audio_channels=1,
        ),
    )
    await runner.start()
    logger.info("avatar runner started — publishing tracks")

    done = asyncio.Event()

    def _maybe_finish(*_args) -> None:
        remotes = list(room.remote_participants.values())
        agents = [p for p in remotes if (agent_identity and p.identity == agent_identity)
                  or p.kind == rtc.ParticipantKind.PARTICIPANT_KIND_AGENT]
        if not agents:
            done.set()

    room.on("participant_disconnected", _maybe_finish)
    room.on("disconnected", lambda *_: done.set())

    started = time.time()
    try:
        await asyncio.wait_for(done.wait(), timeout=max_session_s)
        reason = "agent_left"
    except asyncio.TimeoutError:
        reason = "max_session_s"

    stats = {
        "session_s": round(time.time() - started, 1),
        "base_data_s": base_s,
        "end_reason": reason,
        **generator.stats,
    }
    logger.info(f"session ended: {stats}")
    try:
        await runner.aclose()
    except Exception:
        logger.exception("runner close failed")
    try:
        await room.disconnect()
    except Exception:
        pass
    return stats
