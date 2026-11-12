from types import SimpleNamespace

import numpy as np

from wowfish.audio import AudioChunk, devices
from wowfish.audio.detector import BiteEvent
from wowfish.capture import WindowRef
from wowfish.config import AudioConfig, Config, GameConfig, VisionConfig
from wowfish.engine import events as ev
from wowfish.engine.engine import FishingEngine
from wowfish.engine.state import State
from wowfish.gui.presenter import Presenter, ScoreTrace
from wowfish.gui.setup_page import SetupPage
from wowfish.input._win32 import Win32InputDevice
from wowfish.input.keys import MAC_KEYCODES, VK_CODES, X11_NAMES


def test_addon_awaits_gear_and_only_presses_on_bite():
    cfg = Config(GameConfig(interact_key="num0"), VisionConfig(), AudioConfig())
    keys = []
    detector = SimpleNamespace(feed=lambda _: [BiteEvent("bite", 0.8, 0.0)], reset=lambda: None)
    device = SimpleNamespace(is_game_focused=lambda _: True, press_key=lambda k, _: keys.append(k))
    window = WindowRef("WoW", 1, 0, 0, 100, 100, handle=1)
    engine = FishingEngine(cfg, window, capturer=None, input_device=device, audio_source=None,
                           detector=detector, sleep=lambda _: None)
    chunk = AudioChunk(np.zeros(1024, dtype=np.float32), 22050, 0.1)
    engine._addon_step(0, [chunk])
    assert engine.state is State.CASTING and keys == ["num0"]
    engine.state = State.WAIT_BITE
    engine._addon_step(1, [chunk])
    assert keys == ["num0", "num0"] and engine.state is State.INTERACT
    assert (engine.stats.bites_heard, engine.stats.catches, engine.stats.misses) == (1, 0, 0)
    engine._addon_step(2, [chunk])
    assert keys == ["num0", "num0"]
    engine._addon_step(4, [chunk])
    assert engine.state is State.CASTING and keys == ["num0", "num0", "num0"]


def test_windows_no_foreground_window_and_numpad():
    device = Win32InputDevice.__new__(Win32InputDevice)
    device._u = SimpleNamespace(GetForegroundWindow=lambda: None)
    window = WindowRef("WoW", 1, 0, 0, 100, 100, handle=1)
    assert device.is_game_focused(window) is False
    assert VK_CODES["num0"] == 0x60
    assert X11_NAMES["num0"] == "KP_0"
    assert MAC_KEYCODES["num0"] == 0x52


def test_game_audio_uses_game_process(monkeypatch):
    monkeypatch.setattr(devices.sys, "platform", "win32")
    monkeypatch.setattr(devices, "_open_wasapi", lambda pid, **_: pid)
    assert devices.open_capture("game", pid=123) == 123


def test_capture_numpad_zero_with_numlock_on_or_off():
    values = []
    widget = SimpleNamespace(delete=lambda *_: None, insert=lambda _, value: values.append(value))
    page = SimpleNamespace(keys_note=SimpleNamespace(configure=lambda **_: None))
    for keysym, keycode in (("KP_0", 96), ("KP_Insert", 45), ("0", 96)):
        SetupPage._capture_key(page, SimpleNamespace(keysym=keysym, keycode=keycode, widget=widget))
    assert values == ["num0", "num0", "num0"]


def test_addon_does_not_accept_uncalibrated_template_match():
    cfg = Config(GameConfig(), VisionConfig(), AudioConfig())
    window = WindowRef("WoW", 1, 0, 0, 100, 100, handle=1)
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    capturer = SimpleNamespace(grab=lambda: frame)
    detector = SimpleNamespace(reset=lambda: None)
    engine = FishingEngine(
        cfg, window, capturer=capturer,
        input_device=SimpleNamespace(press_key=lambda key, hold: pressed.append(key)),
        audio_source=None,
        detector=detector, calibrate=lambda *_args, **_kw: SimpleNamespace(valid=False),
    )
    pressed = []
    engine.state = State.CASTING
    def fail_fast_path(_):
        raise AssertionError("fast path must not accept the unverified match")

    engine.search = SimpleNamespace(step=fail_fast_path)
    engine._acquire(1.0)
    assert engine.state is State.CASTING
    assert pressed == []
    engine._acquire(5.0)
    assert pressed == []


def test_addon_does_not_press_again_when_gear_search_is_slow():
    cfg = Config(GameConfig(interact_key="num0"), VisionConfig(), AudioConfig())
    window = WindowRef("WoW", 1, 0, 0, 100, 100, handle=1)
    pressed = []
    device = SimpleNamespace(press_key=lambda key, hold: pressed.append(key))
    detector = SimpleNamespace(reset=lambda: None)
    engine = FishingEngine(cfg, window, capturer=None, input_device=device, audio_source=None,
                           detector=detector, sleep=lambda _: None)
    engine.state = State.CASTING
    engine._vision_future = SimpleNamespace(done=lambda: True,
                                            result=lambda: SimpleNamespace(valid=False))
    engine._addon_acquire(10.0)
    assert pressed == []
    assert engine.state is State.CASTING


def test_audio_is_processed_while_vision_is_pending():
    cfg = Config(GameConfig(), VisionConfig(), AudioConfig())
    window = WindowRef("WoW", 1, 0, 0, 100, 100, handle=1)
    heard = []
    detector = SimpleNamespace(feed=lambda chunk: heard.append(chunk) or [])
    engine = FishingEngine(cfg, window, capturer=None, input_device=None, audio_source=None,
                           detector=detector, sleep=lambda _: None)
    engine.state = State.CASTING
    engine._vision_future = SimpleNamespace(done=lambda: False)
    chunk = AudioChunk(np.zeros(1024, dtype=np.float32), 22050, 0.1)
    engine._addon_step(1, [chunk])
    assert heard == [chunk]


def test_addon_recasts_after_cast_timeout_and_grace_in_both_search_states():
    cfg = Config(GameConfig(interact_key="num0"), VisionConfig(), AudioConfig())
    window = WindowRef("WoW", 1, 0, 0, 100, 100, handle=1)
    presses = []
    resets = []
    detector = SimpleNamespace(feed=lambda _: [], reset=lambda: resets.append(True))
    device = SimpleNamespace(is_game_focused=lambda _: True,
                             press_key=lambda key, _: presses.append(key))
    engine = FishingEngine(cfg, window, capturer=None, input_device=device, audio_source=None,
                           detector=detector, sleep=lambda _: None)
    engine._addon_step(0.0, [])
    engine._addon_acquire = lambda _: None
    engine.state = State.WAIT_BITE
    engine._track = lambda _: None
    engine._addon_step(34.99, [])
    assert presses == ["num0"]
    engine._addon_step(35.0, [])
    assert presses == ["num0", "num0"]
    assert engine.state is State.CASTING and engine._t_cast == 35.0
    engine.state = State.WAIT_BITE
    engine._track = lambda _: None
    engine._addon_step(69.99, [])
    assert len(presses) == 2
    engine._addon_step(70.0, [])
    assert presses == ["num0", "num0", "num0"]
    assert engine.state is State.CASTING and len(resets) == 3


def test_bite_on_timeout_takes_priority_and_stale_vision_is_discarded():
    cfg = Config(GameConfig(interact_key="num0"), VisionConfig(), AudioConfig())
    window = WindowRef("WoW", 1, 0, 0, 100, 100, handle=1)
    presses = []
    detector = SimpleNamespace(feed=lambda _: [BiteEvent("bite", 0.8, 0.0)], reset=lambda: None)
    device = SimpleNamespace(is_game_focused=lambda _: True,
                             press_key=lambda key, _: presses.append(key))
    engine = FishingEngine(cfg, window, capturer=None, input_device=device, audio_source=None,
                           detector=detector, sleep=lambda _: None)
    engine.state = State.WAIT_BITE
    chunk = AudioChunk(np.zeros(1024, dtype=np.float32), 22050, 0.1)
    engine._addon_step(35.0, [chunk])
    assert presses == ["num0"] and engine.state is State.INTERACT
    engine.state = State.CASTING
    engine._vision_future = SimpleNamespace(done=lambda: True,
                                            result=lambda: SimpleNamespace(valid=True))
    engine._vision_generation = 1
    engine._cast_generation = 2
    engine._addon_acquire(33.0)
    assert engine._vision_future is None and engine.state is State.CASTING


def test_addon_score_timer_starts_on_cast_and_survives_icon_found(monkeypatch):
    presenter = Presenter.__new__(Presenter)
    presenter.cfg = Config(GameConfig(), VisionConfig(), AudioConfig())
    presenter.trace = ScoreTrace()
    monkeypatch.setattr("wowfish.gui.presenter.image.with_box", lambda image, box: image)
    monkeypatch.setattr("wowfish.gui.presenter.image.bgr_to_png_bytes", lambda image: b"image")
    presenter._apply(ev.KeyPressed(4.0, "num0", "cast"))
    presenter._add_scores(ev.Scores(5.0, {"fishbite": 0.3}, 0.5))
    presenter._apply(ev.IconFound(10.0, np.zeros((2, 2, 3)), (0, 0, 1, 1), 0.9, 31, (1, 1)))
    assert presenter.trace.t0 == 4.0
    assert len(presenter.trace.points) == 1


def test_bite_heard_during_search_is_used_after_confirmation():
    cfg = Config(GameConfig(interact_key="num0"), VisionConfig(), AudioConfig())
    window = WindowRef("WoW", 1, 0, 0, 100, 100, handle=1)
    presses = []
    detector = SimpleNamespace(feed=lambda _: [BiteEvent("bite", 0.8, 0.0)])
    device = SimpleNamespace(is_game_focused=lambda _: True,
                             press_key=lambda key, _: presses.append(key))
    engine = FishingEngine(cfg, window, capturer=None, input_device=device, audio_source=None,
                           detector=detector, sleep=lambda _: None)
    engine.state = State.CASTING
    engine._addon_acquire = lambda _: None
    chunk = AudioChunk(np.zeros(1024, dtype=np.float32), 22050, 0.1)
    engine._addon_step(1.0, [chunk])
    assert not presses
    engine._addon_acquire = lambda _: setattr(engine, "state", State.WAIT_BITE)
    engine._addon_step(1.5, [])
    assert presses == ["num0"]


def test_addon_recasts_after_eight_seconds_without_gear_and_resets_timer():
    cfg = Config(GameConfig(interact_key="num0"), VisionConfig(), AudioConfig())
    window = WindowRef("WoW", 1, 0, 0, 100, 100, handle=1)
    presses = []
    detector = SimpleNamespace(feed=lambda _: [], reset=lambda: None)
    device = SimpleNamespace(is_game_focused=lambda _: True,
                             press_key=lambda key, _: presses.append(key))
    engine = FishingEngine(cfg, window, capturer=None, input_device=device, audio_source=None,
                           detector=detector, sleep=lambda _: None)
    engine._addon_step(0, [])
    engine._addon_acquire = lambda _: None
    engine._addon_step(7.99, [])
    assert len(presses) == 1
    engine._addon_step(8, [])
    assert len(presses) == 2 and engine._t_cast == 8
    engine.state = State.WAIT_BITE
    engine._track = lambda _: False
    engine._misses = engine.lost_after
    engine._addon_step(10, [])
    engine._addon_step(17.99, [])
    assert len(presses) == 2
    engine._addon_step(18, [])
    assert len(presses) == 3
