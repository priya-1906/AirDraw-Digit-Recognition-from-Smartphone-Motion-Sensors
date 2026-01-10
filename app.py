"""Streamlit live demo for AirDraw digit recognition (UDP or Phyphox Remote)."""

from __future__ import annotations

import csv
import io
import json
import os
import re
import socket
import threading
import time
import urllib.request
from collections import deque
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from tensorflow import keras


APP_ROOT = Path(__file__).resolve().parent


def resolve_model_dir() -> Path:
    env_path = os.environ.get("IMU_MODEL_DIR")
    if env_path:
        return Path(env_path)
    candidates = [
        APP_ROOT / "model",
        Path("/models"),
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def resolve_scaler_path() -> Path:
    env_path = os.environ.get("IMU_SCALER_PATH")
    if env_path:
        return Path(env_path)
    candidates = [
        APP_ROOT / "model" / "scaler" / "data.npz",
        Path("/preprocessed_data/train/data.npz"),
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


DEFAULT_MODEL_DIR = resolve_model_dir()
DEFAULT_SCALER_PATH = resolve_scaler_path()

TARGET_LEN = 200
MOVING_AVG_WINDOW = 5

# Phyphox buffer names (detected from remote interface)
DEFAULT_PHY_BUFFER_MAP = {
    "acc_time": "acc_time",
    "accX": "accX",
    "accY": "accY",
    "accZ": "accZ",
    "gyr_time": "gyr_time",
    "gyrX": "gyrX",
    "gyrY": "gyrY",
    "gyrZ": "gyrZ",
}


def apply_moving_average(features: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return features
    if window % 2 == 0:
        window += 1
    pad = window // 2
    kernel = np.ones(window, dtype=float) / window

    padded = np.pad(features, ((pad, pad), (0, 0)), mode="edge")
    smoothed = np.vstack([
        np.convolve(padded[:, i], kernel, mode="valid")
        for i in range(features.shape[1])
    ]).T
    return smoothed


def normalize_timestamp_array(t: np.ndarray) -> np.ndarray:
    if t.size == 0:
        return t
    if np.nanmax(t) > 1e6:
        return t / 1000.0
    return t


def parse_json_message(msg: str, key_map: dict[str, str]) -> dict | None:
    try:
        data = json.loads(msg)
    except Exception:
        return None

    try:
        t = float(data[key_map["t"]])
        ax = float(data[key_map["ax"]])
        ay = float(data[key_map["ay"]])
        az = float(data[key_map["az"]])
        gx = float(data[key_map["gx"]])
        gy = float(data[key_map["gy"]])
        gz = float(data[key_map["gz"]])
    except Exception:
        return None

    return {
        "t": t,
        "ax": ax,
        "ay": ay,
        "az": az,
        "gx": gx,
        "gy": gy,
        "gz": gz,
    }


def parse_csv_message(msg: str, delimiter: str = ",") -> dict | None:
    parts = [p.strip() for p in msg.split(delimiter)]
    if len(parts) < 7:
        return None
    try:
        t = float(parts[0])
        ax, ay, az, gx, gy, gz = map(float, parts[1:7])
    except Exception:
        return None

    return {
        "t": t,
        "ax": ax,
        "ay": ay,
        "az": az,
        "gx": gx,
        "gy": gy,
        "gz": gz,
    }


def _is_number(value: str) -> bool:
    try:
        float(value)
        return True
    except Exception:
        return False


def load_csv_rows(file_bytes: bytes, delimiter: str) -> list[list[str]]:
    text = file_bytes.decode("utf-8", errors="ignore")
    reader = csv.reader(io.StringIO(text), delimiter=delimiter)
    rows = [row for row in reader if any(cell.strip() for cell in row)]
    return rows


def detect_header(rows: list[list[str]]) -> bool:
    if not rows:
        return False
    return any(not _is_number(cell) for cell in rows[0])


def guess_column(col_names: list[str], candidates: list[str]) -> int:
    lower = [c.lower() for c in col_names]
    for cand in candidates:
        cand = cand.lower()
        for i, name in enumerate(lower):
            if name == cand or cand in name:
                return i
    return 0


def extract_csv_series_xyz(
    rows: list[list[str]],
    col_names: list[str],
    mapping: dict[str, str],
    start_row: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    indices = {}
    for key, col in mapping.items():
        if col not in col_names:
            return None
        indices[key] = col_names.index(col)

    t_vals = []
    features = []
    for row in rows[start_row:]:
        if len(row) <= max(indices.values()):
            continue
        try:
            t = float(row[indices["t"]])
            x = float(row[indices["x"]])
            y = float(row[indices["y"]])
            z = float(row[indices["z"]])
        except Exception:
            continue
        t_vals.append(t)
        features.append([x, y, z])

    if len(t_vals) < 5:
        return None

    t_arr = np.array(t_vals, dtype=float)
    X = np.array(features, dtype=float)
    t_arr = normalize_timestamp_array(t_arr)

    mask = np.isfinite(t_arr) & np.all(np.isfinite(X), axis=1)
    t_arr = t_arr[mask]
    X = X[mask]

    if t_arr.size < 5:
        return None

    order = np.argsort(t_arr)
    t_arr = t_arr[order]
    X = X[order]

    if t_arr.size > 1:
        unique_mask = np.concatenate([[True], np.diff(t_arr) != 0])
        t_arr = t_arr[unique_mask]
        X = X[unique_mask]

    return t_arr, X


def extract_csv_series(
    rows: list[list[str]],
    col_names: list[str],
    mapping: dict[str, str],
    start_row: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    indices = {}
    for key, col in mapping.items():
        if col not in col_names:
            return None
        indices[key] = col_names.index(col)

    t_vals = []
    features = []
    for row in rows[start_row:]:
        if len(row) <= max(indices.values()):
            continue
        try:
            t = float(row[indices["t"]])
            ax = float(row[indices["ax"]])
            ay = float(row[indices["ay"]])
            az = float(row[indices["az"]])
            gx = float(row[indices["gx"]])
            gy = float(row[indices["gy"]])
            gz = float(row[indices["gz"]])
        except Exception:
            continue
        t_vals.append(t)
        features.append([ax, ay, az, gx, gy, gz])

    if len(t_vals) < 5:
        return None

    t_arr = np.array(t_vals, dtype=float)
    X = np.array(features, dtype=float)
    t_arr = normalize_timestamp_array(t_arr)

    mask = np.isfinite(t_arr) & np.all(np.isfinite(X), axis=1)
    t_arr = t_arr[mask]
    X = X[mask]

    if t_arr.size < 5:
        return None

    order = np.argsort(t_arr)
    t_arr = t_arr[order]
    X = X[order]

    if t_arr.size > 1:
        unique_mask = np.concatenate([[True], np.diff(t_arr) != 0])
        t_arr = t_arr[unique_mask]
        X = X[unique_mask]

    return t_arr, X


def window_by_duration(t: np.ndarray, X: np.ndarray, duration: float) -> tuple[np.ndarray, np.ndarray] | None:
    if t.size < 5:
        return None
    t_max = t[-1]
    t_min = t_max - duration
    mask = t >= t_min
    if mask.sum() < 5:
        return None
    t = t[mask] - t[mask][0]
    X = X[mask]
    return t, X


def udp_listener(
    host: str,
    port: int,
    buffer: deque,
    lock: threading.Lock,
    stop_event: threading.Event,
    fmt: str,
    key_map: dict[str, str],
    delimiter: str,
) -> None:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((host, port))
    sock.settimeout(0.5)

    while not stop_event.is_set():
        try:
            data, _ = sock.recvfrom(65535)
        except socket.timeout:
            continue
        except Exception:
            break

        text = data.decode("utf-8", errors="ignore").strip()
        if not text:
            continue

        for line in text.splitlines():
            if not line.strip():
                continue
            if fmt == "json":
                sample = parse_json_message(line, key_map)
            else:
                sample = parse_csv_message(line, delimiter)
            if sample is None:
                continue
            with lock:
                buffer.append(sample)

    sock.close()


def build_phyphox_url(base_url: str) -> str:
    base_url = base_url.strip()
    if not base_url.startswith("http://") and not base_url.startswith("https://"):
        base_url = "http://" + base_url
    return base_url.rstrip("/")


def fetch_phyphox_buffers(base_url: str, buffer_names: list[str], timeout: float = 2.0) -> dict:
    # Phyphox expects raw buffer names in the query string (e.g., get?accX&accY&acc_time)
    query = "&".join(buffer_names)
    url = f"{base_url}/get?{query}"
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        data = resp.read().decode("utf-8")
    return json.loads(data)


def fetch_phyphox_html(base_url: str, timeout: float = 2.0) -> str:
    url = f"{base_url}/"
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="ignore")


def fetch_phyphox_config(base_url: str, timeout: float = 2.0) -> dict:
    url = f"{base_url}/config"
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8", errors="ignore"))


def extract_buffers_from_config(config: dict) -> list[str]:
    buffers = config.get("buffers", [])
    names: list[str] = []
    for entry in buffers:
        name = entry.get("name")
        if name and name not in names:
            names.append(name)
    return names


def extract_phyphox_buffer_names(html: str) -> list[str]:
    names: list[str] = []
    for m in re.finditer(r'dataInput\"?:\\[(.*?)\\]', html, re.S):
        names.extend(re.findall(r'"([^"]+)"', m.group(1)))
    unique: list[str] = []
    for name in names:
        if name not in unique:
            unique.append(name)
    return unique


def send_phyphox_command(base_url: str, cmd: str, timeout: float = 2.0) -> bool:
    url = f"{base_url}/control?cmd={cmd}"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            _ = resp.read()
        return True
    except Exception:
        return False


def update_phyphox_state(state: dict, payload: dict) -> None:
    buffers = payload.get("buffer", {})
    for name, buf in buffers.items():
        new_data = buf.get("buffer", [])
        update_mode = buf.get("updateMode", "full")
        size = int(buf.get("size", 0)) if buf.get("size") is not None else 0

        existing = state.get(name, [])
        if update_mode in ("partial", "partialXYZ"):
            if new_data:
                existing.extend(new_data)
        elif update_mode in ("single", "input"):
            if new_data:
                existing.extend(new_data)
        else:
            existing = list(new_data)

        # Trim buffer to a safe length to avoid unbounded growth
        max_len = size if size > 0 else 5000
        if len(existing) > max_len:
            del existing[:len(existing) - max_len]

        state[name] = existing

    status = payload.get("status", {})
    state["_status"] = status
    state["_last_fetch"] = time.time()
    state["_last_error"] = None


def phyphox_listener(
    base_url: str,
    buffer_names: list[str],
    state: dict,
    lock: threading.Lock,
    stop_event: threading.Event,
    poll_interval: float,
) -> None:
    while not stop_event.is_set():
        try:
            payload = fetch_phyphox_buffers(base_url, buffer_names)
        except Exception as exc:
            with lock:
                state["_last_error"] = str(exc)
            time.sleep(poll_interval)
            continue

        with lock:
            update_phyphox_state(state, payload)

        time.sleep(poll_interval)


def load_scaler(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(path)
    return data["feature_mean"], data["feature_std"]


def list_models(model_dir: Path) -> list[Path]:
    if not model_dir.exists():
        return []
    return sorted(model_dir.glob("*.keras"))


def get_window_samples_udp(
    buffer: deque,
    lock: threading.Lock,
    duration: float,
) -> tuple[np.ndarray, np.ndarray] | None:
    with lock:
        samples = list(buffer)

    if len(samples) < 5:
        return None

    samples.sort(key=lambda s: s["t"])
    t = np.array([s["t"] for s in samples], dtype=float)
    t = normalize_timestamp_array(t)

    t_max = t[-1]
    t_min = t_max - duration
    mask = t >= t_min
    if mask.sum() < 5:
        return None

    t = t[mask] - t[mask][0]
    window = [s for i, s in enumerate(samples) if mask[i]]
    X = np.column_stack([
        [s["ax"] for s in window],
        [s["ay"] for s in window],
        [s["az"] for s in window],
        [s["gx"] for s in window],
        [s["gy"] for s in window],
        [s["gz"] for s in window],
    ])
    return t, X


def sanitize_series(t, x, y, z) -> tuple[np.ndarray, np.ndarray] | None:
    n = min(len(t), len(x), len(y), len(z))
    if n < 5:
        return None
    t = np.array(t[:n], dtype=float)
    X = np.column_stack([x[:n], y[:n], z[:n]]).astype(float)

    t = normalize_timestamp_array(t)
    mask = np.isfinite(t) & np.all(np.isfinite(X), axis=1)
    t = t[mask]
    X = X[mask]

    if t.size < 5:
        return None

    order = np.argsort(t)
    t = t[order]
    X = X[order]

    if t.size > 1:
        unique_mask = np.concatenate([[True], np.diff(t) != 0])
        t = t[unique_mask]
        X = X[unique_mask]

    return t, X


def get_phyphox_series(state: dict, lock: threading.Lock, buffer_map: dict) -> dict | None:
    with lock:
        acc_time = list(state.get(buffer_map["acc_time"], []))
        acc_x = list(state.get(buffer_map["accX"], []))
        acc_y = list(state.get(buffer_map["accY"], []))
        acc_z = list(state.get(buffer_map["accZ"], []))
        gyr_time = list(state.get(buffer_map["gyr_time"], []))
        gyr_x = list(state.get(buffer_map["gyrX"], []))
        gyr_y = list(state.get(buffer_map["gyrY"], []))
        gyr_z = list(state.get(buffer_map["gyrZ"], []))

    acc = sanitize_series(acc_time, acc_x, acc_y, acc_z)
    gyr = sanitize_series(gyr_time, gyr_x, gyr_y, gyr_z)
    if acc is None or gyr is None:
        return None

    return {"acc": acc, "gyr": gyr}


def align_and_resample_series(
    acc_t: np.ndarray,
    acc_vals: np.ndarray,
    gyr_t: np.ndarray,
    gyr_vals: np.ndarray,
    duration: float,
    target_len: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    acc_t_max = acc_t[-1]
    gyr_t_max = gyr_t[-1]
    acc_t_min = acc_t_max - duration
    gyr_t_min = gyr_t_max - duration

    acc_mask = acc_t >= acc_t_min
    gyr_mask = gyr_t >= gyr_t_min

    acc_t = acc_t[acc_mask]
    acc_vals = acc_vals[acc_mask]
    gyr_t = gyr_t[gyr_mask]
    gyr_vals = gyr_vals[gyr_mask]

    if acc_t.size < 5 or gyr_t.size < 5:
        return None

    t_start = max(acc_t.min(), gyr_t.min())
    t_end = min(acc_t.max(), gyr_t.max())
    if t_end <= t_start:
        return None

    acc_mask = (acc_t >= t_start) & (acc_t <= t_end)
    gyr_mask = (gyr_t >= t_start) & (gyr_t <= t_end)

    acc_t = acc_t[acc_mask] - t_start
    acc_vals = acc_vals[acc_mask]
    gyr_t = gyr_t[gyr_mask] - t_start
    gyr_vals = gyr_vals[gyr_mask]

    t_uniform = np.linspace(0.0, t_end - t_start, target_len)

    acc_interp = np.vstack([
        np.interp(t_uniform, acc_t, acc_vals[:, i]) for i in range(3)
    ]).T
    gyr_interp = np.vstack([
        np.interp(t_uniform, gyr_t, gyr_vals[:, i]) for i in range(3)
    ]).T

    features = np.hstack([acc_interp, gyr_interp])
    return t_uniform, features


def resample_sequence(t: np.ndarray, X: np.ndarray, target_len: int) -> np.ndarray:
    t_uniform = np.linspace(0.0, t[-1], target_len)
    resampled = np.vstack([
        np.interp(t_uniform, t, X[:, i]) for i in range(X.shape[1])
    ]).T
    return resampled


def predict_digit(model, X_scaled: np.ndarray) -> tuple[int, np.ndarray]:
    probs = model.predict(X_scaled[None, ...], verbose=0)[0]
    pred = int(np.argmax(probs))
    return pred, probs


def plot_signals(t: np.ndarray, X: np.ndarray) -> go.Figure:
    fig = go.Figure()
    labels = ["ax", "ay", "az", "gx", "gy", "gz"]
    for i, label in enumerate(labels):
        fig.add_trace(go.Scatter(x=t, y=X[:, i], mode="lines", name=label))
    fig.update_layout(
        height=350,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title="Time (s)",
        yaxis_title="Sensor value",
    )
    return fig


def plot_probabilities(probs: np.ndarray) -> go.Figure:
    digits = [str(i) for i in range(len(probs))]
    fig = go.Figure([
        go.Bar(x=digits, y=probs, marker_color="#2ca02c")
    ])
    fig.update_layout(
        height=300,
        margin=dict(l=10, r=10, t=30, b=10),
        yaxis_title="Probability",
        xaxis_title="Digit",
    )
    return fig


st.set_page_config(page_title="AirDraw Live Demo", layout="wide")

if "udp_buffer" not in st.session_state:
    st.session_state.udp_buffer = deque(maxlen=20000)
if "udp_lock" not in st.session_state:
    st.session_state.udp_lock = threading.Lock()
if "udp_thread" not in st.session_state:
    st.session_state.udp_thread = None
if "udp_stop_event" not in st.session_state:
    st.session_state.udp_stop_event = threading.Event()
if "udp_listening" not in st.session_state:
    st.session_state.udp_listening = False

if "phy_state" not in st.session_state:
    st.session_state.phy_state = {}
if "phy_lock" not in st.session_state:
    st.session_state.phy_lock = threading.Lock()
if "phy_thread" not in st.session_state:
    st.session_state.phy_thread = None
if "phy_stop_event" not in st.session_state:
    st.session_state.phy_stop_event = threading.Event()
if "phy_listening" not in st.session_state:
    st.session_state.phy_listening = False
if "phy_buffer_names" not in st.session_state:
    st.session_state.phy_buffer_names = list(DEFAULT_PHY_BUFFER_MAP.values())
if "phy_buffer_map" not in st.session_state:
    st.session_state.phy_buffer_map = DEFAULT_PHY_BUFFER_MAP.copy()
if "phy_buffers_loaded" not in st.session_state:
    st.session_state.phy_buffers_loaded = False
if "phy_last_test" not in st.session_state:
    st.session_state.phy_last_test = None
if "phy_url" not in st.session_state:
    st.session_state.phy_url = os.environ.get("PHY_PHOX_URL", "")

st.title("AirDraw Live Digit Recognition")
st.write(
    "Stream IMU data (accelerometer + gyroscope) from your phone and classify digits in real time."
)

with st.sidebar:
    st.header("Data Source")
    data_source = st.radio("Select source", ["UDP", "Phyphox Remote", "CSV Upload"], index=1)

    st.header("Model Selection")
    model_dir = Path(st.text_input("Model folder", str(DEFAULT_MODEL_DIR)))
    model_files = list_models(model_dir)
    if model_files:
        model_path = st.selectbox("Choose model", model_files, format_func=lambda p: p.name)
    else:
        model_path = None
        st.warning("No .keras models found in the selected folder.")

    scaler_path = Path(st.text_input("Scaler file (.npz)", str(DEFAULT_SCALER_PATH)))

    st.header("Capture")
    capture_seconds = st.slider("Capture window (seconds)", 1.0, 2.0, 1.5, 0.1)
    target_len = st.number_input("Target length", value=TARGET_LEN, min_value=50, max_value=500)
    use_smoothing = st.checkbox("Apply moving average", value=True)

    st.header("Connection Settings")

    if data_source == "UDP":
        host = st.text_input("Bind address", "0.0.0.0")
        port = st.number_input("UDP port", min_value=1, max_value=65535, value=5000)
        fmt = st.selectbox("Message format", ["json", "csv"], index=0)
        delimiter = st.text_input("CSV delimiter", ",")

        if fmt == "json":
            st.subheader("JSON key mapping")
            key_map = {
                "t": st.text_input("timestamp key", "timestamp"),
                "ax": st.text_input("ax key", "ax"),
                "ay": st.text_input("ay key", "ay"),
                "az": st.text_input("az key", "az"),
                "gx": st.text_input("gx key", "gx"),
                "gy": st.text_input("gy key", "gy"),
                "gz": st.text_input("gz key", "gz"),
            }
        else:
            key_map = {"t": "", "ax": "", "ay": "", "az": "", "gx": "", "gy": "", "gz": ""}

        st.header("Controls")
        col_a, col_b = st.columns(2)
        start_clicked = col_a.button("Start UDP listener")
        stop_clicked = col_b.button("Stop UDP listener")
        clear_clicked = st.button("Clear UDP buffer")
    elif data_source == "Phyphox Remote":
        phy_url = st.text_input(
            "Phyphox URL",
            value=st.session_state.phy_url,
            placeholder="http://<phone-ip>:8080",
            help="Open Phyphox → Remote Access to see the URL.",
        )
        if phy_url:
            st.session_state.phy_url = phy_url
        poll_interval = st.slider("Poll interval (seconds)", 0.05, 1.0, 0.2, 0.05)

        st.subheader("Phyphox buffers")
        auto_load = st.checkbox("Auto-load buffer names", value=True)
        load_buffers = st.button("Load buffer names from Phyphox")
        base_url_preview = build_phyphox_url(phy_url)
        if (auto_load and not st.session_state.phy_buffers_loaded) or load_buffers:
            try:
                config = fetch_phyphox_config(base_url_preview)
                names = extract_buffers_from_config(config)
                if not names:
                    html = fetch_phyphox_html(base_url_preview)
                    names = extract_phyphox_buffer_names(html)
                if names:
                    st.session_state.phy_buffer_names = names
                    st.session_state.phy_buffers_loaded = True
                    st.success(f"Loaded {len(names)} buffer names.")
                else:
                    st.warning("No buffer names found in the Phyphox page.")
            except Exception as exc:
                st.error(f"Failed to load buffer names: {exc}")

        buffer_options = st.session_state.phy_buffer_names
        if buffer_options:
            def _pick_default(key: str) -> int:
                current = st.session_state.phy_buffer_map.get(key, "")
                if current in buffer_options:
                    return buffer_options.index(current)
                return 0

            st.session_state.phy_buffer_map["acc_time"] = st.selectbox(
                "acc_time buffer", buffer_options, index=_pick_default("acc_time"), key="acc_time_buf"
            )
            st.session_state.phy_buffer_map["accX"] = st.selectbox(
                "accX buffer", buffer_options, index=_pick_default("accX"), key="accx_buf"
            )
            st.session_state.phy_buffer_map["accY"] = st.selectbox(
                "accY buffer", buffer_options, index=_pick_default("accY"), key="accy_buf"
            )
            st.session_state.phy_buffer_map["accZ"] = st.selectbox(
                "accZ buffer", buffer_options, index=_pick_default("accZ"), key="accz_buf"
            )
            st.session_state.phy_buffer_map["gyr_time"] = st.selectbox(
                "gyr_time buffer", buffer_options, index=_pick_default("gyr_time"), key="gyr_time_buf"
            )
            st.session_state.phy_buffer_map["gyrX"] = st.selectbox(
                "gyrX buffer", buffer_options, index=_pick_default("gyrX"), key="gyrx_buf"
            )
            st.session_state.phy_buffer_map["gyrY"] = st.selectbox(
                "gyrY buffer", buffer_options, index=_pick_default("gyrY"), key="gyry_buf"
            )
            st.session_state.phy_buffer_map["gyrZ"] = st.selectbox(
                "gyrZ buffer", buffer_options, index=_pick_default("gyrZ"), key="gyrz_buf"
            )
        else:
            st.info("No buffer names available yet. Use the load button above.")

        st.header("Controls")
        col_a, col_b = st.columns(2)
        start_clicked = col_a.button("Start Phyphox listener")
        stop_clicked = col_b.button("Stop Phyphox listener")
        clear_clicked = st.button("Clear Phyphox buffers")

        st.subheader("Measurement Controls")
        auto_start_meas = st.checkbox("Auto-start measurement on listener start", value=True)
        col_c, col_d, col_e = st.columns(3)
        start_meas = col_c.button("Start measurement")
        stop_meas = col_d.button("Stop measurement")
        clear_meas = col_e.button("Clear measurement")

        st.subheader("Connection Test")
        test_clicked = st.button("Test Phyphox connection")
    else:
        csv_delimiter = st.text_input("CSV delimiter", ",")
        csv_has_header = st.checkbox("CSV has header row", value=True)

if data_source == "UDP":
    if clear_clicked:
        with st.session_state.udp_lock:
            st.session_state.udp_buffer.clear()

    if start_clicked and not st.session_state.udp_listening:
        st.session_state.udp_stop_event = threading.Event()
        st.session_state.udp_thread = threading.Thread(
            target=udp_listener,
            args=(
                host,
                int(port),
                st.session_state.udp_buffer,
                st.session_state.udp_lock,
                st.session_state.udp_stop_event,
                fmt,
                key_map,
                delimiter,
            ),
            daemon=True,
        )
        st.session_state.udp_thread.start()
        st.session_state.udp_listening = True

    if stop_clicked and st.session_state.udp_listening:
        st.session_state.udp_stop_event.set()
        st.session_state.udp_listening = False
elif data_source == "Phyphox Remote":
    if clear_clicked:
        with st.session_state.phy_lock:
            st.session_state.phy_state = {}

    if start_clicked and not st.session_state.phy_listening:
        if not phy_url:
            st.error("Please enter your Phyphox URL first.")
        else:
            st.session_state.phy_stop_event = threading.Event()
            base_url = build_phyphox_url(phy_url)
            buffer_map = st.session_state.phy_buffer_map
            buffer_names = [
                buffer_map["acc_time"],
                buffer_map["accX"],
                buffer_map["accY"],
                buffer_map["accZ"],
                buffer_map["gyr_time"],
                buffer_map["gyrX"],
                buffer_map["gyrY"],
                buffer_map["gyrZ"],
            ]
            st.session_state.phy_thread = threading.Thread(
                target=phyphox_listener,
                args=(
                    base_url,
                    buffer_names,
                    st.session_state.phy_state,
                    st.session_state.phy_lock,
                    st.session_state.phy_stop_event,
                    float(poll_interval),
                ),
                daemon=True,
            )
            st.session_state.phy_thread.start()
            st.session_state.phy_listening = True
            if auto_start_meas:
                if send_phyphox_command(base_url, "start"):
                    st.success("Auto-started Phyphox measurement.")
                else:
                    st.error("Auto-start failed. Check the Phyphox URL.")

    if stop_clicked and st.session_state.phy_listening:
        st.session_state.phy_stop_event.set()
        st.session_state.phy_listening = False

    # Send measurement controls directly to Phyphox
    base_url = build_phyphox_url(phy_url) if phy_url else ""
    if start_meas:
        if not base_url:
            st.error("Please enter your Phyphox URL first.")
        elif send_phyphox_command(base_url, "start"):
            st.success("Phyphox measurement started.")
        else:
            st.error("Failed to start measurement. Check the Phyphox URL.")
    if stop_meas:
        if not base_url:
            st.error("Please enter your Phyphox URL first.")
        elif send_phyphox_command(base_url, "stop"):
            st.success("Phyphox measurement stopped.")
        else:
            st.error("Failed to stop measurement. Check the Phyphox URL.")
    if clear_meas:
        if not base_url:
            st.error("Please enter your Phyphox URL first.")
        elif send_phyphox_command(base_url, "clear"):
            with st.session_state.phy_lock:
                st.session_state.phy_state = {}
            st.success("Phyphox measurement cleared.")
        else:
            st.error("Failed to clear measurement. Check the Phyphox URL.")

    if test_clicked:
        if not base_url:
            st.error("Please enter your Phyphox URL first.")
        else:
            buffer_map = st.session_state.phy_buffer_map
            buffer_names = [
                buffer_map["acc_time"],
                buffer_map["accX"],
                buffer_map["accY"],
                buffer_map["accZ"],
                buffer_map["gyr_time"],
                buffer_map["gyrX"],
                buffer_map["gyrY"],
                buffer_map["gyrZ"],
            ]
            try:
                payload = fetch_phyphox_buffers(base_url, buffer_names, timeout=3.0)
                st.session_state.phy_last_test = payload
                st.success("Phyphox connection OK.")
            except Exception as exc:
                st.session_state.phy_last_test = None
                st.error(f"Phyphox test failed: {exc}")

csv_series = None
csv_rows_count = None
csv_duration = None
csv_acc_series = None
csv_gyr_series = None
csv_acc_rows = None
csv_gyr_rows = None
csv_mode = None

if data_source == "CSV Upload":
    st.subheader("CSV Upload")
    csv_mode = st.radio(
        "CSV mode",
        ["Combined (acc + gyro in one file)", "Separate accel + gyro files"],
        horizontal=True,
    )

    if csv_mode == "Combined (acc + gyro in one file)":
        uploaded = st.file_uploader("Upload combined CSV", type=["csv"], key="combined_csv")
        if uploaded is not None:
            rows = load_csv_rows(uploaded.getvalue(), csv_delimiter)
            if not rows:
                st.error("CSV appears to be empty.")
            else:
                detected_header = detect_header(rows)
                if detected_header != csv_has_header:
                    st.info(f"Header detection suggests header={detected_header}.")

                if csv_has_header:
                    col_names = [c.strip() if c.strip() else f"col_{i}" for i, c in enumerate(rows[0])]
                    start_row = 1
                else:
                    col_names = [f"col_{i}" for i in range(len(rows[0]))]
                    start_row = 0

                csv_rows_count = len(rows) - start_row
                st.write(f"Detected {csv_rows_count} data rows.")

                # Mapping UI
                default_indices = {
                    "t": guess_column(col_names, ["time (s)", "timestamp", "time", "t"]),
                    "ax": guess_column(col_names, ["acceleration x", "acc x", "accx", "ax"]),
                    "ay": guess_column(col_names, ["acceleration y", "acc y", "accy", "ay"]),
                    "az": guess_column(col_names, ["acceleration z", "acc z", "accz", "az"]),
                    "gx": guess_column(col_names, ["gyroscope x", "gyro x", "gyrx", "gx"]),
                    "gy": guess_column(col_names, ["gyroscope y", "gyro y", "gyry", "gy"]),
                    "gz": guess_column(col_names, ["gyroscope z", "gyro z", "gyrz", "gz"]),
                }

                if not csv_has_header:
                    default_indices = {"t": 0, "ax": 1, "ay": 2, "az": 3, "gx": 4, "gy": 5, "gz": 6}

                def _safe_index(idx: int) -> int:
                    return idx if 0 <= idx < len(col_names) else 0

                mapping = {
                    "t": st.selectbox("Timestamp column", col_names, index=_safe_index(default_indices["t"]), key="csv_t"),
                    "ax": st.selectbox("ax column", col_names, index=_safe_index(default_indices["ax"]), key="csv_ax"),
                    "ay": st.selectbox("ay column", col_names, index=_safe_index(default_indices["ay"]), key="csv_ay"),
                    "az": st.selectbox("az column", col_names, index=_safe_index(default_indices["az"]), key="csv_az"),
                    "gx": st.selectbox("gx column", col_names, index=_safe_index(default_indices["gx"]), key="csv_gx"),
                    "gy": st.selectbox("gy column", col_names, index=_safe_index(default_indices["gy"]), key="csv_gy"),
                    "gz": st.selectbox("gz column", col_names, index=_safe_index(default_indices["gz"]), key="csv_gz"),
                }

                csv_series = extract_csv_series(rows, col_names, mapping, start_row)
                if csv_series is None:
                    st.error("Unable to parse CSV with the selected columns.")
                else:
                    t_csv, _ = csv_series
                    if t_csv.size > 1:
                        csv_duration = float(t_csv[-1] - t_csv[0])
    else:
        acc_file = st.file_uploader("Upload Accelerometer CSV", type=["csv"], key="acc_csv")
        gyr_file = st.file_uploader("Upload Gyroscope CSV", type=["csv"], key="gyr_csv")

        if acc_file is not None and gyr_file is not None:
            acc_rows = load_csv_rows(acc_file.getvalue(), csv_delimiter)
            gyr_rows = load_csv_rows(gyr_file.getvalue(), csv_delimiter)

            if not acc_rows or not gyr_rows:
                st.error("One or both CSV files are empty.")
            else:
                acc_detected = detect_header(acc_rows)
                gyr_detected = detect_header(gyr_rows)
                if acc_detected != csv_has_header:
                    st.info(f"Accel header detection suggests header={acc_detected}.")
                if gyr_detected != csv_has_header:
                    st.info(f"Gyro header detection suggests header={gyr_detected}.")

                if csv_has_header:
                    acc_cols = [c.strip() if c.strip() else f"col_{i}" for i, c in enumerate(acc_rows[0])]
                    gyr_cols = [c.strip() if c.strip() else f"col_{i}" for i, c in enumerate(gyr_rows[0])]
                    acc_start = 1
                    gyr_start = 1
                else:
                    acc_cols = [f"col_{i}" for i in range(len(acc_rows[0]))]
                    gyr_cols = [f"col_{i}" for i in range(len(gyr_rows[0]))]
                    acc_start = 0
                    gyr_start = 0

                csv_acc_rows = len(acc_rows) - acc_start
                csv_gyr_rows = len(gyr_rows) - gyr_start
                st.write(f"Accel rows: {csv_acc_rows} | Gyro rows: {csv_gyr_rows}")

                acc_defaults = {
                    "t": guess_column(acc_cols, ["time (s)", "timestamp", "time", "t"]),
                    "x": guess_column(acc_cols, ["acceleration x", "acc x", "accx", "ax"]),
                    "y": guess_column(acc_cols, ["acceleration y", "acc y", "accy", "ay"]),
                    "z": guess_column(acc_cols, ["acceleration z", "acc z", "accz", "az"]),
                }
                gyr_defaults = {
                    "t": guess_column(gyr_cols, ["time (s)", "timestamp", "time", "t"]),
                    "x": guess_column(gyr_cols, ["gyroscope x", "gyro x", "gyrx", "gx"]),
                    "y": guess_column(gyr_cols, ["gyroscope y", "gyro y", "gyry", "gy"]),
                    "z": guess_column(gyr_cols, ["gyroscope z", "gyro z", "gyrz", "gz"]),
                }

                def _safe_index(idx: int, cols: list[str]) -> int:
                    return idx if 0 <= idx < len(cols) else 0

                st.markdown("**Accelerometer mapping**")
                acc_map = {
                    "t": st.selectbox("Accel time column", acc_cols, index=_safe_index(acc_defaults["t"], acc_cols), key="acc_t"),
                    "x": st.selectbox("Accel x column", acc_cols, index=_safe_index(acc_defaults["x"], acc_cols), key="acc_x"),
                    "y": st.selectbox("Accel y column", acc_cols, index=_safe_index(acc_defaults["y"], acc_cols), key="acc_y"),
                    "z": st.selectbox("Accel z column", acc_cols, index=_safe_index(acc_defaults["z"], acc_cols), key="acc_z"),
                }

                st.markdown("**Gyroscope mapping**")
                gyr_map = {
                    "t": st.selectbox("Gyro time column", gyr_cols, index=_safe_index(gyr_defaults["t"], gyr_cols), key="gyr_t"),
                    "x": st.selectbox("Gyro x column", gyr_cols, index=_safe_index(gyr_defaults["x"], gyr_cols), key="gyr_x"),
                    "y": st.selectbox("Gyro y column", gyr_cols, index=_safe_index(gyr_defaults["y"], gyr_cols), key="gyr_y"),
                    "z": st.selectbox("Gyro z column", gyr_cols, index=_safe_index(gyr_defaults["z"], gyr_cols), key="gyr_z"),
                }

                csv_acc_series = extract_csv_series_xyz(acc_rows, acc_cols, acc_map, acc_start)
                csv_gyr_series = extract_csv_series_xyz(gyr_rows, gyr_cols, gyr_map, gyr_start)

                if csv_acc_series is None or csv_gyr_series is None:
                    st.error("Unable to parse accel/gyro CSV files with the selected columns.")
                else:
                    acc_t, _ = csv_acc_series
                    gyr_t, _ = csv_gyr_series
                    if acc_t.size > 1 and gyr_t.size > 1:
                        csv_duration = float(min(acc_t[-1] - acc_t[0], gyr_t[-1] - gyr_t[0]))

status_col1, status_col2, status_col3 = st.columns(3)

if data_source == "UDP":
    status_col1.metric("Listener", "Running" if st.session_state.udp_listening else "Stopped")
    status_col2.metric("Buffered samples", len(st.session_state.udp_buffer))
    with st.session_state.udp_lock:
        last_ts = st.session_state.udp_buffer[-1]["t"] if st.session_state.udp_buffer else None
    status_col3.metric("Last timestamp", f"{last_ts:.3f}" if last_ts is not None else "-")
elif data_source == "Phyphox Remote":
    status_col1.metric("Listener", "Running" if st.session_state.phy_listening else "Stopped")
    with st.session_state.phy_lock:
        raw_counts = {
            "acc_time": len(st.session_state.phy_state.get(st.session_state.phy_buffer_map["acc_time"], [])),
            "gyr_time": len(st.session_state.phy_state.get(st.session_state.phy_buffer_map["gyr_time"], [])),
        }
    series = get_phyphox_series(
        st.session_state.phy_state,
        st.session_state.phy_lock,
        st.session_state.phy_buffer_map,
    )
    if series:
        acc_t, _ = series["acc"]
        gyr_t, _ = series["gyr"]
        status_col2.metric("Acc samples", acc_t.size)
        status_col3.metric("Gyro samples", gyr_t.size)
    else:
        status_col2.metric("Acc samples", raw_counts.get("acc_time", 0))
        status_col3.metric("Gyro samples", raw_counts.get("gyr_time", 0))
        with st.session_state.phy_lock:
            phy_status = st.session_state.phy_state.get("_status", {})
            last_error = st.session_state.phy_state.get("_last_error")
        measuring = phy_status.get("measuring")
        if measuring is not None:
            st.info(f"Phyphox measuring: {measuring}")
        if last_error:
            st.error(f"Phyphox error: {last_error}")

        # Show last test summary if available
        if st.session_state.phy_last_test:
            status = st.session_state.phy_last_test.get("status", {})
            st.info(f"Last test: measuring={status.get('measuring')} countDown={status.get('countDown')}")

            buf = st.session_state.phy_last_test.get("buffer", {})
            if buf:
                sample_counts = {name: len(data.get('buffer', [])) for name, data in buf.items()}
                st.write("Buffer sample counts:", sample_counts)
else:
    if csv_mode == "Combined (acc + gyro in one file)":
        status_col1.metric("CSV file", "Loaded" if csv_series else "Not loaded")
        status_col2.metric("Rows", csv_rows_count if csv_rows_count is not None else "-")
        if csv_duration is not None:
            status_col3.metric("Duration (s)", f"{csv_duration:.2f}")
        else:
            status_col3.metric("Duration (s)", "-")
    else:
        status_col1.metric("CSV files", "Loaded" if (csv_acc_series and csv_gyr_series) else "Not loaded")
        status_col2.metric("Accel rows", csv_acc_rows if csv_acc_rows is not None else "-")
        status_col3.metric("Gyro rows", csv_gyr_rows if csv_gyr_rows is not None else "-")
        if csv_duration is not None:
            st.info(f"Overlap duration (approx): {csv_duration:.2f}s")

st.subheader("Live Preview")
if data_source == "UDP":
    preview = get_window_samples_udp(st.session_state.udp_buffer, st.session_state.udp_lock, capture_seconds)
    if preview:
        t_preview, X_preview = preview
        st.plotly_chart(plot_signals(t_preview, X_preview), use_container_width=True)
    else:
        st.info("Not enough buffered data for preview. Start the listener and move the phone.")
elif data_source == "Phyphox Remote":
    series = get_phyphox_series(
        st.session_state.phy_state,
        st.session_state.phy_lock,
        st.session_state.phy_buffer_map,
    )
    if series:
        acc_t, acc_vals = series["acc"]
        gyr_t, gyr_vals = series["gyr"]
        aligned = align_and_resample_series(
            acc_t,
            acc_vals,
            gyr_t,
            gyr_vals,
            capture_seconds,
            int(target_len),
        )
        if aligned:
            t_preview, X_preview = aligned
            st.plotly_chart(plot_signals(t_preview, X_preview), use_container_width=True)
        else:
            st.info("Not enough overlapping accel/gyro data for preview.")
    else:
        st.info("Waiting for Phyphox data. Start the experiment in Phyphox or use Start measurement.")
else:
    if csv_mode == "Combined (acc + gyro in one file)":
        if csv_series:
            t_csv, X_csv = csv_series
            window = window_by_duration(t_csv, X_csv, capture_seconds)
            if window:
                t_preview, X_preview = window
                st.plotly_chart(plot_signals(t_preview, X_preview), use_container_width=True)
            else:
                st.info("Not enough CSV data within the selected window.")
        else:
            st.info("Upload a combined CSV file to preview.")
    else:
        if csv_acc_series and csv_gyr_series:
            acc_t, acc_vals = csv_acc_series
            gyr_t, gyr_vals = csv_gyr_series
            aligned = align_and_resample_series(
                acc_t,
                acc_vals,
                gyr_t,
                gyr_vals,
                capture_seconds,
                int(target_len),
            )
            if aligned:
                t_preview, X_preview = aligned
                st.plotly_chart(plot_signals(t_preview, X_preview), use_container_width=True)
            else:
                st.info("Not enough overlapping accel/gyro data for preview.")
        else:
            st.info("Upload both accelerometer and gyroscope CSV files to preview.")

st.subheader("Prediction")
if model_path is None:
    st.warning("Select a model to enable prediction.")
else:
    if st.button("Capture and Predict"):
        prediction_ready = True
        X_resampled = None

        if not scaler_path.exists():
            st.error("Scaler file not found. Please set the correct path.")
            prediction_ready = False
        else:
            if data_source == "UDP":
                window = get_window_samples_udp(
                    st.session_state.udp_buffer,
                    st.session_state.udp_lock,
                    capture_seconds,
                )
                if not window:
                    st.error("Not enough data in buffer. Try again after moving the phone.")
                    prediction_ready = False
                else:
                    t_win, X_win = window
                    X_resampled = resample_sequence(t_win, X_win, int(target_len))
            elif data_source == "Phyphox Remote":
                series = get_phyphox_series(
                    st.session_state.phy_state,
                    st.session_state.phy_lock,
                    st.session_state.phy_buffer_map,
                )
                if not series:
                    st.error("No Phyphox data available yet.")
                    prediction_ready = False
                else:
                    acc_t, acc_vals = series["acc"]
                    gyr_t, gyr_vals = series["gyr"]
                    aligned = align_and_resample_series(
                        acc_t,
                        acc_vals,
                        gyr_t,
                        gyr_vals,
                        capture_seconds,
                        int(target_len),
                    )
                    if not aligned:
                        st.error("Not enough overlapping accel/gyro data for prediction.")
                        prediction_ready = False
                    else:
                        _, X_resampled = aligned
            else:
                if csv_mode == "Combined (acc + gyro in one file)":
                    if csv_series is None:
                        st.error("No CSV data available yet.")
                        prediction_ready = False
                    else:
                        t_csv, X_csv = csv_series
                        window = window_by_duration(t_csv, X_csv, capture_seconds)
                        if not window:
                            st.error("Not enough CSV data within the selected window.")
                            prediction_ready = False
                        else:
                            t_win, X_win = window
                            X_resampled = resample_sequence(t_win, X_win, int(target_len))
                else:
                    if csv_acc_series is None or csv_gyr_series is None:
                        st.error("No accel/gyro CSV data available yet.")
                        prediction_ready = False
                    else:
                        acc_t, acc_vals = csv_acc_series
                        gyr_t, gyr_vals = csv_gyr_series
                        aligned = align_and_resample_series(
                            acc_t,
                            acc_vals,
                            gyr_t,
                            gyr_vals,
                            capture_seconds,
                            int(target_len),
                        )
                        if not aligned:
                            st.error("Not enough overlapping accel/gyro data for prediction.")
                            prediction_ready = False
                        else:
                            _, X_resampled = aligned

        if prediction_ready and X_resampled is not None:
            if use_smoothing:
                X_resampled = apply_moving_average(X_resampled, MOVING_AVG_WINDOW)

            feature_mean, feature_std = load_scaler(scaler_path)
            X_scaled = (X_resampled - feature_mean) / feature_std

            model = keras.models.load_model(model_path)
            pred, probs = predict_digit(model, X_scaled)

            st.success(f"Predicted digit: {pred}")
            st.plotly_chart(plot_probabilities(probs), use_container_width=True)

st.caption(
    "UDP format: JSON or CSV with timestamp, ax, ay, az, gx, gy, gz. "
    "Phyphox Remote: enable Remote Access and run an experiment with accel+gyro. "
    "CSV Upload: include timestamp + ax/ay/az/gx/gy/gz columns."
)
