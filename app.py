from __future__ import annotations

from io import BytesIO
import os
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, render_template, request, send_file


app = Flask(__name__)
MAX_UPLOAD_MB = 100
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024


def _extract_points(array: np.ndarray) -> np.ndarray:
    if isinstance(array, np.lib.npyio.NpzFile):
        raise ValueError("Please upload a .npy file, not a .npz archive.")

    if not isinstance(array, np.ndarray):
        raise ValueError("The uploaded file did not contain a NumPy array.")

    if array.size == 0:
        raise ValueError("The uploaded array is empty.")

    if array.dtype.names:
        field_lookup = {name.lower(): name for name in array.dtype.names}
        required = ("x", "y", "z")
        if not all(field in field_lookup for field in required):
            raise ValueError(
                "Structured arrays must include x, y, and z fields."
            )
        return np.column_stack(
            [array[field_lookup[field]].reshape(-1) for field in required]
        ).astype(np.float64, copy=False)

    if not np.issubdtype(array.dtype, np.number):
        raise ValueError("The uploaded array must contain numeric values.")

    if array.ndim == 1:
        if array.shape[0] != 3:
            raise ValueError(
                "1D arrays must contain exactly 3 values to become one XYZ point."
            )
        return array.reshape(1, 3).astype(np.float64, copy=False)

    if array.shape[-1] < 3:
        raise ValueError(
            "The uploaded array needs at least 3 values in its last dimension."
        )

    flattened = array.reshape(-1, array.shape[-1])[:, :3]
    return flattened.astype(np.float64, copy=False)


def convert_npy_bytes_to_xyz(file_bytes: bytes) -> tuple[bytes, int, str]:
    with BytesIO(file_bytes) as buffer:
        array = np.load(buffer, allow_pickle=False)

    points = _extract_points(array)
    output_lines = "\n".join(
        " ".join(f"{value:.8f}" for value in row) for row in points
    )
    xyz_bytes = f"{output_lines}\n".encode("utf-8")
    return xyz_bytes, len(points), str(array.shape)


@app.get("/")
def index() -> str:
    return render_template("index.html", max_upload_mb=MAX_UPLOAD_MB)


@app.get("/healthz")
def healthz() -> tuple[object, int]:
    return jsonify({"status": "ok"}), 200


@app.get("/google<token>.html")
def google_site_verification(token: str) -> tuple[str, int, dict[str, str]]:
    return (
        f"google-site-verification: google{token}.html",
        200,
        {"Content-Type": "text/html; charset=utf-8"},
    )


@app.post("/api/convert")
def convert() -> tuple[object, int] | object:
    uploaded_file = request.files.get("file")
    if uploaded_file is None or uploaded_file.filename == "":
        return jsonify({"error": "Choose a .npy file to convert."}), 400

    file_name = Path(uploaded_file.filename)
    if file_name.suffix.lower() != ".npy":
        return jsonify({"error": "Only .npy files are supported."}), 400

    try:
        xyz_bytes, point_count, source_shape = convert_npy_bytes_to_xyz(
            uploaded_file.read()
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception:
        return (
            jsonify(
                {
                    "error": (
                        "This file could not be converted. Make sure it is a valid "
                        "NumPy .npy file with point data."
                    )
                }
            ),
            500,
        )

    download_name = f"{file_name.stem}.xyz"
    response = send_file(
        BytesIO(xyz_bytes),
        as_attachment=True,
        download_name=download_name,
        mimetype="text/plain; charset=utf-8",
    )
    response.headers["X-Point-Count"] = str(point_count)
    response.headers["X-Source-Shape"] = source_shape
    return response


@app.errorhandler(413)
def file_too_large(_error: Exception) -> tuple[object, int]:
    return (
        jsonify(
            {
                "error": (
                    f"File too large. Upload a file smaller than {MAX_UPLOAD_MB} MB."
                )
            }
        ),
        413,
    )


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    app.run(host="0.0.0.0", port=port, debug=False)
