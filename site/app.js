const MAX_FILE_MB = 100;

const form = document.getElementById("upload-form");
const fileInput = document.getElementById("file-input");
const selectedFile = document.getElementById("selected-file");
const submitButton = document.getElementById("submit-button");
const downloadSampleButton = document.getElementById("download-sample-button");
const statusPanel = document.getElementById("status-panel");
const statusTitle = document.getElementById("status-title");
const statusCopy = document.getElementById("status-copy");
const dropzone = document.getElementById("dropzone");
const previewPanel = document.getElementById("preview-panel");
const previewSummary = document.getElementById("preview-summary");
const previewOutput = document.getElementById("preview-output");

function setStatus(title, copy, isError = false) {
  statusPanel.hidden = false;
  statusPanel.dataset.state = isError ? "error" : "success";
  statusTitle.textContent = title;
  statusCopy.textContent = copy;
}

function updateSelectedFile() {
  const [file] = fileInput.files;
  selectedFile.textContent = file ? file.name : "No file selected";
}

function detectNativeLittleEndian() {
  const view = new DataView(new ArrayBuffer(2));
  view.setUint16(0, 0x00ff, true);
  return new Uint8Array(view.buffer)[0] === 0xff;
}

function parseHeaderText(headerText) {
  const fortranMatch = headerText.match(/'fortran_order':\s*(True|False)/);
  const shapeMatch = headerText.match(/'shape':\s*\(([^)]*)\)/);
  const descrListMatch = headerText.match(/'descr':\s*(\[[\s\S]*?\])\s*,\s*'fortran_order'/);
  const descrStringMatch = headerText.match(/'descr':\s*'([^']+)'/);

  if (!fortranMatch || !shapeMatch || (!descrListMatch && !descrStringMatch)) {
    throw new Error("Unsupported .npy header format.");
  }

  const shape = shapeMatch[1]
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean)
    .map((item) => Number.parseInt(item, 10));

  if (!shape.length || shape.some((dimension) => !Number.isFinite(dimension) || dimension < 0)) {
    throw new Error("Could not understand the array shape in this .npy file.");
  }

  return {
    fortranOrder: fortranMatch[1] === "True",
    shape,
    descrText: descrListMatch ? descrListMatch[1] : descrStringMatch[1],
    isStructured: Boolean(descrListMatch),
  };
}

function parsePrimitiveDtype(descriptor) {
  const match = descriptor.match(/^([<>|=]?)([A-Za-z])(\d+)$/);
  if (!match) {
    throw new Error(`Unsupported NumPy dtype descriptor: ${descriptor}`);
  }

  const [, endianMarker, kind, sizeText] = match;
  const size = Number.parseInt(sizeText, 10);
  const nativeLittleEndian = detectNativeLittleEndian();
  const littleEndian = endianMarker === "<" || endianMarker === "|" || (endianMarker === "=" && nativeLittleEndian) || endianMarker === "";

  return {
    descriptor,
    kind,
    size,
    littleEndian,
  };
}

function parseStructuredDtype(descriptorText) {
  const fields = [];
  const fieldPattern = /\(\s*['"]([^'"]+)['"]\s*,\s*['"]([^'"]+)['"](?:\s*,\s*\([^)]+\))?\s*\)/g;
  let match;
  let offset = 0;

  while ((match = fieldPattern.exec(descriptorText)) !== null) {
    const name = match[1];
    const dtype = parsePrimitiveDtype(match[2]);
    fields.push({
      name,
      dtype,
      offset,
    });
    offset += dtype.size;
  }

  if (!fields.length) {
    throw new Error("Structured arrays are only supported when field dtypes are simple numeric scalars.");
  }

  return {
    fields,
    recordSize: offset,
  };
}

function readFloat16(view, byteOffset, littleEndian) {
  const bits = view.getUint16(byteOffset, littleEndian);
  const sign = (bits & 0x8000) ? -1 : 1;
  const exponent = (bits & 0x7c00) >> 10;
  const fraction = bits & 0x03ff;

  if (exponent === 0) {
    return fraction === 0 ? sign * 0 : sign * 2 ** (-14) * (fraction / 1024);
  }

  if (exponent === 0x1f) {
    return fraction === 0 ? sign * Infinity : Number.NaN;
  }

  return sign * 2 ** (exponent - 15) * (1 + fraction / 1024);
}

function readScalar(view, byteOffset, dtype) {
  switch (dtype.kind) {
    case "f":
      if (dtype.size === 2) return readFloat16(view, byteOffset, dtype.littleEndian);
      if (dtype.size === 4) return view.getFloat32(byteOffset, dtype.littleEndian);
      if (dtype.size === 8) return view.getFloat64(byteOffset, dtype.littleEndian);
      break;
    case "i":
      if (dtype.size === 1) return view.getInt8(byteOffset);
      if (dtype.size === 2) return view.getInt16(byteOffset, dtype.littleEndian);
      if (dtype.size === 4) return view.getInt32(byteOffset, dtype.littleEndian);
      if (dtype.size === 8) return Number(view.getBigInt64(byteOffset, dtype.littleEndian));
      break;
    case "u":
      if (dtype.size === 1) return view.getUint8(byteOffset);
      if (dtype.size === 2) return view.getUint16(byteOffset, dtype.littleEndian);
      if (dtype.size === 4) return view.getUint32(byteOffset, dtype.littleEndian);
      if (dtype.size === 8) return Number(view.getBigUint64(byteOffset, dtype.littleEndian));
      break;
    case "b":
      if (dtype.size === 1) return view.getUint8(byteOffset) ? 1 : 0;
      break;
    default:
      break;
  }

  throw new Error(`Unsupported dtype for browser conversion: ${dtype.descriptor}`);
}

function product(values) {
  return values.reduce((accumulator, value) => accumulator * value, 1);
}

function unravelIndex(index, shape) {
  const indices = new Array(shape.length).fill(0);
  for (let pointer = shape.length - 1; pointer >= 0; pointer -= 1) {
    indices[pointer] = index % shape[pointer];
    index = Math.floor(index / shape[pointer]);
  }
  return indices;
}

function storageIndex(indices, shape, fortranOrder) {
  if (fortranOrder) {
    let index = 0;
    let stride = 1;
    for (let pointer = 0; pointer < shape.length; pointer += 1) {
      index += indices[pointer] * stride;
      stride *= shape[pointer];
    }
    return index;
  }

  let index = 0;
  for (let pointer = 0; pointer < shape.length; pointer += 1) {
    index = index * shape[pointer] + indices[pointer];
  }
  return index;
}

function parseNpy(arrayBuffer) {
  const bytes = new Uint8Array(arrayBuffer);
  const magic = String.fromCharCode(...bytes.slice(0, 6));
  if (magic !== "\u0093NUMPY") {
    throw new Error("This file is not a valid NumPy .npy file.");
  }

  const major = bytes[6];
  let headerLength;
  let cursor;

  if (major === 1) {
    headerLength = bytes[8] | (bytes[9] << 8);
    cursor = 10;
  } else if (major === 2 || major === 3) {
    headerLength = bytes[8] | (bytes[9] << 8) | (bytes[10] << 16) | (bytes[11] << 24);
    cursor = 12;
  } else {
    throw new Error(`Unsupported .npy format version: ${major}.${bytes[7]}`);
  }

  const headerText = new TextDecoder(major === 3 ? "utf-8" : "latin1")
    .decode(bytes.slice(cursor, cursor + headerLength))
    .trim();

  const header = parseHeaderText(headerText);
  const dataOffset = cursor + headerLength;
  const view = new DataView(arrayBuffer, dataOffset);

  if (header.isStructured) {
    const structured = parseStructuredDtype(header.descrText);
    return {
      dataOffset,
      view,
      shape: header.shape,
      fortranOrder: header.fortranOrder,
      structured,
      dtypeLabel: "structured",
      elementCount: product(header.shape),
      isStructured: true,
    };
  }

  const dtype = parsePrimitiveDtype(header.descrText);
  return {
    dataOffset,
    view,
    shape: header.shape,
    fortranOrder: header.fortranOrder,
    dtype,
    dtypeLabel: header.descrText,
    elementCount: product(header.shape),
    isStructured: false,
  };
}

function getNumericValue(parsed, indices) {
  const index = storageIndex(indices, parsed.shape, parsed.fortranOrder);
  return readScalar(parsed.view, index * parsed.dtype.size, parsed.dtype);
}

function getStructuredValue(parsed, indices, fieldName) {
  const field = parsed.structured.fields.find((item) => item.name.toLowerCase() === fieldName.toLowerCase());
  if (!field) {
    throw new Error("Structured arrays must include x, y, and z fields.");
  }
  const index = storageIndex(indices, parsed.shape, parsed.fortranOrder);
  const byteOffset = (index * parsed.structured.recordSize) + field.offset;
  return readScalar(parsed.view, byteOffset, field.dtype);
}

function extractPoints(parsed) {
  if (parsed.isStructured) {
    const pointCount = product(parsed.shape);
    const points = [];
    for (let row = 0; row < pointCount; row += 1) {
      const indices = unravelIndex(row, parsed.shape);
      points.push([
        getStructuredValue(parsed, indices, "x"),
        getStructuredValue(parsed, indices, "y"),
        getStructuredValue(parsed, indices, "z"),
      ]);
    }
    return points;
  }

  if (parsed.shape.length === 1) {
    if (parsed.shape[0] !== 3) {
      throw new Error("1D arrays must contain exactly 3 values to become one XYZ point.");
    }
    return [[
      getNumericValue(parsed, [0]),
      getNumericValue(parsed, [1]),
      getNumericValue(parsed, [2]),
    ]];
  }

  const lastDimension = parsed.shape[parsed.shape.length - 1];
  if (lastDimension < 3) {
    throw new Error("The uploaded array needs at least 3 values in its last dimension.");
  }

  const leadingShape = parsed.shape.slice(0, -1);
  const pointCount = product(leadingShape);
  const points = [];

  for (let row = 0; row < pointCount; row += 1) {
    const pointIndices = unravelIndex(row, leadingShape);
    points.push([
      getNumericValue(parsed, [...pointIndices, 0]),
      getNumericValue(parsed, [...pointIndices, 1]),
      getNumericValue(parsed, [...pointIndices, 2]),
    ]);
  }

  return points;
}

function formatShape(shape) {
  return `(${shape.join(", ")})`;
}

function formatValue(value) {
  if (!Number.isFinite(value)) {
    return String(value);
  }
  return value.toFixed(8);
}

function buildXyz(points) {
  return `${points.map((row) => row.map(formatValue).join(" ")).join("\n")}\n`;
}

function downloadText(text, fileName) {
  const blob = new Blob([text], { type: "text/plain;charset=utf-8" });
  const objectUrl = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = objectUrl;
  anchor.download = fileName;
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  URL.revokeObjectURL(objectUrl);
}

function renderPreview(points, parsed) {
  previewPanel.hidden = false;
  previewSummary.textContent = `${points.length} points from source shape ${formatShape(parsed.shape)} (${parsed.dtypeLabel})`;
  previewOutput.textContent = points.slice(0, 5).map((row) => row.map(formatValue).join(" ")).join("\n");
}

fileInput.addEventListener("change", updateSelectedFile);

["dragenter", "dragover"].forEach((eventName) => {
  dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropzone.dataset.dragging = "true";
  });
});

["dragleave", "drop"].forEach((eventName) => {
  dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    if (eventName === "drop" && event.dataTransfer.files.length) {
      fileInput.files = event.dataTransfer.files;
      updateSelectedFile();
    }
    dropzone.dataset.dragging = "false";
  });
});

downloadSampleButton.addEventListener("click", () => {
  const sample = [
    [1, 2, 3],
    [4.5, 5.5, 6.5],
    [7.25, 8.25, 9.25],
  ];
  downloadText(buildXyz(sample), "sample.xyz");
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const [file] = fileInput.files;
  if (!file) {
    setStatus("No file selected", "Choose a .npy file first.", true);
    return;
  }

  if (!file.name.toLowerCase().endsWith(".npy")) {
    setStatus("Unsupported file", "Only .npy files are supported.", true);
    return;
  }

  if (file.size > MAX_FILE_MB * 1024 * 1024) {
    setStatus("File too large", `Choose a file smaller than ${MAX_FILE_MB} MB.`, true);
    return;
  }

  submitButton.disabled = true;
  submitButton.textContent = "Converting...";
  setStatus("Working on it", "Reading the NumPy file locally in your browser.");

  try {
    const arrayBuffer = await file.arrayBuffer();
    const parsed = parseNpy(arrayBuffer);
    const points = extractPoints(parsed);
    const xyzText = buildXyz(points);
    const downloadName = file.name.replace(/\.npy$/i, ".xyz");
    downloadText(xyzText, downloadName);
    renderPreview(points, parsed);
    setStatus(
      "Conversion complete",
      `${points.length} points exported from source shape ${formatShape(parsed.shape)}.`
    );
  } catch (error) {
    setStatus(
      "Conversion failed",
      error instanceof Error ? error.message : "This file could not be converted.",
      true
    );
  } finally {
    submitButton.disabled = false;
    submitButton.textContent = "Convert to XYZ";
  }
});
