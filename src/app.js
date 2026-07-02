const GENRES = [
  "blues",
  "classical",
  "country",
  "disco",
  "hiphop",
  "jazz",
  "metal",
  "pop",
  "reggae",
  "rock",
];

const PROFILES = {
  blues: { tempo: 82, centroid: 1700, zcr: 0.08, low: 0.48, mid: 0.36, high: 0.16, dynamics: 0.42 },
  classical: { tempo: 74, centroid: 1300, zcr: 0.04, low: 0.38, mid: 0.45, high: 0.17, dynamics: 0.62 },
  country: { tempo: 104, centroid: 2100, zcr: 0.09, low: 0.33, mid: 0.45, high: 0.22, dynamics: 0.45 },
  disco: { tempo: 122, centroid: 2600, zcr: 0.12, low: 0.42, mid: 0.35, high: 0.23, dynamics: 0.34 },
  hiphop: { tempo: 92, centroid: 1900, zcr: 0.11, low: 0.56, mid: 0.3, high: 0.14, dynamics: 0.28 },
  jazz: { tempo: 118, centroid: 2400, zcr: 0.08, low: 0.34, mid: 0.45, high: 0.21, dynamics: 0.58 },
  metal: { tempo: 142, centroid: 3700, zcr: 0.18, low: 0.3, mid: 0.35, high: 0.35, dynamics: 0.24 },
  pop: { tempo: 118, centroid: 2900, zcr: 0.13, low: 0.34, mid: 0.39, high: 0.27, dynamics: 0.31 },
  reggae: { tempo: 78, centroid: 1800, zcr: 0.07, low: 0.49, mid: 0.38, high: 0.13, dynamics: 0.39 },
  rock: { tempo: 132, centroid: 3200, zcr: 0.15, low: 0.36, mid: 0.35, high: 0.29, dynamics: 0.32 },
};

const WEIGHTS = {
  balanced: { tempo: 1, centroid: 1, zcr: 0.9, low: 1, mid: 0.7, high: 0.9, dynamics: 0.7 },
  rhythm: { tempo: 1.7, centroid: 0.7, zcr: 0.8, low: 1.1, mid: 0.5, high: 0.7, dynamics: 0.9 },
  tone: { tempo: 0.6, centroid: 1.6, zcr: 1.1, low: 1, mid: 0.9, high: 1.2, dynamics: 0.5 },
};

const els = {
  file: document.querySelector("#audio-file"),
  fileLabel: document.querySelector("#file-label"),
  demoButton: document.querySelector("#demo-button"),
  clearButton: document.querySelector("#clear-button"),
  downloadButton: document.querySelector("#download-button"),
  normalize: document.querySelector("#normalize-toggle"),
  windowSize: document.querySelector("#window-size"),
  windowOutput: document.querySelector("#window-output"),
  waveform: document.querySelector("#waveform"),
  spectrum: document.querySelector("#spectrum"),
  topGenre: document.querySelector("#top-genre"),
  confidence: document.querySelector("#confidence"),
  metrics: document.querySelector("#metrics"),
  scores: document.querySelector("#scores"),
  modeButtons: [...document.querySelectorAll("[data-mode]")],
};

let audioContext;
let currentAudio = null;
let currentReport = null;
let mode = "balanced";

function getAudioContext() {
  if (!audioContext) {
    audioContext = new (window.AudioContext || window.webkitAudioContext)();
  }
  return audioContext;
}

function setCanvasSize(canvas) {
  const rect = canvas.getBoundingClientRect();
  const scale = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, Math.floor(rect.width * scale));
  canvas.height = Math.max(1, Math.floor(rect.height * scale));
  return scale;
}

function monoChannel(buffer) {
  const length = buffer.length;
  const channels = buffer.numberOfChannels;
  const out = new Float32Array(length);
  for (let channel = 0; channel < channels; channel += 1) {
    const data = buffer.getChannelData(channel);
    for (let i = 0; i < length; i += 1) out[i] += data[i] / channels;
  }
  return out;
}

function normalizeAudio(samples) {
  let peak = 0;
  for (const value of samples) peak = Math.max(peak, Math.abs(value));
  if (peak < 1e-8) return samples;
  return Float32Array.from(samples, (value) => value / peak);
}

function downsample(samples, maxPoints) {
  if (samples.length <= maxPoints) return [...samples];
  const step = samples.length / maxPoints;
  const points = [];
  for (let i = 0; i < maxPoints; i += 1) {
    const start = Math.floor(i * step);
    const end = Math.min(samples.length, Math.floor((i + 1) * step));
    let sum = 0;
    for (let j = start; j < end; j += 1) sum += Math.abs(samples[j]);
    points.push(sum / Math.max(1, end - start));
  }
  return points;
}

function computeSpectrum(samples, sampleRate, fftSize) {
  const windowed = new Float32Array(fftSize);
  const start = Math.max(0, Math.floor((samples.length - fftSize) / 2));
  for (let i = 0; i < fftSize; i += 1) {
    const sample = samples[start + i] || 0;
    const hann = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / (fftSize - 1));
    windowed[i] = sample * hann;
  }

  const bins = Math.floor(fftSize / 2);
  const magnitudes = new Float32Array(bins);
  for (let k = 0; k < bins; k += 1) {
    let real = 0;
    let imag = 0;
    for (let n = 0; n < fftSize; n += 1) {
      const phase = (2 * Math.PI * k * n) / fftSize;
      real += windowed[n] * Math.cos(phase);
      imag -= windowed[n] * Math.sin(phase);
    }
    magnitudes[k] = Math.sqrt(real * real + imag * imag);
  }

  let total = 0;
  let weighted = 0;
  let low = 0;
  let mid = 0;
  let high = 0;
  for (let i = 1; i < magnitudes.length; i += 1) {
    const freq = (i * sampleRate) / fftSize;
    const mag = magnitudes[i];
    total += mag;
    weighted += mag * freq;
    if (freq < 250) low += mag;
    else if (freq < 4000) mid += mag;
    else high += mag;
  }

  total = Math.max(total, 1e-8);
  return {
    magnitudes,
    centroid: weighted / total,
    low: low / total,
    mid: mid / total,
    high: high / total,
  };
}

function estimateTempo(samples, sampleRate) {
  const frame = 1024;
  const hop = 512;
  const energies = [];
  for (let i = 0; i + frame < samples.length; i += hop) {
    let energy = 0;
    for (let j = 0; j < frame; j += 1) energy += samples[i + j] * samples[i + j];
    energies.push(Math.sqrt(energy / frame));
  }

  if (energies.length < 8) return 0;
  const novelty = [];
  for (let i = 1; i < energies.length; i += 1) novelty.push(Math.max(0, energies[i] - energies[i - 1]));

  let bestLag = 0;
  let bestScore = -Infinity;
  const framesPerSecond = sampleRate / hop;
  const minLag = Math.max(1, Math.floor((60 / 180) * framesPerSecond));
  const maxLag = Math.max(minLag + 1, Math.floor((60 / 60) * framesPerSecond));

  for (let lag = minLag; lag <= maxLag; lag += 1) {
    let score = 0;
    for (let i = lag; i < novelty.length; i += 1) score += novelty[i] * novelty[i - lag];
    if (score > bestScore) {
      bestScore = score;
      bestLag = lag;
    }
  }

  return bestLag ? (60 * framesPerSecond) / bestLag : 0;
}

function analyze(samples, sampleRate) {
  const fftSize = Number(els.windowSize.value);
  const source = els.normalize.checked ? normalizeAudio(samples) : samples;
  const spectrum = computeSpectrum(source, sampleRate, fftSize);
  let rms = 0;
  let peak = 0;
  let crossings = 0;
  for (let i = 0; i < source.length; i += 1) {
    const value = source[i];
    rms += value * value;
    peak = Math.max(peak, Math.abs(value));
    if (i > 0 && Math.sign(value) !== Math.sign(source[i - 1])) crossings += 1;
  }
  rms = Math.sqrt(rms / Math.max(1, source.length));

  const zcr = crossings / Math.max(1, source.length - 1);
  const crest = peak / Math.max(rms, 1e-8);
  const dynamics = Math.min(1, Math.max(0, (crest - 1.2) / 6));
  const tempo = estimateTempo(source, sampleRate);

  return {
    duration: source.length / sampleRate,
    sampleRate,
    rms,
    peak,
    zcr,
    tempo,
    centroid: spectrum.centroid,
    low: spectrum.low,
    mid: spectrum.mid,
    high: spectrum.high,
    dynamics,
    magnitudes: spectrum.magnitudes,
    samples: source,
  };
}

function distanceScore(features, profile, weights) {
  const distance =
    weights.tempo * Math.abs(features.tempo - profile.tempo) / 90 +
    weights.centroid * Math.abs(features.centroid - profile.centroid) / 3800 +
    weights.zcr * Math.abs(features.zcr - profile.zcr) / 0.22 +
    weights.low * Math.abs(features.low - profile.low) / 0.6 +
    weights.mid * Math.abs(features.mid - profile.mid) / 0.6 +
    weights.high * Math.abs(features.high - profile.high) / 0.6 +
    weights.dynamics * Math.abs(features.dynamics - profile.dynamics) / 0.8;
  return Math.exp(-Math.max(0, distance) * 2.2);
}

function predict(features) {
  const raw = GENRES.map((genre) => ({
    genre,
    score: distanceScore(features, PROFILES[genre], WEIGHTS[mode]),
  }));
  const sum = raw.reduce((acc, item) => acc + item.score, 0) || 1;
  return raw
    .map((item) => ({ ...item, probability: item.score / sum }))
    .sort((a, b) => b.probability - a.probability);
}

function drawWaveform(samples) {
  const canvas = els.waveform;
  const scale = setCanvasSize(canvas);
  const ctx = canvas.getContext("2d");
  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = "#eef2f7";
  ctx.fillRect(0, 0, w, h);
  ctx.strokeStyle = "#0f766e";
  ctx.lineWidth = 2 * scale;
  ctx.beginPath();

  const points = downsample(samples, Math.floor(w / scale));
  points.forEach((value, i) => {
    const x = (i / Math.max(1, points.length - 1)) * w;
    const y = h / 2 - value * h * 0.42;
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();

  ctx.strokeStyle = "rgba(24, 27, 34, 0.22)";
  ctx.beginPath();
  ctx.moveTo(0, h / 2);
  ctx.lineTo(w, h / 2);
  ctx.stroke();
}

function drawSpectrum(magnitudes) {
  const canvas = els.spectrum;
  const scale = setCanvasSize(canvas);
  const ctx = canvas.getContext("2d");
  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = "#eef2f7";
  ctx.fillRect(0, 0, w, h);
  const bins = downsample(magnitudes, Math.floor(w / (5 * scale)));
  const max = Math.max(...bins, 1e-8);
  const barWidth = w / bins.length;
  bins.forEach((mag, i) => {
    const value = Math.log10(1 + (9 * mag) / max);
    const barHeight = value * h * 0.88;
    ctx.fillStyle = i % 3 === 0 ? "#c2410c" : "#374151";
    ctx.fillRect(i * barWidth, h - barHeight, Math.max(1, barWidth - scale), barHeight);
  });
}

function renderMetrics(features) {
  const data = [
    ["Duration", `${features.duration.toFixed(1)}s`],
    ["RMS", features.rms.toFixed(3)],
    ["Tempo", `${Math.round(features.tempo)} BPM`],
    ["Centroid", `${Math.round(features.centroid)} Hz`],
    ["Zero Cross", features.zcr.toFixed(3)],
    ["High Band", `${Math.round(features.high * 100)}%`],
  ];
  els.metrics.innerHTML = data
    .map(([label, value]) => `<div><span>${label}</span><strong>${value}</strong></div>`)
    .join("");
}

function renderScores(predictions) {
  els.scores.innerHTML = predictions
    .map((item) => {
      const pct = Math.round(item.probability * 100);
      return `
        <div class="score-row">
          <span class="score-name">${item.genre}</span>
          <span class="score-track"><span class="score-fill" style="width:${pct}%"></span></span>
          <span class="score-value">${pct}%</span>
        </div>`;
    })
    .join("");
}

function updateReport(name, features, predictions) {
  currentReport = {
    source: name,
    generatedAt: new Date().toISOString(),
    modelMode: mode,
    topPrediction: predictions[0],
    metrics: {
      durationSeconds: Number(features.duration.toFixed(3)),
      sampleRate: features.sampleRate,
      rms: Number(features.rms.toFixed(5)),
      peak: Number(features.peak.toFixed(5)),
      estimatedTempoBpm: Number(features.tempo.toFixed(2)),
      spectralCentroidHz: Number(features.centroid.toFixed(2)),
      zeroCrossingRate: Number(features.zcr.toFixed(5)),
      lowBandShare: Number(features.low.toFixed(5)),
      midBandShare: Number(features.mid.toFixed(5)),
      highBandShare: Number(features.high.toFixed(5)),
    },
    predictions: predictions.map(({ genre, probability }) => ({
      genre,
      probability: Number(probability.toFixed(5)),
    })),
  };
  els.downloadButton.disabled = false;
}

function renderAnalysis(name, features, predictions) {
  els.topGenre.textContent = predictions[0].genre;
  els.confidence.textContent = `${Math.round(predictions[0].probability * 100)}% confidence`;
  renderMetrics(features);
  renderScores(predictions);
  drawWaveform(features.samples);
  drawSpectrum(features.magnitudes);
  updateReport(name, features, predictions);
}

function runAnalysis(name, samples, sampleRate) {
  const features = analyze(samples, sampleRate);
  const predictions = predict(features);
  currentAudio = { name, samples, sampleRate };
  renderAnalysis(name, features, predictions);
}

async function loadFile(file) {
  const context = getAudioContext();
  const arrayBuffer = await file.arrayBuffer();
  const audioBuffer = await context.decodeAudioData(arrayBuffer);
  const samples = monoChannel(audioBuffer);
  els.fileLabel.textContent = file.name;
  runAnalysis(file.name, samples, audioBuffer.sampleRate);
}

function createDemoMix() {
  const sampleRate = 44100;
  const duration = 8;
  const length = sampleRate * duration;
  const samples = new Float32Array(length);
  for (let i = 0; i < length; i += 1) {
    const t = i / sampleRate;
    const beat = Math.sin(2 * Math.PI * 2.05 * t) > 0.72 ? 1 : 0;
    const bass = Math.sin(2 * Math.PI * 62 * t) * 0.32;
    const guitar = Math.sin(2 * Math.PI * 220 * t + Math.sin(2 * Math.PI * 5 * t)) * 0.18;
    const hat = Math.sin(2 * Math.PI * 6800 * t) * (Math.sin(2 * Math.PI * 8.2 * t) > 0.35 ? 0.06 : 0);
    const kick = beat * Math.sin(2 * Math.PI * 78 * t) * Math.exp(-18 * (t % 0.49));
    samples[i] = bass + guitar + hat + kick;
  }
  els.fileLabel.textContent = "demo-mashup.wav";
  runAnalysis("demo-mashup.wav", normalizeAudio(samples), sampleRate);
}

function resetApp() {
  currentAudio = null;
  currentReport = null;
  els.file.value = "";
  els.fileLabel.textContent = "Choose audio";
  els.topGenre.textContent = "Waiting";
  els.confidence.textContent = "Upload or load a demo mix";
  els.downloadButton.disabled = true;
  els.metrics.innerHTML = ["Duration", "RMS", "Tempo", "Centroid", "Zero Cross", "High Band"]
    .map((label) => `<div><span>${label}</span><strong>-</strong></div>`)
    .join("");
  els.scores.innerHTML = "";
  drawWaveform(new Float32Array(1));
  drawSpectrum(new Float32Array(1));
}

els.file.addEventListener("change", async (event) => {
  const [file] = event.target.files;
  if (!file) return;
  try {
    await loadFile(file);
  } catch (error) {
    els.fileLabel.textContent = "Could not decode audio";
    console.error(error);
  }
});

els.demoButton.addEventListener("click", createDemoMix);
els.clearButton.addEventListener("click", resetApp);
els.windowSize.addEventListener("input", () => {
  els.windowOutput.textContent = els.windowSize.value;
  if (currentAudio) runAnalysis(currentAudio.name, currentAudio.samples, currentAudio.sampleRate);
});
els.normalize.addEventListener("change", () => {
  if (currentAudio) runAnalysis(currentAudio.name, currentAudio.samples, currentAudio.sampleRate);
});
els.modeButtons.forEach((button) => {
  button.addEventListener("click", () => {
    mode = button.dataset.mode;
    els.modeButtons.forEach((item) => item.classList.toggle("active", item === button));
    if (currentAudio) runAnalysis(currentAudio.name, currentAudio.samples, currentAudio.sampleRate);
  });
});
els.downloadButton.addEventListener("click", () => {
  if (!currentReport) return;
  const blob = new Blob([JSON.stringify(currentReport, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = "messy-mashup-report.json";
  link.click();
  URL.revokeObjectURL(url);
});

window.addEventListener("resize", () => {
  if (currentAudio) {
    runAnalysis(currentAudio.name, currentAudio.samples, currentAudio.sampleRate);
  } else {
    resetApp();
  }
});

resetApp();
