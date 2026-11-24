// Log-mel features matching NeMo's AudioToMelSpectrogramPreprocessor for this model:
// preemphasis 0.97, symmetric Hann(400) in a 512-point FFT, hop 160, zero "center" padding,
// power spectrum, Slaney mel filterbank (shipped as fb.bin), log(x + 2^-24), per-feature normalisation.
const N_FFT = 512;
const WIN = 400;
const HOP = 160;
const N_MELS = 80;
const N_BINS = N_FFT / 2 + 1;
const PREEMPH = 0.97;
const LOG_GUARD = 2 ** -24;
const STD_EPS = 1e-5;

const window = new Float64Array(WIN);
for (let n = 0; n < WIN; n++) window[n] = 0.5 - 0.5 * Math.cos((2 * Math.PI * n) / (WIN - 1));

const bitReverse = new Uint16Array(N_FFT);
for (let i = 0; i < N_FFT; i++) {
  let r = 0;
  for (let b = 0, v = i; b < Math.log2(N_FFT); b++, v >>= 1) r = (r << 1) | (v & 1);
  bitReverse[i] = r;
}
const cosTable = new Float64Array(N_FFT / 2);
const sinTable = new Float64Array(N_FFT / 2);
for (let i = 0; i < N_FFT / 2; i++) {
  cosTable[i] = Math.cos((2 * Math.PI * i) / N_FFT);
  sinTable[i] = -Math.sin((2 * Math.PI * i) / N_FFT);
}

function fft(re, im) {
  for (let i = 0; i < N_FFT; i++) {
    const j = bitReverse[i];
    if (j > i) {
      [re[i], re[j]] = [re[j], re[i]];
      [im[i], im[j]] = [im[j], im[i]];
    }
  }
  for (let size = 2; size <= N_FFT; size <<= 1) {
    const half = size >> 1;
    const step = N_FFT / size;
    for (let start = 0; start < N_FFT; start += size) {
      for (let k = 0; k < half; k++) {
        const c = cosTable[k * step];
        const s = sinTable[k * step];
        const a = start + k;
        const b = a + half;
        const tr = re[b] * c - im[b] * s;
        const ti = re[b] * s + im[b] * c;
        re[b] = re[a] - tr;
        im[b] = im[a] - ti;
        re[a] += tr;
        im[a] += ti;
      }
    }
  }
}

/**
 * @param {Float32Array} wave mono 16 kHz samples
 * @param {Float32Array} fb mel filterbank, row-major [80][257]
 * @returns {{features: Float32Array, frames: number, length: number}} features laid out [80][frames];
 *   `length` is the number of valid frames (the last frame is zero, as in NeMo).
 */
export function computeFeatures(wave, fb) {
  const total = wave.length;
  const emphasised = new Float64Array(total);
  emphasised[0] = wave[0];
  for (let i = 1; i < total; i++) emphasised[i] = wave[i] - PREEMPH * wave[i - 1];

  const length = Math.floor(total / HOP);
  const frames = length + 1;
  const logMel = new Float64Array(N_MELS * frames);
  const re = new Float64Array(N_FFT);
  const im = new Float64Array(N_FFT);
  const power = new Float64Array(N_BINS);
  const offset = (N_FFT - WIN) / 2;

  for (let t = 0; t < frames; t++) {
    re.fill(0);
    im.fill(0);
    for (let n = 0; n < WIN; n++) {
      const index = t * HOP + offset + n - N_FFT / 2;
      if (index >= 0 && index < total) re[offset + n] = emphasised[index] * window[n];
    }
    fft(re, im);
    for (let k = 0; k < N_BINS; k++) power[k] = re[k] * re[k] + im[k] * im[k];
    for (let m = 0; m < N_MELS; m++) {
      let sum = 0;
      const row = m * N_BINS;
      for (let k = 0; k < N_BINS; k++) sum += fb[row + k] * power[k];
      logMel[m * frames + t] = Math.log(sum + LOG_GUARD);
    }
  }

  const features = new Float32Array(N_MELS * frames);
  for (let m = 0; m < N_MELS; m++) {
    let mean = 0;
    for (let t = 0; t < length; t++) mean += logMel[m * frames + t];
    mean /= length;
    let variance = 0;
    for (let t = 0; t < length; t++) variance += (logMel[m * frames + t] - mean) ** 2;
    const std = Math.sqrt(variance / (length - 1)) + STD_EPS;
    for (let t = 0; t < length; t++) features[m * frames + t] = (logMel[m * frames + t] - mean) / std;
  }
  return { features, frames, length };
}
