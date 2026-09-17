/**
 * Studio Sound Synthesizer via Web Audio API
 * Generates delicate, luxury haptic ticks and feedback without external audio files.
 */

let audioCtx: AudioContext | null = null;
let soundEnabled = false;

export function initAudio() {
  if (typeof window === 'undefined') return;
  if (!audioCtx) {
    const AudioContextClass = window.AudioContext || (window as any).webkitAudioContext;
    if (AudioContextClass) {
      audioCtx = new AudioContextClass();
    }
  }
  if (audioCtx && audioCtx.state === 'suspended') {
    audioCtx.resume();
  }
}

export function setSoundEnabled(enabled: boolean) {
  soundEnabled = enabled;
  if (enabled) {
    initAudio();
    playClick(800, 0.04);
  }
}

export function getSoundEnabled(): boolean {
  return soundEnabled;
}

/**
 * High-precision click/tick for token generation
 */
export function playTokenTick(tokenIndex: number = 0) {
  if (!soundEnabled || !audioCtx) return;
  try {
    const now = audioCtx.currentTime;
    const osc = audioCtx.createOscillator();
    const gain = audioCtx.createGain();

    // Subtle pitch variation per character to sound like delicate mechanical clockwork
    const freq = 600 + (tokenIndex % 12) * 45;
    osc.type = 'sine';
    osc.frequency.setValueAtTime(freq, now);

    // Micro envelope: 15ms duration with exponential drop
    gain.gain.setValueAtTime(0.025, now);
    gain.gain.exponentialRampToValueAtTime(0.0001, now + 0.018);

    osc.connect(gain);
    gain.connect(audioCtx.destination);

    osc.start(now);
    osc.stop(now + 0.02);
  } catch {
    // Graceful fallback if audio is blocked
  }
}

/**
 * UI button click feedback
 */
export function playClick(freq: number = 440, duration: number = 0.05) {
  if (!soundEnabled || !audioCtx) return;
  try {
    const now = audioCtx.currentTime;
    const osc = audioCtx.createOscillator();
    const gain = audioCtx.createGain();

    osc.type = 'triangle';
    osc.frequency.setValueAtTime(freq, now);
    osc.frequency.exponentialRampToValueAtTime(freq * 0.5, now + duration);

    gain.gain.setValueAtTime(0.04, now);
    gain.gain.exponentialRampToValueAtTime(0.0001, now + duration);

    osc.connect(gain);
    gain.connect(audioCtx.destination);

    osc.start(now);
    osc.stop(now + duration);
  } catch {
    // Silently ignore
  }
}



