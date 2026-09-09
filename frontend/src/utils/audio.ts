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

/**
 * Liquid Paint Splash & Wave Transition Sound
 * Synthesizes an organic fluid droplet / resonant wave when theme fills the screen.
 */
export function playThemeSplash(targetTheme: 'light' | 'dark') {
  if (!soundEnabled || !audioCtx) return;
  try {
    const now = audioCtx.currentTime;

    // Primary splash wave oscillator
    const osc = audioCtx.createOscillator();
    const gain = audioCtx.createGain();

    // Secondary harmonic overtone for liquid sheen
    const overtone = audioCtx.createOscillator();
    const overtoneGain = audioCtx.createGain();

    if (targetTheme === 'light') {
      // Ascending crystalline dawn chime
      osc.type = 'sine';
      osc.frequency.setValueAtTime(420, now);
      osc.frequency.exponentialRampToValueAtTime(880, now + 0.18);

      overtone.type = 'triangle';
      overtone.frequency.setValueAtTime(840, now);
      overtone.frequency.exponentialRampToValueAtTime(1760, now + 0.15);
    } else {
      // Descending deep velvet dusk resonance
      osc.type = 'sine';
      osc.frequency.setValueAtTime(780, now);
      osc.frequency.exponentialRampToValueAtTime(260, now + 0.22);

      overtone.type = 'triangle';
      overtone.frequency.setValueAtTime(1560, now);
      overtone.frequency.exponentialRampToValueAtTime(520, now + 0.2);
    }

    gain.gain.setValueAtTime(0.045, now);
    gain.gain.exponentialRampToValueAtTime(0.0001, now + 0.32);

    overtoneGain.gain.setValueAtTime(0.02, now);
    overtoneGain.gain.exponentialRampToValueAtTime(0.0001, now + 0.24);

    osc.connect(gain);
    overtone.connect(overtoneGain);
    gain.connect(audioCtx.destination);
    overtoneGain.connect(audioCtx.destination);

    osc.start(now);
    overtone.start(now);
    osc.stop(now + 0.35);
    overtone.stop(now + 0.35);
  } catch {
    // Graceful fallback
  }
}

