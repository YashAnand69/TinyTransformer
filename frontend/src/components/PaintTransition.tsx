import React, { useEffect, useRef, useState } from 'react';
import { playThemeSplash } from '../utils/audio';

export type Theme = 'light' | 'dark';

interface PaintRippleData {
  x: number;
  y: number;
  targetTheme: Theme;
  startTime: number;
  endRadius: number;
  useFallbackOverlay: boolean;
}

// Global emitter so any component or button can trigger the paint transition
type Listener = (data: PaintRippleData) => void;
const listeners: Set<Listener> = new Set();

export function triggerPaintThemeTransition(
  e: React.MouseEvent | MouseEvent | { clientX: number; clientY: number },
  nextTheme: Theme,
  applyTheme: (t: Theme) => void
) {
  const x = 'clientX' in e ? e.clientX : window.innerWidth / 2;
  const y = 'clientY' in e ? e.clientY : window.innerHeight / 2;
  const endRadius = Math.hypot(
    Math.max(x, window.innerWidth - x),
    Math.max(y, window.innerHeight - y)
  );

  // Trigger liquid chime sound
  playThemeSplash(nextTheme);

  const hasViewTransition =
    typeof document !== 'undefined' &&
    'startViewTransition' in document &&
    !window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  // Broadcast ripple visual layer
  const rippleData: PaintRippleData = {
    x,
    y,
    targetTheme: nextTheme,
    startTime: performance.now(),
    endRadius,
    useFallbackOverlay: !hasViewTransition
  };

  listeners.forEach((fn) => fn(rippleData));

  if (hasViewTransition) {
    const transition = (document as any).startViewTransition(() => {
      document.documentElement.setAttribute('data-theme', nextTheme);
      applyTheme(nextTheme);
    });

    transition.ready
      .then(() => {
        document.documentElement.animate(
          {
            clipPath: [
              `circle(0px at ${x}px ${y}px)`,
              `circle(${endRadius}px at ${x}px ${y}px)`
            ]
          },
          {
            duration: 650,
            easing: 'cubic-bezier(0.2, 0.85, 0.32, 1)',
            pseudoElement: '::view-transition-new(root)'
          }
        );
      })
      .catch(() => {
        document.documentElement.setAttribute('data-theme', nextTheme);
        applyTheme(nextTheme);
      });
  } else {
    // Fallback path: switch theme halfway through the canvas ripple fill
    setTimeout(() => {
      document.documentElement.setAttribute('data-theme', nextTheme);
      applyTheme(nextTheme);
    }, 280);
  }
}

/**
 * PaintTransitionOverlay:
 * Renders an organic viscous paint wave ring and fluid droplets radiating
 * from the toggle button across the screen in sync with the theme wipe.
 */
export default function PaintTransitionOverlay() {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [activeRipple, setActiveRipple] = useState<PaintRippleData | null>(null);

  useEffect(() => {
    const handler: Listener = (data) => {
      setActiveRipple(data);
    };
    listeners.add(handler);
    return () => {
      listeners.delete(handler);
    };
  }, []);

  useEffect(() => {
    if (!activeRipple) return;
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let animId: number;
    const { x, y, targetTheme, startTime, endRadius, useFallbackOverlay } = activeRipple;
    const duration = 650;

    // Generate random micro paint droplets around the expanding wave crest
    const dropletCount = 28;
    const droplets = Array.from({ length: dropletCount }, () => ({
      angle: Math.random() * Math.PI * 2,
      speedFactor: 0.88 + Math.random() * 0.24,
      size: 2.5 + Math.random() * 4.5,
      offset: (Math.random() - 0.5) * 40
    }));

    const resize = () => {
      const dpr = window.devicePixelRatio || 1;
      canvas.width = window.innerWidth * dpr;
      canvas.height = window.innerHeight * dpr;
      ctx.scale(dpr, dpr);
    };
    resize();

    const render = (now: number) => {
      const elapsed = now - startTime;
      const progress = Math.min(elapsed / duration, 1.0);

      // Quartic ease-out for liquid deceleration
      const ease = 1 - Math.pow(1 - progress, 3.8);
      const currentRadius = ease * (endRadius + 40);

      ctx.clearRect(0, 0, window.innerWidth, window.innerHeight);

      if (useFallbackOverlay) {
        // Fullscreen paint fill disc for browsers without View Transitions API
        ctx.save();
        ctx.beginPath();
        ctx.arc(x, y, currentRadius, 0, Math.PI * 2);
        ctx.fillStyle = targetTheme === 'light' ? '#f8fafc' : '#09090b';
        ctx.fill();
        ctx.restore();
      }

      // Viscous Paint Surface-Tension Ring at the wave crest
      if (progress < 0.98) {
        ctx.save();
        ctx.beginPath();

        // Wobble the perimeter with high-frequency surface tension waves
        const steps = 90;
        for (let i = 0; i <= steps; i++) {
          const theta = (i / steps) * Math.PI * 2;
          const wobble = Math.sin(theta * 7 + progress * 8) * (8 * (1 - progress));
          const r = currentRadius + wobble;
          const px = x + Math.cos(theta) * r;
          const py = y + Math.sin(theta) * r;
          if (i === 0) {
            ctx.moveTo(px, py);
          } else {
            ctx.lineTo(px, py);
          }
        }
        ctx.closePath();

        // Glowing paint wave boundary gradient
        const fadeAlpha = Math.sin(progress * Math.PI); // Smooth in and out
        ctx.lineWidth = 14 * (1 - progress * 0.7);
        ctx.strokeStyle =
          targetTheme === 'light'
            ? `rgba(37, 99, 235, ${0.45 * fadeAlpha})`
            : `rgba(244, 244, 245, ${0.45 * fadeAlpha})`;
        ctx.shadowColor = targetTheme === 'light' ? '#3b82f6' : '#ffffff';
        ctx.shadowBlur = 18 * fadeAlpha;
        ctx.stroke();

        // Flying micro paint droplets
        droplets.forEach((d) => {
          const rDrop = currentRadius * d.speedFactor + d.offset;
          const dx = x + Math.cos(d.angle) * rDrop;
          const dy = y + Math.sin(d.angle) * rDrop;
          ctx.beginPath();
          ctx.arc(dx, dy, d.size * (1 - progress * 0.8), 0, Math.PI * 2);
          ctx.fillStyle =
            targetTheme === 'light'
              ? `rgba(59, 130, 246, ${0.7 * fadeAlpha})`
              : `rgba(255, 255, 255, ${0.75 * fadeAlpha})`;
          ctx.fill();
        });

        ctx.restore();
      }

      if (progress < 1.0) {
        animId = requestAnimationFrame(render);
      } else {
        ctx.clearRect(0, 0, window.innerWidth, window.innerHeight);
        setActiveRipple(null);
      }
    };

    animId = requestAnimationFrame(render);

    return () => {
      cancelAnimationFrame(animId);
    };
  }, [activeRipple]);

  if (!activeRipple) return null;

  return (
    <canvas
      ref={canvasRef}
      className="paint-ripple-canvas"
      style={{
        position: 'fixed',
        inset: 0,
        width: '100vw',
        height: '100vh',
        pointerEvents: 'none',
        zIndex: 10000
      }}
    />
  );
}
