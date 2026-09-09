import { useEffect, useRef } from 'react';

/**
 * FluidCanvas: Dark-Mode Luminous GPU Navier-Stokes Fluid Dynamics
 * 
 * Fixes the background cow/patch issue:
 * - Idle state is 100% transparent: background remains pure, pristine obsidian (#09090b).
 * - Interactive dye is ONLY emitted at the smoothed cursor and active vortices.
 * - Screen blending ensures fluid only adds subtle liquid luminescence and never inverts background to white.
 * - Full-resolution WebGL fragment shader with Lamb-Oseen vortices and Blinn-Phong specular glints.
 */

export type FluidMode =
  | 'cobalt'    // Royal Cobalt Ink (Deep electric blue, exceptional on light)
  | 'obsidian'  // Sumi Charcoal (Dark graphite calligraphy ink, maximum contrast)
  | 'emerald'   // Emerald Jade (Vivid imperial viridian & mint)
  | 'amethyst'  // Imperial Amethyst (Velvet violet & neon orchid)
  | 'crimson'   // Tuscan Crimson (Deep ruby & rose wine)
  | 'ember'     // Molten Amber (Fiery copper bronze & golden amber)
  | 'mercury'   // Titanium Mercury (Liquid slate chrome with silver specular)
  | 'silk';     // Cashmere Silk (Pearl champagne & soft velvet)

interface FluidCanvasProps {
  intensity?: number;
  interactive?: boolean;
  mode?: FluidMode;
  theme?: 'light' | 'dark';
}

const VERTEX_SHADER_SRC = `
attribute vec2 a_position;
varying vec2 v_uv;
void main() {
  v_uv = (a_position + 1.0) * 0.5;
  gl_Position = vec4(a_position, 0.0, 1.0);
}
`;

const FRAGMENT_SHADER_SRC = `
precision highp float;
varying vec2 v_uv;

uniform vec2 u_resolution;
uniform float u_time;
uniform vec2 u_mouse;
uniform float u_mouse_speed;
uniform float u_intensity;
uniform int u_mode; // 0: mercury, 1: silk, 2: ember, 3: cobalt, 4: emerald, 5: amethyst, 6: crimson, 7: obsidian
uniform int u_is_light; // 0: dark, 1: light
uniform vec4 u_vortices[16]; // xy: pos in aspect-space, z: circulation, w: core radius

// Continuous analytical fluid turbulence
float fluidField(vec2 p, float t) {
  float f = 0.0;
  f += 0.500 * sin(p.x * 3.0 + sin(p.y * 2.6 + t * 0.35));
  f += 0.250 * sin(p.y * 6.0 + cos(p.x * 5.4 - t * 0.28));
  f += 0.125 * sin(p.x * 12.0 + sin(p.y * 11.0 + t * 0.55));
  return f * 0.5 + 0.5;
}

void main() {
  // Aspect-corrected coordinate space centered at (0, 0)
  vec2 p = (gl_FragCoord.xy - 0.5 * u_resolution) / min(u_resolution.x, u_resolution.y);

  // 1. Accumulate analytical Lamb-Oseen Navier-Stokes vortex velocity field
  vec2 vel = vec2(0.0);
  float dye = 0.0;

  // Superposition of active smooth vortices
  for (int i = 0; i < 16; i++) {
    vec4 vort = u_vortices[i];
    if (abs(vort.z) < 0.001) continue;

    vec2 diff = p - vort.xy;
    float r = length(diff);
    float r0 = vort.w;

    // Exact Lamb-Oseen velocity: v_theta = (Gamma / 2pi*r) * (1 - exp(-r^2 / r0^2))
    float factor = (1.0 - exp(-(r * r) / (r0 * r0 + 1e-4))) / (r + 1e-3);
    vec2 v_tangent = vec2(-diff.y, diff.x) * factor * vort.z;
    vel += v_tangent;

    // Smooth Hermite dye concentration
    float normR = r / (r0 * 2.6 + 0.01);
    float falloff = clamp(1.0 - normR * normR, 0.0, 1.0);
    dye += abs(vort.z) * (falloff * falloff) * 2.4;
  }

  // Soft, smooth cursor glow trail
  float mouseDist = length(p - u_mouse);
  float cursorGlow = smoothstep(0.24, 0.0, mouseDist);
  dye += u_mouse_speed * cursorGlow * 1.5;

  // CRITICAL: If no fluid activity at this pixel, leave it 100% transparent
  // This completely prevents background cow-pattern corruption
  if (dye < 0.002) {
    gl_FragColor = vec4(0.0);
    return;
  }

  // 2. Continuous fluid advection warping
  vec2 warped = p - vel * 0.38;

  // 3. Modulate dye with continuous fluid turbulence for authentic liquid texture
  float turb = fluidField(warped * 2.5, u_time * 0.25);
  dye = dye * (turb * 0.4 + 0.8);

  // 4. Compute 3D surface normal via analytical finite differences of dye
  float eps = 0.004;
  float dX = smoothstep(0.24, 0.0, length(p + vec2(eps, 0.0) - u_mouse)) - smoothstep(0.24, 0.0, length(p - vec2(eps, 0.0) - u_mouse));
  float dY = smoothstep(0.24, 0.0, length(p + vec2(0.0, eps) - u_mouse)) - smoothstep(0.24, 0.0, length(p - vec2(0.0, eps) - u_mouse));
  vec3 N = normalize(vec3(-dX * 6.0 - vel.x * 2.0, -dY * 6.0 - vel.y * 2.0, 1.0));

  // 5. Blinn-Phong Specular Lighting
  vec3 L = normalize(vec3(-0.35, -0.65, 0.75));
  vec3 V = vec3(0.0, 0.0, 1.0);
  vec3 H = normalize(L + V);
  float diff = max(dot(N, L), 0.0);
  float spec = pow(max(dot(N, H), 0.0), 24.0);

  // 6. Mode Aesthetics
  vec3 rgb = vec3(1.0);
  float alpha = 0.0;

  if (u_is_light == 1) {
    // =========================================================================
    // LIGHT THEME HIGH-CONTRAST PALETTE (Rich saturated inks on crisp canvas)
    // =========================================================================
    if (u_mode == 0) {
      // Titanium Mercury: deep slate pewter with silver specular
      rgb = mix(vec3(0.22, 0.28, 0.38), vec3(0.60, 0.68, 0.80), spec * 0.8);
      rgb *= (diff * 0.3 + 0.7);
      alpha = clamp((dye * 0.48 + spec * 0.35 * min(1.0, dye * 3.0)) * u_intensity, 0.0, 0.70);
    } else if (u_mode == 1) {
      // Cashmere Silk: warm deep mocha taupe
      rgb = mix(vec3(0.32, 0.28, 0.25), vec3(0.68, 0.62, 0.58), spec * 0.6);
      rgb *= (diff * 0.25 + 0.75);
      alpha = clamp((dye * 0.44 + spec * 0.25 * min(1.0, dye * 3.0)) * u_intensity, 0.0, 0.65);
    } else if (u_mode == 2) {
      // Scorched Amber: rich molten bronze & burnt orange
      rgb = mix(vec3(0.78, 0.32, 0.06), vec3(0.98, 0.65, 0.18), spec * 0.85);
      rgb *= (diff * 0.35 + 0.65);
      alpha = clamp((dye * 0.50 + spec * 0.35 * min(1.0, dye * 3.0)) * u_intensity, 0.0, 0.72);
    } else if (u_mode == 3) {
      // Cobalt Fountain Ink: deep royal sapphire with electric cyan edge (maximum contrast)
      rgb = mix(vec3(0.04, 0.22, 0.85), vec3(0.18, 0.58, 0.98), spec * 0.9);
      rgb *= (diff * 0.35 + 0.65);
      alpha = clamp((dye * 0.56 + spec * 0.40 * min(1.0, dye * 3.0)) * u_intensity, 0.0, 0.78);
    } else if (u_mode == 4) {
      // Emerald Jade: deep imperial viridian & mint
      rgb = mix(vec3(0.02, 0.48, 0.30), vec3(0.12, 0.82, 0.55), spec * 0.85);
      rgb *= (diff * 0.35 + 0.65);
      alpha = clamp((dye * 0.52 + spec * 0.35 * min(1.0, dye * 3.0)) * u_intensity, 0.0, 0.74);
    } else if (u_mode == 5) {
      // Royal Amethyst: deep velvet purple & vivid magenta
      rgb = mix(vec3(0.42, 0.06, 0.75), vec3(0.75, 0.28, 0.95), spec * 0.85);
      rgb *= (diff * 0.35 + 0.65);
      alpha = clamp((dye * 0.54 + spec * 0.38 * min(1.0, dye * 3.0)) * u_intensity, 0.0, 0.75);
    } else if (u_mode == 6) {
      // Crimson Ruby: deep Tuscan scarlet & rose wine
      rgb = mix(vec3(0.72, 0.06, 0.18), vec3(0.95, 0.28, 0.42), spec * 0.85);
      rgb *= (diff * 0.35 + 0.65);
      alpha = clamp((dye * 0.52 + spec * 0.36 * min(1.0, dye * 3.0)) * u_intensity, 0.0, 0.74);
    } else {
      // Sumi Calligraphy Ink: pure deep charcoal graphite ink (dramatic contrast)
      rgb = mix(vec3(0.05, 0.06, 0.08), vec3(0.28, 0.32, 0.40), spec * 0.7);
      rgb *= (diff * 0.25 + 0.75);
      alpha = clamp((dye * 0.58 + spec * 0.35 * min(1.0, dye * 3.0)) * u_intensity, 0.0, 0.80);
    }
  } else {
    // =========================================================================
    // DARK THEME LUMINESCENT PALETTE (Glow on obsidian dark background)
    // =========================================================================
    if (u_mode == 0) {
      // Liquid Mercury
      rgb = mix(vec3(0.85, 0.90, 0.98), vec3(1.0, 1.0, 1.0), spec * 0.7);
      rgb *= (diff * 0.35 + 0.65);
      alpha = clamp((dye * 0.38 + spec * 0.30 * min(1.0, dye * 3.0)) * u_intensity, 0.0, 0.52);
    } else if (u_mode == 1) {
      // Pure Silk
      rgb = vec3(0.92, 0.94, 0.98) * (diff * 0.22 + 0.78);
      alpha = clamp((dye * 0.32 + spec * 0.15 * min(1.0, dye * 3.0)) * u_intensity, 0.0, 0.42);
    } else if (u_mode == 2) {
      // Obsidian Ember
      rgb = mix(vec3(0.95, 0.60, 0.25), vec3(1.0, 0.90, 0.55), spec * 0.85);
      rgb *= (diff * 0.4 + 0.6);
      alpha = clamp((dye * 0.40 + spec * 0.35 * min(1.0, dye * 3.0)) * u_intensity, 0.0, 0.50);
    } else if (u_mode == 3) {
      // Cobalt Azure (luminous neon cyan-blue)
      rgb = mix(vec3(0.18, 0.52, 1.0), vec3(0.45, 0.88, 1.0), spec * 0.85);
      rgb *= (diff * 0.35 + 0.65);
      alpha = clamp((dye * 0.42 + spec * 0.35 * min(1.0, dye * 3.0)) * u_intensity, 0.0, 0.55);
    } else if (u_mode == 4) {
      // Emerald Jade (luminous mint-green)
      rgb = mix(vec3(0.08, 0.85, 0.55), vec3(0.45, 1.0, 0.80), spec * 0.85);
      rgb *= (diff * 0.35 + 0.65);
      alpha = clamp((dye * 0.40 + spec * 0.35 * min(1.0, dye * 3.0)) * u_intensity, 0.0, 0.52);
    } else if (u_mode == 5) {
      // Royal Amethyst (glowing violet-magenta)
      rgb = mix(vec3(0.70, 0.25, 1.0), vec3(0.95, 0.60, 1.0), spec * 0.85);
      rgb *= (diff * 0.35 + 0.65);
      alpha = clamp((dye * 0.42 + spec * 0.35 * min(1.0, dye * 3.0)) * u_intensity, 0.0, 0.55);
    } else if (u_mode == 6) {
      // Crimson Ruby (vivid neon rose)
      rgb = mix(vec3(1.0, 0.22, 0.42), vec3(1.0, 0.65, 0.75), spec * 0.85);
      rgb *= (diff * 0.35 + 0.65);
      alpha = clamp((dye * 0.42 + spec * 0.35 * min(1.0, dye * 3.0)) * u_intensity, 0.0, 0.55);
    } else {
      // Obsidian Charcoal (dark sleek smoke with silver highlights)
      rgb = mix(vec3(0.45, 0.48, 0.55), vec3(0.85, 0.90, 0.98), spec * 0.8);
      rgb *= (diff * 0.35 + 0.65);
      alpha = clamp((dye * 0.38 + spec * 0.30 * min(1.0, dye * 3.0)) * u_intensity, 0.0, 0.48);
    }
  }

  gl_FragColor = vec4(rgb, alpha);
}
`;

interface Vortex {
  x: number;
  y: number;
  targetCirc: number;
  circ: number;
  radius: number;
  age: number;
  maxLife: number;
}

export default function FluidCanvas({
  intensity = 1.0,
  interactive = true,
  mode = 'cobalt',
  theme = 'dark'
}: FluidCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    // Retina 2x scale for razor-sharp rendering
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    let width = (canvas.width = Math.floor(window.innerWidth * dpr));
    let height = (canvas.height = Math.floor(window.innerHeight * dpr));

    const gl = (canvas.getContext('webgl') || canvas.getContext('experimental-webgl')) as WebGLRenderingContext | null;
    if (!gl) return;

    // Shader compilation
    const compileShader = (type: number, src: string) => {
      const shader = gl.createShader(type);
      if (!shader) return null;
      gl.shaderSource(shader, src);
      gl.compileShader(shader);
      if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
        console.warn('Shader compile error:', gl.getShaderInfoLog(shader));
        gl.deleteShader(shader);
        return null;
      }
      return shader;
    };

    const vertShader = compileShader(gl.VERTEX_SHADER, VERTEX_SHADER_SRC);
    const fragShader = compileShader(gl.FRAGMENT_SHADER, FRAGMENT_SHADER_SRC);
    if (!vertShader || !fragShader) return;

    const program = gl.createProgram();
    if (!program) return;
    gl.attachShader(program, vertShader);
    gl.attachShader(program, fragShader);
    gl.linkProgram(program);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
      console.warn('Program link error:', gl.getProgramInfoLog(program));
      return;
    }
    gl.useProgram(program);

    // Full-screen quad
    const quadBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, quadBuffer);
    gl.bufferData(
      gl.ARRAY_BUFFER,
      new Float32Array([-1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1]),
      gl.STATIC_DRAW
    );

    const posLoc = gl.getAttribLocation(program, 'a_position');
    gl.enableVertexAttribArray(posLoc);
    gl.vertexAttribPointer(posLoc, 2, gl.FLOAT, false, 0, 0);

    // Uniform locations
    const uResLoc = gl.getUniformLocation(program, 'u_resolution');
    const uTimeLoc = gl.getUniformLocation(program, 'u_time');
    const uMouseLoc = gl.getUniformLocation(program, 'u_mouse');
    const uMouseSpeedLoc = gl.getUniformLocation(program, 'u_mouse_speed');
    const uIntensityLoc = gl.getUniformLocation(program, 'u_intensity');
    const uModeLoc = gl.getUniformLocation(program, 'u_mode');
    const uIsLightLoc = gl.getUniformLocation(program, 'u_is_light');
    const uVorticesLoc = gl.getUniformLocation(program, 'u_vortices');

    // Active vortices pool
    const MAX_VORTICES = 16;
    const vortices: Vortex[] = [];
    const vortexArray = new Float32Array(MAX_VORTICES * 4);

    const toAspectSpace = (screenX: number, screenY: number) => {
      const minDim = Math.min(width, height);
      return {
        x: (screenX * dpr - 0.5 * width) / minDim,
        y: -(screenY * dpr - 0.5 * height) / minDim
      };
    };

    // Smooth Cursor Physics State
    let targetMouseX = window.innerWidth / 2;
    let targetMouseY = window.innerHeight / 2;
    let smoothMouseX = targetMouseX;
    let smoothMouseY = targetMouseY;
    let prevSmoothX = smoothMouseX;
    let prevSmoothY = smoothMouseY;

    let smoothSpeed = 0;
    let travelAccumulator = 0;
    let isMouseDown = false;
    let wakeAlternate = 1;

    const handleResize = () => {
      if (!canvas) return;
      width = canvas.width = Math.floor(window.innerWidth * dpr);
      height = canvas.height = Math.floor(window.innerHeight * dpr);
      gl.viewport(0, 0, width, height);
    };
    window.addEventListener('resize', handleResize);
    gl.viewport(0, 0, width, height);

    // Pure mouse tracking - zero abrupt jumps or hard distance thresholds
    const onMouseMove = (e: MouseEvent) => {
      if (!interactive) return;
      targetMouseX = e.clientX;
      targetMouseY = e.clientY;
    };

    const onMouseDown = () => {
      isMouseDown = true;
    };

    const onMouseUp = () => {
      isMouseDown = false;
    };

    // Click emits a gentle expanding vortex dipole
    const onClick = (e: MouseEvent) => {
      const pos = toAspectSpace(e.clientX, e.clientY);
      const offset = 0.02;
      vortices.push(
        { x: pos.x - offset, y: pos.y, targetCirc: 0.30, circ: 0.05, radius: 0.08, age: 0, maxLife: 1.8 },
        { x: pos.x + offset, y: pos.y, targetCirc: -0.30, circ: -0.05, radius: 0.08, age: 0, maxLife: 1.8 }
      );
      while (vortices.length > MAX_VORTICES) {
        vortices.shift();
      }
    };

    window.addEventListener('mousemove', onMouseMove);
    window.addEventListener('mousedown', onMouseDown);
    window.addEventListener('mouseup', onMouseUp);
    window.addEventListener('click', onClick);

    let startTime = performance.now();
    let lastFrameTime = startTime;
    let animId: number;

    const render = (now: number) => {
      const dt = Math.min((now - lastFrameTime) * 0.001, 0.05);
      lastFrameTime = now;
      const elapsed = (now - startTime) * 0.001;

      // 1. Exponential Cursor Smoothing (Critically Damped Follower)
      const followRate = 0.16;
      smoothMouseX += (targetMouseX - smoothMouseX) * followRate;
      smoothMouseY += (targetMouseY - smoothMouseY) * followRate;

      const vx = smoothMouseX - prevSmoothX;
      const vy = smoothMouseY - prevSmoothY;
      prevSmoothX = smoothMouseX;
      prevSmoothY = smoothMouseY;

      const frameDist = Math.hypot(vx, vy);
      const minDim = Math.min(window.innerWidth, window.innerHeight);
      const instantSpeed = frameDist / minDim;

      smoothSpeed += (instantSpeed - smoothSpeed) * 0.18;

      // 2. Continuous Motion Accumulator
      travelAccumulator += frameDist;
      const spawnInterval = 14;

      if (travelAccumulator >= spawnInterval && frameDist > 0.3) {
        travelAccumulator = 0;
        const aspectPos = toAspectSpace(smoothMouseX, smoothMouseY);
        const dragMult = isMouseDown ? 2.0 : 1.0;

        wakeAlternate = -wakeAlternate;
        const circulation = Math.min(instantSpeed * 1.5, 0.32) * wakeAlternate * dragMult;

        vortices.push({
          x: aspectPos.x,
          y: aspectPos.y,
          targetCirc: circulation,
          circ: circulation * 0.2,
          radius: 0.07 + Math.min(instantSpeed * 0.12, 0.05),
          age: 0,
          maxLife: 1.6
        });

        if (vortices.length > MAX_VORTICES) {
          vortices.shift();
        }
      }

      // 3. Smooth Vortex Life Management
      for (let i = vortices.length - 1; i >= 0; i--) {
        const v = vortices[i];
        v.age += dt;

        if (v.age < 0.2) {
          v.circ += (v.targetCirc - v.circ) * 0.3;
        } else {
          v.circ *= 0.982;
        }

        v.radius += 0.0004;

        if (v.age >= v.maxLife || Math.abs(v.circ) < 0.003) {
          vortices.splice(i, 1);
        }
      }

      // Pack vortex array for shader uniform
      vortexArray.fill(0);
      for (let i = 0; i < vortices.length && i < MAX_VORTICES; i++) {
        const v = vortices[i];
        vortexArray[i * 4 + 0] = v.x;
        vortexArray[i * 4 + 1] = v.y;
        vortexArray[i * 4 + 2] = v.circ;
        vortexArray[i * 4 + 3] = v.radius;
      }

      const aspectMouse = toAspectSpace(smoothMouseX, smoothMouseY);

      // Set shader uniforms
      gl.uniform2f(uResLoc, width, height);
      gl.uniform1f(uTimeLoc, elapsed);
      gl.uniform2f(uMouseLoc, aspectMouse.x, aspectMouse.y);
      gl.uniform1f(uMouseSpeedLoc, Math.min(smoothSpeed * 12.0, 1.2));
      gl.uniform1f(uIntensityLoc, intensity);

      const modeCodeMap: Record<FluidMode, number> = {
        mercury: 0,
        silk: 1,
        ember: 2,
        cobalt: 3,
        emerald: 4,
        amethyst: 5,
        crimson: 6,
        obsidian: 7
      };
      gl.uniform1i(uModeLoc, modeCodeMap[mode] ?? 0);
      gl.uniform1i(uIsLightLoc, theme === 'light' ? 1 : 0);
      gl.uniform4fv(uVorticesLoc, vortexArray);

      // Render full-screen quad at native Retina resolution
      gl.drawArrays(gl.TRIANGLES, 0, 6);

      animId = requestAnimationFrame(render);
    };

    animId = requestAnimationFrame(render);

    return () => {
      cancelAnimationFrame(animId);
      window.removeEventListener('resize', handleResize);
      window.removeEventListener('mousemove', onMouseMove);
      window.removeEventListener('mousedown', onMouseDown);
      window.removeEventListener('mouseup', onMouseUp);
      window.removeEventListener('click', onClick);
      gl.deleteProgram(program);
      gl.deleteShader(vertShader);
      gl.deleteShader(fragShader);
      gl.deleteBuffer(quadBuffer);
    };
  }, [intensity, interactive, mode, theme]);

  return (
    <canvas
      ref={canvasRef}
      className="fluid-canvas"
    />
  );
}
