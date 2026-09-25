import { useState, useEffect, useCallback, useRef } from 'react';
import {
  Terminal,
  Activity,
  Layers,
  Code2,
  Play,
  StepForward,
  RotateCcw,
  Copy,
  Check,
  HelpCircle,
  ChevronDown,
  ChevronUp,
  Cpu,
  Sliders,
  Volume2,
  VolumeX,
  Droplets,
  Eye
} from 'lucide-react';

import FluidCanvas, { type FluidMode } from './components/FluidCanvas';
import GuidedExplainer from './components/GuidedExplainer';
import trainingReport from './data/training_history.json';
import generationBenchmark from './data/generation_benchmark.json';
import curriculumTopics from './data/curriculum_topics.json';
import { MODEL_CODE, TRAIN_CODE } from './data/codebase';
import { playClick, setSoundEnabled } from './utils/audio';
import './App.css';

const API_BASE = import.meta.env.VITE_API_BASE_URL || (import.meta.env.DEV ? 'http://127.0.0.1:8008' : '');

interface Candidate { id: number; char: string; display?: string; prob: number }
interface TokenStep { chosen_id: number; chosen_char: string; chosen_prob?: number; top_candidates: Candidate[] }
interface AttentionData { tokens: string[]; weights: number[][][][]; n_layer: number; n_head: number }
const summary = trainingReport.summary;
const PROGRESSION_TIMELINE = trainingReport.sample_generations.map(sample => {
  const point = trainingReport.history.reduce((best, item) => Math.abs(item.step - sample.step) < Math.abs(best.step - sample.step) ? item : best);
  return { step: sample.step, loss: point.val_loss, perplexity: point.perplexity,
    phase: sample.step === 0 ? 'Start of refinement' : `Training step ${sample.step}`,
    description: sample.note, sample: sample.sample };
});
const chartMax = Math.ceil(Math.max(...trainingReport.history.map(p => Math.max(p.train_loss, p.val_loss))));
const lossPath = (key: 'train_loss' | 'val_loss') => trainingReport.history.map((p,i) => `${i ? 'L' : 'M'} ${40 + 740*p.step/summary.total_steps} ${200 - 180*p[key]/chartMax}`).join(' ');
const formatPrompt = (text: string) => text.startsWith('User:') ? text : `User: ${text.trim()}\nAssistant: `;
async function requestAPI(path: string, body?: unknown, signal?: AbortSignal) {
  const timeout = AbortSignal.timeout(55000);
  const response = await fetch(`${API_BASE}/api/${path}`, {
    method: body === undefined ? 'GET' : 'POST',
    headers: body === undefined ? undefined : { 'Content-Type': 'application/json' },
    body: body === undefined ? undefined : JSON.stringify(body),
    signal: signal ? AbortSignal.any([signal, timeout]) : timeout,
  });
  if (!response.ok) throw new Error(`Request failed (${response.status}). Please try again.`);
  return response.json();
}

export default function App() {
  const [activeTab, setActiveTab] = useState<'playground' | 'loss' | 'attention' | 'architecture' | 'code'>('playground');
  const [showGuideModal, setShowGuideModal] = useState<boolean>(false);

  // Audio & Fluid Settings
  const [soundActive, setSoundActive] = useState<boolean>(false);
  const [fluidMode, setFluidMode] = useState<FluidMode>(() => {
    if (typeof window !== 'undefined') {
      const saved = localStorage.getItem('tinytransformer_fluid_mode');
      if (saved && ['cobalt','obsidian','emerald','amethyst','crimson','ember','mercury','silk'].includes(saved)) return saved as FluidMode;
    }
    return 'cobalt';
  });

  useEffect(() => {
    localStorage.setItem('tinytransformer_fluid_mode', fluidMode);
  }, [fluidMode]);

  // Permanent Dark Mode Setup
  useEffect(() => {
    document.documentElement.setAttribute('data-theme', 'dark');
    if (typeof window !== 'undefined') {
      localStorage.removeItem('tinytransformer_theme');
    }
  }, []);

  // Backend status
  const [backendOnline, setBackendOnline] = useState<boolean | null>(null);
  const [backendDevice, setBackendDevice] = useState<string>('CPU');

  // Playground State
  const [prompt, setPrompt] = useState<string>("User: Who are you?\nAssistant: ");
  const [maxTokens, setMaxTokens] = useState<number>(220);
  const [temperature, setTemperature] = useState<number>(0.35);
  const [topK, setTopK] = useState<number>(10);
  const [topP, setTopP] = useState<number>(0.90);
  const [isGenerating, setIsGenerating] = useState<boolean>(false);
  const [isStepping, setIsStepping] = useState<boolean>(false);
  const [streamingText, setStreamingText] = useState<string>('');
  const [generationStats, setGenerationStats] = useState<{ latency_ms: number; tokens_per_sec: number; tokens_generated: number } | null>(null);
  const [stepDetails, setStepDetails] = useState<TokenStep[]>([]);
  const [selectedStepIndex, setSelectedStepIndex] = useState<number | null>(null);
  const [showLogitDrawer, setShowLogitDrawer] = useState<boolean>(true);
  const [viewMode, setViewMode] = useState<'text' | 'confidence'>('text');
  const [copied, setCopied] = useState<boolean>(false);

  // Attention State
  const [attnInput, setAttnInput] = useState<string>('User: Who are you?');
  const [attnData, setAttnData] = useState<AttentionData | null>(null);
  const [selectedLayer, setSelectedLayer] = useState<number>(0);
  const [selectedHead, setSelectedHead] = useState<number>(0);
  const [isAttnLoading, setIsAttnLoading] = useState<boolean>(false);
  const [hoveredCell, setHoveredCell] = useState<{ qToken: string; kToken: string; score: number } | null>(null);

  const [attentionError, setAttentionError] = useState('');
  const attentionRequest = useRef(0);
  const fetchAttention = useCallback(async (text: string) => {
    const id = ++attentionRequest.current;
    setIsAttnLoading(true); setAttentionError(''); setAttnData(null);
    try {
      const data = await requestAPI('attention', { text });
      if (id === attentionRequest.current) setAttnData(data);
    } catch {
      if (id === attentionRequest.current) setAttentionError('Attention is unavailable. Start the backend or retry the request.');
    } finally {
      if (id === attentionRequest.current) setIsAttnLoading(false);
    }
  }, []);

  // Timeline Scrubber State
  const [scrubberIndex, setScrubberIndex] = useState<number>(PROGRESSION_TIMELINE.length - 1);

  // Code Tab state
  const [selectedCodeFile, setSelectedCodeFile] = useState<'model' | 'train'>('model');
  const [codeCopied, setCodeCopied] = useState<boolean>(false);

  useEffect(() => {
    const controller = new AbortController();
    async function checkHealth() {
      try {
        const data = await requestAPI('health', undefined, controller.signal);
        if (!controller.signal.aborted) {
          setBackendOnline(Boolean(data.model_loaded));
          setBackendDevice(data.device?.toUpperCase() || 'CPU');
        }
      } catch { if (!controller.signal.aborted) setBackendOnline(false); }
    }
    checkHealth();
    const interval = setInterval(checkHealth, 30000);
    return () => { controller.abort(); clearInterval(interval); };
  }, []);
  useEffect(() => { fetchAttention('User: Who are you?'); }, [fetchAttention]);

  const toggleSound = () => {
    const next = !soundActive;
    setSoundActive(next);
    setSoundEnabled(next);
  };

  // Quick Prompt Presets
  const QUICK_PROMPTS = [
    { label: 'Who are you?', text: 'User: Who are you?\nAssistant: ' },
    { label: 'What is attention?', text: 'User: What is attention?\nAssistant: ' },
    { label: 'What is temperature?', text: 'User: What is sampling temperature?\nAssistant: ' },
    { label: 'Architecture overview', text: 'User: What is your architecture?\nAssistant: ' },
    { label: 'Silicon transmutation', text: 'User: What is silicon transmutation?\nAssistant: ' },
    { label: 'Why train from scratch?', text: 'User: Why train from scratch instead of fine-tuning?\nAssistant: ' },
  ];

  const [generationError, setGenerationError] = useState('');
  const [seed, setSeed] = useState('42');
  const generationRequest = useRef<AbortController | null>(null);
  const animation = useRef<number | null>(null);
  const stopGeneration = () => {
    generationRequest.current?.abort();
    if (animation.current !== null) cancelAnimationFrame(animation.current);
    setStepDetails(prev => prev.slice(0, streamingText.length));
    setIsGenerating(false); setIsStepping(false);
  };
  useEffect(() => () => {
    generationRequest.current?.abort();
    if (animation.current !== null) cancelAnimationFrame(animation.current);
  }, []);
  const runGeneration = async (single: boolean) => {
    if (isGenerating || isStepping || !prompt.trim()) return;
    if (seed && (!/^\d+$/.test(seed) || Number(seed) > 4294967295)) {
      setGenerationError('Seed must be a whole number from 0 to 4294967295, or blank.'); return;
    }
    const controller = new AbortController(); generationRequest.current = controller;
    setGenerationError(''); playClick(520, 0.04);
    if (single) setIsStepping(true);
    else { setIsGenerating(true); setStreamingText(''); setStepDetails([]); setGenerationStats(null); setSelectedStepIndex(null); }
    try {
      const data = await requestAPI('generate', {
        prompt: formatPrompt(prompt) + (single ? streamingText : ''),
        max_new_tokens: single ? 1 : maxTokens, temperature, top_k: topK, top_p: topP,
        seed: seed === '' ? undefined : Number(seed),
      }, controller.signal);
      if (controller.signal.aborted) return;
      setBackendOnline(true);
      setGenerationStats({latency_ms:data.latency_ms,tokens_per_sec:data.tokens_per_sec,tokens_generated:data.tokens_generated});
      if (single) {
        setStreamingText(prev => prev + data.generated_text);
        setStepDetails(prev => [...prev, ...(data.step_details || []).filter((step: TokenStep) => step.chosen_char.length === 1)]);
        setIsStepping(false);
      } else {
        setStepDetails((data.step_details || []).filter((step: TokenStep) => step.chosen_char.length === 1));
        const start = performance.now();
        const streamFrame = (now: number) => {
          if (controller.signal.aborted) return;
          const count = Math.min(data.generated_text.length, Math.floor((now-start)*0.14));
          setStreamingText(data.generated_text.slice(0,count));
          if (count < data.generated_text.length) animation.current = requestAnimationFrame(streamFrame);
          else { setIsGenerating(false); animation.current = null; }
        };
        animation.current = requestAnimationFrame(streamFrame);
      }
    } catch (error) {
      if (controller.signal.aborted) return;
      setGenerationError(error instanceof Error ? error.message : 'Generation failed. Please retry.');
      setIsGenerating(false); setIsStepping(false);
    }
  };
  const handleGenerate = () => runGeneration(false);
  const handleStepToken = () => runGeneration(true);

  // Steer generation by manually clicking a candidate token
  const handleChooseCandidate = (candidate: Candidate) => {
    if (isGenerating || isStepping || candidate.char.length !== 1 || !streamingText) return;
    playClick(900, 0.04);
    const index = selectedStepIndex ?? streamingText.length - 1;
    setStreamingText(streamingText.slice(0, index) + candidate.char);
    setStepDetails(previous => previous.slice(0,index+1).map((step,i) => i === index ? {
      ...step, chosen_id: candidate.id, chosen_char: candidate.char, chosen_prob: candidate.prob,
    } : step));
    setSelectedStepIndex(index); setGenerationStats(null);
  };

  // Keyboard shortcut for generation: Cmd + Enter / Ctrl + Enter
  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') {
      e.preventDefault();
      handleGenerate();
    }
  };

  const copyToClipboard = (text: string) => {
    playClick(700, 0.03);
    navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  // Active step details for inspector (either selected token or last generated)
  const activeStep =
    selectedStepIndex !== null && stepDetails[selectedStepIndex]
      ? stepDetails[selectedStepIndex]
      : stepDetails.length > 0
      ? stepDetails[stepDetails.length - 1]
      : null;

  const currentScrubberItem = PROGRESSION_TIMELINE[scrubberIndex];

  return (
    <div className="app-shell">
      {/* 3D Specular Navier-Stokes Fluid Dynamics Canvas (Layered in Background) */}
      <FluidCanvas intensity={0.36} mode={fluidMode} />

      {/* Clean Minimalist Header */}
      <header className="app-header">
        <div className="container header-inner">
          <div className="brand-group">
            <span className="brand-mark" aria-hidden="true"><span /></span>
            <span className="brand-copy"><span className="brand-title">TinyTransformer</span><span className="brand-subtitle">MODEL LAB / V2</span></span>
            <div className={`engine-status ${backendOnline ? 'is-online' : ''}`} role="status" aria-label={backendOnline ? `${backendDevice} active` : backendOnline === null ? 'Connecting to model' : 'Model engine offline'}>
              <span className="engine-dot" />
              <span>{backendOnline ? `${backendDevice} Active` : backendOnline === null ? 'Connecting…' : 'Engine Offline'}</span>
            </div>
          </div>

          {/* Segmented Tab Navigation */}
          <nav className="nav-segmented" aria-label="Lab sections">
            <button
              className={`nav-btn ${activeTab === 'playground' ? 'active' : ''}`}
              onClick={() => setActiveTab('playground')}
            >
              <Terminal size={14} />
              Playground
            </button>
            <button
              className={`nav-btn ${activeTab === 'loss' ? 'active' : ''}`}
              onClick={() => setActiveTab('loss')}
            >
              <Activity size={14} />
              Loss & Evolution
            </button>
            <button
              className={`nav-btn ${activeTab === 'attention' ? 'active' : ''}`}
              onClick={() => setActiveTab('attention')}
            >
              <Layers size={14} />
              Attention
            </button>
            <button
              className={`nav-btn ${activeTab === 'architecture' ? 'active' : ''}`}
              onClick={() => setActiveTab('architecture')}
            >
              <Cpu size={14} />
              Architecture
            </button>
            <button
              className={`nav-btn ${activeTab === 'code' ? 'active' : ''}`}
              onClick={() => setActiveTab('code')}
            >
              <Code2 size={14} />
              Code
            </button>
          </nav>

          {/* Quick Header Controls (Fluid & Sound) */}
          <div className="header-actions">
            
            {/* Fluid Mode Selector */}
            <div style={{ display: 'flex', alignItems: 'center', gap: '4px', background: 'rgba(255, 255, 255, 0.04)', padding: '2px 6px', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border-subtle)' }}>
              <Droplets size={12} color="var(--text-tertiary)" />
              <select
                value={fluidMode}
                onChange={(e) => setFluidMode(e.target.value as FluidMode)}
                style={{
                  background: 'transparent',
                  border: 'none',
                  color: 'var(--text-secondary)',
                  fontSize: '0.72rem',
                  cursor: 'pointer',
                  outline: 'none'
                }}
              >
                <option value="cobalt" style={{ background: 'var(--bg-surface)' }}>Cobalt Ink (High Contrast)</option>
                <option value="obsidian" style={{ background: 'var(--bg-surface)' }}>Sumi Charcoal (Max Contrast)</option>
                <option value="emerald" style={{ background: 'var(--bg-surface)' }}>Emerald Jade</option>
                <option value="amethyst" style={{ background: 'var(--bg-surface)' }}>Royal Amethyst</option>
                <option value="crimson" style={{ background: 'var(--bg-surface)' }}>Tuscan Crimson</option>
                <option value="ember" style={{ background: 'var(--bg-surface)' }}>Molten Amber</option>
                <option value="mercury" style={{ background: 'var(--bg-surface)' }}>Titanium Mercury</option>
                <option value="silk" style={{ background: 'var(--bg-surface)' }}>Cashmere Silk</option>
              </select>
            </div>

            {/* Synthesized Sound Toggle */}
            <button
              className="btn-secondary"
              onClick={toggleSound}
              style={{ fontSize: '0.74rem', padding: '0.35rem 0.6rem' }}
              title={soundActive ? 'Mute Haptic Audio' : 'Enable Mechanical Sound'}
            >
              {soundActive ? <Volume2 size={13} /> : <VolumeX size={13} />}
            </button>

            {/* Explainer Guide */}
            <button
              className="btn-secondary"
              onClick={() => setShowGuideModal(true)}
              style={{ fontSize: '0.78rem', padding: '0.35rem 0.75rem' }}
            >
              <HelpCircle size={14} />
              Guide
            </button>
          </div>
        </div>
      </header>

      {/* Main Content Area */}
      <main className="container main-content">
        <section className="lab-intro" aria-labelledby="lab-title">
          <div className="intro-copy">
            <div className="eyebrow"><span className="eyebrow-line" /> AN OPEN MODEL EXPERIMENT</div>
            <h1 id="lab-title">See every token <em>take shape.</em></h1>
            <p>Write a prompt, generate a response, and inspect the tokens, attention, and training behind every answer.</p>
            <div className="intro-actions">
              <button className="intro-primary" onClick={() => { setActiveTab('playground'); window.setTimeout(() => document.getElementById('prompt-input')?.focus(), 0); }}><Play size={15} fill="currentColor" /> Try the playground</button>
              <button className="intro-link" onClick={() => setActiveTab('architecture')}>Explore the model <span aria-hidden="true">↗</span></button>
            </div>
          </div>
          <div className="intro-visual" aria-label="Model specification summary">
            <div className="visual-head"><span className="live-indicator" /> LIVE INFERENCE <span>001 / TINY</span></div>
            <div className="model-orbit" aria-hidden="true"><span className="orbit orbit-one" /><span className="orbit orbit-two" /><span className="orbit-core">T</span><span className="orbit-node node-one" /><span className="orbit-node node-two" /><span className="orbit-node node-three" /></div>
            <div className="visual-stats"><div><strong>{(summary.parameters / 1e6).toFixed(2)}M</strong><span>PARAMETERS</span></div><div><strong>4 × 4</strong><span>LAYERS / HEADS</span></div><div><strong>{summary.block_size}</strong><span>CHAR CONTEXT</span></div></div>
          </div>
        </section>
        <div className="section-heading"><div><span className="section-kicker">{activeTab === 'playground' ? '01 / INTERACT' : activeTab === 'loss' ? '02 / LEARN' : activeTab === 'attention' ? '03 / INSPECT' : activeTab === 'architecture' ? '04 / UNDERSTAND' : '05 / SOURCE'}</span><h2>{activeTab === 'playground' ? 'The playground' : activeTab === 'loss' ? 'Training evolution' : activeTab === 'attention' ? 'Attention explorer' : activeTab === 'architecture' ? 'Inside the model' : 'Under the hood'}</h2></div><span className="section-aside">BUILT FROM SCRATCH · RUNNING ON CPU</span></div>
        
        {/* ====================================================================
            TAB 1: PLAYGROUND (STEPPING, PROBABILITIES, STEERING)
            ==================================================================== */}
        <p className="model-notice"><span aria-hidden="true">✳</span> A small model trained on a narrow teaching curriculum. Its answers can be wrong; token confidence is not factual certainty.</p>
        {activeTab === 'playground' && (
          <div className="tab-pane">
            <div className="two-col-grid">
              
              {/* Left Column: Prompt, Stepping, Output & Candidate Inspector */}
              <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
                
                {/* Prompt Suggestions */}
                <div className="prompt-suggestions">
                  <div className="panel-kicker">START WITH A QUESTION</div>
                  <div className="prompt-pills-row">
                    {QUICK_PROMPTS.map((p, idx) => (
                      <button
                        key={idx}
                        disabled={isGenerating || isStepping}
                        className="prompt-pill"
                        onClick={() => {
                          playClick(600, 0.02);
                          setPrompt(p.text);
                        }}
                      >
                        {p.label}
                      </button>
                    ))}
                  </div>
                </div>

                <div className="card topic-card">
                  <label htmlFor="curriculum-topic" style={{display:'block',marginBottom:'.5rem',fontSize:'.8rem'}}>Explore the {curriculumTopics.length} training topics</label>
                  <select id="curriculum-topic" className="textarea-clean" style={{minHeight:'auto'}} value="" disabled={isGenerating || isStepping}
                    onChange={e=>{if(e.target.value) {setPrompt(formatPrompt(e.target.value));setStreamingText('');setStepDetails([]);setGenerationStats(null);setSelectedStepIndex(null);}}}>
                    <option value="">Choose a question…</option>
                    {curriculumTopics.map(item=><option key={item.topic} value={item.question}>{item.question}</option>)}
                  </select>
                </div>
                {/* Prompt Input Box */}
                <div className="card prompt-card">
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '0.6rem' }}>
                    <label htmlFor="prompt-input" className="panel-title">Your prompt</label>
                    <span style={{ fontSize: '0.72rem', color: 'var(--text-tertiary)' }}>
                      Press Ctrl / ⌘ + Enter
                    </span>
                  </div>
                  <textarea
                    id="prompt-input"
                    aria-label="Input prompt"
                    maxLength={4096}
                    disabled={isGenerating || isStepping}
                    className="textarea-clean"
                    value={prompt}
                    onChange={(e) => setPrompt(e.target.value)}
                    onKeyDown={handleKeyDown}
                    rows={4}
                    placeholder="User: Ask a question...&#10;Assistant: "
                  />
                  <div style={{fontSize:'.74rem',color:'var(--text-tertiary)',marginTop:'.5rem'}}>
                    {Array.from(formatPrompt(prompt)).length} characters · {summary.block_size}-character context
                    {Array.from(formatPrompt(prompt)).length > summary.block_size && ' · Older context will be cropped'}
                  </div>
                  <div className="prompt-actions">
                    <button
                      className="btn-secondary"
                      onClick={() => {
                        playClick(440, 0.03);
                        stopGeneration(); setGenerationError(''); setGenerationStats(null); setSelectedStepIndex(null);
                        setPrompt('');
                        setStreamingText('');
                        setStepDetails([]);
                      }}
                      style={{ fontSize: '0.78rem', padding: '0.35rem 0.65rem' }}
                    >
                      <RotateCcw size={13} />
                      Clear
                    </button>

                    <div style={{ display: 'flex', gap: '0.5rem' }}>
                      {(isGenerating || isStepping) && <button className="btn-secondary" onClick={stopGeneration}>Stop</button>}
                      {/* Step Single Token Button */}
                      <button
                        className="btn-secondary"
                        onClick={handleStepToken}
                        disabled={isGenerating || isStepping || !prompt.trim()}
                        style={{ fontSize: '0.82rem' }}
                        title="Generate exactly 1 token and inspect its candidate probabilities"
                      >
                        <StepForward size={14} />
                        {isStepping ? 'Stepping...' : 'Step Token'}
                      </button>

                      {/* Full Generate Button */}
                      <button
                        className="btn-primary"
                        onClick={handleGenerate}
                        disabled={isGenerating || isStepping || !prompt.trim()}
                      >
                        <Play size={14} fill="currentColor" />
                        {isGenerating ? 'Generating...' : 'Generate'}
                      </button>
                    </div>
                  </div>
                </div>

                {generationError && <div className="card" role="alert" style={{color:'#fca5a5'}}>{generationError}</div>}
                {/* Output Panel with Confidence View Toggle */}
                <div className="card response-card">
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '0.75rem' }}>
                    <div className="response-heading">
                      <span className="panel-title">Model response</span>

                      {/* Text vs Confidence Mode Switcher */}
                      {streamingText && (
                        <div style={{ display: 'flex', gap: '2px', background: 'rgba(255,255,255,0.04)', padding: '2px', borderRadius: 'var(--radius-xs)' }}>
                          <button
                            onClick={() => setViewMode('text')}
                            style={{
                              background: viewMode === 'text' ? 'rgba(255,255,255,0.1)' : 'transparent',
                              border: 'none',
                              color: viewMode === 'text' ? 'var(--text-primary)' : 'var(--text-tertiary)',
                              fontSize: '0.7rem',
                              padding: '2px 6px',
                              borderRadius: '3px',
                              cursor: 'pointer'
                            }}
                          >
                            Text
                          </button>
                          <button
                            onClick={() => setViewMode('confidence')}
                            style={{
                              background: viewMode === 'confidence' ? 'rgba(255,255,255,0.1)' : 'transparent',
                              border: 'none',
                              color: viewMode === 'confidence' ? 'var(--text-primary)' : 'var(--text-tertiary)',
                              fontSize: '0.7rem',
                              padding: '2px 6px',
                              borderRadius: '3px',
                              cursor: 'pointer',
                              display: 'flex',
                              alignItems: 'center',
                              gap: '3px'
                            }}
                          >
                            <Eye size={10} />
                            Confidence Map
                          </button>
                        </div>
                      )}
                    </div>

                    {streamingText && (
                      <button
                        className="btn-secondary"
                        onClick={() => copyToClipboard(prompt + streamingText)}
                        style={{ fontSize: '0.74rem', padding: '0.25rem 0.55rem' }}
                      >
                        {copied ? <Check size={12} /> : <Copy size={12} />}
                        {copied ? 'Copied' : 'Copy'}
                      </button>
                    )}
                  </div>

                  {/* Output Box */}
                  <div className="output-box" role="region" aria-label="Generated response" aria-busy={isGenerating || isStepping}>
                    <span style={{ color: 'var(--text-secondary)' }}>{formatPrompt(prompt)}</span>

                    {viewMode === 'text' ? (
                      <span style={{ color: 'var(--text-primary)', fontWeight: 500 }}>
                        {streamingText}
                      </span>
                    ) : (
                      /* Interactive Token Confidence View */
                      <span>
                        {streamingText.split('').map((char, cIdx) => {
                          const step = stepDetails[cIdx];
                          const prob = step?.chosen_prob ?? 0;
                          const isSelected = selectedStepIndex === cIdx;

                          // Color-code by probability confidence
                          let bg = 'rgba(16, 185, 129, 0.15)'; // high (> 75%)
                          let border = '1px solid rgba(16, 185, 129, 0.3)';
                          if (prob < 0.45) {
                            bg = 'rgba(244, 63, 94, 0.2)'; // low (< 45%)
                            border = '1px solid rgba(244, 63, 94, 0.4)';
                          } else if (prob < 0.75) {
                            bg = 'rgba(245, 158, 11, 0.18)'; // moderate (45-75%)
                            border = '1px solid rgba(245, 158, 11, 0.35)';
                          }

                          return (
                            <span
                              key={cIdx}
                              onClick={() => {
                                playClick(750, 0.02);
                                setSelectedStepIndex(cIdx);
                              }}
                              style={{
                                background: isSelected ? 'rgba(255, 255, 255, 0.35)' : bg,
                                border: isSelected ? '1px solid #ffffff' : border,
                                padding: '1px 2px',
                                margin: '0 1px',
                                borderRadius: '3px',
                                cursor: 'pointer',
                                transition: 'all 0.1s ease',
                                display: 'inline-block'
                              }}
                              title={`Token '${char === ' ' ? '␣' : char}' · Prob: ${(prob * 100).toFixed(1)}%`}
                            >
                              {char === ' ' ? '␣' : char}
                            </span>
                          );
                        })}
                      </span>
                    )}

                    {(isGenerating || isStepping) && <span className="output-cursor" />}
                    {!streamingText && !isGenerating && !isStepping && (
                      <span style={{ color: 'var(--text-tertiary)', fontStyle: 'italic' }}>
                        Generated text will appear here...
                      </span>
                    )}
                  </div>

                  {generationStats && (
                    <div className="stats-bar" style={{ marginTop: '0.85rem' }}>
                      <span>Tokens: {generationStats.tokens_generated}</span>
                      <span>Latency: {generationStats.latency_ms} ms</span>
                      <span>Speed: {generationStats.tokens_per_sec} tok/s</span>
                    </div>
                  )}

                  {/* Token Probabilities Inspector & Steering Deck */}
                  {activeStep && activeStep.top_candidates && (
                    <div className="logit-drawer">
                      <div
                        className="logit-drawer-header"
                        onClick={() => setShowLogitDrawer(!showLogitDrawer)}
                      >
                        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                          <span style={{ fontWeight: 600 }}>
                            {selectedStepIndex !== null
                              ? `Token #${selectedStepIndex + 1} Probabilities ('${activeStep.chosen_char === ' ' ? '␣' : activeStep.chosen_char}')`
                              : `Candidates at this step ('${activeStep.chosen_char === ' ' ? '␣' : activeStep.chosen_char}')`}
                          </span>
                          <span style={{ fontSize: '0.72rem', color: 'var(--text-tertiary)' }}>
                            (Click any candidate to steer generation)
                          </span>
                        </div>
                        {showLogitDrawer ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
                      </div>

                      {showLogitDrawer && (
                        <div style={{ padding: '0.5rem 0 0.75rem' }}>
                          {activeStep.top_candidates.slice(0, 5).map((cand: Candidate, idx: number) => {
                            const pct = Math.round((cand.prob || 0) * 100);
                            return (
                              <div
                                key={idx}
                                className="logit-bar-row"
                                style={{ cursor: 'pointer' }}
                                onClick={() => handleChooseCandidate(cand)}
                                title={`Click to steer text with '${cand.display || cand.char}'`}
                              >
                                <span className="logit-token-label">
                                  {cand.display || cand.char}
                                </span>
                                <div className="logit-bar-track">
                                  <div
                                    className="logit-bar-fill"
                                    style={{ width: `${pct}%` }}
                                  />
                                </div>
                                <span className="logit-percent">{pct}%</span>
                              </div>
                            );
                          })}
                        </div>
                      )}
                    </div>
                  )}

                </div>

              </div>

              {/* Right Column: Clean Settings Deck */}
              <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
                <div className="card settings-card">
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '1rem' }}>
                    <Sliders size={15} color="var(--text-secondary)" />
                    <span style={{ fontSize: '0.84rem', fontWeight: 600, color: 'var(--text-primary)' }}>
                      Sampling Controls
                    </span>
                  </div>

                  <div className="slider-deck">
                    
                    {/* Temperature */}
                    <div className="slider-group">
                      <div className="slider-header">
                        <span className="slider-label">Temperature</span>
                        <span className="slider-value">{temperature.toFixed(2)}</span>
                      </div>
                      <input
                        type="range"
                        className="slider-clean"
                        min="0"
                        max="1.5"
                        step="0.05"
                        aria-label="Temperature"
                        value={temperature}
                        onChange={(e) => setTemperature(parseFloat(e.target.value))}
                      />
                      <span style={{ fontSize: '0.72rem', color: 'var(--text-tertiary)' }}>
                        Lower for deterministic facts, higher for variety.
                      </span>
                    </div>

                    {/* Top-K */}
                    <div className="slider-group">
                      <div className="slider-header">
                        <span className="slider-label">Top-K Cutoff</span>
                        <span className="slider-value">{topK}</span>
                      </div>
                      <input
                        type="range"
                        className="slider-clean"
                        min="1"
                        max="50"
                        step="1"
                        aria-label="Top K"
                        value={topK}
                        onChange={(e) => setTopK(parseInt(e.target.value))}
                      />
                      <span style={{ fontSize: '0.72rem', color: 'var(--text-tertiary)' }}>
                        Restrict to top K most probable tokens.
                      </span>
                    </div>

                    {/* Top-P */}
                    <div className="slider-group">
                      <div className="slider-header">
                        <span className="slider-label">Top-P (Nucleus)</span>
                        <span className="slider-value">{topP.toFixed(2)}</span>
                      </div>
                      <input
                        type="range"
                        className="slider-clean"
                        min="0.1"
                        max="1.0"
                        step="0.05"
                        aria-label="Top P"
                        value={topP}
                        onChange={(e) => setTopP(parseFloat(e.target.value))}
                      />
                      <span style={{ fontSize: '0.72rem', color: 'var(--text-tertiary)' }}>
                        Cumulative probability threshold.
                      </span>
                    </div>

                    {/* Max Tokens */}
                    <div className="slider-group">
                      <div className="slider-header">
                        <span className="slider-label">Sequence Length</span>
                        <span className="slider-value">{maxTokens}</span>
                      </div>
                      <input
                        type="range"
                        className="slider-clean"
                        min="20"
                        max="240"
                        step="10"
                        aria-label="Maximum tokens"
                        value={maxTokens}
                        onChange={(e) => setMaxTokens(parseInt(e.target.value))}
                      />
                      <span style={{ fontSize: '0.72rem', color: 'var(--text-tertiary)' }}>
                        Maximum characters to generate.
                      </span>
                    </div>

                    <div style={{ paddingTop: '0.5rem', borderTop: '1px solid var(--border-subtle)' }}>
                      <button
                        style={{
                          background: 'none',
                          border: 'none',
                          color: 'var(--text-tertiary)',
                          fontSize: '0.74rem',
                          cursor: 'pointer',
                          textDecoration: 'underline'
                        }}
                        onClick={() => {
                          setTemperature(0.35);
                          setTopK(10);
                          setTopP(0.90);
                          setMaxTokens(220);
                        }}
                      >
                        Reset to recommended defaults
                      </button>
                    </div>

                  </div>
                </div>

                <div className="card">
                  <label htmlFor="sampling-seed" style={{display:'block',marginBottom:'.5rem'}}>Sampling seed</label>
                  <input id="sampling-seed" className="textarea-clean" style={{minHeight:'auto'}} value={seed} onChange={e=>setSeed(e.target.value)} inputMode="numeric" placeholder="Blank for random" />
                  <p style={{fontSize:'.75rem',color:'var(--text-tertiary)',marginTop:'.5rem'}}>Repeat a prompt with the same settings and seed to reproduce a response on the same engine.</p>
                </div>
                {/* Model Info Card */}
                <div className="card">
                  <div style={{ fontSize: '0.78rem', fontWeight: 600, color: 'var(--text-secondary)', marginBottom: '0.75rem' }}>
                    Model Specification
                  </div>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem', fontSize: '0.76rem' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                      <span style={{ color: 'var(--text-tertiary)' }}>Architecture</span>
                      <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--text-primary)' }}>Causal Decoder</span>
                    </div>
                    <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                      <span style={{ color: 'var(--text-tertiary)' }}>Parameters</span>
                      <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--text-primary)' }}>{summary.parameters.toLocaleString()}</span>
                    </div>
                    <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                      <span style={{ color: 'var(--text-tertiary)' }}>Layers & Heads</span>
                      <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--text-primary)' }}>4 Layers · 4 Heads</span>
                    </div>
                    <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                      <span style={{ color: 'var(--text-tertiary)' }}>Context Window</span>
                      <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--text-primary)' }}>{summary.block_size} Characters</span>
                    </div>
                    <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                      <span style={{ color: 'var(--text-tertiary)' }}>Hardware</span>
                      <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--text-primary)' }}>{summary.device.toUpperCase()} training</span>
                    </div>
                  </div>
                </div>

              </div>

            </div>
          </div>
        )}

        {/* ====================================================================
            TAB 2: LOSS CURVES & TRAINING EVOLUTION (INTERACTIVE SCRUBBER)
            ==================================================================== */}
        {activeTab === 'loss' && (
          <div className="tab-pane">
            <div className="card" style={{marginBottom:'1.25rem'}}>
              <h3>Held-out phrasing evaluation</h3>
              <p>Greedy answer exact match: {generationBenchmark.held_out.exact_matches}/{generationBenchmark.held_out.total_prompts} on held-out formats; {generationBenchmark.canonical.exact_matches}/{generationBenchmark.canonical.total_prompts} on canonical training questions.</p>
              <p style={{fontSize:'.82rem',color:'var(--text-secondary)',lineHeight:1.7}}>
                Answer-token loss: {trainingReport.evaluation.baseline?.loss.toFixed(3)} (previous model) → {trainingReport.evaluation.candidate.loss.toFixed(3)} (current model).
                The underlying facts appear in training; this tests new question phrasings, not new-topic knowledge. Low loss does not guarantee a correct generated answer.
              </p>
            </div>
            {/* KPI Cards */}
            <div className="kpi-grid">
              <div className="kpi-card">
                <div className="kpi-label">Best Validation Loss</div>
                <div className="kpi-value">{summary.best_val_loss}</div>
                <div style={{ fontSize: '0.72rem', color: '#10b981', marginTop: '0.2rem' }}>
                  Held-out question phrasings
                </div>
              </div>
              <div className="kpi-card">
                <div className="kpi-label">Validation Perplexity</div>
                <div className="kpi-value">{Math.exp(summary.best_val_loss).toFixed(2)}</div>
                <div style={{ fontSize: '0.72rem', color: 'var(--text-tertiary)', marginTop: '0.2rem' }}>
                  exp(validation loss)
                </div>
              </div>
              <div className="kpi-card">
                <div className="kpi-label">Trainable Parameters</div>
                <div className="kpi-value">{summary.parameters.toLocaleString()}</div>
                <div style={{ fontSize: '0.72rem', color: 'var(--text-tertiary)', marginTop: '0.2rem' }}>
                  Float32 weights & biases
                </div>
              </div>
              <div className="kpi-card">
                <div className="kpi-label">Refinement Steps</div>
                <div className="kpi-value">{summary.total_steps.toLocaleString()}</div>
                <div style={{ fontSize: '0.72rem', color: 'var(--text-tertiary)', marginTop: '0.2rem' }}>
                  Cosine learning rate schedule
                </div>
              </div>
            </div>

            {/* Interactive Timeline Scrubber (Emergence of Intelligence) */}
            <div className="card" style={{ marginBottom: '1.5rem' }}>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '0.85rem' }}>
                <div>
                  <h3 style={{ fontSize: '0.92rem', fontWeight: 600, color: 'var(--text-primary)' }}>
                    Interactive Training Evolution Scrubber
                  </h3>
                  <p style={{ fontSize: '0.76rem', color: 'var(--text-tertiary)' }}>
                    Drag the slider to observe how the transformer acquired grammatical and reasoning capabilities across 2,200 steps
                  </p>
                </div>
                <div style={{ fontFamily: 'var(--font-mono)', fontSize: '0.85rem', color: 'var(--text-primary)' }}>
                  Step {currentScrubberItem.step} / {summary.total_steps}
                </div>
              </div>

              {/* Scrubber Slider */}
              <input
                type="range"
                className="slider-clean"
                min="0"
                max={PROGRESSION_TIMELINE.length - 1}
                step="1"
                aria-label="Training step"
                        value={scrubberIndex}
                onChange={(e) => {
                  playClick(500 + parseInt(e.target.value) * 80, 0.02);
                  setScrubberIndex(parseInt(e.target.value));
                }}
                style={{ marginBottom: '1rem' }}
              />

              {/* Scrubber Detail Card */}
              <div style={{ background: 'var(--bg-surface-subtle)', border: '1px solid var(--border-subtle)', borderRadius: 'var(--radius-md)', padding: '1.1rem' }}>
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '0.5rem', flexWrap: 'wrap', gap: '0.5rem' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
                    <span style={{ fontSize: '0.86rem', fontWeight: 600, color: 'var(--text-primary)' }}>
                      {currentScrubberItem.phase}
                    </span>
                    <span className="badge-tag">Loss: {currentScrubberItem.loss}</span>
                    <span className="badge-tag">PPL: {currentScrubberItem.perplexity}</span>
                  </div>
                </div>

                <p style={{ fontSize: '0.78rem', color: 'var(--text-secondary)', marginBottom: '0.75rem', lineHeight: 1.6 }}>
                  {currentScrubberItem.description}
                </p>

                <div style={{
                  background: 'rgba(255, 255, 255, 0.02)',
                  border: '1px solid var(--border-subtle)',
                  borderRadius: 'var(--radius-sm)',
                  padding: '0.75rem',
                  fontFamily: 'var(--font-mono)',
                  fontSize: '0.76rem',
                  color: 'var(--text-primary)',
                  lineHeight: 1.6
                }}>
                  {currentScrubberItem.sample}
                </div>
              </div>
            </div>

            {/* Loss Chart */}
            <div className="card">
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1rem' }}>
                <div>
                  <h3 style={{ fontSize: '0.92rem', fontWeight: 600, color: 'var(--text-primary)' }}>
                    Convergence Trajectory (Train Loss vs. Validation Loss)
                  </h3>
                  <p style={{ fontSize: '0.76rem', color: 'var(--text-tertiary)' }}>
                    Measured answer-token loss during refinement
                  </p>
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '1.25rem', fontSize: '0.75rem' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                    <span style={{ width: '10px', height: '2px', background: '#3b82f6', display: 'inline-block' }} />
                    <span style={{ color: 'var(--text-secondary)' }}>Train Loss</span>
                  </div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                    <span style={{ width: '10px', height: '2px', background: '#10b981', display: 'inline-block' }} />
                    <span style={{ color: 'var(--text-secondary)' }}>Validation Loss</span>
                  </div>
                </div>
              </div>

              {/* Minimal SVG Chart */}
              <div style={{ width: '100%', height: '260px', position: 'relative' }}>
                <svg width="100%" height="100%" viewBox="0 0 800 240" preserveAspectRatio="none">
                  <line x1="40" y1="20" x2="780" y2="20" stroke="rgba(255,255,255,0.06)" strokeDasharray="3 3" />
                  <line x1="40" y1="80" x2="780" y2="80" stroke="rgba(255,255,255,0.06)" strokeDasharray="3 3" />
                  <line x1="40" y1="140" x2="780" y2="140" stroke="rgba(255,255,255,0.06)" strokeDasharray="3 3" />
                  <line x1="40" y1="200" x2="780" y2="200" stroke="rgba(255,255,255,0.12)" />

                  <text x="32" y="24" fill="#71717a" fontSize="10" textAnchor="end" fontFamily="monospace">{chartMax}</text>
                  <text x="32" y="84" fill="#71717a" fontSize="10" textAnchor="end" fontFamily="monospace">{(chartMax*2/3).toFixed(1)}</text>
                  <text x="32" y="144" fill="#71717a" fontSize="10" textAnchor="end" fontFamily="monospace">{(chartMax/3).toFixed(1)}</text>
                  <text x="32" y="204" fill="#71717a" fontSize="10" textAnchor="end" fontFamily="monospace">0.0</text>

                  {/* Train Curve */}
                  <path
                    d={lossPath('train_loss')}
                    fill="none"
                    stroke="#3b82f6"
                    strokeWidth="2.5"
                  />
                  {/* Val Curve */}
                  <path
                    d={lossPath('val_loss')}
                    fill="none"
                    stroke="#10b981"
                    strokeWidth="2.5"
                  />
                </svg>
              </div>
            </div>

          </div>
        )}

        {/* ====================================================================
            TAB 3: ATTENTION MATRIX (CLEAN HEATMAP)
            ==================================================================== */}
        {activeTab === 'attention' && (
          <div className="tab-pane">
            <div className="card" style={{ marginBottom: '1.5rem' }}>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1rem', flexWrap: 'wrap', gap: '1rem' }}>
                <div>
                  <h3 style={{ fontSize: '0.95rem', fontWeight: 600, color: 'var(--text-primary)' }}>
                    Multi-Head Causal Self-Attention Inspector
                  </h3>
                  <p style={{ fontSize: '0.76rem', color: 'var(--text-tertiary)' }}>
                    Live Query &times; Key dot-product affinity matrix across 4 layers and 4 heads
                  </p>
                </div>

                <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
                  {/* Layer Select */}
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                    <span style={{ fontSize: '0.75rem', color: 'var(--text-tertiary)' }}>Layer:</span>
                    <div style={{ display: 'flex', gap: '3px' }}>
                      {[0, 1, 2, 3].map((l) => (
                        <button
                          key={l}
                          className={`nav-btn ${selectedLayer === l ? 'active' : ''}`}
                          style={{ padding: '0.2rem 0.5rem', fontSize: '0.75rem' }}
                          onClick={() => {
                            playClick(600 + l * 50, 0.02);
                            setSelectedLayer(l);
                          }}
                        >
                          L{l + 1}
                        </button>
                      ))}
                    </div>
                  </div>

                  {/* Head Select */}
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                    <span style={{ fontSize: '0.75rem', color: 'var(--text-tertiary)' }}>Head:</span>
                    <div style={{ display: 'flex', gap: '3px' }}>
                      {[0, 1, 2, 3].map((h) => (
                        <button
                          key={h}
                          className={`nav-btn ${selectedHead === h ? 'active' : ''}`}
                          style={{ padding: '0.2rem 0.5rem', fontSize: '0.75rem' }}
                          onClick={() => {
                            playClick(700 + h * 50, 0.02);
                            setSelectedHead(h);
                          }}
                        >
                          H{h + 1}
                        </button>
                      ))}
                    </div>
                  </div>
                </div>
              </div>

              {/* Text to analyze */}
              <div style={{ display: 'flex', gap: '0.5rem', marginBottom: '1.25rem' }}>
                <input
                  type="text"
                  className="textarea-clean"
                  style={{ minHeight: 'auto', padding: '0.5rem 0.85rem' }}
                  aria-label="Text to analyze for attention"
                  maxLength={32}
                  value={attnInput}
                  onChange={(e) => setAttnInput(e.target.value)}
                  placeholder="Text to analyze..."
                />
                <button
                  className="btn-secondary"
                  onClick={() => {
                    playClick(650, 0.03);
                    fetchAttention(attnInput);
                  }}
                  disabled={isAttnLoading}
                >
                  {isAttnLoading ? 'Analyzing...' : 'Analyze'}
                </button>
              </div>

              {attentionError && <p role="alert" style={{color:'#fca5a5'}}>{attentionError}</p>}
              {isAttnLoading && <p role="status">Computing attention from the trained model…</p>}
              {/* Heatmap Grid */}
              {attnData && (
                <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', overflowX: 'auto', padding: '1rem 0' }}>
                  <div style={{ display: 'inline-block' }}>
                    {/* Header Row */}
                    <div style={{ display: 'flex', marginLeft: '32px' }}>
                      {attnData.tokens.map((t: string, colIdx: number) => (
                        <div
                          key={colIdx}
                          style={{
                            width: '28px',
                            height: '24px',
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                            fontSize: '0.72rem',
                            fontFamily: 'var(--font-mono)',
                            color: 'var(--text-tertiary)'
                          }}
                        >
                          {t}
                        </div>
                      ))}
                    </div>

                    {/* Matrix Rows */}
                    {attnData.tokens.map((rowToken: string, rIdx: number) => (
                      <div key={rIdx} style={{ display: 'flex', alignItems: 'center' }}>
                        <div
                          style={{
                            width: '32px',
                            fontSize: '0.72rem',
                            fontFamily: 'var(--font-mono)',
                            color: 'var(--text-tertiary)',
                            textAlign: 'right',
                            paddingRight: '6px'
                          }}
                        >
                          {rowToken}
                        </div>
                        {attnData.tokens.map((colToken: string, cIdx: number) => {
                          const weight =
                            attnData.weights?.[selectedLayer]?.[selectedHead]?.[rIdx]?.[cIdx] || 0;
                          const isCausalMasked = cIdx > rIdx;
                          const opacity = isCausalMasked ? 0.03 : Math.max(0.08, weight);

                          return (
                            <div
                              key={cIdx}
                              onMouseEnter={() => {
                                if (!isCausalMasked) playClick(900 + weight * 200, 0.015);
                                setHoveredCell({
                                  qToken: rowToken,
                                  kToken: colToken,
                                  score: weight
                                });
                              }}
                              onMouseLeave={() => setHoveredCell(null)}
                              style={{
                                width: '28px',
                                height: '28px',
                                margin: '1px',
                                borderRadius: '2px',
                                backgroundColor: isCausalMasked
                                  ? '#18181b'
                                  : `rgba(255, 255, 255, ${opacity})`,
                                border: isCausalMasked
                                  ? '1px solid rgba(255,255,255,0.02)'
                                  : '1px solid rgba(255, 255, 255, 0.1)',
                                cursor: 'crosshair',
                                transition: 'all 0.1s ease'
                              }}
                            />
                          );
                        })}
                      </div>
                    ))}
                  </div>

                  {/* Hover Cell Inspector */}
                  <div style={{ marginTop: '1rem', height: '22px', fontSize: '0.78rem', fontFamily: 'var(--font-mono)' }}>
                    {hoveredCell ? (
                      <span style={{ color: 'var(--text-primary)' }}>
                        Query: '{hoveredCell.qToken}' &rarr; Key: '{hoveredCell.kToken}' &bull; Weight:{' '}
                        {(hoveredCell.score * 100).toFixed(1)}%
                      </span>
                    ) : (
                      <span style={{ color: 'var(--text-tertiary)' }}>
                        Hover over any grid cell to inspect attention affinity
                      </span>
                    )}
                  </div>
                </div>
              )}

            </div>
          </div>
        )}

        {/* ====================================================================
            TAB 4: ARCHITECTURE & SPECS (CLEAN REFERENCE)
            ==================================================================== */}
        {activeTab === 'architecture' && (
          <div className="tab-pane">
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: '1.5rem' }}>
              
              <div className="card">
                <h3 style={{ fontSize: '0.92rem', fontWeight: 600, color: 'var(--text-primary)', marginBottom: '0.85rem' }}>
                  Structural Specifications
                </h3>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem', fontSize: '0.8rem' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid var(--border-subtle)', paddingBottom: '0.4rem' }}>
                    <span style={{ color: 'var(--text-secondary)' }}>Model Type</span>
                    <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--text-primary)' }}>Causal Decoder-Only</span>
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid var(--border-subtle)', paddingBottom: '0.4rem' }}>
                    <span style={{ color: 'var(--text-secondary)' }}>Total Parameters</span>
                    <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--text-primary)' }}>{summary.parameters.toLocaleString()}</span>
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid var(--border-subtle)', paddingBottom: '0.4rem' }}>
                    <span style={{ color: 'var(--text-secondary)' }}>Transformer Blocks</span>
                    <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--text-primary)' }}>4 Layers</span>
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid var(--border-subtle)', paddingBottom: '0.4rem' }}>
                    <span style={{ color: 'var(--text-secondary)' }}>Attention Heads</span>
                    <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--text-primary)' }}>4 Heads (head_dim = {summary.d_model / 4})</span>
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid var(--border-subtle)', paddingBottom: '0.4rem' }}>
                    <span style={{ color: 'var(--text-secondary)' }}>Model Dimension (d_model)</span>
                    <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--text-primary)' }}>{summary.d_model}</span>
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid var(--border-subtle)', paddingBottom: '0.4rem' }}>
                    <span style={{ color: 'var(--text-secondary)' }}>Feed-Forward Expansion (d_mlp)</span>
                    <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--text-primary)' }}>{summary.d_model * 4} (4x d_model)</span>
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid var(--border-subtle)', paddingBottom: '0.4rem' }}>
                    <span style={{ color: 'var(--text-secondary)' }}>Context Window (block_size)</span>
                    <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--text-primary)' }}>{summary.block_size} Characters</span>
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid var(--border-subtle)', paddingBottom: '0.4rem' }}>
                    <span style={{ color: 'var(--text-secondary)' }}>Vocabulary Size</span>
                    <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--text-primary)' }}>{summary.vocab_size} Tokens</span>
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                    <span style={{ color: 'var(--text-secondary)' }}>Activation & Norm</span>
                    <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--text-primary)' }}>GELU · Pre-LayerNorm</span>
                  </div>
                </div>
              </div>

              <div className="card">
                <h3 style={{ fontSize: '0.92rem', fontWeight: 600, color: 'var(--text-primary)', marginBottom: '0.85rem' }}>
                  Mathematical Formulation
                </h3>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '0.85rem', fontSize: '0.8rem', color: 'var(--text-secondary)', lineHeight: 1.6 }}>
                  <div>
                    <div style={{ fontWeight: 600, color: 'var(--text-primary)', marginBottom: '0.25rem' }}>
                      1. Scaled Dot-Product Attention
                    </div>
                    <div style={{ background: '#0d0d10', padding: '0.65rem', borderRadius: 'var(--radius-sm)', fontFamily: 'var(--font-mono)', fontSize: '0.76rem', color: 'var(--text-primary)' }}>
                      Attention(Q, K, V) = softmax((Q · K^T) / √d_k + M) · V
                    </div>
                  </div>
                  <div>
                    <div style={{ fontWeight: 600, color: 'var(--text-primary)', marginBottom: '0.25rem' }}>
                      2. Causal Autoregressive Mask
                    </div>
                    <div style={{ background: '#0d0d10', padding: '0.65rem', borderRadius: 'var(--radius-sm)', fontFamily: 'var(--font-mono)', fontSize: '0.76rem', color: 'var(--text-primary)' }}>
                      M_ij = 0 if j &le; i, else -inf
                    </div>
                  </div>
                  <div>
                    <div style={{ fontWeight: 600, color: 'var(--text-primary)', marginBottom: '0.25rem' }}>
                      3. Pre-LayerNorm Residual Block
                    </div>
                    <div style={{ background: '#0d0d10', padding: '0.65rem', borderRadius: 'var(--radius-sm)', fontFamily: 'var(--font-mono)', fontSize: '0.76rem', color: 'var(--text-primary)' }}>
                      x = x + Attention(LayerNorm(x))<br />
                      x = x + MLP(LayerNorm(x))
                    </div>
                  </div>
                </div>
              </div>

            </div>
          </div>
        )}

        {/* ====================================================================
            TAB 5: CODEBASE (CLEAN CODE VIEWER)
            ==================================================================== */}
        {activeTab === 'code' && (
          <div className="tab-pane">
            <div className="card">
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1rem' }}>
                <div style={{ display: 'flex', gap: '4px' }}>
                  <button
                    className={`nav-btn ${selectedCodeFile === 'model' ? 'active' : ''}`}
                    onClick={() => {
                      playClick(600, 0.02);
                      setSelectedCodeFile('model');
                    }}
                  >
                    model.py
                  </button>
                  <button
                    className={`nav-btn ${selectedCodeFile === 'train' ? 'active' : ''}`}
                    onClick={() => {
                      playClick(650, 0.02);
                      setSelectedCodeFile('train');
                    }}
                  >
                    train.py
                  </button>
                </div>

                <button
                  className="btn-secondary"
                  onClick={() => {
                    const code = selectedCodeFile === 'model' ? MODEL_CODE : TRAIN_CODE;
                    navigator.clipboard.writeText(code);
                    setCodeCopied(true);
                    playClick(750, 0.03);
                    setTimeout(() => setCodeCopied(false), 2000);
                  }}
                  style={{ fontSize: '0.76rem' }}
                >
                  {codeCopied ? <Check size={13} /> : <Copy size={13} />}
                  {codeCopied ? 'Copied' : 'Copy File'}
                </button>
              </div>

              <div style={{
                background: 'var(--code-bg)',
                border: '1px solid var(--border-subtle)',
                borderRadius: 'var(--radius-md)',
                padding: '1.25rem',
                maxHeight: '560px',
                overflowY: 'auto'
              }}>
                <pre style={{
                  fontFamily: 'var(--font-mono)',
                  fontSize: '0.78rem',
                  lineHeight: 1.6,
                  color: 'var(--text-primary)',
                  margin: 0
                }}>
                  <code>{selectedCodeFile === 'model' ? MODEL_CODE : TRAIN_CODE}</code>
                </pre>
              </div>
            </div>
          </div>
        )}

      </main>

      {/* Guided Explainer Modal */}
      <GuidedExplainer isOpen={showGuideModal} onClose={() => setShowGuideModal(false)} />
    </div>
  );
}
