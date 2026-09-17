from pathlib import Path
p=Path('frontend/src/App.tsx');s=p.read_text().replace('useCallback }','useCallback, useRef }')
s=s.replace("import { MODEL_CODE, TRAIN_CODE }", "import trainingReport from './data/training_history.json';\nimport { MODEL_CODE, TRAIN_CODE }")
start=s.index('// Progression Scrubber Data');end=s.index('export default function App()',start)
s=s[:start]+'''const summary = trainingReport.summary;
const PROGRESSION_TIMELINE = trainingReport.sample_generations.map(sample => {
  const point = trainingReport.history.reduce((best, item) => Math.abs(item.step - sample.step) < Math.abs(best.step - sample.step) ? item : best);
  return { step: sample.step, loss: point.val_loss, perplexity: point.perplexity,
    phase: sample.step === 0 ? 'Random initialization' : `Training step ${sample.step}`,
    description: sample.note, sample: sample.sample };
});
const chartMax = Math.ceil(Math.max(...trainingReport.history.map(p => Math.max(p.train_loss, p.val_loss))));
const lossPath = (key: 'train_loss' | 'val_loss') => trainingReport.history.map((p,i) => `${i ? 'L' : 'M'} ${40 + 740*p.step/summary.total_steps} ${200 - 180*p[key]/chartMax}`).join(' ');
const formatPrompt = (text: string) => text.startsWith('User:') ? text : `User: ${text.trim()}\\nAssistant: `;
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

'''+s[end:]
s=s.replace("useState<number>(110)","useState<number>(220)")
s=s.replace("useState<any>(() => getInitialAttention())","useState<any>(null)")
s=s.replace("useState<number>(5)","useState<number>(PROGRESSION_TIMELINE.length - 1)")
s=s.replace("useState<string>('MPS')","useState<string>('CPU')")
start=s.index('  // Fetch Attention Weights');end=s.index('  // Timeline Scrubber State',start)
s=s[:start]+'''  const [attentionError, setAttentionError] = useState('');
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

'''+s[end:]
start=s.index('  // Check backend health');end=s.index('  const toggleSound',start)
s=s[:start]+'''  useEffect(() => {
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

'''+s[end:]
start=s.index('  // Full Autoregressive Generation');end=s.index('  // Steer generation',start)
s=s[:start]+'''  const [generationError, setGenerationError] = useState('');
  const [seed, setSeed] = useState('42');
  const generationRequest = useRef<AbortController | null>(null);
  const animation = useRef<number | null>(null);
  const stopGeneration = () => {
    generationRequest.current?.abort();
    if (animation.current !== null) cancelAnimationFrame(animation.current);
    setIsGenerating(false); setIsStepping(false);
  };
  useEffect(() => () => {
    generationRequest.current?.abort();
    if (animation.current !== null) cancelAnimationFrame(animation.current);
  }, []);
  const runGeneration = async (single: boolean) => {
    if (isGenerating || isStepping || !prompt.trim()) return;
    if (seed && (!/^\\d+$/.test(seed) || Number(seed) > 4294967295)) {
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
        setStepDetails(prev => [...prev, ...(data.step_details || [])]);
        setIsStepping(false);
      } else {
        setStepDetails(data.step_details || []);
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

'''+s[end:]
s=s.replace("playClick(440, 0.03);\n                        setPrompt('');", "playClick(440, 0.03);\n                        stopGeneration(); setGenerationError(''); setGenerationStats(null);\n                        setPrompt('');")
s=s.replace('                    className="textarea-clean"\n                    value={prompt}', '                    aria-label="Input prompt"\n                    maxLength={4096}\n                    disabled={isGenerating || isStepping}\n                    className="textarea-clean"\n                    value={prompt}')
s=s.replace('                  <div style={{ display: \'flex\', alignItems: \'center\', justifyContent: \'space-between\', marginTop: \'0.85rem\' }}>', '''                  <div style={{fontSize:'.74rem',color:'var(--text-tertiary)',marginTop:'.5rem'}}>
                    {Array.from(formatPrompt(prompt)).length} characters · {summary.block_size}-character context
                    {Array.from(formatPrompt(prompt)).length > summary.block_size && ' · Older context will be cropped'}
                  </div>
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginTop: '0.85rem' }}>''')
s=s.replace('                      {/* Step Single Token Button */}', '''                      {(isGenerating || isStepping) && <button className="btn-secondary" onClick={stopGeneration}>Stop</button>}
                      {/* Step Single Token Button */}''')
s=s.replace('                {/* Output Panel with Confidence View Toggle */}', '''                {generationError && <div className="card" role="alert" style={{color:'#fca5a5'}}>{generationError}</div>}
                {/* Output Panel with Confidence View Toggle */}''')
s=s.replace('                {/* Model Info Card */}', '''                <div className="card">
                  <label htmlFor="sampling-seed" style={{display:'block',marginBottom:'.5rem'}}>Sampling seed</label>
                  <input id="sampling-seed" className="textarea-clean" style={{minHeight:'auto'}} value={seed} onChange={e=>setSeed(e.target.value)} inputMode="numeric" placeholder="Blank for random" />
                  <p style={{fontSize:'.75rem',color:'var(--text-tertiary)',marginTop:'.5rem'}}>Repeat a prompt with the same settings and seed to reproduce a response on the same engine.</p>
                </div>
                {/* Model Info Card */}''')
s=s.replace('              {/* Heatmap Grid */}', '''              {attentionError && <p role="alert" style={{color:'#fca5a5'}}>{attentionError}</p>}
              {isAttnLoading && <p role="status">Computing attention from the trained model…</p>}
              {/* Heatmap Grid */}''')
s=s.replace('              {/* Prompt Suggestions */}', '              {/* Prompt Suggestions */}')
s=s.replace('        {activeTab === \'playground\' && (', '''        <p className="model-notice">A small model trained from scratch on a narrow teaching curriculum. Outputs can be wrong; token confidence is not factual certainty.</p>
        {activeTab === 'playground' && (''',1)
s=s.replace('813K Params','{(summary.parameters / 1e6).toFixed(2)}M Params').replace('>813,184<','>{summary.parameters.toLocaleString()}<').replace('>128 Tokens<','>{summary.block_size} Characters<').replace('>128<','>{summary.d_model}<').replace('>86 Characters<','>{summary.vocab_size} Tokens<').replace('>Apple Silicon (MPS)<','>{summary.device.toUpperCase()} training<')
s=s.replace('>0.0676<','>{summary.best_val_loss}<').replace('>1.07<','>{Math.exp(summary.best_val_loss).toFixed(2)}<').replace('>2,200<','>{summary.total_steps.toLocaleString()}<').replace('↓ 97.6% error reduction','Held-out question phrasings').replace('e^loss (near-optimal)','exp(validation loss)').replace('/ 2200','/ {summary.total_steps}').replace('Tracking loss from random initialization down to 0.0676','Measured answer-token loss at each evaluation step')
s=s.replace('>3.0<','>{chartMax}<').replace('>2.0<','>{(chartMax*2/3).toFixed(1)}<').replace('>1.0<','>{(chartMax/3).toFixed(1)}<').replace('d="M 50 30 Q 150 140, 280 180 T 520 195 T 770 198"', 'd={lossPath(\'train_loss\')}').replace('d="M 50 35 Q 160 145, 290 184 T 530 196 T 770 198"', 'd={lossPath(\'val_loss\')}')
s=s.replace('setMaxTokens(110)','setMaxTokens(220)').replace('max="200"','max="256"')
s=s.replace('                  value={attnInput}', '                  aria-label="Text to analyze for attention"\n                  maxLength={32}\n                  value={attnInput}')
s=s.replace('                    <div style={{\n                      fontFamily:', '                    <div style={{\n                      fontFamily:')
p.write_text(s)
