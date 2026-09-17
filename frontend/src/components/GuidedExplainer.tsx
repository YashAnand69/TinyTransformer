import trainingReport from '../data/training_history.json';
import { X, Compass, Cpu, Sliders, Lightbulb } from 'lucide-react';

interface GuidedExplainerProps {
  isOpen: boolean;
  onClose: () => void;
}

export default function GuidedExplainer({ isOpen, onClose }: GuidedExplainerProps) {
  if (!isOpen) return null;

  return (
    <div
      style={{
        position: 'fixed',
        inset: 0,
        backgroundColor: 'var(--backdrop-modal)',
        backdropFilter: 'blur(12px)',
        WebkitBackdropFilter: 'blur(12px)',
        zIndex: 100,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        padding: '1.5rem',
        animation: 'fadeIn 0.15s ease-out'
      }}
      onClick={onClose}
    >
      <div
        className="card"
        style={{
          maxWidth: '680px',
          width: '100%',
          maxHeight: '88vh',
          overflowY: 'auto',
          background: 'var(--bg-surface)',
          border: '1px solid var(--border-strong)',
          boxShadow: 'var(--card-shadow-hover)',
          borderRadius: 'var(--radius-lg)',
          padding: '1.75rem'
        }}
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1.5rem', borderBottom: '1px solid var(--border-subtle)', paddingBottom: '1rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
            <div style={{
              width: '32px',
              height: '32px',
              borderRadius: 'var(--radius-sm)',
              background: 'rgba(255, 255, 255, 0.06)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              border: '1px solid var(--border-subtle)'
            }}>
              <Compass size={16} color="var(--text-primary)" />
            </div>
            <div>
              <h3 style={{ fontSize: '1.05rem', color: 'var(--text-primary)', fontWeight: 600 }}>Architecture & Prompting Guide</h3>
              <p style={{ fontSize: '0.78rem', color: 'var(--text-secondary)' }}>How this scratch-trained model operates and how to get optimal responses</p>
            </div>
          </div>
          <button
            onClick={onClose}
            style={{
              background: 'transparent',
              border: 'none',
              color: 'var(--text-secondary)',
              cursor: 'pointer',
              padding: '0.4rem',
              borderRadius: 'var(--radius-xs)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center'
            }}
          >
            <X size={18} />
          </button>
        </div>

        {/* Content sections */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
          
          {/* Card 1: Scratch vs API */}
          <div className="card-subtle">
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.4rem' }}>
              <Cpu size={15} color="var(--text-primary)" />
              <h4 style={{ fontSize: '0.86rem', color: 'var(--text-primary)', fontWeight: 600 }}>1. Built From Scratch (No Third-Party APIs)</h4>
            </div>
            <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', lineHeight: 1.6 }}>
              Unlike wrappers around external APIs (e.g. OpenAI or Anthropic), this model is a {trainingReport.summary.parameters.toLocaleString()} parameter Causal Decoder Transformer coded mathematically in pure PyTorch and trained directly from random Gaussian noise.
            </p>
          </div>

          {/* Card 2: How to talk to it */}
          <div className="card-subtle">
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.4rem' }}>
              <Lightbulb size={15} color="var(--text-primary)" />
              <h4 style={{ fontSize: '0.86rem', color: 'var(--text-primary)', fontWeight: 600 }}>2. Prompt Formatting</h4>
            </div>
            <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', lineHeight: 1.6, marginBottom: '0.5rem' }}>
              Because character-level models predict the next character based on statistical patterns in their training corpus, using standard dialogue markers produces the most structured answers:
            </p>
            <div style={{
              background: 'var(--bg-surface)',
              padding: '0.65rem 0.85rem',
              borderRadius: 'var(--radius-sm)',
              border: '1px solid var(--border-subtle)',
              fontFamily: 'var(--font-mono)',
              fontSize: '0.78rem',
              color: 'var(--text-primary)'
            }}>
              User: Who are you?<br />
              Assistant: 
            </div>
          </div>

          {/* Card 3: Demystifying the Sliders */}
          <div className="card-subtle">
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.6rem' }}>
              <Sliders size={15} color="var(--text-primary)" />
              <h4 style={{ fontSize: '0.86rem', color: 'var(--text-primary)', fontWeight: 600 }}>3. Sampling Hyperparameters</h4>
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.75rem' }}>
              <div style={{ background: 'var(--bg-surface)', padding: '0.75rem', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border-subtle)' }}>
                <div style={{ fontSize: '0.78rem', fontWeight: 600, color: 'var(--text-primary)' }}>Temperature</div>
                <div style={{ fontSize: '0.72rem', color: 'var(--text-secondary)', marginTop: '0.2rem', lineHeight: 1.5 }}>
                  Lower (0.1 - 0.4) produces deterministic, factual answers. Higher values (&gt; 0.8) increase variety and entropy.
                </div>
              </div>
              <div style={{ background: 'var(--bg-surface)', padding: '0.75rem', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border-subtle)' }}>
                <div style={{ fontSize: '0.78rem', fontWeight: 600, color: 'var(--text-primary)' }}>Top-K Truncation</div>
                <div style={{ fontSize: '0.72rem', color: 'var(--text-secondary)', marginTop: '0.2rem', lineHeight: 1.5 }}>
                  Restricts sampling to the top K most probable candidates, preventing unlikely characters from being chosen.
                </div>
              </div>
              <div style={{ background: 'var(--bg-surface)', padding: '0.75rem', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border-subtle)' }}>
                <div style={{ fontSize: '0.78rem', fontWeight: 600, color: 'var(--text-primary)' }}>Top-P (Nucleus)</div>
                <div style={{ fontSize: '0.72rem', color: 'var(--text-secondary)', marginTop: '0.2rem', lineHeight: 1.5 }}>
                  Dynamically retains only candidate tokens whose cumulative probability exceeds P (e.g. 0.90).
                </div>
              </div>
              <div style={{ background: 'var(--bg-surface)', padding: '0.75rem', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border-subtle)' }}>
                <div style={{ fontSize: '0.78rem', fontWeight: 600, color: 'var(--text-primary)' }}>Token Count</div>
                <div style={{ fontSize: '0.72rem', color: 'var(--text-secondary)', marginTop: '0.2rem', lineHeight: 1.5 }}>
                  Sets the generation budget (number of characters to predict autoregressively).
                </div>
              </div>
            </div>
          </div>

        </div>

        {/* Footer */}
        <div style={{ marginTop: '1.5rem', display: 'flex', justifyContent: 'flex-end' }}>
          <button className="btn-primary" onClick={onClose}>
            Done
          </button>
        </div>
      </div>
    </div>
  );
}
