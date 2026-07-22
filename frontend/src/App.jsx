import { useState } from 'react';
import ReviewForm from './components/ReviewForm.jsx';
import EmotionCard from './components/EmotionCard.jsx';
import TopicBars from './components/TopicBars.jsx';

export default function App() {
  const [result,  setResult]  = useState(null);
  const [loading, setLoading] = useState(false);
  const [error,   setError]   = useState('');

  async function analyze(text) {
    setLoading(true);
    setError('');
    setResult(null);
    try {
      const res = await fetch('/api/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);
      setResult(data);
    } catch (e) {
      setError(e.message || 'Something went wrong.');
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="min-h-screen w-full flex items-start justify-center py-10 px-4">
      <div className="w-full max-w-3xl">
        <header className="mb-8 text-center">
          <h1 className="text-3xl md:text-4xl font-bold tracking-tight">
            Customer Review Analyzer
          </h1>
          <p className="mt-2 text-slate-400">
            Fine-tuned DistilBERT · Emotion + Business Topics
          </p>
        </header>

        <ReviewForm onSubmit={analyze} loading={loading} />

        {error && (
          <div className="mt-6 rounded-lg border border-red-500/40 bg-red-500/10 px-4 py-3 text-red-300">
            {error}
          </div>
        )}

        {result && (
          <div className="mt-8 grid gap-6 md:grid-cols-2">
            <EmotionCard emotion={result.emotion} />
            <TopicBars   topics={result.topics}   />
          </div>
        )}

        <footer className="mt-12 text-center text-xs text-slate-500">
          React · Node/Express · FastAPI · PyTorch
        </footer>
      </div>
    </div>
  );
}
