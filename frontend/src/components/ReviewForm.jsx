import { useState } from 'react';

const SAMPLES = [
  "Delivery arrived two weeks late and the box was crushed.",
  "Absolutely love the new interface, so clean and easy to use!",
  "Charged twice, no refund yet. Furious!",
];

export default function ReviewForm({ onSubmit, loading }) {
  const [text, setText] = useState('');

  function submit(e) {
    e.preventDefault();
    const t = text.trim();
    if (!t || loading) return;
    onSubmit(t);
  }

  return (
    <form
      onSubmit={submit}
      className="rounded-2xl border border-slate-700/70 bg-slate-900/60 backdrop-blur p-5 shadow-xl"
    >
      <label className="block text-sm font-medium text-slate-300 mb-2">
        Customer review
      </label>
      <textarea
        value={text}
        onChange={(e) => setText(e.target.value)}
        rows={5}
        maxLength={2000}
        placeholder="Paste or type a customer review…"
        className="w-full resize-none rounded-lg border border-slate-700 bg-slate-950/60
                   px-4 py-3 text-slate-100 placeholder-slate-500 outline-none
                   focus:border-indigo-500 focus:ring-2 focus:ring-indigo-500/30"
      />

      <div className="mt-3 flex flex-wrap gap-2">
        {SAMPLES.map((s) => (
          <button
            key={s}
            type="button"
            onClick={() => setText(s)}
            className="text-xs px-2 py-1 rounded-full border border-slate-700
                       text-slate-400 hover:text-slate-100 hover:border-slate-500"
          >
            {s.slice(0, 40)}…
          </button>
        ))}
      </div>

      <div className="mt-4 flex items-center justify-between">
        <span className="text-xs text-slate-500">{text.length}/2000</span>
        <button
          type="submit"
          disabled={loading || !text.trim()}
          className="inline-flex items-center gap-2 rounded-lg bg-indigo-600
                     px-5 py-2.5 text-sm font-semibold text-white
                     hover:bg-indigo-500 disabled:bg-slate-700 disabled:text-slate-400
                     transition-colors"
        >
          {loading && (
            <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24" fill="none">
              <circle className="opacity-25" cx="12" cy="12" r="10"
                      stroke="currentColor" strokeWidth="4" />
              <path className="opacity-75" fill="currentColor"
                    d="M4 12a8 8 0 018-8v4a4 4 0 00-4 4H4z" />
            </svg>
          )}
          {loading ? 'Analyzing…' : 'Analyze Review'}
        </button>
      </div>
    </form>
  );
}
