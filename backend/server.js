// Express gateway that forwards analyze requests to the FastAPI ML service.
// Adds validation, timeouts, rate limiting, structured errors, health probe.

import 'dotenv/config';
import express from 'express';
import helmet from 'helmet';
import cors from 'cors';
import morgan from 'morgan';
import rateLimit from 'express-rate-limit';
import axios from 'axios';

const PORT = Number(process.env.PORT || 4000);

// Ensure ML_SERVICE_URL has no trailing slash
const ML_SERVICE_URL = (process.env.ML_SERVICE_URL || 'http://localhost:8000').replace(/\/$/, '');
const ML_TIMEOUT_MS = Number(process.env.ML_TIMEOUT_MS || 15000);
const CORS_ORIGIN = process.env.CORS_ORIGIN || '*'; // Allows production frontend

const app = express();

// Disable cross-origin resource policy restriction for API deployment
app.use(helmet({ crossOriginResourcePolicy: false }));

// CORS configuration to support both local and production environments
app.use(cors({ 
  origin: CORS_ORIGIN === '*' ? true : [CORS_ORIGIN, 'http://localhost:5173'],
  methods: ['GET', 'POST', 'OPTIONS'],
  allowedHeaders: ['Content-Type', 'Authorization']
}));

app.use(express.json({ limit: '32kb' }));
app.use(morgan('tiny'));

// Rate-limit only the analyze endpoint (60 req / min / IP)
const analyzeLimiter = rateLimit({
  windowMs: 60 * 1000,
  max: 60,
  standardHeaders: true,
  legacyHeaders: false,
  message: { error: 'Too many requests, please slow down.' },
});

// Shared axios client to the ML service
const ml = axios.create({
  baseURL: ML_SERVICE_URL,
  timeout: ML_TIMEOUT_MS,
  headers: { 'Content-Type': 'application/json' },
});

// ---------- Routes ----------
app.get('/health', async (_req, res) => {
  try {
    const { data } = await ml.get('/health');
    return res.json({ gateway: 'ok', ml: data });
  } catch (err) {
    return res.status(503).json({ 
      gateway: 'ok', 
      ml: 'unreachable',
      detail: err.message 
    });
  }
});

app.post('/api/analyze', analyzeLimiter, async (req, res) => {
  const text = typeof req.body?.text === 'string' ? req.body.text.trim() : '';
  if (!text) {
    return res.status(400).json({ error: '`text` is required.' });
  }
  if (text.length > 2000) {
    return res.status(413).json({ error: '`text` exceeds 2000 characters.' });
  }

  try {
    console.log(`[Gateway] Posting to ML Service: ${ML_SERVICE_URL}/predict`);
    const { data } = await ml.post('/predict', { text });
    
    // Ensure response is always sent as JSON payload
    return res.status(200).json(data || {});
  } catch (err) {
    console.error('[Gateway ML Error]', err.message);
    if (err.response) {
      // ML service returned an error status
      return res.status(err.response.status || 502).json({
        error: 'ML service error',
        detail: err.response.data?.detail || err.response.statusText,
      });
    }
    if (err.code === 'ECONNABORTED') {
      return res.status(504).json({ error: 'ML service timed out.' });
    }
    return res.status(502).json({
      error: 'Cannot reach ML service',
      detail: err.message,
    });
  }
});

// 404
app.use((_req, res) => res.status(404).json({ error: 'Not found' }));

// Generic error handler (last)
app.use((err, _req, res, _next) => {
  console.error('[gateway]', err);
  res.status(500).json({ error: 'Internal server error' });
});

app.listen(PORT, () => {
  console.log(`Gateway listening on port ${PORT}`);
  console.log(`Forwarding to ML service at ${ML_SERVICE_URL}`);
});
