import { Resend } from 'resend'

// Only initialize Resend if the API key exists
const resend = process.env.RESEND_API_KEY
  ? new Resend(process.env.RESEND_API_KEY)
  : null

const FROM   = process.env.RESEND_FROM_EMAIL ?? 'noreply@cadxstudio.in'
const APP    = process.env.NEXT_PUBLIC_APP_URL ?? 'https://cadxstudio.in'

// ── Email templates (unchanged) ──────────────────────────
function baseHtml(title: string, body: string): string {
  return `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>${title}</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: #0A0A0F; color: #ffffff; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; }
  .wrap { max-width: 560px; margin: 40px auto; background: #111118; border-radius: 16px; overflow: hidden; border: 1px solid rgba(255,255,255,0.08); }
  .header { background: linear-gradient(135deg, #1C1C2A, #111118); padding: 32px; border-bottom: 1px solid rgba(255,255,255,0.06); }
  .logo { font-size: 20px; font-weight: 700; color: #fff; }
  .logo span { color: #3B82F6; }
  .body { padding: 32px; }
  h1 { font-size: 22px; font-weight: 700; color: #fff; margin-bottom: 12px; line-height: 1.3; }
  p  { font-size: 15px; color: #94A3B8; line-height: 1.7; margin-bottom: 16px; }
  .btn { display: inline-block; background: #2563EB; color: #fff !important; font-weight: 600; font-size: 14px; padding: 12px 28px; border-radius: 10px; text-decoration: none; margin: 8px 0 24px; }
  .pill { display: inline-block; font-size: 12px; font-weight: 600; padding: 4px 12px; border-radius: 99px; }
  .pill-green { background: rgba(16,185,129,0.15); color: #10B981; border: 1px solid rgba(16,185,129,0.3); }
  .pill-blue  { background: rgba(37,99,235,0.15);  color: #3B82F6; border: 1px solid rgba(37,99,235,0.3);  }
  .stat-row { display: flex; gap: 12px; margin-bottom: 20px; }
  .stat { flex: 1; background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.07); border-radius: 10px; padding: 14px; text-align: center; }
  .stat-val { font-size: 20px; font-weight: 700; color: #fff; margin-bottom: 2px; }
  .stat-label { font-size: 11px; color: #94A3B8; }
  .footer { padding: 20px 32px; border-top: 1px solid rgba(255,255,255,0.06); }
  .footer p { font-size: 12px; color: #4B5563; margin: 0; }
</style>
</head>
<body>
<div class="wrap">
  <div class="header">
    <div class="logo">CadX <span>Studio</span></div>
  </div>
  <div class="body">${body}</div>
  <div class="footer">
    <p>CadX Studio · Made in India 🇮🇳 · <a href="${APP}/unsubscribe" style="color:#4B5563">Unsubscribe</a></p>
  </div>
</div>
</body>
</html>`
}

// ── Safe email sending wrapper ───────────────────────────
async function safeSend(options: { from: string; to: string; subject: string; html: string }) {
  if (!resend) {
    console.warn('RESEND_API_KEY missing – email not sent')
    return
  }
  try {
    await resend.emails.send(options)
  } catch (err) {
    console.error('Email send error:', err)
  }
}

// ── Exported functions (now safe) ────────────────────────
export async function sendWelcomeEmail(to: string, name: string) {
  const html = baseHtml('Welcome to CadX Studio', `
    <h1>Welcome, ${name || 'Engineer'} 👋</h1>
    <p>You're now part of India's first AI-powered CAD/CAM platform. Here's what you can do right now on your free plan:</p>
    <div class="stat-row">
      <div class="stat"><div class="stat-val">10</div><div class="stat-label">AI CAD generations</div></div>
      <div class="stat"><div class="stat-val">5</div><div class="stat-label">G-code programs</div></div>
      <div class="stat"><div class="stat-val">3</div><div class="stat-label">File checks</div></div>
    </div>
    <a href="${APP}/dashboard" class="btn">Open your dashboard →</a>
    <p>Quick start: go to <strong style="color:#fff">Text-to-CAD</strong> and type a part description. Your first 3D model generates in under 30 seconds.</p>
    <p>Questions? Reply to this email — we read every message.</p>
    <p style="color:#4B5563; font-size:13px;">— Team CadX Studio</p>
  `)
  return safeSend({
    from:    FROM,
    to,
    subject: 'Welcome to CadX Studio — your AI CAD/CAM platform',
    html,
  })
}

export async function sendPaymentConfirmation(
  to: string,
  name: string,
  plan: string,
  amountInr: number,
  endsAt: string,
  paymentId: string
) {
  const html = baseHtml('Payment confirmed — CadX Pro', `
    <span class="pill pill-green">✓ Payment successful</span>
    <h1 style="margin-top:16px">You're now on CadX Pro!</h1>
    <p>Thank you, ${name}. Your Pro subscription is now active.</p>
    <div class="stat-row">
      <div class="stat"><div class="stat-val">∞</div><div class="stat-label">AI CAD generations</div></div>
      <div class="stat"><div class="stat-val">∞</div><div class="stat-label">G-code programs</div></div>
      <div class="stat"><div class="stat-val">STEP</div><div class="stat-label">Full export formats</div></div>
    </div>
    <a href="${APP}/dashboard" class="btn">Go to your dashboard →</a>
    <p><strong style="color:#fff">Invoice details</strong><br/>
    Amount: ₹${(amountInr / 100).toLocaleString('en-IN')}<br/>
    Plan: ${plan.charAt(0).toUpperCase() + plan.slice(1)}<br/>
    Valid until: ${new Date(endsAt).toLocaleDateString('en-IN', { day:'numeric', month:'long', year:'numeric' })}<br/>
    Payment ID: <code style="font-size:11px;color:#94A3B8">${paymentId}</code></p>
    <p style="font-size:13px;color:#94A3B8">A GST invoice will be sent separately if applicable. For billing queries, reply to this email.</p>
  `)
  return safeSend({
    from:    FROM,
    to,
    subject: `✓ CadX Pro activated — ₹${(amountInr/100).toLocaleString('en-IN')} received`,
    html,
  })
}

export async function sendUsageLimitWarning(
  to: string,
  name: string,
  type: string,
  used: number,
  limit: number
) {
  const html = baseHtml('You\'re approaching your free limit', `
    <span class="pill pill-blue">Usage alert</span>
    <h1 style="margin-top:16px">Almost at your free limit</h1>
    <p>Hi ${name}, you've used <strong style="color:#fff">${used} of ${limit}</strong> free ${type} generations this month.</p>
    <p>Upgrade to Pro to get unlimited generations, STEP export, and Generative Design — all for ₹1,999/month.</p>
    <a href="${APP}/pricing" class="btn">Upgrade to Pro →</a>
    <p style="font-size:13px;color:#94A3B8">Your free limit resets on the 1st of each month. No charges until you upgrade.</p>
  `)
  return safeSend({
    from:    FROM,
    to,
    subject: `You've used ${used}/${limit} free ${type} generations this month`,
    html,
  })
}

export async function sendJobCompleteEmail(
  to: string,
  name: string,
  jobType: string,
  jobId: string,
  durationSec: number
) {
  const labels: Record<string, string> = {
    'ai-cad':       'AI CAD model',
    'text-to-cad':  'Text-to-CAD model',
    'ai-cam':       'AI CAM G-code',
    'text-to-cam':  'Text-to-CAM G-code',
    'generative':   'Generative Design result',
    'upload-check': 'File analysis report',
  }
  const label = labels[jobType] ?? 'AI result'

  const html = baseHtml(`Your ${label} is ready`, `
    <span class="pill pill-green">✓ Complete in ${durationSec}s</span>
    <h1 style="margin-top:16px">Your ${label} is ready</h1>
    <p>Hi ${name}, your ${label} has been generated successfully and is waiting in your dashboard.</p>
    <a href="${APP}/dashboard" class="btn">View result →</a>
    <p style="font-size:13px;color:#94A3B8">Job ID: <code>${jobId}</code></p>
  `)
  return safeSend({
    from:    FROM,
    to,
    subject: `✓ Your ${label} is ready — CadX Studio`,
    html,
  })
}