/**
 * Insights Dashboard mount — self-contained React island.
 *
 * Creates its own floating "📊 Insights" trigger button + fullscreen overlay
 * (no index.html containers needed beyond the script tag), following the same
 * window-global pattern as the other widgets. Use window.InsightsDashboard.show().
 */

import React from 'react';
import { createRoot, Root } from 'react-dom/client';
import { InsightsDashboard } from './InsightsDashboard';
import './styles.css'; // trigger button styles must load before first render

let root: Root | null = null;
let overlayEl: HTMLDivElement | null = null;

function ensureDom(): HTMLDivElement {
  if (overlayEl) return overlayEl;

  overlayEl = document.createElement('div');
  overlayEl.id = 'insights-dashboard-overlay';
  overlayEl.className = 'ins-overlay';
  document.body.appendChild(overlayEl);

  // Platform pill, top-right — mirrors the "Live Session" indicator placement.
  const trigger = document.createElement('button');
  trigger.id = 'insights-dashboard-trigger';
  trigger.className = 'ins-trigger';
  trigger.innerHTML = '<span class="ins-trigger-dot"></span>Insights';
  trigger.addEventListener('click', showInsightsDashboard);
  document.body.appendChild(trigger);

  return overlayEl;
}

export function showInsightsDashboard(): void {
  const overlay = ensureDom();
  if (!root) root = createRoot(overlay);
  root.render(<InsightsDashboard onClose={hideInsightsDashboard} />);
  overlay.classList.add('visible');
}

export function hideInsightsDashboard(): void {
  overlayEl?.classList.remove('visible');
  // Unmount so polling stops while hidden.
  root?.unmount();
  root = null;
}

declare global {
  interface Window {
    InsightsDashboard: {
      show: typeof showInsightsDashboard;
      hide: typeof hideInsightsDashboard;
    };
  }
}

window.InsightsDashboard = { show: showInsightsDashboard, hide: hideInsightsDashboard };

function boot(): void {
  ensureDom();
  // Deep link: /#insights opens the dashboard directly (also handy for demos).
  if (window.location.hash === '#insights') showInsightsDashboard();
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', boot);
} else {
  setTimeout(boot, 100);
}

export default { show: showInsightsDashboard, hide: hideInsightsDashboard };
