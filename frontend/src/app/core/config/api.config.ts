import { InjectionToken } from '@angular/core';
import { environment } from '../../environments/environment';

declare global {
  /** Set this to point the UI at an API on another origin. */
  var __RAG_API_BASE__: string | undefined;
}

/**
 * Where the chat API lives.
 *
 * The default is the empty string, which means "the same origin as the
 * page". The development server proxies `/chat`, `/memory` and `/health`
 * to the API, so the browser never makes a cross-origin request and the
 * stream behaves the same as it would in production behind a proxy.
 *
 * Set `window.__RAG_API_BASE__` to `http://127.0.0.1:8000` to call the
 * API directly instead - the API allows the origins listed in
 * `CORS_ORIGINS`.
 */
export const API_BASE_URL = new InjectionToken<string>('API_BASE_URL', {
  providedIn: 'root',
  factory: () => environment.apiBaseUrl,
});
