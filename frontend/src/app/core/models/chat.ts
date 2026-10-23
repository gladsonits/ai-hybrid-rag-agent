/**
 * The shapes the chat API speaks.
 *
 * These mirror the pydantic models in `src/api/schemas.py` on the Python
 * side - the event names, the field names and the meaning of every flag
 * are the same, so a change there is a change here.
 */

/** How the memory recognised the question. */
export type CacheMatch = 'exact' | 'semantic';

/** A passage the answer was built from. */
export interface ChatSource {
  chunk_id: number;
  text: string;
  metadata: Record<string, unknown>;
  score: number | null;
}

/** A passage the answer actually referred to. */
export interface ChatCitation {
  marker: number;
  chunk_id: number;
  text: string;
  metadata: Record<string, unknown>;
  score: number | null;
}

/** The kinds of event the stream carries. */
export type ChatEventType = 'sources' | 'token' | 'citations' | 'done' | 'error';

/** One server-sent event from `POST /chat`. */
export interface ChatEvent {
  type: ChatEventType;
  token?: string;
  sources?: ChatSource[];
  citations?: ChatCitation[];
  conversation_id?: string;
  error?: string;
  /** Whether the turn was answered from the memory rather than by the model. */
  cache_hit?: boolean;
  /** How the memory recognised the question, if it did. */
  cache_match?: CacheMatch | null;
}

/** What the UI sends to `POST /chat`. */
export interface ChatRequest {
  message: string;
  conversation_id: string;
}

/** One frequently asked question, from `GET /memory/faqs`. */
export interface FaqEntry {
  question: string;
  asks: number;
  hits: number;
  misses: number;
  hit_rate: number;
  saved_ms: number;
  cost_ms: number;
  last_asked_at: number;
}

/** How far along a turn is. */
export type MessageStatus = 'streaming' | 'complete' | 'error';

/** A turn of the conversation, as the UI holds it. */
export interface ChatMessage {
  id: number;
  role: 'user' | 'assistant';
  text: string;
  sources: ChatSource[];
  citations: ChatCitation[];
  cacheHit: boolean;
  cacheMatch: CacheMatch | null;
  status: MessageStatus;
  error?: string;
}
