import { HttpClient } from '@angular/common/http';
import { inject, Injectable } from '@angular/core';
import { map, Observable } from 'rxjs';

import { API_BASE_URL } from '../config/api.config';
import { ChatEvent, ChatRequest, FaqEntry } from '../models/chat';
import { SseDecoder } from '../sse/sse-decoder';

/**
 * The one place the UI talks to the chat API.
 *
 * The chat endpoint answers with a stream of server-sent events, so it
 * is read with `fetch` and the body is pulled apart as it arrives -
 * `HttpClient` would buffer the whole answer first, which is the one
 * thing the stream is there to avoid. The two plain endpoints - the
 * health check and the list of frequently asked questions - are ordinary
 * requests and go through `HttpClient`.
 */
@Injectable({ providedIn: 'root' })
export class ChatApiService {
  private readonly http = inject(HttpClient);
  private readonly baseUrl = inject(API_BASE_URL);

  /**
   * Send a question and hand over each event as it arrives.
   *
   * @param request the question and the conversation it belongs to
   * @param onEvent called once per event, in the order the API sent them
   * @param signal aborts the request and ends the stream
   */
  async streamChat(
    request: ChatRequest,
    onEvent: (event: ChatEvent) => void,
    signal: AbortSignal,
  ): Promise<void> {
    const response = await fetch(`${this.baseUrl}/chat`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Accept: 'text/event-stream',
      },
      body: JSON.stringify(request),
      signal,
    });

    if (!response.ok) {
      throw new Error(await describeFailure(response));
    }
    if (response.body === null) {
      throw new Error('The API sent an answer with no body to read.');
    }

    const reader = response.body.getReader();
    const text = new TextDecoder();
    const decoder = new SseDecoder();

    for (;;) {
      const { done, value } = await reader.read();
      if (done) {
        break;
      }
      for (const frame of decoder.push(text.decode(value, { stream: true }))) {
        onEvent(readEvent(frame.event, frame.data));
      }
    }

    for (const frame of decoder.flush()) {
      onEvent(readEvent(frame.event, frame.data));
    }
  }

  /** The questions that are asked often, and what is known about them. */
  faqs(): Observable<FaqEntry[]> {
    return this.http
      .get<{ faqs: FaqEntry[] }>(`${this.baseUrl}/memory/faqs`)
      .pipe(map((response) => response.faqs));
  }

  /** Whether the API is up. The check does not touch the database. */
  health(): Observable<boolean> {
    return this.http
      .get<{ status: string }>(`${this.baseUrl}/health`)
      .pipe(map((response) => response.status === 'ok'));
  }
}

/**
 * Read one event out of the stream.
 *
 * @throws when the event is not one this UI knows how to show
 */
function readEvent(name: string, data: string): ChatEvent {
  let payload: unknown;
  try {
    payload = JSON.parse(data);
  } catch {
    throw new Error(`The API sent an event that could not be read: ${name}`);
  }

  const event = payload as ChatEvent;
  if (typeof event?.type !== 'string') {
    throw new Error(`The API sent an event with no type: ${name}`);
  }
  return event;
}

/**
 * Turn a failed response into a message worth showing.
 */
async function describeFailure(response: Response): Promise<string> {
  try {
    const body = (await response.json()) as { detail?: string };
    if (typeof body.detail === 'string') {
      return body.detail;
    }
  } catch {
    // The body was not the JSON the API usually sends - fall through.
  }
  return `The API answered with ${response.status} ${response.statusText}.`;
}
