import {
  afterRenderEffect,
  Component,
  ElementRef,
  inject,
  signal,
  viewChild,
} from '@angular/core';

import { ChatEvent, ChatMessage } from '../../core/models/chat';
import { ChatApiService } from '../../core/services/chat-api.service';
import { Composer } from './composer';
import { FaqDrawer } from './faq-drawer';
import { MessageBubble } from './message-bubble';

/** Where the conversation id is kept between visits. */
const CONVERSATION_KEY = 'rag.conversation-id';

/** The questions offered on the empty screen. */
const SUGGESTIONS = [
  'What projects did Rose work on?',
  'What was Rose’s role in the TruStage project?',
  'Which cloud and DevOps tools are listed?',
  'Summarise the professional experience.',
];

/**
 * The chat.
 *
 * It holds the conversation, sends the questions and folds the events
 * that come back into the turns. The API keeps the history of the
 * conversation in PostgreSQL under the `conversation_id`, so this page
 * only has to keep the same id for every question of a conversation -
 * which is what lets a follow-up be answered in the light of the
 * questions before it.
 */
@Component({
  selector: 'app-chat-page',
  imports: [Composer, FaqDrawer, MessageBubble],
  templateUrl: './chat-page.html',
  styleUrl: './chat-page.css',
})
export class ChatPage {
  private readonly api = inject(ChatApiService);

  /** The turns of the conversation, oldest first. */
  readonly messages = signal<ChatMessage[]>([]);

  /** Whether a turn is in flight. */
  readonly busy = signal(false);

  /** Whether the API answered the health check. */
  readonly apiOnline = signal<boolean | null>(null);

  /** Whether the list of frequently asked questions is open. */
  readonly faqOpen = signal(false);

  /** The conversation the questions belong to. */
  readonly conversationId = signal(readConversationId());

  /** The questions offered on the empty screen. */
  readonly suggestions = SUGGESTIONS;

  private nextId = 0;
  private controller: AbortController | null = null;

  /** Whether the view is pinned to the bottom of the conversation. */
  private stickToBottom = true;

  private readonly scroller = viewChild<ElementRef<HTMLDivElement>>('scroller');

  constructor() {
    this.api.health().subscribe({
      next: (online) => this.apiOnline.set(online),
      error: () => this.apiOnline.set(false),
    });

    // Follow the answer as it arrives, unless the reader has scrolled up
    // to look at an earlier turn.
    afterRenderEffect(() => {
      this.messages();
      const host = this.scroller()?.nativeElement;
      if (host !== undefined && this.stickToBottom) {
        host.scrollTop = host.scrollHeight;
      }
    });
  }

  /** Ask a question. */
  async send(text: string): Promise<void> {
    const question = text.trim();
    if (question === '' || this.busy()) {
      return;
    }

    this.busy.set(true);
    this.append({ role: 'user', text: question });
    const id = this.append({ role: 'assistant', text: '', status: 'streaming' });

    const controller = new AbortController();
    this.controller = controller;

    try {
      await this.api.streamChat(
        { message: question, conversation_id: this.conversationId() },
        (event) => this.apply(id, event),
        controller.signal,
      );
    } catch (error) {
      if (!controller.signal.aborted) {
        this.patch(id, (message) => ({
          ...message,
          status: 'error',
          error: describe(error),
        }));
      }
    } finally {
      this.busy.set(false);
      this.controller = null;
      this.patch(id, (message) =>
        message.status === 'streaming' ? { ...message, status: 'complete' } : message,
      );
    }
  }

  /** Stop the turn in flight. */
  stop(): void {
    this.controller?.abort();
  }

  /** Start a new conversation, and forget the old one. */
  newConversation(): void {
    this.controller?.abort();
    this.messages.set([]);
    const id = createConversationId();
    this.conversationId.set(id);
    writeConversationId(id);
  }

  /** Note whether the reader is at the bottom of the conversation. */
  onScroll(): void {
    const host = this.scroller()?.nativeElement;
    if (host === undefined) {
      return;
    }
    const distance = host.scrollHeight - host.scrollTop - host.clientHeight;
    this.stickToBottom = distance < 80;
  }

  /** Fold one event of the stream into the turn it belongs to. */
  private apply(id: number, event: ChatEvent): void {
    switch (event.type) {
      case 'sources':
        this.patch(id, (message) => ({
          ...message,
          sources: event.sources ?? [],
          cacheHit: event.cache_hit ?? message.cacheHit,
          cacheMatch: event.cache_match ?? message.cacheMatch,
        }));
        break;
      case 'token':
        this.patch(id, (message) => ({
          ...message,
          text: message.text + (event.token ?? ''),
        }));
        break;
      case 'citations':
        this.patch(id, (message) => ({
          ...message,
          citations: event.citations ?? [],
          cacheHit: event.cache_hit ?? message.cacheHit,
          cacheMatch: event.cache_match ?? message.cacheMatch,
        }));
        break;
      case 'done':
        this.patch(id, (message) => ({ ...message, status: 'complete' }));
        break;
      case 'error':
        this.patch(id, (message) => ({
          ...message,
          status: 'error',
          error: event.error ?? 'The chat turn failed.',
        }));
        break;
    }
  }

  /** Add a turn to the conversation and return its id. */
  private append(message: {
    role: ChatMessage['role'];
    text: string;
    status?: ChatMessage['status'];
  }): number {
    const id = ++this.nextId;
    this.messages.update((messages) => [
      ...messages,
      {
        id,
        role: message.role,
        text: message.text,
        sources: [],
        citations: [],
        cacheHit: false,
        cacheMatch: null,
        status: message.status ?? 'complete',
      },
    ]);
    return id;
  }

  /** Change one turn of the conversation. */
  private patch(id: number, change: (message: ChatMessage) => ChatMessage): void {
    this.messages.update((messages) =>
      messages.map((message) => (message.id === id ? change(message) : message)),
    );
  }
}

/** The id of the conversation, kept between visits. */
function readConversationId(): string {
  const stored = readStoredConversationId();
  if (stored !== null && stored !== '') {
    return stored;
  }
  const id = createConversationId();
  writeConversationId(id);
  return id;
}

function readStoredConversationId(): string | null {
  try {
    return globalThis.localStorage?.getItem(CONVERSATION_KEY) ?? null;
  } catch {
    return null;
  }
}

function writeConversationId(id: string): void {
  try {
    globalThis.localStorage?.setItem(CONVERSATION_KEY, id);
  } catch {
    // The browser will not let it be stored - the conversation still works
    // for as long as the page is open.
  }
}

function createConversationId(): string {
  return globalThis.crypto?.randomUUID?.() ?? `c-${Date.now().toString(36)}`;
}

/** Say what went wrong, in a way a reader can act on. */
function describe(error: unknown): string {
  if (error instanceof Error) {
    return error.message;
  }
  return 'The chat turn failed.';
}
