import { Component, effect, HostListener, inject, input, output, signal } from '@angular/core';

import { FaqEntry } from '../../core/models/chat';
import { ChatApiService } from '../../core/services/chat-api.service';

/**
 * The questions that are asked often.
 *
 * The API lists them with the statistics behind each one: how often the
 * question has been asked, how often the memory served it, and how much
 * model time that saved.
 */
@Component({
  selector: 'app-faq-drawer',
  templateUrl: './faq-drawer.html',
  styleUrl: './faq-drawer.css',
})
export class FaqDrawer {
  /** Whether the drawer is open. */
  readonly open = input(false);

  /** The drawer was closed. */
  readonly closed = output<void>();

  /** Ask one of the questions. */
  readonly ask = output<string>();

  private readonly api = inject(ChatApiService);

  /** The questions, once they have been read. */
  readonly faqs = signal<FaqEntry[] | null>(null);

  /** Whether they are being read. */
  readonly loading = signal(false);

  /** Whether they could not be read. */
  readonly failed = signal(false);

  constructor() {
    // The list is read each time the drawer is opened, so a question
    // asked in the meantime is in it.
    effect(() => {
      if (this.open()) {
        this.load();
      }
    });
  }

  /** Close the drawer. */
  close(): void {
    this.closed.emit();
  }

  /** Ask one of the questions. */
  askQuestion(question: string): void {
    this.ask.emit(question);
  }

  /** Close the drawer when the reader presses escape. */
  @HostListener('document:keydown.escape')
  onEscape(): void {
    if (this.open()) {
      this.close();
    }
  }

  /** How long ago the question was last asked. */
  ago(timestamp: number): string {
    const seconds = Math.max(0, Date.now() / 1000 - timestamp);
    if (seconds < 60) {
      return 'just now';
    }
    const minutes = Math.floor(seconds / 60);
    if (minutes < 60) {
      return `${minutes}m ago`;
    }
    const hours = Math.floor(minutes / 60);
    if (hours < 24) {
      return `${hours}h ago`;
    }
    return `${Math.floor(hours / 24)}d ago`;
  }

  /** Read the list of frequently asked questions. */
  private load(): void {
    this.loading.set(true);
    this.failed.set(false);
    this.api.faqs().subscribe({
      next: (faqs) => {
        this.faqs.set(faqs);
        this.loading.set(false);
      },
      error: () => {
        this.failed.set(true);
        this.loading.set(false);
      },
    });
  }
}
