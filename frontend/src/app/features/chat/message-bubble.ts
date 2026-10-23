import { Component, computed, ElementRef, inject, input, signal } from '@angular/core';
import { DomSanitizer } from '@angular/platform-browser';

import { ChatMessage } from '../../core/models/chat';
import { renderMarkdown } from '../../shared/markdown';
import { SourcesList } from './sources-list';

/**
 * One turn of the conversation.
 *
 * The answer is markdown, so it is rendered as markdown and sanitized
 * before it reaches the page. The citation markers the model wrote are
 * clickable: they take the reader to the passage they point at.
 */
@Component({
  selector: 'app-message-bubble',
  imports: [SourcesList],
  templateUrl: './message-bubble.html',
  styleUrl: './message-bubble.css',
})
export class MessageBubble {
  /** The turn to show. */
  readonly message = input.required<ChatMessage>();

  private readonly sanitizer = inject(DomSanitizer);
  private readonly host = inject<ElementRef<HTMLElement>>(ElementRef);

  /** The answer, as HTML that is safe to put on the page. */
  readonly html = computed(() =>
    this.sanitizer.bypassSecurityTrustHtml(renderMarkdown(this.message().text)),
  );

  /** Whether the answer has started arriving. */
  readonly waiting = computed(
    () => this.message().status === 'streaming' && this.message().text === '',
  );

  /** The marker of the passage that is being pointed at. */
  readonly highlighted = signal<number | null>(null);

  /**
   * Follow a citation marker to the passage it points at.
   */
  onAnswerClick(event: MouseEvent): void {
    const target = event.target as HTMLElement | null;
    const chip = target?.closest<HTMLElement>('[data-marker]');
    const marker = chip?.dataset['marker'];
    if (marker === undefined) {
      return;
    }

    const passage = this.host.nativeElement.querySelector<HTMLElement>(
      `[data-source-marker="${marker}"]`,
    );
    if (passage === null) {
      return;
    }

    passage.scrollIntoView({ behavior: 'smooth', block: 'center' });
    this.highlighted.set(Number(marker));
    setTimeout(() => this.highlighted.set(null), 1600);
  }
}
