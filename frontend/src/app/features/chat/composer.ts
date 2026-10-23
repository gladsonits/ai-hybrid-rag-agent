import { Component, ElementRef, input, output, signal, viewChild } from '@angular/core';

/**
 * The question box.
 *
 * It owns the text that has been typed and hands it over when the
 * question is asked. While a turn is in flight the send button becomes
 * a stop button, so a long answer can be cut short.
 */
@Component({
  selector: 'app-composer',
  templateUrl: './composer.html',
  styleUrl: './composer.css',
})
export class Composer {
  /** Whether a turn is in flight. */
  readonly busy = input(false);

  /** The question to ask. */
  readonly send = output<string>();

  /** Stop the turn in flight. */
  readonly stop = output<void>();

  /** The text that has been typed. */
  readonly text = signal('');

  private readonly field = viewChild<ElementRef<HTMLTextAreaElement>>('field');

  /** Take what was typed. */
  onInput(event: Event): void {
    const field = event.target as HTMLTextAreaElement;
    this.text.set(field.value);
    grow(field);
  }

  /** Enter asks the question, Shift + Enter starts a new line. */
  onKeydown(event: KeyboardEvent): void {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      this.submit();
    }
  }

  /** Ask the question that has been typed. */
  submit(): void {
    const question = this.text().trim();
    if (question === '' || this.busy()) {
      return;
    }
    this.send.emit(question);
    this.text.set('');
    const field = this.field()?.nativeElement;
    if (field !== undefined) {
      field.value = '';
      grow(field);
      field.focus();
    }
  }
}

/** Make the box as tall as the text needs it to be. */
function grow(field: HTMLTextAreaElement): void {
  field.style.height = 'auto';
  field.style.height = `${Math.min(field.scrollHeight, 200)}px`;
}
