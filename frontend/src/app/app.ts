import { Component } from '@angular/core';

import { ChatPage } from './features/chat/chat-page';

/**
 * The shell the chat sits in.
 */
@Component({
  selector: 'app-root',
  imports: [ChatPage],
  templateUrl: './app.html',
  styleUrl: './app.css',
})
export class App {}
