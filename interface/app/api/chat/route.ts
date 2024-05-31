import { openai } from '@ai-sdk/openai';
import { mistral } from '@ai-sdk/mistral';
import { streamText } from 'ai';

export async function POST(req: Request) {
  const { messages } = await req.json();

  const result = await streamText({
    model: openai('gpt-3.5-turbo'),
    messages,
  });

  return result.toAIStreamResponse();
}