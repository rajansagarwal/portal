import { openai } from "@ai-sdk/openai";
import { mistral } from "@ai-sdk/mistral";
import { streamText } from "ai";
import fetch from "node-fetch";

export async function POST(req: Request) {
  const { messages } = await req.json();
  const userMessage = messages[messages.length - 1].content;

  const apiUrl = `http://127.0.0.1:5000/search`;
  const apiResponse = await fetch(apiUrl, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ query: userMessage }),
  });
  // @ts-ignore
  const { results } = await apiResponse.json();

  const topResults = Object.entries(results).slice(0, 3);

  // @ts-ignore
  const audioContext = topResults.map(([filename, summary], index) => ({
    role: "system",
    content: `Filename: ${filename} Content: ${summary}}`,
  }));

  // Prepare prompt for AI to use context in its answer
  const prompt = {
    model: openai("gpt-3.5-turbo"),
    messages: [
      ...messages,
      ...audioContext,
      {
        role: "system",
        content:
          "Please directly answer the question concisely, unless specified by user, using the content, and cite it at the end of the sentence using the format '[citation number in increasing order from 1](reference file id)'. Do not go on tangents. You are a helpful assistant with access to key parts and references into my life. Make inferences if needed. Always add a lot of detail and use at least one source. Do NOT add a list of references at the end.",
      },
      { role: "user", content: userMessage },
    ],
  };

  // Generating AI stream response
  const result = await streamText(prompt);

  return result.toAIStreamResponse();
}
