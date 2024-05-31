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
  const { audio_results } = await apiResponse.json();

  const topResults = audio_results.slice(0, 2);

  // @ts-ignore
  const audioContext = topResults.map((result, index) => ({
    role: "system",
    content: `Audio Reference: ${result["Video Filename"]} ${result.Summary} Visual: ${result["Frame Description"]}`,
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
          "Please answer the question using the referenced summaries, cited as [citation number in increasing order](file name). You are a helpful assistant with access to key parts and references into my life. Answer as a chatbot that's friendly and sometimes asks questions, if it is needed. Make inferences if needed. Always use at least one source. At the end, give a list of 2 short follow up questions, phrased in the style of the user's previous questions",
      },
      { role: "user", content: userMessage },
    ],
  };

  // Generating AI stream response
  const result = await streamText(prompt);

  return result.toAIStreamResponse();
}
