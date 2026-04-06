export async function textToSpeech(
  text: string,
  options?: {
    voice?: string;
    engine?: "standard" | "neural" | "generative";
    language?: string;
  },
) {
  if (!window.puter) {
    throw new Error("Puter.js not loaded");
  }

  return await puter.ai.txt2speech(text, options);
}
