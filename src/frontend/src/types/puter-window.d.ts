// src/types/puter.d.ts
export {}; // makes this a module

declare global {
  interface Window {
    puter: {
      ai: {
        txt2speech: (
          text: string,
          options?:
            | string
            | {
                voice?: string;
                engine?: "standard" | "neural" | "generative";
                language?: string;
              },
        ) => Promise<HTMLAudioElement>;
      };
    };
  }
}
