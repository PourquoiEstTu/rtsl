declare global {
  const puter: {
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

export {};
