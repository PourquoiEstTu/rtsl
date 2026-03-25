// TranslationBox.spec.ts
import { mount } from "@vue/test-utils";
import { describe, it, expect } from "vitest";
import TranslationBox from "../TranslationBox.vue";

// Used for checking to see how text appears

describe("TranslationBox.vue", () => {
  it("Renders title and default translation message", () => {
    const wrapper = mount(TranslationBox, {
      propsData: {
        translation: "Waiting for sign input..."
      }});
    expect(wrapper.text()).toContain("Translation:");
    expect(wrapper.text()).toContain("Waiting for sign input...");
  });
});
