import { mount } from "@vue/test-utils";
import { describe, it, expect } from "vitest";
import ToastService from 'primevue/toastservice';
import Camera from "../Camera.vue";

describe("CameraPage", () => {
  it("renders layout correctly", () => {
    const wrapper = mount(Camera, {
      global: {
        plugins: [ToastService],
        stubs: {
          Sidebar: true,
          Button: true,
          TranslationBox: true,
        },
      },
      props: {
        logo: "/test-logo.png",
      },
    });

    // Check Sidebar stub
    expect(wrapper.findComponent({ name: "Sidebar" }).exists()).toBe(true);

    // Check logo image
    const logoImg = wrapper.find("img.camera-logo");
    expect(logoImg.exists()).toBe(true);
    expect(logoImg.attributes("alt")).toBe("RTSL Logo");
    expect(logoImg.attributes("src")).toBe(
      "/src/assets/logo_without_text-removebg-preview.png"
    );

    // Check video element
    expect(wrapper.find("video").exists()).toBe(true);

    // Check button
    expect(wrapper.findComponent({ name: "Button" }).exists()).toBe(true);

    // Check TranslationBox
    expect(wrapper.findComponent({ name: "TranslationBox" }).exists()).toBe(
      true
    );

    // Check static text
    expect(wrapper.text()).toContain("RTSL");
    expect(wrapper.text()).toContain("Real-Time Sign Language");
  });
});
