import { mount } from "@vue/test-utils";
import { describe, it, expect } from "vitest";
import ToastService from "primevue/toastservice";
import Camera from "../Camera.vue";

function setScreenWidth(width: number) {
  Object.defineProperty(window, "innerWidth", {
    writable: true,
    configurable: true,
    value: width,
  });
}

describe("CameraPage", () => {
  it("renders layout correctly", () => {
    setScreenWidth(1200); // ensure desktop mode

    const wrapper = mount(Camera, {
      global: {
        plugins: [ToastService],
        stubs: {
          Sidebar: true,
          Button: true,
          TranslationBox: true,
          ChatHistoryButton: true,
          PhoneSidebarButton: true,
          KeypointTransceiver: true,
        },
      },
    });

    // Sidebar elements
    expect(wrapper.findComponent({ name: "Sidebar" }).exists()).toBe(true);

    const logoImg = wrapper.find("img.camera-logo");
    expect(logoImg.exists()).toBe(true);
    expect(logoImg.attributes("alt")).toBe("RTSL Logo");

    expect(wrapper.text()).toContain("RTSL");
    expect(wrapper.text()).toContain("Real-Time Sign Language");

    // Main screen elements
    expect(wrapper.findComponent({ name: "Button" }).exists()).toBe(true);
    expect(wrapper.findComponent({ name: "TranslationBox" }).exists()).toBe(
      true
    );

    // KeypointTransceiver
    expect(
      wrapper.findComponent({ name: "KeypointTransceiver" }).exists()
    ).toBe(true);

    // ChatHistoryButton
    expect(wrapper.findComponent({ name: "ChatHistoryButton" }).exists()).toBe(
      true
    );

    // Background gradient element
    expect(wrapper.find(".background-gradient").exists()).toBe(true);

    // Button container
    expect(wrapper.find(".button-container").exists()).toBe(true);

    // PhoneSidebarButton should NOT appear on desktop
    expect(wrapper.findComponent({ name: "PhoneSidebarButton" }).exists()).toBe(
      false
    );
  });

  it("shows PhoneSidebarButton on small screens", () => {
    setScreenWidth(500);

    const wrapper = mount(Camera, {
      global: {
        plugins: [ToastService],
        stubs: {
          Sidebar: true,
          Button: true,
          TranslationBox: true,
          ChatHistoryButton: true,
          PhoneSidebarButton: true,
          KeypointTransceiver: true,
        },
      },
    });

    expect(wrapper.findComponent({ name: "PhoneSidebarButton" }).exists()).toBe(
      true
    );

    // Sidebar should be hidden in mobile layout
    expect(wrapper.findComponent({ name: "Sidebar" }).exists()).toBe(false);
  });
});
