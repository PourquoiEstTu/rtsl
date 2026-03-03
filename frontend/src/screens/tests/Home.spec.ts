// Home.spec.ts
import { mount } from "@vue/test-utils";
import { describe, it, expect, vi, beforeEach } from "vitest";
import Home from "../Home.vue";

// Unit Tests
// Checking amount of bottons
// Checking if bottons have the right text
// Checking amount of buttons at a specific screen size?
// Checking page rendering

// Integration Tests
// Testing to see if every button routes to the right page

// Mock vue-router
const pushMock = vi.fn();

vi.mock("vue-router", () => ({
  useRouter: () => ({
    push: pushMock,
  }),
}));

describe("Home.vue", () => {
  beforeEach(() => {
    pushMock.mockClear();
  });

  it("Renders header content correctly", () => {
    const wrapper = mount(Home);

    const logoImg = wrapper.find("img.home-logo");
    expect(logoImg.exists()).toBe(true);
    expect(logoImg.attributes("alt")).toBe("RTSL Logo");
    expect(logoImg.attributes("src")).toBe(
      "/src/assets/logo_without_text-removebg-preview.png"
    );

    expect(wrapper.text()).toContain("RTSL");
    expect(wrapper.text()).toContain("Real-Time Sign Language Assistant");
  });

  it("Renders Camera, Web Extension, and About Us cards", () => {
    const wrapper = mount(Home);

    expect(wrapper.find('[data-testid="camera-card"]').exists()).toBe(true);
    expect(wrapper.find('[data-testid="plugin-card"]').exists()).toBe(true);
    expect(wrapper.find('[data-testid="about-card"]').exists()).toBe(true);
  });

  it("Navigates to 'Translate' when Camera card clicked", async () => {
    const wrapper = mount(Home);

    const cards = wrapper.findAll(".home-card");
    await cards[0]!.trigger("click");

    expect(pushMock).toHaveBeenCalledWith("/camera");
  });

  it("Navigates to 'About Us' when the About card is clicked", async () => {
    const wrapper = mount(Home);

    const cards = wrapper.findAll(".home-card");
    await cards[cards.length - 1]!.trigger("click");

    expect(pushMock).toHaveBeenCalledWith("/about");
  });

  it("Updates the cards to add / remove 'Web Extension' when window resizes", async () => {
    // Case where the extension card is removed
    window.innerWidth = 500;
    const wrapper = mount(Home);
    expect(wrapper.text()).not.toContain("Download the"); // Will try to adjust
    // Case where the extension card exists
    window.innerWidth = 1300;
    window.dispatchEvent(new Event("resize"));
    await wrapper.vm.$nextTick();
    expect(wrapper.text()).toContain("Download the"); // Will try to adjust
  });
});
