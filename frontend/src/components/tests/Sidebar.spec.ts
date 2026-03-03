import { mount } from "@vue/test-utils";
import { type Router, createRouter, createMemoryHistory } from "vue-router";
import Sidebar from "../Sidebar.vue";
import { describe, it, expect, beforeEach, vi } from "vitest";

const Home = { template: "<div>Home Page</div>" };
const Translator = { template: "<div>Translator Page</div>" };
const About = { template: "<div>About Page</div>" };

describe("Sidebar navigation", () => {
  let router: Router;

  beforeEach(async () => {
    router = createRouter({
      history: createMemoryHistory(),
      routes: [
        { path: "/", component: Home },
        { path: "/camera", component: Translator },
        { path: "/about", component: About },
      ],
    });

    router.push("/");
    await router.isReady();
  });

  const mountSidebar = () =>
    mount(Sidebar, {
      global: {
        plugins: [router],
        stubs: {
          RouterLink: false, // use real RouterLink
        },
      },
    });

  it("Renders all links in Sidebar", () => {
    const wrapper = mountSidebar();
    const links = wrapper.findAllComponents({ name: "RouterLink" });

    expect(links).toHaveLength(3);
    expect(links[0]!.text()).toBe("Home");
    expect(links[1]!.text()).toBe("Translator");
    expect(links[2]!.text()).toBe("About");
  });

  it("Navigates to / (Home) when clicking Translator", async () => {
    const wrapper = mountSidebar();
    const links = wrapper.findAllComponents({ name: "RouterLink" });

    const pushSpy = vi.spyOn(router as any, "push");
    await links[0]!.trigger("click");
    expect(pushSpy).toHaveBeenCalledWith("/");
  });

  it("Navigates to /camera when clicking Translator", async () => {
    const wrapper = mountSidebar();
    const links = wrapper.findAllComponents({ name: "RouterLink" });

    const pushSpy = vi.spyOn(router as any, "push");
    await links[1]!.trigger("click");
    expect(pushSpy).toHaveBeenCalledWith("/camera");
  });

  it("Navigates to /about when clicking About", async () => {
    const wrapper = mountSidebar();
    const links = wrapper.findAllComponents({ name: "RouterLink" });

    const pushSpy = vi.spyOn(router as any, "push");
    await links[2]!.trigger("click");
    expect(pushSpy).toHaveBeenCalledWith("/about");
  });
});
