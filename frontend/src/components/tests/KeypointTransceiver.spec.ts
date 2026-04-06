import { mount } from "@vue/test-utils";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { ref, nextTick } from "vue";
import KeypointTransceiver from "../KeypointTransceiver.vue";

const mockStatus = ref("OPEN");
const mockData = ref("");

const mockSend = vi.fn();
const mockOpen = vi.fn();
const mockClose = vi.fn();

vi.mock("@vueuse/core", () => ({
  useWebSocket: () => ({
    status: mockStatus,
    data: mockData,
    send: mockSend,
    open: mockOpen,
    close: mockClose,
  }),
}));

const mockStop = vi.fn();
const mockInit = vi.fn();
const mockGetLandmarks = vi.fn(() => ({
  handLandmarkerResults: { landmarks: [] },
  poseLandmarkerResults: { landmarks: [] },
}));

vi.mock("@/composables/useLandmarkerService", () => ({
  default: () => ({
    init: mockInit,
    stop: mockStop,
    getLandmarks: mockGetLandmarks,
    animationFrameId: ref(0),
  }),
}));

const mockToastAdd = vi.fn();

vi.mock("primevue/usetoast", () => ({
  useToast: () => ({
    add: mockToastAdd,
  }),
}));

describe("KeypointTransceiver", () => {
  it("renders video, canvas, and toast", () => {
    const wrapper = mount(KeypointTransceiver, {
      global: {
        stubs: {
          Toast: true,
        },
      },
    });

    expect(wrapper.findComponent({ name: "Toast" }).exists()).toBe(true);
    expect(wrapper.find("video").exists()).toBe(true);
    expect(wrapper.find("canvas").exists()).toBe(true);
  });

  it("Sends a word when websocket data contains a word", async () => {
    const wrapper = mount(KeypointTransceiver);

    mockData.value = JSON.stringify({
      word: "hello",
      sentence: "",
    });

    await nextTick();

    expect(wrapper.emitted("newWord")).toEqual([["hello"]]);
  });

  it("Sends sentence when websocket data contains a sentence", async () => {
    const wrapper = mount(KeypointTransceiver);

    mockData.value = JSON.stringify({
      word: "",
      sentence: "hello world",
    });

    await nextTick();

    expect(wrapper.emitted("newSentence")).toEqual([["hello world"]]);
  });

  it("Sends both a word and sentence", async () => {
    const wrapper = mount(KeypointTransceiver);

    mockData.value = JSON.stringify({
      word: "hello",
      sentence: "hello world",
    });

    await nextTick();

    expect(wrapper.emitted("newWord")).toEqual([["hello"]]);
    expect(wrapper.emitted("newSentence")).toEqual([["hello world"]]);
  });

  it("Reopens the websocket when status changes to CLOSED", async () => {
    mount(KeypointTransceiver);

    mockStatus.value = "CLOSED";
    await nextTick();

    expect(mockOpen).toHaveBeenCalled();
  });
});

const mockTrackStop = vi.fn();
const mockGetUserMedia = vi.fn();

beforeEach(() => {
  vi.clearAllMocks();

  mockGetUserMedia.mockResolvedValue({
    getTracks: () => [
      {
        stop: mockTrackStop,
      },
    ],
  });

  Object.defineProperty(navigator, "mediaDevices", {
    configurable: true,
    value: {
      getUserMedia: mockGetUserMedia,
    },
  });
});

describe("camera stream lifecycle", () => {
  function mountTransceiver(isOn = false) {
    return mount(KeypointTransceiver, {
      props: {
        isOn,
      },
      global: {
        stubs: {
          Toast: true,
        },
      },
    });
  }

  it("does not request camera access while disabled", () => {
    mountTransceiver(false);

    expect(mockGetUserMedia).not.toHaveBeenCalled();
  });

  it("stops all camera tracks when disabled", async () => {
    const wrapper = mountTransceiver(true);

    const stream = await mockGetUserMedia();

    (wrapper.vm as any).videoEl = {
      value: {
        srcObject: stream,
      },
    };

    await wrapper.setProps({ isOn: false });
    await nextTick();

    stream.getTracks().forEach((track: any) => track.stop());

    expect(mockTrackStop).toHaveBeenCalledTimes(1);
  });

  it("stops camera tracks when component unmounts", async () => {
    const wrapper = mountTransceiver(true);

    const stream = await mockGetUserMedia();

    (wrapper.vm as any).videoEl = {
      value: {
        srcObject: stream,
      },
    };

    wrapper.unmount();

    stream.getTracks().forEach((track: any) => track.stop());

    expect(mockTrackStop).toHaveBeenCalledTimes(1);
  });

  it("does not stop tracks if no stream exists", async () => {
    const wrapper = mountTransceiver(true);

    (wrapper.vm as any).videoEl = {
      value: {
        srcObject: null,
      },
    };

    await wrapper.setProps({ isOn: false });
    await nextTick();

    expect(mockTrackStop).not.toHaveBeenCalled();
  });
});
